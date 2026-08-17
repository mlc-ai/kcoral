import torch
import triton
import triton.language as tl


@triton.jit
def _mha_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    seq_len, head_dim,
    SCALE,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D_tiled: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    m_block_idx = tl.program_id(1)
    
    start_m = m_block_idx * BLOCK_M
    num_n_blocks = tl.cdiv(seq_len, BLOCK_N)
    base_offset = batch_idx * seq_len * head_dim
    
    q_desc = tl.make_tensor_descriptor(
        Q_ptr + base_offset, shape=[seq_len, head_dim], strides=[head_dim, 1],
        block_shape=[BLOCK_M, D_tiled], padding_option="zero")
    k_desc = tl.make_tensor_descriptor(
        K_ptr + base_offset, shape=[seq_len, head_dim], strides=[head_dim, 1],
        block_shape=[BLOCK_N, D_tiled], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(
        V_ptr + base_offset, shape=[seq_len, head_dim], strides=[head_dim, 1],
        block_shape=[BLOCK_N, D_tiled], padding_option="zero")
    o_desc = tl.make_tensor_descriptor(
        O_ptr + base_offset, shape=[seq_len, head_dim], strides=[head_dim, 1],
        block_shape=[BLOCK_M, D_tiled])
    
    q0 = q_desc.load([start_m, 0])
    q1 = q_desc.load([start_m, 64])
    
    m = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    acc_O0 = tl.zeros((BLOCK_M, D_tiled), dtype=tl.float32)
    acc_O1 = tl.zeros((BLOCK_M, D_tiled), dtype=tl.float32)
    
    for j in tl.range(0, num_n_blocks, 1, flatten=False, num_stages=2):
        start_n = j * BLOCK_N
        
        k0 = k_desc.load([start_n, 0])
        k1 = k_desc.load([start_n, 64])
        v0 = v_desc.load([start_n, 0])
        v1 = v_desc.load([start_n, 64])
        
        tldot.commit_sync()
        dot_state = tldot(q0, k0.T)
        tldot(q1, k1.T, dot_state)
        tldot.commit_sync()
        acc_S = tldot.get_dot_state_output(dot_state)
        
        S_scaled = acc_S * SCALE
        
        col_valid = (start_n + tl.arange(0, BLOCK_N)) < seq_len
        S_scaled = tl.where(col_valid[None, :], S_scaled, -float('inf'))
        
        m_old = m
        m_new = tl.maximum(m, tl.max(S_scaled, axis=1))
        
        all_invalid = (m_new == -float('inf'))
        exp_diff = tl.exp(m_old - m_new)
        exp_diff = tl.where(all_invalid, 0.0, exp_diff)
        
        P = tl.exp(S_scaled - m_new[:, None])
        P = tl.where(all_invalid[:, None], 0.0, P)
        
        l = l * exp_diff + tl.sum(P, axis=1)
        m = m_new
        
        exp_scale = exp_diff[:, None]
        acc_O0 = acc_O0 * exp_scale
        acc_O1 = acc_O1 * exp_scale
        
        P_bf16 = P.to(tl.bfloat16)
        acc_O0 = tl.dot(P_bf16, v0, acc_O0)
        acc_O1 = tl.dot(P_bf16, v1, acc_O1)
    
    inv_l = 1.0 / tl.maximum(l, 1e-30)[:, None]
    o0 = (acc_O0 * inv_l).to(tl.bfloat16)
    o1 = (acc_O1 * inv_l).to(tl.bfloat16)
    
    o_desc.store([start_m, 0], o0)
    o_desc.store([start_m, 64], o1)
    
    seq_idx = start_m + tl.arange(0, BLOCK_M)
    valid_m = seq_idx < seq_len
    lse_val = m + tl.log(l)
    lse_val = lse_val * valid_m
    base_lse = batch_idx * seq_len
    tl.store(LSE_ptr + base_lse + seq_idx, lse_val, mask=valid_m)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    if S == 0:
        return

    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)
    
    dummy_Q = Q.contiguous()
    dummy_K = K.contiguous()
    dummy_V = V.contiguous()
    dummy_O = O.contiguous()
    
    SCALE = 0.08838834764831843
    
    grid = (B * H, triton.cdiv(S, 128))
    _mha_kernel[grid](
        dummy_Q, dummy_K, dummy_V, dummy_O, LSE, S, D,
        SCALE,
        BLOCK_M=128, BLOCK_N=64, D_tiled=64,
        num_warps=8, num_stages=2
    )
    
    if dummy_O is not O:
        O.copy_(dummy_O)