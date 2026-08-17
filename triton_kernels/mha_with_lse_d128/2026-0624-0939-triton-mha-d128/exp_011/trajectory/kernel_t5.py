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
    NUM_SMS: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    
    num_m_blocks = tl.cdiv(seq_len, BLOCK_M)
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
    
    for m_block_idx in tl.range(batch_idx, num_m_blocks, NUM_SMS, flatten=False):
        start_m = m_block_idx * BLOCK_M
        
        q0 = q_desc.load([start_m, 0])
        q1 = q_desc.load([start_m, 64])
        
        m = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
        l = tl.zeros((BLOCK_M,), dtype=tl.float32)
        
        acc_O = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
        o_left, o_right = tl.split(acc_O)
        
        for j in tl.range(0, num_n_blocks, 1, flatten=False, num_stages=3):
            start_n = j * BLOCK_N
            
            k0 = k_desc.load([start_n, 0])
            k1 = k_desc.load([start_n, 64])
            v0 = v_desc.load([start_n, 0])
            v1 = v_desc.load([start_n, 64])
            
            acc_S = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for i in tl.static_range(2):
                q = q0 if i == 0 else q1
                k = k0 if i == 0 else k1
                acc_S = tl.dot(q, k.T, acc_S)
            
            S_scaled = acc_S * SCALE
            
            m_old = m
            m_new = tl.maximum(m, tl.max(S_scaled, axis=1))
            
            all_invalid = (m_new == -float('inf'))
            exp_diff = tl.exp(m_old - m_new)
            exp_diff = tl.where(all_invalid, 0.0, exp_diff)
            
            P = tl.exp(S_scaled - m_new[:, None])
            P = tl.where(all_invalid[:, None], 0.0, P)
            
            valid_k = (start_n + tl.arange(0, BLOCK_N)) < seq_len
            P = P * valid_k[None, :]
            
            l = l * exp_diff + tl.sum(P, axis=1)
            m = m_new
            
            exp_scale = exp_diff[:, None]
            o_left = o_left * exp_scale
            o_right = o_right * exp_scale
            
            P_bf16 = P.to(tl.bfloat16)
            
            for i in tl.static_range(2):
                p = P_bf16 if i == 0 else P_bf16
                v = v0 if i == 0 else v1
                o = o_left if i == 0 else o_right
                o_expanded = tl.unsqueeze(o, 2)
                v_expanded = tl.unsqueeze(v, 1)
                o = tl.sum(o_expanded * v_expanded, 2)
                o_left = o if i == 0 else o_left
                o_right = o if i == 1 else o_right
        
        inv_l = 1.0 / tl.maximum(l, 1e-30)[:, None]
        acc_O = tl.join(o_left, o_right)
        o_out = (acc_O * inv_l).to(tl.bfloat16)
        
        o_left_out, o_right_out = tl.split(o_out)
        o_desc.store([start_m, 0], o_left_out)
        o_desc.store([start_m, 64], o_right_out)
        
        seq_idx = start_m + tl.arange(0, BLOCK_M)
        valid_m = seq_idx < seq_len
        lse_val = m + tl.log(l)
        lse_val = lse_val * valid_m
        lse_val = tl.where(l == 0.0, 0.0, lse_val)
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
    
    SCALE = 1.0 / (128.0 ** 0.5)
    
    NUM_SMS = torch.cuda.get_device_properties(Q.device).multi_processor_count
    
    grid = (min(NUM_SMS, B * H * triton.cdiv(S, 64)),)
    _mha_kernel[grid](
        Q, K, V, O, LSE, S, D,
        SCALE,
        BLOCK_M=64, BLOCK_N=64, D_tiled=64, NUM_SMS=NUM_SMS,
        num_warps=8, num_stages=3
    )