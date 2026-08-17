import torch
import triton
import triton.language as tl


@triton.jit
def _mha_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S_len, H, scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    start_pid = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    
    base_offset = (b * H + h) * S_len * BLOCK_K
    Q_base = Q_ptr + base_offset
    K_base = K_ptr + base_offset
    V_base = V_ptr + base_offset
    O_base = O_ptr + base_offset
    
    Q_desc = tl.make_tensor_descriptor(
        Q_base, shape=[S_len, BLOCK_K], strides=[BLOCK_K, 1],
        block_shape=[BLOCK_M, BLOCK_K], padding_option="zero")
    K_desc = tl.make_tensor_descriptor(
        K_base, shape=[S_len, BLOCK_K], strides=[BLOCK_K, 1],
        block_shape=[BLOCK_N, BLOCK_K], padding_option="zero")
    V_desc = tl.make_tensor_descriptor(
        V_base, shape=[S_len, BLOCK_K], strides=[BLOCK_K, 1],
        block_shape=[BLOCK_N, BLOCK_K], padding_option="zero")
    O_desc = tl.make_tensor_descriptor(
        O_base, shape=[S_len, BLOCK_K], strides=[BLOCK_K, 1],
        block_shape=[BLOCK_M, BLOCK_K])
    
    num_pid_m = tl.cdiv(S_len, BLOCK_M)
    num_k_tiles = tl.cdiv(S_len, BLOCK_N)
    
    q_rows = tl.arange(0, BLOCK_M)
    k_rows = tl.arange(0, BLOCK_N)
    
    for tile_id in tl.range(start_pid, num_pid_m, NUM_SMS, flatten=False, warp_specialize=WARP_SPECIALIZE):
        pid_m = tile_id
        start_m = pid_m * BLOCK_M
        
        Q = Q_desc.load([start_m, 0])
        Q_fp32 = Q.to(tl.float32)
        
        m_local = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
        s_local = tl.full((BLOCK_M,), 0.0, dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        
        for k_tile in range(num_k_tiles):
            start_k = k_tile * BLOCK_N
            K = K_desc.load([start_k, 0])
            V = V_desc.load([start_k, 0])
            
            K_fp32 = K.to(tl.float32)
            V_fp32 = V.to(tl.float32)
            
            S_qk = tl.dot(Q_fp32, K_fp32.T) * scale
            
            k_idx = start_k + k_rows
            valid_k = k_idx < S_len
            S_qk = tl.where(valid_k[None, :], S_qk, -float('inf'))
            
            m_curr = tl.max(S_qk, axis=1)
            m_new = tl.maximum(m_local, m_curr)
            p = tl.exp(S_qk - m_new[:, None])
            s_curr = tl.sum(p, axis=1)
            s_local = s_local * tl.exp(m_local - m_new) + s_curr
            acc = acc * tl.exp(m_local - m_new)[:, None] + tl.dot(p, V_fp32)
            m_local = m_new
        
        valid_m = q_rows < (S_len - start_m)
        d = 1.0 / s_local
        O_tile = acc * d[:, None]
        O_fp16 = O_tile.to(tl.bfloat16)
        O_desc.store([start_m, 0], O_fp16)
        
        lse = m_local + tl.log(s_local)
        lse_off = b * H * S_len + h * S_len + start_m + q_rows
        tl.store(LSE_ptr + lse_off, lse, mask=valid_m)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S_len, D = Q.shape
    scale = 1.0 / (D ** 0.5)
    
    if B == 1 and H == 1:
        NUM_SMS = 132
    else:
        total_BH = B * H
        tiles_per_SM = max(1, 132 // total_BH)
        NUM_SMS = min(total_BH * tiles_per_SM, 132)
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 128
    
    grid = (min(NUM_SMS, triton.cdiv(S_len, BLOCK_M)), B, H)
    _mha_kernel[grid](
        Q, K, V, O, LSE,
        S_len, H, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        NUM_SMS=NUM_SMS, WARP_SPECIALIZE=False,
        num_warps=8, num_stages=2,
    )