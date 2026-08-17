import torch
import triton
import triton.language as tl


@triton.jit
def _mha_fwd(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S,
    scale,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    bh_idx = tl.program_id(1)
    B = tl.num_programs(1) // H
    b_idx = bh_idx // H
    h_idx = bh_idx % H
    
    start_m = pid_m * BLOCK_M
    
    base_Q = Q_ptr + b_idx * H * S * D + h_idx * S * D
    base_K = K_ptr + b_idx * H * S * D + h_idx * S * D
    base_V = V_ptr + b_idx * H * S * D + h_idx * S * D
    base_O = O_ptr + b_idx * H * S * D + h_idx * S * D
    base_LSE = LSE_ptr + b_idx * H * S + h_idx * S
    
    row = tl.arange(0, BLOCK_M)
    col = tl.arange(0, 64)
    
    q_idx = start_m + row
    mask_q = q_idx[:, None] < S
    
    q0 = tl.load(base_Q + q_idx[:, None] * D + col[None, :], mask=mask_q, other=0.0)
    q1 = tl.load(base_Q + q_idx[:, None] * D + (col[None, :] + 64), mask=mask_q, other=0.0)
    
    o0 = tl.zeros((BLOCK_M, 64), tl.float32)
    o1 = tl.zeros((BLOCK_M, 64), tl.float32)
    
    running_max = tl.full((BLOCK_M,), -1e20, tl.float32)
    running_sum = tl.full((BLOCK_M,), 0.0, tl.float32)
    
    n_start = 0
    n_end = (start_m + BLOCK_M + BLOCK_N - 1) // BLOCK_N
    
    for n in range(n_start, n_end):
        k_idx = n * BLOCK_N + col
        mask_k = k_idx[None, :] < S
        
        k0 = tl.load(base_K + k_idx[None, :] * D + col[:, None], mask=mask_k, other=0.0)
        k1 = tl.load(base_K + k_idx[None, :] * D + (col[:, None] + 64), mask=mask_k, other=0.0)
        
        v0 = tl.load(base_V + k_idx[None, :] * D + col[:, None], mask=mask_k, other=0.0)
        v1 = tl.load(base_V + k_idx[None, :] * D + (col[:, None] + 64), mask=mask_k, other=0.0)
        
        p = tl.dot(q0, k0.T) + tl.dot(q1, k1.T)
        p = p * scale
        
        causal_mask = q_idx[:, None] >= k_idx[None, :]
        valid_mask = causal_mask & (q_idx[:, None] < S) & (k_idx[None, :] < S)
        p = tl.where(valid_mask, p, -1e20)
        
        m_local = tl.max(p, axis=1)
        new_max = tl.maximum(running_max, m_local)
        
        scale_prev = tl.math.exp(running_max - new_max)
        running_sum = running_sum * scale_prev + tl.sum(tl.math.exp(p - new_max), axis=1)
        
        o0 = o0 * scale_prev[:, None]
        o1 = o1 * scale_prev[:, None]
        
        s = tl.math.exp(p - new_max)
        s = s / tl.sum(s, axis=1)[:, None]
        
        o0 += tl.dot(s, v0)
        o1 += tl.dot(s, v1)
        
        running_max = new_max
        
    mask_o = q_idx[:, None] < S
    tl.store(base_O + q_idx[:, None] * D + col[None, :], o0, mask=mask_o)
    tl.store(base_O + q_idx[:, None] * D + (col[None, :] + 64), o1, mask=mask_o)
    
    lse = running_max + tl.math.log(running_sum)
    tl.store(base_LSE + q_idx, lse, mask=q_idx < S)


# Pre-calculate H to avoid scope issues inside the epilogues or launches
H = 48

def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H_val, S, D = Q.shape
    assert H_val == H, f"Expected H={H}, got {H_val}"
    
    grid = ((S + 63) // 64, B * H)
    scale = 1.0 / (D ** 0.5)
    
    base_Q = Q.contiguous().data_ptr()
    base_K = K.contiguous().data_ptr()
    base_V = V.contiguous().data_ptr()
    base_O = O.contiguous().data_ptr()
    base_LSE = LSE.contiguous().data_ptr()
    
    _mha_fwd[grid](
        base_Q, base_K, base_V, base_O, base_LSE, S, scale, D=D, BLOCK_M=64, BLOCK_N=64,
        num_warps=4, num_stages=2
    )