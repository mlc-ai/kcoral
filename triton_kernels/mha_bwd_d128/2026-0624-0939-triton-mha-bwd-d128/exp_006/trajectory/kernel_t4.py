import torch
import triton
import triton.language as tl


@triton.jit
def _bwd_query_kernel(
    Q, K, V, dO, L, O, dQ,
    S_len, tau, d,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    i = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    
    q_offsets = i * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = (q_offsets[:, None] < S_len)
    
    base_q = Q + b * H * S_len * d + h * S_len * d + i * BLOCK_M * d
    Q_i = tl.load(base_q + q_offsets[:, None] * d + tl.arange(0, 128)[None, :], mask=row_mask, other=0.0)
    
    base_do = dO + b * H * S_len * d + h * S_len * d + i * BLOCK_M * d
    dO_i = tl.load(base_do + q_offsets[:, None] * d + tl.arange(0, 128)[None, :], mask=row_mask, other=0.0)
    
    base_o = O + b * H * S_len * d + h * S_len * d + i * BLOCK_M * d
    O_i = tl.load(base_o + q_offsets[:, None] * d + tl.arange(0, 128)[None, :], mask=row_mask, other=0.0)
    
    D_i_unreduced = tl.sum(dO_i * O_i, axis=1)
    row_squeeze_mask = (q_offsets < S_len)
    D_i = D_i_unreduced * row_squeeze_mask
    
    base_l = L + b * H * S_len + h * S_len + q_offsets
    L_i = tl.load(base_l, mask=(q_offsets < S_len), other=0.0)
    
    dQ_i = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    
    num_blocks_k = tl.cdiv(S_len, BLOCK_N)
    for j in range(num_blocks_k):
        k_offsets = j * BLOCK_N + tl.arange(0, BLOCK_N)
        col_mask = (k_offsets[None, :] < S_len)
        
        base_k = K + b * H * S_len * d + h * S_len * d
        K_j = tl.load(base_k + k_offsets[:, None] * d + tl.arange(0, 128)[None, :], mask=col_mask, other=0.0)
        
        base_v = V + b * H * S_len * d + h * S_len * d
        V_j = tl.load(base_v + k_offsets[:, None] * d + tl.arange(0, 128)[None, :], mask=col_mask, other=0.0)
        
        S = tl.dot(Q_i, K_j.T) 
        scaled_S = S * tau
        
        P = tl.exp(scaled_S - L_i[:, None])
        P = P * row_mask * col_mask
        
        dP = tl.dot(dO_i, V_j.T) 
        dS = P * (dP - D_i[:, None]) * tau
        dS = dS * row_mask * col_mask
        
        dQ_i = tl.dot(dS, K_j, dQ_i)

    base_dq = dQ + b * H * S_len * d + h * S_len * d + i * BLOCK_M * d
    dq_ptr = base_dq + q_offsets[:, None] * d + tl.arange(0, 128)[None, :]
    tl.store(dq_ptr, dQ_i.to(tl.bfloat16), mask=row_mask)


@triton.jit
def _bwd_key_kernel(
    Q, K, V, dO, L, O, dK, dV,
    S_len, tau, d,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    j = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    
    k_offsets = j * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = (k_offsets[None, :] < S_len)
    
    base = K + b * H * S_len * d + h * S_len * d
    K_j = tl.load(base + k_offsets[:, None] * d + tl.arange(0, 128)[None, :], mask=col_mask, other=0.0)
    
    base_v = V + b * H * S_len * d + h * S_len * d
    V_j = tl.load(base_v + k_offsets[:, None] * d + tl.arange(0, 128)[None, :], mask=col_mask, other=0.0)
    
    dK_j = tl.zeros((BLOCK_N, 128), dtype=tl.float32)
    dV_j = tl.zeros((BLOCK_N, 128), dtype=tl.float32)
    
    num_blocks_q = tl.cdiv(S_len, BLOCK_M)
    for i in range(num_blocks_q):
        q_offsets = i * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = (q_offsets[:, None] < S_len)
        
        base_q = Q + b * H * S_len * d + h * S_len * d
        Q_i = tl.load(base_q + q_offsets[:, None] * d + tl.arange(0, 128)[None, :], mask=row_mask, other=0.0)
        
        base_do = dO + b * H * S_len * d + h * S_len * d
        dO_i = tl.load(base_do + q_offsets[:, None] * d + tl.arange(0, 128)[None, :], mask=row_mask, other=0.0)
        
        base_o = O + b * H * S_len * d + h * S_len * d
        O_i = tl.load(base_o + q_offsets[:, None] * d + tl.arange(0, 128)[None, :], mask=row_mask, other=0.0)
        
        D_i_unreduced = tl.sum(dO_i * O_i, axis=1)
        
        base_l = L + b * H * S_len + h * S_len + q_offsets
        L_i = tl.load(base_l, mask=(q_offsets < S_len), other=0.0)
        
        S = tl.dot(Q_i, K_j.T) 
        scaled_S = S * tau
        
        P = tl.exp(scaled_S - L_i[:, None])
        P = P * row_mask * col_mask
        
        dP = tl.dot(dO_i, V_j.T) 
        dS = P * (dP - D_i_unreduced[:, None]) * tau
        dS = dS * row_mask * col_mask
        
        dV_j = tl.dot(P.T, dO_i, dV_j)
        dK_j = tl.dot(dS.T, Q_i, dK_j)

    base_dk = dK + b * H * S_len * d + h * S_len * d + j * BLOCK_N * d
    dk_ptr = base_dk + k_offsets[:, None] * d + tl.arange(0, 128)[None, :]
    tl.store(dk_ptr, dK_j.to(tl.bfloat16), mask=col_mask)
    
    base_dv = dV + b * H * S_len * d + h * S_len * d + j * BLOCK_N * d
    dv_ptr = base_dv + k_offsets[:, None] * d + tl.arange(0, 128)[None, :]
    tl.store(dv_ptr, dV_j.to(tl.bfloat16), mask=col_mask)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S_len = Q.shape[0], Q.shape[1], Q.shape[2]
    d = 128
    tau = 1.0 / (d ** 0.5)
    
    num_blocks_q = triton.cdiv(S_len, 64)
    grid_query = (num_blocks_q, B, H)
    _bwd_query_kernel[grid_query](
        Q, K, V, dO, L, O, dQ, S_len, tau, d,
        BLOCK_M=64, BLOCK_N=64,
        num_warps=8,
        num_stages=3,
    )
    
    num_blocks_k = triton.cdiv(S_len, 64)
    grid_key = (num_blocks_k, B, H)
    _bwd_key_kernel[grid_key](
        Q, K, V, dO, L, O, dK, dV, S_len, tau, d,
        BLOCK_M=64, BLOCK_N=64,
        num_warps=8,
        num_stages=3,
    )