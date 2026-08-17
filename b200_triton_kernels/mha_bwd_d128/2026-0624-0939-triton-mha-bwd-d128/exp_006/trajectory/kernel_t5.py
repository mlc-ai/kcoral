import torch
import triton
import triton.language as tl


@triton.jit
def _bwd_query_kernel(
    Q, K, V, dO, L, O, dQ,
    S_len, tau, d, 
    batch_stride, head_stride, batch_stride_l, head_stride_l,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    i = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    
    q_offsets = i * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = (q_offsets[:, None] < S_len)
    
    base_q = Q + b * batch_stride + h * head_stride + i * BLOCK_M * d
    Q_i0 = tl.load(base_q + q_offsets[:, None] * d + tl.arange(0, 64)[None, :], mask=row_mask, other=0.0)
    Q_i1 = tl.load(base_q + q_offsets[:, None] * d + (tl.arange(0, 64) + 64)[None, :], mask=row_mask, other=0.0)
    
    base_do = dO + b * batch_stride + h * head_stride + i * BLOCK_M * d
    dO_i0 = tl.load(base_do + q_offsets[:, None] * d + tl.arange(0, 64)[None, :], mask=row_mask, other=0.0)
    dO_i1 = tl.load(base_do + q_offsets[:, None] * d + (tl.arange(0, 64) + 64)[None, :], mask=row_mask, other=0.0)
    
    base_o = O + b * batch_stride + h * head_stride + i * BLOCK_M * d
    O_i0 = tl.load(base_o + q_offsets[:, None] * d + tl.arange(0, 64)[None, :], mask=row_mask, other=0.0)
    O_i1 = tl.load(base_o + q_offsets[:, None] * d + (tl.arange(0, 64) + 64)[None, :], mask=row_mask, other=0.0)
    
    D_i_unreduced = tl.sum(dO_i0 * O_i0 + dO_i1 * O_i1, axis=1)
    D_i = D_i_unreduced * (q_offsets < S_len)
    
    base_l = L + b * batch_stride_l + h * head_stride_l + q_offsets
    L_i = tl.load(base_l, mask=(q_offsets < S_len), other=0.0)
    
    dQ_i0 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    dQ_i1 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    
    num_blocks_k = tl.cdiv(S_len, BLOCK_N)
    for j in range(num_blocks_k):
        k_offsets = j * BLOCK_N + tl.arange(0, BLOCK_N)
        col_mask = (k_offsets[None, :] < S_len)
        
        base_k = K + b * batch_stride + h * head_stride + j * BLOCK_N * d
        K_j0 = tl.load(base_k + k_offsets[:, None] * d + tl.arange(0, 64)[None, :], mask=col_mask, other=0.0)
        K_j1 = tl.load(base_k + k_offsets[:, None] * d + (tl.arange(0, 64) + 64)[None, :], mask=col_mask, other=0.0)
        
        base_v = V + b * batch_stride + h * head_stride + j * BLOCK_N * d
        V_j0 = tl.load(base_v + k_offsets[:, None] * d + tl.arange(0, 64)[None, :], mask=col_mask, other=0.0)
        V_j1 = tl.load(base_v + k_offsets[:, None] * d + (tl.arange(0, 64) + 64)[None, :], mask=col_mask, other=0.0)
        
        S = tl.dot(Q_i0, K_j0.T) + tl.dot(Q_i1, K_j1.T)
        scaled_S = S * tau
        
        P = tl.exp(scaled_S - L_i[:, None])
        P = P * row_mask * col_mask
        
        dP = tl.dot(dO_i0, V_j0.T) + tl.dot(dO_i1, V_j1.T)
        
        dS = P * (dP - D_i[:, None]) * tau
        dS = dS * row_mask * col_mask
        
        dQ_i0 = tl.dot(dS, K_j0, dQ_i0)
        dQ_i1 = tl.dot(dS, K_j1, dQ_i1)

    base_dq = dQ + b * batch_stride + h * head_stride + i * BLOCK_M * d
    dq0_ptr = base_dq + q_offsets[:, None] * d + tl.arange(0, 64)[None, :]
    dq1_ptr = base_dq + q_offsets[:, None] * d + (tl.arange(0, 64) + 64)[None, :]
    tl.store(dq0_ptr, dQ_i0.to(tl.bfloat16), mask=row_mask)
    tl.store(dq1_ptr, dQ_i1.to(tl.bfloat16), mask=row_mask)


@triton.jit
def _bwd_key_kernel(
    Q, K, V, dO, L, O, dK, dV,
    S_len, tau, d, 
    batch_stride, head_stride, batch_stride_l, head_stride_l,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    j = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    
    k_offsets = j * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = (k_offsets[None, :] < S_len)
    
    base_k = K + b * batch_stride + h * head_stride + j * BLOCK_N * d
    K_j0 = tl.load(base_k + k_offsets[:, None] * d + tl.arange(0, 64)[None, :], mask=col_mask, other=0.0)
    K_j1 = tl.load(base_k + k_offsets[:, None] * d + (tl.arange(0, 64) + 64)[None, :], mask=col_mask, other=0.0)
    
    base_v = V + b * batch_stride + h * head_stride + j * BLOCK_N * d
    V_j0 = tl.load(base_v + k_offsets[:, None] * d + tl.arange(0, 64)[None, :], mask=col_mask, other=0.0)
    V_j1 = tl.load(base_v + k_offsets[:, None] * d + (tl.arange(0, 64) + 64)[None, :], mask=col_mask, other=0.0)
    
    dK_j0 = tl.zeros((BLOCK_N, 64), dtype=tl.float32)
    dK_j1 = tl.zeros((BLOCK_N, 64), dtype=tl.float32)
    dV_j0 = tl.zeros((BLOCK_N, 64), dtype=tl.float32)
    dV_j1 = tl.zeros((BLOCK_N, 64), dtype=tl.float32)
    
    num_blocks_q = tl.cdiv(S_len, BLOCK_M)
    for i in range(num_blocks_q):
        q_offsets = i * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = (q_offsets[:, None] < S_len)
        
        base_q = Q + b * batch_stride + h * head_stride + i * BLOCK_M * d
        Q_i0 = tl.load(base_q + q_offsets[:, None] * d + tl.arange(0, 64)[None, :], mask=row_mask, other=0.0)
        Q_i1 = tl.load(base_q + q_offsets[:, None] * d + (tl.arange(0, 64) + 64)[None, :], mask=row_mask, other=0.0)
        
        base_do = dO + b * batch_stride + h * head_stride + i * BLOCK_M * d
        dO_i0 = tl.load(base_do + q_offsets[:, None] * d + tl.arange(0, 64)[None, :], mask=row_mask, other=0.0)
        dO_i1 = tl.load(base_do + q_offsets[:, None] * d + (tl.arange(0, 64) + 64)[None, :], mask=row_mask, other=0.0)
        
        base_o = O + b * batch_stride + h * head_stride + i * BLOCK_M * d
        O_i0 = tl.load(base_o + q_offsets[:, None] * d + tl.arange(0, 64)[None, :], mask=row_mask, other=0.0)
        O_i1 = tl.load(base_o + q_offsets[:, None] * d + (tl.arange(0, 64) + 64)[None, :], mask=row_mask, other=0.0)
        
        D_i_unreduced = tl.sum(dO_i0 * O_i0 + dO_i1 * O_i1, axis=1)
        D_i = D_i_unreduced * (q_offsets < S_len)
        
        base_l = L + b * batch_stride_l + h * head_stride_l + q_offsets
        L_i = tl.load(base_l, mask=(q_offsets < S_len), other=0.0)
        
        S = tl.dot(Q_i0, K_j0.T) + tl.dot(Q_i1, K_j1.T)
        scaled_S = S * tau
        
        P = tl.exp(scaled_S - L_i[:, None])
        P = P * row_mask * col_mask
        
        dP = tl.dot(dO_i0, V_j0.T) + tl.dot(dO_i1, V_j1.T)
        
        dS = P * (dP - D_i[:, None]) * tau
        dS = dS * row_mask * col_mask
        
        dV_j0 = tl.dot(P.T, dO_i0, dV_j0)
        dV_j1 = tl.dot(P.T, dO_i1, dV_j1)
        
        dK_j0 = tl.dot(dS.T, Q_i0, dK_j0)
        dK_j1 = tl.dot(dS.T, Q_i1, dK_j1)

    base_dk = dK + b * batch_stride + h * head_stride + j * BLOCK_N * d
    dk0_ptr = base_dk + k_offsets[:, None] * d + tl.arange(0, 64)[None, :]
    dk1_ptr = base_dk + k_offsets[:, None] * d + (tl.arange(0, 64) + 64)[None, :]
    tl.store(dk0_ptr, dK_j0.to(tl.bfloat16), mask=col_mask)
    tl.store(dk1_ptr, dK_j1.to(tl.bfloat16), mask=col_mask)
    
    base_dv = dV + b * batch_stride + h * head_stride + j * BLOCK_N * d
    dv0_ptr = base_dv + k_offsets[:, None] * d + tl.arange(0, 64)[None, :]
    dv1_ptr = base_dv + k_offsets[:, None] * d + (tl.arange(0, 64) + 64)[None, :]
    tl.store(dv0_ptr, dV_j0.to(tl.bfloat16), mask=col_mask)
    tl.store(dv1_ptr, dV_j1.to(tl.bfloat16), mask=col_mask)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S_len = Q.shape[0], Q.shape[1], Q.shape[2]
    d = 128
    tau = 1.0 / (d ** 0.5)
    
    batch_stride = H * S_len * d
    head_stride = S_len * d
    batch_stride_l = H * S_len
    head_stride_l = S_len
    
    num_blocks_q = triton.cdiv(S_len, 64)
    grid_query = (num_blocks_q, B, H)
    _bwd_query_kernel[grid_query](
        Q, K, V, dO, L, O, dQ, S_len, tau, d, 
        batch_stride, head_stride, batch_stride_l, head_stride_l,
        BLOCK_M=64, BLOCK_N=64,
        num_warps=8,
        num_stages=3,
    )
    
    num_blocks_k = triton.cdiv(S_len, 64)
    grid_key = (num_blocks_k, B, H)
    _bwd_key_kernel[grid_key](
        Q, K, V, dO, L, O, dK, dV, S_len, tau, d, 
        batch_stride, head_stride, batch_stride_l, head_stride_l,
        BLOCK_M=64, BLOCK_N=64,
        num_warps=8,
        num_stages=3,
    )