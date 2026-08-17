import torch
import triton
import triton.language as tl


@triton.jit
def _bwd_query_kernel(
    Q, K, V, dO, L, O, dQ,
    S_len, tau, d,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr,
):
    i = tl.program_id(0)
    bh = tl.program_id(1)
    
    rows = tl.arange(0, BLOCK_Q)
    q_offsets = i * BLOCK_Q + rows
    
    mask_q = q_offsets[:, None] < S_len
    
    base_q = Q + bh * S_len * d + i * BLOCK_Q * d
    Q_0 = tl.load(base_q + rows[:, None] * d + tl.arange(0, 64)[None, :], mask=mask_q, other=0.0)
    Q_1 = tl.load(base_q + rows[:, None] * d + (tl.arange(0, 64) + 64)[None, :], mask=mask_q, other=0.0)
    
    base_do = dO + bh * S_len * d + i * BLOCK_Q * d
    dO_0 = tl.load(base_do + rows[:, None] * d + tl.arange(0, 64)[None, :], mask=mask_q, other=0.0)
    dO_1 = tl.load(base_do + rows[:, None] * d + (tl.arange(0, 64) + 64)[None, :], mask=mask_q, other=0.0)
    
    base_o = O + bh * S_len * d + i * BLOCK_Q * d
    O_0 = tl.load(base_o + rows[:, None] * d + tl.arange(0, 64)[None, :], mask=mask_q, other=0.0)
    O_1 = tl.load(base_o + rows[:, None] * d + (tl.arange(0, 64) + 64)[None, :], mask=mask_q, other=0.0)
    
    D_i_unreduced = tl.sum(dO_0 * O_0 + dO_1 * O_1, axis=1)
    
    L_base = L + bh * S_len + q_offsets
    L_i = tl.load(L_base, mask=(q_offsets < S_len), other=0.0)
    
    dQ_i_0 = tl.zeros((BLOCK_Q, 64), dtype=tl.float32)
    dQ_i_1 = tl.zeros((BLOCK_Q, 64), dtype=tl.float32)
    
    num_blocks_k = tl.cdiv(S_len, BLOCK_K)
    
    for k_idx in range(num_blocks_k):
        k_rows = tl.arange(0, BLOCK_K)
        k_offsets = k_idx * BLOCK_K + k_rows
        
        mask_k = k_offsets[:, None] < S_len
        
        base_k = K + bh * S_len * d + k_idx * BLOCK_K * d
        K_0 = tl.load(base_k + k_rows[:, None] * d + tl.arange(0, 64)[None, :], mask=mask_k, other=0.0)
        K_1 = tl.load(base_k + k_rows[:, None] * d + (tl.arange(0, 64) + 64)[None, :], mask=mask_k, other=0.0)
        
        base_v = V + bh * S_len * d + k_idx * BLOCK_K * d
        V_0 = tl.load(base_v + k_rows[:, None] * d + tl.arange(0, 64)[None, :], mask=mask_k, other=0.0)
        V_1 = tl.load(base_v + k_rows[:, None] * d + (tl.arange(0, 64) + 64)[None, :], mask=mask_k, other=0.0)
        
        S = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
        S = tl.dot(Q_0, K_0.T, S)
        S = tl.dot(Q_1, K_1.T, S)
        S = S * tau
        
        P = tl.exp(S - L_i[:, None])
        
        dP = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
        dP = tl.dot(dO_0, V_0.T, dP)
        dP = tl.dot(dO_1, V_1.T, dP)
        
        P_safe = P * mask_q
        dS = P_safe * (dP - D_i_unreduced[:, None]) * tau
        
        dQ_i_0 = tl.dot(dS, K_0, dQ_i_0)
        dQ_i_1 = tl.dot(dS, K_1, dQ_i_1)

    base_dq = dQ + bh * S_len * d + i * BLOCK_Q * d
    dq_0_ptr = base_dq + rows[:, None] * d + tl.arange(0, 64)[None, :]
    dq_1_ptr = base_dq + rows[:, None] * d + (tl.arange(0, 64) + 64)[None, :]
    
    tl.store(dq_0_ptr, dQ_i_0.to(tl.bfloat16), mask=mask_q)
    tl.store(dq_1_ptr, dQ_i_1.to(tl.bfloat16), mask=mask_q)


@triton.jit
def _bwd_key_kernel(
    Q, K, V, dO, L, O, dK, dV,
    S_len, tau, d,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr,
):
    i = tl.program_id(0)
    bh = tl.program_id(1)
    
    rows = tl.arange(0, BLOCK_K)
    k_offsets = i * BLOCK_K + rows
    
    mask_k = k_offsets[:, None] < S_len
    
    base_k = K + bh * S_len * d + i * BLOCK_K * d
    K_0 = tl.load(base_k + rows[:, None] * d + tl.arange(0, 64)[None, :], mask=mask_k, other=0.0)
    K_1 = tl.load(base_k + rows[:, None] * d + (tl.arange(0, 64) + 64)[None, :], mask=mask_k, other=0.0)
    
    base_v = V + bh * S_len * d + i * BLOCK_K * d
    V_0 = tl.load(base_v + rows[:, None] * d + tl.arange(0, 64)[None, :], mask=mask_k, other=0.0)
    V_1 = tl.load(base_v + rows[:, None] * d + (tl.arange(0, 64) + 64)[None, :], mask=mask_k, other=0.0)
    
    dK_j_0 = tl.zeros((BLOCK_K, 64), dtype=tl.float32)
    dK_j_1 = tl.zeros((BLOCK_K, 64), dtype=tl.float32)
    dV_j_0 = tl.zeros((BLOCK_K, 64), dtype=tl.float32)
    dV_j_1 = tl.zeros((BLOCK_K, 64), dtype=tl.float32)
    
    num_blocks_q = tl.cdiv(S_len, BLOCK_Q)
    
    for q_idx in range(num_blocks_q):
        q_rows = tl.arange(0, BLOCK_Q)
        q_offsets_k = q_idx * BLOCK_Q + q_rows
        
        mask_q_in_loop = q_offsets_k[:, None] < S_len
        
        base_qk = Q + bh * S_len * d + q_idx * BLOCK_Q * d
        Q_0 = tl.load(base_qk + q_rows[:, None] * d + tl.arange(0, 64)[None, :], mask=mask_q_in_loop, other=0.0)
        Q_1 = tl.load(base_qk + q_rows[:, None] * d + (tl.arange(0, 64) + 64)[None, :], mask=mask_q_in_loop, other=0.0)
        
        base_dok = dO + bh * S_len * d + q_idx * BLOCK_Q * d
        dO_0 = tl.load(base_dok + q_rows[:, None] * d + tl.arange(0, 64)[None, :], mask=mask_q_in_loop, other=0.0)
        dO_1 = tl.load(base_dok + q_rows[:, None] * d + (tl.arange(0, 64) + 64)[None, :], mask=mask_q_in_loop, other=0.0)
        
        base_ok = O + bh * S_len * d + q_idx * BLOCK_Q * d
        O_0 = tl.load(base_ok + q_rows[:, None] * d + tl.arange(0, 64)[None, :], mask=mask_q_in_loop, other=0.0)
        O_1 = tl.load(base_ok + q_rows[:, None] * d + (tl.arange(0, 64) + 64)[None, :], mask=mask_q_in_loop, other=0.0)
        
        D_i_unreduced = tl.sum(dO_0 * O_0 + dO_1 * O_1, axis=1)
        
        L_base_k = L + bh * S_len + q_offsets_k
        L_k = tl.load(L_base_k, mask=(q_offsets_k < S_len), other=0.0)
        
        S = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
        S = tl.dot(Q_0, K_0.T, S)
        S = tl.dot(Q_1, K_1.T, S)
        S = S * tau
        
        P = tl.exp(S - L_k[:, None])
        
        dP = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
        dP = tl.dot(dO_0, V_0.T, dP)
        dP = tl.dot(dO_1, V_1.T, dP)
        
        P_safe = P * mask_q_in_loop
        dS = P_safe * (dP - D_i_unreduced[:, None]) * tau
        
        dV_j_0 = tl.dot(P.T, dO_0, dV_j_0)
        dV_j_1 = tl.dot(P.T, dO_1, dV_j_1)
        
        dK_j_0 = tl.dot(dS.T, Q_0, dK_j_0)
        dK_j_1 = tl.dot(dS.T, Q_1, dK_j_1)

    base_dk = dK + bh * S_len * d + i * BLOCK_K * d
    dk_0_ptr = base_dk + rows[:, None] * d + tl.arange(0, 64)[None, :]
    dk_1_ptr = base_dk + rows[:, None] * d + (tl.arange(0, 64) + 64)[None, :]
    
    tl.store(dk_0_ptr, dK_j_0.to(tl.bfloat16), mask=mask_k)
    tl.store(dk_1_ptr, dK_j_1.to(tl.bfloat16), mask=mask_k)
    
    base_dv = dV + bh * S_len * d + i * BLOCK_K * d
    dv_0_ptr = base_dv + rows[:, None] * d + tl.arange(0, 64)[None, :]
    dv_1_ptr = base_dv + rows[:, None] * d + (tl.arange(0, 64) + 64)[None, :]
    
    tl.store(dv_0_ptr, dV_j_0.to(tl.bfloat16), mask=mask_k)
    tl.store(dv_1_ptr, dV_j_1.to(tl.bfloat16), mask=mask_k)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S_len = Q.shape[0], Q.shape[1], Q.shape[2]
    d = 128
    tau = 1.0 / (d ** 0.5)
    
    num_blocks_q = triton.cdiv(S_len, 64)
    grid_query = (num_blocks_q, B * H)
    _bwd_query_kernel[grid_query](
        Q, K, V, dO, L, O, dQ, S_len, tau, d,
        BLOCK_Q=64, BLOCK_K=64,
        num_warps=4,
    )
    
    num_blocks_k = triton.cdiv(S_len, 64)
    grid_key = (num_blocks_k, B * H)
    _bwd_key_kernel[grid_key](
        Q, K, V, dO, L, O, dK, dV, S_len, tau, d,
        BLOCK_Q=64, BLOCK_K=64,
        num_warps=4,
    )