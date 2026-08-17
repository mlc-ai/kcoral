import torch
import triton
import triton.language as tl


@triton.jit
def _precompute_D_kernel(
    O, dO, D,
    S_len, B, H, d,
):
    idx = tl.program_id(0)
    n = idx % S_len
    bh = idx // S_len
    base = dO + bh * S_len * d + n * d
    offsets = base + tl.arange(0, 128)
    do_vals = tl.load(offsets)
    o_vals = tl.load(O + bh * S_len * d + n * d + tl.arange(0, 128))
    d_val = tl.sum(do_vals * o_vals)
    if n < S_len:
        dl = D + bh * S_len + n
        tl.store(dl, d_val)


@triton.jit
def _bwd_query_kernel(
    Q, K, V, dO, L, D, dQ,
    S_len, B, H, d, tau,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    i = tl.program_id(0)
    dummy = tl.program_id(1)
    bh = tl.program_id(2)
    
    rows = tl.arange(0, BLOCK_Q)
    cols = tl.arange(0, BLOCK_D)
    q_offsets = i * BLOCK_Q + rows
    
    mask_q = q_offsets[:, None] < S_len
    
    Q_base = Q + bh * H * S_len * d + i * BLOCK_Q * d
    Q_i = tl.load(Q_base + rows[:, None] * d + cols[None, :], mask=mask_q, other=0.0)
    
    dO_base = dO + bh * H * S_len * d + i * BLOCK_Q * d
    dO_i = tl.load(dO_base + rows[:, None] * d + cols[None, :], mask=mask_q, other=0.0)
    
    L_base = L + bh * S_len + q_offsets
    L_i = tl.load(L_base, mask=(q_offsets < S_len), other=0.0)
    
    D_base = D + bh * S_len + q_offsets
    D_i = tl.load(D_base, mask=(q_offsets < S_len), other=0.0)
    
    dQ_i = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)
    
    num_blocks_k = tl.cdiv(S_len, BLOCK_K)
    
    for k_idx in range(num_blocks_k):
        k_rows = tl.arange(0, BLOCK_K)
        k_cols = tl.arange(0, BLOCK_D)
        
        K_base = K + bh * H * S_len * d + k_idx * BLOCK_K * d
        K_j = tl.load(K_base + k_rows[:, None] * d + k_cols[None, :], mask=(k_rows[:, None] < S_len), other=0.0)
        
        V_base = V + bh * H * S_len * d + k_idx * BLOCK_K * d
        V_j = tl.load(V_base + k_rows[:, None] * d + k_cols[None, :], mask=(k_rows[:, None] < S_len), other=0.0)
        
        S = tau * tl.dot(Q_i, K_j.T)
        P = tl.exp(S - L_i[:, None])
        dP = tl.dot(dO_i, V_j.T)
        dS = P * (dP - D_i[:, None]) * tau
        
        dQ_i = tl.dot(dS, K_j, dQ_i)
    
    dQ_base = dQ + bh * H * S_len * d + i * BLOCK_Q * d
    dQ_ptrs = dQ_base + rows[:, None] * d + cols[None, :]
    dQ_i_bf16 = dQ_i.to(tl.bfloat16)
    tl.store(dQ_ptrs, dQ_i_bf16, mask=mask_q)


@triton.jit
def _bwd_key_kernel(
    Q, K, V, dO, L, D, dK, dV,
    S_len, B, H, d, tau,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    i = tl.program_id(0)
    dummy = tl.program_id(1)
    bh = tl.program_id(2)
    
    rows = tl.arange(0, BLOCK_K)
    cols = tl.arange(0, BLOCK_D)
    k_offsets = i * BLOCK_K + rows
    
    mask_k = k_offsets[:, None] < S_len
    
    K_base = K + bh * H * S_len * d + i * BLOCK_K * d
    K_j = tl.load(K_base + rows[:, None] * d + cols[None, :], mask=mask_k, other=0.0)
    
    V_base = V + bh * H * S_len * d + i * BLOCK_K * d
    V_j = tl.load(V_base + rows[:, None] * d + cols[None, :], mask=mask_k, other=0.0)
    
    dK_j = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)
    dV_j = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)
    
    num_blocks_q = tl.cdiv(S_len, BLOCK_Q)
    
    for k_idx in range(num_blocks_q):
        q_rows = tl.arange(0, BLOCK_Q)
        q_cols = tl.arange(0, BLOCK_D)
        q_offsets_k = k_idx * BLOCK_Q + q_rows
        
        mask_q = q_offsets_k[:, None] < S_len
        
        Q_base_k = Q + bh * H * S_len * d + k_idx * BLOCK_Q * d
        Q_k = tl.load(Q_base_k + q_rows[:, None] * d + q_cols[None, :], mask=mask_q, other=0.0)
        
        dO_base_k = dO + bh * H * S_len * d + k_idx * BLOCK_Q * d
        dO_k = tl.load(dO_base_k + q_rows[:, None] * d + q_cols[None, :], mask=mask_q, other=0.0)
        
        L_base_k = L + bh * S_len + q_offsets_k
        L_k = tl.load(L_base_k, mask=(q_offsets_k < S_len), other=0.0)
        
        D_base_k = D + bh * S_len + q_offsets_k
        D_k_buf = tl.load(D_base_k, mask=(q_offsets_k < S_len), other=0.0)
        
        O_base_k = O + bh * H * S_len * d + k_idx * BLOCK_Q * d
        O_k = tl.load(O_base_k + q_rows[:, None] * d + q_cols[None, :], mask=mask_q, other=0.0)
        D_k = tl.sum(dO_k * O_k, axis=1)
        
        S = tau * tl.dot(Q_k, K_j.T)
        P = tl.exp(S - L_k[:, None])
        dP = tl.dot(dO_k, V_j.T)
        dS = P * (dP - D_k[None, :]) * tau
        
        P_T = P.T
        dS_T = dS.T
        
        dV_j = tl.dot(P_T, dO_k, dV_j)
        dK_j = tl.dot(dS_T, Q_k, dK_j)
    
    dK_base = dK + bh * H * S_len * d + i * BLOCK_K * d
    dK_ptrs = dK_base + rows[:, None] * d + cols[None, :]
    dK_j_bf16 = dK_j.to(tl.bfloat16)
    tl.store(dK_ptrs, dK_j_bf16, mask=mask_k)
    
    dV_base = dV + bh * H * S_len * d + i * BLOCK_K * d
    dV_ptrs = dV_base + rows[:, None] * d + cols[None, :]
    dV_j_bf16 = dV_j.to(tl.bfloat16)
    tl.store(dV_ptrs, dV_j_bf16, mask=mask_k)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S_len = Q.shape[0], Q.shape[1], Q.shape[2]
    d = 128
    tau = 1.0 / (d ** 0.5)
    
    D = torch.empty((B, H, S_len), dtype=torch.float32, device=Q.device)
    
    grid_D = (B * H * S_len,)
    _precompute_D_kernel[grid_D](O, dO, D, S_len, B, H, d)
    
    dummy_dim = 1
    num_blocks_q = triton.cdiv(S_len, 64)
    grid_query = (num_blocks_q, dummy_dim, B * H)
    _bwd_query_kernel[grid_query](
        Q, K, V, dO, L, D, dQ, S_len, B, H, d, tau,
        BLOCK_Q=64, BLOCK_K=64, BLOCK_D=128,
        num_warps=4,
    )
    
    num_blocks_k = triton.cdiv(S_len, 64)
    grid_key = (num_blocks_k, dummy_dim, B * H)
    _bwd_key_kernel[grid_key](
        Q, K, V, dO, L, D, dK, dV, S_len, B, H, d, tau,
        BLOCK_Q=64, BLOCK_K=64, BLOCK_D=128,
        num_warps=4,
    )