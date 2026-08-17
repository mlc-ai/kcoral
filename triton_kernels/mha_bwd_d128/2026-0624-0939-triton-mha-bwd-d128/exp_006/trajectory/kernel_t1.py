import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _bwd_query_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, dQ_desc,
    L,
    S_len, tau,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr,
):
    i = tl.program_id(0)
    bh = tl.program_id(1)
    
    rows_q = tl.arange(0, BLOCK_Q)
    q_offsets = i * BLOCK_Q + rows_q
    
    mask_q = q_offsets[:, None] < S_len
    
    base_row_i = bh * S_len + i * 128
    Q_0 = Q_desc.load([base_row_i, 0])
    Q_1 = Q_desc.load([base_row_i, 64])
    
    dO_0 = dO_desc.load([base_row_i, 0])
    dO_1 = dO_desc.load([base_row_i, 64])
    
    O_0 = O_desc.load([base_row_i, 0])
    O_1 = O_desc.load([base_row_i, 64])
    
    D_i_unreduced = tl.sum(dO_0 * O_0 + dO_1 * O_1, axis=1)
    
    L_base = L + bh * S_len + q_offsets
    L_i = tl.load(L_base, mask=(q_offsets < S_len), other=0.0)
    
    dQ_i = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
    
    num_blocks_k = tl.cdiv(S_len, BLOCK_K)
    
    for k_idx in range(num_blocks_k):
        base_row_j = bh * S_len + k_idx * 128
        
        K_0 = K_desc.load([base_row_j, 0])
        K_1 = K_desc.load([base_row_j, 64])
        
        V_0 = V_desc.load([base_row_j, 0])
        V_1 = V_desc.load([base_row_j, 64])
        
        S_partial = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
        for Q_0_b, K_0_b in zip([Q_0, Q_1], [K_0, K_1]):
            S_partial = tl.dot(Q_0_b, K_0_b.T, S_partial)
        S = S_partial * tau
        
        P = tl.exp(S - L_i[:, None])
        
        dP_partial = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
        for dO_0_b, V_0_b in zip([dO_0, dO_1], [V_0, V_1]):
            dP_partial = tl.dot(dO_0_b, V_0_b.T, dP_partial)
        dP = dP_partial
        
        P_safe = P * mask_q
        dS = P_safe * (dP - D_i[:, None]) * tau
        
        dS_reshaped = tl.reshape(dS, (BLOCK_Q, BLOCK_K))
        
        dQ_partial = tl.zeros((BLOCK_Q, 64), dtype=tl.float32)
        for ds_split, k_split in zip([dS[:, :64], dS[:, 64:]], [K_0, K_1]):
            dQ_partial = tl.dot(ds_split, k_split, dQ_partial)
        dQ_i[:, :64] = dQ_partial
        
        dQ_partial_1 = tl.zeros((BLOCK_Q, 64), dtype=tl.float32)
        for ds_split, k_split in zip([dS[:, :64], dS[:, 64:]], [K_0, K_1]):
            dQ_partial_1 = tl.dot(ds_split, k_split, dQ_partial_1)
        dQ_i[:, 64:] = dQ_partial_1

    dQ_0 = dQ_i[:, :64].to(tl.bfloat16)
    dQ_1 = dQ_i[:, 64:].to(tl.bfloat16)
    
    Q_desc_dq.store([base_row_i, 0], dQ_0)
    Q_desc_dq.store([base_row_i, 64], dQ_1)


@triton.jit
def _bwd_key_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, dK_desc, dV_desc,
    L,
    S_len, tau,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr,
):
    i = tl.program_id(0)
    bh = tl.program_id(1)
    
    base_row_j = bh * S_len + i * 128
    
    K_0 = K_desc.load([base_row_j, 0])
    K_1 = K_desc.load([base_row_j, 64])
    
    V_0 = V_desc.load([base_row_j, 0])
    V_1 = V_desc.load([base_row_j, 64])
    
    dK_j_0 = tl.zeros((BLOCK_K, 64), dtype=tl.float32)
    dK_j_1 = tl.zeros((BLOCK_K, 64), dtype=tl.float32)
    dV_j_0 = tl.zeros((BLOCK_K, 64), dtype=tl.float32)
    dV_j_1 = tl.zeros((BLOCK_K, 64), dtype=tl.float32)
    
    num_blocks_q = tl.cdiv(S_len, BLOCK_Q)
    
    for q_idx in range(num_blocks_q):
        base_row_i = bh * S_len + q_idx * 128
        
        Q_0 = Q_desc.load([base_row_i, 0])
        Q_1 = Q_desc.load([base_row_i, 64])
        
        dO_0 = dO_desc.load([base_row_i, 0])
        dO_1 = dO_desc.load([base_row_i, 64])
        
        O_0 = O_desc.load([base_row_i, 0])
        O_1 = O_desc.load([base_row_i, 64])
        
        D_i_unreduced = tl.sum(dO_0 * O_0 + dO_1 * O_1, axis=1)
        
        q_offsets_local = q_idx * 128 + tl.arange(0, 128)
        L_base_k = L + bh * S_len + q_offsets_local
        L_k = tl.load(L_base_k, mask=(q_offsets_local < S_len), other=0.0)
        
        S_partial = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
        for Q_0_b, K_0_b in zip([Q_0, Q_1], [K_0, K_1]):
            S_partial = tl.dot(Q_0_b, K_0_b.T, S_partial)
        S = S_partial * tau
        
        P = tl.exp(S - L_k[:, None])
        
        dP_partial = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
        for dO_0_b, V_0_b in zip([dO_0, dO_1], [V_0, V_1]):
            dP_partial = tl.dot(dO_0_b, V_0_b.T, dP_partial)
        dP = dP_partial
        
        mask_q_in_loop = (q_offsets_local[:, None] < S_len)
        P_safe = P * mask_q_in_loop
        dS = P_safe * (dP - D_i_unreduced[:, None]) * tau
        
        dS_T = dS.T
        P_T = P.T
        
        dV_partial_0 = tl.zeros((BLOCK_K, 64), dtype=tl.float32)
        for p_split, do_split in zip([P_T[:, :64], P_T[:, 64:]], [dO_0, dO_1]):
            dV_partial_0 = tl.dot(p_split, do_split, dV_partial_0)
        dV_j_0 = dV_partial_0 
        
        dV_partial_1 = tl.zeros((BLOCK_K, 64), dtype=tl.float32)
        for p_split, do_split in zip([P_T[:, :64], P_T[:, 64:]], [dO_0, dO_1]):
            dV_partial_1 = tl.dot(p_split, do_split, dV_partial_1)
        dV_j_1 = dV_partial_1
        
        dK_partial_0 = tl.zeros((BLOCK_K, 64), dtype=tl.float32)
        for ds_split, q_split in zip([dS_T[:, :64], dS_T[:, 64:]], [Q_0, Q_1]):
            dK_partial_0 = tl.dot(ds_split, q_split, dK_partial_0)
        dK_j_0 = dK_partial_0
        
        dK_partial_1 = tl.zeros((BLOCK_K, 64), dtype=tl.float32)
        for ds_split, q_split in zip([dS_T[:, :64], dS_T[:, 64:]], [Q_0, Q_1]):
            dK_partial_1 = tl.dot(ds_split, q_split, dK_partial_1)
        dK_j_1 = dK_partial_1

    K_desc_dk.store([base_row_j, 0], dK_j_0.to(tl.bfloat16))
    K_desc_dk.store([base_row_j, 64], dK_j_1.to(tl.bfloat16))
    
    K_desc_dv.store([base_row_j, 0], dV_j_0.to(tl.bfloat16))
    K_desc_dv.store([base_row_j, 64], dV_j_1.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S_len = Q.shape[0], Q.shape[1], Q.shape[2]
    d = 128
    tau = 1.0 / (d ** 0.5)
    
    Q_desc = TensorDescriptor.from_tensor(Q, [128, 64])
    K_desc = TensorDescriptor.from_tensor(K, [128, 64])
    V_desc = TensorDescriptor.from_tensor(V, [128, 64])
    O_desc = TensorDescriptor.from_tensor(O, [128, 64])
    dO_desc = TensorDescriptor.from_tensor(dO, [128, 64])
    dQ_desc = TensorDescriptor.from_tensor(dQ, [128, 64])
    dK_desc = TensorDescriptor.from_tensor(dK, [128, 64])
    dV_desc = TensorDescriptor.from_tensor(dV, [128, 64])
    
    dummy_dim = 1
    num_blocks_q = triton.cdiv(S_len, 128)
    grid_query = (num_blocks_q, dummy_dim, B * H)
    _bwd_query_kernel[grid_query](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, dQ_desc,
        L, S_len, tau,
        BLOCK_Q=128, BLOCK_K=128,
        num_warps=8,
    )
    
    num_blocks_k = triton.cdiv(S_len, 128)
    grid_key = (num_blocks_k, dummy_dim, B * H)
    _bwd_key_kernel[grid_key](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, dK_desc, dV_desc,
        L, S_len, tau,
        BLOCK_Q=128, BLOCK_K=128,
        num_warps=8,
    )