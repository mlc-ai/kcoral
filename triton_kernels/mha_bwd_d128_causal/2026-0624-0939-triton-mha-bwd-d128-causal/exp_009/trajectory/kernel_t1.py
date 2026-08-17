import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _precompute_D_kernel(p_dO, p_O, p_D, n_elements, BLOCK: tl.constexpr):
    start_idx = tl.program_id(0) * BLOCK
    idx = start_idx + tl.arange(0, BLOCK)
    mask = idx < n_elements
    d_offs = tl.arange(0, 128)
    d_offsets = idx[:, None] * 128 + d_offs[None, :]
    dO_val = tl.load(p_dO + d_offsets, mask=mask[:, None], other=0.0)
    O_val = tl.load(p_O + d_offsets, mask=mask[:, None], other=0.0)
    D_val = tl.sum(dO_val * O_val, axis=1)
    tl.store(p_D + idx, D_val, mask=mask)


@triton.jit
def _bwd_dq_kernel(
    desc_Q, desc_K, desc_V, desc_dO, desc_dQ,
    p_L, p_D,
    S, tau,
    c_L_b, c_L_h,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    i = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    
    off_q = i * BLOCK_M
    
    Q_i = desc_Q.load([b, h, off_q, 0])
    dO_i = desc_dO.load([b, h, off_q, 0])
    
    row_offs = off_q + tl.arange(0, BLOCK_M)
    mask_l = row_offs < S
    L_i = tl.load(p_L + b * c_L_b + h * c_L_h + row_offs, mask=mask_l, other=0.0)
    D_i = tl.load(p_D + b * c_L_b + h * c_L_h + row_offs, mask=mask_l, other=0.0)
    
    dQ_i = tl.zeros((BLOCK_M, 128), tl.float32)
    
    q_pos = off_q + tl.arange(0, BLOCK_M)
    
    for j in range(i + 1):
        off_k = j * BLOCK_N
        K_j = desc_K.load([b, h, off_k, 0])
        V_j = desc_V.load([b, h, off_k, 0])
        
        raw_S = tl.dot(Q_i, K_j.T)
        raw_dP = tl.dot(dO_i, V_j.T)
        
        k_pos = off_k + tl.arange(0, BLOCK_N)
        
        diag_mask = tl.full((BLOCK_M, BLOCK_N), fill_value=0.0, dtype=tl.float32)
        valid = (q_pos[:, None] >= k_pos[None, :]) & (q_pos[:, None] < S) & (k_pos[None, :] < S)
        invalid = ~valid
        diag_mask = tl.where(invalid, float('nan'), diag_mask)
        
        S_mat = raw_S * tau + diag_mask
        P = tl.exp(S_mat - L_i[:, None])
        dS = P * (raw_dP - D_i[:, None]) * tau
        
        dS = dS * diag_mask
        
        dQ_i = tl.dot(dS, K_j, acc=dQ_i)
    
    desc_dQ.store([b, h, off_q, 0], dQ_i.to(tl.bfloat16), boundary_check=(2,))


@triton.jit
def _bwd_dkv_kernel(
    desc_Q, desc_K, desc_V, desc_dO, desc_dK, desc_dV,
    p_L, p_D,
    S, tau,
    c_L_b, c_L_h,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    j = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    
    num_blocks_r = tl.cdiv(S, BLOCK_M)
    off_k = j * BLOCK_N
    
    K_j = desc_K.load([b, h, off_k, 0])
    V_j = desc_V.load([b, h, off_k, 0])
    
    dK_j = tl.zeros((BLOCK_N, 128), tl.float32)
    dV_j = tl.zeros((BLOCK_N, 128), tl.float32)
    
    k_pos = off_k + tl.arange(0, BLOCK_N)
    
    for i in range(j, num_blocks_r):
        off_q = i * BLOCK_M
        Q_i = desc_Q.load([b, h, off_q, 0])
        dO_i = desc_dO.load([b, h, off_q, 0])
        
        row_offs = off_q + tl.arange(0, BLOCK_M)
        mask_l = row_offs < S
        L_i = tl.load(p_L + b * c_L_b + h * c_L_h + row_offs, mask=mask_l, other=0.0)
        D_i = tl.load(p_D + b * c_L_b + h * c_L_h + row_offs, mask=mask_l, other=0.0)
        
        raw_S = tl.dot(Q_i, K_j.T)
        raw_dP = tl.dot(dO_i, V_j.T)
        
        q_pos = off_q + tl.arange(0, BLOCK_M)
        
        diag_mask = tl.full((BLOCK_M, BLOCK_N), fill_value=0.0, dtype=tl.float32)
        valid = (q_pos[:, None] >= k_pos[None, :]) & (q_pos[:, None] < S) & (k_pos[None, :] < S)
        invalid = ~valid
        diag_mask = tl.where(invalid, float('nan'), diag_mask)
        
        S_mat = raw_S * tau + diag_mask
        P = tl.exp(S_mat - L_i[:, None])
        dS = P * (raw_dP - D_i[:, None]) * tau
        
        P_masked = P * diag_mask
        dS_masked = dS * diag_mask
        
        dV_j = tl.dot(P_masked.T, dO_i, acc=dV_j)
        dK_j = tl.dot(dS_masked.T, Q_i, acc=dK_j)
    
    desc_dK.store([b, h, off_k, 0], dK_j.to(tl.bfloat16), boundary_check=(2,))
    desc_dV.store([b, h, off_k, 0], dV_j.to(tl.bfloat16), boundary_check=(2,))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    assert d == 128
    n_elements = B * H * S
    tau = 1.0 / (d ** 0.5)
    
    D = torch.empty((B, H, S), dtype=torch.float32, device=Q.device)
    
    BLOCK_M = 128
    BLOCK_N = 128
    
    grid_D = (triton.cdiv(n_elements, 32),)
    _precompute_D_kernel[grid_D](
        dO.data_ptr(), O.data_ptr(), D.data_ptr(),
        n_elements,
        BLOCK=32,
    )
    
    desc_Q = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M, d])
    desc_K = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N, d])
    desc_V = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N, d])
    desc_dO = TensorDescriptor.from_tensor(dO, [1, 1, BLOCK_M, d])
    desc_dQ = TensorDescriptor.from_tensor(dQ, [1, 1, BLOCK_M, d])
    desc_dK = TensorDescriptor.from_tensor(dK, [1, 1, BLOCK_N, d])
    desc_dV = TensorDescriptor.from_tensor(dV, [1, 1, BLOCK_N, d])
    
    num_blocks_m = triton.cdiv(S, BLOCK_M)
    num_blocks_n = triton.cdiv(S, BLOCK_N)
    
    grid_dQ = (num_blocks_m, B, H)
    _bwd_dq_kernel[grid_dQ](
        desc_Q, desc_K, desc_V, desc_dO, desc_dQ,
        L.data_ptr(), D.data_ptr(),
        S, tau,
        H * S, S,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_stages=3,
    )
    
    grid_dKV = (num_blocks_n, B, H)
    _bwd_dkv_kernel[grid_dKV](
        desc_Q, desc_K, desc_V, desc_dO, desc_dK, desc_dV,
        L.data_ptr(), D.data_ptr(),
        S, tau,
        H * S, S,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_stages=3,
    )