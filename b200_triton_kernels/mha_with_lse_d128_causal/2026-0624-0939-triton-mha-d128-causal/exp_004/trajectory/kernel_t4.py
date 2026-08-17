import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_fwd_kernel(
    Q_desc, K_desc, V_desc, O_desc, LSE_ptr,
    H, S_len, D, scale,
    stride_LSE_b, stride_LSE_h, stride_LSE_s,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    bh_idx = tl.program_id(0)
    step = tl.program_id(1)
    row = step * BLOCK_M
    
    if row >= S_len:
        return
    
    b = bh_idx // H
    h = bh_idx % H
    b_h_offset = bh_idx * S_len
    
    smem_Q = tl.empty((BLOCK_M, D), tl.bfloat16)
    smem_K_0 = tl.empty((BLOCK_N, D), tl.bfloat16)
    smem_K_1 = tl.empty((BLOCK_N, D), tl.bfloat16)
    smem_V_0 = tl.empty((BLOCK_N, D), tl.bfloat16)
    smem_V_1 = tl.empty((BLOCK_N, D), tl.bfloat16)
    
    Q_desc.load([b_h_offset + row, 0], storage=smem_Q)
    tl.async_wait(1)
    
    acc_o = tl.zeros((BLOCK_M, D), tl.float32)
    m = tl.full((BLOCK_M,), -float('inf'), tl.float32)
    l = tl.full((BLOCK_M,), 0.0, tl.float32)
    
    j_max_causal = (row + BLOCK_M + BLOCK_N - 1) // BLOCK_N
    j_max_len = (S_len + BLOCK_N - 1) // BLOCK_N
    j_max = min(j_max_causal, j_max_len)
    
    if j_max > 0:
        K_desc.load([b_h_offset + 0, 0], storage=smem_K_0)
        V_desc.load([b_h_offset + 0, 0], storage=smem_V_0)
    
    for j in range(j_max):
        stage = j % 2
        col = j * BLOCK_N
        
        if j + 1 < j_max:
            next_stage = (j + 1) % 2
            next_col = (j + 1) * BLOCK_N
            if next_stage == 0:
                K_desc.load([b_h_offset + next_col, 0], storage=smem_K_0)
                V_desc.load([b_h_offset + next_col, 0], storage=smem_V_0)
            else:
                K_desc.load([b_h_offset + next_col, 0], storage=smem_K_1)
                V_desc.load([b_h_offset + next_col, 0], storage=smem_V_1)
        
        tl.async_wait(2)
        
        if stage == 0:
            K_smem = smem_K_0
            V_smem = smem_V_0
        else:
            K_smem = smem_K_1
            V_smem = smem_V_1
            
        S = tl.dot(smem_Q, K_smem.T) * scale
        
        q_indices = row + tl.arange(0, BLOCK_M)
        k_indices = col + tl.arange(0, BLOCK_N)
        mask = (k_indices[None, :] <= q_indices[:, None]) & \
               (k_indices[None, :] < S_len) & \
               (q_indices[:, None] < S_len)
        S = tl.where(mask, S, -float('inf'))
        
        m_old = m
        m_new = tl.maximum(m, tl.max(S, axis=1))
        P = tl.exp(S - m_new)
        l = l * tl.exp(m_old - m_new) + tl.sum(P, axis=1)
        
        acc_o = acc_o * tl.exp(m_old - m_new)[:, None] + tl.dot(P, V_smem)
        m = m_new
    
    O = acc_o / l[:, None]
    valid = l > 0
    O = tl.where(valid[:, None], O, 0.0)
    O_desc.store([b_h_offset + row, 0], O.to(tl.bfloat16))
    
    LSE_val = tl.where(l > 0, m + tl.log(l), -float('inf'))
    off_row = tl.arange(0, BLOCK_M)
    ptr_LSE = LSE_ptr + b * stride_LSE_b + h * stride_LSE_h + (row + off_row) * stride_LSE_s
    mask_LSE = (row + off_row) < S_len
    tl.store(ptr_LSE, LSE_val, mask=mask_LSE)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S_len, D = Q.shape
    
    Q_2d = Q.view(B * H * S_len, D)
    K_2d = K.view(B * H * S_len, D)
    V_2d = V.view(B * H * S_len, D)
    O_2d = O.view(B * H * S_len, D)
    
    BLOCK_M = 64
    BLOCK_N = 64
    
    Q_desc = TensorDescriptor.from_tensor(Q_2d, [BLOCK_M, D])
    K_desc = TensorDescriptor.from_tensor(K_2d, [BLOCK_N, D])
    V_desc = TensorDescriptor.from_tensor(V_2d, [BLOCK_N, D])
    O_desc = TensorDescriptor.from_tensor(O_2d, [BLOCK_M, D])
    
    stride_LSE_b = LSE.stride(0)
    stride_LSE_h = LSE.stride(1)
    stride_LSE_s = LSE.stride(2)
    
    num_q_tiles = triton.cdiv(S_len, BLOCK_M)
    grid = (B * H, num_q_tiles)
    scale = 1.0 / (D ** 0.5)
    
    _mha_fwd_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, LSE,
        H, S_len, D, scale,
        stride_LSE_b, stride_LSE_h, stride_LSE_s,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )