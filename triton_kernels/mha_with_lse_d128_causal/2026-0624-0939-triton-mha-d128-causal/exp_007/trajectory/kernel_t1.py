import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math


@triton.jit
def _mha_kernel(
    Q_desc, 
    K_desc, 
    V_desc, 
    O_desc, 
    LSE_ptr, 
    H, 
    S, 
    scale,
    BLOCK_Q: tl.constexpr, 
    BLOCK_K: tl.constexpr,
):
    batch_idx = tl.program_id(2)
    head_idx = tl.program_id(1)
    i = tl.program_id(0)
    
    offset_q = i * BLOCK_Q
    
    num_tiles_s = S // 64
    base_row = (batch_idx * H + head_idx) * num_tiles_s
    
    q_offset_0 = base_row + i
    q_offset_1 = base_row + i + num_tiles_s
    
    Q_i_0 = Q_desc.load_2d(q_offset_0, [0, 0])
    Q_i_1 = Q_desc.load_2d(q_offset_1, [0, 0])
    
    O_0 = tl.zeros((64, 64), tl.float32)
    O_1 = tl.zeros((64, 64), tl.float32)
    m = tl.full((64, 1), -float('inf'), tl.float32)
    l = tl.full((64, 1), 0.0, tl.float32)
    
    global_q = offset_q + tl.arange(0, 64)
    
    # --- Unmasked Phase: Strictly Lower Triangular ---
    for j in range(0, i):
        k_offset_0 = base_row + j
        k_offset_1 = base_row + j + num_tiles_s
        K_j_0 = K_desc.load_2d(k_offset_0, [0, 0])
        K_j_1 = K_desc.load_2d(k_offset_1, [0, 0])
        
        S_att_unm = (tl.dot(Q_i_0, K_j_0.T) + tl.dot(Q_i_1, K_j_1.T)) * scale 
        
        m_prev = m
        m = tl.maximum(m, tl.max(S_att_unm, axis=1, keepdims=True))
        P_unm = tl.exp(S_att_unm - m)
        
        l = l * tl.exp(m_prev - m) + tl.sum(P_unm, axis=1, keepdims=True)
        O_0 = O_0 * tl.exp(m_prev - m)
        O_1 = O_1 * tl.exp(m_prev - m)
        
        v_offset_0 = base_row + j
        v_offset_1 = base_row + j + num_tiles_s
        V_j_0 = V_desc.load_2d(v_offset_0, [0, 0])
        V_j_1 = V_desc.load_2d(v_offset_1, [0, 0])
        
        O_0 = O_0 + tl.dot(P_unm, V_j_0)
        O_1 = O_1 + tl.dot(P_unm, V_j_1)
        
    # --- Masked Phase: Exact Causal Boundaries ---
    j = i
    k_offset_0 = base_row + j
    k_offset_1 = base_row + j + num_tiles_s
    K_j_0 = K_desc.load_2d(k_offset_0, [0, 0])
    K_j_1 = K_desc.load_2d(k_offset_1, [0, 0])
    
    S_att_m = (tl.dot(Q_i_0, K_j_0.T) + tl.dot(Q_i_1, K_j_1.T)) * scale 
    
    row_idx = offset_q + tl.arange(0, 64)
    col_idx = offset_q + tl.arange(0, 64)
    valid_mask = (row_idx[:, None] >= col_idx[None, :]) & \
                 (row_idx[:, None] < S) & \
                 (col_idx[None, :] < S)
    
    S_att_m = tl.where(valid_mask, S_att_m, -float('inf'))
    
    m_prev = m
    m = tl.maximum(m, tl.max(S_att_m, axis=1, keepdims=True))
    P_m = tl.where(valid_mask, tl.exp(S_att_m - m), 0.0)
    
    l = l * tl.exp(m_prev - m) + tl.sum(P_m, axis=1, keepdims=True)
    O_0 = O_0 * tl.exp(m_prev - m)
    O_1 = O_1 * tl.exp(m_prev - m)
    
    v_offset_0 = base_row + j
    v_offset_1 = base_row + j + num_tiles_s
    V_j_0 = V_desc.load_2d(v_offset_0, [0, 0])
    V_j_1 = V_desc.load_2d(v_offset_1, [0, 0])
    
    O_0 = O_0 + tl.dot(P_m, V_j_0)
    O_1 = O_1 + tl.dot(P_m, V_j_1)
    
    # Final Normalization
    inv_l = 1.0 / l
    O_0 = O_0 * inv_l
    O_1 = O_1 * inv_l
    
    o_offset_0 = base_row + i
    o_offset_1 = base_row + i + num_tiles_s
    o_desc.store_2d(o_offset_0, [0, 0], O_0.to(tl.bfloat16), mask=(global_q[:, None] < S))
    o_desc.store_2d(o_offset_1, [0, 0], O_1.to(tl.bfloat16), mask=(global_q[:, None] < S))
    
    m_squeezed = m.squeeze(1)
    l_squeezed = l.squeeze(1)
    lse = m_squeezed + tl.log(l_squeezed)
    
    global_q = offset_q + tl.arange(0, 64)
    lse_ptr = LSE_ptr + batch_idx * H * S + head_idx * S + global_q
    tl.store(lse_ptr, lse, mask=(global_q < S))


def run(Q, K, V, O, LSE):
    """Optimized causal multi-head attention computation returning precise outputs and accurate LSE"""
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    scale = 1.0 / math.sqrt(D)
    
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)
    triton.set_allocator(alloc_fn)
    
    Q_desc = TensorDescriptor.from_tensor(Q.contiguous(), [64, 64])
    K_desc = TensorDescriptor.from_tensor(K.contiguous(), [64, 64])
    V_desc = TensorDescriptor.from_tensor(V.contiguous(), [64, 64])
    O_desc = TensorDescriptor.from_tensor(O.contiguous(), [64, 64])
    
    num_blocks_q = triton.cdiv(S, 64)
    grid = (num_blocks_q, H, B)
    
    _mha_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, LSE,
        H, S, scale,
        BLOCK_Q=64, BLOCK_K=64,
        num_warps=4, num_stages=4,
    )