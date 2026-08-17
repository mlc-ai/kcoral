import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _bwd_dKdV_kernel(
    K_desc, V_desc, Q_desc, dO_desc, O_desc, L_ptr, dK_desc, dV_desc,
    num_blocks_r, seq_len, num_heads, scale,
    BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """Compute dK and dV. Grid maps 1-to-1 with Key blocks."""
    pid_j = tl.program_id(1)
    b = tl.program_id(2)
    h = tl.program_id(3)
    
    offset_k = pid_j * BLOCK_C
    
    K0 = K_desc.load([b, h, offset_k, 0])
    K1 = K_desc.load([b, h, offset_k, 64])
    V0 = V_desc.load([b, h, offset_k, 0])
    V1 = V_desc.load([b, h, offset_k, 64])
    
    acc_dK0 = tl.zeros((BLOCK_C, 64), dtype=tl.float32)
    acc_dK1 = tl.zeros((BLOCK_C, 64), dtype=tl.float32)
    acc_dV0 = tl.zeros((BLOCK_C, 64), dtype=tl.float32)
    acc_dV1 = tl.zeros((BLOCK_C, 64), dtype=tl.float32)
    
    for i in range(pid_j, num_blocks_r):
        offset_q = i * BLOCK_R
        
        Q0 = Q_desc.load([b, h, offset_q, 0])
        Q1 = Q_desc.load([b, h, offset_q, 64])
        dO0 = dO_desc.load([b, h, offset_q, 0])
        dO1 = dO_desc.load([b, h, offset_q, 64])
        O0 = O_desc.load([b, h, offset_q, 0])
        O1 = O_desc.load([b, h, offset_q, 64])
        
        D_i_unmasked = tl.sum(dO0 * O0, axis=-1) + tl.sum(dO1 * O1, axis=-1)
        
        l_base = (b * num_heads + h) * seq_len + offset_q
        L_i_unmasked = tl.load(L_ptr + l_base + tl.arange(0, BLOCK_R), 
                               mask=(offset_q + tl.arange(0, BLOCK_R)) < seq_len, 
                               other=0.0)
        
        row_mask = ((offset_q + tl.arange(0, BLOCK_R)) < seq_len)
        D_i = D_i_unmasked * row_mask
        
        S = tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)
        dP = tl.dot(dO0, V0.T) + tl.dot(dO1, V1.T)
        
        row_idx = offset_q + tl.arange(0, BLOCK_R)[:, None]
        col_idx = offset_k + tl.arange(0, BLOCK_C)[None, :]
        valid = (row_idx >= col_idx) & (row_idx < seq_len) & (col_idx < seq_len)
        
        P = tl.exp(S * scale - L_i_unmasked[:, None])
        P = P * valid
        
        dS = P * (dP - D_i[:, None]) * scale
        dS = dS * valid
        
        acc_dV0 = tl.dot(P.T, dO0, acc_dV0)
        acc_dV1 = tl.dot(P.T, dO1, acc_dV1)
        acc_dK0 = tl.dot(dS.T, Q0, acc_dK0)
        acc_dK1 = tl.dot(dS.T, Q1, acc_dK1)
        
    dK_desc.store([b, h, offset_k, 0], acc_dK0.to(tl.bfloat16))
    dK_desc.store([b, h, offset_k, 64], acc_dK1.to(tl.bfloat16))
    dV_desc.store([b, h, offset_k, 0], acc_dV0.to(tl.bfloat16))
    dV_desc.store([b, h, offset_k, 64], acc_dV1.to(tl.bfloat16))


@triton.jit
def _bwd_dQ_kernel(
    Q_desc, dO_desc, O_desc, K_desc, V_desc, L_ptr, dQ_desc,
    num_blocks_c, seq_len, num_heads, scale,
    BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """Compute dQ. Grid maps 1-to-1 with Query blocks."""
    pid_i = tl.program_id(1)
    b = tl.program_id(2)
    h = tl.program_id(3)
    
    offset_q = pid_i * BLOCK_R
    
    Q0 = Q_desc.load([b, h, offset_q, 0])
    Q1 = Q_desc.load([b, h, offset_q, 64])
    dO0 = dO_desc.load([b, h, offset_q, 0])
    dO1 = dO_desc.load([b, h, offset_q, 64])
    O0 = O_desc.load([b, h, offset_q, 0])
    O1 = O_desc.load([b, h, offset_q, 64])
    
    D_i_unmasked = tl.sum(dO0 * O0, axis=-1) + tl.sum(dO1 * O1, axis=-1)
    row_mask = ((offset_q + tl.arange(0, BLOCK_R)) < seq_len)
    D_i = D_i_unmasked * row_mask
    
    l_base = (b * num_heads + h) * seq_len + offset_q
    L_i_unmasked = tl.load(L_ptr + l_base + tl.arange(0, BLOCK_R),
                           mask=row_mask,
                           other=0.0)
    
    acc_dQ0 = tl.zeros((BLOCK_R, 64), dtype=tl.float32)
    acc_dQ1 = tl.zeros((BLOCK_R, 64), dtype=tl.float32)
    
    for j in range(0, pid_i + 1):
        offset_k = j * BLOCK_C
        
        K0 = K_desc.load([b, h, offset_k, 0])
        K1 = K_desc.load([b, h, offset_k, 64])
        V0 = V_desc.load([b, h, offset_k, 0])
        V1 = V_desc.load([b, h, offset_k, 64])
        
        S = tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)
        dP = tl.dot(dO0, V0.T) + tl.dot(dO1, V1.T)
        
        row_idx = offset_q + tl.arange(0, BLOCK_R)[:, None]
        col_idx = offset_k + tl.arange(0, BLOCK_C)[None, :]
        valid = (row_idx >= col_idx) & (row_idx < seq_len) & (col_idx < seq_len)
        
        P = tl.exp(S * scale - L_i_unmasked[:, None])
        P = P * valid
        
        dS = P * (dP - D_i[:, None]) * scale
        dS = dS * valid
        
        acc_dQ0 = tl.dot(dS, K0, acc_dQ0)
        acc_dQ1 = tl.dot(dS, K1, acc_dQ1)
        
    dQ_desc.store([b, h, offset_q, 0], acc_dQ0.to(tl.bfloat16))
    dQ_desc.store([b, h, offset_q, 64], acc_dQ1.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Orchestrate the causal multi-head attention backward pass."""
    torch.cuda.set_device(Q.device)
    
    b, h, seq_len, head_dim = Q.shape
    num_heads = h
    scale = 1.0 / (head_dim ** 0.5)
    num_blocks = triton.cdiv(seq_len, 64)
    
    Q_desc = TensorDescriptor.from_tensor(Q, [1, 1, 64, 64])
    K_desc = TensorDescriptor.from_tensor(K, [1, 1, 64, 64])
    V_desc = TensorDescriptor.from_tensor(V, [1, 1, 64, 64])
    O_desc = TensorDescriptor.from_tensor(O, [1, 1, 64, 64])
    dO_desc = TensorDescriptor.from_tensor(dO, [1, 1, 64, 64])
    dQ_desc = TensorDescriptor.from_tensor(dQ, [1, 1, 64, 64])
    dK_desc = TensorDescriptor.from_tensor(dK, [1, 1, 64, 64])
    dV_desc = TensorDescriptor.from_tensor(dV, [1, 1, 64, 64])
    
    grid_3d = (1, num_blocks, b, num_heads)
    
    _bwd_dKdV_kernel[grid_3d](
        K_desc, V_desc, Q_desc, dO_desc, O_desc, L, dK_desc, dV_desc,
        num_blocks, seq_len, num_heads, scale,
        BLOCK_R=64, BLOCK_C=64,
        num_warps=8, num_stages=2
    )
    
    _bwd_dQ_kernel[grid_3d](
        Q_desc, dO_desc, O_desc, K_desc, V_desc, L, dQ_desc,
        num_blocks, seq_len, num_heads, scale,
        BLOCK_R=64, BLOCK_C=64,
        num_warps=8, num_stages=2
    )