import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _bwd_dKdV_kernel(
    K_desc, V_desc, Q_desc, dO_desc, O_desc, L_ptr, dK_desc, dV_desc,
    num_blocks_r, seq_len, scale,
    BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """Compute dK and dV."""
    pid_j = tl.program_id(0)
    b_h = tl.program_id(2)
    
    offset_k = pid_j * BLOCK_C
    
    K_tile = K_desc.load([b_h, offset_k, 0])
    V_tile = V_desc.load([b_h, offset_k, 0])
    
    acc_dK = tl.zeros((BLOCK_C, 128), dtype=tl.float32)
    acc_dV = tl.zeros((BLOCK_C, 128), dtype=tl.float32)
    
    for i in range(pid_j, num_blocks_r):
        offset_q = i * BLOCK_R
        
        Q_tile = Q_desc.load([b_h, offset_q, 0])
        dO_tile = dO_desc.load([b_h, offset_q, 0])
        O_tile = O_desc.load([b_h, offset_q, 0])
        
        D_i = tl.sum(dO_tile * O_tile, axis=-1)
        
        l_base = b_h * seq_len + offset_q
        L_i = tl.load(L_ptr + l_base + tl.arange(0, BLOCK_R), 
                       mask=(offset_q + tl.arange(0, BLOCK_R)) < seq_len, 
                       other=0.0)
        
        S = tl.dot(Q_tile, K_tile.T) 
        dP = tl.dot(dO_tile, V_tile.T) 
        
        row_idx = offset_q + tl.arange(0, BLOCK_R)[:, None]
        col_idx = offset_k + tl.arange(0, BLOCK_C)[None, :]
        valid = (row_idx >= col_idx) & (row_idx < seq_len) & (col_idx < seq_len)
        
        P = tl.exp(S * scale - L_i[:, None])
        P = P * valid
        
        dS = P * (dP - D_i[:, None]) * scale
        dS = dS * valid
        
        acc_dV = tl.dot(P.T, dO_tile, acc_dV)
        acc_dK = tl.dot(dS.T, Q_tile, acc_dK)
        
    dK_desc.store([b_h, offset_k, 0], acc_dK.to(tl.bfloat16))
    dV_desc.store([b_h, offset_k, 0], acc_dV.to(tl.bfloat16))


@triton.jit
def _bwd_dQ_kernel(
    Q_desc, dO_desc, O_desc, K_desc, V_desc, L_ptr, dQ_desc,
    num_blocks_c, seq_len, scale,
    BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """Compute dQ."""
    pid_i = tl.program_id(0)
    b_h = tl.program_id(2)
    
    offset_q = pid_i * BLOCK_R
    
    Q_tile = Q_desc.load([b_h, offset_q, 0])
    dO_tile = dO_desc.load([b_h, offset_q, 0])
    O_tile = O_desc.load([b_h, offset_q, 0])
    
    D_i = tl.sum(dO_tile * O_tile, axis=-1)
    
    l_base = b_h * seq_len + offset_q
    L_i = tl.load(L_ptr + l_base + tl.arange(0, BLOCK_R),
                   mask=(offset_q + tl.arange(0, BLOCK_R)) < seq_len,
                   other=0.0)
    
    acc_dQ = tl.zeros((BLOCK_R, 128), dtype=tl.float32)
    
    for j in range(0, pid_i + 1):
        offset_k = j * BLOCK_C
        
        K_tile = K_desc.load([b_h, offset_k, 0])
        V_tile = V_desc.load([b_h, offset_k, 0])
        
        S = tl.dot(Q_tile, K_tile.T)
        dP = tl.dot(dO_tile, V_tile.T)
        
        row_idx = offset_q + tl.arange(0, BLOCK_R)[:, None]
        col_idx = offset_k + tl.arange(0, BLOCK_C)[None, :]
        valid = (row_idx >= col_idx) & (row_idx < seq_len) & (col_idx < seq_len)
        
        P = tl.exp(S * scale - L_i[:, None])
        P = P * valid
        
        dS = P * (dP - D_i[:, None]) * scale
        dS = dS * valid
        
        acc_dQ = tl.dot(dS, K_tile, acc_dQ)
        
    dQ_desc.store([b_h, offset_q, 0], acc_dQ.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Orchestrate the causal multi-head attention backward pass."""
    torch.cuda.set_device(Q.device)
    
    b, h, seq_len, head_dim = Q.shape
    scale = 1.0 / (head_dim ** 0.5)
    
    Q_c = Q.contiguous().view(b * h, seq_len, 128)
    K_c = K.contiguous().view(b * h, seq_len, 128)
    V_c = V.contiguous().view(b * h, seq_len, 128)
    O_c = O.contiguous().view(b * h, seq_len, 128)
    dO_c = dO.contiguous().view(b * h, seq_len, 128)
    dQ_c = dQ.contiguous().view(b * h, seq_len, 128)
    dK_c = dK.contiguous().view(b * h, seq_len, 128)
    dV_c = dV.contiguous().view(b * h, seq_len, 128)
    
    Q_desc = TensorDescriptor.from_tensor(Q_c, [1, 64, 128])
    K_desc = TensorDescriptor.from_tensor(K_c, [1, 64, 128])
    V_desc = TensorDescriptor.from_tensor(V_c, [1, 64, 128])
    O_desc = TensorDescriptor.from_tensor(O_c, [1, 64, 128])
    dO_desc = TensorDescriptor.from_tensor(dO_c, [1, 64, 128])
    dQ_desc = TensorDescriptor.from_tensor(dQ_c, [1, 64, 128])
    dK_desc = TensorDescriptor.from_tensor(dK_c, [1, 64, 128])
    dV_desc = TensorDescriptor.from_tensor(dV_c, [1, 64, 128])
    
    num_blocks_r = triton.cdiv(seq_len, 64)
    num_blocks_c = triton.cdiv(seq_len, 64)
    
    grid_dKdV = (num_blocks_c, 1, b * h)
    _bwd_dKdV_kernel[grid_dKdV](
        K_desc, V_desc, Q_desc, dO_desc, O_desc, L, dK_desc, dV_desc,
        num_blocks_r, seq_len, scale,
        BLOCK_R=64, BLOCK_C=64,
        num_warps=4, num_stages=3
    )
    
    grid_dQ = (num_blocks_r, 1, b * h)
    _bwd_dQ_kernel[grid_dQ](
        Q_desc, dO_desc, O_desc, K_desc, V_desc, L, dQ_desc,
        num_blocks_c, seq_len, scale,
        BLOCK_R=64, BLOCK_C=64,
        num_warps=4, num_stages=3
    )