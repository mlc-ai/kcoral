import torch
import triton
import triton.language as tl
import math
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def bwd_dQ_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L_ptr, dQ_ptr,
    S, scale, num_blocks, stride_dQ_b, BLOCK_D: tl.constexpr
):
    bh = tl.program_id(1)
    q_blk = tl.program_id(0)
    q_offset = q_blk * 128
    
    Q0 = Q_desc.load([0, bh, q_offset, 0])
    Q1 = Q_desc.load([0, bh, q_offset, BLOCK_D])
    
    O0 = O_desc.load([0, bh, q_offset, 0])
    O1 = O_desc.load([0, bh, q_offset, BLOCK_D])
    
    dO0 = dO_desc.load([0, bh, q_offset, 0])
    dO1 = dO_desc.load([0, bh, q_offset, BLOCK_D])
    
    D = tl.sum(O0 * dO0, axis=-1, keep_dims=True) + tl.sum(O1 * dO1, axis=-1, keep_dims=True)
    
    q_rows = q_offset + tl.arange(0, 128)
    L = tl.load(L_ptr + bh * S + q_rows, mask=q_rows < S, other=0.0)
    
    dQ0 = 0.0
    dQ1 = 0.0
    
    for k_blk in range(0, q_blk + 1):
        k_offset = k_blk * 128
        
        K0 = K_desc.load([0, bh, k_offset, 0])
        K1 = K_desc.load([0, bh, k_offset, BLOCK_D])
        
        V0 = V_desc.load([0, bh, k_offset, 0])
        V1 = V_desc.load([0, bh, k_offset, BLOCK_D])
        
        S_mat = tl.dot(Q0, K0.T)
        S_mat = tl.dot(Q1, K1.T, acc=S_mat)
        
        d_P = tl.dot(dO0, V0.T)
        d_P = tl.dot(dO1, V1.T, acc=d_P)
        
        k_rows = k_offset + tl.arange(0, 128)
        mask = (q_rows[:, None] >= k_rows[None, :])
        
        S_valid = S_mat * scale - L[None, :]
        P_ij = tl.where(mask, tl.exp(S_valid), 0.0)
            
        dS = P_ij * (d_P - D) * scale
        
        dQ0 = tl.dot(dS, K0, acc=dQ0)
        dQ1 = tl.dot(dS, K1, acc=dQ1)
        
    ptr_Q0 = dQ_ptr + bh * stride_dQ_b + q_rows[:, None] * 128 + tl.arange(0, BLOCK_D)[None, :]
    ptr_Q1 = dQ_ptr + bh * stride_dQ_b + q_rows[:, None] * 128 + BLOCK_D + tl.arange(0, BLOCK_D)[None, :]
    
    tl.store(ptr_Q0, dQ0.to(tl.bfloat16), mask=q_rows[:, None] < S)
    tl.store(ptr_Q1, dQ1.to(tl.bfloat16), mask=q_rows[:, None] < S)


@triton.jit
def bwd_dKV_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L_ptr, dK_ptr, dV_ptr,
    S, scale, num_blocks, stride_dK_b, stride_dV_b, BLOCK_D: tl.constexpr
):
    bh = tl.program_id(1)
    k_blk = tl.program_id(0)
    k_offset = k_blk * 128
    
    k_rows = k_offset + tl.arange(0, 128)
    
    K0 = K_desc.load([0, bh, k_offset, 0])
    K1 = K_desc.load([0, bh, k_offset, BLOCK_D])
    
    V0 = V_desc.load([0, bh, k_offset, 0])
    V1 = V_desc.load([0, bh, k_offset, BLOCK_D])
    
    d_K0 = 0.0
    d_K1 = 0.0
    d_V0 = 0.0
    d_V1 = 0.0
    
    for q_blk in range(k_blk, num_blocks):
        q_offset = q_blk * 128
        
        Q0 = Q_desc.load([0, bh, q_offset, 0])
        Q1 = Q_desc.load([0, bh, q_offset, BLOCK_D])
        
        O0 = O_desc.load([0, bh, q_offset, 0])
        O1 = O_desc.load([0, bh, q_offset, BLOCK_D])
        
        dO0 = dO_desc.load([0, bh, q_offset, 0])
        dO1 = dO_desc.load([0, bh, q_offset, BLOCK_D])
        
        D = tl.sum(O0 * dO0, axis=-1, keep_dims=True) + tl.sum(O1 * dO1, axis=-1, keep_dims=True)
        
        q_rows = q_offset + tl.arange(0, 128)
        L = tl.load(L_ptr + bh * S + q_rows, mask=q_rows < S, other=0.0)
        
        S_mat = tl.dot(Q0, K0.T)
        S_mat = tl.dot(Q1, K1.T, acc=S_mat)
        
        d_P = tl.dot(dO0, V0.T)
        d_P = tl.dot(dO1, V1.T, acc=d_P)
        
        mask = (q_rows[:, None] >= k_rows[None, :])
        
        S_valid = S_mat * scale - L[None, :]
        P_ij = tl.where(mask, tl.exp(S_valid), 0.0)
            
        dS = P_ij * (d_P - D) * scale
        
        d_K0 = tl.dot(dS.T, Q0, acc=d_K0)
        d_K1 = tl.dot(dS.T, Q1, acc=d_K1)
        
        d_V0 = tl.dot(P_ij.T, dO0, acc=d_V0)
        d_V1 = tl.dot(P_ij.T, dO1, acc=d_V1)
        
    ptr_K0 = dK_ptr + bh * stride_dK_b + k_rows[:, None] * 128 + tl.arange(0, BLOCK_D)[None, :]
    ptr_K1 = dK_ptr + bh * stride_dK_b + k_rows[:, None] * 128 + BLOCK_D + tl.arange(0, BLOCK_D)[None, :]
    ptr_V0 = dV_ptr + bh * stride_dV_b + k_rows[:, None] * 128 + tl.arange(0, BLOCK_D)[None, :]
    ptr_V1 = dV_ptr + bh * stride_dV_b + k_rows[:, None] * 128 + BLOCK_D + tl.arange(0, BLOCK_D)[None, :]
    
    tl.store(ptr_K0, d_K0.to(tl.bfloat16), mask=k_rows[:, None] < S)
    tl.store(ptr_K1, d_K1.to(tl.bfloat16), mask=k_rows[:, None] < S)
    tl.store(ptr_V0, d_V0.to(tl.bfloat16), mask=k_rows[:, None] < S)
    tl.store(ptr_V1, d_V1.to(tl.bfloat16), mask=k_rows[:, None] < S)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    scale = 1.0 / math.sqrt(d)
    
    BLOCK_D = 64
    num_blocks = triton.cdiv(S, 128)
    grid = (num_blocks, B * H)
    
    Q_desc = TensorDescriptor.from_tensor(Q, [1, 1, 128, BLOCK_D])
    K_desc = TensorDescriptor.from_tensor(K, [1, 1, 128, BLOCK_D])
    V_desc = TensorDescriptor.from_tensor(V, [1, 1, 128, BLOCK_D])
    O_desc = TensorDescriptor.from_tensor(O, [1, 1, 128, BLOCK_D])
    dO_desc = TensorDescriptor.from_tensor(dO, [1, 1, 128, BLOCK_D])
    
    stride_dQ_b = dQ.stride(1)
    stride_dK_b = dK.stride(1)
    stride_dV_b = dV.stride(1)
    
    bwd_dKV_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dK_ptr=dK, dV_ptr=dV, 
        S=S, scale=scale, num_blocks=num_blocks,
        stride_dK_b=stride_dK_b, stride_dV_b=stride_dV_b, BLOCK_D=BLOCK_D, num_stages=3
    )
    
    bwd_dQ_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dQ_ptr=dQ, 
        S=S, scale=scale, num_blocks=num_blocks, stride_dQ_b=stride_dQ_b, BLOCK_D=BLOCK_D, num_stages=3
    )