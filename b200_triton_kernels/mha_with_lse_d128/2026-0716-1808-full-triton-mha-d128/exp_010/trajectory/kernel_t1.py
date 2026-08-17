import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _attention_kernel(
    q_desc,
    k_desc,
    v_desc,
    o_desc,
    lse_ptr,
    S_seq,
    D,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Optimized FlashAttention kernel for bf16 inputs targeting Hopper SM90.
    
    Computes Output O and Log-Sum-Exp (LSE) for non-causal multi-head attention.
    Uses a two-pass approach (QK^T followed by PV) pipelined through TMA descriptors.
    """
    row_start = tl.program_id(0) * BLOCK_M
    b_h = tl.program_id(1)
    
    m = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    out0 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    out1 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    
    q0 = q_desc.load([b_h * S_seq + row_start, 0], boundary_check=(0,))
    q1 = q_desc.load([b_h * S_seq + row_start, 64], boundary_check=(0,))
    
    scale = 1.0 / (D ** 0.5)
    
    num_kv_blocks = tl.cdiv(S_seq, BLOCK_N)
    
    row_idx = tl.arange(0, BLOCK_M)
    valid_rows = row_start + row_idx < S_seq
    
    if num_kv_blocks > 0:
        k0_0 = k_desc.load([b_h * S_seq + 0, 0], boundary_check=(0,))
        k1_0 = k_desc.load([b_h * S_seq + 0, 64], boundary_check=(0,))
        v0_0 = v_desc.load([b_h * S_seq + 0, 0], boundary_check=(0,))
        v1_0 = v_desc.load([b_h * S_seq + 0, 64], boundary_check=(0,))
        
    col_idx = tl.arange(0, BLOCK_N)
    
    for j in range(num_kv_blocks):
        kv_start = j * BLOCK_N
        
        if j < num_kv_blocks - 1:
            next_kv_start = (j + 1) * BLOCK_N
            k0_next = k_desc.load([b_h * S_seq + next_kv_start, 0], boundary_check=(0,))
            k1_next = k_desc.load([b_h * S_seq + next_kv_start, 64], boundary_check=(0,))
            v0_next = v_desc.load([b_h * S_seq + next_kv_start, 0], boundary_check=(0,))
            v1_next = v_desc.load([b_h * S_seq + next_kv_start, 64], boundary_check=(0,))
        
        acc_S = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        # Wait implicitly handled by compiler for TMA loads 
        acc_S = tl.dot(q0, k0_0.T, acc_S)
        acc_S = tl.dot(q1, k1_0.T, acc_S)
        
        S = acc_S * scale
        
        mask = (kv_start + col_idx[None, :]) < S_seq
        S = tl.where(mask, S, -float('inf'))
        
        m_prev = m
        row_max = tl.max(S, axis=1)
        m = tl.maximum(m_prev, row_max)
        
        P = tl.exp(S - m[None, :])
        P = tl.where(mask, P, 0.0)
        
        row_sum = tl.sum(P, axis=1)
        exp_scale = tl.exp(m_prev - m)
        exp_scale = tl.where(valid_rows, exp_scale, 0.0)
        l = l * exp_scale + row_sum
        
        out0 *= exp_scale[:, None]
        out1 *= exp_scale[:, None]
        
        out0 = tl.dot(P, v0_0, out0)
        out1 = tl.dot(P, v1_0, out1)
        
        if j < num_kv_blocks - 1:
            k0_0 = k0_next
            k1_0 = k1_next
            v0_0 = v0_next
            v1_0 = v1_next
            
    else:
        inv_l = tl.zeros((BLOCK_M,), dtype=tl.float32)
        
    inv_l = 1.0 / l
    inv_l = tl.where(valid_rows, inv_l, 0.0)
    out0 *= inv_l[:, None]
    out1 *= inv_l[:, None]
    
    o_desc.store([b_h * S_seq + row_start, 0], out0)
    o_desc.store([b_h * S_seq + row_start, 64], out1)
    
    lse = m + tl.log(l)
    tl.store(lse_ptr + (b_h * S_seq + row_start) + row_idx, lse, mask=valid_rows)


def run(Q, K, V, O, LSE):
    """Compute Non-causal Multi-Head Attention O and LSE into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    BLOCK_M = 128
    BLOCK_N = 128 
    
    q_desc = TensorDescriptor(Q, dims=[B * H * S, D], stride=[D, 1], elem_ty=Q.dtype, padding_option="zero")
    q_desc.block_shape = [BLOCK_M, 64]
    
    k_desc = TensorDescriptor(K, dims=[B * H * S, D], stride=[D, 1], elem_ty=K.dtype, padding_option="zero")
    k_desc.block_shape = [BLOCK_N, 64]
    
    v_desc = TensorDescriptor(V, dims=[B * H * S, D], stride=[D, 1], elem_ty=V.dtype, padding_option="zero")
    v_desc.block_shape = [BLOCK_N, 64]
    
    o_desc = TensorDescriptor(O, dims=[B * H * S, D], stride=[D, 1], elem_ty=O.dtype)
    o_desc.block_shape = [BLOCK_M, 64]
    
    grid = (triton.cdiv(S, BLOCK_M), B * H)
    _attention_kernel[grid](
        q_desc, k_desc, v_desc, o_desc, LSE,
        S, D, BLOCK_M, BLOCK_N,
        num_warps=4, num_stages=2
    )