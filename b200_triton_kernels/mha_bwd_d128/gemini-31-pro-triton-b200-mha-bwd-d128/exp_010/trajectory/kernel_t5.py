import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

@triton.jit
def bwd_kernel(
    q_desc, k_desc, v_desc, o_desc, do_desc, dq_desc,
    dK, dV, L,
    stride_dk_b, stride_dk_h, stride_dk_s, stride_dk_d,
    stride_dv_b, stride_dv_h, stride_dv_s, stride_dv_d,
    stride_l_b, stride_l_h, stride_l_s,
    B, H, S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b_idx = pid_bh // H
    h_idx = pid_bh % H
    
    m_start = pid_m * BLOCK_M
    
    # Load completely stationary M-block tiles via TMA 
    q_tile = tl.reshape(q_desc.load([b_idx, h_idx, m_start, 0]), (BLOCK_M, D))
    o_tile = tl.reshape(o_desc.load([b_idx, h_idx, m_start, 0]), (BLOCK_M, D))
    do_tile = tl.reshape(do_desc.load([b_idx, h_idx, m_start, 0]), (BLOCK_M, D))
    
    off_m = m_start + tl.arange(0, BLOCK_M)
    valid_m = off_m < S
    
    # Load 1D LSE log values with standard pointers
    l_ptr = L + b_idx * stride_l_b + h_idx * stride_l_h + off_m * stride_l_s
    l_tile = tl.load(l_ptr, mask=valid_m, other=0.0)
    
    # Precompute rowwise sum(dO * O, axis=1) once per M-block iteration
    delta = tl.sum(do_tile.to(tl.float32) * o_tile.to(tl.float32), axis=1)
    
    dq_acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    
    # Base pointers mapped for dK and dV sequence independent atomics
    off_n = tl.arange(0, BLOCK_N)
    off_d = tl.arange(0, D)
    dk_ptrs_base = dK + b_idx * stride_dk_b + h_idx * stride_dk_h + off_d[None, :] * stride_dk_d
    dv_ptrs_base = dV + b_idx * stride_dv_b + h_idx * stride_dv_h + off_d[None, :] * stride_dv_d
    
    num_n_blocks = tl.cdiv(S, BLOCK_N)
    for n_idx in tl.range(0, num_n_blocks, num_stages=3):
        n_start = n_idx * BLOCK_N
        
        # Pipelined loading of stream K and V tiles
        k_tile = tl.reshape(k_desc.load([b_idx, h_idx, n_start, 0]), (BLOCK_N, D))
        v_tile = tl.reshape(v_desc.load([b_idx, h_idx, n_start, 0]), (BLOCK_N, D))
        
        # S = Q @ K^T
        s_mat = tl.dot(q_tile, k_tile.T, out_dtype=tl.float32) * scale
        
        # Bounds masking
        curr_off_n = n_start + off_n
        valid_n = curr_off_n < S
        valid = valid_m[:, None] & valid_n[None, :]
        s_mat = tl.where(valid, s_mat, -float("inf"))
        
        # P = softmax(S)
        p_mat = tl.exp(s_mat - l_tile[:, None])
        
        # dP = dO @ V^T
        dp_mat = tl.dot(do_tile, v_tile.T, out_dtype=tl.float32)
        
        # dS = P * (dP - delta)
        ds_mat = p_mat * (dp_mat - delta[:, None]) * scale
        ds_mat = tl.where(valid, ds_mat, 0.0)
        
        # dQ = dS @ K
        dq_acc = tl.dot(ds_mat.to(q_tile.dtype), k_tile, acc=dq_acc)
        
        # Gather sequential partial outputs 
        dk_partial = tl.dot(ds_mat.T.to(q_tile.dtype), q_tile, out_dtype=tl.float32)
        dv_partial = tl.dot(p_mat.T.to(q_tile.dtype), do_tile, out_dtype=tl.float32)
        
        dk_ptrs = dk_ptrs_base + curr_off_n[:, None] * stride_dk_s
        dv_ptrs = dv_ptrs_base + curr_off_n[:, None] * stride_dv_s
        
        # Protect atomics against out-of-bounds writes using dimension specific masks
        tl.atomic_add(dk_ptrs, dk_partial.to(dK.dtype.element_ty), mask=valid_n[:, None])
        tl.atomic_add(dv_ptrs, dv_partial.to(dV.dtype.element_ty), mask=valid_n[:, None])
        
    # Store finalized fully deterministic dQ M-block back to memory
    dq_desc.store([b_idx, h_idx, m_start, 0], tl.reshape(dq_acc.to(q_tile.dtype), (1, 1, BLOCK_M, D)))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Optimized Flash-Attention Backward kernel using TMA and single-pass execution.
    It parallelizes over queries (M-dimension) and atomically accumulates to dK and dV, 
    halving the overall memory traffic and flops compared to a split-ownership approach.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    scale = 1.0 / math.sqrt(D)
    
    # Must explicitly zero out accumulated dK and dV buffers prior to atomic additions
    dK.zero_()
    dV.zero_()
    
    BLOCK_M = 128
    BLOCK_N = 64
    
    # Pre-configure tightly bound Tensor Descriptors
    q_desc = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M, D])
    k_desc = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N, D])
    v_desc = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N, D])
    o_desc = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M, D])
    do_desc = TensorDescriptor.from_tensor(dO, [1, 1, BLOCK_M, D])
    dq_desc = TensorDescriptor.from_tensor(dQ, [1, 1, BLOCK_M, D])
    
    grid = (triton.cdiv(S, BLOCK_M), B * H)
    
    bwd_kernel[grid](
        q_desc, k_desc, v_desc, o_desc, do_desc, dq_desc,
        dK, dV, L,
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        B, H, S, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D,
        num_warps=8, num_stages=3
    )