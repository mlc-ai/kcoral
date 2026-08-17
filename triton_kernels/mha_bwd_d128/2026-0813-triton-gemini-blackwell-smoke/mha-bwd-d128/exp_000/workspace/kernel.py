import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _bwd_dq_kernel(
    q_desc, k_desc, v_desc, o_desc, do_desc, l_ptr, dq_desc,
    S, stride_lb, stride_lh, stride_ls, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr, H: tl.constexpr
):
    q_tile = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    
    q_tile_offset = q_tile * BLOCK_M
    
    # Load Q, dO, and O exactly once for the current query tile
    q = tl.reshape(q_desc.load([b, h, q_tile_offset, 0]), (BLOCK_M, d))
    do = tl.reshape(do_desc.load([b, h, q_tile_offset, 0]), (BLOCK_M, d))
    o = tl.reshape(o_desc.load([b, h, q_tile_offset, 0]), (BLOCK_M, d))
    
    offs_m = q_tile_offset + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    
    l_ptrs = l_ptr + b * stride_lb + h * stride_lh + offs_m * stride_ls
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    # Precompute row-wise delta for the backward pass
    delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
    
    dq = tl.zeros((BLOCK_M, d), dtype=tl.float32)
    
    num_kv_tiles = tl.cdiv(S, BLOCK_N)
    for kv_tile in tl.range(0, num_kv_tiles, num_stages=3):
        kv_tile_offset = kv_tile * BLOCK_N
        
        k = tl.reshape(k_desc.load([b, h, kv_tile_offset, 0]), (BLOCK_N, d))
        v = tl.reshape(v_desc.load([b, h, kv_tile_offset, 0]), (BLOCK_N, d))
        
        offs_n = kv_tile_offset + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        mask = mask_m[:, None] & mask_n[None, :]
        
        scores = tl.dot(q, k.T) * scale
        scores = tl.where(mask, scores, -float('inf'))
        
        # Fast exp2 equivalent of exp(scores - L)
        RCP_LN2 = 1.4426950408889634
        p = tl.exp2((scores - l[:, None]) * RCP_LN2)
        
        dp = tl.dot(do, v.T)
        ds = tl.where(mask, p * (dp - delta[:, None]) * scale, 0.0)
        
        dq += tl.dot(ds.to(tl.bfloat16), k)
        
    dq_desc.store([b, h, q_tile_offset, 0], tl.reshape(dq.to(tl.bfloat16), (1, 1, BLOCK_M, d)))


@triton.jit
def _bwd_dk_dv_kernel(
    q_desc, k_desc, v_desc, o_desc, do_desc, l_ptr, dk_desc, dv_desc,
    S, stride_lb, stride_lh, stride_ls, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr, H: tl.constexpr
):
    kv_tile = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    
    kv_tile_offset = kv_tile * BLOCK_N
    
    # Load K and V exactly once for the current kv_tile
    k = tl.reshape(k_desc.load([b, h, kv_tile_offset, 0]), (BLOCK_N, d))
    v = tl.reshape(v_desc.load([b, h, kv_tile_offset, 0]), (BLOCK_N, d))
    
    offs_n = kv_tile_offset + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    
    dk = tl.zeros((BLOCK_N, d), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, d), dtype=tl.float32)
    
    num_q_tiles = tl.cdiv(S, BLOCK_M)
    for q_tile in tl.range(0, num_q_tiles, num_stages=2):
        q_tile_offset = q_tile * BLOCK_M
        
        q = tl.reshape(q_desc.load([b, h, q_tile_offset, 0]), (BLOCK_M, d))
        do = tl.reshape(do_desc.load([b, h, q_tile_offset, 0]), (BLOCK_M, d))
        o = tl.reshape(o_desc.load([b, h, q_tile_offset, 0]), (BLOCK_M, d))
        
        offs_m = q_tile_offset + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        mask = mask_n[:, None] & mask_m[None, :]
        
        l_ptrs = l_ptr + b * stride_lb + h * stride_lh + offs_m * stride_ls
        l = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        # Inline delta computation to avoid separate TMA descriptor state
        delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
        
        scores_t = tl.dot(k, q.T) * scale
        scores_t = tl.where(mask, scores_t, -float('inf'))
        
        RCP_LN2 = 1.4426950408889634
        p_t = tl.exp2((scores_t - l[None, :]) * RCP_LN2)
        
        dv += tl.dot(p_t.to(tl.bfloat16), do)
        
        dp_t = tl.dot(v, do.T)
        ds_t = tl.where(mask, p_t * (dp_t - delta[None, :]) * scale, 0.0)
        
        dk += tl.dot(ds_t.to(tl.bfloat16), q)
        
    dk_desc.store([b, h, kv_tile_offset, 0], tl.reshape(dk.to(tl.bfloat16), (1, 1, BLOCK_N, d)))
    dv_desc.store([b, h, kv_tile_offset, 0], tl.reshape(dv.to(tl.bfloat16), (1, 1, BLOCK_N, d)))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes backward gradients for standard SDPA non-causal attention.
    It maps cleanly to native Blackwell Standard Triton via `TensorDescriptor` 
    with perfectly balanced split-ownership program instances, registering highly 
    tuned shapes to keep 228KiB SMEM bounds exactly tight with maximal stage pipelining.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    scale = 1.0 / (d ** 0.5)
    
    # ------------------------------------------------------------------
    # 1. dQ Exclusive Ownership Launch
    # ------------------------------------------------------------------
    BLOCK_M_DQ = 128
    BLOCK_N_DQ = 64
    num_q_tiles = triton.cdiv(S, BLOCK_M_DQ)
    
    q_desc_dq = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M_DQ, d])
    do_desc_dq = TensorDescriptor.from_tensor(dO, [1, 1, BLOCK_M_DQ, d])
    o_desc_dq = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M_DQ, d])
    dq_desc = TensorDescriptor.from_tensor(dQ, [1, 1, BLOCK_M_DQ, d])
    
    k_desc_dq = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N_DQ, d])
    v_desc_dq = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N_DQ, d])
    
    _bwd_dq_kernel[(num_q_tiles, B * H)](
        q_desc_dq, k_desc_dq, v_desc_dq, o_desc_dq, do_desc_dq, L, dq_desc,
        S, L.stride(0), L.stride(1), L.stride(2), scale,
        BLOCK_M=BLOCK_M_DQ, BLOCK_N=BLOCK_N_DQ, d=d, H=H,
        num_warps=8, num_stages=3
    )
    
    # ------------------------------------------------------------------
    # 2. dK, dV Exclusive Ownership Launch
    # ------------------------------------------------------------------
    BLOCK_M_DK = 128
    BLOCK_N_DK = 64
    num_kv_tiles = triton.cdiv(S, BLOCK_N_DK)
    
    q_desc_dk = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M_DK, d])
    do_desc_dk = TensorDescriptor.from_tensor(dO, [1, 1, BLOCK_M_DK, d])
    o_desc_dk = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M_DK, d])
    
    k_desc_dk = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N_DK, d])
    v_desc_dk = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N_DK, d])
    dk_desc = TensorDescriptor.from_tensor(dK, [1, 1, BLOCK_N_DK, d])
    dv_desc = TensorDescriptor.from_tensor(dV, [1, 1, BLOCK_N_DK, d])
    
    _bwd_dk_dv_kernel[(num_kv_tiles, B * H)](
        q_desc_dk, k_desc_dk, v_desc_dk, o_desc_dk, do_desc_dk, L, dk_desc, dv_desc,
        S, L.stride(0), L.stride(1), L.stride(2), scale,
        BLOCK_M=BLOCK_M_DK, BLOCK_N=BLOCK_N_DK, d=d, H=H,
        num_warps=8, num_stages=2
    )