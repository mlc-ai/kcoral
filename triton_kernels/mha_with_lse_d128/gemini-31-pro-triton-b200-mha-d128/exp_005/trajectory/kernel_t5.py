import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        # TMA Configurations (Small stages for 128x128 to prevent SMEM bounds errors on SM100)
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "USE_TMA": True}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "USE_TMA": True}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "USE_TMA": True}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "USE_TMA": True}, num_warps=8, num_stages=3),
        
        # Pointer Configurations (Highly predictable pipeline lowering)
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "USE_TMA": False}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "USE_TMA": False}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "USE_TMA": False}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "USE_TMA": False}, num_warps=8, num_stages=3),
    ],
    key=["S"],
)
@triton.jit
def _mha_fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    S, H, scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
    USE_TMA: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
):
    # L2 Cache Broadcasting Swizzle: Consecutive PIDs map to the same Batch/Head to maximally reuse identical K & V sequences
    pid = tl.program_id(0)
    grid_m = tl.cdiv(S, BLOCK_M)
    
    pid_bh = pid // grid_m
    pid_m = pid % grid_m
    
    b = pid_bh // H
    h = pid_bh % H
    
    start_m = pid_m * BLOCK_M
    
    q_base = Q + b * stride_qb + h * stride_qh
    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh
    o_base = O + b * stride_ob + h * stride_oh
    
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    
    if USE_TMA:
        q_desc = tl.make_tensor_descriptor(q_base, shape=[S, D], strides=[stride_qs, 1], block_shape=[BLOCK_M, D], padding_option="zero")
        k_desc = tl.make_tensor_descriptor(k_base, shape=[S, D], strides=[stride_ks, 1], block_shape=[BLOCK_N, D], padding_option="zero")
        v_desc = tl.make_tensor_descriptor(v_base, shape=[S, D], strides=[stride_vs, 1], block_shape=[BLOCK_N, D], padding_option="zero")
        o_desc = tl.make_tensor_descriptor(o_base, shape=[S, D], strides=[stride_os, 1], block_shape=[BLOCK_M, D], padding_option="zero")
        q = q_desc.load([start_m, 0])
    else:
        q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
        if DIVISIBLE_M:
            q = tl.load(q_ptrs)
        else:
            m_mask = offs_m < S
            q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)
            
    # Base-2 Logarithmic Shift Hoisting 
    RCP_LN2: tl.constexpr = 1.4426950408889634
    q = (q.to(tl.float32) * (scale * RCP_LN2)).to(tl.bfloat16)
    
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    
    limit = (S // BLOCK_N) * BLOCK_N
    
    if USE_TMA:
        for start_n in tl.range(0, limit, BLOCK_N):
            k = k_desc.load([start_n, 0])
            v = v_desc.load([start_n, 0])
            
            # Explicitly enforce WGMMA Matrix Multiplications outputs map to reliable FP32 accumulations natively
            scores = tl.dot(q, k.T, out_dtype=tl.float32)
            m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
            
            alpha = tl.math.exp2(m_i - m_ij)
            p = tl.math.exp2(scores - m_ij[:, None])
            
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(tl.bfloat16), v, acc, out_dtype=tl.float32)
            m_i = m_ij
            
        if not DIVISIBLE_N:
            if S % BLOCK_N != 0:
                k = k_desc.load([limit, 0])
                v = v_desc.load([limit, 0])
                scores = tl.dot(q, k.T, out_dtype=tl.float32)
                
                offs_n = limit + tl.arange(0, BLOCK_N)
                valid_score = offs_n[None, :] < S
                scores = tl.where(valid_score, scores, -float("inf"))
                
                m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
                alpha = tl.math.exp2(m_i - m_ij)
                p = tl.math.exp2(scores - m_ij[:, None])
                
                l_i = l_i * alpha + tl.sum(p, axis=1)
                acc = acc * alpha[:, None]
                acc = tl.dot(p.to(tl.bfloat16), v, acc, out_dtype=tl.float32)
                m_i = m_ij
    else:
        offs_n = tl.arange(0, BLOCK_N)
        k_ptrs = k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        
        for start_n in tl.range(0, limit, BLOCK_N): 
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
            
            scores = tl.dot(q, k.T, out_dtype=tl.float32)
            m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
            
            alpha = tl.math.exp2(m_i - m_ij)
            p = tl.math.exp2(scores - m_ij[:, None])
            
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(tl.bfloat16), v, acc, out_dtype=tl.float32)
            m_i = m_ij
            
            k_ptrs += BLOCK_N * stride_ks
            v_ptrs += BLOCK_N * stride_vs
            
        if not DIVISIBLE_N:
            if S % BLOCK_N != 0:
                n_mask = (limit + offs_n) < S
                k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)
                v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)
                
                scores = tl.dot(q, k.T, out_dtype=tl.float32)
                scores = tl.where(n_mask[None, :], scores, -float("inf"))
                
                m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
                alpha = tl.math.exp2(m_i - m_ij)
                p = tl.math.exp2(scores - m_ij[:, None])
                
                l_i = l_i * alpha + tl.sum(p, axis=1)
                acc = acc * alpha[:, None]
                acc = tl.dot(p.to(tl.bfloat16), v, acc, out_dtype=tl.float32)
                m_i = m_ij

    # Pure execution flow. Given that mathematically at least one element covers sequence ranges, l_i > 0 unconditionally  
    inv_l_i = 1.0 / l_i
    output = acc * inv_l_i[:, None]
    
    if USE_TMA:
        o_desc.store([start_m, 0], output.to(tl.bfloat16))
    else:
        o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
        if DIVISIBLE_M:
            tl.store(o_ptrs, output.to(tl.bfloat16))
        else:
            m_mask = offs_m < S
            tl.store(o_ptrs, output.to(tl.bfloat16), mask=m_mask[:, None])
            
    LN2: tl.constexpr = 0.6931471805599453
    lse = (m_i + tl.math.log2(l_i)) * LN2
    
    lse_ptrs = LSE + b * stride_lseb + h * stride_lseh + offs_m * stride_lses
    if DIVISIBLE_M:
        tl.store(lse_ptrs, lse)
    else:
        m_mask = offs_m < S
        tl.store(lse_ptrs, lse, mask=m_mask)


def run(Q, K, V, O, LSE):
    """Destination-passing multi-head attention forward directly updating O and LSE vectors natively."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    scale = D ** -0.5
    
    # 1D Launch ensures deterministic L2 hardware-level mapping natively
    divisible_m = (S % 64 == 0)
    divisible_n = (S % 64 == 0)
    
    grid = lambda META: (B * H * triton.cdiv(S, META["BLOCK_M"]), )
    
    _mha_fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, H, scale,
        D=128,
        DIVISIBLE_M=divisible_m,
        DIVISIBLE_N=divisible_n,
    )