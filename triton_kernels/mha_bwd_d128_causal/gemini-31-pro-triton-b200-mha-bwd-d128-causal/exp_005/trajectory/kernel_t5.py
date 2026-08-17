import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

@triton.jit
def _bwd_kernel_dq(
    desc_q, desc_k, desc_v, desc_o, desc_do, L, desc_dq,
    stride_lb, stride_lh, stride_ls,
    B, H, S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr
):
    start_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    # Cast to int64 to avoid arbitrary offsets overflowing PyTorch's 1D manual pointers
    off_b = (pid_bh // H).to(tl.int64)
    off_h = (pid_bh % H).to(tl.int64)
    
    m_start = start_m * BLOCK_M
    
    # Fully offloaded asynchronous TMA loads for resident Q ownership block
    q = tl.reshape(desc_q.load([off_b, off_h, m_start, 0]), [BLOCK_M, D])
    do = tl.reshape(desc_do.load([off_b, off_h, m_start, 0]), [BLOCK_M, D])
    o = tl.reshape(desc_o.load([off_b, off_h, m_start, 0]), [BLOCK_M, D])
    
    offs_m = m_start + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    
    # LSE is 1D per head, fallback to pointers as it takes virtually no bandwidth
    l_ptrs = L + off_b * stride_lb + off_h * stride_lh + offs_m * stride_ls
    lse = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    # Precalculate scale delta exactly once. Ensure padded loads are strictly ignored.
    o_fp32 = tl.where(mask_m[:, None], o.to(tl.float32), 0.0)
    do_fp32 = tl.where(mask_m[:, None], do.to(tl.float32), 0.0)
    delta = tl.sum(o_fp32 * do_fp32, axis=1)
    
    dq = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    
    # Compute max causality limits bounds to securely prune trailing K blocks
    max_offs_m = tl.minimum(S, m_start + BLOCK_M)
    end_n = (max_offs_m + BLOCK_N - 1) // BLOCK_N
    
    RCP_LN2 = 1.4426950408889634
    
    for start_n in range(0, end_n):
        n_start = start_n * BLOCK_N
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        
        # TMA block streams 
        k = tl.reshape(desc_k.load([off_b, off_h, n_start, 0]), [BLOCK_N, D])
        v = tl.reshape(desc_v.load([off_b, off_h, n_start, 0]), [BLOCK_N, D])
        
        # MMA ops explicitly directed to FP32 accumulators
        scores = tl.dot(q, k.T, out_dtype=tl.float32) * scale
        
        causal = offs_m[:, None] >= offs_n[None, :]
        valid = causal & mask_m[:, None] & mask_n[None, :]
        
        scores = tl.where(valid, scores, float("-inf"))
        p = tl.math.exp2((scores - lse[:, None]) * RCP_LN2)
        p = tl.where(valid, p, 0.0)
        
        dp = tl.dot(do, v.T, out_dtype=tl.float32)
        ds = p * (dp - delta[:, None]) * scale
        ds = tl.where(valid, ds, 0.0)
        
        dq += tl.dot(ds.to(tl.bfloat16), k, out_dtype=tl.float32)
        
    # Re-expand dimensions to strictly honor 4D descriptor contract before store
    dq = tl.reshape(dq, [1, 1, BLOCK_M, D])
    desc_dq.store([off_b, off_h, m_start, 0], dq.to(tl.bfloat16))


@triton.jit
def _bwd_kernel_dk_dv(
    desc_q, desc_k, desc_v, desc_o, desc_do, L, desc_dk, desc_dv,
    stride_lb, stride_lh, stride_ls,
    B, H, S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr
):
    start_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    off_b = (pid_bh // H).to(tl.int64)
    off_h = (pid_bh % H).to(tl.int64)
    
    n_start = start_n * BLOCK_N
    offs_n = n_start + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    
    k = tl.reshape(desc_k.load([off_b, off_h, n_start, 0]), [BLOCK_N, D])
    v = tl.reshape(desc_v.load([off_b, off_h, n_start, 0]), [BLOCK_N, D])
    
    dk = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    
    start_m_initial = n_start // BLOCK_M
    end_m = (S + BLOCK_M - 1) // BLOCK_M
    
    RCP_LN2 = 1.4426950408889634
    
    for start_m in range(start_m_initial, end_m):
        m_start = start_m * BLOCK_M
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        
        q = tl.reshape(desc_q.load([off_b, off_h, m_start, 0]), [BLOCK_M, D])
        do = tl.reshape(desc_do.load([off_b, off_h, m_start, 0]), [BLOCK_M, D])
        o = tl.reshape(desc_o.load([off_b, off_h, m_start, 0]), [BLOCK_M, D])
        
        l_ptrs = L + off_b * stride_lb + off_h * stride_lh + offs_m * stride_ls
        lse = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        o_fp32 = tl.where(mask_m[:, None], o.to(tl.float32), 0.0)
        do_fp32 = tl.where(mask_m[:, None], do.to(tl.float32), 0.0)
        delta = tl.sum(o_fp32 * do_fp32, axis=1)
        
        scores_t = tl.dot(k, q.T, out_dtype=tl.float32) * scale
        
        causal = offs_m[None, :] >= offs_n[:, None]
        valid_t = causal & mask_n[:, None] & mask_m[None, :]
        
        scores_t = tl.where(valid_t, scores_t, float("-inf"))
        p_t = tl.math.exp2((scores_t - lse[None, :]) * RCP_LN2)
        p_t = tl.where(valid_t, p_t, 0.0)
        
        dv += tl.dot(p_t.to(tl.bfloat16), do, out_dtype=tl.float32)
        
        dp_t = tl.dot(v, do.T, out_dtype=tl.float32)
        ds_t = p_t * (dp_t - delta[None, :]) * scale
        ds_t = tl.where(valid_t, ds_t, 0.0)
        
        dk += tl.dot(ds_t.to(tl.bfloat16), q, out_dtype=tl.float32)
        
    dk = tl.reshape(dk, [1, 1, BLOCK_N, D])
    dv = tl.reshape(dv, [1, 1, BLOCK_N, D])
    
    desc_dk.store([off_b, off_h, n_start, 0], dk.to(tl.bfloat16))
    desc_dv.store([off_b, off_h, n_start, 0], dv.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes causal Multi-Head Attention backward with Blackwell decoupled ownership strategy.
    Implements standard TensorDescriptors mapped to SM100 limits for optimal TMEM loading.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    scale = 1.0 / (D ** 0.5)
    
    # Keeps exact symmetric limits keeping max peak registers inside the 255 hardcap
    BLOCK_M = 64
    BLOCK_N = 64
    
    desc_q = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M, D])
    desc_k = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N, D])
    desc_v = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N, D])
    desc_o = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M, D])
    desc_do = TensorDescriptor.from_tensor(dO, [1, 1, BLOCK_M, D])
    desc_dq = TensorDescriptor.from_tensor(dQ, [1, 1, BLOCK_M, D])
    desc_dk = TensorDescriptor.from_tensor(dK, [1, 1, BLOCK_N, D])
    desc_dv = TensorDescriptor.from_tensor(dV, [1, 1, BLOCK_N, D])
    
    grid_dq = (triton.cdiv(S, BLOCK_M), B * H)
    _bwd_kernel_dq[grid_dq](
        desc_q, desc_k, desc_v, desc_o, desc_do, L, desc_dq,
        L.stride(0), L.stride(1), L.stride(2),
        B, H, S, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D,
        num_warps=4, num_stages=3
    )
    
    grid_dkdv = (triton.cdiv(S, BLOCK_N), B * H)
    _bwd_kernel_dk_dv[grid_dkdv](
        desc_q, desc_k, desc_v, desc_o, desc_do, L, desc_dk, desc_dv,
        L.stride(0), L.stride(1), L.stride(2),
        B, H, S, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D,
        num_warps=4, num_stages=3
    )