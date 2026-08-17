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
    
    # Strictly enforce 32-bit bounds for TMA offset coordinates
    coord_b = (pid_bh // H).to(tl.int32)
    coord_h = (pid_bh % H).to(tl.int32)
    
    m_start = start_m * BLOCK_M
    coord_m = m_start.to(tl.int32)
    
    # Asynchronous TMEM offloaded loads
    q = tl.reshape(desc_q.load([coord_b, coord_h, coord_m, 0]), [BLOCK_M, D])
    do = tl.reshape(desc_do.load([coord_b, coord_h, coord_m, 0]), [BLOCK_M, D])
    o = tl.reshape(desc_o.load([coord_b, coord_h, coord_m, 0]), [BLOCK_M, D])
    
    offs_m = m_start + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    
    # 1D LSE scale handled cleanly with basic pointers 
    l_ptrs = L + coord_b.to(tl.int64) * stride_lb + coord_h.to(tl.int64) * stride_lh + offs_m * stride_ls
    lse = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    # Precalculate scale delta exactly once for the resident Q block.
    # Handle mask explicitly to ensure descriptor boundary pads are ignored.
    o_fp32 = tl.where(mask_m[:, None], o.to(tl.float32), 0.0)
    do_fp32 = tl.where(mask_m[:, None], do.to(tl.float32), 0.0)
    delta = tl.sum(o_fp32 * do_fp32, axis=1)
    
    dq = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    
    # Compute limit bounds to securely prune trailing non-causal K blocks early
    max_offs_m = tl.minimum(S, m_start + BLOCK_M)
    end_n = (max_offs_m + BLOCK_N - 1) // BLOCK_N
    
    RCP_LN2 = 1.4426950408889634
    
    for start_n in range(0, end_n):
        n_start = start_n * BLOCK_N
        coord_n = n_start.to(tl.int32)
        
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        
        k = tl.reshape(desc_k.load([coord_b, coord_h, coord_n, 0]), [BLOCK_N, D])
        v = tl.reshape(desc_v.load([coord_b, coord_h, coord_n, 0]), [BLOCK_N, D])
        
        # Native Tensor Core math mapping optimally using FP32 accumulators
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
        
    # Re-expand dimensions to strictly honor 4D TMA descriptor contract before store
    dq = tl.reshape(dq, [1, 1, BLOCK_M, D])
    desc_dq.store([coord_b, coord_h, coord_m, 0], dq.to(tl.bfloat16))


@triton.jit
def _bwd_kernel_dk_dv(
    desc_q, desc_k, desc_v, desc_o, desc_do, L, desc_dk, desc_dv,
    stride_lb, stride_lh, stride_ls,
    B, H, S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr
):
    start_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    coord_b = (pid_bh // H).to(tl.int32)
    coord_h = (pid_bh % H).to(tl.int32)
    
    n_start = start_n * BLOCK_N
    coord_n = n_start.to(tl.int32)
    
    offs_n = n_start + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    
    # KV ownership hold vectors resident in registers
    k = tl.reshape(desc_k.load([coord_b, coord_h, coord_n, 0]), [BLOCK_N, D])
    v = tl.reshape(desc_v.load([coord_b, coord_h, coord_n, 0]), [BLOCK_N, D])
    
    dk = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    
    # Sequence limit iteration beginning at the causally valid diagonal origin 
    start_m_initial = n_start // BLOCK_M
    end_m = (S + BLOCK_M - 1) // BLOCK_M
    
    RCP_LN2 = 1.4426950408889634
    
    for start_m in range(start_m_initial, end_m):
        m_start = start_m * BLOCK_M
        coord_m = m_start.to(tl.int32)
        
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        
        q = tl.reshape(desc_q.load([coord_b, coord_h, coord_m, 0]), [BLOCK_M, D])
        do = tl.reshape(desc_do.load([coord_b, coord_h, coord_m, 0]), [BLOCK_M, D])
        o = tl.reshape(desc_o.load([coord_b, coord_h, coord_m, 0]), [BLOCK_M, D])
        
        l_ptrs = L + coord_b.to(tl.int64) * stride_lb + coord_h.to(tl.int64) * stride_lh + offs_m * stride_ls
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
    
    desc_dk.store([coord_b, coord_h, coord_n, 0], dk.to(tl.bfloat16))
    desc_dv.store([coord_b, coord_h, coord_n, 0], dv.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes causal Multi-Head Attention backward with Blackwell decoupled ownership strategy.
    Leverages 64x64 symmetric pipelining bounding peak registers to <255 limit perfectly, removing local mem spills.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    scale = 1.0 / (D ** 0.5)
    
    BLOCK_M = 64
    BLOCK_N = 64
    
    # Convert directly utilizing 16-byte contiguous rank requirements to form explicit host descriptors
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
        num_warps=4, num_stages=4
    )
    
    grid_dkdv = (triton.cdiv(S, BLOCK_N), B * H)
    _bwd_kernel_dk_dv[grid_dkdv](
        desc_q, desc_k, desc_v, desc_o, desc_do, L, desc_dk, desc_dv,
        L.stride(0), L.stride(1), L.stride(2),
        B, H, S, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D,
        num_warps=4, num_stages=4
    )