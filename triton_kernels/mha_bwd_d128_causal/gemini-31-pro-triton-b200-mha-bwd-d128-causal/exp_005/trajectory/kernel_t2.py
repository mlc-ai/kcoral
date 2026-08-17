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
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    m_start = start_m * BLOCK_M
    
    # TMA Descriptor Loads for Q, dO, O
    q = desc_q.load([pid_b, pid_h, m_start, 0])
    q = tl.reshape(q, [BLOCK_M, D])
    
    do = desc_do.load([pid_b, pid_h, m_start, 0])
    do = tl.reshape(do, [BLOCK_M, D])
    
    o = desc_o.load([pid_b, pid_h, m_start, 0])
    o = tl.reshape(o, [BLOCK_M, D])
    
    offs_m = m_start + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    
    # LSE loaded via manual pointer (1D, aligned)
    l_ptr = L + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ls
    lse = tl.load(l_ptr, mask=mask_m, other=0.0)
    
    # Precompute row-wise delta
    delta_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    
    dq = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    
    # Causal constraint: max N is bounded by the max M in this block
    max_offs_m = tl.minimum(S, m_start + BLOCK_M)
    end_n = (max_offs_m + BLOCK_N - 1) // BLOCK_N
    
    for start_n in range(0, end_n):
        n_start = start_n * BLOCK_N
        
        # TMA Descriptor Loads for K, V
        k = desc_k.load([pid_b, pid_h, n_start, 0])
        k = tl.reshape(k, [BLOCK_N, D])
        
        v = desc_v.load([pid_b, pid_h, n_start, 0])
        v = tl.reshape(v, [BLOCK_N, D])
        
        scores = tl.dot(q, k.T, out_dtype=tl.float32) * scale
        
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        
        # Fast path: skip causal check for blocks entirely below the diagonal
        if n_start + BLOCK_N > m_start:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            valid = causal_mask & mask_n[None, :] & mask_m[:, None]
        else:
            valid = mask_n[None, :] & mask_m[:, None]
            
        scores = tl.where(valid, scores, float("-inf"))
        p = tl.math.exp(scores - lse[:, None])
        p = tl.where(valid, p, 0.0)
        
        dp = tl.dot(do, v.T, out_dtype=tl.float32)
        ds = p * (dp - delta_val[:, None]) * scale
        ds = tl.where(valid, ds, 0.0)
        
        dq += tl.dot(ds.to(q.dtype), k, out_dtype=tl.float32)
        
    # Re-expand dimensions to match 4D descriptor constraint and store
    dq = tl.reshape(dq, [1, 1, BLOCK_M, D])
    desc_dq.store([pid_b, pid_h, m_start, 0], dq.to(q.dtype))


@triton.jit
def _bwd_kernel_dk_dv(
    desc_q, desc_k, desc_v, desc_o, desc_do, L, desc_dk, desc_dv,
    stride_lb, stride_lh, stride_ls,
    B, H, S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr
):
    start_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    n_start = start_n * BLOCK_N
    
    k = desc_k.load([pid_b, pid_h, n_start, 0])
    k = tl.reshape(k, [BLOCK_N, D])
    
    v = desc_v.load([pid_b, pid_h, n_start, 0])
    v = tl.reshape(v, [BLOCK_N, D])
    
    dk = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    
    offs_n = n_start + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    
    # Causal constraint: start M block from the diagonal
    start_m_initial = n_start // BLOCK_M
    end_m = (S + BLOCK_M - 1) // BLOCK_M
    
    for start_m in range(start_m_initial, end_m):
        m_start = start_m * BLOCK_M
        
        q = desc_q.load([pid_b, pid_h, m_start, 0])
        q = tl.reshape(q, [BLOCK_M, D])
        
        do = desc_do.load([pid_b, pid_h, m_start, 0])
        do = tl.reshape(do, [BLOCK_M, D])
        
        o = desc_o.load([pid_b, pid_h, m_start, 0])
        o = tl.reshape(o, [BLOCK_M, D])
        
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        
        l_ptr = L + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ls
        lse = tl.load(l_ptr, mask=mask_m, other=0.0)
        
        delta_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        
        scores_t = tl.dot(k, q.T, out_dtype=tl.float32) * scale
        
        if m_start < n_start + BLOCK_N:
            causal_mask_t = offs_m[None, :] >= offs_n[:, None]
            valid_t = causal_mask_t & mask_n[:, None] & mask_m[None, :]
        else:
            valid_t = mask_n[:, None] & mask_m[None, :]
            
        scores_t = tl.where(valid_t, scores_t, float("-inf"))
        p_t = tl.math.exp(scores_t - lse[None, :])
        p_t = tl.where(valid_t, p_t, 0.0)
        
        dv += tl.dot(p_t.to(q.dtype), do, out_dtype=tl.float32)
        
        dp_t = tl.dot(v, do.T, out_dtype=tl.float32)
        ds_t = p_t * (dp_t - delta_val[None, :]) * scale
        ds_t = tl.where(valid_t, ds_t, 0.0)
        
        dk += tl.dot(ds_t.to(q.dtype), q, out_dtype=tl.float32)
        
    dk = tl.reshape(dk, [1, 1, BLOCK_N, D])
    dv = tl.reshape(dv, [1, 1, BLOCK_N, D])
    
    desc_dk.store([pid_b, pid_h, n_start, 0], dk.to(q.dtype))
    desc_dv.store([pid_b, pid_h, n_start, 0], dv.to(q.dtype))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    scale = 1.0 / (D ** 0.5)
    
    # -----------------------------
    # Region 1: dQ Tile Computation
    # -----------------------------
    BLOCK_M_DQ = 128
    BLOCK_N_DQ = 64
    
    desc_q_dq = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M_DQ, D])
    desc_k_dq = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N_DQ, D])
    desc_v_dq = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N_DQ, D])
    desc_o_dq = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M_DQ, D])
    desc_do_dq = TensorDescriptor.from_tensor(dO, [1, 1, BLOCK_M_DQ, D])
    desc_dq_dq = TensorDescriptor.from_tensor(dQ, [1, 1, BLOCK_M_DQ, D])
    
    grid_dq = (triton.cdiv(S, BLOCK_M_DQ), B * H)
    _bwd_kernel_dq[grid_dq](
        desc_q_dq, desc_k_dq, desc_v_dq, desc_o_dq, desc_do_dq, L, desc_dq_dq,
        L.stride(0), L.stride(1), L.stride(2),
        B, H, S, scale,
        BLOCK_M=BLOCK_M_DQ, BLOCK_N=BLOCK_N_DQ, D=D,
        num_warps=4, num_stages=3
    )
    
    # ---------------------------------
    # Region 2: dK / dV Tile Computation
    # ---------------------------------
    BLOCK_M_DK = 64
    BLOCK_N_DK = 128
    
    desc_q_dk = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M_DK, D])
    desc_k_dk = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N_DK, D])
    desc_v_dk = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N_DK, D])
    desc_o_dk = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M_DK, D])
    desc_do_dk = TensorDescriptor.from_tensor(dO, [1, 1, BLOCK_M_DK, D])
    desc_dk_dk = TensorDescriptor.from_tensor(dK, [1, 1, BLOCK_N_DK, D])
    desc_dv_dk = TensorDescriptor.from_tensor(dV, [1, 1, BLOCK_N_DK, D])
    
    grid_dkdv = (triton.cdiv(S, BLOCK_N_DK), B * H)
    _bwd_kernel_dk_dv[grid_dkdv](
        desc_q_dk, desc_k_dk, desc_v_dk, desc_o_dk, desc_do_dk, L, desc_dk_dk, desc_dv_dk,
        L.stride(0), L.stride(1), L.stride(2),
        B, H, S, scale,
        BLOCK_M=BLOCK_M_DK, BLOCK_N=BLOCK_N_DK, D=D,
        num_warps=4, num_stages=3
    )