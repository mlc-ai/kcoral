import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def bwd_dq_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, dQ_desc,
    L_ptr, stride_lb, stride_lh, stride_ls,
    S, H,
    inv_sqrt_d,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    start_m = pid_m * BLOCK_M
    # Safe boundary early exit
    if start_m >= S:
        return
        
    # Query dependencies (Loop invariant)
    q = Q_desc.load([pid_bh, start_m, 0])
    q = tl.reshape(q, [BLOCK_M, d])
    
    do = dO_desc.load([pid_bh, start_m, 0])
    do = tl.reshape(do, [BLOCK_M, d])
    
    o = O_desc.load([pid_bh, start_m, 0])
    o = tl.reshape(o, [BLOCK_M, d])
    
    # Precompute fast row sums inline
    do_o = do.to(tl.float32) * o.to(tl.float32)
    D_M = tl.sum(do_o, axis=1)
    
    # Access strides securely for 1D/2D metadata logic
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    l_offset = pid_b * stride_lb + pid_h * stride_lh + (start_m + tl.arange(0, BLOCK_M)) * stride_ls
    l = tl.load(L_ptr + l_offset, mask=(start_m + tl.arange(0, BLOCK_M)) < S, other=0.0)
    
    dq = tl.zeros([BLOCK_M, d], dtype=tl.float32)
    m_offs = start_m + tl.arange(0, BLOCK_M)
    
    # Calculate bounded alignment strictly for safe looping and overlapping causality checks 
    N_max = start_m + BLOCK_M
    if N_max > S:
        N_max = S
    N_max_aligned = ((N_max + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    
    # Loop over causally related key blocks
    for start_n in tl.range(0, N_max_aligned, BLOCK_N, num_stages=3):
        k = K_desc.load([pid_bh, start_n, 0])
        k = tl.reshape(k, [BLOCK_N, d])
        
        v = V_desc.load([pid_bh, start_n, 0])
        v = tl.reshape(v, [BLOCK_N, d])
        
        p = tl.dot(q, k.T, out_dtype=tl.float32)
        p = p * inv_sqrt_d
        
        # Apply standard padding + causal masking combinations over scaled product space
        n_offs = start_n + tl.arange(0, BLOCK_N)
        mask = (m_offs[:, None] >= n_offs[None, :]) & (m_offs[:, None] < S) & (n_offs[None, :] < S)
        p = tl.where(mask, p, -float('inf'))
        
        # S_{ij} derivatives
        s = tl.exp(p - l[:, None])
        s = tl.where(mask, s, 0.0)
        
        dp_unscaled = tl.dot(do, v.T, out_dtype=tl.float32)
        
        ds = s * (dp_unscaled - D_M[:, None])
        ds = ds * inv_sqrt_d
        
        dq += tl.dot(ds.to(k.dtype), k, out_dtype=tl.float32)
        
    # Store bounded memory
    dQ_desc.store([pid_bh, start_m, 0], tl.reshape(dq.to(q.dtype), [1, BLOCK_M, d]))


@triton.jit
def bwd_dk_dv_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, dK_desc, dV_desc,
    L_ptr, stride_lb, stride_lh, stride_ls,
    S, H,
    inv_sqrt_d,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    start_n = pid_n * BLOCK_N
    # Safe boundary early exit
    if start_n >= S:
        return
        
    # Key dependencies (Loop invariant)
    k = K_desc.load([pid_bh, start_n, 0])
    k = tl.reshape(k, [BLOCK_N, d])
    
    v = V_desc.load([pid_bh, start_n, 0])
    v = tl.reshape(v, [BLOCK_N, d])
    
    dk = tl.zeros([BLOCK_N, d], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, d], dtype=tl.float32)
    
    n_offs = start_n + tl.arange(0, BLOCK_N)
    
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    start_m_aligned = (start_n // BLOCK_M) * BLOCK_M
    M_max_aligned = ((S + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    
    # Loop over casually related query blocks
    for start_m in tl.range(start_m_aligned, M_max_aligned, BLOCK_M, num_stages=3):
        q = Q_desc.load([pid_bh, start_m, 0])
        q = tl.reshape(q, [BLOCK_M, d])
        
        do = dO_desc.load([pid_bh, start_m, 0])
        do = tl.reshape(do, [BLOCK_M, d])
        
        o = O_desc.load([pid_bh, start_m, 0])
        o = tl.reshape(o, [BLOCK_M, d])
        
        do_o = do.to(tl.float32) * o.to(tl.float32)
        D_M = tl.sum(do_o, axis=1)
        
        l_offset = pid_b * stride_lb + pid_h * stride_lh + (start_m + tl.arange(0, BLOCK_M)) * stride_ls
        l = tl.load(L_ptr + l_offset, mask=(start_m + tl.arange(0, BLOCK_M)) < S, other=0.0)
        
        p = tl.dot(q, k.T, out_dtype=tl.float32)
        p = p * inv_sqrt_d
        
        m_offs = start_m + tl.arange(0, BLOCK_M)
        mask = (m_offs[:, None] >= n_offs[None, :]) & (m_offs[:, None] < S) & (n_offs[None, :] < S)
        p = tl.where(mask, p, -float('inf'))
        
        # dV logic accumulation
        s = tl.exp(p - l[:, None])
        s = tl.where(mask, s, 0.0)
        
        dv += tl.dot(s.T.to(do.dtype), do, out_dtype=tl.float32)
        
        # dK logic accumulation
        dp_unscaled = tl.dot(do, v.T, out_dtype=tl.float32)
        
        ds = s * (dp_unscaled - D_M[:, None])
        ds = ds * inv_sqrt_d
        
        dk += tl.dot(ds.T.to(q.dtype), q, out_dtype=tl.float32)
        
    dK_desc.store([pid_bh, start_n, 0], tl.reshape(dk.to(k.dtype), [1, BLOCK_N, d]))
    dV_desc.store([pid_bh, start_n, 0], tl.reshape(dv.to(v.dtype), [1, BLOCK_N, d]))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Perform causal multi-head attention backward in destination-passing style.
    Calculates inline gradient operations and writes cleanly into dQ, dK, dV.
    """
    with torch.cuda.device(Q.device):
        B, H, S, d = Q.shape
        
        BLOCK_M = 64
        BLOCK_N = 64
        
        # Preconfigure 3D Host Blackwell Descriptors resolving boundary logic padding accurately 
        Q_3d = Q.view(B * H, S, d)
        K_3d = K.view(B * H, S, d)
        V_3d = V.view(B * H, S, d)
        O_3d = O.view(B * H, S, d)
        dO_3d = dO.view(B * H, S, d)
        dQ_3d = dQ.view(B * H, S, d)
        dK_3d = dK.view(B * H, S, d)
        dV_3d = dV.view(B * H, S, d)
        
        Q_desc = TensorDescriptor.from_tensor(Q_3d, [1, BLOCK_M, d])
        K_desc = TensorDescriptor.from_tensor(K_3d, [1, BLOCK_N, d])
        V_desc = TensorDescriptor.from_tensor(V_3d, [1, BLOCK_N, d])
        O_desc = TensorDescriptor.from_tensor(O_3d, [1, BLOCK_M, d])
        dO_desc = TensorDescriptor.from_tensor(dO_3d, [1, BLOCK_M, d])
        dQ_desc = TensorDescriptor.from_tensor(dQ_3d, [1, BLOCK_M, d])
        dK_desc = TensorDescriptor.from_tensor(dK_3d, [1, BLOCK_N, d])
        dV_desc = TensorDescriptor.from_tensor(dV_3d, [1, BLOCK_N, d])
        
        inv_sqrt_d = 1.0 / math.sqrt(d)
        
        # Execution #1 - Accumulate Queries over Keys
        grid_dq = (triton.cdiv(S, BLOCK_M), B * H)
        bwd_dq_kernel[grid_dq](
            Q_desc, K_desc, V_desc, O_desc, dO_desc, dQ_desc,
            L, L.stride(0), L.stride(1), L.stride(2),
            S, H,
            inv_sqrt_d=inv_sqrt_d,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, d=d,
            num_warps=4, num_stages=3
        )
        
        # Execution #2 - Accumulate Keys/Values over Queries
        grid_dkv = (triton.cdiv(S, BLOCK_N), B * H)
        bwd_dk_dv_kernel[grid_dkv](
            Q_desc, K_desc, V_desc, O_desc, dO_desc, dK_desc, dV_desc,
            L, L.stride(0), L.stride(1), L.stride(2),
            S, H,
            inv_sqrt_d=inv_sqrt_d,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, d=d,
            num_warps=4, num_stages=3
        )