import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


def dq_pre_hook(kwargs):
    BM = kwargs['BLOCK_M']
    BN = kwargs['BLOCK_N']
    d = kwargs['d']
    kwargs['Q_desc'] = TensorDescriptor.from_tensor(kwargs['Q'], [1, 1, BM, d])
    kwargs['K_desc'] = TensorDescriptor.from_tensor(kwargs['K'], [1, 1, BN, d])
    kwargs['V_desc'] = TensorDescriptor.from_tensor(kwargs['V'], [1, 1, BN, d])
    kwargs['O_desc'] = TensorDescriptor.from_tensor(kwargs['O'], [1, 1, BM, d])
    kwargs['dO_desc'] = TensorDescriptor.from_tensor(kwargs['dO'], [1, 1, BM, d])
    kwargs['dQ_desc'] = TensorDescriptor.from_tensor(kwargs['dQ'], [1, 1, BM, d])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'STAGES': 3}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'STAGES': 4}, num_warps=4, num_stages=4),
    ],
    key=['S'],
    pre_hook=dq_pre_hook
)
@triton.jit
def bwd_dq_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, dQ_desc,
    Q, K, V, O, dO, dQ,
    L_ptr, stride_lb, stride_lh, stride_ls,
    S, H, inv_sqrt_d,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr,
    STAGES: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    start_m = pid_m * BLOCK_M
    if start_m >= S:
        return
        
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    # Fully loop-invariant Query, Output, and upstream gradients
    q = tl.reshape(Q_desc.load([pid_b, pid_h, start_m, 0]), [BLOCK_M, d])
    do = tl.reshape(dO_desc.load([pid_b, pid_h, start_m, 0]), [BLOCK_M, d])
    o = tl.reshape(O_desc.load([pid_b, pid_h, start_m, 0]), [BLOCK_M, d])
    
    do_o = do.to(tl.float32) * o.to(tl.float32)
    D_M = tl.sum(do_o, axis=1)
    
    m_offs = start_m + tl.arange(0, BLOCK_M)
    l_offset = pid_b * stride_lb + pid_h * stride_lh + m_offs * stride_ls
    l = tl.load(L_ptr + l_offset, mask=m_offs < S, other=float('inf'))
    
    dq = tl.zeros([BLOCK_M, d], dtype=tl.float32)
    N_unmasked = (start_m // BLOCK_N) * BLOCK_N
    
    # Primary logic: Heavy software-pipelined, full blocks with no causality conditionals needed
    if N_unmasked > 0:
        for start_n in tl.range(0, N_unmasked, BLOCK_N, num_stages=STAGES):
            k = tl.reshape(K_desc.load([pid_b, pid_h, start_n, 0]), [BLOCK_N, d])
            v = tl.reshape(V_desc.load([pid_b, pid_h, start_n, 0]), [BLOCK_N, d])
            
            p = tl.dot(q, k.T, out_dtype=tl.float32) * inv_sqrt_d
            s = tl.exp(p - l[:, None])
            
            dp_unscaled = tl.dot(do, v.T, out_dtype=tl.float32)
            ds = (dp_unscaled - D_M[:, None]) * s * inv_sqrt_d
            
            dq += tl.dot(ds.to(q.dtype), k, out_dtype=tl.float32)
            
    # Trailing logic: Minimal causal boundary overlap mapping loop
    start_n_masked = N_unmasked
    while start_n_masked <= start_m and start_n_masked < S:
        k = tl.reshape(K_desc.load([pid_b, pid_h, start_n_masked, 0]), [BLOCK_N, d])
        v = tl.reshape(V_desc.load([pid_b, pid_h, start_n_masked, 0]), [BLOCK_N, d])
        
        p = tl.dot(q, k.T, out_dtype=tl.float32) * inv_sqrt_d
        
        n_offs = start_n_masked + tl.arange(0, BLOCK_N)
        mask = m_offs[:, None] >= n_offs[None, :]
        p = tl.where(mask, p, -float('inf'))
        
        s = tl.exp(p - l[:, None])
        
        dp_unscaled = tl.dot(do, v.T, out_dtype=tl.float32)
        ds = (dp_unscaled - D_M[:, None]) * s * inv_sqrt_d
        
        dq += tl.dot(ds.to(q.dtype), k, out_dtype=tl.float32)
        start_n_masked += BLOCK_N
        
    dQ_desc.store([pid_b, pid_h, start_m, 0], tl.reshape(dq.to(q.dtype), [1, 1, BLOCK_M, d]))


def dk_dv_pre_hook(kwargs):
    BM = kwargs['BLOCK_M']
    BN = kwargs['BLOCK_N']
    d = kwargs['d']
    kwargs['Q_desc'] = TensorDescriptor.from_tensor(kwargs['Q'], [1, 1, BM, d])
    kwargs['K_desc'] = TensorDescriptor.from_tensor(kwargs['K'], [1, 1, BN, d])
    kwargs['V_desc'] = TensorDescriptor.from_tensor(kwargs['V'], [1, 1, BN, d])
    kwargs['O_desc'] = TensorDescriptor.from_tensor(kwargs['O'], [1, 1, BM, d])
    kwargs['dO_desc'] = TensorDescriptor.from_tensor(kwargs['dO'], [1, 1, BM, d])
    kwargs['dK_desc'] = TensorDescriptor.from_tensor(kwargs['dK'], [1, 1, BN, d])
    kwargs['dV_desc'] = TensorDescriptor.from_tensor(kwargs['dV'], [1, 1, BN, d])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'STAGES': 3}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'STAGES': 4}, num_warps=4, num_stages=4),
    ],
    key=['S'],
    pre_hook=dk_dv_pre_hook
)
@triton.jit
def bwd_dk_dv_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, dK_desc, dV_desc,
    Q, K, V, O, dO, dK, dV,
    L_ptr, stride_lb, stride_lh, stride_ls,
    S, H, inv_sqrt_d,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr,
    STAGES: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    start_n = pid_n * BLOCK_N
    if start_n >= S:
        return
        
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    k = tl.reshape(K_desc.load([pid_b, pid_h, start_n, 0]), [BLOCK_N, d])
    v = tl.reshape(V_desc.load([pid_b, pid_h, start_n, 0]), [BLOCK_N, d])
    
    dk = tl.zeros([BLOCK_N, d], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, d], dtype=tl.float32)
    
    n_offs = start_n + tl.arange(0, BLOCK_N)
    start_m_aligned = (start_n // BLOCK_M) * BLOCK_M
    S_aligned = ((S + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    
    M_unmasked = ((start_n + BLOCK_N + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    if M_unmasked > S_aligned:
        M_unmasked = S_aligned
        
    # Trailing logic: Pre-processed overlapping causal bounders strictly managed 
    start_m_masked = start_m_aligned
    while start_m_masked < M_unmasked and start_m_masked < S:
        q = tl.reshape(Q_desc.load([pid_b, pid_h, start_m_masked, 0]), [BLOCK_M, d])
        do = tl.reshape(dO_desc.load([pid_b, pid_h, start_m_masked, 0]), [BLOCK_M, d])
        o = tl.reshape(O_desc.load([pid_b, pid_h, start_m_masked, 0]), [BLOCK_M, d])
        
        do_o = do.to(tl.float32) * o.to(tl.float32)
        D_M_masked = tl.sum(do_o, axis=1)
        
        m_offs = start_m_masked + tl.arange(0, BLOCK_M)
        l_offset = pid_b * stride_lb + pid_h * stride_lh + m_offs * stride_ls
        l = tl.load(L_ptr + l_offset, mask=m_offs < S, other=float('inf'))
        
        p = tl.dot(q, k.T, out_dtype=tl.float32) * inv_sqrt_d
        
        mask = m_offs[:, None] >= n_offs[None, :]
        p = tl.where(mask, p, -float('inf'))
        
        s = tl.exp(p - l[:, None])
        
        dv += tl.dot(s.T.to(do.dtype), do, out_dtype=tl.float32)
        
        dp_unscaled = tl.dot(do, v.T, out_dtype=tl.float32)
        ds = (dp_unscaled - D_M_masked[:, None]) * s * inv_sqrt_d
        
        dk += tl.dot(ds.T.to(q.dtype), q, out_dtype=tl.float32)
        start_m_masked += BLOCK_M

    # Primary logic: High volume software-pipelined accumulation 
    if M_unmasked < S_aligned:
        for start_m in tl.range(M_unmasked, S_aligned, BLOCK_M, num_stages=STAGES):
            q = tl.reshape(Q_desc.load([pid_b, pid_h, start_m, 0]), [BLOCK_M, d])
            do = tl.reshape(dO_desc.load([pid_b, pid_h, start_m, 0]), [BLOCK_M, d])
            o = tl.reshape(O_desc.load([pid_b, pid_h, start_m, 0]), [BLOCK_M, d])
            
            do_o = do.to(tl.float32) * o.to(tl.float32)
            D_M_unmasked = tl.sum(do_o, axis=1)
            
            m_offs = start_m + tl.arange(0, BLOCK_M)
            l_offset = pid_b * stride_lb + pid_h * stride_lh + m_offs * stride_ls
            l = tl.load(L_ptr + l_offset, mask=m_offs < S, other=float('inf'))
            
            p = tl.dot(q, k.T, out_dtype=tl.float32) * inv_sqrt_d
            p = tl.where(n_offs[None, :] < S, p, -float('inf'))  # Safe-guard sequence overlaps limits
            s = tl.exp(p - l[:, None])
            
            dv += tl.dot(s.T.to(do.dtype), do, out_dtype=tl.float32)
            
            dp_unscaled = tl.dot(do, v.T, out_dtype=tl.float32)
            ds = (dp_unscaled - D_M_unmasked[:, None]) * s * inv_sqrt_d
            
            dk += tl.dot(ds.T.to(q.dtype), q, out_dtype=tl.float32)
            
    dK_desc.store([pid_b, pid_h, start_n, 0], tl.reshape(dk.to(k.dtype), [1, 1, BLOCK_N, d]))
    dV_desc.store([pid_b, pid_h, start_n, 0], tl.reshape(dv.to(v.dtype), [1, 1, BLOCK_N, d]))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Perform causal multi-head attention backward accurately in destination-passing style.
    Calculates inline gradient operations and writes cleanly into dQ, dK, dV.
    """
    with torch.cuda.device(Q.device):
        B, H, S, d = Q.shape
        inv_sqrt_d = 1.0 / math.sqrt(d)
        stride_ls = L.stride(2) if L.dim() >= 3 else 1
        
        # Dispatch 1: Compute Query Gradients independently
        grid_dq = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
        bwd_dq_kernel[grid_dq](
            Q_desc=None, K_desc=None, V_desc=None, O_desc=None, dO_desc=None, dQ_desc=None,
            Q=Q, K=K, V=V, O=O, dO=dO, dQ=dQ,
            L_ptr=L, stride_lb=L.stride(0), stride_lh=L.stride(1), stride_ls=stride_ls,
            S=S, H=H, inv_sqrt_d=inv_sqrt_d,
            d=d
        )
        
        # Dispatch 2: Compute Key and Value Gradients independently
        grid_dkv = lambda META: (triton.cdiv(S, META['BLOCK_N']), B * H)
        bwd_dk_dv_kernel[grid_dkv](
            Q_desc=None, K_desc=None, V_desc=None, O_desc=None, dO_desc=None, dK_desc=None, dV_desc=None,
            Q=Q, K=K, V=V, O=O, dO=dO, dK=dK, dV=dV,
            L_ptr=L, stride_lb=L.stride(0), stride_lh=L.stride(1), stride_ls=stride_ls,
            S=S, H=H, inv_sqrt_d=inv_sqrt_d,
            d=d
        )