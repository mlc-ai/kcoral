import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


def dq_pre_hook(kwargs):
    BM = kwargs['BLOCK_M']
    BN = kwargs['BLOCK_N']
    d = kwargs['d']
    kwargs['Q_desc'] = TensorDescriptor.from_tensor(kwargs['Q_3d'], [1, BM, d])
    kwargs['K_desc'] = TensorDescriptor.from_tensor(kwargs['K_3d'], [1, BN, d])
    kwargs['V_desc'] = TensorDescriptor.from_tensor(kwargs['V_3d'], [1, BN, d])
    kwargs['O_desc'] = TensorDescriptor.from_tensor(kwargs['O_3d'], [1, BM, d])
    kwargs['dO_desc'] = TensorDescriptor.from_tensor(kwargs['dO_3d'], [1, BM, d])
    kwargs['dQ_desc'] = TensorDescriptor.from_tensor(kwargs['dQ_3d'], [1, BM, d])

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'WARP_SPECIALIZE': True, 'STAGES': 3}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'WARP_SPECIALIZE': False, 'STAGES': 3}, num_warps=4, num_stages=3),
    ],
    key=['S'],
    pre_hook=dq_pre_hook
)
@triton.jit
def bwd_dq_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, dQ_desc,
    Q_3d, K_3d, V_3d, O_3d, dO_3d, dQ_3d,
    L_ptr, stride_lb, stride_lh, stride_ls,
    S, H, inv_sqrt_d,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr, STAGES: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    start_m = pid_m * BLOCK_M
    if start_m >= S:
        return
        
    q = Q_desc.load([pid_bh, start_m, 0])
    do = dO_desc.load([pid_bh, start_m, 0])
    o = O_desc.load([pid_bh, start_m, 0])
    
    q = tl.reshape(q, [BLOCK_M, d])
    do = tl.reshape(do, [BLOCK_M, d])
    o = tl.reshape(o, [BLOCK_M, d])
    
    do_o = do.to(tl.float32) * o.to(tl.float32)
    D_M = tl.sum(do_o, axis=1)
    
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    m_offs = start_m + tl.arange(0, BLOCK_M)
    l_offset = pid_b * stride_lb + pid_h * stride_lh + m_offs * stride_ls
    l = tl.load(L_ptr + l_offset, mask=m_offs < S, other=float('inf'))
    
    dq = tl.zeros([BLOCK_M, d], dtype=tl.float32)
    N_unmasked = (start_m // BLOCK_N) * BLOCK_N
    
    # Fast unmasked looping
    for start_n in tl.range(0, N_unmasked, BLOCK_N, num_stages=STAGES, warp_specialize=WARP_SPECIALIZE):
        k = K_desc.load([pid_bh, start_n, 0])
        v = V_desc.load([pid_bh, start_n, 0])
        
        k = tl.reshape(k, [BLOCK_N, d])
        v = tl.reshape(v, [BLOCK_N, d])
        
        p = tl.dot(q, k.T, out_dtype=tl.float32)
        p = p * inv_sqrt_d
        s = tl.exp(p - l[:, None])
        
        dp_unscaled = tl.dot(do, v.T, out_dtype=tl.float32)
        ds = s * (dp_unscaled - D_M[:, None])
        ds = ds * inv_sqrt_d
        
        dq += tl.dot(ds.to(k.dtype), k, out_dtype=tl.float32)
        
    # Masked trailing edges
    start_n_masked = N_unmasked
    while start_n_masked <= start_m and start_n_masked < S:
        k = K_desc.load([pid_bh, start_n_masked, 0])
        v = V_desc.load([pid_bh, start_n_masked, 0])
        
        k = tl.reshape(k, [BLOCK_N, d])
        v = tl.reshape(v, [BLOCK_N, d])
        
        p = tl.dot(q, k.T, out_dtype=tl.float32)
        p = p * inv_sqrt_d
        
        n_offs = start_n_masked + tl.arange(0, BLOCK_N)
        mask = m_offs[:, None] >= n_offs[None, :]
        p = tl.where(mask, p, -float('inf'))
        
        s = tl.exp(p - l[:, None])
        
        dp_unscaled = tl.dot(do, v.T, out_dtype=tl.float32)
        ds = s * (dp_unscaled - D_M[:, None])
        ds = ds * inv_sqrt_d
        
        dq += tl.dot(ds.to(k.dtype), k, out_dtype=tl.float32)
        start_n_masked += BLOCK_N
        
    dQ_desc.store([pid_bh, start_m, 0], tl.reshape(dq.to(q.dtype), [1, BLOCK_M, d]))


def dk_dv_pre_hook(kwargs):
    BM = kwargs['BLOCK_M']
    BN = kwargs['BLOCK_N']
    d = kwargs['d']
    kwargs['Q_desc'] = TensorDescriptor.from_tensor(kwargs['Q_3d'], [1, BM, d])
    kwargs['K_desc'] = TensorDescriptor.from_tensor(kwargs['K_3d'], [1, BN, d])
    kwargs['V_desc'] = TensorDescriptor.from_tensor(kwargs['V_3d'], [1, BN, d])
    kwargs['O_desc'] = TensorDescriptor.from_tensor(kwargs['O_3d'], [1, BM, d])
    kwargs['dO_desc'] = TensorDescriptor.from_tensor(kwargs['dO_3d'], [1, BM, d])
    kwargs['dK_desc'] = TensorDescriptor.from_tensor(kwargs['dK_3d'], [1, BN, d])
    kwargs['dV_desc'] = TensorDescriptor.from_tensor(kwargs['dV_3d'], [1, BN, d])

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'WARP_SPECIALIZE': True, 'STAGES': 3}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'WARP_SPECIALIZE': False, 'STAGES': 3}, num_warps=4, num_stages=3),
    ],
    key=['S'],
    pre_hook=dk_dv_pre_hook
)
@triton.jit
def bwd_dk_dv_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, dK_desc, dV_desc,
    Q_3d, K_3d, V_3d, O_3d, dO_3d, dK_3d, dV_3d,
    L_ptr, stride_lb, stride_lh, stride_ls,
    S, H, inv_sqrt_d,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr, STAGES: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    start_n = pid_n * BLOCK_N
    if start_n >= S:
        return
        
    k = K_desc.load([pid_bh, start_n, 0])
    v = V_desc.load([pid_bh, start_n, 0])
    k = tl.reshape(k, [BLOCK_N, d])
    v = tl.reshape(v, [BLOCK_N, d])
    
    dk = tl.zeros([BLOCK_N, d], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, d], dtype=tl.float32)
    
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    n_offs = start_n + tl.arange(0, BLOCK_N)
    start_m_masked = (start_n // BLOCK_M) * BLOCK_M
    M_unmasked = ((start_n + BLOCK_N + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    
    while start_m_masked < M_unmasked and start_m_masked < S:
        q = tl.reshape(Q_desc.load([pid_bh, start_m_masked, 0]), [BLOCK_M, d])
        do = tl.reshape(dO_desc.load([pid_bh, start_m_masked, 0]), [BLOCK_M, d])
        o = tl.reshape(O_desc.load([pid_bh, start_m_masked, 0]), [BLOCK_M, d])
        
        do_o = do.to(tl.float32) * o.to(tl.float32)
        D_M_masked = tl.sum(do_o, axis=1)
        
        m_offs = start_m_masked + tl.arange(0, BLOCK_M)
        l_offset = pid_b * stride_lb + pid_h * stride_lh + m_offs * stride_ls
        l = tl.load(L_ptr + l_offset, mask=m_offs < S, other=float('inf'))
        
        p = tl.dot(q, k.T, out_dtype=tl.float32)
        p = p * inv_sqrt_d
        
        mask = m_offs[:, None] >= n_offs[None, :]
        p = tl.where(mask, p, -float('inf'))
        
        s = tl.exp(p - l[:, None])
        
        dv += tl.dot(s.T.to(do.dtype), do, out_dtype=tl.float32)
        
        dp_unscaled = tl.dot(do, v.T, out_dtype=tl.float32)
        ds = (dp_unscaled - D_M_masked[:, None]) * s * inv_sqrt_d
        
        dk += tl.dot(ds.T.to(q.dtype), q, out_dtype=tl.float32)
        start_m_masked += BLOCK_M

    S_aligned = ((S + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    for start_m in tl.range(M_unmasked, S_aligned, BLOCK_M, num_stages=STAGES, warp_specialize=WARP_SPECIALIZE):
        q = tl.reshape(Q_desc.load([pid_bh, start_m, 0]), [BLOCK_M, d])
        do = tl.reshape(dO_desc.load([pid_bh, start_m, 0]), [BLOCK_M, d])
        o = tl.reshape(O_desc.load([pid_bh, start_m, 0]), [BLOCK_M, d])
        
        do_o = do.to(tl.float32) * o.to(tl.float32)
        D_M_unmasked = tl.sum(do_o, axis=1)
        
        m_offs = start_m + tl.arange(0, BLOCK_M)
        l_offset = pid_b * stride_lb + pid_h * stride_lh + m_offs * stride_ls
        l = tl.load(L_ptr + l_offset, mask=m_offs < S, other=float('inf'))
        
        p = tl.dot(q, k.T, out_dtype=tl.float32)
        p = p * inv_sqrt_d
        s = tl.exp(p - l[:, None])
        
        dv += tl.dot(s.T.to(do.dtype), do, out_dtype=tl.float32)
        
        dp_unscaled = tl.dot(do, v.T, out_dtype=tl.float32)
        ds = (dp_unscaled - D_M_unmasked[:, None]) * s * inv_sqrt_d
        
        dk += tl.dot(ds.T.to(q.dtype), q, out_dtype=tl.float32)
        
    dK_desc.store([pid_bh, start_n, 0], tl.reshape(dk.to(k.dtype), [1, BLOCK_N, d]))
    dV_desc.store([pid_bh, start_n, 0], tl.reshape(dv.to(v.dtype), [1, BLOCK_N, d]))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Destination-passing runtime wrapper for attention backward"""
    with torch.cuda.device(Q.device):
        B, H, S, d = Q.shape
        
        Q_3d = Q.view(B * H, S, d)
        K_3d = K.view(B * H, S, d)
        V_3d = V.view(B * H, S, d)
        O_3d = O.view(B * H, S, d)
        dO_3d = dO.view(B * H, S, d)
        dQ_3d = dQ.view(B * H, S, d)
        dK_3d = dK.view(B * H, S, d)
        dV_3d = dV.view(B * H, S, d)
        
        inv_sqrt_d = 1.0 / math.sqrt(d)
        
        # Dispatch 1
        grid_dq = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
        bwd_dq_kernel[grid_dq](
            Q_desc=None, K_desc=None, V_desc=None, O_desc=None, dO_desc=None, dQ_desc=None,
            Q_3d=Q_3d, K_3d=K_3d, V_3d=V_3d, O_3d=O_3d, dO_3d=dO_3d, dQ_3d=dQ_3d,
            L_ptr=L, stride_lb=L.stride(0), stride_lh=L.stride(1), stride_ls=L.stride(2) if L.dim() >= 3 else 1,
            S=S, H=H, inv_sqrt_d=inv_sqrt_d,
            d=d
        )
        
        # Dispatch 2 
        grid_dkv = lambda META: (triton.cdiv(S, META['BLOCK_N']), B * H)
        bwd_dk_dv_kernel[grid_dkv](
            Q_desc=None, K_desc=None, V_desc=None, O_desc=None, dO_desc=None, dK_desc=None, dV_desc=None,
            Q_3d=Q_3d, K_3d=K_3d, V_3d=V_3d, O_3d=O_3d, dO_3d=dO_3d, dK_3d=dK_3d, dV_3d=dV_3d,
            L_ptr=L, stride_lb=L.stride(0), stride_lh=L.stride(1), stride_ls=L.stride(2) if L.dim() >= 3 else 1,
            S=S, H=H, inv_sqrt_d=inv_sqrt_d,
            d=d
        )