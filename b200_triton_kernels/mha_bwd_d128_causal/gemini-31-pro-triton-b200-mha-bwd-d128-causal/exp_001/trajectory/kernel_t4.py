import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

def bwd_dq_pre_hook(args):
    BLOCK_M = args['BLOCK_M']
    BLOCK_N = args['BLOCK_N']
    d = args['d']
    args['q_desc'] = TensorDescriptor.from_tensor(args['Q'], [1, 1, BLOCK_M, d])
    args['k_desc'] = TensorDescriptor.from_tensor(args['K'], [1, 1, BLOCK_N, d])
    args['v_desc'] = TensorDescriptor.from_tensor(args['V'], [1, 1, BLOCK_N, d])
    args['o_desc'] = TensorDescriptor.from_tensor(args['O'], [1, 1, BLOCK_M, d])
    args['do_desc'] = TensorDescriptor.from_tensor(args['dO'], [1, 1, BLOCK_M, d])
    args['dq_desc'] = TensorDescriptor.from_tensor(args['dQ'], [1, 1, BLOCK_M, d])

def bwd_dk_dv_pre_hook(args):
    BLOCK_M = args['BLOCK_M']
    BLOCK_N = args['BLOCK_N']
    d = args['d']
    args['q_desc'] = TensorDescriptor.from_tensor(args['Q'], [1, 1, BLOCK_M, d])
    args['k_desc'] = TensorDescriptor.from_tensor(args['K'], [1, 1, BLOCK_N, d])
    args['v_desc'] = TensorDescriptor.from_tensor(args['V'], [1, 1, BLOCK_N, d])
    args['o_desc'] = TensorDescriptor.from_tensor(args['O'], [1, 1, BLOCK_M, d])
    args['do_desc'] = TensorDescriptor.from_tensor(args['dO'], [1, 1, BLOCK_M, d])
    args['dk_desc'] = TensorDescriptor.from_tensor(args['dK'], [1, 1, BLOCK_N, d])
    args['dv_desc'] = TensorDescriptor.from_tensor(args['dV'], [1, 1, BLOCK_N, d])

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
    ],
    key=['S'],
    pre_hook=bwd_dq_pre_hook
)
@triton.jit
def bwd_dq_kernel(
    Q, K, V, O, dO, dQ,
    q_desc, k_desc, v_desc, o_desc, do_desc, dq_desc,
    L, stride_lb, stride_lh, stride_ls,
    S,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    start_m = pid_m * BLOCK_M

    # Load block inputs via TMA descriptors
    q = tl.reshape(q_desc.load([pid_b, pid_h, start_m, 0]), (BLOCK_M, d))
    o = tl.reshape(o_desc.load([pid_b, pid_h, start_m, 0]), (BLOCK_M, d))
    do = tl.reshape(do_desc.load([pid_b, pid_h, start_m, 0]), (BLOCK_M, d))

    offs_m = start_m + tl.arange(0, BLOCK_M)
    l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ls
    l = tl.load(l_ptrs, mask=offs_m < S, other=0.0)

    # Precalculate sum(O * dO, dim=-1) for the current Q block
    Di = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)

    dq = tl.zeros((BLOCK_M, d), dtype=tl.float32)
    scale = 0.08838834764831845  # 1.0 / sqrt(128)

    max_n = start_m + BLOCK_M
    if max_n > S:
        max_n = S
    
    n_tiles = tl.cdiv(max_n, BLOCK_N)

    for step in range(n_tiles):
        start_n = step * BLOCK_N
        
        k = tl.reshape(k_desc.load([pid_b, pid_h, start_n, 0]), (BLOCK_N, d))
        v = tl.reshape(v_desc.load([pid_b, pid_h, start_n, 0]), (BLOCK_N, d))

        s = tl.dot(q, k.T, out_dtype=tl.float32) * scale
        
        causal_mask = offs_m[:, None] >= (start_n + tl.arange(0, BLOCK_N))[None, :]
        mask = causal_mask & (offs_m[:, None] < S) & ((start_n + tl.arange(0, BLOCK_N))[None, :] < S)
        s = tl.where(mask, s, float('-inf'))

        p = tl.exp(s - l[:, None])

        ds = tl.dot(do, v.T, out_dtype=tl.float32)
        dp = p * (ds - Di[:, None]) * scale

        dq += tl.dot(dp.to(q.dtype), k, out_dtype=tl.float32)

    # Store computed gradients via TMA descriptor
    dq_desc.store([pid_b, pid_h, start_m, 0], tl.reshape(dq.to(q.dtype), (1, 1, BLOCK_M, d)))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
    ],
    key=['S'],
    pre_hook=bwd_dk_dv_pre_hook
)
@triton.jit
def bwd_dk_dv_kernel(
    Q, K, V, O, dO, dK, dV,
    q_desc, k_desc, v_desc, o_desc, do_desc, dk_desc, dv_desc,
    L, stride_lb, stride_lh, stride_ls,
    S,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    start_n = pid_n * BLOCK_N

    k = tl.reshape(k_desc.load([pid_b, pid_h, start_n, 0]), (BLOCK_N, d))
    v = tl.reshape(v_desc.load([pid_b, pid_h, start_n, 0]), (BLOCK_N, d))

    dk = tl.zeros((BLOCK_N, d), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, d), dtype=tl.float32)

    scale = 0.08838834764831845

    # Causal masking implies M >= N.
    start_m_block = (start_n // BLOCK_M) * BLOCK_M
    n_tiles = tl.cdiv(S - start_m_block, BLOCK_M)
    
    offs_n = start_n + tl.arange(0, BLOCK_N)

    for step in range(n_tiles):
        start_m = start_m_block + step * BLOCK_M
        
        q = tl.reshape(q_desc.load([pid_b, pid_h, start_m, 0]), (BLOCK_M, d))
        o = tl.reshape(o_desc.load([pid_b, pid_h, start_m, 0]), (BLOCK_M, d))
        do = tl.reshape(do_desc.load([pid_b, pid_h, start_m, 0]), (BLOCK_M, d))

        offs_m = start_m + tl.arange(0, BLOCK_M)
        
        l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ls
        l = tl.load(l_ptrs, mask=offs_m < S, other=0.0)
        
        Di = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)

        # Transpose inherently supported via RHS operand orientation.
        s_T = tl.dot(k, q.T, out_dtype=tl.float32) * scale
        
        causal_mask = offs_m[None, :] >= offs_n[:, None]
        mask = causal_mask & (offs_n[:, None] < S) & (offs_m[None, :] < S)
        s_T = tl.where(mask, s_T, float('-inf'))

        p_T = tl.exp(s_T - l[None, :])

        dv += tl.dot(p_T.to(v.dtype), do, out_dtype=tl.float32)

        ds_T = tl.dot(v, do.T, out_dtype=tl.float32)
        dp_T = p_T * (ds_T - Di[None, :]) * scale

        dk += tl.dot(dp_T.to(k.dtype), q, out_dtype=tl.float32)

    # Store via TMA descriptors natively configured.
    dk_desc.store([pid_b, pid_h, start_n, 0], tl.reshape(dk.to(k.dtype), (1, 1, BLOCK_N, d)))
    dv_desc.store([pid_b, pid_h, start_n, 0], tl.reshape(dv.to(v.dtype), (1, 1, BLOCK_N, d)))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes causal multi-head attention backward pass.
    TMA (Tensor Memory Accelerator) efficiently manages memory movement using descriptors.
    """
    with torch.cuda.device(Q.device):
        B, H, S, d = Q.shape
        lse = L.view(B, H, S)
        
        # We pass a valid TensorDescriptor layout to correctly prime the argument types for caching.
        # It is overwritten before compilation per configuration trial inside the pre_hook.
        dummy_desc = TensorDescriptor.from_tensor(Q, [1, 1, 128, d])

        grid_dq = lambda META: (triton.cdiv(S, META['BLOCK_M']), H, B)
        bwd_dq_kernel[grid_dq](
            Q, K, V, O, dO, dQ,
            dummy_desc, dummy_desc, dummy_desc, dummy_desc, dummy_desc, dummy_desc,
            lse, lse.stride(0), lse.stride(1), lse.stride(2),
            S, d=d
        )

        grid_dk_dv = lambda META: (triton.cdiv(S, META['BLOCK_N']), H, B)
        bwd_dk_dv_kernel[grid_dk_dv](
            Q, K, V, O, dO, dK, dV,
            dummy_desc, dummy_desc, dummy_desc, dummy_desc, dummy_desc, dummy_desc, dummy_desc,
            lse, lse.stride(0), lse.stride(1), lse.stride(2),
            S, d=d
        )
        
        return dQ, dK, dV