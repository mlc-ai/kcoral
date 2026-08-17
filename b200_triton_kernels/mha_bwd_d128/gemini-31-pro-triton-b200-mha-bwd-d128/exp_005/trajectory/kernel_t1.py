import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

def dq_pre_hook(kwargs):
    BM = kwargs['BLOCK_M']
    BN = kwargs['BLOCK_N']
    d = kwargs['d']
    kwargs['q_desc'] = TensorDescriptor.from_tensor(kwargs['Q'], [1, 1, BM, d])
    kwargs['k_desc'] = TensorDescriptor.from_tensor(kwargs['K'], [1, 1, BN, d])
    kwargs['v_desc'] = TensorDescriptor.from_tensor(kwargs['V'], [1, 1, BN, d])
    kwargs['o_desc'] = TensorDescriptor.from_tensor(kwargs['O'], [1, 1, BM, d])
    kwargs['do_desc'] = TensorDescriptor.from_tensor(kwargs['dO'], [1, 1, BM, d])
    kwargs['dq_desc'] = TensorDescriptor.from_tensor(kwargs['dQ'], [1, 1, BM, d])

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'LOOP_STAGES': 3}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'LOOP_STAGES': 2}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'LOOP_STAGES': 3}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'LOOP_STAGES': 4}, num_warps=4, num_stages=4),
    ],
    key=['S'],
    pre_hook=dq_pre_hook,
)
@triton.jit
def bwd_dq_kernel(
    Q, K, V, O, dO, dQ,
    q_desc, k_desc, v_desc, o_desc, do_desc, dq_desc,
    L, stride_lb, stride_lh, stride_ls,
    B, H, S, scale,
    d: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, LOOP_STAGES: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    # Load resident query tiles via TMA (out-of-bounds natively padded with 0)
    q_blk = q_desc.load([b, h, pid_m * BLOCK_M, 0])
    q = tl.reshape(q_blk, (BLOCK_M, d))
    
    o_blk = o_desc.load([b, h, pid_m * BLOCK_M, 0])
    o = tl.reshape(o_blk, (BLOCK_M, d))
    
    do_blk = do_desc.load([b, h, pid_m * BLOCK_M, 0])
    do = tl.reshape(do_blk, (BLOCK_M, d))

    # Load logsumexp
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    l_ptrs = L + b * stride_lb + h * stride_lh + offs_m * stride_ls
    l = tl.load(l_ptrs, mask=offs_m < S, other=0.0)

    # Precompute row-wise delta for this Q block
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)

    dq_acc = tl.zeros((BLOCK_M, d), dtype=tl.float32)

    num_kv_tiles = tl.cdiv(S, BLOCK_N)
    LOG2_E = 1.4426950408889634
    mask_m = offs_m < S

    # Stream KV tiles
    for kv_tile in tl.range(0, num_kv_tiles, num_stages=LOOP_STAGES):
        k_blk = k_desc.load([b, h, kv_tile * BLOCK_N, 0])
        k = tl.reshape(k_blk, (BLOCK_N, d))
        
        v_blk = v_desc.load([b, h, kv_tile * BLOCK_N, 0])
        v = tl.reshape(v_blk, (BLOCK_N, d))
        
        scores = tl.dot(q, k.trans(1, 0)) * scale
        
        offs_n = kv_tile * BLOCK_N + tl.arange(0, BLOCK_N)
        valid = mask_m[:, None] & (offs_n[None, :] < S)
        scores = tl.where(valid, scores, float("-inf"))
        
        p = tl.math.exp2((scores - l[:, None]) * LOG2_E)
        p = tl.where(valid, p, 0.0)
        
        dp = tl.dot(do, v.trans(1, 0))
        
        ds = p * (dp - delta[:, None]) * scale
        ds = tl.where(valid, ds, 0.0)
        
        dq_acc += tl.dot(ds.to(tl.bfloat16), k)

    # Store computed dQ tile back via TMA
    dq_blk = tl.reshape(dq_acc.to(tl.bfloat16), (1, 1, BLOCK_M, d))
    dq_desc.store([b, h, pid_m * BLOCK_M, 0], dq_blk)


def dk_dv_pre_hook(kwargs):
    BM = kwargs['BLOCK_M']
    BN = kwargs['BLOCK_N']
    d = kwargs['d']
    kwargs['q_desc'] = TensorDescriptor.from_tensor(kwargs['Q'], [1, 1, BM, d])
    kwargs['k_desc'] = TensorDescriptor.from_tensor(kwargs['K'], [1, 1, BN, d])
    kwargs['v_desc'] = TensorDescriptor.from_tensor(kwargs['V'], [1, 1, BN, d])
    kwargs['o_desc'] = TensorDescriptor.from_tensor(kwargs['O'], [1, 1, BM, d])
    kwargs['do_desc'] = TensorDescriptor.from_tensor(kwargs['dO'], [1, 1, BM, d])
    kwargs['dk_desc'] = TensorDescriptor.from_tensor(kwargs['dK'], [1, 1, BN, d])
    kwargs['dv_desc'] = TensorDescriptor.from_tensor(kwargs['dV'], [1, 1, BN, d])

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'LOOP_STAGES': 3}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'LOOP_STAGES': 4}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'LOOP_STAGES': 3}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'LOOP_STAGES': 2}, num_warps=4, num_stages=2),
    ],
    key=['S'],
    pre_hook=dk_dv_pre_hook,
)
@triton.jit
def bwd_dk_dv_kernel(
    Q, K, V, O, dO, dK, dV,
    q_desc, k_desc, v_desc, o_desc, do_desc, dk_desc, dv_desc,
    L, stride_lb, stride_lh, stride_ls,
    B, H, S, scale,
    d: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, LOOP_STAGES: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    # Load resident KV tiles via TMA
    k_blk = k_desc.load([b, h, pid_n * BLOCK_N, 0])
    k = tl.reshape(k_blk, (BLOCK_N, d))
    
    v_blk = v_desc.load([b, h, pid_n * BLOCK_N, 0])
    v = tl.reshape(v_blk, (BLOCK_N, d))

    dk_acc = tl.zeros((BLOCK_N, d), dtype=tl.float32)
    dv_acc = tl.zeros((BLOCK_N, d), dtype=tl.float32)

    num_q_tiles = tl.cdiv(S, BLOCK_M)
    LOG2_E = 1.4426950408889634

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S

    # Stream Q tiles
    for q_tile in tl.range(0, num_q_tiles, num_stages=LOOP_STAGES):
        q_blk = q_desc.load([b, h, q_tile * BLOCK_M, 0])
        q = tl.reshape(q_blk, (BLOCK_M, d))
        
        o_blk = o_desc.load([b, h, q_tile * BLOCK_M, 0])
        o = tl.reshape(o_blk, (BLOCK_M, d))
        
        do_blk = do_desc.load([b, h, q_tile * BLOCK_M, 0])
        do = tl.reshape(do_blk, (BLOCK_M, d))

        offs_m = q_tile * BLOCK_M + tl.arange(0, BLOCK_M)
        l_ptrs = L + b * stride_lb + h * stride_lh + offs_m * stride_ls
        l = tl.load(l_ptrs, mask=offs_m < S, other=0.0)

        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)

        scores = tl.dot(k, q.trans(1, 0)) * scale

        valid = mask_n[:, None] & (offs_m[None, :] < S)
        scores = tl.where(valid, scores, float("-inf"))

        p = tl.math.exp2((scores - l[None, :]) * LOG2_E)
        p = tl.where(valid, p, 0.0)

        dv_acc += tl.dot(p.to(tl.bfloat16), do)

        dp_t = tl.dot(v, do.trans(1, 0))

        ds_t = p * (dp_t - delta[None, :]) * scale
        ds_t = tl.where(valid, ds_t, 0.0)

        dk_acc += tl.dot(ds_t.to(tl.bfloat16), q)

    # Store computed dK and dV tiles back via TMA
    dk_blk = tl.reshape(dk_acc.to(tl.bfloat16), (1, 1, BLOCK_N, d))
    dk_desc.store([b, h, pid_n * BLOCK_N, 0], dk_blk)
    
    dv_blk = tl.reshape(dv_acc.to(tl.bfloat16), (1, 1, BLOCK_N, d))
    dv_desc.store([b, h, pid_n * BLOCK_N, 0], dv_blk)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes backward gradients for multi-head attention without causal masking.
    Operates using disjoint split output ownership via TMA descriptors for optimal Blackwell hardware utilization.
    """
    torch.cuda.set_device(Q.device)
    B_val, H_val, S_val, d_val = Q.shape
    
    scale = 1.0 / (d_val ** 0.5)
    stride_ls = L.stride(2) if L.dim() >= 3 else L.stride(-1)

    def grid_dq(META):
        return (triton.cdiv(S_val, META['BLOCK_M']), B_val * H_val)

    # Launch Region 1: Q-tile owners calculate dQ
    bwd_dq_kernel[grid_dq](
        Q=Q, K=K, V=V, O=O, dO=dO, dQ=dQ,
        q_desc=None, k_desc=None, v_desc=None, o_desc=None, do_desc=None, dq_desc=None,
        L=L, stride_lb=L.stride(0), stride_lh=L.stride(1), stride_ls=stride_ls,
        B=B_val, H=H_val, S=S_val, scale=scale,
        d=d_val
    )

    def grid_dk(META):
        return (triton.cdiv(S_val, META['BLOCK_N']), B_val * H_val)

    # Launch Region 2: KV-tile owners calculate dK and dV
    bwd_dk_dv_kernel[grid_dk](
        Q=Q, K=K, V=V, O=O, dO=dO, dK=dK, dV=dV,
        q_desc=None, k_desc=None, v_desc=None, o_desc=None, do_desc=None, dk_desc=None, dv_desc=None,
        L=L, stride_lb=L.stride(0), stride_lh=L.stride(1), stride_ls=stride_ls,
        B=B_val, H=H_val, S=S_val, scale=scale,
        d=d_val
    )