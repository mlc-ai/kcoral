import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

# Setup Triton allocator for descriptor infrastructure storage natively on host/device interactions
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def dq_pre_hook(kwargs):
    Q = kwargs['Q']
    K = kwargs['K']
    V = kwargs['V']
    O = kwargs['O']
    dO = kwargs['dO']
    dQ = kwargs['dQ']
    BLOCK_M = kwargs['BLOCK_M']
    BLOCK_N = kwargs['BLOCK_N']
    BLOCK_D = kwargs['BLOCK_D']
    
    # Overwrite the dummy descriptors with properly sized host TMA descriptors mapped physically
    kwargs['Q_desc'] = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M, BLOCK_D])
    kwargs['K_desc'] = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N, BLOCK_D])
    kwargs['V_desc'] = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N, BLOCK_D])
    kwargs['O_desc'] = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M, BLOCK_D])
    kwargs['dO_desc'] = TensorDescriptor.from_tensor(dO, [1, 1, BLOCK_M, BLOCK_D])
    kwargs['dQ_desc'] = TensorDescriptor.from_tensor(dQ, [1, 1, BLOCK_M, BLOCK_D])

def dkdv_pre_hook(kwargs):
    Q = kwargs['Q']
    K = kwargs['K']
    V = kwargs['V']
    O = kwargs['O']
    dO = kwargs['dO']
    dK = kwargs['dK']
    dV = kwargs['dV']
    BLOCK_M = kwargs['BLOCK_M']
    BLOCK_N = kwargs['BLOCK_N']
    BLOCK_D = kwargs['BLOCK_D']
    
    kwargs['Q_desc'] = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M, BLOCK_D])
    kwargs['K_desc'] = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N, BLOCK_D])
    kwargs['V_desc'] = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N, BLOCK_D])
    kwargs['O_desc'] = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M, BLOCK_D])
    kwargs['dO_desc'] = TensorDescriptor.from_tensor(dO, [1, 1, BLOCK_M, BLOCK_D])
    kwargs['dK_desc'] = TensorDescriptor.from_tensor(dK, [1, 1, BLOCK_N, BLOCK_D])
    kwargs['dV_desc'] = TensorDescriptor.from_tensor(dV, [1, 1, BLOCK_N, BLOCK_D])

def get_dq_configs():
    configs = []
    # Exhaustively explore stages mapping cleanly to Blackwell's 228KiB limit shared boundaries.
    for ws in [False, True]:
        configs.append(triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'NUM_STAGES': 2, 'WARP_SPECIALIZE': ws}, num_warps=8, num_stages=2, pre_hook=dq_pre_hook))
        configs.append(triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'NUM_STAGES': 3, 'WARP_SPECIALIZE': ws}, num_warps=8, num_stages=3, pre_hook=dq_pre_hook))
        configs.append(triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'NUM_STAGES': 3, 'WARP_SPECIALIZE': ws}, num_warps=4, num_stages=3, pre_hook=dq_pre_hook))
        configs.append(triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'NUM_STAGES': 3, 'WARP_SPECIALIZE': ws}, num_warps=4, num_stages=3, pre_hook=dq_pre_hook))
    return configs

def get_dkdv_configs():
    configs = []
    for ws in [False, True]:
        configs.append(triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'NUM_STAGES': 2, 'WARP_SPECIALIZE': ws}, num_warps=8, num_stages=2, pre_hook=dkdv_pre_hook))
        configs.append(triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'NUM_STAGES': 3, 'WARP_SPECIALIZE': ws}, num_warps=8, num_stages=3, pre_hook=dkdv_pre_hook))
        configs.append(triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'NUM_STAGES': 3, 'WARP_SPECIALIZE': ws}, num_warps=8, num_stages=3, pre_hook=dkdv_pre_hook))
        configs.append(triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'NUM_STAGES': 3, 'WARP_SPECIALIZE': ws}, num_warps=4, num_stages=3, pre_hook=dkdv_pre_hook))
    return configs

@triton.autotune(configs=get_dq_configs(), key=['S'])
@triton.jit
def bwd_kernel_dq(
    Q, K, V, O, dO, L, dQ,
    Q_desc, K_desc, V_desc, O_desc, dO_desc, dQ_desc,
    stride_lb, stride_lh, stride_ls,
    B, H, S, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    NUM_STAGES: tl.constexpr, WARP_SPECIALIZE: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offset_m = pid_m * BLOCK_M
    
    # Extract M-block natively from physical descriptors completely outside the inner loop
    q = tl.reshape(Q_desc.load([b, h, offset_m, 0]), [BLOCK_M, BLOCK_D])
    do = tl.reshape(dO_desc.load([b, h, offset_m, 0]), [BLOCK_M, BLOCK_D])
    o = tl.reshape(O_desc.load([b, h, offset_m, 0]), [BLOCK_M, BLOCK_D])
    
    # Compute the fixed delta offset externally utilizing vector ALUs cleanly
    D_val = tl.sum(tl.cast(do, tl.float32) * tl.cast(o, tl.float32), axis=1)
    
    offs_m = offset_m + tl.arange(0, BLOCK_M)
    l_ptrs = L + b * stride_lb + h * stride_lh + offs_m * stride_ls
    mask_m = offs_m < S
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)

    # Gradient accumulator for Q firmly bound to register arrays
    dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    num_n_steps = tl.cdiv(S, BLOCK_N)
    
    # Native pipelining automatically leveraging Blackwell's TMA instruction queues
    for j in tl.range(0, num_n_steps, num_stages=NUM_STAGES, warp_specialize=WARP_SPECIALIZE):
        offset_n = j * BLOCK_N
        k = tl.reshape(K_desc.load([b, h, offset_n, 0]), [BLOCK_N, BLOCK_D])
        v = tl.reshape(V_desc.load([b, h, offset_n, 0]), [BLOCK_N, BLOCK_D])
        
        # Native MMA schedule resolving directly (Row x Col inherently bound via trans)
        s = tl.dot(q, tl.trans(k)) * sm_scale
        
        offs_n = offset_n + tl.arange(0, BLOCK_N)
        mask = mask_m[:, None] & (offs_n[None, :] < S)
        
        p = tl.where(mask, tl.exp(s - l[:, None]), 0.0)
        p_bf16 = tl.cast(p, tl.bfloat16)
        
        dp = tl.dot(do, tl.trans(v))
        
        ds = p * (dp - D_val[:, None]) * sm_scale
        ds_bf16 = tl.cast(ds, tl.bfloat16)
        
        dq = tl.dot(ds_bf16, k, acc=dq)
        
    dQ_desc.store([b, h, offset_m, 0], tl.reshape(tl.cast(dq, tl.bfloat16), [1, 1, BLOCK_M, BLOCK_D]))

@triton.autotune(configs=get_dkdv_configs(), key=['S'])
@triton.jit
def bwd_kernel_dk_dv(
    Q, K, V, O, dO, L, dK, dV,
    Q_desc, K_desc, V_desc, O_desc, dO_desc, dK_desc, dV_desc,
    stride_lb, stride_lh, stride_ls,
    B, H, S, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    NUM_STAGES: tl.constexpr, WARP_SPECIALIZE: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offset_n = pid_n * BLOCK_N
    
    k = tl.reshape(K_desc.load([b, h, offset_n, 0]), [BLOCK_N, BLOCK_D])
    v = tl.reshape(V_desc.load([b, h, offset_n, 0]), [BLOCK_N, BLOCK_D])
    
    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    
    offs_n = offset_n + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    
    num_m_steps = tl.cdiv(S, BLOCK_M)
    for i in tl.range(0, num_m_steps, num_stages=NUM_STAGES, warp_specialize=WARP_SPECIALIZE):
        offset_m = i * BLOCK_M
        q = tl.reshape(Q_desc.load([b, h, offset_m, 0]), [BLOCK_M, BLOCK_D])
        do = tl.reshape(dO_desc.load([b, h, offset_m, 0]), [BLOCK_M, BLOCK_D])
        o = tl.reshape(O_desc.load([b, h, offset_m, 0]), [BLOCK_M, BLOCK_D])
        
        offs_m = offset_m + tl.arange(0, BLOCK_M)
        l_ptrs = L + b * stride_lb + h * stride_lh + offs_m * stride_ls
        l = tl.load(l_ptrs, mask=offs_m < S, other=0.0)
        
        # Parallelizing D reduction intrinsically inside the pipeline stages overlapping safely with TMA delays
        D_val = tl.sum(tl.cast(do, tl.float32) * tl.cast(o, tl.float32), axis=1)
        
        mask = mask_n[:, None] & (offs_m[None, :] < S)
        
        # Evaluate transposed states completely free of memory footprint scaling via logical layout alignments
        s_T = tl.dot(k, tl.trans(q)) * sm_scale
        
        p_T = tl.where(mask, tl.exp(s_T - l[None, :]), 0.0)
        p_T_bf16 = tl.cast(p_T, tl.bfloat16)
        
        dv = tl.dot(p_T_bf16, do, acc=dv)
        
        dp_T = tl.dot(v, tl.trans(do))
        ds_T = p_T * (dp_T - D_val[None, :]) * sm_scale
        ds_T_bf16 = tl.cast(ds_T, tl.bfloat16)
        
        dk = tl.dot(ds_T_bf16, q, acc=dk)
        
    dK_desc.store([b, h, offset_n, 0], tl.reshape(tl.cast(dk, tl.bfloat16), [1, 1, BLOCK_N, BLOCK_D]))
    dV_desc.store([b, h, offset_n, 0], tl.reshape(tl.cast(dv, tl.bfloat16), [1, 1, BLOCK_N, BLOCK_D]))

def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Main destination-passing backward logic ensuring all Blackwell 5th-generation Tensor Cores and TMA flows are robustly achieved.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    sm_scale = 1.0 / math.sqrt(d)
    
    # Generating correctly sized runtime dummy descriptors so that the JIT type-resolution cleanly accepts them before pre-hooks replace with properly tailored autotuned scales.
    dummy_block = [1, 1, 64, d]
    Q_desc = TensorDescriptor.from_tensor(Q, dummy_block)
    K_desc = TensorDescriptor.from_tensor(K, dummy_block)
    V_desc = TensorDescriptor.from_tensor(V, dummy_block)
    O_desc = TensorDescriptor.from_tensor(O, dummy_block)
    dO_desc = TensorDescriptor.from_tensor(dO, dummy_block)
    dQ_desc = TensorDescriptor.from_tensor(dQ, dummy_block)
    dK_desc = TensorDescriptor.from_tensor(dK, dummy_block)
    dV_desc = TensorDescriptor.from_tensor(dV, dummy_block)
    
    grid_dq = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
    bwd_kernel_dq[grid_dq](
        Q, K, V, O, dO, L, dQ,
        Q_desc, K_desc, V_desc, O_desc, dO_desc, dQ_desc,
        L.stride(0), L.stride(1), L.stride(2),
        B, H, S, sm_scale,
        BLOCK_D=d
    )
    
    grid_dk_dv = lambda META: (triton.cdiv(S, META['BLOCK_N']), B * H)
    bwd_kernel_dk_dv[grid_dk_dv](
        Q, K, V, O, dO, L, dK, dV,
        Q_desc, K_desc, V_desc, O_desc, dO_desc, dK_desc, dV_desc,
        L.stride(0), L.stride(1), L.stride(2),
        B, H, S, sm_scale,
        BLOCK_D=d
    )
    return dQ, dK, dV