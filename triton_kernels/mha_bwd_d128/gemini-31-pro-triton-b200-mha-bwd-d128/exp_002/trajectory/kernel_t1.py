import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


def dq_pre_hook(kwargs):
    BLOCK_M = kwargs["BLOCK_M"]
    BLOCK_N = kwargs["BLOCK_N"]
    d = 128
    kwargs["Q_desc"] = TensorDescriptor.from_tensor(kwargs["Q_desc"], [1, 1, BLOCK_M, d])
    kwargs["K_desc"] = TensorDescriptor.from_tensor(kwargs["K_desc"], [1, 1, BLOCK_N, d])
    kwargs["V_desc"] = TensorDescriptor.from_tensor(kwargs["V_desc"], [1, 1, BLOCK_N, d])
    kwargs["O_desc"] = TensorDescriptor.from_tensor(kwargs["O_desc"], [1, 1, BLOCK_M, d])
    kwargs["dO_desc"] = TensorDescriptor.from_tensor(kwargs["dO_desc"], [1, 1, BLOCK_M, d])
    kwargs["dQ_desc"] = TensorDescriptor.from_tensor(kwargs["dQ_desc"], [1, 1, BLOCK_M, d])


def dkv_pre_hook(kwargs):
    BLOCK_M = kwargs["BLOCK_M"]
    BLOCK_N = kwargs["BLOCK_N"]
    d = 128
    kwargs["Q_desc"] = TensorDescriptor.from_tensor(kwargs["Q_desc"], [1, 1, BLOCK_M, d])
    kwargs["K_desc"] = TensorDescriptor.from_tensor(kwargs["K_desc"], [1, 1, BLOCK_N, d])
    kwargs["V_desc"] = TensorDescriptor.from_tensor(kwargs["V_desc"], [1, 1, BLOCK_N, d])
    kwargs["O_desc"] = TensorDescriptor.from_tensor(kwargs["O_desc"], [1, 1, BLOCK_M, d])
    kwargs["dO_desc"] = TensorDescriptor.from_tensor(kwargs["dO_desc"], [1, 1, BLOCK_M, d])
    kwargs["dK_desc"] = TensorDescriptor.from_tensor(kwargs["dK_desc"], [1, 1, BLOCK_N, d])
    kwargs["dV_desc"] = TensorDescriptor.from_tensor(kwargs["dV_desc"], [1, 1, BLOCK_N, d])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
    ],
    key=['S'],
    pre_hook=dq_pre_hook,
)
@triton.jit
def bwd_dq_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, dQ_desc,
    L, stride_lb, stride_lh, stride_ls,
    S, alpha, H: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
    q = Q_desc.load([b, h, pid_m * BLOCK_M, 0])
    o = O_desc.load([b, h, pid_m * BLOCK_M, 0])
    do = dO_desc.load([b, h, pid_m * BLOCK_M, 0])
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    
    L_ptr = L + b * stride_lb + h * stride_lh + offs_m * stride_ls
    l = tl.load(L_ptr, mask=mask_m, other=0.0)
    
    o_fp32 = o.to(tl.float32)
    do_fp32 = do.to(tl.float32)
    d_val = tl.sum(o_fp32 * do_fp32, axis=1)
    
    dq = tl.zeros([BLOCK_M, 128], dtype=tl.float32)
    
    for start_n in range(0, S, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        
        k = K_desc.load([b, h, start_n, 0])
        v = V_desc.load([b, h, start_n, 0])
        
        s_attn = tl.dot(q, k.T)
        s_attn = s_attn * alpha
        p = tl.exp(s_attn - l[:, None])
        p = tl.where(mask_m[:, None] & mask_n[None, :], p, 0.0)
        
        dp = tl.dot(do, v.T)
        
        da = p * (dp - d_val[:, None])
        da_bf16 = da.to(tl.bfloat16)
        
        dq += tl.dot(da_bf16, k)
        
    dq = dq * alpha
    dQ_desc.store([b, h, pid_m * BLOCK_M, 0], dq.to(tl.bfloat16))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 64}, num_warps=4, num_stages=3),
    ],
    key=['S'],
    pre_hook=dkv_pre_hook,
)
@triton.jit
def bwd_dkv_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, dK_desc, dV_desc,
    L, stride_lb, stride_lh, stride_ls,
    S, alpha, H: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
    k = K_desc.load([b, h, pid_n * BLOCK_N, 0])
    v = V_desc.load([b, h, pid_n * BLOCK_N, 0])
    
    dk = tl.zeros([BLOCK_N, 128], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, 128], dtype=tl.float32)
    
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    
    for start_m in range(0, S, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        
        q = Q_desc.load([b, h, start_m, 0])
        o = O_desc.load([b, h, start_m, 0])
        do = dO_desc.load([b, h, start_m, 0])
        
        L_ptr = L + b * stride_lb + h * stride_lh + offs_m * stride_ls
        l = tl.load(L_ptr, mask=mask_m, other=0.0)
        
        o_fp32 = o.to(tl.float32)
        do_fp32 = do.to(tl.float32)
        d_val = tl.sum(o_fp32 * do_fp32, axis=1)
        
        s_attn = tl.dot(q, k.T)
        s_attn = s_attn * alpha
        p = tl.exp(s_attn - l[:, None])
        p = tl.where(mask_m[:, None] & mask_n[None, :], p, 0.0)
        
        p_bf16 = p.to(tl.bfloat16)
        dv += tl.dot(p_bf16.T, do)
        
        dp = tl.dot(do, v.T)
        da = p * (dp - d_val[:, None])
        da_bf16 = da.to(tl.bfloat16)
        
        dk += tl.dot(da_bf16.T, q)
        
    dk = dk * alpha
    
    dK_desc.store([b, h, pid_n * BLOCK_N, 0], dk.to(tl.bfloat16))
    dV_desc.store([b, h, pid_n * BLOCK_N, 0], dv.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    alpha = 1.0 / math.sqrt(d)
    
    # Kernel 1: Calculate dQ iteratively over K and V blocks
    grid_dq = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
    bwd_dq_kernel[grid_dq](
        Q_desc=Q, K_desc=K, V_desc=V, 
        O_desc=O, dO_desc=dO, dQ_desc=dQ,
        L=L, stride_lb=L.stride(0), stride_lh=L.stride(1), stride_ls=L.stride(2),
        S=S, alpha=alpha, H=H
    )
    
    # Kernel 2: Calculate dK and dV iteratively over Q and O blocks
    grid_dkv = lambda META: (triton.cdiv(S, META['BLOCK_N']), B * H)
    bwd_dkv_kernel[grid_dkv](
        Q_desc=Q, K_desc=K, V_desc=V, 
        O_desc=O, dO_desc=dO, dK_desc=dK, dV_desc=dV,
        L=L, stride_lb=L.stride(0), stride_lh=L.stride(1), stride_ls=L.stride(2),
        S=S, alpha=alpha, H=H
    )
    
    return dQ, dK, dV