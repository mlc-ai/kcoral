import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def bwd_dq_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, dQ_desc,
    L,
    sm_scale: tl.constexpr,
    stride_lb, stride_lh, stride_ls,
    B, H, S, d: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offset_m = pid_m * BLOCK_M
    
    Q_m = Q_desc.load((b, h, offset_m, 0))
    O_m = O_desc.load((b, h, offset_m, 0))
    dO_m = dO_desc.load((b, h, offset_m, 0))
    
    L_ptr = L + b * stride_lb + h * stride_lh
    offs_m = offset_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    L_m = tl.load(L_ptr + offs_m * stride_ls, mask=mask_m, other=0.0)

    D_m = tl.sum(tl.cast(O_m, tl.float32) * tl.cast(dO_m, tl.float32), axis=1)
    
    dQ_acc = tl.zeros((BLOCK_M, d), tl.float32)
    
    num_n_blocks = tl.cdiv(S, BLOCK_N)
    
    # Use tl.range for proper compiler pipelining of the N loop
    for start_n in tl.range(0, num_n_blocks):
        offset_n = start_n * BLOCK_N
        
        K_n = K_desc.load((b, h, offset_n, 0))
        V_n = V_desc.load((b, h, offset_n, 0))
        
        S_mn = tl.dot(Q_m, tl.trans(K_n), out_dtype=tl.float32) * sm_scale
        
        offs_n = offset_n + tl.arange(0, BLOCK_N)
        mask = mask_m[:, None] & (offs_n[None, :] < S)
        S_mn = tl.where(mask, S_mn, float("-inf"))
        
        P_mn = tl.exp(S_mn - L_m[:, None])
        
        dP_mn = tl.dot(dO_m, tl.trans(V_n), out_dtype=tl.float32)
        dS_mn = P_mn * (dP_mn - D_m[:, None])
        
        dS_scaled = tl.cast(dS_mn * sm_scale, tl.bfloat16)
        dQ_acc = tl.dot(dS_scaled, K_n, dQ_acc, out_dtype=tl.float32)

    dQ_desc.store((b, h, offset_m, 0), tl.cast(dQ_acc, tl.bfloat16))


@triton.jit
def bwd_dk_dv_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, dK_desc, dV_desc,
    L,
    sm_scale: tl.constexpr,
    stride_lb, stride_lh, stride_ls,
    B, H, S, d: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offset_n = pid_n * BLOCK_N
    
    K_n = K_desc.load((b, h, offset_n, 0))
    V_n = V_desc.load((b, h, offset_n, 0))
    
    offs_n = offset_n + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    
    L_ptr = L + b * stride_lb + h * stride_lh
    
    dK_acc = tl.zeros((BLOCK_N, d), tl.float32)
    dV_acc = tl.zeros((BLOCK_N, d), tl.float32)

    num_m_blocks = tl.cdiv(S, BLOCK_M)
    for start_m in tl.range(0, num_m_blocks):
        offset_m = start_m * BLOCK_M
        
        Q_m = Q_desc.load((b, h, offset_m, 0))
        O_m = O_desc.load((b, h, offset_m, 0))
        dO_m = dO_desc.load((b, h, offset_m, 0))
        
        offs_m = offset_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        L_m = tl.load(L_ptr + offs_m * stride_ls, mask=mask_m, other=0.0)

        D_m = tl.sum(tl.cast(O_m, tl.float32) * tl.cast(dO_m, tl.float32), axis=1)

        S_mn = tl.dot(Q_m, tl.trans(K_n), out_dtype=tl.float32) * sm_scale
        
        mask = mask_m[:, None] & mask_n[None, :]
        S_mn = tl.where(mask, S_mn, float("-inf"))
        
        P_mn = tl.exp(S_mn - L_m[:, None])
        
        P_b16 = tl.cast(P_mn, tl.bfloat16)
        dV_acc = tl.dot(tl.trans(P_b16), dO_m, dV_acc, out_dtype=tl.float32)
        
        dP_mn = tl.dot(dO_m, tl.trans(V_n), out_dtype=tl.float32)
        dS_mn = P_mn * (dP_mn - D_m[:, None])
        
        dS_scaled = tl.cast(dS_mn * sm_scale, tl.bfloat16)
        dK_acc = tl.dot(tl.trans(dS_scaled), Q_m, dK_acc, out_dtype=tl.float32)

    dK_desc.store((b, h, offset_n, 0), tl.cast(dK_acc, tl.bfloat16))
    dV_desc.store((b, h, offset_n, 0), tl.cast(dV_acc, tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes gradients dQ, dK, dV for a multi-head attention backward pass.
    Uses Hopper TMA (Tensor Memory Accelerator) via Python-side TensorDescriptors
    to entirely bypass setup overhead and ensure maximum pipelined bandwidth.
    """
    torch.cuda.set_device(Q.device)
    if L.dim() == 4:
        L = L.squeeze(-1)
        
    B, H, S, d_val = Q.shape
    sm_scale = 1.0 / math.sqrt(d_val)
    
    # We statically specialize the best known H100 block configurations to avoid autotune hooks.
    BLOCK_M_DQ = 128
    BLOCK_N_DQ = 64
    BLOCK_M_DK = 64
    BLOCK_N_DK = 128
    
    # Preconstruct exactly sized descriptors on the host for TMA. 
    # Notice that standard Triton TensorDescriptors naturally infer the nested 4D strides.
    Q_desc_dq  = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M_DQ, d_val])
    O_desc_dq  = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M_DQ, d_val])
    dO_desc_dq = TensorDescriptor.from_tensor(dO, [1, 1, BLOCK_M_DQ, d_val])
    dQ_desc    = TensorDescriptor.from_tensor(dQ, [1, 1, BLOCK_M_DQ, d_val])
    
    K_desc_dq  = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N_DQ, d_val])
    V_desc_dq  = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N_DQ, d_val])
    
    Q_desc_dk  = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M_DK, d_val])
    O_desc_dk  = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M_DK, d_val])
    dO_desc_dk = TensorDescriptor.from_tensor(dO, [1, 1, BLOCK_M_DK, d_val])
    
    K_desc_dk  = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N_DK, d_val])
    V_desc_dk  = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N_DK, d_val])
    dK_desc    = TensorDescriptor.from_tensor(dK, [1, 1, BLOCK_N_DK, d_val])
    dV_desc    = TensorDescriptor.from_tensor(dV, [1, 1, BLOCK_N_DK, d_val])

    # Pass 1: compute dQ
    grid_dq = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H, 1)
    bwd_dq_kernel[grid_dq](
        Q_desc_dq, K_desc_dq, V_desc_dq, O_desc_dq, dO_desc_dq, dQ_desc,
        L, sm_scale,
        L.stride(0), L.stride(1), L.stride(2),
        B, H, S, d=d_val,
        BLOCK_M=BLOCK_M_DQ, BLOCK_N=BLOCK_N_DQ,
        num_warps=4, num_stages=3
    )
    
    # Pass 2: compute dK, dV
    grid_dk_dv = lambda META: (triton.cdiv(S, META["BLOCK_N"]), B * H, 1)
    bwd_dk_dv_kernel[grid_dk_dv](
        Q_desc_dk, K_desc_dk, V_desc_dk, O_desc_dk, dO_desc_dk, dK_desc, dV_desc,
        L, sm_scale,
        L.stride(0), L.stride(1), L.stride(2),
        B, H, S, d=d_val,
        BLOCK_M=BLOCK_M_DK, BLOCK_N=BLOCK_N_DK,
        num_warps=4, num_stages=3
    )