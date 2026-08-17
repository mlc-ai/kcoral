import math
import torch
import triton
import triton.language as tl

def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
    ],
    key=["S"],
)
@triton.jit
def bwd_dq_kernel(
    Q, K, V, O, dO, dQ, L,
    sm_scale: tl.constexpr,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_dob, stride_doh, stride_dos,
    stride_dqb, stride_dqh, stride_dqs,
    stride_lb, stride_lh, stride_ls,
    B, H, S, d: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Triton perfectly hoists invariant descriptor configurations to the host
    Q_desc = tl.make_tensor_descriptor(Q, shape=[B, H, S, d], strides=[stride_qb, stride_qh, stride_qs, 1], block_shape=[1, 1, BLOCK_M, d], padding_option="zero")
    O_desc = tl.make_tensor_descriptor(O, shape=[B, H, S, d], strides=[stride_ob, stride_oh, stride_os, 1], block_shape=[1, 1, BLOCK_M, d], padding_option="zero")
    dO_desc = tl.make_tensor_descriptor(dO, shape=[B, H, S, d], strides=[stride_dob, stride_doh, stride_dos, 1], block_shape=[1, 1, BLOCK_M, d], padding_option="zero")
    dQ_desc = tl.make_tensor_descriptor(dQ, shape=[B, H, S, d], strides=[stride_dqb, stride_dqh, stride_dqs, 1], block_shape=[1, 1, BLOCK_M, d])
    
    K_desc = tl.make_tensor_descriptor(K, shape=[B, H, S, d], strides=[stride_kb, stride_kh, stride_ks, 1], block_shape=[1, 1, BLOCK_N, d], padding_option="zero")
    V_desc = tl.make_tensor_descriptor(V, shape=[B, H, S, d], strides=[stride_vb, stride_vh, stride_vs, 1], block_shape=[1, 1, BLOCK_N, d], padding_option="zero")

    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    offset_m = pid_m * BLOCK_M
    
    Q_m = Q_desc.load([b, h, offset_m, 0])
    O_m = O_desc.load([b, h, offset_m, 0])
    dO_m = dO_desc.load([b, h, offset_m, 0])
    
    L_ptr = L + b * stride_lb + h * stride_lh
    offs_m = offset_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    L_m = tl.load(L_ptr + offs_m * stride_ls, mask=mask_m, other=0.0)

    # Pre-calculated across the local M block before iterating N 
    D_m = tl.sum(tl.cast(O_m, tl.float32) * tl.cast(dO_m, tl.float32), axis=1)
    
    dQ_acc = tl.zeros((BLOCK_M, d), tl.float32)
    
    num_n_blocks = tl.cdiv(S, BLOCK_N)
    for start_n in range(0, num_n_blocks):
        offset_n = start_n * BLOCK_N
        
        K_n = K_desc.load([b, h, offset_n, 0])
        V_n = V_desc.load([b, h, offset_n, 0])
        
        S_mn = tl.dot(Q_m, tl.trans(K_n), out_dtype=tl.float32) * sm_scale
        
        offs_n = offset_n + tl.arange(0, BLOCK_N)
        mask = mask_m[:, None] & (offs_n[None, :] < S)
        S_mn = tl.where(mask, S_mn, float("-inf"))
        
        P_mn = tl.exp(S_mn - L_m[:, None])
        
        dP_mn = tl.dot(dO_m, tl.trans(V_n), out_dtype=tl.float32)
        dS_mn = P_mn * (dP_mn - D_m[:, None])
        
        dS_scaled = tl.cast(dS_mn * sm_scale, tl.bfloat16)
        dQ_acc = tl.dot(dS_scaled, K_n, dQ_acc, out_dtype=tl.float32)

    dQ_desc.store([b, h, offset_m, 0], tl.cast(dQ_acc, tl.bfloat16))

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128, "BLOCK_M": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 128, "BLOCK_M": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 64, "BLOCK_M": 128}, num_warps=4, num_stages=3),
    ],
    key=["S"],
)
@triton.jit
def bwd_dk_dv_kernel(
    Q, K, V, O, dO, dK, dV, L,
    sm_scale: tl.constexpr,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_dob, stride_doh, stride_dos,
    stride_dkb, stride_dkh, stride_dks,
    stride_dvb, stride_dvh, stride_dvs,
    stride_lb, stride_lh, stride_ls,
    B, H, S, d: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    K_desc = tl.make_tensor_descriptor(K, shape=[B, H, S, d], strides=[stride_kb, stride_kh, stride_ks, 1], block_shape=[1, 1, BLOCK_N, d], padding_option="zero")
    V_desc = tl.make_tensor_descriptor(V, shape=[B, H, S, d], strides=[stride_vb, stride_vh, stride_vs, 1], block_shape=[1, 1, BLOCK_N, d], padding_option="zero")
    dK_desc = tl.make_tensor_descriptor(dK, shape=[B, H, S, d], strides=[stride_dkb, stride_dkh, stride_dks, 1], block_shape=[1, 1, BLOCK_N, d])
    dV_desc = tl.make_tensor_descriptor(dV, shape=[B, H, S, d], strides=[stride_dvb, stride_dvh, stride_dvs, 1], block_shape=[1, 1, BLOCK_N, d])
    
    Q_desc = tl.make_tensor_descriptor(Q, shape=[B, H, S, d], strides=[stride_qb, stride_qh, stride_qs, 1], block_shape=[1, 1, BLOCK_M, d], padding_option="zero")
    O_desc = tl.make_tensor_descriptor(O, shape=[B, H, S, d], strides=[stride_ob, stride_oh, stride_os, 1], block_shape=[1, 1, BLOCK_M, d], padding_option="zero")
    dO_desc = tl.make_tensor_descriptor(dO, shape=[B, H, S, d], strides=[stride_dob, stride_doh, stride_dos, 1], block_shape=[1, 1, BLOCK_M, d], padding_option="zero")

    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    offset_n = pid_n * BLOCK_N
    
    K_n = K_desc.load([b, h, offset_n, 0])
    V_n = V_desc.load([b, h, offset_n, 0])
    
    offs_n = offset_n + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    
    L_ptr = L + b * stride_lb + h * stride_lh
    
    dK_acc = tl.zeros((BLOCK_N, d), tl.float32)
    dV_acc = tl.zeros((BLOCK_N, d), tl.float32)

    num_m_blocks = tl.cdiv(S, BLOCK_M)
    for start_m in range(0, num_m_blocks):
        offset_m = start_m * BLOCK_M
        
        Q_m = Q_desc.load([b, h, offset_m, 0])
        O_m = O_desc.load([b, h, offset_m, 0])
        dO_m = dO_desc.load([b, h, offset_m, 0])
        
        offs_m = offset_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        L_m = tl.load(L_ptr + offs_m * stride_ls, mask=mask_m, other=0.0)

        # In-line block computation allows us to quickly reuse L2-cached blocks without allocations
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

    dK_desc.store([b, h, offset_n, 0], tl.cast(dK_acc, tl.bfloat16))
    dV_desc.store([b, h, offset_n, 0], tl.cast(dV_acc, tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Dest-passing entry point for attention backward gradient materialization mapping directly
    to efficient Hopper TMA 4D block abstractions guaranteeing maximum interconnect usage.
    """
    torch.cuda.set_device(Q.device)
    
    if L.dim() == 4:
        L = L.squeeze(-1)
        
    B, H, S, d = Q.shape
    sm_scale = 1.0 / math.sqrt(d)
    
    grid_dq = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H, 1)
    bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, dQ, L,
        sm_scale,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        dO.stride(0), dO.stride(1), dO.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2),
        L.stride(0), L.stride(1), L.stride(2),
        B, H, S, d=d
    )
    
    grid_dk_dv = lambda META: (triton.cdiv(S, META["BLOCK_N"]), B * H, 1)
    bwd_dk_dv_kernel[grid_dk_dv](
        Q, K, V, O, dO, dK, dV, L,
        sm_scale,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        dO.stride(0), dO.stride(1), dO.stride(2),
        dK.stride(0), dK.stride(1), dK.stride(2),
        dV.stride(0), dV.stride(1), dV.stride(2),
        L.stride(0), L.stride(1), L.stride(2),
        B, H, S, d=d
    )