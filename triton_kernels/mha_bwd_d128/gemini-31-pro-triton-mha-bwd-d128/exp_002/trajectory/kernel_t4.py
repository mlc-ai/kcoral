import math
import torch
import triton
import triton.language as tl


def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    ],
    key=["S"],
)
@triton.jit
def bwd_dq_kernel(
    Q, K, V, O, dO, L, dQ,
    sm_scale: tl.constexpr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    B, H, S,
    d: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    Q_ptr = Q + b * stride_qb + h * stride_qh
    K_ptr = K + b * stride_kb + h * stride_kh
    V_ptr = V + b * stride_vb + h * stride_vh
    O_ptr = O + b * stride_ob + h * stride_oh
    dO_ptr = dO + b * stride_dob + h * stride_doh
    dQ_ptr = dQ + b * stride_dqb + h * stride_dqh

    # Reconstruct robust 2D device-descriptors (no out-of-bound TMA dimensionality issues)
    Q_desc = tl.make_tensor_descriptor(
        Q_ptr, shape=[S, d], strides=[stride_qs, 1], block_shape=[BLOCK_M, d], padding_option="zero"
    )
    O_desc = tl.make_tensor_descriptor(
        O_ptr, shape=[S, d], strides=[stride_os, 1], block_shape=[BLOCK_M, d], padding_option="zero"
    )
    dO_desc = tl.make_tensor_descriptor(
        dO_ptr, shape=[S, d], strides=[stride_dos, 1], block_shape=[BLOCK_M, d], padding_option="zero"
    )
    dQ_desc = tl.make_tensor_descriptor(
        dQ_ptr, shape=[S, d], strides=[stride_dqs, 1], block_shape=[BLOCK_M, d]
    )
    K_desc = tl.make_tensor_descriptor(
        K_ptr, shape=[S, d], strides=[stride_ks, 1], block_shape=[BLOCK_N, d], padding_option="zero"
    )
    V_desc = tl.make_tensor_descriptor(
        V_ptr, shape=[S, d], strides=[stride_vs, 1], block_shape=[BLOCK_N, d], padding_option="zero"
    )

    offset_m = pid_m * BLOCK_M
    Q_m = Q_desc.load([offset_m, 0])
    O_m = O_desc.load([offset_m, 0])
    dO_m = dO_desc.load([offset_m, 0])
    
    L_ptr = L + b * stride_lb + h * stride_lh
    offs_m = offset_m + tl.arange(0, BLOCK_M)
    L_m = tl.load(L_ptr + offs_m * stride_ls, mask=offs_m < S, other=0.0)

    # Pre-calculated invariant row sums before hitting inner bottleneck
    D_m = tl.sum(tl.cast(O_m, tl.float32) * tl.cast(dO_m, tl.float32), axis=1)
    
    dQ_acc = tl.zeros((BLOCK_M, d), tl.float32)
    num_n_blocks = tl.cdiv(S, BLOCK_N)
    
    for start_n in tl.range(0, num_n_blocks):
        offset_n = start_n * BLOCK_N
        K_n = K_desc.load([offset_n, 0])
        V_n = V_desc.load([offset_n, 0])
        
        S_mn = tl.dot(Q_m, tl.trans(K_n), out_dtype=tl.float32) * sm_scale
        
        offs_n = offset_n + tl.arange(0, BLOCK_N)
        mask = (offs_m[:, None] < S) & (offs_n[None, :] < S)
        S_mn = tl.where(mask, S_mn, float("-inf"))
        
        P_mn = tl.exp(S_mn - L_m[:, None])
        
        dP_mn = tl.dot(dO_m, tl.trans(V_n), out_dtype=tl.float32)
        dS_mn = P_mn * (dP_mn - D_m[:, None])
        
        dS_scaled = tl.cast(dS_mn * sm_scale, tl.bfloat16)
        dQ_acc = tl.dot(dS_scaled, K_n, dQ_acc, out_dtype=tl.float32)

    dQ_desc.store([offset_m, 0], tl.cast(dQ_acc, tl.bfloat16))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    ],
    key=["S"],
)
@triton.jit
def bwd_dk_dv_kernel(
    Q, K, V, O, dO, L, dK, dV,
    sm_scale: tl.constexpr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    B, H, S,
    d: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    Q_ptr = Q + b * stride_qb + h * stride_qh
    K_ptr = K + b * stride_kb + h * stride_kh
    V_ptr = V + b * stride_vb + h * stride_vh
    O_ptr = O + b * stride_ob + h * stride_oh
    dO_ptr = dO + b * stride_dob + h * stride_doh
    dK_ptr = dK + b * stride_dkb + h * stride_dkh
    dV_ptr = dV + b * stride_dvb + h * stride_dvh

    Q_desc = tl.make_tensor_descriptor(
        Q_ptr, shape=[S, d], strides=[stride_qs, 1], block_shape=[BLOCK_M, d], padding_option="zero"
    )
    O_desc = tl.make_tensor_descriptor(
        O_ptr, shape=[S, d], strides=[stride_os, 1], block_shape=[BLOCK_M, d], padding_option="zero"
    )
    dO_desc = tl.make_tensor_descriptor(
        dO_ptr, shape=[S, d], strides=[stride_dos, 1], block_shape=[BLOCK_M, d], padding_option="zero"
    )
    K_desc = tl.make_tensor_descriptor(
        K_ptr, shape=[S, d], strides=[stride_ks, 1], block_shape=[BLOCK_N, d], padding_option="zero"
    )
    V_desc = tl.make_tensor_descriptor(
        V_ptr, shape=[S, d], strides=[stride_vs, 1], block_shape=[BLOCK_N, d], padding_option="zero"
    )
    dK_desc = tl.make_tensor_descriptor(
        dK_ptr, shape=[S, d], strides=[stride_dks, 1], block_shape=[BLOCK_N, d]
    )
    dV_desc = tl.make_tensor_descriptor(
        dV_ptr, shape=[S, d], strides=[stride_dvs, 1], block_shape=[BLOCK_N, d]
    )

    offset_n = pid_n * BLOCK_N
    K_n = K_desc.load([offset_n, 0])
    V_n = V_desc.load([offset_n, 0])
    
    offs_n = offset_n + tl.arange(0, BLOCK_N)
    L_ptr = L + b * stride_lb + h * stride_lh
    
    dK_acc = tl.zeros((BLOCK_N, d), tl.float32)
    dV_acc = tl.zeros((BLOCK_N, d), tl.float32)

    num_m_blocks = tl.cdiv(S, BLOCK_M)
    for start_m in tl.range(0, num_m_blocks):
        offset_m = start_m * BLOCK_M
        
        Q_m = Q_desc.load([offset_m, 0])
        O_m = O_desc.load([offset_m, 0])
        dO_m = dO_desc.load([offset_m, 0])
        
        offs_m = offset_m + tl.arange(0, BLOCK_M)
        L_m = tl.load(L_ptr + offs_m * stride_ls, mask=offs_m < S, other=0.0)

        D_m = tl.sum(tl.cast(O_m, tl.float32) * tl.cast(dO_m, tl.float32), axis=1)

        # Implicit mathematical formulation preventing massive register transpositions
        # Produces result `[BLOCK_N, BLOCK_M]` mapping purely to WGMMA A (registers) & WGMMA B (SRAM)
        S_nm = tl.dot(K_n, tl.trans(Q_m), out_dtype=tl.float32) * sm_scale
        
        mask = (offs_n[:, None] < S) & (offs_m[None, :] < S)
        S_nm = tl.where(mask, S_nm, float("-inf"))
        
        P_nm = tl.exp(S_nm - L_m[None, :])
        P_b16 = tl.cast(P_nm, tl.bfloat16)
        
        dV_acc = tl.dot(P_b16, dO_m, dV_acc, out_dtype=tl.float32)
        
        dP_nm = tl.dot(V_n, tl.trans(dO_m), out_dtype=tl.float32)
        dS_nm = P_nm * (dP_nm - D_m[None, :])
        
        dS_scaled = tl.cast(dS_nm * sm_scale, tl.bfloat16)
        dK_acc = tl.dot(dS_scaled, Q_m, dK_acc, out_dtype=tl.float32)

    dK_desc.store([offset_n, 0], tl.cast(dK_acc, tl.bfloat16))
    dV_desc.store([offset_n, 0], tl.cast(dV_acc, tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes gradients dQ, dK, dV.
    """
    torch.cuda.set_device(Q.device)
    
    if L.dim() == 4:
        L = L.squeeze(-1)
        
    B, H, S, d = Q.shape
    sm_scale = 1.0 / math.sqrt(d)
    
    # Pass 1: compute dQ
    grid_dq = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H, 1)
    bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, L, dQ,
        sm_scale,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        B, H, S,
        d=d
    )
    
    # Pass 2: compute dK, dV
    grid_dk_dv = lambda META: (triton.cdiv(S, META["BLOCK_N"]), B * H, 1)
    bwd_dk_dv_kernel[grid_dk_dv](
        Q, K, V, O, dO, L, dK, dV,
        sm_scale,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        B, H, S,
        d=d
    )