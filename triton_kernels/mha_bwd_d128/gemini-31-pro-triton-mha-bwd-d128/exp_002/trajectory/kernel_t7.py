import math
import torch
import triton
import triton.language as tl

def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.jit
def compute_D_kernel(
    O, dO, D_out,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_db, stride_dh, stride_ds,
    S, d: tl.constexpr,
    BLOCK_M: tl.constexpr
):
    """
    Computes the row sum D_m = sum(O_m * dO_m) and stores it in the first two elements
    of the dQ output buffer to avoid redundant computation and loads in the main kernels.
    Bitcasts the 32-bit float to two 16-bit bfloat16 elements to retain exactly 100% precision.
    """
    pid_m = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    
    O_ptr = O + b * stride_ob + h * stride_oh
    dO_ptr = dO + b * stride_dob + h * stride_doh
    D_ptr = D_out + b * stride_db + h * stride_dh
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, d)
    mask_m = offs_m < S
    
    O_ptrs = O_ptr + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    dO_ptrs = dO_ptr + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    
    o = tl.load(O_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(dO_ptrs, mask=mask_m[:, None], other=0.0)
    
    D = tl.sum(tl.cast(o, tl.float32) * tl.cast(do, tl.float32), axis=1)
    
    D_bits = tl.cast(D, tl.uint32, bitcast=True)
    D_low16 = tl.cast(D_bits & 0xFFFF, tl.uint16)
    D_high16 = tl.cast(D_bits >> 16, tl.uint16)
    
    D_low_bf16 = tl.cast(D_low16, tl.bfloat16, bitcast=True)
    D_high_bf16 = tl.cast(D_high16, tl.bfloat16, bitcast=True)
    
    tl.store(D_ptr + offs_m * stride_ds + 0, D_low_bf16, mask=mask_m)
    tl.store(D_ptr + offs_m * stride_ds + 1, D_high_bf16, mask=mask_m)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def bwd_dq_kernel(
    Q, K, V, dO, L, dQ,
    sm_scale: tl.constexpr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
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
    dO_ptr = dO + b * stride_dob + h * stride_doh
    dQ_ptr = dQ + b * stride_dqb + h * stride_dqh

    Q_desc = tl.make_tensor_descriptor(Q_ptr, shape=[S, d], strides=[stride_qs, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    dO_desc = tl.make_tensor_descriptor(dO_ptr, shape=[S, d], strides=[stride_dos, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    dQ_desc = tl.make_tensor_descriptor(dQ_ptr, shape=[S, d], strides=[stride_dqs, 1], block_shape=[BLOCK_M, d])
    
    K_desc = tl.make_tensor_descriptor(K_ptr, shape=[S, d], strides=[stride_ks, 1], block_shape=[BLOCK_N, d], padding_option="zero")
    V_desc = tl.make_tensor_descriptor(V_ptr, shape=[S, d], strides=[stride_vs, 1], block_shape=[BLOCK_N, d], padding_option="zero")

    offset_m = pid_m * BLOCK_M
    Q_m = Q_desc.load([offset_m, 0])
    dO_m = dO_desc.load([offset_m, 0])
    
    L_ptr = L + b * stride_lb + h * stride_lh
    D_ptr = dQ + b * stride_dqb + h * stride_dqh
    
    offs_m = offset_m + tl.arange(0, BLOCK_M)
    m_is_full = (pid_m + 1) * BLOCK_M <= S
    
    L_m = tl.load(L_ptr + offs_m * stride_ls, mask=offs_m < S, other=0.0)
    
    # Retrieve precomputed exact 32-bit D_m
    D_low_bf16 = tl.load(D_ptr + offs_m * stride_dqs + 0, mask=offs_m < S, other=0.0)
    D_high_bf16 = tl.load(D_ptr + offs_m * stride_dqs + 1, mask=offs_m < S, other=0.0)
    D_low16 = tl.cast(D_low_bf16, tl.uint16, bitcast=True)
    D_high16 = tl.cast(D_high_bf16, tl.uint16, bitcast=True)
    D_bits = tl.cast(D_low16, tl.uint32) | (tl.cast(D_high16, tl.uint32) << 16)
    D_m = tl.cast(D_bits, tl.float32, bitcast=True)
    
    dQ_acc = tl.zeros((BLOCK_M, d), tl.float32)
    num_n_blocks = tl.cdiv(S, BLOCK_N)
    
    for start_n in tl.range(0, num_n_blocks):
        offset_n = start_n * BLOCK_N
        K_n = K_desc.load([offset_n, 0])
        V_n = V_desc.load([offset_n, 0])
        
        S_mn = tl.dot(Q_m, tl.trans(K_n), out_dtype=tl.float32) * sm_scale
        
        n_is_full = (start_n + 1) * BLOCK_N <= S
        if not (m_is_full and n_is_full):
            offs_n = offset_n + tl.arange(0, BLOCK_N)
            mask = (offs_m[:, None] < S) & (offs_n[None, :] < S)
            S_mn = tl.where(mask, S_mn, float("-inf"))
        
        P_mn = tl.exp(S_mn - L_m[:, None])
        dP_mn = tl.dot(dO_m, tl.trans(V_n), out_dtype=tl.float32)
        dS_mn = (P_mn * sm_scale) * (dP_mn - D_m[:, None])
        
        dS_scaled = tl.cast(dS_mn, tl.bfloat16)
        dQ_acc = tl.dot(dS_scaled, K_n, dQ_acc, out_dtype=tl.float32)

    dQ_desc.store([offset_m, 0], tl.cast(dQ_acc, tl.bfloat16))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def bwd_dk_dv_kernel(
    Q, K, V, dO, dQ, L, dK, dV,
    sm_scale: tl.constexpr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
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
    dO_ptr = dO + b * stride_dob + h * stride_doh
    dK_ptr = dK + b * stride_dkb + h * stride_dkh
    dV_ptr = dV + b * stride_dvb + h * stride_dvh

    Q_desc = tl.make_tensor_descriptor(Q_ptr, shape=[S, d], strides=[stride_qs, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    dO_desc = tl.make_tensor_descriptor(dO_ptr, shape=[S, d], strides=[stride_dos, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    K_desc = tl.make_tensor_descriptor(K_ptr, shape=[S, d], strides=[stride_ks, 1], block_shape=[BLOCK_N, d], padding_option="zero")
    V_desc = tl.make_tensor_descriptor(V_ptr, shape=[S, d], strides=[stride_vs, 1], block_shape=[BLOCK_N, d], padding_option="zero")
    dK_desc = tl.make_tensor_descriptor(dK_ptr, shape=[S, d], strides=[stride_dks, 1], block_shape=[BLOCK_N, d])
    dV_desc = tl.make_tensor_descriptor(dV_ptr, shape=[S, d], strides=[stride_dvs, 1], block_shape=[BLOCK_N, d])

    offset_n = pid_n * BLOCK_N
    K_n = K_desc.load([offset_n, 0])
    V_n = V_desc.load([offset_n, 0])
    
    offs_n = offset_n + tl.arange(0, BLOCK_N)
    n_is_full = (pid_n + 1) * BLOCK_N <= S
    
    L_ptr = L + b * stride_lb + h * stride_lh
    D_ptr = dQ + b * stride_dqb + h * stride_dqh
    
    dK_acc = tl.zeros((BLOCK_N, d), tl.float32)
    dV_acc = tl.zeros((BLOCK_N, d), tl.float32)

    num_m_blocks = tl.cdiv(S, BLOCK_M)
    
    for start_m in tl.range(0, num_m_blocks):
        offset_m = start_m * BLOCK_M
        
        Q_m = Q_desc.load([offset_m, 0])
        dO_m = dO_desc.load([offset_m, 0])
        
        offs_m = offset_m + tl.arange(0, BLOCK_M)
        L_m = tl.load(L_ptr + offs_m * stride_ls, mask=offs_m < S, other=0.0)
        
        D_low_bf16 = tl.load(D_ptr + offs_m * stride_dqs + 0, mask=offs_m < S, other=0.0)
        D_high_bf16 = tl.load(D_ptr + offs_m * stride_dqs + 1, mask=offs_m < S, other=0.0)
        D_low16 = tl.cast(D_low_bf16, tl.uint16, bitcast=True)
        D_high16 = tl.cast(D_high_bf16, tl.uint16, bitcast=True)
        D_bits = tl.cast(D_low16, tl.uint32) | (tl.cast(D_high16, tl.uint32) << 16)
        D_m = tl.cast(D_bits, tl.float32, bitcast=True)

        S_nm = tl.dot(K_n, tl.trans(Q_m), out_dtype=tl.float32) * sm_scale
        
        m_is_full = (start_m + 1) * BLOCK_M <= S
        if not (n_is_full and m_is_full):
            mask = (offs_n[:, None] < S) & (offs_m[None, :] < S)
            S_nm = tl.where(mask, S_nm, float("-inf"))
        
        P_nm = tl.exp(S_nm - L_m[None, :])
        P_b16 = tl.cast(P_nm, tl.bfloat16)
        
        dV_acc = tl.dot(P_b16, dO_m, dV_acc, out_dtype=tl.float32)
        
        dP_nm = tl.dot(V_n, tl.trans(dO_m), out_dtype=tl.float32)
        dS_nm = (P_nm * sm_scale) * (dP_nm - D_m[None, :])
        
        dS_scaled = tl.cast(dS_nm, tl.bfloat16)
        dK_acc = tl.dot(dS_scaled, Q_m, dK_acc, out_dtype=tl.float32)

    dK_desc.store([offset_n, 0], tl.cast(dK_acc, tl.bfloat16))
    dV_desc.store([offset_n, 0], tl.cast(dV_acc, tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes exact FlashAttention backward gradients eliminating massive HBM pressure
    via an inline Workspace reuse pattern mapping perfectly to Hopper WGMMA constraints.
    """
    torch.cuda.set_device(Q.device)
    
    if L.dim() == 4:
        L = L.squeeze(-1)
        
    B, H, S, d = Q.shape
    sm_scale = 1.0 / math.sqrt(d)
    
    grid_D = (triton.cdiv(S, 128), H, B)
    compute_D_kernel[grid_D](
        O, dO, dQ,
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dQ.stride(0), dQ.stride(1), dQ.stride(2),
        S, d, BLOCK_M=128
    )
    
    grid_dk_dv = lambda META: (triton.cdiv(S, META["BLOCK_N"]), B * H, 1)
    bwd_dk_dv_kernel[grid_dk_dv](
        Q, K, V, dO, dQ, L, dK, dV,
        sm_scale,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        B, H, S, d=d
    )
    
    grid_dq = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H, 1)
    bwd_dq_kernel[grid_dq](
        Q, K, V, dO, L, dQ,
        sm_scale,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        B, H, S, d=d
    )