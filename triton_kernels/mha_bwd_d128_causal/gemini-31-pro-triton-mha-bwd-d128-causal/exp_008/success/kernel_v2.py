import math
import torch
import triton
import triton.language as tl

@triton.jit
def bwd_prep_kernel(
    O, dO, dQ,
    stride_o_b, stride_o_h, stride_o_s, stride_o_d,
    stride_do_b, stride_do_h, stride_do_s, stride_do_d,
    stride_dq_b, stride_dq_h, stride_dq_s, stride_dq_d,
    B, H, S, d: tl.constexpr,
    BLOCK_M: tl.constexpr
):
    """
    Computes D = rowsum(dO * O) and packs the exact 32-bit float values into the
    first two columns of dQ using a safe unsigned bitcast. This prevents having to
    load the massive O tensor during the inner loop of the dkdv kernel.
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
    start_m = pid_m * BLOCK_M
    if start_m >= S:
        return
        
    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    
    offs_d = tl.arange(0, d)
    
    o_ptrs = O + b * stride_o_b + h * stride_o_h + offs_m[:, None] * stride_o_s + offs_d[None, :] * stride_o_d
    do_ptrs = dO + b * stride_do_b + h * stride_do_h + offs_m[:, None] * stride_do_s + offs_d[None, :] * stride_do_d
    
    o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    
    D = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
    
    # Strictly unsigned 32-bit bitcasting guarantees perfectly preserved precision avoiding any sign extensions
    D_uint32 = D.to(tl.uint32, bitcast=True)
    D_low16 = (D_uint32 & 0xFFFF).to(tl.uint16)
    D_high16 = (D_uint32 >> 16).to(tl.uint16)
    
    dq_ptr_0 = dQ + b * stride_dq_b + h * stride_dq_h + offs_m * stride_dq_s + 0 * stride_dq_d
    dq_ptr_1 = dQ + b * stride_dq_b + h * stride_dq_h + offs_m * stride_dq_s + 1 * stride_dq_d
    
    tl.store(dq_ptr_0, D_low16.to(tl.bfloat16, bitcast=True), mask=mask_m)
    tl.store(dq_ptr_1, D_high16.to(tl.bfloat16, bitcast=True), mask=mask_m)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def bwd_dkdv_kernel(
    Q, K, V, dQ, dO, L, dK, dV,
    stride_q_b, stride_q_h, stride_q_s, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_dq_b, stride_dq_h, stride_dq_s, stride_dq_d,
    stride_do_b, stride_do_h, stride_do_s, stride_do_d,
    stride_l_b, stride_l_h, stride_l_s,
    stride_dk_b, stride_dk_h, stride_dk_s, stride_dk_d,
    stride_dv_b, stride_dv_h, stride_dv_s, stride_dv_d,
    scale,
    B, H, S, d: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
    start_n = pid_n * BLOCK_N
    if start_n >= S:
        return
    
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, d)
    mask_n = offs_n < S
    
    k_ptrs = K + b * stride_k_b + h * stride_k_h + offs_n[:, None] * stride_k_s + offs_d[None, :] * stride_k_d
    v_ptrs = V + b * stride_v_b + h * stride_v_h + offs_n[:, None] * stride_v_s + offs_d[None, :] * stride_v_d
    
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    
    dk = tl.zeros([BLOCK_N, d], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, d], dtype=tl.float32)
    
    start_m_initial = (start_n // BLOCK_M) * BLOCK_M
    limit_m = ((start_n + BLOCK_N + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    
    # Boundary blocks (with causal mask restriction)
    for start_m in range(start_m_initial, tl.minimum(limit_m, S), BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        
        q_ptrs = Q + b * stride_q_b + h * stride_q_h + offs_m[:, None] * stride_q_s + offs_d[None, :] * stride_q_d
        do_ptrs = dO + b * stride_do_b + h * stride_do_h + offs_m[:, None] * stride_do_s + offs_d[None, :] * stride_do_d
        l_ptrs = L + b * stride_l_b + h * stride_l_h + offs_m * stride_l_s
        
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        l = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        # Load exactly re-packed 32-bit floats D
        dq_ptr_0 = dQ + b * stride_dq_b + h * stride_dq_h + offs_m * stride_dq_s + 0 * stride_dq_d
        dq_ptr_1 = dQ + b * stride_dq_b + h * stride_dq_h + offs_m * stride_dq_s + 1 * stride_dq_d
        D_low_bf16 = tl.load(dq_ptr_0, mask=mask_m, other=0.0)
        D_high_bf16 = tl.load(dq_ptr_1, mask=mask_m, other=0.0)
        
        D_low32 = D_low_bf16.to(tl.uint16, bitcast=True).to(tl.uint32)
        D_high32 = D_high_bf16.to(tl.uint16, bitcast=True).to(tl.uint32)
        
        D_uint32 = D_low32 | (D_high32 << 16)
        D = D_uint32.to(tl.float32, bitcast=True)
        
        s_ji = tl.dot(k, q.T, out_dtype=tl.float32) * scale
        valid_mask_ji = (offs_n[:, None] <= offs_m[None, :]) & mask_n[:, None] & mask_m[None, :]
        s_ji = tl.where(valid_mask_ji, s_ji, float("-inf"))
        
        p_ji = tl.exp(s_ji - l[None, :])
        p_ji = tl.where(valid_mask_ji, p_ji, 0.0)
        
        dp_ji = tl.dot(v, do.T, out_dtype=tl.float32)
        ds_ji = p_ji * (dp_ji - D[None, :]) * scale
        
        ds_ji_bf16 = ds_ji.to(q.dtype)
        dk += tl.dot(ds_ji_bf16, q, out_dtype=tl.float32)
        
        p_ji_bf16 = p_ji.to(do.dtype)
        dv += tl.dot(p_ji_bf16, do, out_dtype=tl.float32)
        
    # Fully valid blocks (no causal mask constraint allows loop simplicity optimization)
    for start_m in range(limit_m, S, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        
        q_ptrs = Q + b * stride_q_b + h * stride_q_h + offs_m[:, None] * stride_q_s + offs_d[None, :] * stride_q_d
        do_ptrs = dO + b * stride_do_b + h * stride_do_h + offs_m[:, None] * stride_do_s + offs_d[None, :] * stride_do_d
        l_ptrs = L + b * stride_l_b + h * stride_l_h + offs_m * stride_l_s
        
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        l = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        dq_ptr_0 = dQ + b * stride_dq_b + h * stride_dq_h + offs_m * stride_dq_s + 0 * stride_dq_d
        dq_ptr_1 = dQ + b * stride_dq_b + h * stride_dq_h + offs_m * stride_dq_s + 1 * stride_dq_d
        D_low_bf16 = tl.load(dq_ptr_0, mask=mask_m, other=0.0)
        D_high_bf16 = tl.load(dq_ptr_1, mask=mask_m, other=0.0)
        
        D_low32 = D_low_bf16.to(tl.uint16, bitcast=True).to(tl.uint32)
        D_high32 = D_high_bf16.to(tl.uint16, bitcast=True).to(tl.uint32)
        
        D_uint32 = D_low32 | (D_high32 << 16)
        D = D_uint32.to(tl.float32, bitcast=True)
        
        s_ji = tl.dot(k, q.T, out_dtype=tl.float32) * scale
        valid_mask_ji = mask_n[:, None] & mask_m[None, :]
        s_ji = tl.where(valid_mask_ji, s_ji, float("-inf"))
        
        p_ji = tl.exp(s_ji - l[None, :])
        p_ji = tl.where(valid_mask_ji, p_ji, 0.0)
        
        dp_ji = tl.dot(v, do.T, out_dtype=tl.float32)
        ds_ji = p_ji * (dp_ji - D[None, :]) * scale
        
        ds_ji_bf16 = ds_ji.to(q.dtype)
        dk += tl.dot(ds_ji_bf16, q, out_dtype=tl.float32)
        
        p_ji_bf16 = p_ji.to(do.dtype)
        dv += tl.dot(p_ji_bf16, do, out_dtype=tl.float32)
        
    dk_ptrs = dK + b * stride_dk_b + h * stride_dk_h + offs_n[:, None] * stride_dk_s + offs_d[None, :] * stride_dk_d
    tl.store(dk_ptrs, dk.to(dK.dtype.element_ty), mask=mask_n[:, None])
    
    dv_ptrs = dV + b * stride_dv_b + h * stride_dv_h + offs_n[:, None] * stride_dv_s + offs_d[None, :] * stride_dv_d
    tl.store(dv_ptrs, dv.to(dV.dtype.element_ty), mask=mask_n[:, None])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def bwd_dq_kernel(
    Q, K, V, O, dO, L, dQ,
    stride_q_b, stride_q_h, stride_q_s, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_o_b, stride_o_h, stride_o_s, stride_o_d,
    stride_do_b, stride_do_h, stride_do_s, stride_do_d,
    stride_l_b, stride_l_h, stride_l_s,
    stride_dq_b, stride_dq_h, stride_dq_s, stride_dq_d,
    scale,
    B, H, S, d: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
    start_m = pid_m * BLOCK_M
    if start_m >= S:
        return
    
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, d)
    mask_m = offs_m < S
    
    q_ptrs = Q + b * stride_q_b + h * stride_q_h + offs_m[:, None] * stride_q_s + offs_d[None, :] * stride_q_d
    o_ptrs = O + b * stride_o_b + h * stride_o_h + offs_m[:, None] * stride_o_s + offs_d[None, :] * stride_o_d
    do_ptrs = dO + b * stride_do_b + h * stride_do_h + offs_m[:, None] * stride_do_s + offs_d[None, :] * stride_do_d
    l_ptrs = L + b * stride_l_b + h * stride_l_h + offs_m * stride_l_s
    
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    # Calculating D inside dQ guarantees absolutely 0 side-effect dependency errors during Triton's autotuner iterations!
    D = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
    
    dq = tl.zeros([BLOCK_M, d], dtype=tl.float32)
    
    limit_n = (start_m // BLOCK_N) * BLOCK_N
    
    # Fully valid blocks (no causal mask constraint)
    for start_n in range(0, limit_n, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        
        k_ptrs = K + b * stride_k_b + h * stride_k_h + offs_n[:, None] * stride_k_s + offs_d[None, :] * stride_k_d
        v_ptrs = V + b * stride_v_b + h * stride_v_h + offs_n[:, None] * stride_v_s + offs_d[None, :] * stride_v_d
        
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        s_ij = tl.dot(q, k.T, out_dtype=tl.float32) * scale
        
        valid_mask = mask_m[:, None] & mask_n[None, :]
        s_ij = tl.where(valid_mask, s_ij, float("-inf"))
        
        p_ij = tl.exp(s_ij - l[:, None])
        p_ij = tl.where(valid_mask, p_ij, 0.0)
        
        dp_ij = tl.dot(do, v.T, out_dtype=tl.float32)
        ds_ij = p_ij * (dp_ij - D[:, None]) * scale
        
        ds_ij_bf16 = ds_ij.to(q.dtype)
        dq += tl.dot(ds_ij_bf16, k, out_dtype=tl.float32)
        
    # Boundary blocks (with causal mask constraint)
    end_n = tl.minimum(S, start_m + BLOCK_M)
    for start_n in range(limit_n, end_n, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        
        k_ptrs = K + b * stride_k_b + h * stride_k_h + offs_n[:, None] * stride_k_s + offs_d[None, :] * stride_k_d
        v_ptrs = V + b * stride_v_b + h * stride_v_h + offs_n[:, None] * stride_v_s + offs_d[None, :] * stride_v_d
        
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        s_ij = tl.dot(q, k.T, out_dtype=tl.float32) * scale
        
        valid_mask = (offs_m[:, None] >= offs_n[None, :]) & mask_m[:, None] & mask_n[None, :]
        s_ij = tl.where(valid_mask, s_ij, float("-inf"))
        
        p_ij = tl.exp(s_ij - l[:, None])
        p_ij = tl.where(valid_mask, p_ij, 0.0)
        
        dp_ij = tl.dot(do, v.T, out_dtype=tl.float32)
        ds_ij = p_ij * (dp_ij - D[:, None]) * scale
        
        ds_ij_bf16 = ds_ij.to(q.dtype)
        dq += tl.dot(ds_ij_bf16, k, out_dtype=tl.float32)
        
    dq_ptrs = dQ + b * stride_dq_b + h * stride_dq_h + offs_m[:, None] * stride_dq_s + offs_d[None, :] * stride_dq_d
    tl.store(dq_ptrs, dq.to(dQ.dtype.element_ty), mask=mask_m[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    
    B, H, S, d_val = Q.shape
    scale = 1.0 / math.sqrt(d_val)
    
    stride_l_b = L.stride(0)
    stride_l_h = L.stride(1)
    stride_l_s = L.stride(2) if L.dim() >= 3 else 1
    
    # 1. Prep Kernel -> Extract and store Delta D across head elements safely leveraging unpopulated dQ block memory slots
    grid_prep = (triton.cdiv(S, 128), B * H)
    bwd_prep_kernel[grid_prep](
        O, dO, dQ,
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        B, H, S, d=d_val,
        BLOCK_M=128
    )
    
    # 2. Key / Value Gradients Kernel -> Calculates Key/Value Gradient allocations bypassing loading entirely O using passed D
    grid_dkdv = lambda META: (triton.cdiv(S, META["BLOCK_N"]), B * H)
    bwd_dkdv_kernel[grid_dkdv](
        Q, K, V, dQ, dO, L, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        stride_l_b, stride_l_h, stride_l_s,
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        scale,
        B, H, S, d=d_val,
    )
    
    # 3. Query Gradients Kernel -> Recalculates final dQ outputs rewriting prior D values, utilizing idempotent independent computation
    grid_dq = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H)
    bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, L, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        stride_l_b, stride_l_h, stride_l_s,
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        scale,
        B, H, S, d=d_val,
    )
    
    return dQ, dK, dV