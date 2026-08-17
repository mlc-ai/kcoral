import math
import torch
import triton
import triton.language as tl

def get_dk_dv_configs():
    # We restrict num_stages for large blocks to ensure we stay within the 228KB Hopper shared memory limit.
    # Q, dO, and O are loaded inside the loop and will be staged.
    return [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    ]

@triton.autotune(
    configs=get_dk_dv_configs(),
    key=["S"],
)
@triton.jit
def bwd_kernel_dk_dv(
    Q, K, V, O, dO, L, dK, dV,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_dob, stride_doh, stride_dos,
    stride_dkb, stride_dkh, stride_dks,
    stride_dvb, stride_dvh, stride_dvs,
    stride_lb, stride_lh, stride_ls,
    S, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    start_n = pid_n * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    offs_d = tl.arange(0, d)
    
    # Load corresponding K and V blocks once for the outer loop iteration
    k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :]
    v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :]
    
    k_j = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v_j = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    
    dk_j = tl.zeros((BLOCK_N, d), tl.float32)
    dv_j = tl.zeros((BLOCK_N, d), tl.float32)
    
    num_m_blocks = tl.cdiv(S, BLOCK_M)
    
    # Loop over all blocks of M (Q, dO, O)
    for start_m in range(0, num_m_blocks * BLOCK_M, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        
        q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :]
        do_ptrs = dO + pid_b * stride_dob + pid_h * stride_doh + offs_m[:, None] * stride_dos + offs_d[None, :]
        o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :]
        
        q_i = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        do_i = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        o_i = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
        
        # Calculate D_i (row sum of element-wise multiply dO * O) on the fly
        d_i = tl.sum(tl.cast(do_i, tl.float32) * tl.cast(o_i, tl.float32), axis=1)
        
        l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ls
        l_i = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        # S_i,j = (Q_i @ K_j^T) * scale
        s_ij = tl.dot(q_i, k_j.T, out_dtype=tl.float32) * sm_scale
        p_ij = tl.exp(s_ij - l_i[:, None])
        
        # Eliminate out-of-bounds padded values to ensure exact 0 gradient contribution
        mask_mn = mask_m[:, None] & mask_n[None, :]
        p_ij = tl.where(mask_mn, p_ij, 0.0)
        
        # dP_ij = dO_i @ V_j^T
        dp_ij = tl.dot(do_i, v_j.T, out_dtype=tl.float32)
        # dS_ij = P_ij * (dP_ij - D_i)
        ds_ij = p_ij * (dp_ij - d_i[:, None])
        
        # Cast P and dS values back to low-precision matching inputs for fast tensor core dot
        p_ij_bf16 = tl.cast(p_ij, tl.bfloat16)
        ds_ij_bf16 = tl.cast(ds_ij * sm_scale, tl.bfloat16)
        
        # dV_j += P_ij^T @ dO_i
        dv_j = tl.dot(p_ij_bf16.T, do_i, dv_j, out_dtype=tl.float32)
        # dK_j += dS_ij^T @ Q_i
        dk_j = tl.dot(ds_ij_bf16.T, q_i, dk_j, out_dtype=tl.float32)
        
    dk_ptrs = dK + pid_b * stride_dkb + pid_h * stride_dkh + offs_n[:, None] * stride_dks + offs_d[None, :]
    dv_ptrs = dV + pid_b * stride_dvb + pid_h * stride_dvh + offs_n[:, None] * stride_dvs + offs_d[None, :]
    
    tl.store(dk_ptrs, tl.cast(dk_j, tl.bfloat16), mask=mask_n[:, None])
    tl.store(dv_ptrs, tl.cast(dv_j, tl.bfloat16), mask=mask_n[:, None])

def get_dq_configs():
    # In this kernel, Q, dO, and O are loaded outside the loop so they do not stress staging memory.
    # K and V are loaded inside the loop and staged.
    return [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    ]

@triton.autotune(
    configs=get_dq_configs(),
    key=["S"],
)
@triton.jit
def bwd_kernel_dq(
    Q, K, V, O, dO, L, dQ,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_dob, stride_doh, stride_dos,
    stride_dqb, stride_dqh, stride_dqs,
    stride_lb, stride_lh, stride_ls,
    S, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    offs_d = tl.arange(0, d)
    
    # Load corresponding Q, dO, and O blocks once for the outer loop iteration
    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :]
    do_ptrs = dO + pid_b * stride_dob + pid_h * stride_doh + offs_m[:, None] * stride_dos + offs_d[None, :]
    o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :]
    
    q_i = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    do_i = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    o_i = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    
    # Calculate D_i strictly locally for zero workspace allocation requirement
    d_i = tl.sum(tl.cast(do_i, tl.float32) * tl.cast(o_i, tl.float32), axis=1)
    
    l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ls
    l_i = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    dq_i = tl.zeros((BLOCK_M, d), tl.float32)
    
    num_n_blocks = tl.cdiv(S, BLOCK_N)
    
    # Loop over all blocks of N (K, V)
    for start_n in range(0, num_n_blocks * BLOCK_N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        
        k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :]
        v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :]
        
        k_j = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v_j = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        # S_i,j = (Q_i @ K_j^T) * scale
        s_ij = tl.dot(q_i, k_j.T, out_dtype=tl.float32) * sm_scale
        p_ij = tl.exp(s_ij - l_i[:, None])
        
        # Eliminate out-of-bounds padded values to ensure exact 0 gradient contribution
        mask_mn = mask_m[:, None] & mask_n[None, :]
        p_ij = tl.where(mask_mn, p_ij, 0.0)
        
        # dP_ij = dO_i @ V_j^T
        dp_ij = tl.dot(do_i, v_j.T, out_dtype=tl.float32)
        # dS_ij = P_ij * (dP_ij - D_i)
        ds_ij = p_ij * (dp_ij - d_i[:, None])
        
        ds_ij_bf16 = tl.cast(ds_ij * sm_scale, tl.bfloat16)
        
        # dQ_i += dS_ij @ K_j
        dq_i = tl.dot(ds_ij_bf16, k_j, dq_i, out_dtype=tl.float32)
        
    dq_ptrs = dQ + pid_b * stride_dqb + pid_h * stride_dqh + offs_m[:, None] * stride_dqs + offs_d[None, :]
    tl.store(dq_ptrs, tl.cast(dq_i, tl.bfloat16), mask=mask_m[:, None])

def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes the FlashAttention backward pass gradient w.r.t Q, K, and V.
    Arguments:
        Q, K, V: Attention inputs (B, H, S, d) of dtype bfloat16.
        O, dO: Forward output and output gradient (B, H, S, d) of dtype bfloat16.
        L: Forward log-sum-exp (B, H, S) of dtype float32.
        dQ, dK, dV: Preallocated output gradient buffers (B, H, S, d) of dtype bfloat16.
    """
    with torch.cuda.device(Q.device):
        B_sz, H, S, d = Q.shape
        sm_scale = 1.0 / math.sqrt(d)
        
        # Dispatch computing dK and dV via one kernel
        grid_dk_dv = lambda META: (B_sz, H, triton.cdiv(S, META["BLOCK_N"]))
        bwd_kernel_dk_dv[grid_dk_dv](
            Q, K, V, O, dO, L, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            L.stride(0), L.stride(1), L.stride(2),
            S, sm_scale,
            d=d
        )
        
        # Dispatch computing dQ via another kernel
        grid_dq = lambda META: (B_sz, H, triton.cdiv(S, META["BLOCK_M"]))
        bwd_kernel_dq[grid_dq](
            Q, K, V, O, dO, L, dQ,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            L.stride(0), L.stride(1), L.stride(2),
            S, sm_scale,
            d=d
        )