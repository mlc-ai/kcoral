import math
import torch
import triton
import triton.language as tl

# Configure Triton's descriptor allocator for device-side TMA descriptor creation.
# This is required when using tl.make_tensor_descriptor on Hopper.
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def get_dq_configs():
    # Provide multiple tile sizes. num_stages is carefully restricted to stay
    # comfortably within Hopper's ~228KB usable shared memory limit per SM.
    return [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
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
    S, sm_scale, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    
    # Calculate base pointers for the specific batch and head
    q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
    do_ptr = dO + pid_b * stride_dob + pid_h * stride_doh
    o_ptr = O + pid_b * stride_ob + pid_h * stride_oh
    dq_ptr = dQ + pid_b * stride_dqb + pid_h * stride_dqh
    
    k_ptr = K + pid_b * stride_kb + pid_h * stride_kh
    v_ptr = V + pid_b * stride_vb + pid_h * stride_vh
    
    # Create TMA descriptors
    q_desc = tl.make_tensor_descriptor(q_ptr, shape=[S, d], strides=[stride_qs, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    do_desc = tl.make_tensor_descriptor(do_ptr, shape=[S, d], strides=[stride_dos, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    o_desc = tl.make_tensor_descriptor(o_ptr, shape=[S, d], strides=[stride_os, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    dq_desc = tl.make_tensor_descriptor(dq_ptr, shape=[S, d], strides=[stride_dqs, 1], block_shape=[BLOCK_M, d])
    
    k_desc = tl.make_tensor_descriptor(k_ptr, shape=[S, d], strides=[stride_ks, 1], block_shape=[BLOCK_N, d], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(v_ptr, shape=[S, d], strides=[stride_vs, 1], block_shape=[BLOCK_N, d], padding_option="zero")
    
    # Load Q, dO, O for the current M block once via TMA
    q_i = q_desc.load([start_m, 0])
    do_i = do_desc.load([start_m, 0])
    o_i = o_desc.load([start_m, 0])
    
    # Precompute D_i (row sum of pointwise multiply dO * O) strictly locally for zero workspace
    d_i = tl.sum(tl.cast(do_i, tl.float32) * tl.cast(o_i, tl.float32), axis=1)
    
    l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ls
    l_i = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    dq_i = tl.zeros((BLOCK_M, d), tl.float32)
    
    num_n_blocks = tl.cdiv(S, BLOCK_N)
    for n_idx in range(num_n_blocks):
        start_n = n_idx * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        
        k_j = k_desc.load([start_n, 0])
        v_j = v_desc.load([start_n, 0])
        
        # S_i,j = (Q_i @ K_j^T) * scale
        s_ij = tl.dot(q_i, k_j.T, out_dtype=tl.float32) * sm_scale
        
        # Properly mask padding to -inf so exp(-inf) cleanly evaluates to 0.0
        s_ij = tl.where(mask_m[:, None], s_ij, float("-inf"))
        s_ij = tl.where(mask_n[None, :], s_ij, float("-inf"))
        
        p_ij = tl.exp(s_ij - l_i[:, None])
        
        # dP_ij = dO_i @ V_j^T
        dp_ij = tl.dot(do_i, v_j.T, out_dtype=tl.float32)
        
        # dS_ij = P_ij * (dP_ij - D_i)
        ds_ij = p_ij * (dp_ij - d_i[:, None])
        
        ds_ij_bf16 = tl.cast(ds_ij * sm_scale, tl.bfloat16)
        
        # dQ_i += dS_ij @ K_j
        dq_i = tl.dot(ds_ij_bf16, k_j, dq_i, out_dtype=tl.float32)
        
    dq_desc.store([start_m, 0], tl.cast(dq_i, tl.bfloat16))


def get_dk_dv_configs():
    return [
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
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
    S, sm_scale, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    start_n = pid_n * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    
    q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
    do_ptr = dO + pid_b * stride_dob + pid_h * stride_doh
    o_ptr = O + pid_b * stride_ob + pid_h * stride_oh
    k_ptr = K + pid_b * stride_kb + pid_h * stride_kh
    v_ptr = V + pid_b * stride_vb + pid_h * stride_vh
    dk_ptr = dK + pid_b * stride_dkb + pid_h * stride_dkh
    dv_ptr = dV + pid_b * stride_dvb + pid_h * stride_dvh
    
    q_desc = tl.make_tensor_descriptor(q_ptr, shape=[S, d], strides=[stride_qs, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    do_desc = tl.make_tensor_descriptor(do_ptr, shape=[S, d], strides=[stride_dos, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    o_desc = tl.make_tensor_descriptor(o_ptr, shape=[S, d], strides=[stride_os, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    
    k_desc = tl.make_tensor_descriptor(k_ptr, shape=[S, d], strides=[stride_ks, 1], block_shape=[BLOCK_N, d], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(v_ptr, shape=[S, d], strides=[stride_vs, 1], block_shape=[BLOCK_N, d], padding_option="zero")
    dk_desc = tl.make_tensor_descriptor(dk_ptr, shape=[S, d], strides=[stride_dks, 1], block_shape=[BLOCK_N, d])
    dv_desc = tl.make_tensor_descriptor(dv_ptr, shape=[S, d], strides=[stride_dvs, 1], block_shape=[BLOCK_N, d])
    
    k_j = k_desc.load([start_n, 0])
    v_j = v_desc.load([start_n, 0])
    
    dk_j = tl.zeros((BLOCK_N, d), tl.float32)
    dv_j = tl.zeros((BLOCK_N, d), tl.float32)
    
    num_m_blocks = tl.cdiv(S, BLOCK_M)
    for m_idx in range(num_m_blocks):
        start_m = m_idx * BLOCK_M
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        
        q_i = q_desc.load([start_m, 0])
        do_i = do_desc.load([start_m, 0])
        o_i = o_desc.load([start_m, 0])
        
        d_i = tl.sum(tl.cast(do_i, tl.float32) * tl.cast(o_i, tl.float32), axis=1)
        
        l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ls
        l_i = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        s_ij = tl.dot(q_i, k_j.T, out_dtype=tl.float32) * sm_scale
        s_ij = tl.where(mask_m[:, None], s_ij, float("-inf"))
        s_ij = tl.where(mask_n[None, :], s_ij, float("-inf"))
        
        p_ij = tl.exp(s_ij - l_i[:, None])
        dp_ij = tl.dot(do_i, v_j.T, out_dtype=tl.float32)
        ds_ij = p_ij * (dp_ij - d_i[:, None])
        
        p_ij_bf16 = tl.cast(p_ij, tl.bfloat16)
        ds_ij_bf16 = tl.cast(ds_ij * sm_scale, tl.bfloat16)
        
        # dV_j += P_ij^T @ dO_i
        dv_j = tl.dot(p_ij_bf16.T, do_i, dv_j, out_dtype=tl.float32)
        # dK_j += dS_ij^T @ Q_i
        dk_j = tl.dot(ds_ij_bf16.T, q_i, dk_j, out_dtype=tl.float32)
        
    dk_desc.store([start_n, 0], tl.cast(dk_j, tl.bfloat16))
    dv_desc.store([start_n, 0], tl.cast(dv_j, tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes the FlashAttention backward pass gradient w.r.t Q, K, and V.
    """
    with torch.cuda.device(Q.device):
        B_sz, H_sz, S, d = Q.shape
        sm_scale = 1.0 / math.sqrt(d)
        
        # We group the grid launch by heads (B_sz * H_sz) on the Y axis, which 
        # leverages optimal Hopper L2 locality across blocks of identical heads.
        
        grid_dq = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B_sz * H_sz)
        bwd_kernel_dq[grid_dq](
            Q, K, V, O, dO, L, dQ,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            L.stride(0), L.stride(1), L.stride(2),
            S, sm_scale, H_sz, d=d
        )
        
        grid_dk_dv = lambda META: (triton.cdiv(S, META["BLOCK_N"]), B_sz * H_sz)
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
            S, sm_scale, H_sz, d=d
        )