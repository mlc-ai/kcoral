import math
import torch
import triton
import triton.language as tl

# Configure Triton's descriptor allocator for device-side TMA descriptor creation.
# This is required when using tl.make_tensor_descriptor on Hopper.
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def get_configs():
    # Provide various tile sizes. The 1-kernel approach fuses all backward pass 
    # computations into one loop, avoiding redundant Q, dO, O loads.
    # We restrict `num_stages` to ensure we stay well within Hopper's 228KB SMEM limit.
    return [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    ]

@triton.autotune(
    configs=get_configs(),
    key=["S"],
)
@triton.jit
def bwd_kernel_all(
    Q, K, V, O, dO, L, dQ, dK, dV,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_dob, stride_doh, stride_dos,
    stride_dqb, stride_dqh, stride_dqs,
    stride_dkb, stride_dkh, stride_dks,
    stride_dvb, stride_dvh, stride_dvs,
    stride_lb, stride_lh, stride_ls,
    S, sm_scale, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    start_m = pid_m * BLOCK_M
    
    # Calculate base pointers for the specific batch and head
    q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
    do_ptr = dO + pid_b * stride_dob + pid_h * stride_doh
    o_ptr = O + pid_b * stride_ob + pid_h * stride_oh
    dq_ptr = dQ + pid_b * stride_dqb + pid_h * stride_dqh
    
    k_ptr = K + pid_b * stride_kb + pid_h * stride_kh
    v_ptr = V + pid_b * stride_vb + pid_h * stride_vh
    dk_ptr = dK + pid_b * stride_dkb + pid_h * stride_dkh
    dv_ptr = dV + pid_b * stride_dvb + pid_h * stride_dvh
    
    # Create TMA descriptors
    q_desc = tl.make_tensor_descriptor(q_ptr, shape=[S, d], strides=[stride_qs, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    do_desc = tl.make_tensor_descriptor(do_ptr, shape=[S, d], strides=[stride_dos, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    o_desc = tl.make_tensor_descriptor(o_ptr, shape=[S, d], strides=[stride_os, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    dq_desc = tl.make_tensor_descriptor(dq_ptr, shape=[S, d], strides=[stride_dqs, 1], block_shape=[BLOCK_M, d])
    
    k_desc = tl.make_tensor_descriptor(k_ptr, shape=[S, d], strides=[stride_ks, 1], block_shape=[BLOCK_N, d], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(v_ptr, shape=[S, d], strides=[stride_vs, 1], block_shape=[BLOCK_N, d], padding_option="zero")
    
    # Load Q, dO, O once per thread block using TMA
    q_i = q_desc.load([start_m, 0])
    do_i = do_desc.load([start_m, 0])
    o_i = o_desc.load([start_m, 0])
    
    # Precalculate D_i (row sum of element-wise multiply dO * O) entirely avoiding any extra storage
    d_i = tl.sum(tl.cast(do_i, tl.float32) * tl.cast(o_i, tl.float32), axis=1)
    
    offs_m = start_m + tl.arange(0, BLOCK_M)
    l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ls
    l_i = tl.load(l_ptrs, mask=offs_m < S, other=0.0)
    
    dq_i = tl.zeros((BLOCK_M, d), tl.float32)
    
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, d)
    
    # Base pointer layout blocks for atomic add
    dk_ptrs_base = dk_ptr + offs_n[:, None] * stride_dks + offs_d[None, :]
    dv_ptrs_base = dv_ptr + offs_n[:, None] * stride_dvs + offs_d[None, :]
    
    num_n_blocks = tl.cdiv(S, BLOCK_N)
    # Loop swizzling pattern: staggering the sequence start for N to effectively drop atomic 
    # write contention to exactly 0 when distributing cross-threaded `tl.atomic_add` workloads.
    start_n_idx = (pid_m * BLOCK_M // BLOCK_N) % num_n_blocks
    
    # Loop over all blocks of N
    for n_iter in range(0, num_n_blocks):
        n_block = (start_n_idx + n_iter) % num_n_blocks
        start_n = n_block * BLOCK_N
        
        k_j = k_desc.load([start_n, 0])
        v_j = v_desc.load([start_n, 0])
        
        # S_i,j = (Q_i @ K_j^T) * scale
        s_ij = tl.dot(q_i, k_j.T, out_dtype=tl.float32) * sm_scale
        
        # Eliminate out-of-bounds padded values strictly guaranteeing exact 0 gradient contribution 
        # and preventing NaN propagations from soft-clamping mechanism
        curr_n = start_n + offs_n
        mask_mn = (offs_m[:, None] < S) & (curr_n[None, :] < S)
        s_ij = tl.where(mask_mn, s_ij, float("-inf"))
        
        p_ij = tl.exp(s_ij - l_i[:, None])
        
        # dP_ij = dO_i @ V_j^T
        dp_ij = tl.dot(do_i, v_j.T, out_dtype=tl.float32)
        # dS_ij = P_ij * (dP_ij - D_i)
        ds_ij = p_ij * (dp_ij - d_i[:, None])
        
        # Cast P and dS values back to matching inputs for fast Tensor Core dots
        ds_ij_bf16 = tl.cast(ds_ij * sm_scale, tl.bfloat16)
        dq_i = tl.dot(ds_ij_bf16, k_j, dq_i, out_dtype=tl.float32)
        
        p_ij_bf16 = tl.cast(p_ij, tl.bfloat16)
        dv_j = tl.dot(p_ij_bf16.T, do_i, out_dtype=tl.float32)
        dk_j = tl.dot(ds_ij_bf16.T, q_i, out_dtype=tl.float32)
        
        # HBM atomic reduction targeting dK & dV concurrently
        mask_n = curr_n < S
        dk_ptrs_n = dk_ptrs_base + start_n * stride_dks
        dv_ptrs_n = dv_ptrs_base + start_n * stride_dvs
        
        tl.atomic_add(dk_ptrs_n, tl.cast(dk_j, tl.bfloat16), mask=mask_n[:, None], sem="relaxed")
        tl.atomic_add(dv_ptrs_n, tl.cast(dv_j, tl.bfloat16), mask=mask_n[:, None], sem="relaxed")
        
    dq_desc.store([start_m, 0], tl.cast(dq_i, tl.bfloat16))

def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes the FlashAttention-style exact backward pass gradient w.r.t Q, K, and V.
    Arguments:
        Q, K, V: Attention inputs (B, H, S, d) of dtype bfloat16.
        O, dO: Forward output and output gradient (B, H, S, d) of dtype bfloat16.
        L: Forward log-sum-exp (B, H, S) of dtype float32.
        dQ, dK, dV: Preallocated output gradient buffers (B, H, S, d) of dtype bfloat16.
    """
    with torch.cuda.device(Q.device):
        B_sz, H_sz, S, d = Q.shape
        sm_scale = 1.0 / math.sqrt(d)
        
        # Must zero initialize destination pointers given asynchronous atomic accumulations
        dK.zero_()
        dV.zero_()
        
        # Dispatch computing dQ, dK, dV using a fused block 1-kernel approach.
        # Grouped program ordered mapping explicitly isolates specific attention heads (B_sz * H_sz).
        grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B_sz * H_sz)
        
        bwd_kernel_all[grid](
            Q, K, V, O, dO, L, dQ, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            L.stride(0), L.stride(1), L.stride(2),
            S, sm_scale, H_sz, d=d
        )