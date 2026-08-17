import math
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def bwd_dq_kernel(
    Q, K, V, O, dO, L, dQ,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    B, H, S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    q_bh_offset = pid_b * stride_qb + pid_h * stride_qh
    k_bh_offset = pid_b * stride_kb + pid_h * stride_kh
    v_bh_offset = pid_b * stride_vb + pid_h * stride_vh
    o_bh_offset = pid_b * stride_ob + pid_h * stride_oh
    do_bh_offset = pid_b * stride_dob + pid_h * stride_doh
    l_bh_offset = pid_b * stride_lb + pid_h * stride_lh
    dq_bh_offset = pid_b * stride_dqb + pid_h * stride_dqh
    
    q_desc = tl.make_tensor_descriptor(
        Q + q_bh_offset, shape=[S, d], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        O + o_bh_offset, shape=[S, d], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    do_desc = tl.make_tensor_descriptor(
        dO + do_bh_offset, shape=[S, d], strides=[stride_dos, stride_dod],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        K + k_bh_offset, shape=[S, d], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V + v_bh_offset, shape=[S, d], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    dq_desc = tl.make_tensor_descriptor(
        dQ + dq_bh_offset, shape=[S, d], strides=[stride_dqs, stride_dqd],
        block_shape=[BLOCK_M, d]
    )
    
    m_offset = pid_m * BLOCK_M
    q_i = q_desc.load([m_offset, 0])
    o_i = o_desc.load([m_offset, 0])
    do_i = do_desc.load([m_offset, 0])
    
    offs_m = m_offset + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    
    l_ptrs = L + l_bh_offset + offs_m * stride_ls
    l_i = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    # Precompute row sums in fp32
    D_i = tl.sum(tl.cast(do_i, tl.float32) * tl.cast(o_i, tl.float32), axis=1)
    
    dq_acc = tl.zeros((BLOCK_M, d), dtype=tl.float32)
    
    num_blocks_n = tl.cdiv(S, BLOCK_N)
    # Causal masking: limit inner loop over N to blocks that can contain keys where n <= m
    max_n_idx = (m_offset + BLOCK_M - 1) // BLOCK_N
    max_n_idx = tl.minimum(max_n_idx, num_blocks_n - 1)
    
    for n in range(0, max_n_idx + 1):
        n_offset = n * BLOCK_N
        k_j = k_desc.load([n_offset, 0])
        v_j = v_desc.load([n_offset, 0])
        
        acc_s = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        s_ij = tl.dot(q_i, k_j.T, acc=acc_s)
        s_ij = s_ij * scale
        
        offs_n = n_offset + tl.arange(0, BLOCK_N)
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        valid_mask = causal_mask & mask_m[:, None] & (offs_n[None, :] < S)
        
        p_ij = tl.exp(s_ij - l_i[:, None])
        p_ij = tl.where(valid_mask, p_ij, 0.0)
        
        acc_dp = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        dp_ij = tl.dot(do_i, v_j.T, acc=acc_dp)
        
        # dS calculation scaled
        ds_ij = p_ij * (dp_ij - D_i[:, None])
        ds_ij = ds_ij * scale
        
        ds_ij_bf16 = tl.cast(ds_ij, tl.bfloat16)
        dq_acc = tl.dot(ds_ij_bf16, k_j, acc=dq_acc)
        
    dq_desc.store([m_offset, 0], dq_acc.to(tl.bfloat16))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    ],
    key=["S"],
)
@triton.jit
def bwd_dk_dv_kernel(
    Q, K, V, O, dO, L, dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    B, H, S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    q_bh_offset = pid_b * stride_qb + pid_h * stride_qh
    k_bh_offset = pid_b * stride_kb + pid_h * stride_kh
    v_bh_offset = pid_b * stride_vb + pid_h * stride_vh
    o_bh_offset = pid_b * stride_ob + pid_h * stride_oh
    do_bh_offset = pid_b * stride_dob + pid_h * stride_doh
    l_bh_offset = pid_b * stride_lb + pid_h * stride_lh
    dk_bh_offset = pid_b * stride_dkb + pid_h * stride_dkh
    dv_bh_offset = pid_b * stride_dvb + pid_h * stride_dvh
    
    k_desc = tl.make_tensor_descriptor(
        K + k_bh_offset, shape=[S, d], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V + v_bh_offset, shape=[S, d], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    q_desc = tl.make_tensor_descriptor(
        Q + q_bh_offset, shape=[S, d], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        O + o_bh_offset, shape=[S, d], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    do_desc = tl.make_tensor_descriptor(
        dO + do_bh_offset, shape=[S, d], strides=[stride_dos, stride_dod],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    dk_desc = tl.make_tensor_descriptor(
        dK + dk_bh_offset, shape=[S, d], strides=[stride_dks, stride_dkd],
        block_shape=[BLOCK_N, d]
    )
    dv_desc = tl.make_tensor_descriptor(
        dV + dv_bh_offset, shape=[S, d], strides=[stride_dvs, stride_dvd],
        block_shape=[BLOCK_N, d]
    )
    
    n_offset = pid_n * BLOCK_N
    k_j = k_desc.load([n_offset, 0])
    v_j = v_desc.load([n_offset, 0])
    
    offs_n = n_offset + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    
    dk_acc = tl.zeros((BLOCK_N, d), dtype=tl.float32)
    dv_acc = tl.zeros((BLOCK_N, d), dtype=tl.float32)
    
    num_blocks_m = tl.cdiv(S, BLOCK_M)
    # Causal masking logic: avoid completely skipping query blocks before n block
    start_m = (pid_n * BLOCK_N) // BLOCK_M
    
    for m in range(start_m, num_blocks_m):
        m_offset = m * BLOCK_M
        
        q_i = q_desc.load([m_offset, 0])
        o_i = o_desc.load([m_offset, 0])
        do_i = do_desc.load([m_offset, 0])
        
        offs_m = m_offset + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        
        l_ptrs = L + l_bh_offset + offs_m * stride_ls
        l_i = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        D_i = tl.sum(tl.cast(do_i, tl.float32) * tl.cast(o_i, tl.float32), axis=1)
        
        acc_s = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        s_ij = tl.dot(q_i, k_j.T, acc=acc_s)
        s_ij = s_ij * scale
        
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        valid_mask = causal_mask & mask_m[:, None] & mask_n[None, :]
        
        p_ij = tl.exp(s_ij - l_i[:, None])
        p_ij = tl.where(valid_mask, p_ij, 0.0)
        
        acc_dp = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        dp_ij = tl.dot(do_i, v_j.T, acc=acc_dp)
        
        ds_ij = p_ij * (dp_ij - D_i[:, None])
        ds_ij = ds_ij * scale
        
        ds_ij_bf16 = tl.cast(ds_ij, tl.bfloat16)
        p_ij_bf16 = tl.cast(p_ij, tl.bfloat16)
        
        dv_acc = tl.dot(p_ij_bf16.T, do_i, acc=dv_acc)
        dk_acc = tl.dot(ds_ij_bf16.T, q_i, acc=dk_acc)
        
    dk_desc.store([n_offset, 0], dk_acc.to(tl.bfloat16))
    dv_desc.store([n_offset, 0], dv_acc.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Destination-passing driver for SDPA Backward on Hopper SM90 using TMA descriptor paths.
    """
    # Setup allocator strictly for device-created TMA descriptors storage
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)
    
    triton.set_allocator(alloc_fn)

    with torch.cuda.device(Q.device):
        B, H, S, D = Q.shape
        scale = 1.0 / math.sqrt(D)
        
        # Squeeze L if 4D
        L_3d = L.squeeze(-1) if L.dim() == 4 else L
        
        grid_dq = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H)
        bwd_dq_kernel[grid_dq](
            Q, K, V, O, dO, L_3d, dQ,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            L_3d.stride(0), L_3d.stride(1), L_3d.stride(2),
            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
            B, H, S, scale,
            d=128
        )
        
        grid_dkv = lambda META: (triton.cdiv(S, META["BLOCK_N"]), B * H)
        bwd_dk_dv_kernel[grid_dkv](
            Q, K, V, O, dO, L_3d, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            L_3d.stride(0), L_3d.stride(1), L_3d.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
            dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
            B, H, S, scale,
            d=128
        )
        
        return dQ, dK, dV