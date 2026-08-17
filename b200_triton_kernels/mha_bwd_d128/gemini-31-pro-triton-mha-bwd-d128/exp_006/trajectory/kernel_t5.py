import torch
import triton
import triton.language as tl

# Set up the Triton allocator required for device-created tensor descriptors (TMA)
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)


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
    S, scale: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    # Base pointers for the current Batch and Head
    offset_q = pid_b * stride_qb + pid_h * stride_qh
    offset_k = pid_b * stride_kb + pid_h * stride_kh
    offset_v = pid_b * stride_vb + pid_h * stride_vh
    offset_o = pid_b * stride_ob + pid_h * stride_oh
    offset_do = pid_b * stride_dob + pid_h * stride_doh
    offset_dq = pid_b * stride_dqb + pid_h * stride_dqh

    # Create TMA descriptors for 2D slices (S x d)
    # The last stride is always 1 for the innermost dimension.
    q_desc = tl.make_tensor_descriptor(
        Q + offset_q, shape=[S, d], strides=[stride_qs, 1],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        O + offset_o, shape=[S, d], strides=[stride_os, 1],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    do_desc = tl.make_tensor_descriptor(
        dO + offset_do, shape=[S, d], strides=[stride_dos, 1],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    dq_desc = tl.make_tensor_descriptor(
        dQ + offset_dq, shape=[S, d], strides=[stride_dqs, 1],
        block_shape=[BLOCK_M, d]
    )
    
    k_desc = tl.make_tensor_descriptor(
        K + offset_k, shape=[S, d], strides=[stride_ks, 1],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V + offset_v, shape=[S, d], strides=[stride_vs, 1],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )

    offset_m = pid_m * BLOCK_M
    
    # Load Q, O, dO once for the current M block
    q = q_desc.load([offset_m, 0])
    o = o_desc.load([offset_m, 0])
    do = do_desc.load([offset_m, 0])
    
    offs_m = offset_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    
    l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ls
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    # Precompute rowsum(dO * O) in FP32
    d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    
    # Initialize FP32 accumulator for dQ
    dq = tl.zeros((BLOCK_M, d), tl.float32)
    
    num_blocks_n = tl.cdiv(S, BLOCK_N)
    
    for n in range(num_blocks_n):
        offset_n = n * BLOCK_N
        offs_n = offset_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        
        # Load K and V for the inner loop
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])
        
        # S_ij = Q @ K^T
        acc_s = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        s_val = tl.dot(q, tl.trans(k), acc_s)
        s_val = s_val * scale
        
        # P = exp(S - L), strictly applying bounds mask
        p = tl.exp(s_val - l[:, None])
        p = tl.where(mask_m[:, None] & mask_n[None, :], p, 0.0)
        
        # dP = dO @ V^T
        acc_dp = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        dp = tl.dot(do, tl.trans(v), acc_dp)
        
        # dS = P * (dP - D)
        ds = p * (dp - d_val[:, None]) * scale
        
        # dQ += dS @ K
        dq = tl.dot(ds.to(q.dtype), k, dq)
        
    dq_desc.store([offset_m, 0], dq.to(q.dtype))


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
    S, scale: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    offset_q = pid_b * stride_qb + pid_h * stride_qh
    offset_k = pid_b * stride_kb + pid_h * stride_kh
    offset_v = pid_b * stride_vb + pid_h * stride_vh
    offset_o = pid_b * stride_ob + pid_h * stride_oh
    offset_do = pid_b * stride_dob + pid_h * stride_doh
    offset_dk = pid_b * stride_dkb + pid_h * stride_dkh
    offset_dv = pid_b * stride_dvb + pid_h * stride_dvh

    q_desc = tl.make_tensor_descriptor(
        Q + offset_q, shape=[S, d], strides=[stride_qs, 1],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        O + offset_o, shape=[S, d], strides=[stride_os, 1],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    do_desc = tl.make_tensor_descriptor(
        dO + offset_do, shape=[S, d], strides=[stride_dos, 1],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        K + offset_k, shape=[S, d], strides=[stride_ks, 1],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V + offset_v, shape=[S, d], strides=[stride_vs, 1],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    dk_desc = tl.make_tensor_descriptor(
        dK + offset_dk, shape=[S, d], strides=[stride_dks, 1],
        block_shape=[BLOCK_N, d]
    )
    dv_desc = tl.make_tensor_descriptor(
        dV + offset_dv, shape=[S, d], strides=[stride_dvs, 1],
        block_shape=[BLOCK_N, d]
    )

    offset_n = pid_n * BLOCK_N
    
    # Load K and V once for the current N block
    k = k_desc.load([offset_n, 0])
    v = v_desc.load([offset_n, 0])
    
    offs_n = offset_n + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    
    dk = tl.zeros((BLOCK_N, d), tl.float32)
    dv = tl.zeros((BLOCK_N, d), tl.float32)
    
    num_blocks_m = tl.cdiv(S, BLOCK_M)
    for m in range(num_blocks_m):
        offset_m = m * BLOCK_M
        
        # Pipeline-load Q, O, dO
        q = q_desc.load([offset_m, 0])
        o = o_desc.load([offset_m, 0])
        do = do_desc.load([offset_m, 0])
        
        offs_m = offset_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        
        l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ls
        l = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        
        # Exploit (Q K^T)^T = K Q^T for avoiding register transposes. 
        # Resolves perfectly on WGMMA.
        acc_s = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        s_val_t = tl.dot(k, tl.trans(q), acc_s)
        s_val_t = s_val_t * scale
        
        p_t = tl.exp(s_val_t - l[None, :])
        p_t = tl.where(mask_n[:, None] & mask_m[None, :], p_t, 0.0)
        
        acc_dp = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        dp_t = tl.dot(v, tl.trans(do), acc_dp)
        
        ds_t = p_t * (dp_t - d_val[None, :]) * scale
        
        # dV += P^T @ dO
        dv = tl.dot(p_t.to(q.dtype), do, dv)
        # dK += dS^T @ Q
        dk = tl.dot(ds_t.to(q.dtype), q, dk)
        
    dk_desc.store([offset_n, 0], dk.to(k.dtype))
    dv_desc.store([offset_n, 0], dv.to(v.dtype))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes FlashAttention-3 backward pass (destination-passing variant).
    Leverages Hopper TMA and WGMMA natively with standard Triton `tl.make_tensor_descriptor`.
    """
    with torch.cuda.device(Q.device):
        B, H, S, d = Q.shape
        scale = 1.0 / (d ** 0.5)

        # 128 (inner) x 64 (outer) fits precisely into H100 CTA resource budgets
        BLOCK_M_DQ = 128
        BLOCK_N_DQ = 64
        grid_dq = (triton.cdiv(S, BLOCK_M_DQ), H, B)

        bwd_kernel_dq[grid_dq](
            Q, K, V, O, dO, L, dQ,
            Q.stride(0), Q.stride(1), Q.stride(2), 
            K.stride(0), K.stride(1), K.stride(2), 
            V.stride(0), V.stride(1), V.stride(2), 
            O.stride(0), O.stride(1), O.stride(2), 
            dO.stride(0), dO.stride(1), dO.stride(2), 
            dQ.stride(0), dQ.stride(1), dQ.stride(2), 
            L.stride(0), L.stride(1), L.stride(2),
            S, scale,
            BLOCK_M=BLOCK_M_DQ, BLOCK_N=BLOCK_N_DQ, d=d,
            num_warps=4, num_stages=3
        )

        # For dK/dV, keeping inner loop loaded tensors tight prevents SMEM spilling 
        # while using TMA effectively.
        BLOCK_M_DK = 64
        BLOCK_N_DK = 128
        grid_dk_dv = (triton.cdiv(S, BLOCK_N_DK), H, B)

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
            S, scale,
            BLOCK_M=BLOCK_M_DK, BLOCK_N=BLOCK_N_DK, d=d,
            num_warps=4, num_stages=3
        )