import torch
import triton
import triton.language as tl


@triton.jit
def bwd_kernel_dq(
    Q, K, V, O, dO, L, dQ,
    stride_q_b, stride_q_h, stride_q_s, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_o_b, stride_o_h, stride_o_s, stride_o_d,
    stride_do_b, stride_do_h, stride_do_s, stride_do_d,
    stride_l_b, stride_l_h, stride_l_s,
    stride_dq_b, stride_dq_h, stride_dq_s, stride_dq_d,
    B, H, S, d_scale,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    start_q = pid_q * BLOCK_Q
    offs_q = start_q + tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, BLOCK_D)

    offset_q = pid_b * stride_q_b + pid_h * stride_q_h
    offset_k = pid_b * stride_k_b + pid_h * stride_k_h
    offset_v = pid_b * stride_v_b + pid_h * stride_v_h
    offset_o = pid_b * stride_o_b + pid_h * stride_o_h
    offset_do = pid_b * stride_do_b + pid_h * stride_do_h
    offset_l = pid_b * stride_l_b + pid_h * stride_l_h

    q_ptrs = Q + offset_q + offs_q[:, None] * stride_q_s + offs_d[None, :] * stride_q_d
    o_ptrs = O + offset_o + offs_q[:, None] * stride_o_s + offs_d[None, :] * stride_o_d
    do_ptrs = dO + offset_do + offs_q[:, None] * stride_do_s + offs_d[None, :] * stride_do_d
    l_ptrs = L + offset_l + offs_q * stride_l_s

    mask_q = offs_q < S
    
    q = tl.load(q_ptrs, mask=mask_q[:, None], other=0.0)
    o = tl.load(o_ptrs, mask=mask_q[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=mask_q[:, None], other=0.0)
    l = tl.load(l_ptrs, mask=mask_q, other=0.0)

    # Pre-reduce per Q block independently in registers
    d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    
    dq_acc = tl.zeros([BLOCK_Q, BLOCK_D], dtype=tl.float32)

    # Leverage causality logic constraint to slice off unneeded operations mapping the K iteration boundary
    end_k = tl.minimum((start_q + BLOCK_Q - 1) // BLOCK_K + 1, tl.cdiv(S, BLOCK_K))

    offs_k = tl.arange(0, BLOCK_K)
    k_ptrs = K + offset_k + offs_k[:, None] * stride_k_s + offs_d[None, :] * stride_k_d
    v_ptrs = V + offset_v + offs_k[:, None] * stride_v_s + offs_d[None, :] * stride_v_d

    for i in range(end_k):
        start_k = i * BLOCK_K
        offs_k_curr = start_k + offs_k
        mask_k = offs_k_curr < S

        k = tl.load(k_ptrs, mask=mask_k[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_k[:, None], other=0.0)

        # q is row-major, k.T is natively transposed (col-major). Matches perfect WGMMA layout parameters!
        s_qk = tl.dot(q, tl.trans(k)) * d_scale

        # Unconditional pipeline mask. Prevents stalling Triton's loop software pipeliner with conditional block divergence.
        mask = (offs_q[:, None] >= offs_k_curr[None, :]) & mask_q[:, None] & mask_k[None, :]
        s_qk = tl.where(mask, s_qk, float('-inf'))
        p = tl.exp(s_qk - l[:, None])
        p = tl.where(mask, p, 0.0)

        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - d_val[:, None]) * d_scale

        dq_acc += tl.dot(ds.to(q.dtype), k)

        k_ptrs += BLOCK_K * stride_k_s
        v_ptrs += BLOCK_K * stride_v_s

    offset_dq = pid_b * stride_dq_b + pid_h * stride_dq_h
    dq_ptrs = dQ + offset_dq + offs_q[:, None] * stride_dq_s + offs_d[None, :] * stride_dq_d
    tl.store(dq_ptrs, dq_acc.to(dQ.dtype.element_ty), mask=mask_q[:, None])


@triton.jit
def bwd_kernel_dk_dv(
    Q, K, V, O, dO, L, dK, dV,
    stride_q_b, stride_q_h, stride_q_s, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_o_b, stride_o_h, stride_o_s, stride_o_d,
    stride_do_b, stride_do_h, stride_do_s, stride_do_d,
    stride_l_b, stride_l_h, stride_l_s,
    stride_dk_b, stride_dk_h, stride_dk_s, stride_dk_d,
    stride_dv_b, stride_dv_h, stride_dv_s, stride_dv_d,
    B, H, S, d_scale,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_k = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    start_k = pid_k * BLOCK_K
    offs_k = start_k + tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)

    offset_q = pid_b * stride_q_b + pid_h * stride_q_h
    offset_k = pid_b * stride_k_b + pid_h * stride_k_h
    offset_v = pid_b * stride_v_b + pid_h * stride_v_h
    offset_o = pid_b * stride_o_b + pid_h * stride_o_h
    offset_do = pid_b * stride_do_b + pid_h * stride_do_h
    offset_l = pid_b * stride_l_b + pid_h * stride_l_h

    k_ptrs = K + offset_k + offs_k[:, None] * stride_k_s + offs_d[None, :] * stride_k_d
    v_ptrs = V + offset_v + offs_k[:, None] * stride_v_s + offs_d[None, :] * stride_v_d
    
    mask_k = offs_k < S
    
    k = tl.load(k_ptrs, mask=mask_k[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_k[:, None], other=0.0)

    dk_acc = tl.zeros([BLOCK_K, BLOCK_D], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_K, BLOCK_D], dtype=tl.float32)

    # Establish logical start of the Q iteration constraint bounds minimizing iteration depth
    start_q_first = (start_k // BLOCK_Q) * BLOCK_Q
    num_q_steps = tl.cdiv(S - start_q_first, BLOCK_Q)

    offs_q = tl.arange(0, BLOCK_Q)
    q_ptrs = Q + offset_q + (start_q_first + offs_q)[:, None] * stride_q_s + offs_d[None, :] * stride_q_d
    o_ptrs = O + offset_o + (start_q_first + offs_q)[:, None] * stride_o_s + offs_d[None, :] * stride_o_d
    do_ptrs = dO + offset_do + (start_q_first + offs_q)[:, None] * stride_do_s + offs_d[None, :] * stride_do_d
    l_ptrs = L + offset_l + (start_q_first + offs_q) * stride_l_s

    for i in range(num_q_steps):
        start_q = start_q_first + i * BLOCK_Q
        offs_q_curr = start_q + offs_q
        mask_q = offs_q_curr < S

        q = tl.load(q_ptrs, mask=mask_q[:, None], other=0.0)
        o = tl.load(o_ptrs, mask=mask_q[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_q[:, None], other=0.0)
        l = tl.load(l_ptrs, mask=mask_q, other=0.0)

        d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)

        # Native Optimization: Evaluated inversely by `k @ q.T` ensuring mathematically rigorous transposed formats implicitly!
        s_kq = tl.dot(k, tl.trans(q)) * d_scale

        mask = (offs_k[:, None] <= offs_q_curr[None, :]) & mask_k[:, None] & mask_q[None, :]
        s_kq = tl.where(mask, s_kq, float('-inf'))
        p_kq = tl.exp(s_kq - l[None, :])
        p_kq = tl.where(mask, p_kq, 0.0)

        dv_acc += tl.dot(p_kq.to(do.dtype), do)
        
        # Iteration format inherently sidesteps register transposition layout faults
        dp_kq = tl.dot(v, tl.trans(do))
        ds_kq = p_kq * (dp_kq - d_val[None, :]) * d_scale
        
        dk_acc += tl.dot(ds_kq.to(q.dtype), q)

        q_ptrs += BLOCK_Q * stride_q_s
        o_ptrs += BLOCK_Q * stride_o_s
        do_ptrs += BLOCK_Q * stride_do_s
        l_ptrs += BLOCK_Q * stride_l_s

    offset_dk = pid_b * stride_dk_b + pid_h * stride_dk_h
    offset_dv = pid_b * stride_dv_b + pid_h * stride_dv_h
    dk_ptrs = dK + offset_dk + offs_k[:, None] * stride_dk_s + offs_d[None, :] * stride_dk_d
    dv_ptrs = dV + offset_dv + offs_k[:, None] * stride_dv_s + offs_d[None, :] * stride_dv_d
    
    tl.store(dk_ptrs, dk_acc.to(dK.dtype.element_ty), mask=mask_k[:, None])
    tl.store(dv_ptrs, dv_acc.to(dV.dtype.element_ty), mask=mask_k[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes standard causal multi-head attention backward pass directly translating gradients to target buffers dynamically.
    No mutation overhead utilizing highly efficient WGMMA-optimal WGMMA Tensor Core intrinsics cleanly across H100 hardware paths.
    """
    with torch.cuda.device(Q.device):
        B, H, S, d = Q.shape
        d_scale = 1.0 / (d ** 0.5)
        
        BLOCK_D = 128
        
        # Dimensioning optimally caps Shared Memory footprint constraints while securing wide instruction parallelism latency-hiding overhead 
        BLOCK_Q_DQ = 128
        BLOCK_K_DQ = 64
        
        grid_dq = (triton.cdiv(S, BLOCK_Q_DQ), B * H)
        bwd_kernel_dq[grid_dq](
            Q, K, V, O, dO, L, dQ,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
            B, H, S, d_scale,
            BLOCK_Q_DQ, BLOCK_K_DQ, BLOCK_D,
            num_warps=8, num_stages=3
        )
        
        BLOCK_Q_DK = 64
        BLOCK_K_DK = 128
        
        grid_dk_dv = (triton.cdiv(S, BLOCK_K_DK), B * H)
        bwd_kernel_dk_dv[grid_dk_dv](
            Q, K, V, O, dO, L, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
            dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
            B, H, S, d_scale,
            BLOCK_Q_DK, BLOCK_K_DK, BLOCK_D,
            num_warps=8, num_stages=3
        )
        
        return dQ, dK, dV