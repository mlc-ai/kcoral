import torch
import triton
import triton.language as tl

# Configure Triton's descriptor allocator on the host for device-created descriptors.
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        triton.Config({
            'BM_DQ': 128, 'BN_DQ': 128,
            'BM_DK': 128, 'BN_DK': 128,
            'STAGES': 2
        }, num_warps=8, num_stages=2),
        triton.Config({
            'BM_DQ': 128, 'BN_DQ': 64,
            'BM_DK': 64, 'BN_DK': 128,
            'STAGES': 3
        }, num_warps=4, num_stages=3),
        triton.Config({
            'BM_DQ': 64, 'BN_DQ': 128,
            'BM_DK': 128, 'BN_DK': 64,
            'STAGES': 3
        }, num_warps=4, num_stages=3),
        triton.Config({
            'BM_DQ': 64, 'BN_DQ': 64,
            'BM_DK': 64, 'BN_DK': 64,
            'STAGES': 4
        }, num_warps=4, num_stages=4),
        triton.Config({
            'BM_DQ': 128, 'BN_DQ': 64,
            'BM_DK': 128, 'BN_DK': 64,
            'STAGES': 3
        }, num_warps=8, num_stages=3),
        triton.Config({
            'BM_DQ': 128, 'BN_DQ': 128,
            'BM_DK': 128, 'BN_DK': 128,
            'STAGES': 2
        }, num_warps=4, num_stages=2),
    ],
    key=['S']
)
@triton.jit
def bwd_kernel(
    Q, K, V, O, dO, L,
    dQ, dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    B, H, S, scale,
    d: tl.constexpr,
    BM_DQ: tl.constexpr, BN_DQ: tl.constexpr,
    BM_DK: tl.constexpr, BN_DK: tl.constexpr,
    STAGES: tl.constexpr
):
    pid = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    num_kv_tiles = (S + BN_DK - 1) // BN_DK

    # Conversion factor from natural log LSE to base-2 LSE
    LOG2_E = 1.4426950408889634

    if pid < num_kv_tiles:
        # =====================================================================
        # REGION 2: Exclusive owner for a dK and dV tile
        # =====================================================================
        pid_n = pid
        BLOCK_N = BN_DK
        BLOCK_M = BM_DK
        
        # Define descriptors for resident KV and output tiles
        k_base = K + b * stride_kb + h * stride_kh
        v_base = V + b * stride_vb + h * stride_vh
        dk_base = dK + b * stride_dkb + h * stride_dkh
        dv_base = dV + b * stride_dvb + h * stride_dvh

        k_desc = tl.make_tensor_descriptor(
            k_base, shape=[S, d], strides=[stride_ks, 1],
            block_shape=[BLOCK_N, d], padding_option="zero"
        )
        v_desc = tl.make_tensor_descriptor(
            v_base, shape=[S, d], strides=[stride_vs, 1],
            block_shape=[BLOCK_N, d], padding_option="zero"
        )
        dk_desc = tl.make_tensor_descriptor(
            dk_base, shape=[S, d], strides=[stride_dks, 1],
            block_shape=[BLOCK_N, d], padding_option="zero"
        )
        dv_desc = tl.make_tensor_descriptor(
            dv_base, shape=[S, d], strides=[stride_dvs, 1],
            block_shape=[BLOCK_N, d], padding_option="zero"
        )

        # Load resident KV tiles via TMA
        k = k_desc.load([pid_n * BLOCK_N, 0])
        v = v_desc.load([pid_n * BLOCK_N, 0])

        dk_acc = tl.zeros((BLOCK_N, d), dtype=tl.float32)
        dv_acc = tl.zeros((BLOCK_N, d), dtype=tl.float32)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S

        # Define descriptors for Q stream tiles
        q_base = Q + b * stride_qb + h * stride_qh
        o_base = O + b * stride_ob + h * stride_oh
        do_base = dO + b * stride_dob + h * stride_doh

        q_desc = tl.make_tensor_descriptor(
            q_base, shape=[S, d], strides=[stride_qs, 1],
            block_shape=[BLOCK_M, d], padding_option="zero"
        )
        o_desc = tl.make_tensor_descriptor(
            o_base, shape=[S, d], strides=[stride_os, 1],
            block_shape=[BLOCK_M, d], padding_option="zero"
        )
        do_desc = tl.make_tensor_descriptor(
            do_base, shape=[S, d], strides=[stride_dos, 1],
            block_shape=[BLOCK_M, d], padding_option="zero"
        )

        loop_q_tiles = (S + BLOCK_M - 1) // BLOCK_M
        for q_tile in tl.range(0, loop_q_tiles, num_stages=STAGES):
            q = q_desc.load([q_tile * BLOCK_M, 0])
            o = o_desc.load([q_tile * BLOCK_M, 0])
            do = do_desc.load([q_tile * BLOCK_M, 0])

            offs_m = q_tile * BLOCK_M + tl.arange(0, BLOCK_M)
            l_ptrs = L + b * stride_lb + h * stride_lh + offs_m * stride_ls
            l = tl.load(l_ptrs, mask=offs_m < S, other=0.0)

            # Precompute rowwise delta for this Q block on the fly
            delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)

            scores = tl.dot(k, q.T) * scale

            valid = mask_n[:, None] & (offs_m[None, :] < S)
            scores = tl.where(valid, scores, float("-inf"))

            p = tl.math.exp2((scores - l[None, :]) * LOG2_E)
            p = tl.where(valid, p, 0.0)

            dv_acc += tl.dot(p.to(tl.bfloat16), do)

            dp_t = tl.dot(v, do.T)

            ds_t = p * (dp_t - delta[None, :]) * scale
            ds_t = tl.where(valid, ds_t, 0.0)

            dk_acc += tl.dot(ds_t.to(tl.bfloat16), q)

        # Store complete dK and dV tiles via TMA
        dk_desc.store([pid_n * BLOCK_N, 0], dk_acc.to(tl.bfloat16))
        dv_desc.store([pid_n * BLOCK_N, 0], dv_acc.to(tl.bfloat16))

    else:
        # =====================================================================
        # REGION 1: Exclusive owner for a dQ tile
        # =====================================================================
        pid_m = pid - num_kv_tiles
        BLOCK_M = BM_DQ
        BLOCK_N = BN_DQ

        # Define descriptors for resident Q and output tiles
        q_base = Q + b * stride_qb + h * stride_qh
        o_base = O + b * stride_ob + h * stride_oh
        do_base = dO + b * stride_dob + h * stride_doh
        dq_base = dQ + b * stride_dqb + h * stride_dqh
        
        q_desc = tl.make_tensor_descriptor(
            q_base, shape=[S, d], strides=[stride_qs, 1],
            block_shape=[BLOCK_M, d], padding_option="zero"
        )
        o_desc = tl.make_tensor_descriptor(
            o_base, shape=[S, d], strides=[stride_os, 1],
            block_shape=[BLOCK_M, d], padding_option="zero"
        )
        do_desc = tl.make_tensor_descriptor(
            do_base, shape=[S, d], strides=[stride_dos, 1],
            block_shape=[BLOCK_M, d], padding_option="zero"
        )
        dq_desc = tl.make_tensor_descriptor(
            dq_base, shape=[S, d], strides=[stride_dqs, 1],
            block_shape=[BLOCK_M, d], padding_option="zero"
        )

        # Load resident Q tiles via TMA
        q = q_desc.load([pid_m * BLOCK_M, 0])
        o = o_desc.load([pid_m * BLOCK_M, 0])
        do = do_desc.load([pid_m * BLOCK_M, 0])

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        l_ptrs = L + b * stride_lb + h * stride_lh + offs_m * stride_ls
        mask_m = offs_m < S
        l = tl.load(l_ptrs, mask=mask_m, other=0.0)

        # Precompute rowwise delta for this resident Q block
        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)

        dq_acc = tl.zeros((BLOCK_M, d), dtype=tl.float32)

        # Define descriptors for KV stream tiles
        k_base = K + b * stride_kb + h * stride_kh
        v_base = V + b * stride_vb + h * stride_vh
        k_desc = tl.make_tensor_descriptor(
            k_base, shape=[S, d], strides=[stride_ks, 1],
            block_shape=[BLOCK_N, d], padding_option="zero"
        )
        v_desc = tl.make_tensor_descriptor(
            v_base, shape=[S, d], strides=[stride_vs, 1],
            block_shape=[BLOCK_N, d], padding_option="zero"
        )

        loop_kv_tiles = (S + BLOCK_N - 1) // BLOCK_N
        for kv_tile in tl.range(0, loop_kv_tiles, num_stages=STAGES):
            k = k_desc.load([kv_tile * BLOCK_N, 0])
            v = v_desc.load([kv_tile * BLOCK_N, 0])
            
            scores = tl.dot(q, k.T) * scale
            
            offs_n = kv_tile * BLOCK_N + tl.arange(0, BLOCK_N)
            valid = mask_m[:, None] & (offs_n[None, :] < S)
            scores = tl.where(valid, scores, float("-inf"))
            
            p = tl.math.exp2((scores - l[:, None]) * LOG2_E)
            p = tl.where(valid, p, 0.0)
            
            dp = tl.dot(do, v.T)
            
            ds = p * (dp - delta[:, None]) * scale
            ds = tl.where(valid, ds, 0.0)
            
            dq_acc += tl.dot(ds.to(tl.bfloat16), k)

        # Store complete dQ tile via TMA
        dq_desc.store([pid_m * BLOCK_M, 0], dq_acc.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes exact backward gradients for multi-head attention without causal masking.
    
    Operates on bf16 tensors with shape [B, H, S, d]. Output dQ, dK, and dV are fully stored
    using a unified flat-grid layout with split output ownership to maximize parallel compute,
    leverage efficient scheduling, and eliminate atomic operations.
    """
    with torch.cuda.device(Q.device):
        B_val, H_val, S_val, d_val = Q.shape
        scale = 1.0 / (d_val ** 0.5)
        stride_ls = L.stride(2) if L.dim() >= 3 else L.stride(-1)

        # Launch unified flat-grid kernel spanning both Q owners and KV owners
        def grid(META):
            num_kv = triton.cdiv(S_val, META['BN_DK'])
            num_q = triton.cdiv(S_val, META['BM_DQ'])
            return (num_kv + num_q, B_val * H_val)

        bwd_kernel[grid](
            Q, K, V, O, dO, L,
            dQ, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            L.stride(0), L.stride(1), stride_ls,
            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
            dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
            dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
            B_val, H_val, S_val, scale,
            d=d_val
        )