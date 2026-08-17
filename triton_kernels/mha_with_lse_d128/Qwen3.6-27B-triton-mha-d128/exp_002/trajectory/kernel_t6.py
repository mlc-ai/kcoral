import torch
import triton
import triton.language as tl


@triton.jit
def _mha_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    B, H, S, D,
    SCALE,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    n_bh = B * H
    n_m = tl.cdiv(S, BLOCK_M)

    # Load Q once per query tile for this (b,h)
    off_d = tl.arange(0, D)
    
    for idx in range(pid, n_bh * n_m, tl.num_programs(0)):
        bh = idx // n_m
        pm = idx % n_m
        b = bh // H
        h = bh % H

        off_m = pm * BLOCK_M + tl.arange(0, BLOCK_M)

        q_offs = b * stride_qb + h * stride_qh
        q_mask = off_m[:, None] < S
        
        Q_ptrs = Q + q_offs + off_m[:, None] * stride_qs + off_d[None, :] * stride_qd
        Q_tile = tl.load(Q_ptrs, mask=q_mask, other=0.0, eviction_policy="evict_last")

        acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
        m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        l_i = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)

        k_base = K + b * stride_kb + h * stride_kh
        v_base = V + b * stride_vb + h * stride_vh

        for sn in range(tl.cdiv(S, BLOCK_N)):
            off_n = sn * BLOCK_N + tl.arange(0, BLOCK_N)
            
            k_ptrs = k_base + off_n[:, None] * stride_ks + off_d[None, :] * stride_kd
            km = (off_n[:, None] < S) & (off_d[None, :] < D)
            K_tile = tl.load(k_ptrs, mask=km, other=0.0)

            s = tl.dot(Q_tile, K_tile.T) * SCALE
            
            sm = (off_m[:, None] < S) & (off_n[None, :] < S)
            s = tl.where(sm, s, float("-inf"))

            mn = tl.maximum(m_i, tl.max(s, axis=1))
            a = tl.exp(m_i - mn)
            p = tl.exp(s - mn[:, None])
            
            acc *= a[:, None]

            v_ptrs = v_base + off_n[:, None] * stride_vs + off_d[None, :] * stride_vd
            vm = (off_n[:, None] < S) & (off_d[None, :] < D)
            V_tile = tl.load(v_ptrs, mask=vm, other=0.0)

            acc += tl.dot(p, V_tile.to(tl.float32))
            l_i = l_i * a + tl.sum(p, axis=1)
            m_i = mn

        acc /= l_i[:, None]
        
        o_base = O + b * stride_ob + h * stride_oh
        o_ptrs = o_base + off_m[:, None] * stride_os + off_d[None, :] * stride_od
        tl.store(o_ptrs, acc.to(tl.bfloat16), mask=q_mask)

        lse_base = LSE + b * stride_lb + h * stride_lh
        tl.store(lse_base + off_m * stride_ls, m_i + tl.log(l_i), mask=(off_m < S))


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    scale = 1.0 / float(D ** 0.5)
    
    sm_count = torch.cuda.get_device_properties(Q.device).multi_processor_count
    
    BM = 64
    BN = 32
    
    # Cap grid at SM count for persistent scheduling
    np = min(sm_count, B * H * triton.cdiv(S, BM))
    
    _mha_kernel[(np,)](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S, D, scale,
        BLOCK_M=BM, BLOCK_N=BN,
        num_warps=4, num_stages=3,
    )