import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    B, H, S,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    # Load Q
    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    
    q_valid_1d = offs_m < S
    q_valid_2d = q_valid_1d[:, None]
    q = tl.load(q_ptrs, mask=q_valid_2d, other=0.0)

    # Initialize online softmax state
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)

    RCP_LN2: tl.constexpr = 1.4426950408889634
    LN2: tl.constexpr = 0.6931471805599453

    # Causal limits keys to the max sequence index of the queries in this block
    max_kv_len = tl.minimum(S, (pid_m + 1) * BLOCK_M)
    num_kv_tiles = tl.cdiv(max_kv_len, BLOCK_N)

    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    
    offs_n_init = tl.arange(0, BLOCK_N)
    k_ptrs = k_base + offs_n_init[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = v_base + offs_n_init[:, None] * stride_vs + offs_d[None, :] * stride_vd

    for kv_tile in range(0, num_kv_tiles):
        offs_n = kv_tile * BLOCK_N + offs_n_init
        n_valid_2d = (offs_n < S)[:, None]
        
        k = tl.load(k_ptrs, mask=n_valid_2d, other=0.0)
        v = tl.load(v_ptrs, mask=n_valid_2d, other=0.0)

        # Q @ K.T (Output is FP32)
        scores = tl.dot(q, k.T, tl.zeros((BLOCK_M, BLOCK_N), tl.float32)) * sm_scale

        # Causal and padding masks
        n_valid_row = (offs_n < S)[None, :]
        valid_mask = (offs_m[:, None] >= offs_n[None, :]) & q_valid_2d & n_valid_row

        scores = tl.where(valid_mask, scores * RCP_LN2, -float("inf"))

        # Online softmax recurrence
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        safe_m_ij = tl.where(m_ij == -float("inf"), 0.0, m_ij)
        
        alpha = tl.math.exp2(m_i - safe_m_ij)
        p = tl.math.exp2(scores - safe_m_ij[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        
        # P @ V
        acc = tl.dot(p.to(tl.bfloat16), v, acc)

        m_i = m_ij

        # Advance KV pointers
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs

    # Normalize output
    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / safe_l_i[:, None]

    # LogSumExp (natural log) computation matching PyTorch backend
    lse_log2 = tl.where(l_i == 0.0, -float("inf"), (m_i + tl.math.log2(safe_l_i)) * LN2)

    # Store Attention output
    o_base = O + pid_b * stride_ob + pid_h * stride_oh
    o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, out.to(tl.bfloat16), mask=q_valid_2d)

    # Store LogSumExp
    lse_base = LSE + pid_b * stride_lb + pid_h * stride_lh
    lse_ptrs = lse_base + offs_m * stride_ls
    tl.store(lse_ptrs, lse_log2, mask=q_valid_1d)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    sm_scale = 1.0 / (D ** 0.5)

    BLOCK_M = 128
    BLOCK_N = 64
    grid = (triton.cdiv(S, BLOCK_M), B * H)

    _fwd_kernel[grid](
        Q, K, V, sm_scale,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S,
        D=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=3
    )