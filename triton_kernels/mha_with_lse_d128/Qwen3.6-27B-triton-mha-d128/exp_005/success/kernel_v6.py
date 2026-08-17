import torch
import triton
import triton.language as tl


def _alloc_fn(size: int, alignment: int, stream):
    """Allocator for device-created tensor descriptors (infrastructure only)."""
    return torch.empty(size, device="cuda", dtype=torch.int8)


@triton.jit
def _mha_fwd_kernel(
    Q, K, V,
    Out, Lse,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    batch_size, num_heads, seq_len, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_SIZE: tl.constexpr,
):
    """Persistent-grid multi-head attention forward.

    Each CTA owns a contiguous stripe of (batch, head) work-items and
    processes every query-block for those heads sequentially, maximising
    L2 reuse of that head's Q/K/V data.
    """
    pid = tl.program_id(0)
    grid_dim = tl.num_programs(0)

    zh_total = batch_size * num_heads

    # Simple round-robin assignment: each CTA gets ceil(total/grid) items
    items_per_cta = tl.cdiv(zh_total, grid_dim)
    start_zh = pid * items_per_cta
    end_zh = min(start_zh + items_per_cta, zh_total)

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_SIZE)
    num_k_steps = tl.cdiv(seq_len, BLOCK_N)

    for item_id in range(start_zh, end_zh):
        off_b = item_id // num_heads
        off_h = item_id % num_heads

        Q_base = Q + off_b * stride_qb + off_h * stride_qh
        K_base = K + off_b * stride_kb + off_h * stride_kh
        V_base = V + off_b * stride_vb + off_h * stride_vh
        O_base  = Out + off_b * stride_ob + off_h * stride_oh
        LSE_base = Lse + off_b * stride_lb + off_h * stride_lh

        num_q_tiles = tl.cdiv(seq_len, BLOCK_M)

        for tile_m in range(num_q_tiles):
            offs_m_local = tl.arange(0, BLOCK_M)
            offs_m = tile_m * BLOCK_M + offs_m_local
            q_valid = offs_m < seq_len

            Q_tile = tl.load(Q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
                             mask=q_valid[:, None], other=0.0)
            dtype_in = Q_tile.dtype

            m_i = tl.full([BLOCK_M], float('-inf'), tl.float32)
            l_i = tl.full([BLOCK_M], 1.0, tl.float32)
            acc_o = tl.zeros([BLOCK_M, HEAD_SIZE], tl.float32)

            for step in range(num_k_steps):
                n_idx = step * BLOCK_N + offs_n
                n_valid = n_idx < seq_len

                K_tile = tl.load(K_base + n_idx[:, None] * stride_ks + offs_d[None, :] * stride_kd,
                                 mask=n_valid[:, None], other=0.0)
                V_tile = tl.load(V_base + n_idx[:, None] * stride_vs + offs_d[None, :] * stride_vd,
                                 mask=n_valid[:, None], other=0.0)

                scores = tl.dot(Q_tile, K_tile.T) * scale
                scores = tl.where(n_valid[None, :], scores, float('-inf'))

                m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
                p = tl.exp(scores - m_ij[:, None])
                alpha = tl.exp(m_i - m_ij)

                acc_o = acc_o * alpha[:, None] + tl.dot(p.to(dtype_in), V_tile)

                l_i = l_i * alpha + tl.sum(p, axis=1)
                m_i = m_ij

            o_final = acc_o / l_i[:, None]

            tl.store(O_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od,
                     o_final.to(dtype_in), mask=q_valid[:, None])
            tl.store(LSE_base + offs_m * stride_ls,
                     m_i + tl.log(l_i), mask=q_valid)


def run(Q, K, V, O, LSE):
    """Destination-passing entry point."""
    torch.cuda.set_device(Q.device)

    triton.set_allocator(_alloc_fn)

    B, H, S, D = Q.shape
    scale = 1.0 / float(D ** 0.5)

    BLOCK_M = 64
    BLOCK_N = 64
    num_warps = 4
    num_stages = 3

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    num_sms = props.multi_processor_count

    grid = (min(num_sms, B * H),)

    _mha_fwd_kernel[grid](
        Q, K, V,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S, scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_SIZE=D,
        num_warps=num_warps,
        num_stages=num_stages,
    )