import torch
import triton
import triton.language as tl

def alloc_fn(size: int, alignment: int, stream):
    # Device descriptor creation needs Triton's descriptor allocator configured on the host.
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def check_tma_supported(q, k, v, o):
    # TMA requirements: base is 16-byte aligned; last stride is 1; leading strides 16-byte aligned in bytes.
    def is_aligned(tensor):
        return (tensor.data_ptr() % 16 == 0) and \
               ((tensor.stride(-2) * tensor.element_size()) % 16 == 0) and \
               (tensor.stride(-1) == 1)
    return is_aligned(q) and is_aligned(k) and is_aligned(v) and is_aligned(o)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def _attention_kernel_tma(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_lb, stride_lh, stride_ls,
    S,
    softmax_scale_log2,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    S_IS_MULTIPLE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    o_base = O + pid_b * stride_ob + pid_h * stride_oh
    lse_base = LSE + pid_b * stride_lb + pid_h * stride_lh

    q_desc = tl.make_tensor_descriptor(
        q_base,
        shape=[S, BLOCK_D],
        strides=[stride_qs, 1],
        block_shape=[BLOCK_M, BLOCK_D],
        padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_base,
        shape=[S, BLOCK_D],
        strides=[stride_ks, 1],
        block_shape=[BLOCK_N, BLOCK_D],
        padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_base,
        shape=[S, BLOCK_D],
        strides=[stride_vs, 1],
        block_shape=[BLOCK_N, BLOCK_D],
        padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        o_base,
        shape=[S, BLOCK_D],
        strides=[stride_os, 1],
        block_shape=[BLOCK_M, BLOCK_D],
        padding_option="zero"
    )

    offset_m = pid_m * BLOCK_M
    q = q_desc.load([offset_m, 0])

    # Pre-scale Q in base-2 units outside the loop
    q_scaled = (q * softmax_scale_log2).to(q.dtype)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

    offs_m = offset_m + tl.arange(0, BLOCK_M)

    for k_idx in range(0, tl.cdiv(S, BLOCK_N)):
        offset_n = k_idx * BLOCK_N
        
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])
        
        scores = tl.dot(q_scaled, k.T)
        
        # Omit boundary and tail mask work on fully valid blocks entirely
        if not S_IS_MULTIPLE:
            offs_n = offset_n + tl.arange(0, BLOCK_N)
            valid_score = (offs_m[:, None] < S) & (offs_n[None, :] < S)
            scores = tl.where(valid_score, scores, -float("inf"))
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        safe_m_ij = tl.where(m_ij == -float("inf"), 0.0, m_ij)
        
        alpha = tl.math.exp2(m_i - safe_m_ij)
        p = tl.math.exp2(scores - safe_m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(q.dtype), v.to(q.dtype), acc)
        m_i = m_ij

    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / safe_l_i[:, None]
    
    o_desc.store([offset_m, 0], out.to(q.dtype))

    # Output log-sum-exp uses base-e convention
    LN2: tl.constexpr = 0.6931471805599453
    lse = (m_i + tl.math.log2(safe_l_i)) * LN2
    lse = tl.where(l_i == 0.0, -float("inf"), lse)

    lse_ptrs = lse_base + offs_m * stride_ls
    if S_IS_MULTIPLE:
        tl.store(lse_ptrs, lse)
    else:
        tl.store(lse_ptrs, lse, mask=(offs_m < S))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def _attention_kernel_ptr(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    S,
    softmax_scale_log2,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    S_IS_MULTIPLE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    o_base = O + pid_b * stride_ob + pid_h * stride_oh
    lse_base = LSE + pid_b * stride_lb + pid_h * stride_lh

    q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    if S_IS_MULTIPLE:
        q = tl.load(q_ptrs)
    else:
        mask_q = (offs_m[:, None] < S) & (offs_d[None, :] < BLOCK_D)
        q = tl.load(q_ptrs, mask=mask_q, other=0.0)

    q_scaled = (q * softmax_scale_log2).to(q.dtype)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

    for k_idx in range(0, tl.cdiv(S, BLOCK_N)):
        offs_n = k_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        
        k_ptrs = k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        
        if S_IS_MULTIPLE:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            mask_k = (offs_n[:, None] < S) & (offs_d[None, :] < BLOCK_D)
            k = tl.load(k_ptrs, mask=mask_k, other=0.0)
            v = tl.load(v_ptrs, mask=mask_k, other=0.0)
        
        scores = tl.dot(q_scaled, k.T)
        
        if not S_IS_MULTIPLE:
            valid_score = (offs_m[:, None] < S) & (offs_n[None, :] < S)
            scores = tl.where(valid_score, scores, -float("inf"))
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        safe_m_ij = tl.where(m_ij == -float("inf"), 0.0, m_ij)
        
        alpha = tl.math.exp2(m_i - safe_m_ij)
        p = tl.math.exp2(scores - safe_m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(q.dtype), v.to(q.dtype), acc)
        m_i = m_ij

    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / safe_l_i[:, None]
    
    LN2: tl.constexpr = 0.6931471805599453
    lse = (m_i + tl.math.log2(safe_l_i)) * LN2
    lse = tl.where(l_i == 0.0, -float("inf"), lse)

    o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    lse_ptrs = lse_base + offs_m * stride_ls

    if S_IS_MULTIPLE:
        tl.store(o_ptrs, out.to(q.dtype))
        tl.store(lse_ptrs, lse)
    else:
        tl.store(o_ptrs, out.to(q.dtype), mask=mask_q)
        tl.store(lse_ptrs, lse, mask=(offs_m < S))


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    softmax_scale = 1.0 / (D ** 0.5)
    RCP_LN2 = 1.4426950408889634
    softmax_scale_log2 = softmax_scale * RCP_LN2

    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B, H)
    
    # 128 is the maximum BLOCK size configured in autotuning
    s_is_multiple = bool(S % 128 == 0)

    if check_tma_supported(Q, K, V, O):
        _attention_kernel_tma[grid](
            Q, K, V, O, LSE,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
            S,
            softmax_scale_log2,
            BLOCK_D=128,
            S_IS_MULTIPLE=s_is_multiple,
        )
    else:
        _attention_kernel_ptr[grid](
            Q, K, V, O, LSE,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
            S,
            softmax_scale_log2,
            BLOCK_D=128,
            S_IS_MULTIPLE=s_is_multiple,
        )