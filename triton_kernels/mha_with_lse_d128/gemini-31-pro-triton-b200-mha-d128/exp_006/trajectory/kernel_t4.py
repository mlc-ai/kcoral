import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

def check_tma_supported(q, k, v, o):
    """
    TMA requirements: base is 16-byte aligned; last stride is 1;
    leading strides must be 16-byte aligned in bytes.
    """
    def is_aligned(tensor):
        if tensor.data_ptr() % 16 != 0: return False
        if tensor.stride(-1) != 1: return False
        # Leading strides must be 16-byte aligned in bytes
        for s in tensor.stride()[:-1]:
            if (s * tensor.element_size()) % 16 != 0:
                return False
        return True
    return is_aligned(q) and is_aligned(k) and is_aligned(v) and is_aligned(o)


def _pre_hook(kwargs):
    """
    Autotune pre-hook: Updates Host TensorDescriptors to match the dynamically
    chosen BLOCK_M and BLOCK_N dimensions before standard Triton launches them.
    """
    bm = kwargs["BLOCK_M"]
    bn = kwargs["BLOCK_N"]
    bd = kwargs["BLOCK_D"]
    kwargs["q_desc"] = TensorDescriptor.from_tensor(kwargs["Q"], [1, 1, bm, bd])
    kwargs["k_desc"] = TensorDescriptor.from_tensor(kwargs["K"], [1, 1, bn, bd])
    kwargs["v_desc"] = TensorDescriptor.from_tensor(kwargs["V"], [1, 1, bn, bd])
    kwargs["o_desc"] = TensorDescriptor.from_tensor(kwargs["O"], [1, 1, bm, bd])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "NUM_STAGES": 3}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "NUM_STAGES": 3}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "NUM_STAGES": 3}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "NUM_STAGES": 3}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "NUM_STAGES": 3}, num_warps=4, num_stages=3),
    ],
    key=["S"],
    pre_hook=_pre_hook
)
@triton.jit
def _attention_kernel_host_tma(
    Q, K, V, O,  # Dummy variables to allow pre_hook to reference original tensors
    q_desc, k_desc, v_desc, o_desc,
    LSE,
    stride_lb, stride_lh, stride_ls,
    S,
    softmax_scale_log2,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    offset_m = pid_m * BLOCK_M

    # TMA load returns a 4D block matching descriptor dimensions; Reshape logically to 2D
    q_4d = q_desc.load([pid_b, pid_h, offset_m, 0])
    q = tl.reshape(q_4d, [BLOCK_M, BLOCK_D])
    
    # Pre-scale in base-2 directly inside registers
    q_scaled = (q * softmax_scale_log2).to(q.dtype)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

    num_full_blocks = S // BLOCK_N

    # Explicit software pipelining overrides configured with num_stages
    for k_idx in tl.range(0, num_full_blocks, num_stages=NUM_STAGES):
        offset_n = k_idx * BLOCK_N
        
        k_4d = k_desc.load([pid_b, pid_h, offset_n, 0])
        v_4d = v_desc.load([pid_b, pid_h, offset_n, 0])
        
        k = tl.reshape(k_4d, [BLOCK_N, BLOCK_D])
        v = tl.reshape(v_4d, [BLOCK_N, BLOCK_D])
        
        scores = tl.dot(q_scaled, k.T)
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(q.dtype), v, acc)
        m_i = m_ij

    if S % BLOCK_N != 0:
        offset_n = num_full_blocks * BLOCK_N
        
        k_4d = k_desc.load([pid_b, pid_h, offset_n, 0])
        v_4d = v_desc.load([pid_b, pid_h, offset_n, 0])
        
        k = tl.reshape(k_4d, [BLOCK_N, BLOCK_D])
        v = tl.reshape(v_4d, [BLOCK_N, BLOCK_D])
        
        scores = tl.dot(q_scaled, k.T)
        
        offs_n_tail = offset_n + tl.arange(0, BLOCK_N)
        scores = tl.where(offs_n_tail[None, :] < S, scores, float("-inf"))
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(q.dtype), v, acc)
        m_i = m_ij

    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    inv_l_i = 1.0 / safe_l_i
    out = acc * inv_l_i[:, None]
    
    # Store with TMA hardware zero-pad boundaries gracefully ignoring OOB write boundaries 
    out_4d = tl.reshape(out.to(q.dtype), [1, 1, BLOCK_M, BLOCK_D])
    o_desc.store([pid_b, pid_h, offset_m, 0], out_4d)
    
    # Output Log-Sum-Exp mapping conventions back to Base-e Natural Log
    LN2: tl.constexpr = 0.6931471805599453
    lse = (m_i + tl.math.log2(safe_l_i)) * LN2

    lse_base_ptr = LSE + pid_b * stride_lb + pid_h * stride_lh
    offs_m = offset_m + tl.arange(0, BLOCK_M)
    lse_ptrs = lse_base_ptr + offs_m * stride_ls

    is_full_m = (pid_m + 1) * BLOCK_M <= S
    if is_full_m:
        tl.store(lse_ptrs, lse)
    else:
        tl.store(lse_ptrs, lse, mask=(offs_m < S))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_warps=4, num_stages=3),
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
    S, softmax_scale_log2,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
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
    
    lse_base_ptr = LSE + pid_b * stride_lb + pid_h * stride_lh

    q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    
    is_full_m = (pid_m + 1) * BLOCK_M <= S
    if is_full_m:
        q = tl.load(q_ptrs)
    else:
        mask_q = (offs_m[:, None] < S)
        q = tl.load(q_ptrs, mask=mask_q, other=0.0)

    q_scaled = (q * softmax_scale_log2).to(q.dtype)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

    offs_n = tl.arange(0, BLOCK_N)
    k_ptrs = k_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd

    num_full_blocks = S // BLOCK_N

    for k_idx in range(0, num_full_blocks):
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        
        scores = tl.dot(q_scaled, k.T)
        
        if not is_full_m:
            scores = tl.where(offs_m[:, None] < S, scores, float("-inf"))
            m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
            safe_m_ij = tl.where(m_ij == float("-inf"), 0.0, m_ij)
            alpha = tl.math.exp2(m_i - safe_m_ij)
            p = tl.math.exp2(scores - safe_m_ij[:, None])
        else:
            m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
            alpha = tl.math.exp2(m_i - m_ij)
            p = tl.math.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(q.dtype), v, acc)
        m_i = m_ij
        
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs

    has_tail = (S % BLOCK_N != 0)
    if has_tail:
        offset_n = num_full_blocks * BLOCK_N
        offs_n_tail = offset_n + tl.arange(0, BLOCK_N)
        
        mask_k = (offs_n_tail[:, None] < S)
        k = tl.load(k_ptrs, mask=mask_k, other=0.0)
        v = tl.load(v_ptrs, mask=mask_k, other=0.0)
        
        scores = tl.dot(q_scaled, k.T)
        if is_full_m:
            scores = tl.where(offs_n_tail[None, :] < S, scores, float("-inf"))
        else:
            valid_score = (offs_m[:, None] < S) & (offs_n_tail[None, :] < S)
            scores = tl.where(valid_score, scores, float("-inf"))
            
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        safe_m_ij = tl.where(m_ij == float("-inf"), 0.0, m_ij)
        
        alpha = tl.math.exp2(m_i - safe_m_ij)
        p = tl.math.exp2(scores - safe_m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(q.dtype), v, acc)
        m_i = m_ij

    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    inv_l_i = 1.0 / safe_l_i
    out = acc * inv_l_i[:, None]
    
    LN2: tl.constexpr = 0.6931471805599453
    lse = (m_i + tl.math.log2(safe_l_i)) * LN2
    lse = tl.where(l_i == 0.0, float("-inf"), lse)

    o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    lse_ptrs = lse_base_ptr + offs_m * stride_ls

    if is_full_m:
        tl.store(o_ptrs, out.to(q.dtype))
        tl.store(lse_ptrs, lse)
    else:
        mask_q = (offs_m[:, None] < S)
        tl.store(o_ptrs, out.to(q.dtype), mask=mask_q)
        tl.store(lse_ptrs, lse, mask=(offs_m < S))


def run(Q, K, V, O, LSE):
    """
    Computes a Multi-Head Attention forward pass returning output vectors and Log-Sum-Exp (LSE).
    Targeting NVIDIA Blackwell architecture featuring 4D Descriptors and MUFU instruction math.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    softmax_scale = 1.0 / (D ** 0.5)
    RCP_LN2 = 1.4426950408889634
    softmax_scale_log2 = softmax_scale * RCP_LN2

    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B, H)
    
    if check_tma_supported(Q, K, V, O):
        # We start by constructing generic shape descriptors, dynamically overwritten locally on launch pre_hook configs.
        q_desc = TensorDescriptor.from_tensor(Q, [1, 1, 128, 128])
        k_desc = TensorDescriptor.from_tensor(K, [1, 1, 128, 128])
        v_desc = TensorDescriptor.from_tensor(V, [1, 1, 128, 128])
        o_desc = TensorDescriptor.from_tensor(O, [1, 1, 128, 128])

        _attention_kernel_host_tma[grid](
            Q, K, V, O,
            q_desc, k_desc, v_desc, o_desc,
            LSE,
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
            S, softmax_scale_log2,
            BLOCK_D=128
        )
    else:
        _attention_kernel_ptr[grid](
            Q, K, V, O, LSE,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
            S, softmax_scale_log2,
            BLOCK_D=128
        )