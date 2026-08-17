import torch
import triton
import triton.language as tl

def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def _mha_fwd_kernel(
    Q, K, V, sm_scale,
    O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    H, S,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    
    if start_m * BLOCK_M >= S:
        return
        
    off_hz = tl.program_id(1)
    off_b = off_hz // H
    off_h = off_hz % H
    
    Q_ptr = Q + off_b * stride_qb + off_h * stride_qh
    K_ptr = K + off_b * stride_kb + off_h * stride_kh
    V_ptr = V + off_b * stride_vb + off_h * stride_vh
    O_ptr = O + off_b * stride_ob + off_h * stride_oh
    
    q_desc = tl.make_tensor_descriptor(
        Q_ptr, shape=[S, BLOCK_D], strides=[stride_qs, stride_qd], block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        K_ptr, shape=[S, BLOCK_D], strides=[stride_ks, stride_kd], block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V_ptr, shape=[S, BLOCK_D], strides=[stride_vs, stride_vd], block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        O_ptr, shape=[S, BLOCK_D], strides=[stride_os, stride_od], block_shape=[BLOCK_M, BLOCK_D]
    )

    q = q_desc.load([start_m * BLOCK_M, 0])
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    is_valid_m = offs_m < S
    
    m_i = tl.where(is_valid_m, float("-inf"), 0.0)
    l_i = tl.where(is_valid_m, 0.0, 1.0)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    limit_n = tl.minimum(start_m * BLOCK_M, S)
    n_unmasked_blocks = limit_n // BLOCK_N
    max_k_idx = tl.minimum((start_m + 1) * BLOCK_M, S)
    n_blocks = (max_k_idx + BLOCK_N - 1) // BLOCK_N
    
    for start_n_idx in range(0, n_unmasked_blocks):
        start_n = start_n_idx * BLOCK_N
        
        k = k_desc.load([start_n, 0])
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k.T, qk)
        qk = qk * sm_scale
        
        qk = tl.where(is_valid_m[:, None], qk, float("-inf"))
        
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        l_i_new = alpha * l_i + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        
        v = v_desc.load([start_n, 0])
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        
        m_i = m_i_new
        l_i = l_i_new
        
    offs_n = tl.arange(0, BLOCK_N)
    for start_n_idx in range(n_unmasked_blocks, n_blocks):
        start_n = start_n_idx * BLOCK_N
        offs_n_curr = start_n + offs_n
        
        k = k_desc.load([start_n, 0])
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k.T, qk)
        qk = qk * sm_scale
        
        causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
        valid_mask = causal_mask & is_valid_m[:, None] & (offs_n_curr[None, :] < S)
        qk = tl.where(valid_mask, qk, float("-inf"))
        
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        p = tl.where(valid_mask, p, 0.0)
        
        l_i_new = alpha * l_i + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        
        v = v_desc.load([start_n, 0])
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        
        m_i = m_i_new
        l_i = l_i_new

    acc = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)
    
    lse_ptrs = LSE + off_b * stride_lseb + off_h * stride_lseh + offs_m * stride_lses
    tl.store(lse_ptrs, lse, mask=is_valid_m)
    
    o_desc.store([start_m * BLOCK_M, 0], acc.to(tl.bfloat16))


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    if S == 0:
        return
        
    sm_scale = 1.0 / (D ** 0.5)
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H, 1)
    
    _mha_fwd_kernel[grid](
        Q, K, V, sm_scale,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        H, S,
        BLOCK_D=128
    )