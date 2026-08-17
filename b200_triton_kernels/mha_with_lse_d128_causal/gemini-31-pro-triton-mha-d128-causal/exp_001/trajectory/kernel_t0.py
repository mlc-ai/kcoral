import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, O, LSE,
    sm_scale,
    S,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    H,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    batch_head = tl.program_id(1)
    
    batch_idx = batch_head // H
    head_idx = batch_head % H
    
    # Early exit if the entire block is out of bounds
    if start_m * BLOCK_M >= S:
        return
        
    q_base = Q + batch_idx * stride_qb + head_idx * stride_qh
    k_base = K + batch_idx * stride_kb + head_idx * stride_kh
    v_base = V + batch_idx * stride_vb + head_idx * stride_vh
    o_base = O + batch_idx * stride_ob + head_idx * stride_oh
    lse_base = LSE + batch_idx * stride_lseb + head_idx * stride_lseh
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    
    q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q_mask = offs_m < S
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # Since BLOCK_M is always a multiple of BLOCK_N in our configs, we can precisely calculate end_n
    end_n = (start_m + 1) * BLOCK_M // BLOCK_N
    
    k_ptrs = k_base + tl.arange(0, BLOCK_N)[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = v_base + tl.arange(0, BLOCK_N)[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    for start_n in range(0, end_n):
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        k_mask = offs_n < S
        
        k = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
        
        qk_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k.T, acc=qk_acc)
        qk = qk * sm_scale
        
        # We enforce the causal mask (offs_m >= offs_n) and ensure we don't read keys past S.
        valid_mask = (offs_m[:, None] >= offs_n[None, :]) & k_mask[None, :]
        
        # If the query is out of bounds (q_mask is False), we avoid masking with -inf to 
        # prevent NaNs during exp( -inf - (-inf) ). These out-of-bound rows compute garbage
        # safely but are masked out in the final tl.store.
        final_mask = valid_mask | (~q_mask[:, None])
        qk = tl.where(final_mask, qk, float("-inf"))
        
        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        l_i_new = alpha * l_i + tl.sum(p, axis=1)
        
        acc = acc * alpha[:, None]
        p_bf16 = p.to(tl.bfloat16)
        acc = tl.dot(p_bf16, v, acc=acc)
        
        m_i = m_i_new
        l_i = l_i_new
        
        # Advance key/value pointers to the next block along the sequence dimension
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs

    acc = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)
    
    o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=q_mask[:, None])
    
    lse_ptrs = lse_base + offs_m * stride_lses
    tl.store(lse_ptrs, lse, mask=q_mask)

def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    sm_scale = 1.0 / (D ** 0.5)
    
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H)
    
    _attn_fwd_kernel[grid](
        Q, K, V, O, LSE,
        sm_scale,
        S,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        H,
        BLOCK_D=D,
    )