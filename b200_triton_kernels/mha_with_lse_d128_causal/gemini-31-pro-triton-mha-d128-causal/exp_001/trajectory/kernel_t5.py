import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=5),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, O, LSE,
    sm_scale_log2,
    S,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    H,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(S, BLOCK_M)
    
    # Swizzle program ID to dramatically improve L2 cache hit rate for K and V.
    # We group tiles so that multiple SMs process tiles for the same batch_head concurrently.
    GROUP_M: tl.constexpr = 8
    grid_bh = tl.num_programs(0) // grid_m
    
    num_pid_in_group = GROUP_M * grid_bh
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(grid_m - first_pid_m, GROUP_M)
    
    start_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    batch_head = (pid % num_pid_in_group) // group_size_m
    
    batch_idx = batch_head // H
    head_idx = batch_head % H
    
    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    
    # Base pointers
    q_ptrs = Q + batch_idx * stride_qb + head_idx * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    k_ptrs = K + batch_idx * stride_kb + head_idx * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + batch_idx * stride_vb + head_idx * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    
    # Fast-path detection: if this M block is completely within sequence bounds, we skip ALL Q/K/V sequence bounds masks
    is_m_full = (start_m + 1) * BLOCK_M <= S
    
    hi = tl.minimum(S, (start_m + 1) * BLOCK_M)
    num_full_blocks = tl.minimum(S, start_m * BLOCK_M) // BLOCK_N
    end_n_blocks = tl.cdiv(hi, BLOCK_N)
    
    if is_m_full:
        # ---- FAST PATH: Zero bounds-masking overhead ----
        q = tl.load(q_ptrs)
        q = (q * sm_scale_log2).to(tl.bfloat16)
        
        # Phase 1: Fully unmasked sequence blocks
        for i in range(0, num_full_blocks):
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
            
            qk = tl.dot(q, k.T)
            
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            
            alpha = tl.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            
            acc = acc * alpha[:, None]
            acc += tl.dot(p.to(tl.bfloat16), v)
            
            m_i = m_ij
            k_ptrs += BLOCK_N * stride_ks
            v_ptrs += BLOCK_N * stride_vs
            
        # Phase 2: Causal boundary blocks
        for i in range(num_full_blocks, end_n_blocks):
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
            
            qk = tl.dot(q, k.T)
            
            start_n = i * BLOCK_N
            causal_mask = offs_m[:, None] >= (start_n + offs_n)[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))
            
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            
            alpha = tl.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            
            acc = acc * alpha[:, None]
            acc += tl.dot(p.to(tl.bfloat16), v)
            
            m_i = m_ij
            k_ptrs += BLOCK_N * stride_ks
            v_ptrs += BLOCK_N * stride_vs
            
        acc = acc / l_i[:, None]
        o_ptrs = O + batch_idx * stride_ob + head_idx * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
        tl.store(o_ptrs, acc.to(tl.bfloat16))
        
        ln2 = 0.6931471805599453
        lse = (m_i + tl.log2(l_i)) * ln2
        lse_ptrs = LSE + batch_idx * stride_lseb + head_idx * stride_lseh + offs_m * stride_lses
        tl.store(lse_ptrs, lse)
        
    else:
        # ---- SAFE PATH: Explicit sequence bounds-masking ----
        q_mask = offs_m < S
        q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)
        q = (q * sm_scale_log2).to(tl.bfloat16)
        
        for i in range(0, num_full_blocks):
            # Sequence bounds guaranteed to be valid for Phase 1 because start_n + BLOCK_N <= start_m * BLOCK_M <= S
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
            
            qk = tl.dot(q, k.T)
            
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            
            alpha = tl.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            
            acc = acc * alpha[:, None]
            acc += tl.dot(p.to(tl.bfloat16), v)
            
            m_i = m_ij
            k_ptrs += BLOCK_N * stride_ks
            v_ptrs += BLOCK_N * stride_vs
            
        for i in range(num_full_blocks, end_n_blocks):
            start_n = i * BLOCK_N
            offs_n_curr = start_n + offs_n
            k_mask = offs_n_curr < S
            
            k = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)
            v = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
            
            qk = tl.dot(q, k.T)
            
            causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
            # Leave strictly padded Q rows as 0 (not masked to -inf) to safely evade NaN corruption
            valid_mask = causal_mask | (~q_mask[:, None])
            qk = tl.where(valid_mask, qk, float("-inf"))
            
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            
            alpha = tl.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            
            acc = acc * alpha[:, None]
            acc += tl.dot(p.to(tl.bfloat16), v)
            
            m_i = m_ij
            k_ptrs += BLOCK_N * stride_ks
            v_ptrs += BLOCK_N * stride_vs
            
        acc = acc / l_i[:, None]
        o_ptrs = O + batch_idx * stride_ob + head_idx * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
        tl.store(o_ptrs, acc.to(tl.bfloat16), mask=q_mask[:, None])
        
        ln2 = 0.6931471805599453
        lse = (m_i + tl.log2(l_i)) * ln2
        lse_ptrs = LSE + batch_idx * stride_lseb + head_idx * stride_lseh + offs_m * stride_lses
        tl.store(lse_ptrs, lse, mask=q_mask)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    # Pre-scale configuration incorporating log2(e) for tl.exp2 substitution
    sm_scale = 1.0 / (D ** 0.5)
    log2_e = 1.4426950408889634
    sm_scale_log2 = sm_scale * log2_e
    
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]) * B * H, )
    
    _attn_fwd_kernel[grid](
        Q, K, V, O, LSE,
        sm_scale_log2,
        S,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        H,
        D=D,
    )