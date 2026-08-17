import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
    ],
    key=["seq_len"],
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, sm_scale,
    O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    seq_len, H,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    
    q_offset = b * stride_qb + h * stride_qh
    k_offset = b * stride_kb + h * stride_kh
    v_offset = b * stride_vb + h * stride_vh
    
    # K is loaded transposed to shape [BLOCK_D, BLOCK_N] for tl.dot
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    k_ptrs = K + k_offset + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_ks
    v_ptrs = V + v_offset + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    q_mask = offs_m < seq_len
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.bfloat16)
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # Causal sequence length upper-bound optimization
    k_end = tl.minimum(seq_len, (pid_m + 1) * BLOCK_M)
    
    for start_n in range(0, k_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_curr = start_n + offs_n
        
        k_mask = offs_n_curr < seq_len
        
        k = tl.load(k_ptrs, mask=k_mask[None, :], other=0.0).to(tl.bfloat16)
        v = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0).to(tl.bfloat16)
        
        qk = tl.dot(q, k, out_dtype=tl.float32) * sm_scale
        qk = tl.where(k_mask[None, :], qk, float("-inf"))
        
        # Apply causal mask only if the block crosses the diagonal
        if start_n + BLOCK_N > pid_m * BLOCK_M:
            causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))
            
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(qk - m_new[:, None])
        
        l_new = l_i * alpha + tl.sum(beta, 1)
        
        # Update accumulator: acc_new = acc_old * alpha + beta @ V
        acc = acc * alpha[:, None]
        acc = tl.dot(beta.to(tl.bfloat16), v, acc)
        
        m_i = m_new
        l_i = l_new
        
        # Advance pointers
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs
        
    # Finalize outputs
    acc = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)
    
    o_offset = b * stride_ob + h * stride_oh
    o_ptrs = O + o_offset + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=q_mask[:, None])
    
    lse_offset = b * stride_lseb + h * stride_lseh
    lse_ptrs = LSE + lse_offset + offs_m * stride_lses
    tl.store(lse_ptrs, lse, mask=q_mask)


def run(Q, K, V, O, LSE):
    """
    Computes causal Multi-Head Attention forward pass returning Output and Log-Sum-Exp.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    if S == 0:
        return
    
    sm_scale = 1.0 / (D ** 0.5)
    
    grid = lambda META: (
        triton.cdiv(S, META["BLOCK_M"]),
        B * H,
        1
    )
    
    _attn_fwd_kernel[grid](
        Q, K, V, sm_scale,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, H,
        BLOCK_D=128
    )