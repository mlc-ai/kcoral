import torch
import triton
import triton.language as tl


@triton.jit
def _attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    M,
    sqrt_D,
    stride_bh,
    stride_m,
    stride_bh_lse,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch_head_idx = tl.program_id(0)
    query_block_idx = tl.program_id(1)

    q_row_idx = tl.arange(0, BLOCK_M)
    head_dim = tl.arange(0, 128)
    
    q_offsets = q_row_idx[:, None] * stride_m + head_dim[None, :]
    Q = tl.load(q_ptr + batch_head_idx * stride_bh + q_offsets, mask=(q_row_idx[:, None] < M), other=0.0)
    
    O = tl.zeros((BLOCK_M, 128), tl.float32)
    m = tl.full((BLOCK_M,), float('-inf'), tl.float32)
    l = tl.full((BLOCK_M,), 0.0, tl.float32)

    for j in range(query_block_idx + 1):
        kv_row_idx = tl.arange(0, BLOCK_N)
        kv_offsets = kv_row_idx[:, None] * stride_m + head_dim[None, :]
        
        K = tl.load(k_ptr + batch_head_idx * stride_bh + kv_offsets, mask=(kv_row_idx[:, None] < M), other=0.0)
        V = tl.load(v_ptr + batch_head_idx * stride_bh + kv_offsets, mask=(kv_row_idx[:, None] < M), other=0.0)
        
        S = tl.dot(Q, K.T)
        S = S * (1.0 / sqrt_D)
        
        if j == query_block_idx:
            global_q = query_block_idx * BLOCK_M + q_row_idx[:, None]
            global_k = j * BLOCK_N + kv_row_idx[None, :]
            S = tl.where(global_q >= global_k, S, -1e20)
            
        row_max = tl.max(S, axis=1)
        m_old = m
        m = tl.maximum(m_old, row_max)
        
        exp_m_diff = tl.exp(m_old - m)
        l *= exp_m_diff
        O *= exp_m_diff[:, None]
        
        P = tl.exp(S - m[:, None])
        l += tl.sum(P, axis=1)
        
        O = tl.dot(P, V) + O
        
    out_ptr = o_ptr + batch_head_idx * stride_bh + q_row_idx[:, None] * stride_m + head_dim[None, :]
    tl.store(out_ptr, O.to(tl.bfloat16), mask=(q_row_idx[:, None] < M))
    
    lse_ptr = lse_ptr + batch_head_idx * stride_bh_lse + q_row_idx
    tl.store(lse_ptr, m + tl.log(l), mask=(q_row_idx < M))


def run(Q, K, V, O, LSE):
    """Compute causal multi-head attention O and LSE into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    B, H, M, D = Q.shape
    sqrt_D = D ** 0.5
    
    stride_bh = M * D
    stride_m = D
    stride_bh_lse = M
    
    BLOCK_M = 128
    BLOCK_N = 128
    
    grid = (B * H, triton.cdiv(M, BLOCK_M))
    
    _attention_kernel[grid](
        Q, K, V, O, LSE,
        M, sqrt_D,
        stride_bh, stride_m, stride_bh_lse,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4,
    )