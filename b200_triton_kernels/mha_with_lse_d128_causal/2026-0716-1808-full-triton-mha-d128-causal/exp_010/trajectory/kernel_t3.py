import torch
import triton
import triton.language as tl


@triton.jit
def swizzle(row, col):
    return (((col // 16) ^ (row // 16)) % 4) * 16 + (col % 16)


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
    BLOCK_K: tl.constexpr,
):
    tid = tl.program_id(0)
    num_blocks_per_bh = triton.cdiv(M, BLOCK_M)
    batch_head_idx = tid // num_blocks_per_bh
    query_block_idx = tid % num_blocks_per_bh

    q_row_idx = tl.arange(0, BLOCK_M)
    
    col_idx_0 = tl.arange(0, BLOCK_K)
    swizzled_offsets_0 = swizzle(q_row_idx, col_idx_0)
    offsets_0 = q_row_idx[:, None] * stride_m + 0 * BLOCK_K + swizzled_offsets_0[None, :]
    Q_0 = tl.load(q_ptr + batch_head_idx * stride_bh + offsets_0, mask=(q_row_idx[:, None] < M), other=0.0)
    
    col_idx_1 = tl.arange(0, BLOCK_K)
    swizzled_offsets_1 = swizzle(q_row_idx, col_idx_1)
    offsets_1 = q_row_idx[:, None] * stride_m + 1 * BLOCK_K + swizzled_offsets_1[None, :]
    Q_1 = tl.load(q_ptr + batch_head_idx * stride_bh + offsets_1, mask=(q_row_idx[:, None] < M), other=0.0)
    
    O_0 = tl.zeros((BLOCK_M, BLOCK_K), tl.float32)
    O_1 = tl.zeros((BLOCK_M, BLOCK_K), tl.float32)
    m = tl.full((BLOCK_M,), float('-inf'), tl.float32)
    l = tl.full((BLOCK_M,), 0.0, tl.float32)

    for j in range(query_block_idx + 1):
        kv_row_idx = tl.arange(0, BLOCK_N)
        
        kv_offsets_0 = kv_row_idx[:, None] * stride_m + 0 * BLOCK_K + swizzle(kv_row_idx, tl.arange(0, BLOCK_K))[None, :]
        K_0 = tl.load(k_ptr + batch_head_idx * stride_bh + kv_offsets_0, mask=(kv_row_idx[:, None] < M), other=0.0)
        V_0 = tl.load(v_ptr + batch_head_idx * stride_bh + kv_offsets_0, mask=(kv_row_idx[:, None] < M), other=0.0)
        
        kv_offsets_1 = kv_row_idx[:, None] * stride_m + 1 * BLOCK_K + swizzle(kv_row_idx, tl.arange(0, BLOCK_K))[None, :]
        K_1 = tl.load(k_ptr + batch_head_idx * stride_bh + kv_offsets_1, mask=(kv_row_idx[:, None] < M), other=0.0)
        V_1 = tl.load(v_ptr + batch_head_idx * stride_bh + kv_offsets_1, mask=(kv_row_idx[:, None] < M), other=0.0)
        
        S = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        S += tl.dot(Q_0, K_0.T)
        S += tl.dot(Q_1, K_1.T)
        S = S * (1.0 / sqrt_D)
        
        apply_mask = (j == query_block_idx)
        mask_ok = query_block_idx * BLOCK_M + q_row_idx[:, None] >= j * BLOCK_N + kv_row_idx[None, :]
        S = tl.where(apply_mask, tl.where(mask_ok, S, -1e20), S)
            
        row_max = tl.max(S, axis=1)
        m_old = m
        m = tl.maximum(m_old, row_max)
        
        exp_m_diff = tl.exp(m_old - m)
        l *= exp_m_diff
        O_0 *= exp_m_diff[:, None]
        O_1 *= exp_m_diff[:, None]
        
        P = tl.exp(S - m[:, None])
        l += tl.sum(P, axis=1)
        
        P_bf16 = P.to(tl.bfloat16)
        
        O_0 += tl.dot(P_bf16, V_0)
        O_1 += tl.dot(P_bf16, V_1)
        
    O_0 /= l[:, None]
    O_1 /= l[:, None]
    
    out_offsets_0 = q_row_idx[:, None] * stride_m + 0 * BLOCK_K + swizzle(q_row_idx, tl.arange(0, BLOCK_K))[None, :]
    out_ptr_0 = o_ptr + batch_head_idx * stride_bh + out_offsets_0
    tl.store(out_ptr_0, O_0.to(tl.bfloat16), mask=(q_row_idx[:, None] < M))
    
    out_offsets_1 = q_row_idx[:, None] * stride_m + 1 * BLOCK_K + swizzle(q_row_idx, tl.arange(0, BLOCK_K))[None, :]
    out_ptr_1 = o_ptr + batch_head_idx * stride_bh + out_offsets_1
    tl.store(out_ptr_1, O_1.to(tl.bfloat16), mask=(q_row_idx[:, None] < M))
    
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
    BLOCK_K = 64
    
    grid = (B * H * triton.cdiv(M, BLOCK_M),)
    
    _attention_kernel[grid](
        Q, K, V, O, LSE,
        M, sqrt_D,
        stride_bh, stride_m, stride_bh_lse,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )