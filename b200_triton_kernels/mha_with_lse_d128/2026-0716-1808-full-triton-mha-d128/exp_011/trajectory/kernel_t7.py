import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attention_kernel(
    ptr_q,
    ptr_k,
    ptr_v,
    O_ptr,
    LSE_ptr,
    S,
    H,
    scale,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HALF_D: tl.constexpr,
):
    bh_idx = tl.program_id(0)
    pid_m = tl.program_id(1)
    
    offset_m = pid_m * BLOCK_M
    row_idx = offset_m + tl.arange(0, BLOCK_M)
    row_idx_global = bh_idx * S + row_idx
    
    q_0 = tl.load(tl.advance(ptr_q, (bh_idx * S + offset_m, 0)))
    q_1 = tl.load(tl.advance(ptr_q, (bh_idx * S + offset_m, HALF_D)))
    
    acc_o_0 = tl.zeros((BLOCK_M, HALF_D), dtype=tl.float32)
    acc_o_1 = tl.zeros((BLOCK_M, HALF_D), dtype=tl.float32)
    m = tl.full((BLOCK_M,), -1e20, dtype=tl.float32)
    l = tl.full((BLOCK_M,), 0.0, dtype=tl.float32)
    
    num_kv_blocks = (S + BLOCK_N - 1) // BLOCK_N
    
    for j in range(num_kv_blocks):
        offset_n = j * BLOCK_N
        
        k_0 = tl.load(tl.advance(ptr_k, (bh_idx * S + offset_n, 0)))
        k_1 = tl.load(tl.advance(ptr_k, (bh_idx * S + offset_n, HALF_D)))
        
        v_0 = tl.load(tl.advance(ptr_v, (bh_idx * S + offset_n, 0)))
        v_1 = tl.load(tl.advance(ptr_v, (bh_idx * S + offset_n, HALF_D)))
        
        acc_s = tl.dot(q_0, k_0.T)
        acc_s = tl.dot(q_1, k_1.T, acc_s)
        acc_s *= scale
        
        kv_idx = offset_n + tl.arange(0, BLOCK_N)
        mask = (kv_idx[None, :] < S) & (row_idx[:, None] < S)
        acc_s = tl.where(mask, acc_s, -1e20)
        
        m_new = tl.max(acc_s, axis=1)
        prev_m = m
        m = tl.maximum(prev_m, m_new)
        
        next_p = tl.exp(acc_s - m[:, None])
        l_new = tl.sum(next_p, axis=1)
        m_scale = tl.exp(prev_m - m)
        l = l * m_scale + l_new
        
        acc_o_0 *= m_scale[:, None]
        acc_o_1 *= m_scale[:, None]
        
        acc_o_0 = tl.dot(next_p, v_0, acc_o_0)
        acc_o_1 = tl.dot(next_p, v_1, acc_o_1)
            
    acc_o_0 /= l[:, None]
    acc_o_1 /= l[:, None]
    
    row_mask = (row_idx < S)[:, None]
    
    col_idx_0 = tl.arange(0, HALF_D)[None, :]
    out_ptr_offset_0 = row_idx_global[:, None] * D + col_idx_0
    tl.store(O_ptr + out_ptr_offset_0, acc_o_0.to(tl.bfloat16), mask=row_mask)
    
    col_idx_1 = HALF_D + tl.arange(0, HALF_D)[None, :]
    out_ptr_offset_1 = row_idx_global[:, None] * D + col_idx_1
    tl.store(O_ptr + out_ptr_offset_1, acc_o_1.to(tl.bfloat16), mask=row_mask)
    
    final_lse = m + tl.log(l)
    lse_ptr_offset = row_idx_global
    store_mask_lse = (row_idx < S)
    tl.store(LSE_ptr + lse_ptr_offset, final_lse, mask=store_mask_lse)


def run(Q, K, V, O, LSE):
    """Compute non-causal multi-head attention with Log-Sum-Exp into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    assert Q.shape == K.shape == V.shape, "Q, K, V must have the same shape"
    assert O.shape == Q.shape, "Output O must match input shape"
    assert LSE.shape == (B, H, S), "LSE must have shape (B, H, S)"
    assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous(), "Inputs must be contiguous"
    
    out_ptr = O.data_ptr()
    lse_ptr = LSE.data_ptr()
    
    scale = 1.0 / (D ** 0.5)
    
    Q_ptr = Q.data_ptr()
    K_ptr = K.data_ptr()
    V_ptr = V.data_ptr()
    
    ptr_q = tl.make_block_ptr(
        Q_ptr, shape=[B * H * S, D], strides=[D, 1],
        block_shape=[128, 64], num_elements=B * H * S * D, dltype=Q_ptr.dtype.element_ty)
    
    ptr_k = tl.make_block_ptr(
        K_ptr, shape=[B * H * S, D], strides=[D, 1],
        block_shape=[128, 64], num_elements=B * H * S * D, dltype=K_ptr.dtype.element_ty)
    
    ptr_v = tl.make_block_ptr(
        V_ptr, shape=[B * H * S, D], strides=[D, 1],
        block_shape=[128, 64], num_elements=B * H * S * D, dltype=V_ptr.dtype.element_ty)
    
    grid = (B * H, triton.cdiv(S, 128))
    _flash_attention_kernel[grid](
        ptr_q, ptr_k, ptr_v,
        out_ptr, lse_ptr,
        S, H, scale,
        D=D, BLOCK_M=128, BLOCK_N=128, HALF_D=64,
        num_warps=4, num_stages=2,
    )