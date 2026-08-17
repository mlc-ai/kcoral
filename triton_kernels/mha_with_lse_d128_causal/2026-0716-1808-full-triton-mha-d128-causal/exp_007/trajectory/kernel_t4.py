import torch
import triton
import triton.language as tl


@triton.jit
def _attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S_len, B, H, D: tl.constexpr = 128,
):
    """
    Standard single-CTA-per-query-block FlashAttention kernel.
    
    Computes causal multi-head attention forward pass with Log-Sum-Exp outputs.
    Uses simple pointer loads and explicit masks. All calculations performed in fp32 
    to avoid mixed dtype tensor core restrictions.
    """
    scale = 1.0 / (D ** 0.5)
    
    bh_id = tl.program_id(0)
    q_blk = tl.program_id(1)
    
    valid_q = (q_blk * 128 + tl.arange(0, 128)) < S_len
    if not valid_q[0]:
        return
    
    O_acc = tl.zeros((128, 128), tl.float32)
    m = tl.full((128,), -1e38, tl.float32)
    l = tl.full((128,), 0.0, tl.float32)
    
    max_k_blk = min(q_blk, (S_len - 1) // 128)
    
    # Precompute column offsets and load Q once for the entire causal sequence
    col_offsets = tl.arange(0, D)
    q_row_offsets = (bh_id * S_len + q_blk * 128) + tl.arange(0, 128)
    Q = tl.load(Q_ptr + q_row_offsets[:, None] * D + col_offsets[None, :], 
                mask=valid_q[:, None], other=0.0).to(tl.float32)
    
    for k_blk in range(0, max_k_blk + 1):
        row_offsets = (bh_id * S_len + k_blk * 128) + tl.arange(0, 128)
        k_row_valid = (row_offsets < (bh_id + 1) * S_len)
        
        K = tl.load(K_ptr + row_offsets[:, None] * D + col_offsets[None, :], 
                    mask=k_row_valid[:, None], other=0.0).to(tl.float32)
        V = tl.load(V_ptr + row_offsets[:, None] * D + col_offsets[None, :], 
                    mask=k_row_valid[:, None], other=0.0).to(tl.float32)
        
        S = tl.dot(Q, K.T) * scale
        
        # Mask completely out-of-bounds items to avoid NaN propagation.
        r_idx = q_blk * 128 + tl.arange(0, 128)
        c_idx = k_blk * 128 + tl.arange(0, 128)
        valid = (c_idx[None, :] <= r_idx[:, None]) & (c_idx[None, :] < S_len)
        S = tl.where(valid, S, -1e38)
        
        m_prev = m
        m = tl.maximum(m, tl.max(S, axis=1))
        P = tl.exp(S - m[:, None])
        
        l = l * tl.exp(m_prev - m) + tl.sum(P, axis=1)
        
        O_acc *= tl.exp(m_prev - m)[:, None]
        O_acc = tl.dot(P, V, acc=O_acc)
    
    O_acc = O_acc / l[:, None]
    lse_val = m + tl.log(l)
    
    out_ptr = O_ptr + bh_id * S_len * D + q_blk * 128 * D
    row_idx = tl.arange(0, 128)[:, None] * D
    col_idx = tl.arange(0, 128)[None, :]
    tl.store(out_ptr + row_idx + col_idx, O_acc.to(tl.bfloat16), mask=valid_q[:, None])
    
    lse_ptr = LSE_ptr + bh_id * S_len + q_blk * 128
    tl.store(lse_ptr + tl.arange(0, 128), lse_val, mask=valid_q)


def run(Q, K, V, O, LSE):
    """Compute causal attention O and LSE into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    B, H, S_len, D = Q.shape
    
    Q_ptr = Q.contiguous()
    K_ptr = K.contiguous()
    V_ptr = V.contiguous()
    
    grid = (B * H, triton.cdiv(S_len, 128))
    
    _attention_kernel[grid](
        Q_ptr, K_ptr, V_ptr, O, LSE,
        S_len, B, H,
        num_warps=4,
        num_stages=1,
    )