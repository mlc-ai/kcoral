import torch
import triton
import triton.language as tl


@triton.jit
def load_chunk(base_ptr, chunk_idx, blk_idx, S_len):
    idx = tl.arange(0, 128 * 32)
    row = idx // 32
    col = chunk_idx * 32 + (idx % 32)
    ptr = base_ptr + row * 128 + col
    mask = (blk_idx * 128 + row) < S_len
    tile = tl.load(ptr, mask=mask, other=0.0)
    return tl.reshape(tile, (128, 32))


@triton.jit
def load_v_chunk(base_ptr, chunk_idx, blk_idx, S_len):
    idx = tl.arange(0, 32 * 128)
    row = idx // 128
    col = idx % 128
    ptr = base_ptr + chunk_idx * 32 * 128 + row * 128 + col
    mask = (blk_idx * 128 + chunk_idx * 32 + row) < S_len
    tile = tl.load(ptr, mask=mask, other=0.0)
    return tl.reshape(tile, (32, 128))


@triton.jit
def store_chunk(base_ptr, O_acc, chunk_idx, q_blk, S_len):
    idx = tl.arange(0, 128 * 32)
    row = idx // 32
    col = chunk_idx * 32 + (idx % 32)
    ptr = base_ptr + row * 128 + col
    mask = (q_blk * 128 + row) < S_len
    O_flat = tl.flatten(O_acc)
    tl.store(ptr, O_flat.to(tl.bfloat16), mask=mask)


@triton.jit
def apply_causal_mask(S, q_blk, k_blk, S_len):
    S_flat = tl.flatten(S)
    idx = tl.arange(0, 16384)
    row = idx // 128
    col = idx % 128
    r_pos = q_blk * 128 + row
    c_pos = k_blk * 128 + col
    valid = (c_pos <= r_pos) & (c_pos < S_len)
    S_flat = tl.where(valid, S_flat, -1e38)
    return tl.reshape(S_flat, (128, 128))


@triton.jit
def _attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S_len,
):
    """
    Minimal, correct FlashAttention kernel.
    
    Computes causal multi-head attention forward pass with Log-Sum-Exp outputs.
    Replaces problematic 2D block indexing with safe 1D block loads, resolves mixed 
    dtype mismatches by casting V to fp32, and fixes several critical numerical issues.
    """
    scale = 1.0 / (128.0 ** 0.5)
    
    bh_id = tl.program_id(0)
    q_blk = tl.program_id(1)
    
    valid_q = (q_blk * 128 + tl.arange(0, 128)) < S_len
    if not valid_q[0]:
        return
    
    O_acc_0 = tl.zeros((128, 32), tl.float32)
    O_acc_1 = tl.zeros((128, 32), tl.float32)
    O_acc_2 = tl.zeros((128, 32), tl.float32)
    O_acc_3 = tl.zeros((128, 32), tl.float32)
    
    m = tl.full((128,), -1e38, tl.float32)
    l = tl.full((128,), 0.0, tl.float32)
    
    max_k_blk = min(q_blk, (S_len - 1) // 128)
    
    Q_base = Q_ptr + bh_id * S_len * 128 + q_blk * 128 * 128
    
    Q_0 = load_chunk(Q_base, 0, q_blk, S_len)
    Q_1 = load_chunk(Q_base, 1, q_blk, S_len)
    Q_2 = load_chunk(Q_base, 2, q_blk, S_len)
    Q_3 = load_chunk(Q_base, 3, q_blk, S_len)
    
    for k_blk in range(0, max_k_blk + 1):
        K_base = K_ptr + bh_id * S_len * 128 + k_blk * 128 * 128
        V_base = V_ptr + bh_id * S_len * 128 + k_blk * 128 * 128
        
        K_0 = load_chunk(K_base, 0, k_blk, S_len)
        K_1 = load_chunk(K_base, 1, k_blk, S_len)
        K_2 = load_chunk(K_base, 2, k_blk, S_len)
        K_3 = load_chunk(K_base, 3, k_blk, S_len)
        
        S = tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T) + tl.dot(Q_2, K_2.T) + tl.dot(Q_3, K_3.T)
        S = S * scale
        S = apply_causal_mask(S, q_blk, k_blk, S_len)
        
        m_prev = m
        block_max = tl.max(S, axis=1)
        m = tl.maximum(m, block_max)
        
        m_expanded = tl.reshape(m, (128, 1))
        P = tl.exp(S - m_expanded)
        
        block_sum = tl.sum(P, axis=1)
        
        # Fix: Use standard Online Softmax correction exp(m_old - m_new)
        correction = tl.exp(m_prev - m)
        correction_expanded = tl.reshape(correction, (128, 1))
        
        l = l * correction + block_sum
        
        # Rescale existing O accumulators to account for block level shifts
        O_acc_0 *= correction_expanded
        O_acc_1 *= correction_expanded
        O_acc_2 *= correction_expanded
        O_acc_3 *= correction_expanded
        
        V_0 = load_v_chunk(V_base, 0, k_blk, S_len).to(tl.float32)
        V_1 = load_v_chunk(V_base, 1, k_blk, S_len).to(tl.float32)
        V_2 = load_v_chunk(V_base, 2, k_blk, S_len).to(tl.float32)
        V_3 = load_v_chunk(V_base, 3, k_blk, S_len).to(tl.float32)
        
        P_0, P_1, P_2, P_3 = tl.split(P, 4, dim=1)
        
        O_0 = tl.dot(P_0, V_0) + tl.dot(P_1, V_1) + tl.dot(P_2, V_2) + tl.dot(P_3, V_3)
        O_0 = tl.reshape(O_0, (128, 128))
        O_0_0, O_0_1, O_0_2, O_0_3 = tl.split(O_0, 4, dim=1)
        
        O_acc_0 += O_0_0
        O_acc_1 += O_0_1
        O_acc_2 += O_0_2
        O_acc_3 += O_0_3
        
    l_expanded = tl.reshape(l, (128, 1))
    O_acc_0 /= l_expanded
    O_acc_1 /= l_expanded
    O_acc_2 /= l_expanded
    O_acc_3 /= l_expanded
    
    lse_val = m + tl.log(l)
    
    O_base = O_ptr + bh_id * S_len * 128 + q_blk * 128 * 128
    store_chunk(O_base, O_acc_0, 0, q_blk, S_len)
    store_chunk(O_base, O_acc_1, 1, q_blk, S_len)
    store_chunk(O_base, O_acc_2, 2, q_blk, S_len)
    store_chunk(O_base, O_acc_3, 3, q_blk, S_len)
    
    lse_ptr = LSE_ptr + bh_id * S_len + q_blk * 128
    lse_flat = tl.flatten(tl.reshape(lse_val, (128, 1)))
    idx = tl.arange(0, 128)
    mask = (q_blk * 128 + idx) < S_len
    tl.store(lse_ptr + idx, lse_flat, mask=mask)


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
        S_len,
        num_warps=4,
        num_stages=1,
    )