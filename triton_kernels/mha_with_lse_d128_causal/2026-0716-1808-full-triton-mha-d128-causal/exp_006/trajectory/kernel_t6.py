import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


D = 128
scale = 1.0 / (D ** 0.5)

@triton.jit
def flash_attention_kernel(
    q_desc, k_desc, v_desc,
    out_O_ptr, out_LSE_ptr,
    S, scale: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, NUM_K_TILES: tl.constexpr
):
    """
    Optimized FlashAttention kernel for causal multi-head attention.
    Uses TensorDescriptor loads and standard tl.dot reductions.
    Implements online softmax with explicit row tracking.
    """
    program_id = tl.program_id(0)
    bh = tl.program_id(1)
    
    row_idx = tl.arange(0, BLOCK_M)
    col_idx = tl.arange(0, BLOCK_N)
    
    q_offset = bh * S + program_id * BLOCK_M
    
    o_out0 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    o_out1 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    l_i = tl.full((BLOCK_M,), 0.0, dtype=tl.float32)
    
    q_left = q_desc.load([q_offset, 0])
    q_right = q_desc.load([q_offset, 64])
    
    # Iterating using a static range bounded by NUM_K_TILES resolves JIT control flow errors.
    for j in range(NUM_K_TILES):
        if j > program_id:
            break
        
        k_offset = bh * S + j * BLOCK_N
        
        k_left = k_desc.load([k_offset, 0])
        k_right = k_desc.load([k_offset, 64])
        v_left = v_desc.load([k_offset, 0])
        v_right = v_desc.load([k_offset, 64])
        
        zero_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        s = tl.dot(q_left, k_left, acc=zero_acc)
        s = tl.dot(q_right, k_right, acc=s)
        
        s = s * scale
        
        q_positions = program_id * BLOCK_M + row_idx
        k_positions = j * BLOCK_N + col_idx
        
        mask_qk = (q_positions[:, None] >= k_positions[None, :]) & (k_positions[None, :] < S)
            
        s = tl.where(mask_qk, s, -float('inf'))
        
        new_m = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.exp(s - new_m[:, None])
        new_l = tl.exp(m_i - new_m) * l_i + tl.sum(p, axis=1)
        
        factor = tl.exp(m_i - new_m)
        o_out0 = o_out0 * factor[:, None]
        o_out1 = o_out1 * factor[:, None]
        
        p = p * mask_qk.to(tl.float32)
        
        o_out0 = tl.dot(p, v_left.T, acc=o_out0)
        o_out1 = tl.dot(p, v_right.T, acc=o_out1)
        
        m_i = new_m
        l_i = new_l
        
    valid_row = (program_id * BLOCK_M + row_idx) < S
    
    row_idx_0 = tl.arange(0, BLOCK_M)
    row_idx_1 = tl.arange(0, BLOCK_M)
    col_idx_0 = tl.arange(0, 64)
    col_idx_1 = tl.arange(0, 64)
    
    out_ptr0 = out_O_ptr + bh * S * D + (program_id * BLOCK_M + row_idx_0[:, None]) * D + col_idx_0[None, :]
    out_ptr1 = out_O_ptr + bh * S * D + (program_id * BLOCK_M + row_idx_1[:, None]) * D + 64 + col_idx_1[None, :]
    
    tl.store(out_ptr0, (o_out0 / l_i[:, None]).to(tl.bfloat16), mask=valid_row[:, None])
    tl.store(out_ptr1, (o_out1 / l_i[:, None]).to(tl.bfloat16), mask=valid_row[:, None])
    
    lse = m_i + tl.log(l_i)
    lse_ptr = out_LSE_ptr + bh * S + program_id * BLOCK_M + row_idx
    tl.store(lse_ptr, lse, mask=valid_row)


def run(Q, K, V, O, LSE):
    """Compute causal multi-head attention O and LSE into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    BLOCK_M = 128
    BLOCK_N = 128
    NUM_K_TILES = S // BLOCK_N
    
    Q_flat = Q.reshape(-1, D)
    K_flat = K.reshape(-1, D)
    V_flat = V.reshape(-1, D)
    
    q_desc = TensorDescriptor.from_tensor(Q_flat, [BLOCK_M, 64])
    k_desc = TensorDescriptor.from_tensor(K_flat, [BLOCK_N, 64])
    v_desc = TensorDescriptor.from_tensor(V_flat, [BLOCK_N, 64])
    
    grid = (triton.cdiv(S, BLOCK_M), B * H)
    
    flash_attention_kernel[grid](
        q_desc, k_desc, v_desc,
        O.data_ptr(), LSE.data_ptr(),
        S, scale, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, NUM_K_TILES=NUM_K_TILES, num_warps=8
    )