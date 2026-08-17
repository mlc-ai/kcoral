import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def flash_attention_kernel(
    q_desc, k_desc, v_desc,
    out_O_ptr, out_LSE_ptr,
    S, scale, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
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
    col_idx_0 = tl.arange(0, 64)
    col_idx_1 = tl.arange(0, 64)
    
    q_offset = bh * S + program_id * BLOCK_M
    
    o_out = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    l_i = tl.full((BLOCK_M,), 0.0, dtype=tl.float32)
    
    # Load the entire Q tile (shape [BLOCK_M, D])
    q = q_desc.load([q_offset, 0])
    
    num_k_tiles = S // BLOCK_N
    
    # Iterate sequentially enforcing causality constraint
    for j in range(num_k_tiles):
        if j > program_id:
            break
            
        k_offset = bh * S + j * BLOCK_N
        
        k = k_desc.load([k_offset, 0])
        v = v_desc.load([k_offset, 0])
        
        k_T = k.T
        
        zero_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        s = tl.dot(q, k_T, acc=zero_acc)
        
        s = s * scale
        
        q_positions = program_id * BLOCK_M + row_idx
        k_positions = j * BLOCK_N + col_idx
        
        mask_qk = (q_positions[:, None] >= k_positions[None, :]) & (k_positions[None, :] < S)
            
        s = tl.where(mask_qk, s, -float('inf'))
        
        new_m = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.exp(s - new_m[:, None])
        new_l = tl.exp(m_i - new_m) * l_i + tl.sum(p, axis=1)
        
        factor = tl.exp(m_i - new_m)
        o_out = o_out * factor[:, None]
        
        p = p * mask_qk.to(tl.float32)
        
        o_out = tl.dot(p, v, o_out)
        
        m_i = new_m
        l_i = new_l
        
    valid_row = (program_id * BLOCK_M + row_idx) < S
    
    out_ptr0 = out_O_ptr + (bh * S * 128 + (program_id * BLOCK_M + row_idx[:, None]) * 128 + col_idx_0[None, :])
    out_ptr1 = out_O_ptr + (bh * S * 128 + (program_id * BLOCK_M + row_idx[:, None]) * 128 + 64 + col_idx_1[None, :])
    
    # Divide o_out by cumulative row sums and cast to bf16
    # Split o_out along dim 1 (columns) into two matching 64-column chunks aligning with pointers
    o_left, o_right = tl.split((o_out / l_i[:, None]).to(tl.bfloat16), dim=1)
    
    tl.store(out_ptr0, o_left, mask=valid_row[:, None])
    tl.store(out_ptr1, o_right, mask=valid_row[:, None])
    
    # Write LSE
    lse = m_i + tl.log(l_i)
    lse_ptr = out_LSE_ptr + bh * S + program_id * BLOCK_M + row_idx
    tl.store(lse_ptr, lse, mask=valid_row)


def run(Q, K, V, O, LSE):
    """Compute causal multi-head attention O and LSE into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    BLOCK_M = 128
    BLOCK_N = 128
    scale = 1.0 / (D ** 0.5)
    
    Q_flat = Q.reshape(-1, D)
    K_flat = K.reshape(-1, D)
    V_flat = V.reshape(-1, D)
    
    q_desc = TensorDescriptor.from_tensor(Q_flat, [BLOCK_M, BLOCK_N])
    k_desc = TensorDescriptor.from_tensor(K_flat, [BLOCK_N, BLOCK_N])
    v_desc = TensorDescriptor.from_tensor(V_flat, [BLOCK_N, BLOCK_N])
    
    grid = (triton.cdiv(S, BLOCK_M), B * H)
    
    flash_attention_kernel[grid](
        q_desc, k_desc, v_desc,
        O.data_ptr(), LSE.data_ptr(),
        S, scale, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=8
    )