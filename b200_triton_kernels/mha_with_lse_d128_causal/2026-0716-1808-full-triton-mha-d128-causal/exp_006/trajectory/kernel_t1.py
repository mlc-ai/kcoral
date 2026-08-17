import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit(__launch_bounds__(max_threads=128))
def flash_attention_kernel(
    q_desc, k_desc, v_desc,
    out_O_ptr, out_LSE_ptr,
    S, scale, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr
):
    """
    Optimized FlashAttention kernel for causal multi-head attention.
    Uses TensorDescriptor loads and standard tl.dot reductions.
    Implements online softmax with explicit row tracking.
    """
    B = 4; H = 48
    
    program_id = tl.program_id(0)
    bh = tl.program_id(1)
    
    row_idx_0 = tl.arange(0, BLOCK_M)
    row_idx_1 = tl.arange(0, BLOCK_M)
    col_idx_0 = tl.arange(0, 64)
    col_idx_1 = tl.arange(0, 64)
    col_idx = tl.arange(0, BLOCK_N)
    
    q_offset = bh * S + program_id * BLOCK_M
    
    o_out0 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    o_out1 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    l_i = tl.full((BLOCK_M,), 0.0, dtype=tl.float32)
    
    # Load the Q tile for this program (spanning entire block width along S)
    q0 = q_desc.load([q_offset, 0])
    q1 = q_desc.load([q_offset, 64])
    
    # Iterate sequentially over key blocks up to current block enforcing causality constraint
    for j in range(program_id + 1):
        k_offset = bh * S + j * BLOCK_N
        
        k0 = k_desc.load([k_offset, 0])
        k1 = k_desc.load([k_offset, 64])
        v0 = v_desc.load([k_offset, 0])
        v1 = v_desc.load([k_offset, 64])
        
        k0_T = k0.T
        k1_T = k1.T
        
        s_acc = tl.dot(q0, k0_T)
        s_acc = tl.dot(q1, k1_T, s_acc)
        
        s_acc = s_acc * scale
        
        if j == program_id:
            mask_qk = ((program_id * BLOCK_M + row_idx_0[:, None]) >= (j * BLOCK_N + col_idx[None, :])) & \
                      ((j * BLOCK_N + col_idx[None, :]) < S)
        else:
            mask_qk = tl.ones((BLOCK_M, BLOCK_N), dtype=tl.int1)
            
        s_acc = tl.where(mask_qk, s_acc, -float('inf'))
        
        new_m = tl.maximum(m_i, tl.max(s_acc, axis=1))
        p = tl.exp(s_acc - new_m[:, None])
        new_l = tl.exp(m_i - new_m) * l_i + tl.sum(p, axis=1)
        
        factor = tl.exp(m_i - new_m)
        o_out0 = o_out0 * factor[:, None]
        o_out1 = o_out1 * factor[:, None]
        
        p = p * mask_qk.to(tl.float32)
        
        o_out0 = tl.dot(p, v0, o_out0)
        o_out1 = tl.dot(p, v1, o_out1)
        
        m_i = new_m
        l_i = new_l
        
    valid_row = (program_id * BLOCK_M + row_idx_0) < S
    
    out_ptr0 = out_O_ptr + (bh * S * D + (program_id * BLOCK_M + row_idx_0[:, None]) * D + col_idx_0[None, :])
    out_ptr1 = out_O_ptr + (bh * S * D + (program_id * BLOCK_M + row_idx_1[:, None]) * D + 64 + col_idx_1[None, :])
    
    tl.store(out_ptr0, (o_out0 / l_i[:, None]).to(tl.bfloat16), mask=valid_row[:, None])
    tl.store(out_ptr1, (o_out1 / l_i[:, None]).to(tl.bfloat16), mask=valid_row[:, None])
    
    lse = m_i + tl.log(l_i)
    lse_ptr = out_LSE_ptr + bh * S + program_id * BLOCK_M + row_idx_0
    tl.store(lse_ptr, lse, mask=valid_row)


def run(Q, K, V, O, LSE):
    """Compute causal multi-head attention O and LSE into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    BLOCK_M = 64
    BLOCK_N = 64
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
        S, scale, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D, num_warps=4
    )