import torch
import triton
import triton.language as tl


@triton.jit
def _attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S, B, H, D,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    
    scale = 1.0 / tl.sqrt(tl.float32(D))
    
    pid_m = tl.program_id(0)
    row_start = pid_m * BLOCK_M
    b_h = tl.program_id(1)
    
    M = B * H * S
    
    q_desc = tl.make_tensor_descriptor(
        Q_ptr, shape=[M, 2, BLOCK_D], strides=[D, BLOCK_D, 1],
        block_shape=[BLOCK_M, 2, BLOCK_D], padding_option="zero")
    k_desc = tl.make_tensor_descriptor(
        K_ptr, shape=[M, 2, BLOCK_D], strides=[D, BLOCK_D, 1],
        block_shape=[BLOCK_N, 2, BLOCK_D], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(
        V_ptr, shape=[M, 2, BLOCK_D], strides=[D, BLOCK_D, 1],
        block_shape=[BLOCK_N, 2, BLOCK_D], padding_option="zero")
    
    q_offset = b_h * S + row_start
    Q0 = q_desc.load([q_offset, 0, 0])
    Q1 = q_desc.load([q_offset, 1, 0])
    
    O0 = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    O1 = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    num_kv_blocks = (S + BLOCK_N - 1) // BLOCK_N
    
    for j in range(min(num_kv_blocks, pid_m + 1)):
        kv_offset = b_h * S + j * BLOCK_N
        K0 = k_desc.load([kv_offset, 0, 0])
        K1 = k_desc.load([kv_offset, 1, 0])
        V0 = v_desc.load([kv_offset, 0, 0])
        V1 = v_desc.load([kv_offset, 1, 0])
        
        K0_T = K0.T
        K1_T = K1.T
        
        acc_QK = tl.dot(Q0, K0_T) + tl.dot(Q1, K1_T)
        
        acc_QK *= scale
        
        global_q_idx = (row_start + tl.arange(0, BLOCK_M))[:, None]
        global_k_idx = (j * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
        valid = (global_k_idx <= global_q_idx) & (global_k_idx < S)
        
        acc_QK = tl.where(valid, acc_QK, float('-inf'))
        
        M_j = tl.max(acc_QK, axis=1)
        m_prev = m
        m = tl.maximum(m, M_j)
        
        P = tl.exp(acc_QK - m)
        
        l = l * tl.exp(m_prev - m) + tl.sum(P, axis=1)
        
        rescale_O = tl.exp(m_prev - m)[:, None]
        O0 = O0 * rescale_O
        O1 = O1 * rescale_O
        
        P = P.to(tl.bfloat16)
        
        O0 += tl.dot(P, V0)
        O1 += tl.dot(P, V1)
        
    l = tl.where(l > 0, l, 1.0)
    
    O0 = O0 / l[:, None]
    O1 = O1 / l[:, None]
    
    row_idx = row_start + tl.arange(0, BLOCK_M)
    LSE_val = m + tl.log(l)
    
    col = tl.arange(0, BLOCK_D)
    
    out_ptr0 = O_ptr + b_h * S * D + row_idx[:, None] * D + col[None, :]
    tl.store(
        out_ptr0,
        O0.to(tl.bfloat16),
        mask=(row_idx[:, None] < S),
        boundary_check=((0, BLOCK_M), (0, BLOCK_D))
    )
    
    out_ptr1 = O_ptr + b_h * S * D + row_idx[:, None] * D + (col[None, :] + BLOCK_D)
    tl.store(
        out_ptr1,
        O1.to(tl.bfloat16),
        mask=(row_idx[:, None] < S),
        boundary_check=((0, BLOCK_M), (0, BLOCK_D))
    )
    
    tl.store(
        LSE_ptr + b_h * S + row_idx,
        LSE_val,
        mask=(row_idx < S),
        boundary_check=((0, BLOCK_M),)
    )


def run(Q, K, V, O, LSE):
    """Compute causal multi-head attention with Log-Sum-Exp."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_D = 64
    
    grid = (triton.cdiv(S, BLOCK_M), B * H)
    
    _attention_kernel[grid](
        Q, K, V, O, LSE,
        S, B, H, D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )