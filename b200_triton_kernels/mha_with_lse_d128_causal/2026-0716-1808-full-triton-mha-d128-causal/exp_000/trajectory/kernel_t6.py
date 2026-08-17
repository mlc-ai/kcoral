import math
import torch
import triton
import triton.language as tl
from triton import cdiv
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def mha_with_lse_opt_kernel(
    Q_desc, K_desc, V_desc, O_desc,
    lse_ptr, seq_len, num_heads,
    stride_h_lse, stride_s_lse,
    scale, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HALF_HEAD_DIM: tl.constexpr
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    
    q_start = pid_y * BLOCK_M
    if q_start >= seq_len:
        return
    
    row = tl.arange(0, BLOCK_M)
    col = tl.arange(0, HALF_HEAD_DIM)
    col_k = tl.arange(0, BLOCK_N)
    
    row_offset = pid_x * seq_len + q_start
    
    q0 = Q_desc.load([row_offset, 0])
    q1 = Q_desc.load([row_offset, HALF_HEAD_DIM])
    
    acc_o0 = tl.zeros((BLOCK_M, HALF_HEAD_DIM), dtype=tl.float32)
    acc_o1 = tl.zeros((BLOCK_M, HALF_HEAD_DIM), dtype=tl.float32)
    
    prev_max = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    prev_sum = tl.full((BLOCK_M,), 0.0, dtype=tl.float32)
    
    num_k_tiles = min(pid_y + 1, (seq_len + BLOCK_N - 1) // BLOCK_N)
    
    for k_tile in range(num_k_tiles):
        k_start = k_tile * BLOCK_N
        k_offset = pid_x * seq_len + k_start
        
        k0 = K_desc.load([k_offset, 0])
        k1 = K_desc.load([k_offset, HALF_HEAD_DIM])
        
        v0 = V_desc.load([k_offset, 0])
        v1 = V_desc.load([k_offset, HALF_HEAD_DIM])
        
        acc_p = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        acc_p = tl.dot(q0, k0.T, acc_p)
        acc_p = tl.dot(q1, k1.T, acc_p)
        
        p = acc_p * scale
        
        global_row = q_start + row
        global_col = k_start + col_k
        mask_causal = (global_row[:, None] >= global_col[None, :]) & \
                      (global_row[:, None] < seq_len) & \
                      (global_col[None, :] < seq_len)
        
        p = tl.where(mask_causal, p, -float('inf'))
        
        curr_max = tl.max(p, axis=1)
        p = p - curr_max[:, None]
        p_exp = tl.exp(p)
        
        curr_sum = tl.sum(p_exp, axis=1)
        
        new_max = tl.maximum(prev_max, curr_max)
        new_sum = prev_sum * tl.exp(prev_max - new_max) + curr_sum * tl.exp(curr_max - new_max)
        
        acc_o0 = acc_o0 * tl.exp(prev_max[:, None] - new_max[:, None])
        acc_o1 = acc_o1 * tl.exp(prev_max[:, None] - new_max[:, None])
        
        p_exp = p_exp * tl.exp(curr_max[:, None] - new_max[:, None])
        
        acc_o0 = tl.dot(p_exp, v0, acc_o0)
        acc_o1 = tl.dot(p_exp, v1, acc_o1)
        
        prev_max = new_max
        prev_sum = new_sum
        
    lse = tl.log(prev_sum) + prev_max
    
    q_start_i32 = q_start.to(tl.int32)
    seq_len_i32 = seq_len.to(tl.int32)
    valid_q = q_start_i32 + row.to(tl.int32) < seq_len_i32
    
    lse_ptr0 = lse_ptr + pid_x * stride_h_lse + q_start * stride_s_lse
    tl.store(lse_ptr0 + row, lse, mask=valid_q)
    
    O_desc.store([row_offset, 0], acc_o0)
    O_desc.store([row_offset, HALF_HEAD_DIM], acc_o1)


def run(Q, K, V, O, LSE):
    """
    Computes Multi-Head Attention forward with causal mask and outputs Log-Sum-Exp (LSE).
    Signature follows standard destination-passing style.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    num_heads = H
    
    scale = 1.0 / math.sqrt(D)
    
    BLOCK_M = 128
    BLOCK_N = 128
    HALF_HEAD_DIM = 64
    
    Q_desc = TensorDescriptor.from_tensor(Q.view(B * H * S, D), [128, 64])
    K_desc = TensorDescriptor.from_tensor(K.view(B * H * S, D), [128, 64])
    V_desc = TensorDescriptor.from_tensor(V.view(B * H * S, D), [128, 64])
    O_desc = TensorDescriptor.from_tensor(O.view(B * H * S, D), [128, 64])
    
    stride_h_lse = LSE.stride(1)
    stride_s_lse = LSE.stride(2)
    
    grid = lambda META: (B * H, cdiv(S, META["BLOCK_M"]))
    
    with torch.no_grad():
        mha_with_lse_opt_kernel[grid](
            Q_desc, K_desc, V_desc, O_desc,
            LSE, S, num_heads,
            stride_h_lse, stride_s_lse,
            scale,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HALF_HEAD_DIM=HALF_HEAD_DIM,
            num_warps=4, num_stages=3,
        )