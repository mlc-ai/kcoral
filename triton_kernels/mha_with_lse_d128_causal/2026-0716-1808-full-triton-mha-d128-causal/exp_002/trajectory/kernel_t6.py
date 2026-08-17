import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _attention_forward_kernel(
    desc_q_ptr,
    desc_k_ptr,
    desc_v_ptr,
    desc_o_ptr,
    lse_ptr,
    S,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    D_HALF: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_q_blk = tl.program_id(2)
    pid_k_blk = tl.program_id(3)
    
    bid_bh = pid_b * 48 + pid_h
    
    q_row_start = pid_q_blk * BLOCK_M
    q_idx = q_row_start + tl.arange(0, BLOCK_M)
    valid_rows = q_idx < S
    
    stride_q_bh = BLOCK_M * HEAD_DIM
    stride_k_bh = BLOCK_N * HEAD_DIM
    stride_v_bh = BLOCK_N * HEAD_DIM
    stride_o_bh = BLOCK_M * HEAD_DIM
    
    out_acc_left = tl.zeros([BLOCK_M, D_HALF], dtype=tl.float32)
    out_acc_right = tl.zeros([BLOCK_M, D_HALF], dtype=tl.float32)
    
    m = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    d = tl.full([BLOCK_M], 0.0, dtype=tl.float32)
    
    q = desc_q_ptr.load([bid_bh * stride_q_bh + q_row_start, 0])
    
    dummy_p = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for k_row_start in tl.range(0, S, BLOCK_N, num_stages=3):
        if k_row_start > q_row_start + BLOCK_M - 1:
            break
            
        k = desc_k_ptr.load([bid_bh * stride_k_bh + k_row_start, 0])
        v = desc_v_ptr.load([bid_bh * stride_v_bh + k_row_start, 0])
        
        k_t = k.T
        k_transposed_h0 = k_t[:, :D_HALF]
        k_transposed_h1 = k_t[:, D_HALF:]
        
        q_h0 = q[:, :D_HALF]
        q_h1 = q[:, D_HALF:]
        
        p = tl.dot(q_h0, k_transposed_h0, output=dummy_p)
        p = tl.dot(q_h1, k_transposed_h1, output=p)
        
        p = p * (1.0 / (HEAD_DIM ** 0.5))
        
        mask = (q_row_start + tl.arange(0, BLOCK_M))[:, None] >= (k_row_start + tl.arange(0, BLOCK_N))[None, :]
        mask = mask & ((k_row_start + tl.arange(0, BLOCK_N))[None, :] < S)
        p = tl.where(mask, p, -float('inf'))
        
        curr_max = tl.maximum(m, tl.max(p, axis=1))
        exp_diff = tl.exp(m - curr_max)
        d = d * exp_diff
        m = curr_max
        
        out_acc_left = out_acc_left * exp_diff[:, None]
        out_acc_right = out_acc_right * exp_diff[:, None]
        
        exp_p = tl.exp(p - m[:, None])
        d += tl.sum(exp_p, axis=1)
        
        left_half = exp_p[:, :128]
        right_half = exp_p[:, 128:]
        
        v_left = v[:, :BLOCK_N // 2]
        v_right = v[:, BLOCK_N // 2:]
        
        out_acc_left = tl.dot(left_half, v_left, output=out_acc_left)
        out_acc_right = tl.dot(right_half, v_right, output=out_acc_right)
        
    final_out_left = (out_acc_left / d[:, None]).to(tl.bfloat16)
    final_out_right = (out_acc_right / d[:, None]).to(tl.bfloat16)
    
    desc_o_ptr.store([bid_bh * stride_o_bh + q_row_start, 0], final_out_left)
    desc_o_ptr.store([bid_bh * stride_o_bh + q_row_start, 64], final_out_right)
    
    lse_ptr_bh = lse_ptr + bid_bh * S
    
    lse = m + tl.log(d)
    lse_store = tl.where(valid_rows, lse, 0.0)
    
    tl.store(lse_ptr_bh + q_idx, lse_store, mask=valid_rows)


def run(Q, K, V, O, LSE):
    """
    Computes causal multi-head attention O and LSE given Q, K, V on the current device.
    Expected shapes:
      Q, K, V : [B, H, S, D]
      O       : [B, H, S, D]
      LSE     : [B, H, S]
    Where B=4, H=48, D=128.
    """
    torch.cuda.set_device(Q.device)
    
    S = Q.shape[2]
    
    BLOCK_M = 256
    BLOCK_N = 256
    HEAD_DIM = 128
    D_HALF = 64
    
    desc_q = TensorDescriptor.from_tensor(Q, [BLOCK_M, HEAD_DIM])
    desc_k = TensorDescriptor.from_tensor(K, [BLOCK_N, HEAD_DIM])
    desc_v = TensorDescriptor.from_tensor(V, [BLOCK_N, HEAD_DIM])
    desc_o = TensorDescriptor.from_tensor(O, [BLOCK_M, HEAD_DIM])
    
    grid = (
        4,
        48,
        triton.cdiv(S, BLOCK_M),
        triton.cdiv(S, BLOCK_N)
    )
    
    _attention_forward_kernel[grid](
        desc_q, desc_k, desc_v, desc_o,
        LSE,
        S,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=HEAD_DIM,
        D_HALF=D_HALF,
        num_warps=8,
        num_stages=3,
    )