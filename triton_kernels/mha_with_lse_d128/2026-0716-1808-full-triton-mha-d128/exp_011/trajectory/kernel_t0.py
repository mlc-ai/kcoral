import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

BLOCK_M = 128
BLOCK_N = 128
HEAD_DIM = 128
HALF_HEAD_DIM = 64


@triton.jit
def _flash_attention_kernel(
    q_desc,
    k_desc,
    v_desc,
    out_ptr,
    lse_ptr,
    S,
    H,
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_HEAD_DIM: tl.constexpr,
):
    bh_idx = tl.program_id(0)
    pid_m = tl.program_id(1)
    
    b_idx = bh_idx // H
    h_idx = bh_idx % H
    
    offset_m = pid_m * BLOCK_M
    row_idx = offset_m + tl.arange(0, BLOCK_M)
    row_idx_global = bh_idx * S + row_idx
    
    q = q_desc.load([bh_idx * S + offset_m, 0])
    q_0 = q[:, :HALF_HEAD_DIM]
    q_1 = q[:, HALF_HEAD_DIM:]
    
    lse_stored = False
    
    for col_half in [0, HALF_HEAD_DIM]:
        acc_o = tl.zeros((BLOCK_M, HALF_HEAD_DIM), dtype=tl.float32)
        m = tl.full((BLOCK_M,), -1e20, dtype=tl.float32)
        l = tl.full((BLOCK_M,), 0.0, dtype=tl.float32)
        
        num_kv_blocks = tl.cdiv(S, BLOCK_N)
        
        for j in range(num_kv_blocks):
            offset_n = j * BLOCK_N
            
            cur_k = k_desc.load([bh_idx * S + offset_n, 0])
            cur_v = v_desc.load([bh_idx * S + offset_n, 0])
            
            k_0 = cur_k[:, :HALF_HEAD_DIM]
            k_1 = cur_k[:, HALF_HEAD_DIM:]
            
            acc_s = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            acc_s = tl.dot(q_0, k_0.T, acc_s)
            acc_s = tl.dot(q_1, k_1.T, acc_s)
            acc_s *= scale
            
            global_kv_idx = offset_n + tl.arange(0, BLOCK_N)
            mask = (global_kv_idx[None, :] < S) & (row_idx[:, None] < S)
            acc_s = tl.where(mask, acc_s, -1e20)
            
            m_new = tl.max(acc_s, axis=1, keep_dims=True)
            prev_m = m
            m = tl.maximum(prev_m, m_new)
            
            next_p = tl.exp(acc_s - m)
            l_new = tl.sum(next_p, axis=1, keep_dims=True)
            m_scale = tl.exp(prev_m - m)
            l = l * m_scale + l_new
            
            prev_o = acc_o * m_scale
            
            v_slice = cur_v[:, col_half:col_half+HALF_HEAD_DIM]
            next_o = tl.dot(next_p, v_slice, prev_o)
            
            acc_o = next_o
        
        acc_o = acc_o / l
        
        store_mask_o = row_idx_global[:, None] < ((bh_idx + 1) * S)
        col_idx = col_half + tl.arange(0, HALF_HEAD_DIM)
        out_ptr_offset = row_idx_global[:, None] * HEAD_DIM + col_idx[None, :]
        tl.store(out_ptr + out_ptr_offset, acc_o, mask=store_mask_o)
        
        if not lse_stored:
            final_lse = m[:, 0] + tl.log(l[:, 0])
            store_mask_lse = row_idx_global < ((bh_idx + 1) * S)
            tl.store(lse_ptr + row_idx_global, final_lse, mask=store_mask_lse)
            lse_stored = True


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
    
    scale = 1.0 / (HEAD_DIM ** 0.5)
    
    q_desc = TensorDescriptor(
        Q.data_ptr(),
        shape=[B * H * S, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
        padding_option="zero",
    )
    k_desc = TensorDescriptor(
        K.data_ptr(),
        shape=[B * H * S, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
        padding_option="zero",
    )
    v_desc = TensorDescriptor(
        V.data_ptr(),
        shape=[B * H * S, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
        padding_option="zero",
    )
    
    grid = (B * H, triton.cdiv(S, BLOCK_M))
    _flash_attention_kernel[grid](
        q_desc, k_desc, v_desc,
        out_ptr, lse_ptr,
        S, H,
        scale=scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=HEAD_DIM,
        HALF_HEAD_DIM=HALF_HEAD_DIM,
        num_warps=4,
        num_stages=1,
    )