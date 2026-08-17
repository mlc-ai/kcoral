import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

BLOCK_M = 128
BLOCK_N = 128
HALF_HEAD_DIM = 64


@triton.jit
def _flash_attention_kernel(
    q_desc,
    k_desc,
    v_desc,
    O,
    LSE,
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
    
    offset_m = pid_m * BLOCK_M
    row_idx = offset_m + tl.arange(0, BLOCK_M)
    row_idx_global = bh_idx * S + row_idx
    
    q = q_desc.load([bh_idx * S + offset_m, 0])
    
    q_flat = tl.reshape(q, (BLOCK_M * HEAD_DIM,))
    q_0, q_1 = tl.split(q_flat)
    q_0 = tl.reshape(q_0, (BLOCK_M, HALF_HEAD_DIM))
    q_1 = tl.reshape(q_1, (BLOCK_M, HALF_HEAD_DIM))
    
    acc_o_0 = tl.zeros((BLOCK_M, HALF_HEAD_DIM), dtype=tl.float32)
    acc_o_1 = tl.zeros((BLOCK_M, HALF_HEAD_DIM), dtype=tl.float32)
    m = tl.full((BLOCK_M,), -1e20, dtype=tl.float32)
    l = tl.full((BLOCK_M,), 0.0, dtype=tl.float32)
    
    num_kv_blocks = tl.cdiv(S, BLOCK_N)
    
    for j in range(num_kv_blocks):
        offset_n = j * BLOCK_N
        
        cur_k = k_desc.load([bh_idx * S + offset_n, 0])
        cur_v = v_desc.load([bh_idx * S + offset_n, 0])
        
        k_flat = tl.reshape(cur_k, (BLOCK_N * HEAD_DIM,))
        k_0, k_1 = tl.split(k_flat)
        k_0 = tl.reshape(k_0, (BLOCK_N, HALF_HEAD_DIM))
        k_1 = tl.reshape(k_1, (BLOCK_N, HALF_HEAD_DIM))
        
        v_flat = tl.reshape(cur_v, (BLOCK_N * HEAD_DIM,))
        v_0, v_1 = tl.split(v_flat)
        v_0 = tl.reshape(v_0, (BLOCK_N, HALF_HEAD_DIM))
        v_1 = tl.reshape(v_1, (BLOCK_N, HALF_HEAD_DIM))
        
        acc_s = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_s = tl.dot(q_0, k_0.T, acc_s)
        acc_s = tl.dot(q_1, k_1.T, acc_s)
        acc_s *= scale
        
        global_kv_idx = offset_n + tl.arange(0, BLOCK_N)
        mask = (global_kv_idx[None, :] < S) & (row_idx[:, None] < S)
        acc_s = tl.where(mask, acc_s, -1e20)
        
        m_new = tl.max(acc_s, axis=1)
        prev_m = m
        m = tl.maximum(prev_m, m_new)
        
        next_p = tl.exp(acc_s - m[:, None])
        l_new = tl.sum(next_p, axis=1)
        m_scale = tl.exp(prev_m - m)
        l = l * m_scale + l_new
        
        acc_o_0 = acc_o_0 * m_scale[:, None]
        acc_o_1 = acc_o_1 * m_scale[:, None]
        
        acc_o_0 = tl.dot(next_p, v_0, acc_o_0)
        acc_o_1 = tl.dot(next_p, v_1, acc_o_1)
        
    acc_o_0 = acc_o_0 / l[:, None]
    acc_o_1 = acc_o_1 / l[:, None]
    
    for i in range(2):
        acc_o = acc_o_0 if i == 0 else acc_o_1
        col_idx = i * HALF_HEAD_DIM + tl.arange(0, HALF_HEAD_DIM)
        out_ptr_offset = row_idx_global[:, None] * HEAD_DIM + col_idx[None, :]
        store_mask_o = (row_idx_global[:, None] < S)
        tl.store(O + out_ptr_offset, acc_o.to(tl.bfloat16), mask=store_mask_o)
        
    final_lse = m + tl.log(l)
    lse_ptr_offset = row_idx_global
    store_mask_lse = (row_idx_global < S)
    tl.store(LSE + lse_ptr_offset, final_lse, mask=store_mask_lse)


def run(Q, K, V, O, LSE):
    """Compute non-causal multi-head attention with Log-Sum-Exp into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    assert Q.shape == K.shape == V.shape, "Q, K, V must have the same shape"
    assert O.shape == Q.shape, "Output O must match input shape"
    assert LSE.shape == (B, H, S), "LSE must have shape (B, H, S)"
    assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous(), "Inputs must be contiguous"
    
    HEAD_DIM = D
    scale = 1.0 / (HEAD_DIM ** 0.5)
    
    q_desc = TensorDescriptor(
        Q.data_ptr(),
        shape=[B * H * S, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    k_desc = TensorDescriptor(
        K.data_ptr(),
        shape=[B * H * S, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    v_desc = TensorDescriptor(
        V.data_ptr(),
        shape=[B * H * S, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    
    grid = (B * H, triton.cdiv(S, BLOCK_M))
    _flash_attention_kernel[grid](
        q_desc, k_desc, v_desc,
        O, LSE,
        S, H,
        scale=scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=HEAD_DIM,
        HALF_HEAD_DIM=HALF_HEAD_DIM,
        num_warps=4,
        num_stages=1,
    )