import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=["S"],
    pre_hook=lambda META, descs: _set_block_shapes(descs, META["BLOCK_M"], META["BLOCK_N"]),
)
@triton.jit
def fused_softmax_kernel(
    q_desc,
    k_desc,
    v_desc,
    O_ptr,
    LSE_ptr,
    S,
    H_val,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    scale: tl.constexpr,
):
    block_i = tl.program_id(0)
    bh_idx = tl.program_id(1)
    
    offset_m = block_i * BLOCK_M
    
    if offset_m >= S:
        return
    
    batch_idx = bh_idx // H_val
    head_idx = bh_idx % H_val
    seq_idx_m = head_idx * S + offset_m
    
    q0 = q_desc.load([batch_idx, seq_idx_m, 0])
    q1 = q_desc.load([batch_idx, seq_idx_m, 64])
    
    acc_o = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    m = tl.full((BLOCK_M,), -1e20, dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    num_kv_blocks = tl.cdiv(S, BLOCK_N)
    max_j = min(num_kv_blocks - 1, block_i)
    
    n_indices = tl.arange(0, BLOCK_N)
    q_indices = offset_m + tl.arange(0, BLOCK_M)
    
    for j in range(0, max_j + 1):
        seq_idx_n = head_idx * S + j * BLOCK_N
        
        k0 = k_desc.load([batch_idx, seq_idx_n, 0])
        k1 = k_desc.load([batch_idx, seq_idx_n, 64])
        
        v0 = v_desc.load([batch_idx, seq_idx_n, 0])
        v1 = v_desc.load([batch_idx, seq_idx_n, 64])
        
        acc_s = tl.dot(q0, k0.T) + tl.dot(q1, k1.T)
        
        acc_s = acc_s * scale
        
        mask = (n_indices[None, :] <= q_indices[:, None]) & (n_indices[None, :] < S)
        acc_s = acc_s * mask + (1 - mask) * (-1e20)
        
        m_old = m
        m_row = tl.reduce(acc_s, 1, tl.maximum)
        m_new = tl.maximum(m_old, m_row)
        
        exp_s = tl.exp(acc_s - m_new)
        exp_s = exp_s * mask 
        
        l_row = tl.reduce(exp_s, 1, tl.sum)
        l_new = l * tl.exp(m_old - m_new) + l_row
        
        o_scale = tl.exp(m_old - m_new)
        acc_o[:, 0:64] = acc_o[:, 0:64] * o_scale[:, None]
        acc_o[:, 64:128] = acc_o[:, 64:128] * o_scale[:, None]
        
        acc_o[:, 0:64] += tl.dot(exp_s, v0)
        acc_o[:, 64:128] += tl.dot(exp_s, v1)
        
        m = m_new
        l = l_new
        
    safe_l = tl.where(l > 0, l, 1.0)
    acc_o[:, 0:64] = acc_o[:, 0:64] / safe_l[:, None]
    acc_o[:, 64:128] = acc_o[:, 64:128] / safe_l[:, None]
    
    lse = tl.where(l > 0, m + tl.log(l), float("-inf"))
    
    acc_o = acc_o.to(tl.bfloat16)
    
    query_mask = (offset_m + tl.arange(0, BLOCK_M)) < S
    o_ptr = O_ptr + (bh_idx * S + offset_m) * HEAD_DIM
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, HEAD_DIM)
    ptrs = o_ptr + rows[:, None] * HEAD_DIM + cols[None, :]
    tl.store(ptrs, acc_o, mask=query_mask[:, None])
    
    lse_ptr = LSE_ptr + bh_idx * S + offset_m
    tl.store(lse_ptr + tl.arange(0, BLOCK_M), lse, mask=query_mask)


def _set_block_shapes(descs, BLOCK_M, BLOCK_N):
    def pre_hook(META):
        for desc in descs:
            desc.block_shape = [1, BLOCK_M, HEAD_DIM]
    return pre_hook


def run(Q, K, V, O, LSE):
    """Compute Causal Multi-Head Attention forward pass returning O and LSE."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    BLOCK_M = 128
    BLOCK_N = 128
    
    scale = 1.0 / float(D)**0.5
    
    q_desc = TensorDescriptor.from_tensor(Q, [B, H * S, D])
    k_desc = TensorDescriptor.from_tensor(K, [B, H * S, D])
    v_desc = TensorDescriptor.from_tensor(V, [B, H * S, D])
    
    grid = (triton.cdiv(S, BLOCK_M), B * H)
    
    fused_softmax_kernel[grid](
        q_desc, k_desc, v_desc, O, LSE, S, H,
        HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, scale=scale,
    )