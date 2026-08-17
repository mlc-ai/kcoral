import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def fused_softmax_kernel(
    q_desc,
    k_desc,
    v_desc,
    o_desc,
    lse_desc,
    S,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    block_i = tl.program_id(0)
    block_j_start = tl.program_id(1)
    bh_idx = tl.program_id(2)
    
    H_val = 48
    batch_idx = bh_idx // H_val
    head_idx = bh_idx % H_val
    
    offset_m = block_i * BLOCK_M
    
    # Load the static query tile once
    q_partial = q_desc.load([batch_idx, head_idx, offset_m, 0])
    
    acc_o = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    m = tl.full((BLOCK_M,), -1e20, dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    num_blocks = tl.cdiv(S, BLOCK_N)
    
    start_j = block_j_start
    for j in range(0, num_blocks):
        idx = (start_j + j) % num_blocks
        offset_n = idx * BLOCK_N
        
        n_indices = offset_n + tl.arange(0, BLOCK_N)
        q_indices = offset_m + tl.arange(0, BLOCK_M)
        
        mask = (n_indices[None, :] <= q_indices[:, None]) & (n_indices[None, :] < S)
        
        k_partial = k_desc.load([batch_idx, head_idx, offset_n, 0])
        v_partial = v_desc.load([batch_idx, head_idx, offset_n, 0])
        
        acc_s = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        for d_chunk in range(0, HEAD_DIM, 32):
            q_slice = q_partial[:, d_chunk:d_chunk+32]
            k_slice = k_partial[:, d_chunk:d_chunk+32]
            acc_s += tl.dot(q_slice, k_slice.T)
            
        acc_s = acc_s * (1.0 / tl.sqrt(tl.float32(HEAD_DIM)))
        
        acc_s = tl.where(mask, acc_s, -1e20)
        
        m_old = m
        m_row = tl.reduce(acc_s, 1, tl.maximum)
        m_new = tl.maximum(m_old, m_row)
        
        exp_s = tl.exp(acc_s - m_new)
        exp_s = tl.where(mask, exp_s, 0.0)
        
        l_row = tl.reduce(exp_s, 1, tl.sum)
        l_new = l * tl.exp(m_old - m_new) + l_row
        
        o_scale = tl.exp(m_old - m_new)
        acc_o = acc_o * o_scale[:, None]
        
        for d_chunk in range(0, HEAD_DIM, 32):
            e_slice = exp_s[:, d_chunk:d_chunk+32]
            v_slice = v_partial[:, d_chunk:d_chunk+32]
            partial_pv = tl.dot(e_slice, v_slice)
            acc_o[:, d_chunk:d_chunk+32] += partial_pv
            
        m = m_new
        l = l_new
        
    acc_o = acc_o / l[:, None]
    
    lse = m + tl.log(l)
    
    acc_o = acc_o.to(tl.bfloat16)
    o_desc.store([batch_idx, head_idx, offset_m, 0], acc_o)
    lse_desc.store([batch_idx, head_idx, offset_m], lse)


def run(Q, K, V, O, LSE):
    """Compute Causal Multi-Head Attention forward pass returning O and LSE."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    # Use 4-dimensional TensorDescriptors aligning perfectly with physical memory layouts
    q_desc = TensorDescriptor.from_tensor(Q, [B, H, S, D])
    k_desc = TensorDescriptor.from_tensor(K, [B, H, S, D])
    v_desc = TensorDescriptor.from_tensor(V, [B, H, S, D])
    o_desc = TensorDescriptor.from_tensor(O, [B, H, S, D])
    lse_desc = TensorDescriptor.from_tensor(LSE, [B, H, S])
    
    # Utilize a fully populated grid mapped across Sequence space and Attention Heads
    grid = (triton.cdiv(S, 128), triton.cdiv(S, 128), B * H)
    
    fused_softmax_kernel[grid](
        q_desc,
        k_desc,
        v_desc,
        o_desc,
        lse_desc,
        S,
        HEAD_DIM=128,
        BLOCK_M=128,
        BLOCK_N=128,
        num_warps=4,
        num_stages=3,
    )