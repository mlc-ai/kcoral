import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 64


@triton.jit
def _flash_attention_kernel(
    q_desc,
    k_desc,
    v_desc,
    O_ptr,
    LSE_ptr,
    S,
    H,
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    bh_idx = tl.program_id(0)
    pid_m = tl.program_id(1)
    
    D = 128
    
    offset_m = pid_m * BLOCK_M
    row_idx = offset_m + tl.arange(0, BLOCK_M)
    row_idx_global = bh_idx * S + row_idx
    
    q = q_desc.load([bh_idx * S + offset_m, 0])
    
    acc_o = [
        tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32),
        tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    ]
    
    m = tl.full((BLOCK_M,), -1e20, dtype=tl.float32)
    l = tl.full((BLOCK_M,), 0.0, dtype=tl.float32)
    
    for j in range(tl.cdiv(S, BLOCK_N)):
        offset_n = j * BLOCK_N
        
        cur_k = k_desc.load([bh_idx * S + offset_n, 0])
        cur_v = v_desc.load([bh_idx * S + offset_n, 0])
        
        acc_s = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for chunk in range(2):
            q_chunk = q[:, chunk * BLOCK_K:(chunk + 1) * BLOCK_K]
            k_chunk = cur_k[:, chunk * BLOCK_K:(chunk + 1) * BLOCK_K]
            acc_s = tl.dot(q_chunk, k_chunk.T, acc_s)
            
        acc_s *= scale
        
        kv_idx = offset_n + tl.arange(0, BLOCK_N)
        mask = kv_idx[None, :] < S
        acc_s = tl.where(mask, acc_s, -1e20)
        
        m_new = tl.max(acc_s, axis=1, keep_dims=True)
        prev_m = m
        m = tl.maximum(prev_m, m_new)
        
        next_p = tl.exp(acc_s - m)
        l_new = tl.sum(next_p, axis=1, keep_dims=True)
        m_scale = tl.exp(prev_m - m)
        l = l * m_scale + l_new
        
        acc_o[0] *= m_scale
        acc_o[1] *= m_scale
        
        for chunk in range(2):
            v_chunk = cur_v[:, chunk * BLOCK_K:(chunk + 1) * BLOCK_K]
            acc_o[chunk] = tl.dot(next_p.to(tl.bfloat16), v_chunk, acc_o[chunk])
            
    acc_o[0] = acc_o[0] / l
    acc_o[1] = acc_o[1] / l
    
    row_mask = (row_idx < S)[:, None]
    
    col_idx_0 = tl.arange(0, BLOCK_K)[None, :]
    out_ptr_offset_0 = row_idx_global[:, None] * D + col_idx_0
    tl.store(O_ptr + out_ptr_offset_0, acc_o[0].to(tl.bfloat16), mask=row_mask)
    
    col_idx_1 = BLOCK_K + tl.arange(0, BLOCK_K)[None, :]
    out_ptr_offset_1 = row_idx_global[:, None] * D + col_idx_1
    tl.store(O_ptr + out_ptr_offset_1, acc_o[1].to(tl.bfloat16), mask=row_mask)
    
    final_lse = m[:, 0] + tl.log(l[:, 0])
    lse_ptr_offset = row_idx_global
    store_mask_lse = (row_idx < S)
    tl.store(LSE_ptr + lse_ptr_offset, final_lse, mask=store_mask_lse)


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
    
    scale = 1.0 / (D ** 0.5)
    
    Q_flat = Q.view(B * H * S, D)
    K_flat = K.view(B * H * S, D)
    V_flat = V.view(B * H * S, D)
    
    q_desc = TensorDescriptor.from_tensor(Q_flat, [BLOCK_M, D])
    k_desc = TensorDescriptor.from_tensor(K_flat, [BLOCK_N, D])
    v_desc = TensorDescriptor.from_tensor(V_flat, [BLOCK_N, D])
    
    grid = (B * H, triton.cdiv(S, BLOCK_M))
    _flash_attention_kernel[grid](
        q_desc, k_desc, v_desc,
        out_ptr, lse_ptr,
        S, H,
        scale=scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )