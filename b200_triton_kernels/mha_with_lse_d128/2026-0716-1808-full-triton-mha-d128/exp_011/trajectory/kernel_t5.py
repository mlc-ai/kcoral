import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "scale": 1.0 / (128 ** 0.5)}),
    ],
    key=[],
)
@triton.jit
def _flash_attention_kernel(
    q_desc,
    k_desc,
    v_desc,
    O_ptr,
    LSE_ptr,
    S,
    H,
    D,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    scale: tl.constexpr,
):
    bh_idx = tl.program_id(0)
    pid_m = tl.program_id(1)
    
    offset_m = pid_m * BLOCK_M
    row_idx = offset_m + tl.arange(0, BLOCK_M)
    row_idx_global = bh_idx * S + row_idx
    
    q_0 = q_desc.load([bh_idx * S + offset_m, 0])
    q_1 = q_desc.load([bh_idx * S + offset_m, BLOCK_K])
    
    acc_o_0 = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    acc_o_1 = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    m = tl.full((BLOCK_M,), -1e20, dtype=tl.float32)
    l = tl.full((BLOCK_M,), 0.0, dtype=tl.float32)
    
    num_kv_blocks = (S + BLOCK_N - 1) // BLOCK_N
    
    for j in range(num_kv_blocks):
        offset_n = j * BLOCK_N
        
        k_0 = k_desc.load([bh_idx * S + offset_n, 0])
        k_1 = k_desc.load([bh_idx * S + offset_n, BLOCK_K])
        
        v_0 = v_desc.load([bh_idx * S + offset_n, 0])
        v_1 = v_desc.load([bh_idx * S + offset_n, BLOCK_K])
        
        acc_s = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_s = tl.dot(q_0, k_0.T, acc_s)
        acc_s = tl.dot(q_1, k_1.T, acc_s)
        acc_s *= scale
        
        kv_idx = offset_n + tl.arange(0, BLOCK_N)
        mask = (kv_idx[None, :] < S) & (row_idx[:, None] < S)
        acc_s = tl.where(mask, acc_s, -1e20)
        
        m_new = tl.max(acc_s, axis=1)
        prev_m = m
        m = tl.maximum(prev_m, m_new)
        
        next_p = tl.exp(acc_s - m[:, None])
        l_new = tl.sum(next_p, axis=1)
        m_scale = tl.exp(prev_m - m)
        l = l * m_scale + l_new
        
        acc_o_0 *= m_scale[:, None]
        acc_o_1 *= m_scale[:, None]
        
        acc_o_0 = tl.dot(next_p.to(tl.bfloat16), v_0, acc_o_0)
        acc_o_1 = tl.dot(next_p.to(tl.bfloat16), v_1, acc_o_1)
            
    acc_o_0 /= l[:, None]
    acc_o_1 /= l[:, None]
    
    row_mask = (row_idx < S)[:, None]
    
    col_idx_0 = tl.arange(0, BLOCK_K)[None, :]
    out_ptr_offset_0 = row_idx_global[:, None] * D + col_idx_0
    tl.store(O_ptr + out_ptr_offset_0, acc_o_0.to(tl.bfloat16), mask=row_mask)
    
    col_idx_1 = BLOCK_K + tl.arange(0, BLOCK_K)[None, :]
    out_ptr_offset_1 = row_idx_global[:, None] * D + col_idx_1
    tl.store(O_ptr + out_ptr_offset_1, acc_o_1.to(tl.bfloat16), mask=row_mask)
    
    final_lse = m + tl.log(l)
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
    
    Q_flat = Q.view(B * H * S, D)
    K_flat = K.view(B * H * S, D)
    V_flat = V.view(B * H * S, D)
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    
    q_desc = TensorDescriptor.from_tensor(Q_flat, [BLOCK_M, BLOCK_K])
    k_desc = TensorDescriptor.from_tensor(K_flat, [BLOCK_N, BLOCK_K])
    v_desc = TensorDescriptor.from_tensor(V_flat, [BLOCK_N, BLOCK_K])
    
    grid = (B * H, triton.cdiv(S, BLOCK_M))
    _flash_attention_kernel[grid](
        q_desc, k_desc, v_desc,
        out_ptr, lse_ptr,
        S, H, D,
        num_warps=4,
        num_stages=2,
    )