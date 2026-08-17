import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_fwd_kernel(
    desc_Q, desc_K, desc_V,
    O_ptr, LSE_ptr,
    S_len, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    batch_head = tl.program_id(0)
    start_n = tl.program_id(1) * BLOCK_M
    offset_s = batch_head * S_len + start_n
    
    num_j = tl.cdiv(S_len, BLOCK_N)
    
    Q = desc_Q.load([offset_s, 0])
    
    q_idx = start_n + tl.arange(0, 64)
    mask_q = (q_idx < S_len)[:, None]
    Q = tl.where(mask_q, Q, 0.0)
    
    O_acc = tl.zeros((64, 128), tl.float32)
    l = tl.zeros((64,), tl.float32)
    m = tl.full((64,), -float('inf'), tl.float32)
    
    for j in range(num_j):
        kv_offset = batch_head * S_len + j * BLOCK_N
        
        K = desc_K.load([kv_offset, 0])
        
        S = tl.dot(Q, K.T) * scale
        
        k_idx = j * BLOCK_N + tl.arange(0, 64)
        mask_k = (k_idx < S_len)[None, :]
        S = tl.where(mask_k, S, -float('inf'))
        
        m_old = m
        row_max = tl.max(S, axis=1)
        m_new = tl.maximum(m_old, row_max)
        
        P = tl.where(m_new == -float('inf'), 0.0, tl.exp(S - m_new[:, None]))
        
        o_scale = tl.exp(m_old - m_new)[:, None]
        O_acc = O_acc * o_scale
        
        l = l * tl.exp(m_old - m_new) + tl.sum(P, axis=1)
        
        V = desc_V.load([kv_offset, 0])
        V_fp32 = V.to(tl.float32)
        
        O_acc += tl.dot(P, V_fp32)
        
        m = m_new
    
    if num_j > 0:
        inv_l = 1.0 / l
        out = (O_acc * inv_l[:, None]).to(tl.bfloat16)
    else:
        m = 0.0
        l = 1.0
        out = tl.zeros((64, 128), tl.bfloat16)
    
    rows = tl.arange(0, 64)
    cols_left = tl.arange(0, 64)
    cols_right = tl.arange(0, 64) + 64
    
    out_left = out[:, :64]
    out_right = out[:, 64:]
    
    out_ptr_left = O_ptr + (offset_s + rows[:, None]) * 128 + cols_left[None, :]
    out_ptr_right = O_ptr + (offset_s + rows[:, None]) * 128 + cols_right[None, :]
    
    mask_o = (q_idx < S_len)[:, None]
    tl.store(out_ptr_left, out_left, mask=mask_o)
    tl.store(out_ptr_right, out_right, mask=mask_o)
    
    lse_val = m + tl.log(l)
    lse_ptr = LSE_ptr + batch_head * S_len + q_idx
    mask_lse = q_idx < S_len
    tl.store(lse_ptr, lse_val, mask=mask_lse)


def run(Q, K, V, O, LSE):
    device = Q.device
    torch.cuda.set_device(device)
    
    B, H, S, D = Q.shape
    num_batch_head = B * H
    
    Q_flat = Q.view(num_batch_head * S, D)
    K_flat = K.view(num_batch_head * S, D)
    V_flat = V.view(num_batch_head * S, D)
    
    desc_Q = TensorDescriptor.from_tensor(Q_flat, [64, 128])
    desc_K = TensorDescriptor.from_tensor(K_flat, [64, 128])
    desc_V = TensorDescriptor.from_tensor(V_flat, [64, 128])
    
    scale = 1.0 / (D ** 0.5)
    
    grid = (num_batch_head, triton.cdiv(S, 64))
    
    LSE_ptr = LSE.data_ptr()
    
    _mha_fwd_kernel[grid](
        desc_Q, desc_K, desc_V,
        O.data_ptr(), LSE_ptr,
        S, scale,
        BLOCK_M=64, BLOCK_N=64,
        num_warps=4, num_stages=3,
    )