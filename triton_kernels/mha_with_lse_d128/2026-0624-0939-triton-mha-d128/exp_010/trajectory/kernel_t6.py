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
    
    # Load Q tile once; reuse against every K block. 
    # Split into smaller chunks to bypass limits and enable manual ptr output mapping.
    Q_left = desc_Q.load([offset_s, 0])
    Q_right = desc_Q.load([offset_s, 64])
    
    q_idx = start_n + tl.arange(0, 64)
    mask_q = (q_idx < S_len)[:, None]
    Q_left = tl.where(mask_q, Q_left, 0.0)
    Q_right = tl.where(mask_q, Q_right, 0.0)
    
    # Explicit float32 accumulation preventing precision loss 
    O_acc_left = tl.zeros((64, 64), tl.float32)
    O_acc_right = tl.zeros((64, 64), tl.float32)
    l = tl.zeros((64,), tl.float32)
    m = tl.full((64,), -float('inf'), tl.float32)
    
    for j in range(num_j):
        kv_offset = batch_head * S_len + j * BLOCK_N
        
        K_left = desc_K.load([kv_offset, 0])
        K_right = desc_K.load([kv_offset, 64])
        
        S = tl.dot(Q_left, K_left.T) + tl.dot(Q_right, K_right.T)
        S = S * scale
        
        k_idx = j * BLOCK_N + tl.arange(0, 64)
        mask_k = (k_idx < S_len)[None, :]
        S = tl.where(mask_k, S, -float('inf'))
        
        m_old = m
        row_max = tl.max(S, axis=1)
        m_new = tl.maximum(m_old, row_max)
        
        P = tl.where(m_new == -float('inf'), 0.0, tl.exp(S - m_new[:, None]))
        
        # Explicitly manage exp arithmetic when `m_old` is initially -inf to prevent NaN generation
        o_scale = tl.where(m_old == -float('inf'), 0.0, tl.exp(m_old - m_new))[:, None]
        
        O_acc_left = O_acc_left * o_scale
        O_acc_right = O_acc_right * o_scale
        
        l = l * tl.where(m_old == -float('inf'), 0.0, tl.exp(m_old - m_new)) + tl.sum(P, axis=1)
        
        V_left = desc_V.load([kv_offset, 0])
        V_right = desc_V.load([kv_offset, 64])
        
        O_acc_left += tl.dot(P, V_left)
        O_acc_right += tl.dot(P, V_right)
        
        m = m_new
    
    inv_l = 1.0 / l
    out_left = (O_acc_left * inv_l[:, None]).to(tl.bfloat16)
    out_right = (O_acc_right * inv_l[:, None]).to(tl.bfloat16)
    
    rows = tl.arange(0, 64)
    cols_left = tl.arange(0, 64)
    cols_right = tl.arange(0, 64) + 64
    
    # Utilize `tl.make_ptr` to resolve shape/store type conflicts when writing outputs
    out_ptr_left = tl.make_ptr(O_ptr + offset_s * 128, (64, 64), (128, 1), (0, 0), ty=tl.bfloat16)
    out_ptr_right = tl.make_ptr(O_ptr + offset_s * 128, (64, 64), (128, 1), (0, 64), ty=tl.bfloat16)
    
    row_f = tl.int_to_float(rows[:, None])
    col_f_left = tl.int_to_float(cols_left[None, :])
    col_f_right = tl.int_to_float(cols_right[None, :])
    
    ptr_left = out_ptr_left + row_f * 128.0 + col_f_left * 1.0
    ptr_right = out_ptr_right + row_f * 128.0 + col_f_right * 1.0
    
    mask_o = (q_idx < S_len)[:, None]
    tl.store(ptr_left, out_left, mask=mask_o)
    tl.store(ptr_right, out_right, mask=mask_o)
    
    lse_val = m + tl.log(l)
    lse_ptr = tl.make_ptr(LSE_ptr + batch_head * S_len, (64,), (1,), (0,), ty=tl.float32)
    ptr_lse = lse_ptr + tl.int_to_float(q_idx)
    mask_lse = q_idx < S_len
    tl.store(ptr_lse, lse_val, mask=mask_lse)


def run(Q, K, V, O, LSE):
    device = Q.device
    torch.cuda.set_device(device)
    
    B, H, S, D = Q.shape
    num_batch_head = B * H
    
    Q_flat = Q.view(num_batch_head * S, D)
    K_flat = K.view(num_batch_head * S, D)
    V_flat = V.view(num_batch_head * S, D)
    
    desc_Q = TensorDescriptor.from_tensor(Q_flat, [64, 64])
    desc_K = TensorDescriptor.from_tensor(K_flat, [64, 64])
    desc_V = TensorDescriptor.from_tensor(V_flat, [64, 64])
    
    scale = 1.0 / (D ** 0.5)
    
    grid = (num_batch_head, triton.cdiv(S, 64))
    
    _mha_fwd_kernel[grid](
        desc_Q, desc_K, desc_V,
        O.data_ptr(), LSE.data_ptr(),
        S, scale,
        BLOCK_M=64, BLOCK_N=64,
        num_warps=4, num_stages=3,
    )