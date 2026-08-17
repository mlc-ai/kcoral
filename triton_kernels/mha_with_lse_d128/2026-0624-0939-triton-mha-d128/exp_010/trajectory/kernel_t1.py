import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_fwd_kernel(
    desc_Q, desc_K, desc_V,
    O_ptr, LSE_ptr,
    S_len, scale, D,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    batch_head = tl.program_id(0)
    start_n = tl.program_id(1) * BLOCK_M
    offset_s = batch_head * S_len + start_n
    
    num_j = tl.cdiv(S_len, BLOCK_N)
    
    Q = desc_Q.load([offset_s, 0])
    
    O_acc = tl.zeros((BLOCK_M, D), tl.float32)
    l = tl.zeros((BLOCK_M,), tl.float32)
    m = tl.full((BLOCK_M,), -float('inf'), tl.float32)
    
    for j in range(num_j):
        kv_offset = batch_head * S_len + j * BLOCK_N
        K = desc_K.load([kv_offset, 0])
        V = desc_V.load([kv_offset, 0])
        
        S = tl.dot(Q, K.T)
        S = S * scale
        
        k_idx = j * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_k = (k_idx < S_len)[None, :]
        S = tl.where(mask_k, S, -float('inf'))
        
        m_old = m
        row_max = tl.max(S, axis=1)
        m_new = tl.maximum(m_old, row_max)
        
        p = tl.exp(S - m_new[:, None])
        
        l = l * tl.exp(m_old - m_new) + tl.sum(p, axis=1)
        
        o_scale = tl.exp(m_old - m_new)[:, None]
        O_acc += tl.dot(p, V)
        O_acc *= o_scale
        
        m = m_new
        
    inv_l = 1.0 / l
    out = (O_acc * inv_l[:, None]).to(tl.bfloat16)
    
    local_rows = start_n + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, D)
    out_ptr = O_ptr + offset_s * D + cols[None, :]
    mask_o = (local_rows < S_len)[:, None]
    tl.store(out_ptr, out, mask=mask_o)
    
    lse_val = m + tl.log(l)
    lse_ptr = LSE_ptr + offset_s
    mask_lse = local_rows < S_len
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
    
    _mha_fwd_kernel[grid](
        desc_Q, desc_K, desc_V,
        O.data_ptr(), LSE.data_ptr(),
        S, scale, D,
        BLOCK_M=64, BLOCK_N=64,
        num_warps=4, num_stages=3,
    )