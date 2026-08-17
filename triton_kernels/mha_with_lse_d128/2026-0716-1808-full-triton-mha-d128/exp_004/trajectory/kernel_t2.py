import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _attention_kernel(
    Q_desc, K_desc, V_desc, O_desc, LSE_ptr,
    S_len, scale, H,
):
    """
    Hopper-optimized FlashAttention kernel targeting WGMMA.
    
    Uses a 32x32 tile for the attention score computation and software pipelining 
    for the Key and Value matrix blocks. The Head Dimension (D=128) is split into 
    two chunks of size 64 to align with the 16x16x16 WGMMA instruction shape.
    
    Grid mapping: (ceil(S/32), B, H)
    - Axis 0: Query block index (each CTA handles 32 query rows)
    - Axis 1: Batch index
    - Axis 2: Attention head index
    """
    
    q_blk = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    
    global_row = q_blk * 32
    bh = b * H + h
    
    q0 = Q_desc.load([bh, global_row, 0]).squeeze(0)
    q1 = Q_desc.load([bh, global_row, 64]).squeeze(0)
    
    acc_O0 = tl.zeros((32, 64), tl.float32)
    acc_O1 = tl.zeros((32, 64), tl.float32)
    m_i = tl.full((32,), -1e38, tl.float32)
    l_i = tl.zeros((32,), 1.0, tl.float32)
    
    num_kv_iters = triton.cdiv(S_len, 32)
    
    for i_j in triton.range(0, num_kv_iters, num_stages=2):
        kv_row = i_j * 32
        
        k0 = K_desc.load([bh, kv_row, 0]).squeeze(0)
        k1 = K_desc.load([bh, kv_row, 64]).squeeze(0)
        
        s = tl.dot(q0, k0.T) + tl.dot(q1, k1.T)
        s = s * scale
        
        kv_offs = kv_row + tl.arange(0, 32)
        kv_mask = kv_offs < S_len
        q_offs = global_row + tl.arange(0, 32)
        q_mask = q_offs < S_len
        s = tl.where(kv_mask[None, :] & q_mask[:, None], s, -1e44)
        
        m_i_prev = m_i
        m_i = tl.maximum(m_i, tl.max(s, axis=1))
        curr_p = tl.exp(s - m_i[:, None])
        l_i = l_i * tl.exp(m_i_prev - m_i) + tl.sum(curr_p, axis=1)
        
        acc_O0 = acc_O0 * tl.exp(m_i_prev - m_i)[:, None]
        acc_O1 = acc_O1 * tl.exp(m_i_prev - m_i)[:, None]
        
        v0 = V_desc.load([bh, kv_row, 0]).squeeze(0)
        v1 = V_desc.load([bh, kv_row, 64]).squeeze(0)
        
        acc_O0 += tl.dot(curr_p, v0)
        acc_O1 += tl.dot(curr_p, v1)
        
    acc_O0 = acc_O0 / l_i[:, None]
    acc_O1 = acc_O1 / l_i[:, None]
    
    O_desc.store([bh, global_row, 0], acc_O0.to(tl.bfloat16), mask=q_mask[:, None])
    O_desc.store([bh, global_row, 64], acc_O1.to(tl.bfloat16), mask=q_mask[:, None])
    
    lse_ptr = LSE_ptr + (bh * S_len) + q_offs
    LSE_val = m_i + tl.log(l_i)
    tl.store(lse_ptr, LSE_val, mask=q_mask)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S_len, D = Q.shape
    
    scale = 1.0 / (D ** 0.5)
    
    Q_desc = TensorDescriptor.from_tensor(Q, [B * H, S_len, D])
    K_desc = TensorDescriptor.from_tensor(K, [B * H, S_len, D])
    V_desc = TensorDescriptor.from_tensor(V, [B * H, S_len, D])
    O_desc = TensorDescriptor.from_tensor(O, [B * H, S_len, D])
    
    grid = (triton.cdiv(S_len, 32), B, H)
    _attention_kernel[grid](Q_desc, K_desc, V_desc, O_desc, LSE, S_len, scale, H, num_warps=4)
    
    return O, LSE