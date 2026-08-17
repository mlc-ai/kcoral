import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_kernel(
    q_desc, k_desc, v_desc, o_desc,
    LSE_ptr, 
    H, S, scale,
    BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr, D: tl.constexpr,
):
    bid_q = tl.program_id(0)
    bid_h = tl.program_id(1)
    bid_b = tl.program_id(2)
    
    qo = bid_q * BLOCK_Q
    
    Q = tl.squeeze(q_desc.load([bid_b, bid_h, qo, 0]))
    
    num_kv = tl.cdiv(S, BLOCK_KV)
    
    acc_o = tl.zeros((BLOCK_Q, D), tl.float32)
    m_i = tl.full((BLOCK_Q, 1), -float('inf'), tl.float32)
    l_i = tl.full((BLOCK_Q, 1), 0.0, tl.float32)
    
    row_indices = tl.arange(0, BLOCK_Q)
    col_indices = tl.arange(0, BLOCK_KV)
    q_seq_o = bid_q * BLOCK_Q + row_indices
    
    for j in range(num_kv):
        ko = j * BLOCK_KV
        
        K = tl.squeeze(k_desc.load([bid_b, bid_h, ko, 0]))
        
        S_tile = tl.dot(Q, K.T) * scale
        
        k_seq_o = j * BLOCK_KV + col_indices
        valid_mask = (q_seq_o[:, None] < S) & (k_seq_o[None, :] < S)
        S_tile = tl.where(valid_mask, S_tile, -float('inf'))
        
        m_old = m_i
        m_new = tl.maximum(m_i, tl.max(S_tile, axis=1, keep_dims=True))
        
        P = tl.where(valid_mask, tl.exp(S_tile - m_new), 0.0)
        
        exp_diff = tl.where(m_new > -float('inf'), tl.exp(m_old - m_new), 0.0)
        l_i = tl.fma(exp_diff, l_i, tl.sum(P, axis=1, keep_dims=True))
        m_i = m_new
        
        V = tl.squeeze(v_desc.load([bid_b, bid_h, ko, 0]))
        
        acc_o = tl.fma(exp_diff, acc_o, tl.dot(P, V))
    
    O = acc_o / (l_i + 1e-30)
    
    o_desc.store([bid_b, bid_h, qo, 0], tl.unsqueeze(tl.unsqueeze(O.to(tl.bfloat16), 0), 0))
    
    lse = m_i + tl.log(l_i + 1e-30)
    lse_base = bid_b * H * S + bid_h * S + qo
    tl.store(LSE_ptr + lse_base + row_indices, lse.squeeze(-1), mask=q_seq_o < S)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    scale = 1.0 / (D ** 0.5)
    
    BLOCK_Q = 64
    BLOCK_KV = 64
    
    q_desc = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_Q, D])
    k_desc = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_KV, D])
    v_desc = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_KV, D])
    o_desc = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_Q, D])
    
    grid = (triton.cdiv(S, BLOCK_Q), H, B)
    
    _mha_kernel[grid](
        q_desc, k_desc, v_desc, o_desc, LSE,
        H, S, scale,
        BLOCK_Q=BLOCK_Q, BLOCK_KV=BLOCK_KV, D=D,
        num_warps=8, num_stages=2,
    )