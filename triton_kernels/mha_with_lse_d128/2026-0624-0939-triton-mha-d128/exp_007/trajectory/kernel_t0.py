import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_kernel(
    q_desc, k_desc, v_desc, o_desc,
    LSE, 
    H, S, scale,
    BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr, D: tl.constexpr,
):
    bid_q = tl.program_id(0)
    bid_h = tl.program_id(1)
    bid_b = tl.program_id(2)
    
    qo = bid_q * BLOCK_Q
    
    Q = q_desc.load([bid_b, bid_h, qo, 0])
    
    num_kv = tl.cdiv(S, BLOCK_KV)
    
    acc_o = tl.zeros((BLOCK_Q, D), tl.float32)
    m_i = tl.full((BLOCK_Q,), -float('inf'), tl.float32)
    l_i = tl.full((BLOCK_Q,), 0.0, tl.float32)
    
    for j in range(num_kv):
        ko = j * BLOCK_KV
        
        K = k_desc.load([bid_b, bid_h, ko, 0])
        V = v_desc.load([bid_b, bid_h, ko, 0])
        
        S_tile = tl.dot(Q, K.T) * scale
        
        k_valid = (ko + tl.arange(0, BLOCK_KV)[None, :]) < S
        S_tile = tl.where(k_valid, S_tile, -float('inf'))
        
        m_old = m_i
        m_new = tl.maximum(m_i, tl.max(S_tile, axis=1, keep_dims=True))
        
        P = tl.exp(S_tile - m_new)
        
        exp_diff = tl.exp(m_old - m_new)
        l_i = tl.fma(exp_diff, l_i, tl.sum(P, axis=1, keep_dims=True))
        m_i = m_new
        
        acc_o = tl.fma(exp_diff, acc_o, tl.dot(P, V))
    
    O = acc_o / (l_i + 1e-30)
    o_desc.store([bid_b, bid_h, qo, 0], O.to(tl.bfloat16))
    
    lse = m_i + tl.log(l_i + 1e-30)
    lse_rows = tl.arange(0, BLOCK_Q)
    lse_base = LSE + bid_b * (H * S) + bid_h * S + qo
    tl.store(lse_base + lse_rows, lse.squeeze(-1), mask=lse_rows < (S - qo))


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    scale = 1.0 / (D ** 0.5)
    
    BLOCK_Q = 64
    BLOCK_KV = 64
    
    q_desc = TensorDescriptor.from_tensor(Q, [BLOCK_Q, D])
    k_desc = TensorDescriptor.from_tensor(K, [BLOCK_KV, D])
    v_desc = TensorDescriptor.from_tensor(V, [BLOCK_KV, D])
    o_desc = TensorDescriptor.from_tensor(O, [BLOCK_Q, D])
    
    grid = (triton.cdiv(S, BLOCK_Q), H, B)
    
    _mha_kernel[grid](
        q_desc, k_desc, v_desc, o_desc, LSE,
        H, S, scale,
        BLOCK_Q=BLOCK_Q, BLOCK_KV=BLOCK_KV, D=D,
        num_warps=8, num_stages=2,
    )