import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_kernel(
    q_desc, k_desc, v_desc, o_desc,
    LSE_ptr, 
    H, S, scale,
    BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr,
):
    bid_q = tl.program_id(0)
    bid_h = tl.program_id(1)
    bid_b = tl.program_id(2)
    
    qo = bid_q * BLOCK_Q
    
    Q_full = q_desc.load([bid_b, bid_h, qo, 0])
    Q0 = tl.squeeze(Q_full)
    
    Q_full_1 = q_desc.load([bid_b, bid_h, qo, 64])
    Q1 = tl.squeeze(Q_full_1)
    
    num_kv = tl.cdiv(S, BLOCK_KV)
    
    acc_o0 = tl.zeros((BLOCK_Q, 64), tl.float32)
    acc_o1 = tl.zeros((BLOCK_Q, 64), tl.float32)
    m_i = tl.full((BLOCK_Q, 1), -float('inf'), tl.float32)
    l_i = tl.full((BLOCK_Q, 1), 0.0, tl.float32)
    
    q_seq_o = bid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_valid = q_seq_o < S
    
    col_indices = tl.arange(0, BLOCK_KV)
    
    for j in range(num_kv):
        ko = j * BLOCK_KV
        
        K_full = k_desc.load([bid_b, bid_h, ko, 0])
        K0 = tl.squeeze(K_full)
        
        K_full_1 = k_desc.load([bid_b, bid_h, ko, 64])
        K1 = tl.squeeze(K_full_1)
        
        S_tile = tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)
        S_tile *= scale
        
        k_seq_o = j * BLOCK_KV + col_indices
        k_valid = k_seq_o < S
        
        valid_mask = q_valid[:, None] & k_valid[None, :]
        S_tile = tl.where(valid_mask, S_tile, -float('inf'))
        
        m_old = m_i
        m_new = tl.maximum(m_i, tl.max(S_tile, axis=1, keep_dims=True))
        
        P = tl.where(valid_mask, tl.exp(S_tile - m_new), 0.0)
        
        exp_diff = tl.where(m_new > -float('inf'), tl.exp(m_old - m_new), 0.0)
        l_i = tl.fma(exp_diff, l_i, tl.sum(P, axis=1, keep_dims=True))
        m_i = m_new
        
        V_full = v_desc.load([bid_b, bid_h, ko, 0])
        V0 = tl.squeeze(V_full)
        
        V_full_1 = v_desc.load([bid_b, bid_h, ko, 64])
        V1 = tl.squeeze(V_full_1)
        
        P_bf16 = P.to(tl.bfloat16)
        
        acc_o0 = tl.fma(exp_diff, acc_o0, tl.dot(P_bf16, V0))
        acc_o1 = tl.fma(exp_diff, acc_o1, tl.dot(P_bf16, V1))
    
    inv_l = 1.0 / (l_i + 1e-30)
    O0 = (acc_o0 * inv_l).to(tl.bfloat16)
    O1 = (acc_o1 * inv_l).to(tl.bfloat16)
    
    out_mask = (qo + tl.arange(0, BLOCK_Q))[:, None] < S
    O0 = tl.where(out_mask, O0, 0.0)
    O1 = tl.where(out_mask, O1, 0.0)
    
    o_desc.store([bid_b, bid_h, qo, 0], tl.unsqueeze(tl.unsqueeze(O0, 0), 0))
    o_desc.store([bid_b, bid_h, qo, 64], tl.unsqueeze(tl.unsqueeze(O1, 0), 0))
    
    lse = m_i + tl.log(l_i + 1e-30)
    lse = tl.where(q_valid[:, None], lse, float('inf'))
    
    lse_base = bid_b * H * S + bid_h * S + qo
    lse_offs = tl.arange(0, BLOCK_Q)
    tl.store(LSE_ptr + lse_base + lse_offs, lse.squeeze(-1), mask=q_valid)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    scale = 1.0 / (D ** 0.5)
    
    BLOCK_Q = 64
    BLOCK_KV = 64
    
    q_desc = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_Q, 64])
    k_desc = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_KV, 64])
    v_desc = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_KV, 64])
    o_desc = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_Q, 64])
    
    num_blocks = triton.cdiv(S, BLOCK_Q)
    grid = (num_blocks, H, B)
    
    _mha_kernel[grid](
        q_desc, k_desc, v_desc, o_desc, LSE,
        H, S, scale,
        BLOCK_Q=BLOCK_Q, BLOCK_KV=BLOCK_KV,
        num_warps=8, num_stages=2,
    )