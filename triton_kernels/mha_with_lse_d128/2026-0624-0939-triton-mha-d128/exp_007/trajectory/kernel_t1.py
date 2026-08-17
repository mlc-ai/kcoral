import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_kernel(
    q_desc, k_desc, v_desc, o_desc,
    LSE, 
    seq_offsets,
    H, S, scale,
    BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr, D_CHUNK: tl.constexpr,
):
    bid_q = tl.program_id(0)
    bid_h = tl.program_id(1)
    bid_b = tl.program_id(2)
    
    qo = bid_q * BLOCK_Q
    
    Q0 = q_desc.load([bid_b, bid_h, qo, 0])
    Q1 = q_desc.load([bid_b, bid_h, qo, 64])
    
    num_kv = tl.cdiv(S, BLOCK_KV)
    
    acc_o0 = tl.zeros((BLOCK_Q, D_CHUNK), tl.float32)
    acc_o1 = tl.zeros((BLOCK_Q, D_CHUNK), tl.float32)
    m_i = tl.full((BLOCK_Q,), -float('inf'), tl.float32)
    l_i = tl.full((BLOCK_Q,), 0.0, tl.float32)
    
    for j in range(num_kv):
        ko = j * BLOCK_KV
        
        K0 = k_desc.load([bid_b, bid_h, ko, 0])
        K1 = k_desc.load([bid_b, bid_h, ko, 64])
        
        S_tile = tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)
        S_tile *= scale
        
        col_indices = tl.arange(0, BLOCK_KV)
        k_seq_o = j * BLOCK_KV + col_indices
        k_valid = k_seq_o < S
        S_tile = tl.where(k_valid[None, :], S_tile, -float('inf'))
        
        m_old = m_i
        m_new = tl.maximum(m_i, tl.max(S_tile, axis=1, keep_dims=True))
        
        P = tl.exp(S_tile - m_new)
        
        exp_diff = tl.exp(m_old - m_new)
        l_i = tl.fma(exp_diff, l_i, tl.sum(P, axis=1, keep_dims=True))
        m_i = m_new
        
        V0 = v_desc.load([bid_b, bid_h, ko, 0])
        V1 = v_desc.load([bid_b, bid_h, ko, 64])
        
        acc_o0 = tl.fma(exp_diff, acc_o0, tl.dot(P, V0))
        acc_o1 = tl.fma(exp_diff, acc_o1, tl.dot(P, V1))
    
    O0 = acc_o0 / (l_i + 1e-30)
    O1 = acc_o1 / (l_i + 1e-30)
    
    o_desc.store([bid_b, bid_h, qo, 0], O0.to(tl.bfloat16))
    o_desc.store([bid_b, bid_h, qo, 64], O1.to(tl.bfloat16))
    
    lse = m_i + tl.log(l_i + 1e-30)
    lse_rows = tl.arange(0, BLOCK_Q)
    seq_o = bid_q * BLOCK_Q + lse_rows
    valid_mask = seq_o < S
    lse_base = LSE + seq_offsets[bid_b, bid_h]
    tl.store(lse_base + seq_o, lse.squeeze(-1), mask=valid_mask)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    scale = 1.0 / (D ** 0.5)
    
    BLOCK_Q = 64
    BLOCK_KV = 64
    D_CHUNK = 64
    
    q_desc = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_Q, D_CHUNK])
    k_desc = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_KV, D_CHUNK])
    v_desc = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_KV, D_CHUNK])
    o_desc = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_Q, D_CHUNK])
    
    seq_offsets = torch.empty((B, H), dtype=torch.int64, device=Q.device)
    for b in range(B):
        for h in range(H):
            seq_offsets[b, h] = b * H * S + h * S
    
    grid = (triton.cdiv(S, BLOCK_Q), H, B)
    
    _mha_kernel[grid](
        q_desc, k_desc, v_desc, o_desc, LSE, seq_offsets,
        H, S, scale,
        BLOCK_Q=BLOCK_Q, BLOCK_KV=BLOCK_KV, D_CHUNK=D_CHUNK,
        num_warps=8, num_stages=2,
    )