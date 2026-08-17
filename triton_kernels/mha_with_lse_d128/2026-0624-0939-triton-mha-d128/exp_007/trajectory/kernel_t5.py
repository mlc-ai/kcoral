import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_kernel(
    q_desc, k_desc, v_desc, o_desc,
    LSE_ptr, 
    H, S, scale,
    BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr, D_CHUNK: tl.constexpr,
):
    bid_q = tl.program_id(0)
    bid_h = tl.program_id(1)
    bid_b = tl.program_id(2)
    
    qo = bid_q * BLOCK_Q
    bh_idx = bid_b * H + bid_h
    q_row_base = bh_idx * S + qo
    
    Q0 = q_desc.load([q_row_base, 0])
    Q1 = q_desc.load([q_row_base, D_CHUNK])
    
    num_kv = tl.cdiv(S, BLOCK_KV)
    
    acc_o0 = tl.zeros((BLOCK_Q, D_CHUNK), tl.float32)
    acc_o1 = tl.zeros((BLOCK_Q, D_CHUNK), tl.float32)
    m_i = tl.full((BLOCK_Q, 1), -float('inf'), tl.float32)
    l_i = tl.full((BLOCK_Q, 1), 0.0, tl.float32)
    
    q_seq_o = bid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_valid = q_seq_o < S
    
    col_indices = tl.arange(0, BLOCK_KV)
    
    for j in range(num_kv):
        ko = j * BLOCK_KV
        k_row_base = bh_idx * S + ko
        
        K0 = k_desc.load([k_row_base, 0])
        K1 = k_desc.load([k_row_base, D_CHUNK])
        
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
        
        V0 = v_desc.load([k_row_base, 0])
        V1 = v_desc.load([k_row_base, D_CHUNK])
        
        P_bf16 = P.to(tl.bfloat16)
        acc_o0 = tl.fma(exp_diff, acc_o0, tl.dot(P_bf16, V0))
        acc_o1 = tl.fma(exp_diff, acc_o1, tl.dot(P_bf16, V1))
    
    inv_l = 1.0 / (l_i + 1e-30)
    O0 = (acc_o0 * inv_l).to(tl.bfloat16)
    O1 = (acc_o1 * inv_l).to(tl.bfloat16)
    
    o_desc.store([q_row_base, 0], O0)
    o_desc.store([q_row_base, D_CHUNK], O1)
    
    lse = m_i + tl.log(l_i + 1e-30)
    
    row_indices = tl.arange(0, BLOCK_Q)
    valid_mask_lse = q_seq_o < S
    lse_base = bid_b * H * S + bid_h * S + qo
    tl.store(LSE_ptr + lse_base + row_indices, lse.squeeze(-1), mask=valid_mask_lse)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    scale = 1.0 / (D ** 0.5)
    
    BLOCK_Q = 64
    BLOCK_KV = 64
    D_CHUNK = 64
    
    Q_flat = Q.view(B * H * S, D)
    K_flat = K.view(B * H * S, D)
    V_flat = V.view(B * H * S, D)
    O_flat = O.view(B * H * S, D)
    
    q_desc = TensorDescriptor.from_tensor(Q_flat, [BLOCK_Q, D_CHUNK])
    k_desc = TensorDescriptor.from_tensor(K_flat, [BLOCK_KV, D_CHUNK])
    v_desc = TensorDescriptor.from_tensor(V_flat, [BLOCK_KV, D_CHUNK])
    o_desc = TensorDescriptor.from_tensor(O_flat, [BLOCK_Q, D_CHUNK])
    
    num_blocks = triton.cdiv(S, BLOCK_Q)
    grid = (num_blocks, H, B)
    
    _mha_kernel[grid](
        q_desc, k_desc, v_desc, o_desc, LSE,
        H, S, scale,
        BLOCK_Q=BLOCK_Q, BLOCK_KV=BLOCK_KV, D_CHUNK=D_CHUNK,
        num_warps=8, num_stages=2,
    )