import torch
import triton
import triton.language as tl


@triton.jit
def _mha_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    H, S, scale,
    BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr, D: tl.constexpr,
):
    bid_q = tl.program_id(0)
    bid_h = tl.program_id(1)
    bid_b = tl.program_id(2)
    
    qo = bid_q * BLOCK_Q
    
    base_offset = bid_b * H * S * D + bid_h * S * D
    
    q_offs = tl.arange(0, BLOCK_Q)
    k_offs = tl.arange(0, BLOCK_KV)
    
    d_offs_0 = tl.arange(0, 64)
    d_offs_1 = tl.arange(0, 64) + 64
    
    Q0_ptrs = Q_ptr + base_offset + (qo + q_offs)[:, None] * D + d_offs_0[None, :]
    Q1_ptrs = Q_ptr + base_offset + (qo + q_offs)[:, None] * D + d_offs_1[None, :]
    
    q_valid = (qo + q_offs) < S
    Q0 = tl.load(Q0_ptrs, mask=q_valid[:, None], other=0.0)
    Q1 = tl.load(Q1_ptrs, mask=q_valid[:, None], other=0.0)
    
    num_kv = tl.cdiv(S, BLOCK_KV)
    
    acc_o0 = tl.zeros((BLOCK_Q, 64), tl.float32)
    acc_o1 = tl.zeros((BLOCK_Q, 64), tl.float32)
    m_i = tl.full((BLOCK_Q, 1), -float('inf'), tl.float32)
    l_i = tl.full((BLOCK_Q, 1), 0.0, tl.float32)
    
    for j in range(num_kv):
        ko = j * BLOCK_KV
        
        K0_ptrs = K_ptr + base_offset + (ko + k_offs)[:, None] * D + d_offs_0[None, :]
        K1_ptrs = K_ptr + base_offset + (ko + k_offs)[:, None] * D + d_offs_1[None, :]
        
        K0 = tl.load(K0_ptrs, mask=(ko + k_offs)[:, None] < S, other=0.0)
        K1 = tl.load(K1_ptrs, mask=(ko + k_offs)[:, None] < S, other=0.0)
        
        S_tile = tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)
        S_tile *= scale
        
        k_valid = (ko + k_offs) < S
        S_tile = tl.where(k_valid[None, :], S_tile, -float('inf'))
        
        m_old = m_i
        m_new = tl.maximum(m_i, tl.max(S_tile, axis=1, keep_dims=True))
        
        P = tl.exp(S_tile - m_new)
        all_invalid = ~q_valid
        P = tl.where(all_invalid[:, None], 0.0, P)
        
        exp_diff = tl.where(m_new > -float('inf'), tl.exp(m_old - m_new), 0.0)
        l_i = tl.fma(exp_diff, l_i, tl.sum(P, axis=1, keep_dims=True))
        m_i = m_new
        
        V0_ptrs = V_ptr + base_offset + (ko + k_offs)[:, None] * D + d_offs_0[None, :]
        V1_ptrs = V_ptr + base_offset + (ko + k_offs)[:, None] * D + d_offs_1[None, :]
        
        V0 = tl.load(V0_ptrs, mask=(ko + k_offs)[:, None] < S, other=0.0)
        V1 = tl.load(V1_ptrs, mask=(ko + k_offs)[:, None] < S, other=0.0)
        
        acc_o0 = tl.fma(exp_diff, acc_o0, tl.dot(P, V0.T))
        acc_o1 = tl.fma(exp_diff, acc_o1, tl.dot(P, V1.T))
    
    inv_l = 1.0 / (l_i + 1e-30)
    O0 = (acc_o0 * inv_l).to(tl.bfloat16)
    O1 = (acc_o1 * inv_l).to(tl.bfloat16)
    
    out_mask = (qo + q_offs)[:, None] < S
    O0 = tl.where(out_mask, O0, 0.0)
    O1 = tl.where(out_mask, O1, 0.0)
    
    O0_ptrs = O_ptr + base_offset + (qo + q_offs)[:, None] * D + d_offs_0[None, :]
    O1_ptrs = O_ptr + base_offset + (qo + q_offs)[:, None] * D + d_offs_1[None, :]
    
    tl.store(O0_ptrs, O0, mask=out_mask)
    tl.store(O1_ptrs, O1, mask=out_mask)
    
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
    
    num_blocks = triton.cdiv(S, BLOCK_Q)
    grid = (num_blocks, H, B)
    
    _mha_kernel[grid](
        Q, K, V, O, LSE,
        H, S, scale,
        BLOCK_Q=BLOCK_Q, BLOCK_KV=BLOCK_KV, D=D,
        num_warps=8, num_stages=3,
    )