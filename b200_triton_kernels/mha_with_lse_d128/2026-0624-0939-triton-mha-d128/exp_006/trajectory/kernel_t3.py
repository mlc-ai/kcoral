import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_kernel(
    desc_Q, desc_K, desc_V, desc_O, LSE,
    S, H, D,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    
    idx = b * H + h
    start_n = pid_m * BLOCK_M
    rowOffsets = start_n + tl.arange(0, BLOCK_M)
    
    # Load Q once onto-chip; reused invariantly for every K iteration.
    Q_0 = desc_Q.load([start_n, 0])  
    
    m_i = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    O_acc = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    
    scale = 1.0 / (D ** 0.5) 
    
    num_blocks_k = tl.cdiv(S, BLOCK_N)
    
    for j in range(num_blocks_k):
        k_idx = j * BLOCK_N
        
        K_j = desc_K.load([k_idx, 0]).to(tl.float32)  
        V_j = desc_V.load([k_idx, 0]).to(tl.float32)
        
        S_tile = tl.dot(Q_0, K_j.T) * scale 
        
        m_local = tl.max(S_tile, axis=1)
        m_new = tl.maximum(m_i, m_local)
        
        exp_old = tl.exp(m_i - m_new)
        P = tl.exp(S_tile - m_new[:, None])
        
        l_local = tl.sum(P, axis=1)
        l_i = exp_old * l_i + l_local
        
        O_acc = O_acc * exp_old[:, None] + tl.dot(P, V_j)
        
        m_i = m_new
    
    safe_l_i = tl.where(l_i > 0, l_i, 1.0)
    O_acc = O_acc / safe_l_i[:, None]
    
    mask_Q = rowOffsets[:, None] < S
    O_acc = tl.where(mask_Q, O_acc, 0.0)
    
    desc_O.store([start_n, 0], O_acc.to(tl.bfloat16))
    
    lse = m_i + tl.log(safe_l_i)
    
    lse_ptr = LSE + idx * S + rowOffsets
    mask_lse = rowOffsets < S
    tl.store(lse_ptr, lse, mask=mask_lse)


def run(Q, K, V, O, LSE):
    """Compute non-causal multi-head attention forward returning O and LSE."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    block_m = 64
    block_n = 64
    
    Q_descs = []
    K_descs = []
    V_descs = []
    O_descs = []
    
    for b in range(B):
        for h in range(H):
            q_ptr_base = Q + (b * H + h) * S * D
            k_ptr_base = K + (b * H + h) * S * D
            v_ptr_base = V + (b * H + h) * S * D
            o_ptr_base = O + (b * H + h) * S * D
            
            Q_descs.append(TensorDescriptor.from_tensor(q_ptr_base, [block_m, D]))
            K_descs.append(TensorDescriptor.from_tensor(k_ptr_base, [block_n, D]))
            V_descs.append(TensorDescriptor.from_tensor(v_ptr_base, [block_n, D]))
            O_descs.append(TensorDescriptor.from_tensor(o_ptr_base, [block_m, D]))
            
    grid = (triton.cdiv(S, block_m), B, H)
    _mha_kernel[grid](
        *Q_descs, *K_descs, *V_descs, *O_descs, LSE,
        S, H, D,
        BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=4, num_stages=3,
    )