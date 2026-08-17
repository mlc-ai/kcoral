import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_kernel(
    *descs,
    S, H, LSE,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    
    idx = b * H + h
    desc_idx = idx * 4
    desc_q0 = descs[desc_idx]
    desc_k0 = descs[desc_idx + 1]
    desc_v0 = descs[desc_idx + 2]
    desc_o0 = descs[desc_idx + 3]
    
    start_n = pid_m * BLOCK_M
    
    # Load Q once; logically [BLOCK_M, 128] but physically as two D=64 tiles
    Q_0_0 = desc_q0.load([start_n, 0])  
    Q_0_1 = desc_q0.load([start_n, 64])  
    
    m_i = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    O_acc_0 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    O_acc_1 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    
    scale = tl.rsqrt(128.0) 
    
    num_blocks_k = tl.cdiv(S, BLOCK_N)
    
    for j in range(num_blocks_k):
        k_idx = j * BLOCK_N
        
        K_j_0 = desc_k0.load([k_idx, 0])  
        K_j_1 = desc_k0.load([k_idx, 64])
        V_j_0 = desc_v0.load([k_idx, 0])  
        V_j_1 = desc_v0.load([k_idx, 64])
        
        S_tile = (tl.dot(Q_0_0, K_j_0.T) + tl.dot(Q_0_1, K_j_1.T)) * scale 
        
        m_local = tl.max(S_tile, axis=1)
        m_new = tl.maximum(m_i, m_local)
        
        exp_old = tl.exp(m_i - m_new)
        exp_old = tl.where(m_i == -float('inf'), 1.0, exp_old)
        
        P = tl.exp(S_tile - m_new[:, None])
        
        l_local = tl.sum(P, axis=1)
        l_i = exp_old * l_i + l_local
        
        exp_old_col = exp_old[:, None] 
        O_acc_0 = O_acc_0 * exp_old_col + tl.dot(P, V_j_0)
        O_acc_1 = O_acc_1 * exp_old_col + tl.dot(P, V_j_1)
        
        m_i = m_new
    
    safe_l_i = tl.where(l_i > 0, l_i, 1.0)
    O_acc_0 = O_acc_0 / safe_l_i[:, None]
    O_acc_1 = O_acc_1 / safe_l_i[:, None]
    
    lse = m_i + tl.log(safe_l_i)
    
    o_ptr_0 = desc_o0.store([start_n, 0], O_acc_0)
    o_ptr_1 = desc_o0.store([start_n, 64], O_acc_1)
    
    lse_offsets = start_n + tl.arange(0, BLOCK_M)
    lse_ptr = LSE + (b * H + h) * S + lse_offsets
    mask_lse = lse_offsets < S
    tl.store(lse_ptr, lse, mask=mask_lse)


def run(Q, K, V, O, LSE):
    """Compute non-causal multi-head attention with LSE output."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    # Allocate expected outputs matching reference tensor shapes
    LSE = torch.empty((B, H, S), dtype=torch.float32, device=Q.device)

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
            
            Q_descs.append(TensorDescriptor.from_tensor(q_ptr_base, [block_m, 64]))
            K_descs.append(TensorDescriptor.from_tensor(k_ptr_base, [block_n, 64]))
            V_descs.append(TensorDescriptor.from_tensor(v_ptr_base, [block_n, 64]))
            O_descs.append(TensorDescriptor.from_tensor(o_ptr_base, [block_m, 64]))
            
    grid = (triton.cdiv(S, block_m), B, H)
    _mha_kernel[grid](
        *Q_descs, *K_descs, *V_descs, *O_descs,
        S, H, LSE,
        BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=4, num_stages=3,
    )