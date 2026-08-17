import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_kernel(
    Q_desc, K_desc, V_desc, O_desc, LSE_ptr,
    S, scale,
    B: tl.constexpr, H: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Causal Multi-Head Attention Forward Pass using FlashAttention algorithm.
    Processes a single query block for a specific (batch, head) pair.
    Uses online softmax for numerical stability.
    """
    idx_b_h = tl.program_id(1)
    q_idx = tl.program_id(0)
    
    if q_idx * BLOCK_Q >= S:
        return
    
    # Load Query tiles
    Q0 = Q_desc.load([idx_b_h, q_idx * BLOCK_Q, 0]).squeeze(0)
    Q1 = Q_desc.load([idx_b_h, q_idx * BLOCK_Q, 64]).squeeze(0)
    
    # Determine number of K/V blocks required by the causal lower-triangular mask
    num_k_blocks = min(q_idx + 1, tl.cdiv(S, BLOCK_K))
    
    o0 = tl.zeros((BLOCK_Q, 64), tl.float32)
    o1 = tl.zeros((BLOCK_Q, 64), tl.float32)
    m_old = tl.full((BLOCK_Q,), -float('inf'), tl.float32)
    l_old = tl.full((BLOCK_Q,), 0.0, tl.float32)
    
    row_idx_abs = q_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
    
    for j in range(num_k_blocks):
        # Load Key and Value tiles
        K0 = K_desc.load([idx_b_h, j * BLOCK_K, 0]).squeeze(0)
        K1 = K_desc.load([idx_b_h, j * BLOCK_K, 64]).squeeze(0)
        V0 = K_desc.load([idx_b_h, j * BLOCK_K, 0]).squeeze(0)
        V1 = K_desc.load([idx_b_h, j * BLOCK_K, 64]).squeeze(0)
        
        # GEMM1: S = Q @ K^T * scale
        S_acc = tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)
        S_val = S_acc * scale
        
        # Apply causal and boundary mask safely (-inf fallback zeroes out exp output)
        col_idx_abs = j * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = (row_idx_abs[:, None] >= col_idx_abs[None, :]) & \
               (row_idx_abs[:, None] < S) & \
               (col_idx_abs[None, :] < S)
        S_val = tl.where(mask, S_val, -float('inf'))
        
        # --- Online Softmax ---
        m_local = tl.max(S_val, axis=1)
        m_new = tl.maximum(m_old, m_local)
        
        P = tl.exp(S_val - m_new[:, None])
        l_local = tl.sum(P, axis=1)
        l_new = l_old * tl.exp(m_old - m_new) + l_local
        
        # Rescale accumulated Output and prepare for GEMM2
        o0 = o0 * tl.exp(m_old - m_new)[:, None]
        o1 = o1 * tl.exp(m_old - m_new)[:, None]
        
        # GEMM2: O_acc += P @ V
        o0 = tl.dot(P, V0, o0)
        o1 = tl.dot(P, V1, o1)
        
        m_old = m_new
        l_old = l_new
    
    # Epilogue: Normalize O by the denominator
    inv_l = 1.0 / l_old
    o0 = o0 * inv_l[:, None]
    o1 = o1 * inv_l[:, None]
    
    # Store the final output tiles via TMA
    O_desc.store([idx_b_h, q_idx * BLOCK_Q, 0], o0[None, :, :].to(tl.bfloat16))
    O_desc.store([idx_b_h, q_idx * BLOCK_Q, 64], o1[None, :, :].to(tl.bfloat16))
    
    # Compute and store Log-Sum-Exp (LSE) securely mapping to our known contiguous [B, H, S] layout
    lse = m_old + tl.log(l_old)
    bh_row_offset = idx_b_h * S
    lse_ptr = LSE_ptr + bh_row_offset + row_idx_abs
    tl.store(lse_ptr, lse, mask=row_idx_abs < S)


def run(Q, K, V, O, LSE):
    """
    Host entry point for causal multi-head attention forward pass.
    Applies robust continuous memory traversals mapped onto Hopper TMA.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    scale = 1.0 / (D ** 0.5)
    
    Q_3d = Q.view(B * H, S, D)
    K_3d = K.view(B * H, S, D)
    V_3d = V.view(B * H, S, D)
    O_3d = O.view(B * H, S, D)
    
    # Establish TMA descriptors for coalesced HBM reads/writes
    q_desc = TensorDescriptor.from_tensor(Q_3d, [1, 64, 64])
    k_desc = TensorDescriptor.from_tensor(K_3d, [1, 64, 64])
    v_desc = TensorDescriptor.from_tensor(V_3d, [1, 64, 64])
    o_desc = TensorDescriptor.from_tensor(O_3d, [1, 64, 64])
    
    # Grid maps 1:1 to (query_block, batch_head) pairs
    grid = (triton.cdiv(S, 64), B * H)
    
    _mha_kernel[grid](
        q_desc, k_desc, v_desc, o_desc, LSE,
        S, scale,
        B=B, H=H,
        BLOCK_Q=64, BLOCK_K=64,
        num_warps=4, num_stages=3,
    )