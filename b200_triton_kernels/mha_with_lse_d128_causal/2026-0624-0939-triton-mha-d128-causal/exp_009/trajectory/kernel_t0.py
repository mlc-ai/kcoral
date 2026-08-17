import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_kernel(
    Q_desc, K_desc, V_desc, O_desc, LSE_ptr,
    B, H, S, D,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Causal Multi-Head Attention Forward Pass using FlashAttention algorithm.
    
    Processes a single query block for a specific (batch, head) pair.
    Uses online softmax for numerical stability.
    """
    # Decode batch/head and query block index from program IDs
    pid_b_h = tl.program_id(1)
    b = pid_b_h // H
    h = pid_b_h % H
    q_idx = tl.program_id(0)
    
    # Early exit if this query block is entirely out of bounds
    if q_idx * BLOCK_Q >= S:
        return
    
    # Compute the flat row offset for this (b, h) pair in the 2D descriptor view
    bh_offset = b * H * S + h * S
    
    # Load the Query tile [BLOCK_Q, D] via TMA
    Q = Q_desc.load([bh_offset + q_idx * BLOCK_Q, 0])
    
    # Causal constraint: only iterate over key blocks up to the diagonal boundary
    num_k_blocks = (min((q_idx + 1) * BLOCK_Q, S) + BLOCK_K - 1) // BLOCK_K
    
    # Initialize accumulation states
    acc_O = tl.zeros((BLOCK_Q, D), tl.float32)
    m_old = tl.full((BLOCK_Q,), -float('inf'), tl.float32)
    l_old = tl.full((BLOCK_Q,), 0.0, tl.float32)
    
    scale = 1.0 / (D ** 0.5)
    row_idx_abs = q_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
    
    for j in range(num_k_blocks):
        # Load Key and Value tiles [BLOCK_K, D] via TMA
        K = K_desc.load([bh_offset + j * BLOCK_K, 0])
        V = V_desc.load([bh_offset + j * BLOCK_K, 0])
        
        # GEMM1: S = Q @ K^T * scale  -> [BLOCK_Q, BLOCK_K]
        S_scores = tl.dot(Q, K.T) * scale
        
        # Apply causal and boundary mask
        col_idx_abs = j * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = (row_idx_abs[:, None] >= col_idx_abs[None, :]) & \
               (row_idx_abs[:, None] < S) & \
               (col_idx_abs[None, :] < S)
        S_scores = tl.where(mask, S_scores, -float('inf'))
        
        # --- Online Softmax ---
        m_local = tl.max(S_scores, axis=1)
        m_new = tl.maximum(m_old, m_local)
        
        P = tl.exp(S_scores - m_new[:, None])
        
        l_local = tl.sum(P, axis=1)
        l_new = l_old * tl.exp(m_old - m_new) + l_local
        
        # Rescale accumulated Output and prepare for GEMM2
        acc_O = acc_O * tl.exp(m_old - m_new)[:, None]
        
        # GEMM2: O_acc += P @ V
        acc_O = tl.dot(P, V, acc_O)
        
        m_old = m_new
        l_old = l_new
    
    # Epilogue: Normalize O by the denominator
    inv_l = 1.0 / l_old
    acc_O = acc_O * inv_l[:, None]
    
    # Store the final output tile via TMA
    O_desc.store([bh_offset + q_idx * BLOCK_Q, 0], acc_O.to(tl.bfloat16))
    
    # Compute and store Log-Sum-Exp (LSE) using standard host-pointer store
    lse = m_old + tl.log(l_old)
    lse_ptr = LSE_ptr + bh_offset + row_idx_abs
    tl.store(lse_ptr, lse, mask=row_idx_abs < S)


def run(Q, K, V, O, LSE):
    """
    Host entry point for causal multi-head attention forward pass.
    
    Args:
        Q: Query tensor [B, H, S, D], bf16
        K: Key tensor [B, H, S, D], bf16
        V: Value tensor [B, H, S, D], bf16
        O: Preallocated output tensor [B, H, S, D], bf16
        LSE: Preallocated log-sum-exp tensor [B, H, S], float32
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    # Construct 2D views for continuous memory traversals matching TMA descriptors
    Q_2d = Q.view(B * H * S, D)
    K_2d = K.view(B * H * S, D)
    V_2d = V.view(B * H * S, D)
    O_2d = O.view(B * H * S, D)
    
    # Establish TMA descriptors for coalesced HBM reads/writes
    q_desc = TensorDescriptor.from_tensor(Q_2d, [64, D])
    k_desc = TensorDescriptor.from_tensor(K_2d, [64, D])
    v_desc = TensorDescriptor.from_tensor(V_2d, [64, D])
    o_desc = TensorDescriptor.from_tensor(O_2d, [64, D])
    
    # Grid maps 1:1 to (query_block, batch_head) pairs
    grid = (triton.cdiv(S, 64), B * H)
    
    _mha_kernel[grid](
        q_desc, k_desc, v_desc, o_desc, LSE,
        B, H, S, D,
        BLOCK_Q=64, BLOCK_K=64,
        num_warps=4, num_stages=2,
    )