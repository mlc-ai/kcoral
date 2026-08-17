import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_kernel(
    Q_desc, K_desc, V_desc, O_desc, LSE_ptr,
    S,
    B: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Causal Multi-Head Attention Forward Pass using FlashAttention algorithm.
    Processes a single query block for a specific (batch, head) pair.
    Uses online softmax for numerical stability and 3D TMA descriptors.
    """
    idx_b_h = tl.program_id(1)
    q_idx = tl.program_id(0)
    
    if q_idx * BLOCK_Q >= S:
        return
    
    # Load the Query tiles [1, 64, 64] -> [64, 64]
    Q0 = Q_desc.load([idx_b_h, q_idx * BLOCK_Q, 0])
    q0 = Q0.squeeze(0)
    Q1 = Q_desc.load([idx_b_h, q_idx * BLOCK_Q, 64])
    q1 = Q1.squeeze(0)
    
    # Causal constraint restricts key block iterations to the lower triangle boundary
    num_k_blocks = min(q_idx + 1, tl.cdiv(S, BLOCK_K))
    
    # Initialize accumulation states
    o0 = tl.zeros((BLOCK_Q, 64), tl.float32)
    o1 = tl.zeros((BLOCK_Q, 64), tl.float32)
    m_old = tl.full((BLOCK_Q,), -float('inf'), tl.float32)
    l_old = tl.full((BLOCK_Q,), 0.0, tl.float32)
    
    scale = 1.0 / (D ** 0.5)
    row_idx_abs = q_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
    
    for j in range(num_k_blocks):
        # Load Key and Value tiles [1, 64, 64] -> [64, 64]
        K0 = K_desc.load([idx_b_h, j * BLOCK_K, 0])
        k0 = K0.squeeze(0)
        K1 = K_desc.load([idx_b_h, j * BLOCK_K, 64])
        k1 = K1.squeeze(0)
        V0 = V_desc.load([idx_b_h, j * BLOCK_K, 0])
        v0 = V0.squeeze(0)
        V1 = V_desc.load([idx_b_h, j * BLOCK_K, 64])
        v1 = V1.squeeze(0)
        
        # GEMM1: S = Q @ K^T * scale  -> [64, 64]
        S_acc = tl.zeros((BLOCK_Q, BLOCK_K), tl.float32)
        S_acc = tl.dot(q0, k0.T, S_acc)
        S_acc = tl.dot(q1, k1.T, S_acc)
        S = S_acc * scale
        
        # Apply causal and boundary mask
        col_idx_abs = j * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = (row_idx_abs[:, None] >= col_idx_abs[None, :]) & \
               (row_idx_abs[:, None] < S) & \
               (col_idx_abs[None, :] < S)
        S = tl.where(mask, S, -float('inf'))
        
        # --- Online Softmax ---
        m_local = tl.max(S, axis=1)
        m_new = tl.maximum(m_old, m_local)
        
        P = tl.exp(S - m_new[:, None])
        
        l_local = tl.sum(P, axis=1)
        l_new = l_old * tl.exp(m_old - m_new) + l_local
        
        # Rescale accumulated Output and prepare for GEMM2
        o0 = o0 * tl.exp(m_old - m_new)[:, None]
        o1 = o1 * tl.exp(m_old - m_new)[:, None]
        
        # GEMM2: O_acc += P @ V
        o0 = tl.dot(P, v0, o0)
        o1 = tl.dot(P, v1, o1)
        
        m_old = m_new
        l_old = l_new
    
    # Epilogue: Normalize O by the denominator
    inv_l = 1.0 / l_old
    o0 = o0 * inv_l[:, None]
    o1 = o1 * inv_l[:, None]
    
    # Store the final output tiles via TMA
    O_desc.store([idx_b_h, q_idx * BLOCK_Q, 0], o0[None, :, :].to(tl.bfloat16))
    O_desc.store([idx_b_h, q_idx * BLOCK_Q, 64], o1[None, :, :].to(tl.bfloat16))
    
    # Compute and store Log-Sum-Exp (LSE)
    lse = m_old + tl.log(l_old)
    tl.store(LSE_ptr + idx_b_h * S + row_idx_abs, lse, mask=row_idx_abs < S)


def run(Q, K, V, O, LSE):
    """
    Host entry point for causal multi-head attention forward pass.
    Uses robust 3D continuous memory traversals mapped perfectly onto TMA block loads.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    # Flatten Batch/Heads cleanly into a single logical dimension for the 3D traversal
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
        S,
        B=B, H=H, D=D,
        BLOCK_Q=64, BLOCK_K=64,
        num_warps=4, num_stages=3,
    )