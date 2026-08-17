import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_fwd_causal(
    Q_desc, K_desc, V_desc, O_desc, LSE_desc,
    S, H, B,
    SCALE,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    D: tl.constexpr,
):
    i = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)
    
    base_offset = batch_idx * H + head_idx
    
    Q_0 = Q_desc.load([base_offset, i * BLOCK_Q, 0])
    Q_1 = Q_desc.load([base_offset, i * BLOCK_Q, D//2])
    
    O_acc_0 = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
    O_acc_1 = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)
    
    m = tl.full((BLOCK_Q,), -float('inf'), dtype=tl.float32)
    l = tl.full((BLOCK_Q,), 0.0, dtype=tl.float32)
    
    ONLINE_SOFTMAX_NINF = -1.0e20
    
    for j in range(i + 1):
        K_0 = K_desc.load([base_offset, j * BLOCK_K, 0])
        K_1 = K_desc.load([base_offset, j * BLOCK_K, D//2])
        
        V_0 = V_desc.load([base_offset, j * BLOCK_K, 0])
        V_1 = V_desc.load([base_offset, j * BLOCK_K, D//2])
        
        acc = tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)
        acc = acc * SCALE
        
        if j == i:
            row = tl.arange(0, BLOCK_Q)
            col = tl.arange(0, BLOCK_K)
            causal_mask = (row[:, None] >= col[None, :])
            acc = tl.where(causal_mask, acc, ONLINE_SOFTMAX_NINF)
        
        m_old = m
        m_curr = tl.max(acc, axis=1)
        m = tl.maximum(m_old, m_curr)
        
        exp_scale = tl.exp(m_old - m)
        P = tl.exp(acc - m)
        
        l_curr = tl.sum(P, axis=1)
        l = l * exp_scale + l_curr
        
        O_acc_0 = O_acc_0 * exp_scale[:, None]
        O_acc_1 = O_acc_1 * exp_scale[:, None]
        
        O_acc_0 = O_acc_0 + tl.dot(P, V_0)
        O_acc_1 = O_acc_1 + tl.dot(P, V_1)
        
    inv_l = 1.0 / l
    O_0 = (O_acc_0 * inv_l[:, None]).to(tl.bfloat16)
    O_1 = (O_acc_1 * inv_l[:, None]).to(tl.bfloat16)
    
    L_val = (m + tl.log(l))[:, None]
    
    O_desc.store([base_offset, i * BLOCK_Q, 0], O_0)
    O_desc.store([base_offset, i * BLOCK_Q, D//2], O_1)
    LSE_desc.store([base_offset, i * BLOCK_Q], L_val)


def run(Q, K, V, O, LSE):
    """Compute Causal MHA Forward Pass with LSE output."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    BLOCK_Q = 64
    BLOCK_K = 64
    
    Q_desc = TensorDescriptor.from_tensor(Q, [BLOCK_Q, D])
    K_desc = TensorDescriptor.from_tensor(K, [BLOCK_K, D])
    V_desc = TensorDescriptor.from_tensor(V, [BLOCK_K, D])
    O_desc = TensorDescriptor.from_tensor(O, [BLOCK_Q, D])
    LSE_desc = TensorDescriptor.from_tensor(LSE, [BLOCK_Q, 1])
    
    scale = 1.0 / (D ** 0.5)
    
    grid = (triton.cdiv(S, BLOCK_Q), H, B)
    
    _mha_fwd_causal[grid](
        Q_desc, K_desc, V_desc, O_desc, LSE_desc,
        S, H, B,
        scale,
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
        D=D,
        num_warps=4,
        num_stages=3,
    )