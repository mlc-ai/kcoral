import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_fwd_causal(
    Q_desc, K_desc, V_desc, O_desc, LSE_desc,
    S, H, B,
    SCALE,
    NUM_SMS: tl.constexpr,
    BLOCK_Q: tl.constexpr = 64,
    D: tl.constexpr = 128,
):
    start_pid = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)
    
    num_pid_m = tl.cdiv(S, BLOCK_Q)
    
    for m_idx in range(start_pid, num_pid_m, NUM_SMS):
        Q = Q_desc.load([batch_idx, head_idx, m_idx * BLOCK_Q, 0])
        
        O_acc = tl.zeros((BLOCK_Q, D), dtype=tl.float32)
        
        m = tl.full((BLOCK_Q,), -float('inf'), dtype=tl.float32)
        l = tl.full((BLOCK_Q,), 0.0, dtype=tl.float32)
        
        q_seq = (m_idx * BLOCK_Q) + tl.arange(0, BLOCK_Q)
        
        for j in range(m_idx + 1):
            K = K_desc.load([batch_idx, head_idx, j * BLOCK_Q, 0])
            V = V_desc.load([batch_idx, head_idx, j * BLOCK_Q, 0])
            
            acc = tl.dot(Q, K.T) 
            acc = acc * SCALE
            
            k_seq = (j * BLOCK_Q) + tl.arange(0, BLOCK_Q)
            causal_mask = (q_seq[:, None] >= k_seq[None, :]) & (q_seq[:, None] < S) & (k_seq[None, :] < S)
            acc = tl.where(causal_mask, acc, -1.0e20)
            
            m_old = m
            m_curr = tl.max(acc, axis=1)
            m = tl.maximum(m_old, m_curr)
            
            exp_scale = tl.exp(m_old - m)
            P = tl.exp(acc - m)
            P = tl.where(causal_mask, P, 0.0)
            
            l_curr = tl.sum(P, axis=1)
            l = l * exp_scale + l_curr
            
            O_acc = O_acc * exp_scale[:, None] + tl.dot(P, V)
            
        inv_l = 1.0 / l
        O = (O_acc * inv_l[:, None]).to(tl.bfloat16)
        O_desc.store([batch_idx, head_idx, m_idx * BLOCK_Q, 0], O)
        
        L_val = m + tl.log(l)
        LSE_desc.store([batch_idx, head_idx, m_idx * BLOCK_Q, 0], L_val[:, None])


def run(Q, K, V, O, LSE):
    """Compute Causal MHA Forward Pass with LSE output."""
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)
    triton.set_allocator(alloc_fn)
    
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    Q_desc = TensorDescriptor.from_tensor(Q, [B, H, S, D], padding_option="zero")
    K_desc = TensorDescriptor.from_tensor(K, [B, H, S, D], padding_option="zero")
    V_desc = TensorDescriptor.from_tensor(V, [B, H, S, D], padding_option="zero")
    O_desc = TensorDescriptor.from_tensor(O, [B, H, S, D])
    LSE_desc = TensorDescriptor.from_tensor(LSE, [B, H, S, 1], padding_option="zero")
    
    scale = 1.0 / (D ** 0.5)
    NUM_SMS = 132 
    
    grid = (min(NUM_SMS, triton.cdiv(S, 64)), H, B)
    
    _mha_fwd_causal[grid](
        Q_desc, K_desc, V_desc, O_desc, LSE_desc,
        S, H, B,
        scale,
        NUM_SMS=NUM_SMS,
        num_warps=4,
        num_stages=4,
    )