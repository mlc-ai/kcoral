import torch
import triton
import triton.language as tl
import math
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_kernel_forward(
    Q_desc, K_desc, V_desc, O_desc, LSE_desc,
    S_len, sqrt_d,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    
    start_seq_q = pid_m * BLOCK_M
    n_blocks = tl.cdiv(S_len, BLOCK_N)
    
    q = Q_desc.load([b, h, start_seq_q, 0])
    
    o_acc = tl.zeros((BLOCK_M, D), tl.float32)
    m = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    for j in range(n_blocks):
        seq_idx = j * BLOCK_N
        
        k = K_desc.load([b, h, seq_idx, 0])
        v = V_desc.load([b, h, seq_idx, 0])
        
        s_acc = tl.dot(q, k.T)
        s_acc /= sqrt_d
        
        col_offsets = seq_idx + tl.arange(0, BLOCK_N)
        mask_k = col_offsets < S_len
        s_acc = tl.where(mask_k[None, :], s_acc, -float('inf'))
        
        m_curr = tl.max(s_acc, axis=1)
        m_new = tl.maximum(m, m_curr)
        scale = tl.exp(m - m_new)
        
        p = tl.exp(s_acc - m_new[:, None])
        l_curr = tl.sum(p, axis=1)
        l = l * scale + l_curr
        
        m = m_new
        
        o_acc = o_acc * scale[:, None] + tl.dot(p, v)
    
    seq_idx_q = start_seq_q + tl.arange(0, BLOCK_M)
    mask_m = seq_idx_q < S_len
    
    l_safe = tl.maximum(l, 1e-30)
    o_acc = o_acc / l_safe[:, None]
    lse = m + tl.log(l_safe)
    
    O_bf16 = o_acc.to(tl.bfloat16)
    O_desc.store([b, h, start_seq_q, 0], O_bf16)
    LSE_desc.store([b, h, start_seq_q], lse)


BLOCK_M = 64
BLOCK_N = 64


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S_len, D = Q.shape
    
    Q_desc = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M, D])
    K_desc = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N, D])
    V_desc = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N, D])
    O_desc = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M, D])
    LSE_desc = TensorDescriptor.from_tensor(LSE, [1, 1, BLOCK_M])
    
    grid = (triton.cdiv(S_len, BLOCK_M), H, B)
    sqrt_d = math.sqrt(D)
    
    _mha_kernel_forward[grid](
        Q_desc, K_desc, V_desc, O_desc, LSE_desc,
        S_len, sqrt_d,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D,
        num_warps=4, num_stages=4
    )