import torch
import triton
import triton.language as tl
import math
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_kernel_forward(
    Q_desc, K_desc, V_desc,
    O, LSE,
    B, H, S_len, sqrt_d,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    
    start_seq_q = pid_m * BLOCK_M
    n_blocks = tl.cdiv(S_len, BLOCK_N)
    
    q_0 = Q_desc.load([b, h, start_seq_q, 0])
    q_1 = Q_desc.load([b, h, start_seq_q, 64])
    
    o_0_acc = tl.zeros((BLOCK_M, 64), tl.float32)
    o_1_acc = tl.zeros((BLOCK_M, 64), tl.float32)
    m = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    for j in range(n_blocks):
        seq_idx = j * BLOCK_N
        
        k_0 = K_desc.load([b, h, seq_idx, 0])
        k_1 = K_desc.load([b, h, seq_idx, 64])
        v_0 = V_desc.load([b, h, seq_idx, 0])
        v_1 = V_desc.load([b, h, seq_idx, 64])
        
        s_acc = tl.dot(q_0, k_0.T)
        s_acc += tl.dot(q_1, k_1.T)
        s_acc /= sqrt_d
        
        col_offsets = seq_idx + tl.arange(0, BLOCK_N)
        mask_k = col_offsets < S_len
        s_acc = tl.where(mask_k[:, None], s_acc, -float('inf'))
        
        m_curr = tl.max(s_acc, axis=1)
        m_new = tl.maximum(m, m_curr)
        scale = tl.exp(m - m_new)
        
        p = tl.exp(s_acc - m_new[:, None])
        l_curr = tl.sum(p, axis=1)
        l = l * scale + l_curr
        
        m = m_new
        
        o_0_acc = o_0_acc * scale[:, None] + tl.dot(p, v_0)
        o_1_acc = o_1_acc * scale[:, None] + tl.dot(p, v_1)
    
    seq_idx_q = start_seq_q + tl.arange(0, BLOCK_M)
    mask_m = seq_idx_q < S_len
    
    o_0_acc = tl.where(l[:, None] > 0, o_0_acc / l[:, None], 0.0)
    o_1_acc = tl.where(l[:, None] > 0, o_1_acc / l[:, None], 0.0)
    lse = tl.where(mask_m, m + tl.log(l), float('-inf'))
    
    out_s = O + b * H * S_len * D + h * S_len * D
    col = tl.arange(0, 64)
    out_ptr0 = out_s + seq_idx_q[:, None] * D + col[None, :]
    out_ptr1 = out_s + seq_idx_q[:, None] * D + col[None, :] + 64
    
    tl.store(out_ptr0, o_0_acc.to(tl.bfloat16), mask=mask_m[:, None])
    tl.store(out_ptr1, o_1_acc.to(tl.bfloat16), mask=mask_m[:, None])
    
    lse_s = LSE + b * H * S_len + h * S_len
    lse_ptr = lse_s + seq_idx_q
    tl.store(lse_ptr, lse, mask=mask_m)


BLOCK_M = 64
BLOCK_N = 64


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S_len, D = Q.shape
    
    Q_desc = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M, 64])
    K_desc = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N, 64])
    V_desc = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N, 64])
    
    grid = (triton.cdiv(S_len, BLOCK_M), H, B)
    sqrt_d = math.sqrt(D)
    
    _mha_kernel_forward[grid](
        Q_desc, K_desc, V_desc, O, LSE,
        B, H, S_len, sqrt_d,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D,
        num_warps=4, num_stages=4
    )