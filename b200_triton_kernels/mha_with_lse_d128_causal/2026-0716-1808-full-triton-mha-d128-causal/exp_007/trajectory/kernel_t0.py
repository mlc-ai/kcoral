import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _attention_kernel(
    q_desc, k_desc, v_desc,
    O_ptr, LSE_ptr,
    S_len,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    bh_id = tl.program_id(0)
    q_blk = tl.program_id(1)
    
    scale = 1.0 / (128.0 ** 0.5)
    
    Q = q_desc.load([bh_id * S_len + q_blk * BLOCK_M, 0])
    
    O_acc = tl.zeros((BLOCK_M, BLOCK_K), tl.float32)
    m = tl.full((BLOCK_M,), -1e38, tl.float32)
    l = tl.full((BLOCK_M,), 0.0, tl.float32)
    
    for k_blk in range(q_blk + 1):
        K = k_desc.load([bh_id * S_len + k_blk * BLOCK_N, 0])
        V = v_desc.load([bh_id * S_len + k_blk * BLOCK_N, 0])
        
        S = tl.dot(Q, K.T) * scale
        
        if k_blk == q_blk:
            r_idx = (q_blk * BLOCK_M + tl.arange(0, BLOCK_M))
            c_idx = (k_blk * BLOCK_N + tl.arange(0, BLOCK_N))
            valid = c_idx[None, :] <= r_idx[:, None]
            S = tl.where(valid, S, -1e38)
        
        m_prev = m
        m = tl.maximum(m, tl.max(S, axis=1))
        
        P = tl.exp(S - m)
        l = l * tl.exp(m_prev - m) + tl.sum(P, axis=1)
        
        O_acc *= tl.exp(m_prev - m)[:, None]
        O_acc = tl.dot(P, V, O_acc)
    
    O_acc = O_acc / l[:, None]
    lse_val = m + tl.log(l)
    
    valid_q = (q_blk * BLOCK_M + tl.arange(0, BLOCK_M)) < S_len
    
    out_ptr = O_ptr + bh_id * S_len * 128 + q_blk * BLOCK_M * 128
    row_idx = tl.arange(0, BLOCK_M)[:, None] * 128
    col_idx = tl.arange(0, BLOCK_K)[None, :]
    tl.store(out_ptr + row_idx + col_idx, O_acc.to(tl.bfloat16), mask=valid_q[:, None])
    
    lse_ptr = LSE_ptr + bh_id * S_len + q_blk * BLOCK_M
    tl.store(lse_ptr + tl.arange(0, BLOCK_M), lse_val, mask=valid_q)


def run(Q, K, V, O, LSE):
    """Compute causal attention O and LSE into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    B, H, S_len, D = Q.shape
    
    grid = (B * H, triton.cdiv(S_len, 128))
    
    Q_2d = Q.view(-1, D)
    K_2d = K.view(-1, D)
    V_2d = V.view(-1, D)
    
    q_desc = TensorDescriptor.from_tensor(Q_2d, [128, 128])
    k_desc = TensorDescriptor.from_tensor(K_2d, [128, 128])
    v_desc = TensorDescriptor.from_tensor(V_2d, [128, 128])
    
    _attention_kernel[grid](
        q_desc, k_desc, v_desc,
        O, LSE,
        S_len,
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=128,
        num_warps=4,
    )