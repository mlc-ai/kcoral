import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _attention_kernel(
    Q_desc, K_desc, V_desc, O_desc, LSE_ptr,
    S_len, scale,
    H: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HEAD_DIM: tl.constexpr = 128,
    lse_s0=None,
    lse_s1=None,
    lse_s2=None,
):
    bh_idx = tl.program_id(0)
    q_blk = tl.program_id(1)
    q_start = q_blk * BLOCK_Q
    
    Q = Q_desc.load([bh_idx * S_len + q_start, 0])
    
    acc_O = tl.zeros((BLOCK_Q, HEAD_DIM), tl.float32)
    exp_sum = tl.zeros((BLOCK_Q,), tl.float32)
    global_max = tl.full((BLOCK_Q,), -float('inf'), tl.float32)
    
    q_offsets = q_start + tl.arange(0, BLOCK_Q)
    q_mask = q_offsets < S_len
    
    num_k_blks = tl.cdiv(S_len, BLOCK_K)
    num_iter = min(q_blk + 1, num_k_blks)
    
    for k_blk in range(num_iter):
        k_start = k_blk * BLOCK_K
        
        K = K_desc.load([bh_idx * S_len + k_start, 0])
        V = V_desc.load([bh_idx * S_len + k_start, 0])
        V_fp32 = V.to(tl.float32)
        
        P = tl.dot(Q, K.T) * scale
        
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask = (q_offsets[:, None] >= k_offsets[None, :]) & \
               q_mask[:, None] & \
               (k_offsets[None, :] < S_len)
        P = tl.where(mask, P, -float('inf'))
        
        block_max = tl.max(P, axis=1, keep_dims=False)
        new_max = tl.maximum(global_max, block_max)
        p_corr = tl.exp(global_max - new_max)
        
        acc_O = acc_O * p_corr[:, None]
        exp_P = tl.exp(P - new_max[:, None])
        exp_sum = exp_sum * p_corr + tl.sum(exp_P, axis=1, keep_dims=False)
        global_max = new_max
        
        acc_O = acc_O + tl.dot(exp_P, V_fp32)
    
    inv_sum = 1.0 / exp_sum
    O = (acc_O * inv_sum[:, None]).to(tl.bfloat16)
    
    O_desc.store([bh_idx * S_len + q_start, 0], O)
    
    b = bh_idx // H
    h = bh_idx % H
    lse = global_max + tl.log(exp_sum)
    lse = tl.where(exp_sum > 0, lse, -float('inf'))
    lse_offsets = b * lse_s0 + h * lse_s1 + (q_start + tl.arange(0, BLOCK_Q)) * lse_s2
    tl.store(LSE_ptr + lse_offsets, lse, mask=q_mask)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    Q_desc = TensorDescriptor.from_tensor(Q.reshape(-1, D), [64, 128])
    K_desc = TensorDescriptor.from_tensor(K.reshape(-1, D), [64, 128])
    V_desc = TensorDescriptor.from_tensor(V.reshape(-1, D), [64, 128])
    O_desc = TensorDescriptor.from_tensor(O.reshape(-1, D), [64, 128])
    
    scale = 1.0 / (D ** 0.5)
    lse_s0, lse_s1, lse_s2 = LSE.stride(0), LSE.stride(1), LSE.stride(2)
    
    grid = (B * H, triton.cdiv(S, 64))
    _attention_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, LSE,
        S, scale,
        H=H, BLOCK_Q=64, BLOCK_K=64,
        lse_s0=lse_s0, lse_s1=lse_s1, lse_s2=lse_s2,
        num_warps=4, num_stages=3,
    )