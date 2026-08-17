import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _attention_kernel(
    Q_desc, K_desc, V_desc, O_desc, LSE_ptr,
    S_len, scale,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
):
    bh_idx = tl.program_id(0)
    q_blk = tl.program_id(1)
    q_start = q_blk * BLOCK_Q
    
    b = bh_idx // H
    h = bh_idx % H
    
    # Load Q tile for this query block
    Q = Q_desc.load([bh_idx * S_len + q_start, 0])
    
    acc_O = tl.zeros((BLOCK_Q, D), tl.float32)
    exp_sum = tl.zeros((BLOCK_Q,), tl.float32)
    max_val = tl.full((BLOCK_Q,), -float('inf'), tl.float32)
    
    q_offsets = q_start + tl.arange(BLOCK_Q)
    
    # Iterate over KV blocks respecting causal mask
    for k_blk in range(q_blk + 1):
        k_start = k_blk * BLOCK_K
        
        K = K_desc.load([bh_idx * S_len + k_start, 0])
        V = V_desc.load([bh_idx * S_len + k_start, 0])
        
        # Q @ K^T
        P = tl.dot(Q, K.T) * scale
        
        k_offsets = k_start + tl.arange(BLOCK_K)
        mask = (q_offsets[:, None] >= k_offsets[None, :]) & \
               (q_offsets[:, None] < S_len) & \
               (k_offsets[None, :] < S_len)
        P = tl.where(mask, P, -float('inf'))
        
        block_max = tl.max(P, axis=1, keep_dims=False)
        new_max = tl.maximum(max_val, block_max)
        p_corr = tl.exp(max_val - new_max)
        
        acc_O = acc_O * p_corr[:, None]
        exp_sum = exp_sum * p_corr + tl.sum(tl.exp(P - new_max[:, None]), axis=1, keep_dims=False)
        max_val = new_max
        
        S_attn = tl.exp(P - max_val[:, None])
        acc_O += tl.dot(S_attn, V)
    
    O = acc_O / exp_sum[:, None]
    O_desc.store([bh_idx * S_len + q_start, 0], O.to(tl.bfloat16))
    
    lse = max_val + tl.log(exp_sum)
    lse_offsets = b * H * S_len + h * S_len + q_start + tl.arange(BLOCK_Q)
    mask_lse = q_offsets < S_len
    tl.store(LSE_ptr + lse_offsets, lse, mask=mask_lse)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    Q_desc = TensorDescriptor.from_tensor(Q.reshape(-1, D), [64, D])
    K_desc = TensorDescriptor.from_tensor(K.reshape(-1, D), [64, D])
    V_desc = TensorDescriptor.from_tensor(V.reshape(-1, D), [64, D])
    O_desc = TensorDescriptor.from_tensor(O.reshape(-1, D), [64, D])
    
    scale = 1.0 / (D ** 0.5)
    
    grid = (B * H, triton.cdiv(S, 64))
    _attention_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, LSE,
        S, scale,
        BLOCK_Q=64, BLOCK_K=64, D=D, H=H,
        num_warps=4, num_stages=3,
    )