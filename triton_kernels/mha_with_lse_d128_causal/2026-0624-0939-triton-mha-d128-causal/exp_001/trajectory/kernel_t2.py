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
):
    bh_idx = tl.program_id(0)
    q_blk = tl.program_id(1)
    q_start = q_blk * BLOCK_Q
    
    # Load Q tile for this query block
    Q = Q_desc.load([bh_idx * S_len + q_start, 0])
    
    acc_O = tl.zeros((BLOCK_Q, HEAD_DIM), tl.float32)
    exp_sum = tl.zeros((BLOCK_Q,), tl.float32)
    global_max = tl.full((BLOCK_Q,), -float('inf'), tl.float32)
    
    q_offsets = q_start + tl.arange(0, BLOCK_Q)
    
    for k_blk in range(q_blk + 1):
        k_start = k_blk * BLOCK_K
        
        K = K_desc.load([bh_idx * S_len + k_start, 0])
        V = V_desc.load([bh_idx * S_len + k_start, 0])
        
        # Q @ K^T
        P = tl.dot(Q, K.T) * scale
        
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask = (q_offsets[:, None] >= k_offsets[None, :]) & \
               (q_offsets[:, None] < S_len) & \
               (k_offsets[None, :] < S_len)
        P = tl.where(mask, P, -float('inf'))
        
        block_max = tl.max(P, axis=1, keep_dims=False)
        new_max = tl.maximum(global_max, block_max)
        p_corr = tl.exp(global_max - new_max)
        
        acc_O = acc_O * p_corr[:, None]
        exp_sum = exp_sum * p_corr + tl.sum(tl.exp(P - new_max[:, None]), axis=1, keep_dims=False)
        global_max = new_max
        
        S_attn = tl.exp(P - global_max[:, None])
        
        # S_attn @ V
        acc_O = acc_O + tl.dot(S_attn, V)
    
    inv_sum = 1.0 / exp_sum
    O = (acc_O * inv_sum[:, None]).to(tl.bfloat16)
    
    O_desc.store([bh_idx * S_len + q_start, 0], O)
    
    b = bh_idx // H
    h = bh_idx % H
    lse = global_max + tl.log(exp_sum)
    lse_offsets = b * H * S_len + h * S_len + q_start + tl.arange(0, BLOCK_Q)
    mask_lse = q_offsets < S_len
    tl.store(LSE_ptr + lse_offsets, lse, mask=mask_lse)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    Q_desc = TensorDescriptor.from_tensor(Q.reshape(-1, D), [64, 128])
    K_desc = TensorDescriptor.from_tensor(K.reshape(-1, D), [64, 128])
    V_desc = TensorDescriptor.from_tensor(V.reshape(-1, D), [64, 128])
    O_desc = TensorDescriptor.from_tensor(O.reshape(-1, D), [64, 128])
    
    scale = 1.0 / (D ** 0.5)
    
    grid = (B * H, triton.cdiv(S, 64))
    _attention_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, LSE,
        S, scale,
        H=H, BLOCK_Q=64, BLOCK_K=64,
        num_warps=4, num_stages=3,
    )