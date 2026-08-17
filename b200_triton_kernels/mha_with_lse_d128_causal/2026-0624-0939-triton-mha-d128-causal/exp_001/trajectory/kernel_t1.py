import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _attention_kernel(
    Q_desc, K_desc, V_desc, O_desc, LSE_ptr,
    S_len, scale,
    HEAD_DIM: tl.constexpr,
    H: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    bh_idx = tl.program_id(0)
    q_blk = tl.program_id(1)
    q_start = q_blk * BLOCK_Q
    
    # Load Q tile for this query block
    Q = Q_desc.load([bh_idx * S_len + q_start, 0])
    Q_0 = Q[:, 0:64]
    Q_1 = Q[:, 64:128]
    
    acc_O_0 = tl.zeros((BLOCK_Q, 64), tl.float32)
    acc_O_1 = tl.zeros((BLOCK_Q, 64), tl.float32)
    exp_sum = tl.zeros((BLOCK_Q,), tl.float32)
    global_max = tl.full((BLOCK_Q,), -float('inf'), tl.float32)
    
    q_offsets = q_start + tl.arange(0, BLOCK_Q)
    
    num_k_blks = tl.cdiv(S_len, BLOCK_K)
    num_iter = min(q_blk + 1, num_k_blks)
    
    for k_blk in range(num_iter):
        k_start = k_blk * BLOCK_K
        
        K = K_desc.load([bh_idx * S_len + k_start, 0])
        V = V_desc.load([bh_idx * S_len + k_start, 0])
        
        K_0 = K[:, 0:64]
        K_1 = K[:, 64:128]
        V_0 = V[:, 0:64]
        V_1 = V[:, 64:128]
        
        # Q @ K^T split over D to maintain 64x64 dims for faster execution
        P = tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)
        P = P * scale
        
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask = (q_offsets[:, None] >= k_offsets[None, :]) & \
               (q_offsets[:, None] < S_len) & \
               (k_offsets[None, :] < S_len)
        P = tl.where(mask, P, -float('inf'))
        
        block_max = tl.max(P, axis=1, keep_dims=False)
        new_max = tl.maximum(global_max, block_max)
        p_corr = tl.exp(global_max - new_max)
        
        acc_O_0 = acc_O_0 * p_corr[:, None]
        acc_O_1 = acc_O_1 * p_corr[:, None]
        exp_sum = exp_sum * p_corr + tl.sum(tl.exp(P - new_max[:, None]), axis=1, keep_dims=False)
        global_max = new_max
        
        S_attn = tl.exp(P - global_max[:, None])
        
        # S_attn @ V split over D
        acc_O_0 = acc_O_0 + tl.dot(S_attn, V_0)
        acc_O_1 = acc_O_1 + tl.dot(S_attn, V_1)
    
    inv_sum = 1.0 / exp_sum
    O_0 = (acc_O_0 * inv_sum[:, None]).to(tl.bfloat16)
    O_1 = (acc_O_1 * inv_sum[:, None]).to(tl.bfloat16)
    
    O_desc.store([bh_idx * S_len + q_start, 0], O_0)
    O_desc.store([bh_idx * S_len + q_start, 64], O_1)
    
    b = bh_idx // H
    h = bh_idx % H
    lse = global_max + tl.log(exp_sum)
    lse_offsets = b * H * S_len + h * S_len + q_start + tl.arange(0, BLOCK_Q)
    mask_lse = q_offsets < S_len
    tl.store(LSE_ptr + lse_offsets, lse, mask=mask_lse)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    Q_desc = TensorDescriptor.from_tensor(Q.reshape(-1, D), [128, 128])
    K_desc = TensorDescriptor.from_tensor(K.reshape(-1, D), [64, 128])
    V_desc = TensorDescriptor.from_tensor(V.reshape(-1, D), [64, 128])
    O_desc = TensorDescriptor.from_tensor(O.reshape(-1, D), [128, 128])
    
    scale = 1.0 / (D ** 0.5)
    
    grid = (B * H, triton.cdiv(S, 128))
    _attention_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, LSE,
        S, scale,
        HEAD_DIM=D, H=H, BLOCK_Q=128, BLOCK_K=64,
        num_warps=4, num_stages=3,
    )