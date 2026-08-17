import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _attention_kernel_q_even(
    Q_desc, K_desc, V_desc, O_desc, LSE,
    S_len, scale,
    H: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    bh_idx = tl.program_id(0)
    q_blk = tl.program_id(1)
    q_start = q_blk * 2 * BLOCK_Q
    
    Q_full = Q_desc.load([bh_idx * S_len + q_start, 0])
    Q_0, Q_1 = tl.split(Q_full)
    
    acc_O_0 = tl.zeros((BLOCK_Q, 64), tl.float32)
    acc_O_1 = tl.zeros((BLOCK_Q, 64), tl.float32)
    exp_sum = tl.zeros((BLOCK_Q,), tl.float32)
    global_max = tl.full((BLOCK_Q,), -float('inf'), tl.float32)
    
    q_offsets = q_start + tl.arange(0, BLOCK_Q)
    q_mask = q_offsets < S_len
    
    num_k_blks = tl.cdiv(S_len, BLOCK_K)
    num_iter = min(q_blk * 2 + 1, num_k_blks)
    
    for k_blk in range(num_iter):
        k_start = k_blk * BLOCK_K
        
        K_full = K_desc.load([bh_idx * S_len + k_start, 0])
        V_full = V_desc.load([bh_idx * S_len + k_start, 0])
        
        K_0, K_1 = tl.split(K_full)
        V_0, V_1 = tl.split(V_full)
        
        V_0_fp32 = V_0.to(tl.float32)
        V_1_fp32 = V_1.to(tl.float32)
        
        P = (tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)) * scale
        
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        
        mask = (q_offsets[:, None] >= k_offsets[None, :]) & \
               q_mask[:, None] & \
               (k_offsets[None, :] < S_len)
        
        P = tl.where(mask, P, -float('inf'))
        
        block_max = tl.max(P, axis=1, keep_dims=False)
        new_max = tl.maximum(global_max, block_max)
        p_corr = tl.exp(global_max - new_max)
        
        acc_O_0 = acc_O_0 * p_corr[:, None]
        acc_O_1 = acc_O_1 * p_corr[:, None]
        
        exp_P = tl.exp(P - new_max[:, None])
        
        exp_sum = exp_sum * p_corr + tl.sum(exp_P, axis=1, keep_dims=False)
        global_max = new_max
        
        acc_O_0 = acc_O_0 + tl.dot(exp_P, V_0_fp32)
        acc_O_1 = acc_O_1 + tl.dot(exp_P, V_1_fp32)
    
    inv_sum = 1.0 / exp_sum
    
    O_0 = (acc_O_0 * inv_sum[:, None]).to(tl.bfloat16)
    O_1 = (acc_O_1 * inv_sum[:, None]).to(tl.bfloat16)
    
    O_desc.store([bh_idx * S_len + q_start, 0], O_0)
    O_desc.store([bh_idx * S_len + q_start, 64], O_1)
    
    b = bh_idx // H
    h = bh_idx % H
    lse = global_max + tl.log(exp_sum)
    lse = tl.where(exp_sum > 0, lse, -float('inf'))
    lse_offsets = b * H * S_len + h * S_len + q_offsets
    tl.store(LSE + lse_offsets, lse, mask=q_mask)


@triton.jit
def _attention_kernel_q_odd(
    Q_desc, K_desc, V_desc, O_desc, LSE,
    S_len, scale,
    H: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    bh_idx = tl.program_id(0)
    q_blk = tl.program_id(1)
    q_start = q_blk * 2 * BLOCK_Q + BLOCK_Q
    
    Q_full = Q_desc.load([bh_idx * S_len + q_start, 0])
    Q_0, Q_1 = tl.split(Q_full)
    
    acc_O_0 = tl.zeros((BLOCK_Q, 64), tl.float32)
    acc_O_1 = tl.zeros((BLOCK_Q, 64), tl.float32)
    exp_sum = tl.zeros((BLOCK_Q,), tl.float32)
    global_max = tl.full((BLOCK_Q,), -float('inf'), tl.float32)
    
    q_offsets = q_start + tl.arange(0, BLOCK_Q)
    q_mask = q_offsets < S_len
    
    num_k_blks = tl.cdiv(S_len, BLOCK_K)
    num_iter = min(q_blk * 2 + 2, num_k_blks)
    
    for k_blk in range(num_iter):
        k_start = k_blk * BLOCK_K
        
        K_full = K_desc.load([bh_idx * S_len + k_start, 0])
        V_full = V_desc.load([bh_idx * S_len + k_start, 0])
        
        K_0, K_1 = tl.split(K_full)
        V_0, V_1 = tl.split(V_full)
        
        V_0_fp32 = V_0.to(tl.float32)
        V_1_fp32 = V_1.to(tl.float32)
        
        P = (tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)) * scale
        
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        
        mask = (q_offsets[:, None] >= k_offsets[None, :]) & \
               q_mask[:, None] & \
               (k_offsets[None, :] < S_len)
        
        P = tl.where(mask, P, -float('inf'))
        
        block_max = tl.max(P, axis=1, keep_dims=False)
        new_max = tl.maximum(global_max, block_max)
        p_corr = tl.exp(global_max - new_max)
        
        acc_O_0 = acc_O_0 * p_corr[:, None]
        acc_O_1 = acc_O_1 * p_corr[:, None]
        
        exp_P = tl.exp(P - new_max[:, None])
        
        exp_sum = exp_sum * p_corr + tl.sum(exp_P, axis=1, keep_dims=False)
        global_max = new_max
        
        acc_O_0 = acc_O_0 + tl.dot(exp_P, V_0_fp32)
        acc_O_1 = acc_O_1 + tl.dot(exp_P, V_1_fp32)
    
    inv_sum = 1.0 / exp_sum
    
    O_0 = (acc_O_0 * inv_sum[:, None]).to(tl.bfloat16)
    O_1 = (acc_O_1 * inv_sum[:, None]).to(tl.bfloat16)
    
    O_desc.store([bh_idx * S_len + q_start, 0], O_0)
    O_desc.store([bh_idx * S_len + q_start, 64], O_1)
    
    b = bh_idx // H
    h = bh_idx % H
    lse = global_max + tl.log(exp_sum)
    lse = tl.where(exp_sum > 0, lse, -float('inf'))
    lse_offsets = b * H * S_len + h * S_len + q_offsets
    tl.store(LSE + lse_offsets, lse, mask=q_mask)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    Q_desc = TensorDescriptor.from_tensor(Q.reshape(-1, D), [64, 128])
    K_desc = TensorDescriptor.from_tensor(K.reshape(-1, D), [64, 128])
    V_desc = TensorDescriptor.from_tensor(V.reshape(-1, D), [64, 128])
    O_desc = TensorDescriptor.from_tensor(O.reshape(-1, D), [64, 128])
    
    scale = 1.0 / (D ** 0.5)
    
    grid = (B * H, triton.cdiv(S, 128))
    
    _attention_kernel_q_even[grid](
        Q_desc, K_desc, V_desc, O_desc, LSE,
        S, scale,
        H=H, BLOCK_Q=64, BLOCK_K=64,
        num_warps=8, num_stages=2,
    )
    
    _attention_kernel_q_odd[grid](
        Q_desc, K_desc, V_desc, O_desc, LSE,
        S, scale,
        H=H, BLOCK_Q=64, BLOCK_K=64,
        num_warps=8, num_stages=2,
    )