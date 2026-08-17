import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _attention_kernel(
    Q_desc, K_desc, V_desc, O_desc, LSE,
    S_len, scale,
    H: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    bh_idx = tl.program_id(0)
    q_blk = tl.program_id(1)
    
    q_start_0 = q_blk * 2 * BLOCK_Q
    q_start_1 = q_blk * 2 * BLOCK_Q + BLOCK_Q
    
    Q_0_full = Q_desc.load([bh_idx * S_len + q_start_0, 0])
    Q_1_full = Q_desc.load([bh_idx * S_len + q_start_1, 0])
    
    Q_0_0, Q_0_1 = tl.split(Q_0_full)
    Q_1_0, Q_1_1 = tl.split(Q_1_full)
    
    acc_O_0_0 = tl.zeros((BLOCK_Q, 64), tl.float32)
    acc_O_0_1 = tl.zeros((BLOCK_Q, 64), tl.float32)
    acc_O_1_0 = tl.zeros((BLOCK_Q, 64), tl.float32)
    acc_O_1_1 = tl.zeros((BLOCK_Q, 64), tl.float32)
    
    exp_sum_0 = tl.full((BLOCK_Q,), 1.0, tl.float32)
    exp_sum_1 = tl.full((BLOCK_Q,), 1.0, tl.float32)
    global_max_0 = tl.full((BLOCK_Q,), -float('inf'), tl.float32)
    global_max_1 = tl.full((BLOCK_Q,), -float('inf'), tl.float32)
    
    q_offsets_0 = q_start_0 + tl.arange(0, BLOCK_Q)
    q_offsets_1 = q_start_1 + tl.arange(0, BLOCK_Q)
    q_mask_0 = q_offsets_0 < S_len
    q_mask_1 = q_offsets_1 < S_len
    
    num_k_blks = tl.cdiv(S_len, BLOCK_K)
    num_iter_0 = min(q_blk * 2 + 1, num_k_blks)
    num_iter_1 = min(q_blk * 2 + 2, num_k_blks)
    num_iter = max(num_iter_0, num_iter_1)
    
    for k_blk in range(num_iter):
        k_start = k_blk * BLOCK_K
        
        K_full = K_desc.load([bh_idx * S_len + k_start, 0])
        V_full = V_desc.load([bh_idx * S_len + k_start, 0])
        
        K_0, K_1 = tl.split(K_full)
        V_0, V_1 = tl.split(V_full)
        
        P_0 = (tl.dot(Q_0_0, K_0.T) + tl.dot(Q_0_1, K_1.T)) * scale
        P_1 = (tl.dot(Q_1_0, K_0.T) + tl.dot(Q_1_1, K_1.T)) * scale
        
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        
        mask_0 = (q_offsets_0[:, None] >= k_offsets[None, :]) & \
                 q_mask_0[:, None] & \
                 (k_offsets[None, :] < S_len)
        mask_1 = (q_offsets_1[:, None] >= k_offsets[None, :]) & \
                 q_mask_1[:, None] & \
                 (k_offsets[None, :] < S_len)
        
        P_0 = tl.where(mask_0, P_0, -float('inf'))
        P_1 = tl.where(mask_1, P_1, -float('inf'))
        
        block_max_0 = tl.max(P_0, axis=1, keep_dims=False)
        block_max_1 = tl.max(P_1, axis=1, keep_dims=False)
        
        new_max_0 = tl.maximum(global_max_0, block_max_0)
        new_max_1 = tl.maximum(global_max_1, block_max_1)
        
        p_corr_0 = tl.exp(global_max_0 - new_max_0)
        p_corr_1 = tl.exp(global_max_1 - new_max_1)
        
        acc_O_0_0 = acc_O_0_0 * p_corr_0[:, None]
        acc_O_0_1 = acc_O_0_1 * p_corr_0[:, None]
        acc_O_1_0 = acc_O_1_0 * p_corr_1[:, None]
        acc_O_1_1 = acc_O_1_1 * p_corr_1[:, None]
        
        exp_P_0 = tl.exp(P_0 - new_max_0[:, None])
        exp_P_1 = tl.exp(P_1 - new_max_1[:, None])
        
        exp_sum_0 = exp_sum_0 * p_corr_0 + tl.sum(exp_P_0, axis=1, keep_dims=False)
        exp_sum_1 = exp_sum_1 * p_corr_1 + tl.sum(exp_P_1, axis=1, keep_dims=False)
        
        global_max_0 = new_max_0
        global_max_1 = new_max_1
        
        acc_O_0_0 = acc_O_0_0 + tl.dot(exp_P_0, V_0)
        acc_O_0_1 = acc_O_0_1 + tl.dot(exp_P_0, V_1)
        acc_O_1_0 = acc_O_1_0 + tl.dot(exp_P_1, V_0)
        acc_O_1_1 = acc_O_1_1 + tl.dot(exp_P_1, V_1)
    
    inv_sum_0 = 1.0 / exp_sum_0
    inv_sum_1 = 1.0 / exp_sum_1
    
    O_0 = tl.cat(acc_O_0_0 * inv_sum_0_0[:, None], acc_O_0_1 * inv_sum_0[:, None], dim=1).to(tl.bfloat16)
    O_1 = tl.cat(acc_O_1_0 * inv_sum_1[:, None], acc_O_1_1 * inv_sum_1[:, None], dim=1).to(tl.bfloat16)
    
    O_desc.store([bh_idx * S_len + q_start_0, 0], O_0)
    O_desc.store([bh_idx * S_len + q_start_1, 0], O_1)
    
    lse_0 = global_max_0 + tl.log(exp_sum_0)
    lse_1 = global_max_1 + tl.log(exp_sum_1)
    lse_0 = tl.where(exp_sum_0 > 0, lse_0, -float('inf'))
    lse_1 = tl.where(exp_sum_1 > 0, lse_1, -float('inf'))
    
    lse_offsets_0 = bh_idx * S_len + q_offsets_0
    lse_offsets_1 = bh_idx * S_len + q_offsets_1
    tl.store(LSE + lse_offsets_0, lse_0, mask=q_mask_0)
    tl.store(LSE + lse_offsets_1, lse_1, mask=q_mask_1)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    Q_desc = TensorDescriptor.from_tensor(Q.reshape(-1, D), [64, 128])
    K_desc = TensorDescriptor.from_tensor(K.reshape(-1, D), [64, 128])
    V_desc = TensorDescriptor.from_tensor(V.reshape(-1, D), [64, 128])
    O_desc = TensorDescriptor.from_tensor(O.reshape(-1, D), [64, 128])
    
    scale = 1.0 / (D ** 0.5)
    
    grid = (B * H, triton.cdiv(S, 128))
    _attention_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, LSE,
        S, scale,
        H=H, BLOCK_Q=64, BLOCK_K=64,
        num_warps=8, num_stages=2,
    )