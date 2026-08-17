import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def kernel(
    Q_desc, K_desc, V_desc, O_desc, LSE_ptr,
    S, scale, num_blocks,
    stride_LSE_b, stride_LSE_h, stride_LSE_s,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    bh = tl.program_id(1)
    q_block_id = tl.program_id(0)
    q_row_start = q_block_id * BLOCK_M
    
    b_idx = bh // 48
    h_idx = bh % 48
    
    Q_tile = Q_desc.load([b_idx, h_idx, q_row_start, 0])
    
    O_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    running_max = tl.full((BLOCK_M,), -1e38, tl.float32)
    running_sum = tl.full((BLOCK_M,), 0.0, tl.float32)
    
    m_arr = tl.arange(0, BLOCK_M)
    n_arr = tl.arange(0, BLOCK_N)
    query_valid = (q_row_start + m_arr) < S
    
    for kv_block_id in range(0, q_block_id + 1):
        kv_row_start = kv_block_id * BLOCK_N
        
        K_tile = K_desc.load([b_idx, h_idx, kv_row_start, 0])
        
        P = tl.dot(Q_tile, K_tile.T)
        P *= scale
        
        causal_mask = (q_row_start + m_arr[:, None]) >= (kv_row_start + n_arr[None, :])
        mask = query_valid[:, None] & causal_mask
        P = tl.where(mask, P, -1e38)
        
        row_max = tl.max(P, axis=1)
        new_max = tl.maximum(running_max, row_max)
        
        running_sum *= tl.exp(running_max - new_max)
        O_acc *= tl.exp(running_max - new_max)[:, None]
        
        P_scaled = P - new_max[:, None]
        exp_P = tl.where(mask, tl.exp(P_scaled), 0.0)
        row_sum = tl.sum(exp_P, axis=1)
        running_sum += row_sum
        
        V_tile = V_desc.load([b_idx, h_idx, kv_row_start, 0])
        O_acc += tl.dot(exp_P.to(tl.bfloat16), V_tile)
                
        running_max = new_max
    
    O_acc /= running_sum[:, None]
    
    lse = running_max + tl.log(running_sum)
    
    row_mask = query_valid
    O_desc.store([b_idx, h_idx, q_row_start, 0], O_acc.to(tl.bfloat16), mask=row_mask[:, None])
    
    lse_ptr = LSE_ptr + b_idx * stride_LSE_b + h_idx * stride_LSE_h
    tl.store(lse_ptr + q_row_start + m_arr, lse, mask=row_mask)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    scale = 1.0 / (D ** 0.5)
    
    BLOCK_M = 128
    BLOCK_N = 128
    
    Q_desc = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M, BLOCK_N])
    K_desc = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N, BLOCK_N])
    V_desc = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N, BLOCK_N])
    O_desc = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M, BLOCK_N])
    
    num_blocks = triton.cdiv(S, BLOCK_M)
    grid = (num_blocks, B * H)
    
    kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, LSE,
        S, scale, num_blocks,
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4,
    )