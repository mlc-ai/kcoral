import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_fwd_kernel(
    desc_Q, desc_K, desc_V,
    O_ptr, LSE_ptr,
    S_len, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
):
    batch_head = tl.program_id(0)
    start_n = tl.program_id(1) * BLOCK_M
    
    # Load Q tile. It stays resident and is reused to project against every K tile.
    Q_left = desc_Q.load([batch_head, start_n, 0])
    Q_right = desc_Q.load([batch_head, start_n, 64])
    
    # Online softmax state initialization
    O_acc_left = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    O_acc_right = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    l = tl.zeros((BLOCK_M,), tl.float32)
    m = tl.full((BLOCK_M,), -float('inf'), tl.float32)
    
    num_j = tl.cdiv(S_len, BLOCK_N)
    
    for j in range(num_j):
        # 1. QK^T phase
        K_left = desc_K.load([batch_head, j * BLOCK_N, 0])
        K_right = desc_K.load([batch_head, j * BLOCK_N, 64])
        
        S = Q_left @ K_left.T + Q_right @ K_right.T
        S = S * scale
        
        # 2. Softmax phase
        m_old = m
        m = tl.maximum(m, tl.max(S, axis=1))
        P = tl.exp(S - m[:, None])
        
        l = l * tl.exp(m_old - m) + tl.sum(P, axis=1)
        
        O_acc_left = O_acc_left * tl.exp(m_old - m)[:, None]
        O_acc_right = O_acc_right * tl.exp(m_old - m)[:, None]
        
        # 3. PV phase
        V_left = desc_V.load([batch_head, j * BLOCK_N, 0])
        V_right = desc_V.load([batch_head, j * BLOCK_N, 64])
        
        O_acc_left += P @ V_left
        O_acc_right += P @ V_right
        
    # Epilogue phase 
    inv_l = 1.0 / l
    out_left = (O_acc_left * inv_l[:, None]).to(tl.bfloat16)
    out_right = (O_acc_right * inv_l[:, None]).to(tl.bfloat16)
    
    rows = tl.arange(0, BLOCK_M)
    cols_left = tl.arange(0, 64)
    cols_right = tl.arange(0, 64) + 64
    
    out_ptr_left = O_ptr + (batch_head * S_len + start_n + rows)[:, None] * D + cols_left[None, :]
    out_ptr_right = O_ptr + (batch_head * S_len + start_n + rows)[:, None] * D + cols_right[None, :]
    
    tl.store(out_ptr_left, out_left)
    tl.store(out_ptr_right, out_right)
    
    lse_val = m + tl.log(l)
    lse_ptr = LSE_ptr + batch_head * S_len + start_n + rows
    tl.store(lse_ptr, lse_val)


def run(Q, K, V, O, LSE):
    device = Q.device
    torch.cuda.set_device(device)
    
    B, H, S, D = Q.shape
    num_batch_head = B * H
    
    Q_3d = Q.view(num_batch_head, S, D)
    K_3d = K.view(num_batch_head, S, D)
    V_3d = V.view(num_batch_head, S, D)
    
    desc_Q = TensorDescriptor.from_tensor(Q_3d, [64, 64])
    desc_K = TensorDescriptor.from_tensor(K_3d, [64, 64])
    desc_V = TensorDescriptor.from_tensor(V_3d, [64, 64])
    
    scale = 1.0 / (D ** 0.5)
    
    O_ptr = O.data_ptr()
    LSE_ptr = LSE.data_ptr()
    
    BLOCK_M = 64
    BLOCK_N = 64
    
    grid = (num_batch_head, triton.cdiv(S, BLOCK_M))
    
    _mha_fwd_kernel[grid](
        desc_Q, desc_K, desc_V,
        O_ptr, LSE_ptr,
        S, scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        D=D,
        num_warps=4,
        num_stages=3,
    )