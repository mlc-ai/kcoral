import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_fwd_kernel(
    Q_desc, K_desc, V_desc, O_ptr, LSE_ptr,
    B, H, S_len, D,
    stride_O_b, stride_O_h, stride_O_s, stride_O_d,
    stride_LSE_b, stride_LSE_h, stride_LSE_s,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    idx = tl.program_id(0)
    num_programs = NUM_SMS
    
    smem_K_0 = tl.empty((BLOCK_N, D), torch.bfloat16)
    smem_K_1 = tl.empty((BLOCK_N, D), torch.bfloat16)
    smem_V_0 = tl.empty((BLOCK_N, D), torch.bfloat16)
    smem_V_1 = tl.empty((BLOCK_N, D), torch.bfloat16)
    smem_Q = tl.empty((BLOCK_M, D), torch.bfloat16)
    
    scale = 1.0 / (D ** 0.5)
    
    total_bh = B * H
    num_q_tiles = tl.cdiv(S_len, BLOCK_M)
    
    for bh_idx in range(idx, total_bh, num_programs):
        b = bh_idx // H
        h = bh_idx % H
        b_h_idx = b * H + h
        
        for step in range(num_q_tiles):
            row = step * BLOCK_M
            if row >= S_len:
                break
            
            Q = Q_desc.load([b_h_idx * S_len + row, 0], storage=smem_Q)
            tl.device_async_task.commit()
            tl.device_async_task.wait()
            
            acc_o = tl.zeros((BLOCK_M, D), tl.float32)
            m = tl.full((BLOCK_M,), -float('inf'), tl.float32)
            l = tl.full((BLOCK_M,), 0.0, tl.float32)
            
            j_max = (row + BLOCK_M + BLOCK_N - 1) // BLOCK_N
            j_max = min(j_max, (S_len + BLOCK_N - 1) // BLOCK_N)
            
            if j_max > 0:
                K_desc.load([b_h_idx * S_len + 0, 0], storage=smem_K_0)
                V_desc.load([b_h_idx * S_len + 0, 0], storage=smem_V_0)
                tl.device_async_task.commit()
            
            for j in range(j_max):
                stage = j % 2
                col = j * BLOCK_N
                
                tl.device_async_task.wait()
                
                if stage == 0:
                    S = tl.dot(Q, smem_K_0.T) * scale
                else:
                    S = tl.dot(Q, smem_K_1.T) * scale
                
                mask = (col + tl.arange(0, BLOCK_N)[None, :]) <= (row + tl.arange(0, BLOCK_M)[:, None])
                S = tl.where(mask, S, -float('inf'))
                
                m_old = m
                m = tl.maximum(m, tl.max(S, axis=1))
                P = tl.exp(S - m)
                l = l * tl.exp(m_old - m) + tl.sum(P, axis=1)
                
                if stage == 0:
                    acc_o = acc_o * tl.exp(m_old - m)[:, None] + tl.dot(P, smem_V_0)
                else:
                    acc_o = acc_o * tl.exp(m_old - m)[:, None] + tl.dot(P, smem_V_1)
                
                if j + 1 < j_max:
                    next_stage = (j + 1) % 2
                    next_col = (j + 1) * BLOCK_N
                    if next_stage == 0:
                        K_desc.load([b_h_idx * S_len + next_col, 0], storage=smem_K_0)
                        V_desc.load([b_h_idx * S_len + next_col, 0], storage=smem_V_0)
                    else:
                        K_desc.load([b_h_idx * S_len + next_col, 0], storage=smem_K_1)
                        V_desc.load([b_h_idx * S_len + next_col, 0], storage=smem_V_1)
                    tl.device_async_task.commit()
            
            O = acc_o / l[:, None]
            
            off_row = tl.arange(0, BLOCK_M)
            off_dim = tl.arange(0, D)
            row_offsets = (b * stride_O_b + h * stride_O_h + (row + off_row) * stride_O_s)[:, None]
            dim_offsets = off_dim[None, :] * stride_O_d
            ptr_O = O_ptr + row_offsets + dim_offsets
            mask_O = (row + off_row)[:, None] < S_len
            tl.store(ptr_O, O, mask=mask_O)
            
            LSE_val = m + tl.log(l)
            ptr_LSE = LSE_ptr + b * stride_LSE_b + h * stride_LSE_h + (row + off_row) * stride_LSE_s
            mask_LSE = (row + off_row) < S_len
            tl.store(ptr_LSE, LSE_val, mask=mask_LSE)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S_len, D = Q.shape
    
    Q_2d = Q.view(B * H * S_len, D)
    K_2d = K.view(B * H * S_len, D)
    V_2d = V.view(B * H * S_len, D)
    
    BLOCK_M = 64
    BLOCK_N = 64
    
    Q_desc = TensorDescriptor.from_tensor(Q_2d, [BLOCK_M, D])
    K_desc = TensorDescriptor.from_tensor(K_2d, [BLOCK_N, D])
    V_desc = TensorDescriptor.from_tensor(V_2d, [BLOCK_N, D])
    
    stride_O_b = O.stride(0)
    stride_O_h = O.stride(1)
    stride_O_s = O.stride(2)
    stride_O_d = O.stride(3)
    
    stride_LSE_b = LSE.stride(0)
    stride_LSE_h = LSE.stride(1)
    stride_LSE_s = LSE.stride(2)
    
    num_programs = min(
        triton.runtime.driver.active.get_current_device_properties()["multiprocessor_count"],
        B * H
    )
    
    grid = (num_programs,)
    _mha_fwd_kernel[grid](
        Q_desc, K_desc, V_desc, O, LSE,
        B, H, S_len, D,
        stride_O_b, stride_O_h, stride_O_s, stride_O_d,
        stride_LSE_b, stride_LSE_h, stride_LSE_s,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        NUM_SMS=num_programs,
        num_warps=4,
    )