import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_fwd_kernel(
    Q_desc, K_desc, V_desc, O_desc, LSE_ptr,
    H, S_len, D, scale, num_q_tiles,
    stride_LSE_b, stride_LSE_h, stride_LSE_s,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    tile_id = tl.program_id(0)
    
    b_h_idx = tile_id // num_q_tiles
    step = tile_id % num_q_tiles
    row = step * BLOCK_M
    
    if row >= S_len:
        pass
    else:
        b_h_offset = b_h_idx * S_len
        
        Q = Q_desc.load([b_h_offset + row, 0], boundary_check=(0,))
        
        acc_o = tl.zeros((BLOCK_M, D), tl.float32)
        m = tl.full((BLOCK_M,), -float('inf'), tl.float32)
        l = tl.full((BLOCK_M,), 0.0, tl.float32)
        
        j_max_causal = tl.cdiv(row + BLOCK_M, BLOCK_N)
        j_max_len = tl.cdiv(S_len, BLOCK_N)
        j_max = min(j_max_causal, j_max_len)
        
        for j in range(j_max):
            K = K_desc.load([b_h_offset + j * BLOCK_N, 0], boundary_check=(0,))
            V = V_desc.load([b_h_offset + j * BLOCK_N, 0], boundary_check=(0,))
            
            S = tl.dot(Q, K.T) * scale
            
            q_idx_base = step * BLOCK_M + tl.arange(0, BLOCK_M)
            k_idx_base = j * BLOCK_N + tl.arange(0, BLOCK_N)
            mask = (k_idx_base[None, :] <= q_idx_base[:, None]) & (k_idx_base[None, :] < S_len)
            S = tl.where(mask, S, -float('inf'))
            
            m_old = m
            m_new = tl.maximum(m, tl.max(S, axis=1, keep_dims=True))
            P = tl.exp(S - m_new)
            l = l * tl.exp(m_old - m_new) + tl.sum(P, axis=1, keep_dims=True)
            
            exp_scale = tl.exp(m_old - m_new)
            acc_o = acc_o * exp_scale + tl.dot(P, V)
            m = m_new.squeeze()
            l = l.squeeze()
        
        O = acc_o / l[:, None]
        valid = (l > 0)[:, None]
        O = tl.where(valid, O, 0.0)
        O_desc.store([b_h_offset + row, 0], O.to(tl.bfloat16))
        
        b = b_h_idx // H
        h = b_h_idx % H
        LSE_val = m + tl.log(l)
        off_row = tl.arange(0, BLOCK_M)
        ptr_LSE = LSE_ptr + b * stride_LSE_b + h * stride_LSE_h + (row + off_row) * stride_LSE_s
        mask_LSE = (row + off_row) < S_len
        tl.store(ptr_LSE, LSE_val, mask=mask_LSE)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S_len, D = Q.shape
    
    Q_2d = Q.view(B * H * S_len, D)
    K_2d = K.view(B * H * S_len, D)
    V_2d = V.view(B * H * S_len, D)
    O_2d = O.view(B * H * S_len, D)
    
    BLOCK_M = 128
    BLOCK_N = 128
    
    Q_desc = TensorDescriptor.from_tensor(Q_2d, [BLOCK_M, D])
    K_desc = TensorDescriptor.from_tensor(K_2d, [BLOCK_N, D])
    V_desc = TensorDescriptor.from_tensor(V_2d, [BLOCK_N, D])
    O_desc = TensorDescriptor.from_tensor(O_2d, [BLOCK_M, D])
    
    stride_LSE_b = LSE.stride(0)
    stride_LSE_h = LSE.stride(1)
    stride_LSE_s = LSE.stride(2)
    
    num_q_tiles = triton.cdiv(S_len, BLOCK_M)
    total_tiles = B * H * num_q_tiles
    
    grid = (total_tiles,)
    scale = 1.0 / (D ** 0.5)
    
    _mha_fwd_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, LSE,
        H, S_len, D, scale, num_q_tiles,
        stride_LSE_b, stride_LSE_h, stride_LSE_s,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )