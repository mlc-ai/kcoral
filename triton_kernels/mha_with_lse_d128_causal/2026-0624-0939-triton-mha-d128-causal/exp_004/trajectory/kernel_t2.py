import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_fwd_kernel(
    Q_desc, K_desc, V_desc, O_desc,
    LSE_ptr,
    B, H, S_len, D, scale,
    stride_LSE_b, stride_LSE_h, stride_LSE_s,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    bh_idx = tl.program_id(0)
    step = tl.program_id(1)
    row = step * BLOCK_M
    
    if row >= S_len:
        return
    
    b = bh_idx // H
    h = bh_idx % H
    b_h_offset = bh_idx * S_len
    
    Q = Q_desc.load([b_h_offset + row, 0])
    
    acc_o = tl.zeros((BLOCK_M, D), tl.float32)
    m = tl.full((BLOCK_M,), -float('inf'), tl.float32)
    l = tl.full((BLOCK_M,), 0.0, tl.float32)
    
    j_max_causal = (row + BLOCK_M + BLOCK_N - 1) // BLOCK_N
    j_max_len = (S_len + BLOCK_N - 1) // BLOCK_N
    j_max = min(j_max_causal, j_max_len)
    
    for j in range(j_max):
        col = j * BLOCK_N
        K = K_desc.load([b_h_offset + col, 0])
        V = V_desc.load([b_h_offset + col, 0])
        
        S = tl.dot(Q, K.T) * scale
        
        q_indices = row + tl.arange(0, BLOCK_M)
        k_indices = col + tl.arange(0, BLOCK_N)
        mask = k_indices[None, :] <= q_indices[:, None]
        S = tl.where(mask, S, -10000.0)
        
        m_old = m
        m_new = tl.maximum(m, tl.max(S, axis=1))
        P = tl.exp(S - m_new)
        l = l * tl.exp(m_old - m_new) + tl.sum(P, axis=1)
        
        acc_o = acc_o * tl.exp(m_old - m_new)[:, None] + tl.dot(P, V)
        m = m_new
    
    O = acc_o / l[:, None]
    O_desc.store([b_h_offset + row, 0], O.to(tl.bfloat16))
    
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
    BLOCK_N = 64
    
    Q_desc = TensorDescriptor.from_tensor(Q_2d, [BLOCK_M, D])
    K_desc = TensorDescriptor.from_tensor(K_2d, [BLOCK_N, D])
    V_desc = TensorDescriptor.from_tensor(V_2d, [BLOCK_N, D])
    O_desc = TensorDescriptor.from_tensor(O_2d, [BLOCK_M, D])
    
    stride_LSE_b = LSE.stride(0)
    stride_LSE_h = LSE.stride(1)
    stride_LSE_s = LSE.stride(2)
    
    num_q_tiles = triton.cdiv(S_len, BLOCK_M)
    
    grid = (B * H, num_q_tiles)
    scale = 1.0 / (D ** 0.5)
    
    props = torch.cuda.get_device_properties("cuda")
    num_programs = props.mp_count
    
    _mha_fwd_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc,
        LSE,
        B, H, S_len, D, scale,
        stride_LSE_b, stride_LSE_h, stride_LSE_s,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )