import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_kernel(
    Q_desc, K_desc, V_desc, O_desc, LSE_ptr,
    S_len, H, scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_M
    b = tl.program_id(1)
    h = tl.program_id(2)
    
    Q = Q_desc.load([b, h, start_m, 0])
    Q = tl.reshape(Q, [BLOCK_M, 128])
    Q_fp32 = Q.to(tl.float32)
    
    m_local = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    s_local = tl.full((BLOCK_M,), 0.0, dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    
    num_k_tiles = tl.cdiv(S_len, BLOCK_N)
    k_idx_base = tl.arange(0, BLOCK_N)
    q_idx_base = tl.arange(0, BLOCK_M)
    
    for k_tile in range(num_k_tiles):
        start_k = k_tile * BLOCK_N
        
        K = K_desc.load([b, h, start_k, 0])
        K = tl.reshape(K, [BLOCK_N, 128])
        K_fp32 = K.to(tl.float32)
        
        V = V_desc.load([b, h, start_k, 0])
        V = tl.reshape(V, [BLOCK_N, 128])
        V_fp32 = V.to(tl.float32)
        
        S_qk = tl.dot(Q_fp32, K_fp32.T) * scale
        
        valid_k = (start_k + k_idx_base) < S_len
        S_qk = tl.where(valid_k[None, :], S_qk, -float('inf'))
        
        m_curr = tl.max(S_qk, axis=1)
        m_new = tl.maximum(m_local, m_curr)
        
        p = tl.exp(S_qk - m_new[:, None])
        
        s_curr = tl.sum(p, axis=1)
        s_local = s_local * tl.exp(m_local - m_new) + s_curr
        
        acc = acc * tl.exp(m_local - m_new)[:, None] + tl.dot(p, V_fp32)
        
        m_local = m_new
    
    valid_m = q_idx_base < (S_len - start_m)
    d = 1.0 / s_local
    O_tile = (acc * d[:, None]).to(tl.bfloat16)
    
    O_tile_4d = tl.reshape(O_tile, [1, 1, BLOCK_M, 128])
    O_desc.store([b, h, start_m, 0], O_tile_4d)
    
    lse = m_local + tl.log(s_local)
    lse_off = b * H * S_len + h * S_len + start_m + q_idx_base
    tl.store(LSE_ptr + lse_off, lse, mask=valid_m)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S_len, D = Q.shape
    scale = 1.0 / (D ** 0.5)
    
    BLOCK_M = 64
    BLOCK_N = 64
    
    Q_desc = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M, D])
    K_desc = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N, D])
    V_desc = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N, D])
    O_desc = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M, D])
    
    grid = (triton.cdiv(S_len, BLOCK_M), B, H)
    _mha_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, LSE,
        S_len, H, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=8, num_stages=3,
    )