import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_kernel(
    Q_desc, K_desc, V_desc, O_desc, LSE_ptr,
    S_len, H, scale, b, h,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    start_m = pid_m * BLOCK_M
    
    Q_0 = Q_desc.load([start_m, 0])
    Q_1 = Q_desc.load([start_m, 1])
    Q_0_fp32 = Q_0.to(tl.float32)
    Q_1_fp32 = Q_1.to(tl.float32)
    
    m_local = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    s_local = tl.full((BLOCK_M,), 0.0, dtype=tl.float32)
    acc_0 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    acc_1 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    
    num_k_tiles = tl.cdiv(S_len, BLOCK_N)
    
    for k_tile in range(num_k_tiles):
        start_k = k_tile * BLOCK_N
        
        K_0 = K_desc.load([start_k, 0])
        K_1 = K_desc.load([start_k, 1])
        V_0 = V_desc.load([start_k, 0])
        V_1 = V_desc.load([start_k, 1])
        
        K_0_fp32 = K_0.to(tl.float32)
        K_1_fp32 = K_1.to(tl.float32)
        V_0_fp32 = V_0.to(tl.float32)
        V_1_fp32 = V_1.to(tl.float32)
        
        S_qk = tl.dot(Q_0_fp32, K_0_fp32.T) * scale + tl.dot(Q_1_fp32, K_1_fp32.T) * scale
        
        k_idx = start_k + tl.arange(0, BLOCK_N)
        valid_k = k_idx < S_len
        S_qk = tl.where(valid_k[None, :], S_qk, -float('inf'))
        
        m_curr = tl.max(S_qk, axis=1)
        m_new = tl.maximum(m_local, m_curr)
        
        diff = (S_qk - m_new[:, None]) * 0.144269504088896f
        p = tl.exp2(diff)
        
        s_curr = tl.sum(p, axis=1)
        exp_log_s = tl.exp2((m_local - m_new) * 0.14427f)
        s_local = s_local * exp_log_s + s_curr
        
        acc_0 = acc_0 * exp_log_s[:, None] + tl.dot(p, V_0_fp32)
        acc_1 = acc_1 * exp_log_s[:, None] + tl.dot(p, V_1_fp32)
        
        m_local = m_new
    
    d = 1.0 / s_local
    O_0 = (acc_0 * d[:, None]).to(tl.bfloat16)
    O_1 = (acc_1 * d[:, None]).to(tl.bfloat16)
    
    O_desc.store([start_m, 0], O_0)
    O_desc.store([start_m, 1], O_1)
    
    q_idx = tl.arange(0, BLOCK_M)
    valid_m = q_idx < (S_len - start_m)
    lse = m_local + tl.log(s_local)
    lse_off = b * H * S_len + h * S_len + start_m + q_idx
    tl.store(LSE_ptr + lse_off, lse, mask=valid_m)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S_len, D = Q.shape
    scale = 1.0 / (D ** 0.5)
    
    BLOCK_M = 128
    BLOCK_N = 64
    
    grid = (triton.cdiv(S_len, BLOCK_M),)
    
    for b in range(B):
        for h in range(H):
            b_h_offset = (b * H + h) * S_len * D * 2
            Q_desc = TensorDescriptor.from_bytes(Q.data_ptr() + b_h_offset, [S_len, D], [D, 1], [BLOCK_M, 64])
            K_desc = TensorDescriptor.from_bytes(K.data_ptr() + b_h_offset, [S_len, D], [D, 1], [BLOCK_N, 64])
            V_desc = TensorDescriptor.from_bytes(V.data_ptr() + b_h_offset, [S_len, D], [D, 1], [BLOCK_N, 64])
            O_desc = TensorDescriptor.from_bytes(O.data_ptr() + b_h_offset, [S_len, D], [D, 1], [BLOCK_M, 64])
            
            _mha_kernel[grid](
                Q_desc, K_desc, V_desc, O_desc, LSE,
                S_len, H, scale, b, h,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                num_warps=8, num_stages=3,
            )