import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_bwd_dq_kernel(
    desc_Q0, desc_Q1, desc_K0, desc_K1, desc_V0, desc_V1, desc_O0, desc_O1, desc_dO0, desc_dO1, desc_dQ0, desc_dQ1,
    L_ptr,
    H, S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    pid_m = tl.program_id(0)
    m_start = pid_m * BLOCK_M
    
    Q0 = desc_Q0.load([b, h, m_start, 0])
    Q1 = desc_Q1.load([b, h, m_start, 0])
    O0 = desc_O0.load([b, h, m_start, 0])
    O1 = desc_O1.load([b, h, m_start, 0])
    dO0 = desc_dO0.load([b, h, m_start, 0])
    dO1 = desc_dO1.load([b, h, m_start, 0])
    
    Q0 = tl.reshape(Q0, [BLOCK_M, 64])
    Q1 = tl.reshape(Q1, [BLOCK_M, 64])
    O0 = tl.reshape(O0, [BLOCK_M, 64])
    O1 = tl.reshape(O1, [BLOCK_M, 64])
    dO0 = tl.reshape(dO0, [BLOCK_M, 64])
    dO1 = tl.reshape(dO1, [BLOCK_M, 64])
    
    l_idx = m_start + tl.arange(0, BLOCK_M)
    L_vec = tl.load(L_ptr + b * H * S + h * S + l_idx, mask=(l_idx < S), other=0.0)
    
    D_vec = tl.sum(O0.to(tl.float32) * dO0.to(tl.float32) + O1.to(tl.float32) * dO1.to(tl.float32), axis=1)
    
    acc_dQ0 = tl.zeros((BLOCK_M, 64), tl.float32)
    acc_dQ1 = tl.zeros((BLOCK_M, 64), tl.float32)
    
    for n_start in range(0, S, BLOCK_N):
        K0 = desc_K0.load([b, h, n_start, 0])
        K1 = desc_K1.load([b, h, n_start, 0])
        V0 = desc_V0.load([b, h, n_start, 0])
        V1 = desc_V1.load([b, h, n_start, 0])
        
        K0 = tl.reshape(K0, [BLOCK_N, 64])
        K1 = tl.reshape(K1, [BLOCK_N, 64])
        V0 = tl.reshape(V0, [BLOCK_N, 64])
        V1 = tl.reshape(V1, [BLOCK_N, 64])
        
        S_val = (tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)) * scale
        P = tl.exp(S_val - L_vec[:, None])
        
        dP = tl.dot(dO0, V0.T) + tl.dot(dO1, V1.T)
        
        dS = P * (dP - D_vec[:, None]) * scale
        
        dS_bf16 = dS.to(tl.bfloat16)
        acc_dQ0 = tl.dot(dS_bf16, K0, acc_dQ0)
        acc_dQ1 = tl.dot(dS_bf16, K1, acc_dQ1)
        
    desc_dQ0.store([b, h, m_start, 0], tl.reshape(acc_dQ0, [1, 1, BLOCK_M, 64]))
    desc_dQ1.store([b, h, m_start, 0], tl.reshape(acc_dQ1, [1, 1, BLOCK_M, 64]))


@triton.jit
def _mha_bwd_dk_dv_kernel(
    desc_Q0, desc_Q1, desc_K0, desc_K1, desc_V0, desc_V1, desc_O0, desc_O1, desc_dO0, desc_dO1, desc_dK0, desc_dK1, desc_dV0, desc_dV1,
    L_ptr,
    H, S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    pid_n = tl.program_id(0)
    n_start = pid_n * BLOCK_N
    
    K0 = desc_K0.load([b, h, n_start, 0])
    K1 = desc_K1.load([b, h, n_start, 0])
    V0 = desc_V0.load([b, h, n_start, 0])
    V1 = desc_V1.load([b, h, n_start, 0])
    
    K0 = tl.reshape(K0, [BLOCK_N, 64])
    K1 = tl.reshape(K1, [BLOCK_N, 64])
    V0 = tl.reshape(V0, [BLOCK_N, 64])
    V1 = tl.reshape(V1, [BLOCK_N, 64])
    
    acc_dK0 = tl.zeros((BLOCK_N, 64), tl.float32)
    acc_dK1 = tl.zeros((BLOCK_N, 64), tl.float32)
    acc_dV0 = tl.zeros((BLOCK_N, 64), tl.float32)
    acc_dV1 = tl.zeros((BLOCK_N, 64), tl.float32)
    
    k_idx = n_start + tl.arange(0, BLOCK_N)
    mask_n = k_idx < S
    
    for m_start in range(0, S, BLOCK_M):
        Q0 = desc_Q0.load([b, h, m_start, 0])
        Q1 = desc_Q1.load([b, h, m_start, 0])
        O0 = desc_O0.load([b, h, m_start, 0])
        O1 = desc_O1.load([b, h, m_start, 0])
        dO0 = desc_dO0.load([b, h, m_start, 0])
        dO1 = desc_dO1.load([b, h, m_start, 0])
        
        Q0 = tl.reshape(Q0, [BLOCK_M, 64])
        Q1 = tl.reshape(Q1, [BLOCK_M, 64])
        O0 = tl.reshape(O0, [BLOCK_M, 64])
        O1 = tl.reshape(O1, [BLOCK_M, 64])
        dO0 = tl.reshape(dO0, [BLOCK_M, 64])
        dO1 = tl.reshape(dO1, [BLOCK_M, 64])
        
        l_idx = m_start + tl.arange(0, BLOCK_M)
        L_vec = tl.load(L_ptr + b * H * S + h * S + l_idx, mask=(l_idx < S), other=0.0)
        
        D_vec = tl.sum(O0.to(tl.float32) * dO0.to(tl.float32) + O1.to(tl.float32) * dO1.to(tl.float32), axis=1)
        
        S_val = (tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)) * scale
        P = tl.exp(S_val - L_vec[:, None])
        P = P * mask_n[None, :]
        
        dP = tl.dot(dO0, V0.T) + tl.dot(dO1, V1.T)
        
        dS = P * (dP - D_vec[:, None]) * scale
        
        dS_bf16 = dS.to(tl.bfloat16)
        acc_dK0 = tl.dot(dS_bf16.T, Q0, acc_dK0)
        acc_dK1 = tl.dot(dS_bf16.T, Q1, acc_dK1)
        
        P_bf16 = P.to(tl.bfloat16)
        acc_dV0 = tl.dot(P_bf16.T, dO0, acc_dV0)
        acc_dV1 = tl.dot(P_bf16.T, dO1, acc_dV1)
        
    desc_dK0.store([b, h, n_start, 0], tl.reshape(acc_dK0, [1, 1, BLOCK_N, 64]))
    desc_dK1.store([b, h, n_start, 0], tl.reshape(acc_dK1, [1, 1, BLOCK_N, 64]))
    desc_dV0.store([b, h, n_start, 0], tl.reshape(acc_dV0, [1, 1, BLOCK_N, 64]))
    desc_dV1.store([b, h, n_start, 0], tl.reshape(acc_dV1, [1, 1, BLOCK_N, 64]))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    scale = 1.0 / math.sqrt(d)
    
    desc_Q0 = TensorDescriptor.from_tensor(Q, [1, 1, 64, 64])
    desc_Q1 = TensorDescriptor.from_tensor(Q, [1, 1, 64, 64])
    desc_K0 = TensorDescriptor.from_tensor(K, [1, 1, 64, 64])
    desc_K1 = TensorDescriptor.from_tensor(K, [1, 1, 64, 64])
    desc_V0 = TensorDescriptor.from_tensor(V, [1, 1, 64, 64])
    desc_V1 = TensorDescriptor.from_tensor(V, [1, 1, 64, 64])
    desc_O0 = TensorDescriptor.from_tensor(O, [1, 1, 64, 64])
    desc_O1 = TensorDescriptor.from_tensor(O, [1, 1, 64, 64])
    desc_dO0 = TensorDescriptor.from_tensor(dO, [1, 1, 64, 64])
    desc_dO1 = TensorDescriptor.from_tensor(dO, [1, 1, 64, 64])
    
    desc_dQ0 = TensorDescriptor.from_tensor(dQ, [1, 1, 64, 64])
    desc_dQ1 = TensorDescriptor.from_tensor(dQ, [1, 1, 64, 64])
    
    desc_dK0 = TensorDescriptor.from_tensor(dK, [1, 1, 64, 64])
    desc_dK1 = TensorDescriptor.from_tensor(dK, [1, 1, 64, 64])
    desc_dV0 = TensorDescriptor.from_tensor(dV, [1, 1, 64, 64])
    desc_dV1 = TensorDescriptor.from_tensor(dV, [1, 1, 64, 64])
    
    num_blocks_q = triton.cdiv(S, 64)
    grid_dq = (num_blocks_q, H, B)
    
    _mha_bwd_dq_kernel[grid_dq](
        desc_Q0, desc_Q1, desc_K0, desc_K1, desc_V0, desc_V1, desc_O0, desc_O1, desc_dO0, desc_dO1, desc_dQ0, desc_dQ1,
        L,
        H, S, scale,
        BLOCK_M=64, BLOCK_N=64,
        num_warps=8,
        num_stages=3,
    )
    
    num_blocks_k = triton.cdiv(S, 64)
    grid_dk = (num_blocks_k, H, B)
    
    _mha_bwd_dk_dv_kernel[grid_dk](
        desc_Q0, desc_Q1, desc_K0, desc_K1, desc_V0, desc_V1, desc_O0, desc_O1, desc_dO0, desc_dO1, desc_dK0, desc_dK1, desc_dV0, desc_dV1,
        L,
        H, S, scale,
        BLOCK_M=64, BLOCK_N=64,
        num_warps=8,
        num_stages=3,
    )