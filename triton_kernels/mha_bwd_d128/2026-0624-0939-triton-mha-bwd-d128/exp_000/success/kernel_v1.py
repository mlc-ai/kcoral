import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_bwd_dq_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L_ptr, dQ_desc,
    S, scale, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    pid_m = tl.program_id(0)
    m_start = pid_m * BLOCK_M
    
    Q_tile = tl.reshape(Q_desc.load([b, h, m_start, 0]), [BLOCK_M, 128])
    O_tile = tl.reshape(O_desc.load([b, h, m_start, 0]), [BLOCK_M, 128])
    dO_tile = tl.reshape(dO_desc.load([b, h, m_start, 0]), [BLOCK_M, 128])
    
    l_idx = m_start + tl.arange(0, BLOCK_M)
    L_vec = tl.load(L_ptr + b * H * S + h * S + l_idx, mask=(l_idx < S), other=0.0)
    
    D_vec = tl.sum(O_tile.to(tl.float32) * dO_tile.to(tl.float32), axis=1)
    
    acc_dQ = tl.zeros((BLOCK_M, 128), tl.float32)
    
    for n_start in range(0, S, BLOCK_N):
        K_tile = tl.reshape(K_desc.load([b, h, n_start, 0]), [BLOCK_N, 128])
        V_tile = tl.reshape(V_desc.load([b, h, n_start, 0]), [BLOCK_N, 128])
        
        S_val = tl.dot(Q_tile, K_tile.T) * scale
        P = tl.exp(S_val - L_vec[:, None])
        dP = tl.dot(dO_tile, V_tile.T)
        dS = P * (dP - D_vec[:, None]) * scale
        
        dS_bf16 = dS.to(tl.bfloat16)
        acc_dQ = tl.dot(dS_bf16, K_tile, acc_dQ)
    
    dQ_desc.store([b, h, m_start, 0], tl.reshape(acc_dQ, [1, 1, BLOCK_M, 128]))


@triton.jit
def _mha_bwd_dk_dv_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L_ptr, dK_desc, dV_desc,
    S, scale, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    pid_n = tl.program_id(0)
    n_start = pid_n * BLOCK_N
    
    K_tile = tl.reshape(K_desc.load([b, h, n_start, 0]), [BLOCK_N, 128])
    V_tile = tl.reshape(V_desc.load([b, h, n_start, 0]), [BLOCK_N, 128])
    
    acc_dK = tl.zeros((BLOCK_N, 128), tl.float32)
    acc_dV = tl.zeros((BLOCK_N, 128), tl.float32)
    
    for m_start in range(0, S, BLOCK_M):
        Q_tile = tl.reshape(Q_desc.load([b, h, m_start, 0]), [BLOCK_M, 128])
        O_tile = tl.reshape(O_desc.load([b, h, m_start, 0]), [BLOCK_M, 128])
        dO_tile = tl.reshape(dO_desc.load([b, h, m_start, 0]), [BLOCK_M, 128])
        
        l_idx = m_start + tl.arange(0, BLOCK_M)
        L_vec = tl.load(L_ptr + b * H * S + h * S + l_idx, mask=(l_idx < S), other=0.0)
        
        D_vec = tl.sum(O_tile.to(tl.float32) * dO_tile.to(tl.float32), axis=1)
        
        S_val = tl.dot(Q_tile, K_tile.T) * scale
        P = tl.exp(S_val - L_vec[:, None])
        dP = tl.dot(dO_tile, V_tile.T)
        dS = P * (dP - D_vec[:, None]) * scale
        
        dS_bf16 = dS.to(tl.bfloat16)
        acc_dK = tl.dot(dS_bf16.T, Q_tile, acc_dK)
        
        P_bf16 = P.to(tl.bfloat16)
        acc_dV = tl.dot(P_bf16.T, dO_tile, acc_dV)
    
    dK_desc.store([b, h, n_start, 0], tl.reshape(acc_dK, [1, 1, BLOCK_N, 128]))
    dV_desc.store([b, h, n_start, 0], tl.reshape(acc_dV, [1, 1, BLOCK_N, 128]))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    scale = 1.0 / math.sqrt(d)
    
    BLOCK_M = 64
    BLOCK_N = 64
    
    Q_desc = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M, 128])
    K_desc = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N, 128])
    V_desc = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N, 128])
    O_desc = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M, 128])
    dO_desc = TensorDescriptor.from_tensor(dO, [1, 1, BLOCK_M, 128])
    dQ_desc = TensorDescriptor.from_tensor(dQ, [1, 1, BLOCK_M, 128])
    dK_desc = TensorDescriptor.from_tensor(dK, [1, 1, BLOCK_N, 128])
    dV_desc = TensorDescriptor.from_tensor(dV, [1, 1, BLOCK_N, 128])
    
    num_blocks_q = triton.cdiv(S, BLOCK_M)
    grid_dq = (num_blocks_q, H, B)
    
    num_blocks_k = triton.cdiv(S, BLOCK_N)
    grid_dk = (num_blocks_k, H, B)
    
    _mha_bwd_dq_kernel[grid_dq](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dQ_desc,
        S, scale, H,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=8, num_stages=2,
    )
    
    _mha_bwd_dk_dv_kernel[grid_dk](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dK_desc, dV_desc,
        S, scale, H,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=8, num_stages=2,
    )