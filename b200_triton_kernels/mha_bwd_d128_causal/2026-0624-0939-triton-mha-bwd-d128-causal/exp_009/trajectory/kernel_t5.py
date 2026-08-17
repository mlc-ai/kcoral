import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _bwd_dq_kernel(
    desc_Q, desc_K, desc_V, desc_dO, desc_O, desc_dQ, desc_L,
    S, tau, num_blocks_causal,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    i = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    
    off_q = i * BLOCK_M
    
    Q_i = desc_Q.load([b, h, off_q, 0])
    Q_i = tl.reshape(Q_i, [BLOCK_M, 128])
    
    dO_i = desc_dO.load([b, h, off_q, 0])
    dO_i = tl.reshape(dO_i, [BLOCK_M, 128])
    
    L_i = desc_L.load([b, h, off_q])
    L_i = tl.reshape(L_i, [BLOCK_M])
    
    O_i = desc_O.load([b, h, off_q, 0])
    O_i = tl.reshape(O_i, [BLOCK_M, 128])
    
    D_i = tl.sum(dO_i * O_i, axis=1)
    
    dQ_i = tl.zeros((BLOCK_M, 128), tl.float32)
    
    q_pos = off_q + tl.arange(0, BLOCK_M)
    
    for j in range(min(i + 1, num_blocks_causal)):
        off_k = j * BLOCK_N
        K_j = desc_K.load([b, h, off_k, 0])
        K_j = tl.reshape(K_j, [BLOCK_N, 128])
        
        V_j = desc_V.load([b, h, off_k, 0])
        V_j = tl.reshape(V_j, [BLOCK_N, 128])
        
        raw_S = tl.dot(Q_i, K_j.T)
        raw_dP = tl.dot(dO_i, V_j.T)
        
        k_pos = off_k + tl.arange(0, BLOCK_N)
        
        S_mat = raw_S * tau
        
        P = tl.exp(S_mat - L_i[:, None])
        dS = P * (raw_dP - D_i[:, None]) * tau
        
        mask = (q_pos[:, None] >= k_pos[None, :]) & (q_pos[:, None] < S) & (k_pos[None, :] < S)
        
        P = P * mask
        dS = dS * mask
        
        dQ_i = tl.dot(dS, K_j, acc=dQ_i)
    
    desc_dQ.store([b, h, off_q, 0], dQ_i.to(tl.bfloat16), boundary_check=(2,))


@triton.jit
def _bwd_dkv_kernel(
    desc_Q, desc_K, desc_V, desc_dO, desc_O, desc_dK, desc_dV, desc_L,
    S, tau, num_blocks_causal,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    j = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    
    off_k = j * BLOCK_N
    
    K_j = desc_K.load([b, h, off_k, 0])
    K_j = tl.reshape(K_j, [BLOCK_N, 128])
    
    V_j = desc_V.load([b, h, off_k, 0])
    V_j = tl.reshape(V_j, [BLOCK_N, 128])
    
    dK_j = tl.zeros((BLOCK_N, 128), tl.float32)
    dV_j = tl.zeros((BLOCK_N, 128), tl.float32)
    
    k_pos = off_k + tl.arange(0, BLOCK_N)
    
    for i in range(j, num_blocks_causal):
        off_q = i * BLOCK_M
        Q_i = desc_Q.load([b, h, off_q, 0])
        Q_i = tl.reshape(Q_i, [BLOCK_M, 128])
        
        dO_i = desc_dO.load([b, h, off_q, 0])
        dO_i = tl.reshape(dO_i, [BLOCK_M, 128])
        
        L_i = desc_L.load([b, h, off_q])
        L_i = tl.reshape(L_i, [BLOCK_M])
        
        O_i = desc_O.load([b, h, off_q, 0])
        O_i = tl.reshape(O_i, [BLOCK_M, 128])
        
        D_i = tl.sum(dO_i * O_i, axis=1)
        
        raw_S = tl.dot(Q_i, K_j.T)
        raw_dP = tl.dot(dO_i, V_j.T)
        
        q_pos = off_q + tl.arange(0, BLOCK_M)
        
        S_mat = raw_S * tau
        
        P = tl.exp(S_mat - L_i[:, None])
        dS = P * (raw_dP - D_i[:, None]) * tau
        
        mask = (q_pos[:, None] >= k_pos[None, :]) & (q_pos[:, None] < S) & (k_pos[None, :] < S)
        
        P_masked = P * mask
        dS_masked = dS * mask
        
        dV_j = tl.dot(P_masked.T, dO_i, acc=dV_j)
        dK_j = tl.dot(dS_masked.T, Q_i, acc=dK_j)
    
    desc_dK.store([b, h, off_k, 0], dK_j.to(tl.bfloat16), boundary_check=(2,))
    desc_dV.store([b, h, off_k, 0], dV_j.to(tl.bfloat16), boundary_check=(2,))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    assert d == 128
    tau = 1.0 / (d ** 0.5)
    
    BLOCK_M = 64
    BLOCK_N = 64
    
    desc_Q = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M, d])
    desc_K = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N, d])
    desc_V = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N, d])
    desc_dO = TensorDescriptor.from_tensor(dO, [1, 1, BLOCK_M, d])
    desc_O  = TensorDescriptor.from_tensor(O,  [1, 1, BLOCK_M, d])
    desc_dQ = TensorDescriptor.from_tensor(dQ, [1, 1, BLOCK_M, d])
    desc_dK = TensorDescriptor.from_tensor(dK, [1, 1, BLOCK_N, d])
    desc_dV = TensorDescriptor.from_tensor(dV, [1, 1, BLOCK_N, d])
    desc_L  = TensorDescriptor.from_tensor(L,  [1, 1, BLOCK_M])
    
    num_blocks_causal = triton.cdiv(S, BLOCK_M)
    
    grid_dQ = (num_blocks_causal, B, H)
    _bwd_dq_kernel[grid_dQ](
        desc_Q, desc_K, desc_V, desc_dO, desc_O, desc_dQ, desc_L,
        S, tau, num_blocks_causal,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_stages=3,
    )
    
    grid_dKV = (num_blocks_causal, B, H)
    _bwd_dkv_kernel[grid_dKV](
        desc_Q, desc_K, desc_V, desc_dO, desc_O, desc_dK, desc_dV, desc_L,
        S, tau, num_blocks_causal,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_stages=3,
    )