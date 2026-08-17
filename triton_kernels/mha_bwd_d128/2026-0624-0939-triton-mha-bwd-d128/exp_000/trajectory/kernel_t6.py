import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _compute_D_kernel(O_ptr, dO_ptr, D_ptr, d):
    idx = tl.program_id(0)
    if idx < S * H:
        b_h = idx // (S * H)
        s = idx % (S * H)
        offset = b_h * S * d + s * d + tl.arange(0, 128)
        o = tl.load(O_ptr + offset)
        do = tl.load(dO_ptr + offset)
        d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32))
        tl.store(D_ptr + idx, d_val)


@triton.jit
def _mha_bwd_dq_kernel(
    desc_Q, desc_K, desc_V, desc_O, desc_dO, desc_dQ,
    L_ptr, D_ptr,
    H, S, scale,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    pid_m = tl.program_id(0)
    m_start = pid_m * 64
    
    Q0 = tl.reshape(desc_Q.load([b, h, m_start, 0]), [64, 64])
    Q1 = tl.reshape(desc_Q.load([b, h, m_start, 64]), [64, 64])
    
    O0 = tl.reshape(desc_O.load([b, h, m_start, 0]), [64, 64])
    O1 = tl.reshape(desc_O.load([b, h, m_start, 64]), [64, 64])
    
    dO0 = tl.reshape(desc_dO.load([b, h, m_start, 0]), [64, 64])
    dO1 = tl.reshape(desc_dO.load([b, h, m_start, 64]), [64, 64])
    
    l_idx = m_start + tl.arange(0, 64)
    L_vec = tl.load(L_ptr + b * H * S + h * S + l_idx, mask=(l_idx < S), other=0.0)
    D_vec = tl.load(D_ptr + b * H * S + h * S + l_idx, mask=(l_idx < S), other=0.0)
    
    acc_dQ0 = tl.zeros((64, 64), tl.float32)
    acc_dQ1 = tl.zeros((64, 64), tl.float32)
    
    for n_start in range(0, S, 64):
        K0 = tl.reshape(desc_K.load([b, h, n_start, 0]), [64, 64])
        K1 = tl.reshape(desc_K.load([b, h, n_start, 64]), [64, 64])
        
        V0 = tl.reshape(desc_V.load([b, h, n_start, 0]), [64, 64])
        V1 = tl.reshape(desc_V.load([b, h, n_start, 64]), [64, 64])
        
        S_val = (tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)) * scale 
        
        P = tl.exp(S_val - L_vec[:, None])
        
        k_idx = n_start + tl.arange(0, 64)
        mask = (l_idx[:, None] < S) & (k_idx[None, :] < S)
        P = P * mask
        
        dP = tl.dot(dO0, V0.T) + tl.dot(dO1, V1.T)
        
        dS = P * (dP - D_vec[:, None]) * scale
        dS = dS * mask
        
        dS_bf16 = dS.to(tl.bfloat16)
        acc_dQ0 = tl.dot(dS_bf16, K0, acc_dQ0)
        acc_dQ1 = tl.dot(dS_bf16, K1, acc_dQ1)
        
    desc_dQ.store([b, h, m_start, 0], tl.reshape(acc_dQ0, [1, 1, 64, 64]))
    desc_dQ.store([b, h, m_start, 64], tl.reshape(acc_dQ1, [1, 1, 64, 64]))


@triton.jit
def _mha_bwd_dk_dv_kernel(
    desc_Q, desc_K, desc_V, desc_O, desc_dO, desc_dK, desc_dV,
    L_ptr, D_ptr,
    H, S, scale,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    pid_n = tl.program_id(0)
    n_start = pid_n * 64
    
    K0 = tl.reshape(desc_K.load([b, h, n_start, 0]), [64, 64])
    K1 = tl.reshape(desc_K.load([b, h, n_start, 64]), [64, 64])
    
    V0 = tl.reshape(desc_V.load([b, h, n_start, 0]), [64, 64])
    V1 = tl.reshape(desc_V.load([b, h, n_start, 64]), [64, 64])
    
    acc_dK0 = tl.zeros((64, 64), tl.float32)
    acc_dK1 = tl.zeros((64, 64), tl.float32)
    acc_dV0 = tl.zeros((64, 64), tl.float32)
    acc_dV1 = tl.zeros((64, 64), tl.float32)
    
    for m_start in range(0, S, 64):
        Q0 = tl.reshape(desc_Q.load([b, h, m_start, 0]), [64, 64])
        Q1 = tl.reshape(desc_Q.load([b, h, m_start, 64]), [64, 64])
        
        O0 = tl.reshape(desc_O.load([b, h, m_start, 0]), [64, 64])
        O1 = tl.reshape(desc_O.load([b, h, m_start, 64]), [64, 64])
        
        dO0 = tl.reshape(desc_dO.load([b, h, m_start, 0]), [64, 64])
        dO1 = tl.reshape(desc_dO.load([b, h, m_start, 64]), [64, 64])
        
        l_idx = m_start + tl.arange(0, 64)
        L_vec = tl.load(L_ptr + b * H * S + h * S + l_idx, mask=(l_idx < S), other=0.0)
        D_vec = tl.load(D_ptr + b * H * S + h * S + l_idx, mask=(l_idx < S), other=0.0)
        
        S_val = (tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)) * scale 
        
        P = tl.exp(S_val - L_vec[:, None])
        
        k_idx = n_start + tl.arange(0, 64)
        mask = (l_idx[:, None] < S) & (k_idx[None, :] < S)
        P = P * mask
        
        dP = tl.dot(dO0, V0.T) + tl.dot(dO1, V1.T)
        
        dS = P * (dP - D_vec[:, None]) * scale
        dS = dS * mask
        
        dS_bf16 = dS.to(tl.bfloat16)
        acc_dK0 = tl.dot(dS_bf16.T, Q0, acc_dK0)
        acc_dK1 = tl.dot(dS_bf16.T, Q1, acc_dK1)
        
        P_bf16 = P.to(tl.bfloat16)
        acc_dV0 = tl.dot(P_bf16.T, dO0, acc_dV0)
        acc_dV1 = tl.dot(P_bf16.T, dO1, acc_dV1)
        
    desc_dK.store([b, h, n_start, 0], tl.reshape(acc_dK0, [1, 1, 64, 64]))
    desc_dK.store([b, h, n_start, 64], tl.reshape(acc_dK1, [1, 1, 64, 64]))
    desc_dV.store([b, h, n_start, 0], tl.reshape(acc_dV0, [1, 1, 64, 64]))
    desc_dV.store([b, h, n_start, 64], tl.reshape(acc_dV1, [1, 1, 64, 64]))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    scale = 1.0 / math.sqrt(d)
    
    D = torch.empty((B, H, S), dtype=torch.float32, device=Q.device)
    
    grid_D = (B * H * S,)
    _compute_D_kernel[grid_D](Q, K, V, O, dO, L, D, S, H, B, d)
    
    desc_Q = TensorDescriptor.from_tensor(Q, [1, 1, 64, 64])
    desc_K = TensorDescriptor.from_tensor(K, [1, 1, 64, 64])
    desc_V = TensorDescriptor.from_tensor(V, [1, 1, 64, 64])
    desc_O = TensorDescriptor.from_tensor(O, [1, 1, 64, 64])
    desc_dO = TensorDescriptor.from_tensor(dO, [1, 1, 64, 64])
    desc_dQ = TensorDescriptor.from_tensor(dQ, [1, 1, 64, 64])
    desc_dK = TensorDescriptor.from_tensor(dK, [1, 1, 64, 64])
    desc_dV = TensorDescriptor.from_tensor(dV, [1, 1, 64, 64])
    
    num_blocks = triton.cdiv(S, 64)
    grid = (num_blocks, H, B)
    
    _mha_bwd_dq_kernel[grid](
        desc_Q, desc_K, desc_V, desc_O, desc_dO, desc_dQ,
        L, D,
        H, S, scale,
        num_warps=4, num_stages=4,
    )
    
    _mha_bwd_dk_dv_kernel[grid](
        desc_Q, desc_K, desc_V, desc_O, desc_dO, desc_dK, desc_dV,
        L, D,
        H, S, scale,
        num_warps=4, num_stages=4,
    )