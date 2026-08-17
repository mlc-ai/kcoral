import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _bwd_dQ_kernel(
    Q_desc,
    K_desc,
    V_desc,
    O_desc,
    dO_desc,
    L_desc,
    dQ_desc,
    S_len,
    scale,
    num_heads: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_z = tl.program_id(2)
    
    bh = pid_z * num_heads + pid_y
    start_seq = pid_x * 128
    
    # Load persistent Query info
    Q_raw = Q_desc.load([bh, start_seq, 0])
    Q = tl.squeeze(Q_raw, dim=0).to(tl.float32)
    
    dO_raw = dO_desc.load([bh, start_seq, 0])
    dO = tl.squeeze(dO_raw, dim=0).to(tl.float32)
    
    O_raw = O_desc.load([bh, start_seq, 0])
    O = tl.squeeze(O_raw, dim=0).to(tl.float32)
    
    # Preprocess D = rowsum(dO * O)
    D_i = tl.sum(dO * O, axis=1)
    
    # Load LogSumExp
    L_i_raw = L_desc.load([bh, start_seq])
    L_i = tl.squeeze(L_i_raw, dim=0)
    
    q_mask = (start_seq + tl.arange(0, 128)) < S_len
    
    acc_dQ = tl.zeros((128, 128), tl.float32)
    num_k = tl.cdiv(S_len, 128)
    
    # Iterate over Key and Value chunks
    for j in tl.range(num_k, num_stages=3):
        K_raw = K_desc.load([bh, j * 128, 0])
        K = tl.squeeze(K_raw, dim=0).to(tl.float32)
        
        V_raw = V_desc.load([bh, j * 128, 0])
        V = tl.squeeze(V_raw, dim=0).to(tl.float32)
        
        # S = Q @ K^T
        S = tl.dot(Q, K.T)
        # dP = dO @ V^T
        dP = tl.dot(dO, V.T)
        
        k_mask = (j * 128 + tl.arange(0, 128)) < S_len
        mask = q_mask[:, None] & k_mask[None, :]
        
        # Safely handle invalid S values to avoid NaNs in exp
        S = tl.where(mask, S, -float('inf'))
        
        # P = exp(S * scale - L)
        P = tl.exp(S * scale - L_i[:, None])
        
        # dS = P * (dP - D) * scale
        dS = P * (dP - D_i[:, None]) * scale
        
        # dQ += dS @ K
        acc_dQ = tl.dot(dS, K, acc_dQ)
    
    dQ_desc.store([bh, start_seq, 0], acc_dQ.to(tl.bfloat16))


@triton.jit
def _bwd_dKdV_kernel(
    Q_desc,
    K_desc,
    V_desc,
    O_desc,
    dO_desc,
    L_desc,
    dK_desc,
    dV_desc,
    S_len,
    scale,
    num_heads: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_z = tl.program_id(2)
    
    bh = pid_z * num_heads + pid_y
    start_seq_k = pid_x * 128
    
    # Load persistent Key and Value info
    K_raw = K_desc.load([bh, start_seq_k, 0])
    K = tl.squeeze(K_raw, dim=0).to(tl.float32)
    
    V_raw = V_desc.load([bh, start_seq_k, 0])
    V = tl.squeeze(V_raw, dim=0).to(tl.float32)
    
    acc_dK = tl.zeros((128, 128), tl.float32)
    acc_dV = tl.zeros((128, 128), tl.float32)
    
    k_mask = (start_seq_k + tl.arange(0, 128)) < S_len
    
    num_q = tl.cdiv(S_len, 128)
    
    # Iterate over Query chunks
    for i in tl.range(num_q, num_stages=3):
        Q_raw = Q_desc.load([bh, i * 128, 0])
        Q = tl.squeeze(Q_raw, dim=0).to(tl.float32)
        
        dO_raw = dO_desc.load([bh, i * 128, 0])
        dO = tl.squeeze(dO_raw, dim=0).to(tl.float32)
        
        O_raw = O_desc.load([bh, i * 128, 0])
        O = tl.squeeze(O_raw, dim=0).to(tl.float32)
        
        D_i = tl.sum(dO * O, axis=1)
        
        L_i_raw = L_desc.load([bh, i * 128])
        L_i = tl.squeeze(L_i_raw, dim=0)
        
        # S = Q @ K^T
        S = tl.dot(Q, K.T)
        # dP = dO @ V^T
        dP = tl.dot(dO, V.T)
        
        q_mask = (i * 128 + tl.arange(0, 128)) < S_len
        mask = q_mask[:, None] & k_mask[None, :]
        
        # Safely handle invalid S values to avoid NaNs in exp
        S = tl.where(mask, S, -float('inf'))
        
        # P = exp(S * scale - L)
        P = tl.exp(S * scale - L_i[:, None])
        
        # dS = P * (dP - D) * scale
        dS = P * (dP - D_i[:, None]) * scale
        
        # dK += dS^T @ Q
        acc_dK = tl.dot(dS.T, Q, acc_dK)
        
        # dV += P^T @ dO
        acc_dV = tl.dot(P.T, dO, acc_dV)
    
    dK_desc.store([bh, start_seq_k, 0], acc_dK.to(tl.bfloat16))
    dV_desc.store([bh, start_seq_k, 0], acc_dV.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    scale = 1.0 / (d ** 0.5)
    
    N = B * H
    Q_3d = Q.view(N, S, d)
    K_3d = K.view(N, S, d)
    V_3d = V.view(N, S, d)
    O_3d = O.view(N, S, d)
    dO_3d = dO.view(N, S, d)
    dQ_3d = dQ.view(N, S, d)
    dK_3d = dK.view(N, S, d)
    dV_3d = dV.view(N, S, d)
    
    Q_desc = TensorDescriptor.from_tensor(Q_3d, [1, 128, 128])
    K_desc = TensorDescriptor.from_tensor(K_3d, [1, 128, 128])
    V_desc = TensorDescriptor.from_tensor(V_3d, [1, 128, 128])
    O_desc = TensorDescriptor.from_tensor(O_3d, [1, 128, 128])
    dO_desc = TensorDescriptor.from_tensor(dO_3d, [1, 128, 128])
    dQ_desc = TensorDescriptor.from_tensor(dQ_3d, [1, 128, 128])
    dK_desc = TensorDescriptor.from_tensor(dK_3d, [1, 128, 128])
    dV_desc = TensorDescriptor.from_tensor(dV_3d, [1, 128, 128])
    
    L_2d = L.view(N, S)
    L_desc = TensorDescriptor.from_tensor(L_2d, [1, 128])
    
    grid_dQ = (triton.cdiv(S, 128), H, B)
    _bwd_dQ_kernel[grid_dQ](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L_desc, dQ_desc,
        S, scale, H, num_warps=8, num_stages=3
    )
    
    grid_dKdV = (triton.cdiv(S, 128), H, B)
    _bwd_dKdV_kernel[grid_dKdV](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L_desc, dK_desc, dV_desc,
        S, scale, H, num_warps=8, num_stages=3
    )