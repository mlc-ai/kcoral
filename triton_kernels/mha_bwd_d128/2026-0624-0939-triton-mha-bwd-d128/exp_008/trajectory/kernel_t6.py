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
    H: tl.constexpr,
    B: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_z = tl.program_id(2)
    
    bh = pid_z * H + pid_y
    start_seq = pid_x * 64
    
    # Load persistent Query info (split d=128 into two 64-wide chunks)
    Q0_raw = Q_desc.load([bh, start_seq, 0])
    Q0 = tl.squeeze(Q0_raw, dim=0).to(tl.float32)
    Q1_raw = Q_desc.load([bh, start_seq, 64])
    Q1 = tl.squeeze(Q1_raw, dim=0).to(tl.float32)
    
    dO0_raw = dO_desc.load([bh, start_seq, 0])
    dO0 = tl.squeeze(dO0_raw, dim=0).to(tl.float32)
    dO1_raw = dO_desc.load([bh, start_seq, 64])
    dO1 = tl.squeeze(dO1_raw, dim=0).to(tl.float32)
    
    O0_raw = O_desc.load([bh, start_seq, 0])
    O0 = tl.squeeze(O0_raw, dim=0).to(tl.float32)
    O1_raw = O_desc.load([bh, start_seq, 64])
    O1 = tl.squeeze(O1_raw, dim=0).to(tl.float32)
    
    # Preprocess D = rowsum(dO * O)
    D_i = tl.sum(dO0 * O0 + dO1 * O1, axis=1)
    
    # Load LogSumExp
    L_i_raw = L_desc.load([bh, start_seq])
    L_i = tl.squeeze(L_i_raw, dim=0)
    
    q_mask = (start_seq + tl.arange(0, 64)) < S_len
    
    acc_dQ0 = tl.zeros((64, 64), tl.float32)
    acc_dQ1 = tl.zeros((64, 64), tl.float32)
    
    num_k = triton.cdiv(S_len, 64)
    
    # Iterate over Key and Value chunks
    for j in range(num_k):
        K0_raw = K_desc.load([bh, j * 64, 0])
        K0 = tl.squeeze(K0_raw, dim=0).to(tl.float32)
        K1_raw = K_desc.load([bh, j * 64, 64])
        K1 = tl.squeeze(K1_raw, dim=0).to(tl.float32)
        
        V0_raw = V_desc.load([bh, j * 64, 0])
        V0 = tl.squeeze(V0_raw, dim=0).to(tl.float32)
        V1_raw = V_desc.load([bh, j * 64, 64])
        V1 = tl.squeeze(V1_raw, dim=0).to(tl.float32)
        
        # S = Q @ K^T
        S = tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)
        # dP = dO @ V^T
        dP = tl.dot(dO0, V0.T) + tl.dot(dO1, V1.T)
        
        k_mask = (j * 64 + tl.arange(0, 64)) < S_len
        mask = q_mask[:, None] & k_mask[None, :]
        
        # Safely handle invalid S values to avoid NaNs in exp
        S = tl.where(mask, S, -float('inf'))
        
        # P = exp(S * scale - L)
        P = tl.exp(S * scale - L_i[:, None])
        
        # dS = P * (dP - D) * scale
        dS = P * (dP - D_i[:, None]) * scale
        
        # dQ += dS @ K
        acc_dQ0 = tl.dot(dS, K0, acc_dQ0)
        acc_dQ1 = tl.dot(dS, K1, acc_dQ1)
    
    dQ_desc.store([bh, start_seq, 0], acc_dQ0.to(tl.bfloat16))
    dQ_desc.store([bh, start_seq, 64], acc_dQ1.to(tl.bfloat16))


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
    H: tl.constexpr,
    B: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_z = tl.program_id(2)
    
    bh = pid_z * H + pid_y
    start_seq_k = pid_x * 64
    
    # Load persistent Key and Value info (split d=128 into two 64-wide chunks)
    K0_raw = K_desc.load([bh, start_seq_k, 0])
    K0 = tl.squeeze(K0_raw, dim=0).to(tl.float32)
    K1_raw = K_desc.load([bh, start_seq_k, 64])
    K1 = tl.squeeze(K1_raw, dim=0).to(tl.float32)
    
    V0_raw = V_desc.load([bh, start_seq_k, 0])
    V0 = tl.squeeze(V0_raw, dim=0).to(tl.float32)
    V1_raw = V_desc.load([bh, start_seq_k, 64])
    V1 = tl.squeeze(V1_raw, dim=0).to(tl.float32)
    
    acc_dK0 = tl.zeros((64, 64), tl.float32)
    acc_dK1 = tl.zeros((64, 64), tl.float32)
    acc_dV0 = tl.zeros((64, 64), tl.float32)
    acc_dV1 = tl.zeros((64, 64), tl.float32)
    
    k_mask = (start_seq_k + tl.arange(0, 64)) < S_len
    
    num_q = triton.cdiv(S_len, 64)
    
    # Iterate over Query chunks
    for i in range(num_q):
        Q0_raw = Q_desc.load([bh, i * 64, 0])
        Q0 = tl.squeeze(Q0_raw, dim=0).to(tl.float32)
        Q1_raw = Q_desc.load([bh, i * 64, 64])
        Q1 = tl.squeeze(Q1_raw, dim=0).to(tl.float32)
        
        dO0_raw = dO_desc.load([bh, i * 64, 0])
        dO0 = tl.squeeze(dO0_raw, dim=0).to(tl.float32)
        dO1_raw = dO_desc.load([bh, i * 64, 64])
        dO1 = tl.squeeze(dO1_raw, dim=0).to(tl.float32)
        
        O0_raw = O_desc.load([bh, i * 64, 0])
        O0 = tl.squeeze(O0_raw, dim=0).to(tl.float32)
        O1_raw = O_desc.load([bh, i * 64, 64])
        O1 = tl.squeeze(O1_raw, dim=0).to(tl.float32)
        
        D_i = tl.sum(dO0 * O0 + dO1 * O1, axis=1)
        
        L_i_raw = L_desc.load([bh, i * 64])
        L_i = tl.squeeze(L_i_raw, dim=0)
        
        # S = Q @ K^T
        S = tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)
        # dP = dO @ V^T
        dP = tl.dot(dO0, V0.T) + tl.dot(dO1, V1.T)
        
        q_mask = (i * 64 + tl.arange(0, 64)) < S_len
        mask = q_mask[:, None] & k_mask[None, :]
        
        # Safely handle invalid S values to avoid NaNs in exp
        S = tl.where(mask, S, -float('inf'))
        
        # P = exp(S * scale - L)
        P = tl.exp(S * scale - L_i[:, None])
        
        # dS = P * (dP - D) * scale
        dS = P * (dP - D_i[:, None]) * scale
        
        # dK += dS^T @ Q
        acc_dK0 = tl.dot(dS.T, Q0, acc_dK0)
        acc_dK1 = tl.dot(dS.T, Q1, acc_dK1)
        
        # dV += P^T @ dO
        acc_dV0 = tl.dot(P.T, dO0, acc_dV0)
        acc_dV1 = tl.dot(P.T, dO1, acc_dV1)
    
    dK_desc.store([bh, start_seq_k, 0], acc_dK0.to(tl.bfloat16))
    dK_desc.store([bh, start_seq_k, 64], acc_dK1.to(tl.bfloat16))
    dV_desc.store([bh, start_seq_k, 0], acc_dV0.to(tl.bfloat16))
    dV_desc.store([bh, start_seq_k, 64], acc_dV1.to(tl.bfloat16))


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
    
    Q_desc = TensorDescriptor.from_tensor(Q_3d, [1, 64, 64])
    K_desc = TensorDescriptor.from_tensor(K_3d, [1, 64, 64])
    V_desc = TensorDescriptor.from_tensor(V_3d, [1, 64, 64])
    O_desc = TensorDescriptor.from_tensor(O_3d, [1, 64, 64])
    dO_desc = TensorDescriptor.from_tensor(dO_3d, [1, 64, 64])
    dQ_desc = TensorDescriptor.from_tensor(dQ_3d, [1, 64, 64])
    dK_desc = TensorDescriptor.from_tensor(dK_3d, [1, 64, 64])
    dV_desc = TensorDescriptor.from_tensor(dV_3d, [1, 64, 64])
    
    L_2d = L.view(N, S)
    L_desc = TensorDescriptor.from_tensor(L_2d, [1, 64])
    
    grid_dQ = (triton.cdiv(S, 64), H, B)
    _bwd_dQ_kernel[grid_dQ](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L_desc, dQ_desc,
        S, scale, H, B,
        num_warps=8, num_stages=3
    )
    
    grid_dKdV = (triton.cdiv(S, 64), H, B)
    _bwd_dKdV_kernel[grid_dKdV](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L_desc, dK_desc, dV_desc,
        S, scale, H, B,
        num_warps=8, num_stages=3
    )