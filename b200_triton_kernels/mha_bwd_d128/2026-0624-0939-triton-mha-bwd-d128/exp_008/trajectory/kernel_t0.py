import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

B_r = 64
B_c = 64


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
    B_r: tl.constexpr,
    B_c: tl.constexpr,
):
    pid_x = tl.program_id(0)  # queries block
    pid_y = tl.program_id(1)  # head
    pid_z = tl.program_id(2)  # batch
    
    bh = pid_z * num_heads + pid_y
    offset_s = pid_x * B_r
    
    # Load persistent Query info
    Q0_raw = Q_desc.load([bh, offset_s, 0])
    Q1_raw = Q_desc.load([bh, offset_s, 64])
    Q0 = tl.squeeze(Q0_raw, dim=0)
    Q1 = tl.squeeze(Q1_raw, dim=0)
    
    dO0_raw = dO_desc.load([bh, offset_s, 0])
    dO1_raw = dO_desc.load([bh, offset_s, 64])
    dO0 = tl.squeeze(dO0_raw, dim=0)
    dO1 = tl.squeeze(dO1_raw, dim=0)
    
    O0_raw = O_desc.load([bh, offset_s, 0])
    O1_raw = O_desc.load([bh, offset_s, 64])
    O0 = tl.squeeze(O0_raw, dim=0)
    O1 = tl.squeeze(O1_raw, dim=0)
    
    # Preprocess D = rowsum(dO * O)
    D_i = tl.sum(dO0 * O0 + dO1 * O1, axis=1)
    
    # Load LogSumExp
    L_i_raw = L_desc.load([bh, offset_s])
    L_i = tl.squeeze(L_i_raw, dim=0)
    
    acc_dQ0 = tl.zeros((B_r, B_c), tl.float32)
    acc_dQ1 = tl.zeros((B_r, B_c), tl.float32)
    
    # Iterate over Key and Value chunks
    for j in range(tl.cdiv(S_len, B_c)):
        K0_raw = K_desc.load([bh, j * B_c, 0])
        K1_raw = K_desc.load([bh, j * B_c, 64])
        K0 = tl.squeeze(K0_raw, dim=0)
        K1 = tl.squeeze(K1_raw, dim=0)
        
        V0_raw = V_desc.load([bh, j * B_c, 0])
        V1_raw = V_desc.load([bh, j * B_c, 64])
        V0 = tl.squeeze(V0_raw, dim=0)
        V1 = tl.squeeze(V1_raw, dim=0)
        
        # S = Q @ K^T
        S = tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)
        # dP = dO @ V^T
        dP = tl.dot(dO0, V0.T) + tl.dot(dO1, V1.T)
        
        k_mask = (j * B_c + tl.arange(0, B_c)) < S_len
        
        # P = exp(S * scale - L)
        P = tl.exp(S * scale - L_i[:, None]) * k_mask[None, :]
        
        # dS = P * (dP - D) * scale
        dS = P * (dP - D_i[:, None]) * scale
        
        # dQ += dS @ K
        acc_dQ0 = tl.dot(dS, K0, acc_dQ0)
        acc_dQ1 = tl.dot(dS, K1, acc_dQ1)
        
    dQ_desc.store([bh, offset_s, 0], tl.unsqueeze(acc_dQ0, dim=0).to(tl.bfloat16))
    dQ_desc.store([bh, offset_s, 64], tl.unsqueeze(acc_dQ1, dim=0).to(tl.bfloat16))


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
    B_r: tl.constexpr,
    B_c: tl.constexpr,
):
    pid_x = tl.program_id(0)  # kv block
    pid_y = tl.program_id(1)  # head
    pid_z = tl.program_id(2)  # batch
    
    bh = pid_z * num_heads + pid_y
    offset_s = pid_x * B_c
    
    # Load persistent Key and Value info
    K0_raw = K_desc.load([bh, offset_s, 0])
    K1_raw = K_desc.load([bh, offset_s, 64])
    K0 = tl.squeeze(K0_raw, dim=0)
    K1 = tl.squeeze(K1_raw, dim=0)
    
    V0_raw = V_desc.load([bh, offset_s, 0])
    V1_raw = V_desc.load([bh, offset_s, 64])
    V0 = tl.squeeze(V0_raw, dim=0)
    V1 = tl.squeeze(V1_raw, dim=0)
    
    acc_dK0 = tl.zeros((B_c, B_r), tl.float32)
    acc_dK1 = tl.zeros((B_c, B_r), tl.float32)
    acc_dV0 = tl.zeros((B_c, B_r), tl.float32)
    acc_dV1 = tl.zeros((B_c, B_r), tl.float32)
    
    # Iterate over Query chunks
    for i in range(tl.cdiv(S_len, B_r)):
        Q0_raw = Q_desc.load([bh, i * B_r, 0])
        Q1_raw = Q_desc.load([bh, i * B_r, 64])
        Q0 = tl.squeeze(Q0_raw, dim=0)
        Q1 = tl.squeeze(Q1_raw, dim=0)
        
        dO0_raw = dO_desc.load([bh, i * B_r, 0])
        dO1_raw = dO_desc.load([bh, i * B_r, 64])
        dO0 = tl.squeeze(dO0_raw, dim=0)
        dO1 = tl.squeeze(dO1_raw, dim=0)
        
        O0_raw = O_desc.load([bh, i * B_r, 0])
        O1_raw = O_desc.load([bh, i * B_r, 64])
        O0 = tl.squeeze(O0_raw, dim=0)
        O1 = tl.squeeze(O1_raw, dim=0)
        
        D_i = tl.sum(dO0 * O0 + dO1 * O1, axis=1)
        L_i_raw = L_desc.load([bh, i * B_r])
        L_i = tl.squeeze(L_i_raw, dim=0)
        
        # S = Q @ K^T
        S = tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)
        # dP = dO @ V^T
        dP = tl.dot(dO0, V0.T) + tl.dot(dO1, V1.T)
        
        # P = exp(S * scale - L)
        P = tl.exp(S * scale - L_i[:, None])
        
        k_mask = (pid_x * B_c + tl.arange(0, B_c)) < S_len
        q_mask = (i * B_r + tl.arange(0, B_r)) < S_len
        
        P = P * k_mask[None, :] * q_mask[:, None]
        
        # dS = P * (dP - D) * scale
        dS = P * (dP - D_i[:, None]) * scale
        
        # dK += dS^T @ Q
        acc_dK0 = tl.dot(dS.T, Q0, acc_dK0)
        acc_dK1 = tl.dot(dS.T, Q1, acc_dK1)
        
        # dV += P^T @ dO
        acc_dV0 = tl.dot(P.T, dO0, acc_dV0)
        acc_dV1 = tl.dot(P.T, dO1, acc_dV1)
        
    dK_desc.store([bh, offset_s, 0], tl.unsqueeze(acc_dK0, dim=0).to(tl.bfloat16))
    dK_desc.store([bh, offset_s, 64], tl.unsqueeze(acc_dK1, dim=0).to(tl.bfloat16))
    dV_desc.store([bh, offset_s, 0], tl.unsqueeze(acc_dV0, dim=0).to(tl.bfloat16))
    dV_desc.store([bh, offset_s, 64], tl.unsqueeze(acc_dV1, dim=0).to(tl.bfloat16))


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
    
    Q_desc = TensorDescriptor.from_tensor(Q_3d, [1, B_r, 64])
    K_desc = TensorDescriptor.from_tensor(K_3d, [1, B_c, 64])
    V_desc = TensorDescriptor.from_tensor(V_3d, [1, B_c, 64])
    O_desc = TensorDescriptor.from_tensor(O_3d, [1, B_r, 64])
    dO_desc = TensorDescriptor.from_tensor(dO_3d, [1, B_r, 64])
    dQ_desc = TensorDescriptor.from_tensor(dQ_3d, [1, B_r, 64])
    dK_desc = TensorDescriptor.from_tensor(dK_3d, [1, B_c, 64])
    dV_desc = TensorDescriptor.from_tensor(dV_3d, [1, B_c, 64])
    
    L_2d = L.view(N, S)
    L_desc = TensorDescriptor.from_tensor(L_2d, [1, B_r])
    
    grid_dQ = (triton.cdiv(S, B_r), H, B)
    _bwd_dQ_kernel[grid_dQ](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L_desc, dQ_desc,
        S, scale, H, B_r, B_c,
        num_warps=8, num_stages=2
    )
    
    grid_dKdV = (triton.cdiv(S, B_c), H, B)
    _bwd_dKdV_kernel[grid_dKdV](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L_desc, dK_desc, dV_desc,
        S, scale, H, B_r, B_c,
        num_warps=8, num_stages=2
    )