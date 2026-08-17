import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

B_r = 64
B_c = 64


@triton.jit
def _bwd_dQ_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L_desc, dQ_desc,
    S_len, scale, num_heads: tl.constexpr,
    B_r: tl.constexpr, B_c: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_z = tl.program_id(2)
    
    bh = pid_z * num_heads + pid_y
    offset_s = pid_x * B_r
    
    # Load full d=128 tiles as float32 to ensure correct `tl.dot` promotion
    Q_raw = Q_desc.load([bh, offset_s, 0])
    Q = tl.squeeze(Q_raw, dim=0).to(tl.float32)
    
    dO_raw = dO_desc.load([bh, offset_s, 0])
    dO = tl.squeeze(dO_raw, dim=0).to(tl.float32)
    
    O_raw = O_desc.load([bh, offset_s, 0])
    O = tl.squeeze(O_raw, dim=0).to(tl.float32)
    
    # Preprocess D = rowsum(dO * O)
    D_i = tl.sum(dO * O, axis=1)
    
    L_i_raw = L_desc.load([bh, offset_s])
    L_i = tl.squeeze(L_i_raw, dim=0)
    
    q_mask = (offset_s + tl.arange(0, B_r)) < S_len
    
    acc_dQ = tl.zeros((B_r, 128), tl.float32)
    
    # Iterate over Key and Value chunks
    for j in range(tl.cdiv(S_len, B_c)):
        K_raw = K_desc.load([bh, j * B_c, 0])
        K = tl.squeeze(K_raw, dim=0).to(tl.float32)
        
        V_raw = V_desc.load([bh, j * B_c, 0])
        V = tl.squeeze(V_raw, dim=0).to(tl.float32)
        
        # S = Q @ K^T
        S = tl.dot(Q, K.T)
        # dP = dO @ V^T
        dP = tl.dot(dO, V.T)
        
        k_mask = (j * B_c + tl.arange(0, B_c)) < S_len
        mask = q_mask[:, None] & k_mask[None, :]
        
        # P = exp(S * scale - L)
        P = tl.exp(S * scale - L_i[:, None]) * mask
        
        # dS = P * (dP - D) * scale
        dS = P * (dP - D_i[:, None]) * scale
        
        # dQ += dS @ K
        acc_dQ = tl.dot(dS, K, acc_dQ)
    
    dQ_desc.store([bh, offset_s, 0], tl.unsqueeze(acc_dQ, dim=0).to(tl.bfloat16))


@triton.jit
def _bwd_dKdV_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L_desc, dK_desc, dV_desc,
    S_len, scale, num_heads: tl.constexpr,
    B_r: tl.constexpr, B_c: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_z = tl.program_id(2)
    
    bh = pid_z * num_heads + pid_y
    offset_s = pid_x * B_c
    
    # Load persistent Key and Value info
    K_raw = K_desc.load([bh, offset_s, 0])
    K = tl.squeeze(K_raw, dim=0).to(tl.float32)
    
    V_raw = V_desc.load([bh, offset_s, 0])
    V = tl.squeeze(V_raw, dim=0).to(tl.float32)
    
    acc_dK = tl.zeros((B_c, 128), tl.float32)
    acc_dV = tl.zeros((B_c, 128), tl.float32)
    
    k_mask = (offset_s + tl.arange(0, B_c)) < S_len
    
    # Iterate over Query chunks
    for i in range(tl.cdiv(S_len, B_r)):
        Q_raw = Q_desc.load([bh, i * B_r, 0])
        Q = tl.squeeze(Q_raw, dim=0).to(tl.float32)
        
        dO_raw = dO_desc.load([bh, i * B_r, 0])
        dO = tl.squeeze(dO_raw, dim=0).to(tl.float32)
        
        O_raw = O_desc.load([bh, i * B_r, 0])
        O = tl.squeeze(O_raw, dim=0).to(tl.float32)
        
        D_i = tl.sum(dO * O, axis=1)
        
        L_i_raw = L_desc.load([bh, i * B_r])
        L_i = tl.squeeze(L_i_raw, dim=0)
        
        # S = Q @ K^T
        S = tl.dot(Q, K.T)
        # dP = dO @ V^T
        dP = tl.dot(dO, V.T)
        
        q_mask = (i * B_r + tl.arange(0, B_r)) < S_len
        mask = q_mask[:, None] & k_mask[None, :]
        
        # P = exp(S * scale - L)
        P = tl.exp(S * scale - L_i[:, None]) * mask
        
        # dS = P * (dP - D) * scale
        dS = P * (dP - D_i[:, None]) * scale
        
        # dK += dS^T @ Q
        acc_dK = tl.dot(dS.T, Q, acc_dK)
        
        # dV += P^T @ dO
        acc_dV = tl.dot(P.T, dO, acc_dV)
    
    dK_desc.store([bh, offset_s, 0], tl.unsqueeze(acc_dK, dim=0).to(tl.bfloat16))
    dV_desc.store([bh, offset_s, 0], tl.unsqueeze(acc_dV, dim=0).to(tl.bfloat16))


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
    
    Q_desc = TensorDescriptor.from_tensor(Q_3d, [1, B_r, 128])
    K_desc = TensorDescriptor.from_tensor(K_3d, [1, B_c, 128])
    V_desc = TensorDescriptor.from_tensor(V_3d, [1, B_c, 128])
    O_desc = TensorDescriptor.from_tensor(O_3d, [1, B_r, 128])
    dO_desc = TensorDescriptor.from_tensor(dO_3d, [1, B_r, 128])
    dQ_desc = TensorDescriptor.from_tensor(dQ_3d, [1, B_r, 128])
    dK_desc = TensorDescriptor.from_tensor(dK_3d, [1, B_c, 128])
    dV_desc = TensorDescriptor.from_tensor(dV_3d, [1, B_c, 128])
    
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