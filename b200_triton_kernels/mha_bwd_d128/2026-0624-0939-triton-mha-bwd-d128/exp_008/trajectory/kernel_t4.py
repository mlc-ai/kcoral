import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

BLOCK_Q = 128
BLOCK_K = 128


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
    
    start_seq = pid_x * BLOCK_Q
    
    # Load persistent Query info
    Q = Q_desc.load([pid_z, pid_y, start_seq, 0]).to(tl.float32)
    dO = dO_desc.load([pid_z, pid_y, start_seq, 0]).to(tl.float32)
    O = O_desc.load([pid_z, pid_y, start_seq, 0]).to(tl.float32)
    
    # Preprocess D = rowsum(dO * O)
    D_i = tl.sum(dO * O, axis=1)
    
    # Load LogSumExp
    L_i = L_desc.load([pid_z, pid_y, start_seq])
    
    q_mask = (start_seq + tl.arange(0, 128)) < S_len
    
    acc_dQ = tl.zeros((128, 128), tl.float32)
    num_k = triton.cdiv(S_len, BLOCK_K)
    
    # Iterate over Key and Value chunks
    for j in tl.range(num_k, num_stages=3):
        K = K_desc.load([pid_z, pid_y, j * BLOCK_K, 0]).to(tl.float32)
        V = V_desc.load([pid_z, pid_y, j * BLOCK_K, 0]).to(tl.float32)
        
        # S = Q @ K^T
        S = tl.dot(Q, K.T)
        # dP = dO @ V^T
        dP = tl.dot(dO, V.T)
        
        k_mask = (j * BLOCK_K + tl.arange(0, 128)) < S_len
        mask = q_mask[:, None] & k_mask[None, :]
        
        # P = exp(S * scale - L)
        P = tl.exp(S * scale - L_i[:, None]) * mask
        
        # dS = P * (dP - D) * scale
        dS = P * (dP - D_i[:, None]) * scale
        
        # dQ += dS @ K
        acc_dQ = tl.dot(dS, K, acc_dQ)
    
    dQ_desc.store([pid_z, pid_y, start_seq, 0], acc_dQ.to(tl.bfloat16))


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
    
    start_seq_k = pid_x * BLOCK_K
    
    # Load persistent Key and Value info
    K = K_desc.load([pid_z, pid_y, start_seq_k, 0]).to(tl.float32)
    V = V_desc.load([pid_z, pid_y, start_seq_k, 0]).to(tl.float32)
    
    acc_dK = tl.zeros((128, 128), tl.float32)
    acc_dV = tl.zeros((128, 128), tl.float32)
    
    k_mask = (start_seq_k + tl.arange(0, 128)) < S_len
    
    num_q = triton.cdiv(S_len, BLOCK_Q)
    
    # Iterate over Query chunks
    for i in tl.range(num_q, num_stages=3):
        Q = Q_desc.load([pid_z, pid_y, i * BLOCK_Q, 0]).to(tl.float32)
        dO = dO_desc.load([pid_z, pid_y, i * BLOCK_Q, 0]).to(tl.float32)
        O = O_desc.load([pid_z, pid_y, i * BLOCK_Q, 0]).to(tl.float32)
        
        D_i = tl.sum(dO * O, axis=1)
        L_i = L_desc.load([pid_z, pid_y, i * BLOCK_Q])
        
        # S = Q @ K^T
        S = tl.dot(Q, K.T)
        # dP = dO @ V^T
        dP = tl.dot(dO, V.T)
        
        q_mask = (i * BLOCK_Q + tl.arange(0, 128)) < S_len
        mask = k_mask[:, None] & q_mask[None, :]
        
        # P = exp(S * scale - L)
        P = tl.exp(S * scale - L_i[:, None]) * mask
        
        # dS = P * (dP - D) * scale
        dS = P * (dP - D_i[:, None]) * scale
        
        # dK += dS^T @ Q
        acc_dK = tl.dot(dS.T, Q, acc_dK)
        
        # dV += P^T @ dO
        acc_dV = tl.dot(P.T, dO, acc_dV)
    
    dK_desc.store([pid_z, pid_y, start_seq_k, 0], acc_dK.to(tl.bfloat16))
    dV_desc.store([pid_z, pid_y, start_seq_k, 0], acc_dV.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    scale = 1.0 / (d ** 0.5)
    
    Q_desc = TensorDescriptor.from_tensor(Q, [B, H, BLOCK_Q, 128])
    K_desc = TensorDescriptor.from_tensor(K, [B, H, BLOCK_K, 128])
    V_desc = TensorDescriptor.from_tensor(V, [B, H, BLOCK_K, 128])
    O_desc = TensorDescriptor.from_tensor(O, [B, H, BLOCK_Q, 128])
    dO_desc = TensorDescriptor.from_tensor(dO, [B, H, BLOCK_Q, 128])
    dQ_desc = TensorDescriptor.from_tensor(dQ, [B, H, BLOCK_Q, 128])
    dK_desc = TensorDescriptor.from_tensor(dK, [B, H, BLOCK_K, 128])
    dV_desc = TensorDescriptor.from_tensor(dV, [B, H, BLOCK_K, 128])
    L_desc = TensorDescriptor.from_tensor(L, [B, H, BLOCK_Q])
    
    grid_dQ = (triton.cdiv(S, BLOCK_Q), H, B)
    _bwd_dQ_kernel[grid_dQ](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L_desc, dQ_desc,
        S, scale, H, num_warps=8, num_stages=3
    )
    
    grid_dKdV = (triton.cdiv(S, BLOCK_K), H, B)
    _bwd_dKdV_kernel[grid_dKdV](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L_desc, dK_desc, dV_desc,
        S, scale, H, num_warps=8, num_stages=3
    )