import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

@triton.jit
def _precompute_D_kernel(dO_ptr, O_ptr, D_ptr, n_rows):
    row_idx = tl.program_id(0)
    if row_idx < n_rows:
        val = 0.0
        for k in range(128):
            val += tl.load(dO_ptr + row_idx * 128 + k) * tl.load(O_ptr + row_idx * 128 + k)
        tl.store(D_ptr + row_idx, val)


@triton.jit
def _bwd_dQ_kernel(
    Q_desc,
    K_desc,
    V_desc,
    O_desc,
    dO_desc,
    L_desc,
    D_desc,
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
    
    Q_desc_bh = Q_desc[bh]
    K_desc_bh = K_desc[bh]
    V_desc_bh = V_desc[bh]
    O_desc_bh = O_desc[bh]
    dO_desc_bh = dO_desc[bh]
    L_desc_bh = L_desc[bh]
    D_desc_bh = D_desc[bh]
    dQ_desc_bh = dQ_desc[bh]
    
    # Load persistent Query info
    Q = Q_desc_bh.load([start_seq, 0]).to(tl.float32)
    dO = dO_desc_bh.load([start_seq, 0]).to(tl.float32)
    O = O_desc_bh.load([start_seq, 0]).to(tl.float32)
    
    # Preprocess D = rowsum(dO * O)
    D_i = tl.sum(dO * O, axis=1)
    
    # Load LogSumExp
    L_i = L_desc_bh.load([start_seq])
    
    q_mask = (start_seq + tl.arange(0, 128)) < S_len
    
    acc_dQ = tl.zeros((128, 128), tl.float32)
    num_k = S_len // 128
    
    # Iterate over Key and Value chunks
    for j in tl.range(num_k, num_stages=3):
        K = K_desc_bh.load([j * 128, 0]).to(tl.float32)
        V = V_desc_bh.load([j * 128, 0]).to(tl.float32)
        
        # S = Q @ K^T
        S = tl.dot(Q, K.T)
        # dP = dO @ V^T
        dP = tl.dot(dO, V.T)
        
        k_mask = (j * 128 + tl.arange(0, 128)) < S_len
        
        # P = exp(S * scale - L)
        P = tl.exp(S * scale - L_i[:, None])
        P = P * k_mask[None, :]
        P = P * q_mask[:, None]
        
        # dS = P * (dP - D) * scale
        dS = P * (dP - D_i[:, None]) * scale
        
        # dQ += dS @ K
        acc_dQ = tl.dot(dS, K, acc_dQ)
    
    dQ_desc_bh.store([start_seq, 0], acc_dQ.to(tl.bfloat16))


@triton.jit
def _bwd_dKdV_kernel(
    Q_desc,
    K_desc,
    V_desc,
    O_desc,
    dO_desc,
    L_desc,
    D_desc,
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
    start_seq = pid_x * 128
    
    Q_desc_bh = Q_desc[bh]
    K_desc_bh = K_desc[bh]
    V_desc_bh = V_desc[bh]
    O_desc_bh = O_desc[bh]
    dO_desc_bh = dO_desc[bh]
    L_desc_bh = L_desc[bh]
    D_desc_bh = D_desc[bh]
    dK_desc_bh = dK_desc[bh]
    dV_desc_bh = dV_desc[bh]
    
    # Load persistent Key and Value info
    K = K_desc_bh.load([start_seq, 0]).to(tl.float32)
    V = V_desc_bh.load([start_seq, 0]).to(tl.float32)
    
    acc_dK = tl.zeros((128, 128), tl.float32)
    acc_dV = tl.zeros((128, 128), tl.float32)
    
    k_mask = (start_seq + tl.arange(0, 128)) < S_len
    
    num_q = S_len // 128
    
    # Iterate over Query chunks
    for i in tl.range(num_q, num_stages=3):
        Q = Q_desc_bh.load([i * 128, 0]).to(tl.float32)
        dO = dO_desc_bh.load([i * 128, 0]).to(tl.float32)
        O = O_desc_bh.load([i * 128, 0]).to(tl.float32)
        
        D_i = tl.sum(dO * O, axis=1)
        L_i = L_desc_bh.load([i * 128])
        
        # S = Q @ K^T
        S = tl.dot(Q, K.T)
        # dP = dO @ V^T
        dP = tl.dot(dO, V.T)
        
        q_mask = (i * 128 + tl.arange(0, 128)) < S_len
        
        # P = exp(S * scale - L)
        P = tl.exp(S * scale - L_i[:, None])
        P = P * k_mask[None, :]
        P = P * q_mask[:, None]
        
        # dS = P * (dP - D) * scale
        dS = P * (dP - D_i[:, None]) * scale
        
        # dK += dS^T @ Q
        acc_dK = tl.dot(dS.T, Q, acc_dK)
        
        # dV += P^T @ dO
        acc_dV = tl.dot(P.T, dO, acc_dV)
    
    dK_desc_bh.store([start_seq, 0], acc_dK.to(tl.bfloat16))
    dV_desc_bh.store([start_seq, 0], acc_dV.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    scale = 1.0 / (d ** 0.5)
    
    # Precompute D entirely on CUDA using a simple kernel
    D = torch.empty((B, H, S), dtype=torch.float32, device=Q.device)
    n_rows = B * H * S
    grid_D = (n_rows,)
    _precompute_D_kernel[grid_D](dO.contiguous().view(n_rows, 128).data_ptr(), 
                                 O.contiguous().view(n_rows, 128).data_ptr(), 
                                 D.contiguous().view(n_rows).data_ptr(), 
                                 n_rows)
    
    dQ_3d = dQ.view(B * H, S, d)
    dK_3d = dK.view(B * H, S, d)
    dV_3d = dV.view(B * H, S, d)
    D_3d = D.view(B * H, S)
    
    Q_desc_list = []
    K_desc_list = []
    V_desc_list = []
    O_desc_list = []
    dO_desc_list = []
    dQ_desc_list = []
    dK_desc_list = []
    dV_desc_list = []
    L_desc_list = []
    D_desc_list = []
    
    for b in range(B):
        for h in range(H):
            Q_bh = Q[b, h, :, :]
            K_bh = K[b, h, :, :]
            V_bh = V[b, h, :, :]
            O_bh = O[b, h, :, :]
            dO_bh = dO[b, h, :, :]
            dQ_bh = dQ_3d[b*H + h, :, :]
            dK_bh = dK_3d[b*H + h, :, :]
            dV_bh = dV_3d[b*H + h, :, :]
            
            Q_desc_list.append(TensorDescriptor.from_tensor(Q_bh, [128, 128]))
            K_desc_list.append(TensorDescriptor.from_tensor(K_bh, [128, 128]))
            V_desc_list.append(TensorDescriptor.from_tensor(V_bh, [128, 128]))
            O_desc_list.append(TensorDescriptor.from_tensor(O_bh, [128, 128]))
            dO_desc_list.append(TensorDescriptor.from_tensor(dO_bh, [128, 128]))
            dQ_desc_list.append(TensorDescriptor.from_tensor(dQ_bh, [128, 128]))
            dK_desc_list.append(TensorDescriptor.from_tensor(dK_bh, [128, 128]))
            dV_desc_list.append(TensorDescriptor.from_tensor(dV_bh, [128, 128]))
            
            L_bh = L[b, h, :]
            L_desc_list.append(TensorDescriptor.from_tensor(L_bh, [128]))
            
            D_bh = D_3d[b*H + h, :]
            D_desc_list.append(TensorDescriptor.from_tensor(D_bh, [128]))
    
    grid_dQ = (S // 128, H, B)
    _bwd_dQ_kernel[grid_dQ](
        Q_desc_list, K_desc_list, V_desc_list, O_desc_list, dO_desc_list, L_desc_list, D_desc_list, dQ_desc_list,
        S, scale, H, num_warps=8, num_stages=3
    )
    
    grid_dKdV = (S // 128, H, B)
    _bwd_dKdV_kernel[grid_dKdV](
        Q_desc_list, K_desc_list, V_desc_list, O_desc_list, dO_desc_list, L_desc_list, D_desc_list, dK_desc_list, dV_desc_list,
        S, scale, H, num_warps=8, num_stages=3
    )