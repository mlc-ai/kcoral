import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _preprocess_D_kernel(
    dO_ptr, O_ptr, D_ptr, S, d_val,
    STRIDE_B: tl.constexpr, STRIDE_S: tl.constexpr,
):
    idx = tl.program_id(0)
    b = idx // S
    s = idx % S
    
    offset = b * STRIDE_B + s * STRIDE_S
    cols = tl.arange(0, d_val)
    
    dO = tl.load(dO_ptr + offset + cols, mask=cols<d_val, other=0.0)
    O  = tl.load(O_ptr  + offset + cols, mask=cols<d_val, other=0.0)
    
    D = tl.sum(dO * O)
    
    if s < S:
        tl.store(D_ptr + idx, D)


@triton.jit
def _bwd_dKdV_kernel(
    Q_desc, K_desc, V_desc, dO_desc, O_desc, L_ptr, D_ptr,
    dK_desc, dV_desc,
    S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    j = tl.program_id(0)
    batch_h = tl.program_id(1)
    
    if j * BLOCK_N >= S:
        return
    
    num_blocks = (S + BLOCK_M - 1) // BLOCK_M
    
    b_h_idx0 = batch_h // 48
    b_h_idx1 = batch_h % 48
    
    K0 = tl.zeros((BLOCK_N, 64), tl.bfloat16)
    K1 = tl.zeros((BLOCK_N, 64), tl.bfloat16)
    V0 = tl.zeros((BLOCK_N, 64), tl.bfloat16)
    V1 = tl.zeros((BLOCK_N, 64), tl.bfloat16)
    
    for k in range(0, 128, 64):
        K_tile = K_desc.load([b_h_idx0, b_h_idx1, j * BLOCK_N, k])
        V_tile = V_desc.load([b_h_idx0, b_h_idx1, j * BLOCK_N, k])
        if k == 0:
            K0 = K_tile
            V0 = V_tile
        else:
            K1 = K_tile
            V1 = V_tile
            
    dK0_acc = tl.zeros((BLOCK_N, 64), tl.float32)
    dK1_acc = tl.zeros((BLOCK_N, 64), tl.float32)
    dV0_acc = tl.zeros((BLOCK_N, 64), tl.float32)
    dV1_acc = tl.zeros((BLOCK_N, 64), tl.float32)
    
    cols = tl.arange(0, BLOCK_N)
    valid_c = (j * BLOCK_N + cols) < S
    
    for i in range(num_blocks):
        Q0 = tl.zeros((BLOCK_M, 64), tl.bfloat16)
        Q1 = tl.zeros((BLOCK_M, 64), tl.bfloat16)
        dO0 = tl.zeros((BLOCK_M, 64), tl.bfloat16)
        dO1 = tl.zeros((BLOCK_M, 64), tl.bfloat16)
        O0 = tl.zeros((BLOCK_M, 64), tl.bfloat16)
        O1 = tl.zeros((BLOCK_M, 64), tl.bfloat16)
        
        for k in range(0, 128, 64):
            Q_tile = Q_desc.load([b_h_idx0, b_h_idx1, i * BLOCK_M, k])
            dO_tile = dO_desc.load([b_h_idx0, b_h_idx1, i * BLOCK_M, k])
            O_tile = O_desc.load([b_h_idx0, b_h_idx1, i * BLOCK_M, k])
            if k == 0:
                Q0 = Q_tile; dO0 = dO_tile; O0 = O_tile
            else:
                Q1 = Q_tile; dO1 = dO_tile; O1 = O_tile
                
        rows = tl.arange(0, BLOCK_M)
        row_idx = i * BLOCK_M + rows
        
        L_idx = batch_h * S + row_idx
        L_mask = row_idx < S
        L_val = tl.load(L_ptr + L_idx, mask=L_mask, other=0.0)
        
        D_idx = batch_h * S + row_idx
        D_mask = row_idx < S
        D_val = tl.load(D_ptr + D_idx, mask=D_mask, other=0.0)
        
        valid_r = row_idx < S
        
        S_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        S_acc = tl.dot(Q0, K0.T, S_acc)
        S_acc = tl.dot(Q1, K1.T, S_acc)
        
        P = tl.exp(S_acc * scale - L_val[:, None])
        P = P * valid_c[None, :]
        P = P * valid_r[:, None]
        
        dP_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        dP_acc = tl.dot(dO0, V0.T, dP_acc)
        dP_acc = tl.dot(dO1, V1.T, dP_acc)
        
        dS = P * (dP_acc - D_val[:, None]) * scale
        dS = dS * valid_c[None, :]
        dS = dS * valid_r[:, None]
        
        dV0_acc = tl.dot(P.T.to(tl.bfloat16), dO0, dV0_acc)
        dV1_acc = tl.dot(P.T.to(tl.bfloat16), dO1, dV1_acc)
        
        dK0_acc = tl.dot(dS.T.to(tl.bfloat16), Q0, dK0_acc)
        dK1_acc = tl.dot(dS.T.to(tl.bfloat16), Q1, dK1_acc)
    
    valid_r = (j * BLOCK_N + cols) < S
    dK0_acc *= valid_r[:, None]
    dK1_acc *= valid_r[:, None]
    dV0_acc *= valid_r[:, None]
    dV1_acc *= valid_r[:, None]
    
    dK0_out = dK0_acc.to(tl.bfloat16)
    dK_desc.store([b_h_idx0, b_h_idx1, j * BLOCK_N, 0], dK0_out)
    dK1_out = dK1_acc.to(tl.bfloat16)
    dK_desc.store([b_h_idx0, b_h_idx1, j * BLOCK_N, 64], dK1_out)
    
    dV0_out = dV0_acc.to(tl.bfloat16)
    dV_desc.store([b_h_idx0, b_h_idx1, j * BLOCK_N, 0], dV0_out)
    dV1_out = dV1_acc.to(tl.bfloat16)
    dV_desc.store([b_h_idx0, b_h_idx1, j * BLOCK_N, 64], dV1_out)


@triton.jit
def _bwd_dQ_kernel(
    Q_desc, K_desc, V_desc, dO_desc, O_desc, L_ptr, D_ptr,
    dQ_desc,
    S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    i = tl.program_id(0)
    batch_h = tl.program_id(1)
    
    if i * BLOCK_M >= S:
        return
    
    num_blocks = (S + BLOCK_N - 1) // BLOCK_N
    
    b_h_idx0 = batch_h // 48
    b_h_idx1 = batch_h % 48
    
    rows = tl.arange(0, BLOCK_M)
    row_idx = i * BLOCK_M + rows
    
    L_idx = batch_h * S + row_idx
    L_mask = row_idx < S
    L_val = tl.load(L_ptr + L_idx, mask=L_mask, other=0.0)
    
    D_idx = batch_h * S + row_idx
    D_mask = row_idx < S
    D_val = tl.load(D_ptr + D_idx, mask=D_mask, other=0.0)
    
    valid_r = row_idx < S
    
    Q0 = tl.zeros((BLOCK_M, 64), tl.bfloat16)
    Q1 = tl.zeros((BLOCK_M, 64), tl.bfloat16)
    dO0 = tl.zeros((BLOCK_M, 64), tl.bfloat16)
    dO1 = tl.zeros((BLOCK_M, 64), tl.bfloat16)
    O0 = tl.zeros((BLOCK_M, 64), tl.bfloat16)
    O1 = tl.zeros((BLOCK_M, 64), tl.bfloat16)
    
    for k in range(0, 128, 64):
        Q_tile = Q_desc.load([b_h_idx0, b_h_idx1, i * BLOCK_M, k])
        dO_tile = dO_desc.load([b_h_idx0, b_h_idx1, i * BLOCK_M, k])
        O_tile = O_desc.load([b_h_idx0, b_h_idx1, i * BLOCK_M, k])
        if k == 0:
            Q0 = Q_tile; dO0 = dO_tile; O0 = O_tile
        else:
            Q1 = Q_tile; dO1 = dO_tile; O1 = O_tile
            
    dQ0_acc = tl.zeros((BLOCK_M, 64), tl.float32)
    dQ1_acc = tl.zeros((BLOCK_M, 64), tl.float32)
    
    for j in range(num_blocks):
        K0 = tl.zeros((BLOCK_N, 64), tl.bfloat16)
        K1 = tl.zeros((BLOCK_N, 64), tl.bfloat16)
        V0 = tl.zeros((BLOCK_N, 64), tl.bfloat16)
        V1 = tl.zeros((BLOCK_N, 64), tl.bfloat16)
        
        for k in range(0, 128, 64):
            K_tile = K_desc.load([b_h_idx0, b_h_idx1, j * BLOCK_N, k])
            V_tile = V_desc.load([b_h_idx0, b_h_idx1, j * BLOCK_N, k])
            if k == 0:
                K0 = K_tile; V0 = V_tile
            else:
                K1 = K_tile; V1 = V_tile
                
        cols = tl.arange(0, BLOCK_N)
        valid_c = (j * BLOCK_N + cols) < S
        
        S_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        S_acc = tl.dot(Q0, K0.T, S_acc)
        S_acc = tl.dot(Q1, K1.T, S_acc)
        
        P = tl.exp(S_acc * scale - L_val[:, None])
        P = P * valid_c[None, :]
        P = P * valid_r[:, None]
        
        dP_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        dP_acc = tl.dot(dO0, V0.T, dP_acc)
        dP_acc = tl.dot(dO1, V1.T, dP_acc)
        
        dS = P * (dP_acc - D_val[:, None]) * scale
        dS = dS * valid_c[None, :]
        dS = dS * valid_r[:, None]
        
        dQ0_acc = tl.dot(dS.to(tl.bfloat16), K0, dQ0_acc)
        dQ1_acc = tl.dot(dS.to(tl.bfloat16), K1, dQ1_acc)
    
    dQ0_acc *= valid_r[:, None]
    dQ1_acc *= valid_r[:, None]
    
    dQ0_out = dQ0_acc.to(tl.bfloat16)
    dQ_desc.store([b_h_idx0, b_h_idx1, i * BLOCK_M, 0], dQ0_out)
    dQ1_out = dQ1_acc.to(tl.bfloat16)
    dQ_desc.store([b_h_idx0, b_h_idx1, i * BLOCK_M, 64], dQ1_out)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    
    B_val, H_val, S_val, d_val = Q.shape
    assert d_val == 128
    
    B_total = B_val * H_val
    
    L_3d = L.view(B_total, S_val)
    
    D = torch.empty((B_total, S_val), dtype=torch.float32, device=Q.device)
    grid_D = (B_total * S_val,)
    _preprocess_D_kernel[grid_D](
        dO.data_ptr(), O.data_ptr(), D.data_ptr(), S_val, d_val, 
        STRIDE_B=O.stride()[0], STRIDE_S=O.stride()[1]
    )
    
    BLOCK_M = 64
    BLOCK_N = 64
    
    scale = 1.0 / (d_val ** 0.5)
    
    Q_desc = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M, 64])
    K_desc = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N, 64])
    V_desc = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N, 64])
    dO_desc = TensorDescriptor.from_tensor(dO, [1, 1, BLOCK_M, 64])
    O_desc = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M, 64])
    dQ_desc = TensorDescriptor.from_tensor(dQ, [1, 1, BLOCK_M, 64])
    dK_desc = TensorDescriptor.from_tensor(dK, [1, 1, BLOCK_N, 64])
    dV_desc = TensorDescriptor.from_tensor(dV, [1, 1, BLOCK_N, 64])
    
    grid_dKdV = (triton.cdiv(S_val, BLOCK_N), B_total)
    _bwd_dKdV_kernel[grid_dKdV](
        Q_desc, K_desc, V_desc, dO_desc, O_desc, L_3d.data_ptr(), D.data_ptr(),
        dK_desc, dV_desc,
        S_val, scale, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=8, num_stages=2
    )
    
    grid_dQ = (triton.cdiv(S_val, BLOCK_M), B_total)
    _bwd_dQ_kernel[grid_dQ](
        Q_desc, K_desc, V_desc, dO_desc, O_desc, L_3d.data_ptr(), D.data_ptr(),
        dQ_desc,
        S_val, scale, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=8, num_stages=2
    )