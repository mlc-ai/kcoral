import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _preprocess_D_kernel(
    dO_ptr, O_ptr, D_ptr, S, d_val,
    STRIDE_B: tl.constexpr, STRIDE_S: tl.constexpr, STRIDE_D: tl.constexpr,
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
    Q0_desc, Q1_desc, K0_desc, K1_desc, V0_desc, V1_desc,
    dO0_desc, dO1_desc, O0_desc, O1_desc,
    L_ptr, dK0_desc, dK1_desc, dV0_desc, dV1_desc,
    S, scale,
    BLOCK_S: tl.constexpr,
):
    j = tl.program_id(0)
    batch_h = tl.program_id(1)
    
    if j * BLOCK_S >= S:
        return
    
    cols = tl.arange(0, BLOCK_S)
    valid_c = ((j * BLOCK_S) + cols) < S
    
    K0 = K0_desc.load([batch_h, j * BLOCK_S, 0])
    K1 = K1_desc.load([batch_h, j * BLOCK_S, 0])
    V0 = V0_desc.load([batch_h, j * BLOCK_S, 0])
    V1 = V1_desc.load([batch_h, j * BLOCK_S, 0])
    
    dK0_acc = tl.zeros((BLOCK_S, 64), tl.float32)
    dK1_acc = tl.zeros((BLOCK_S, 64), tl.float32)
    dV0_acc = tl.zeros((BLOCK_S, 64), tl.float32)
    dV1_acc = tl.zeros((BLOCK_S, 64), tl.float32)
    
    num_blocks = (S + BLOCK_S - 1) // BLOCK_S
    
    col_idx = j * BLOCK_S + cols
    
    for i in range(num_blocks):
        Q0 = Q0_desc.load([batch_h, i * BLOCK_S, 0])
        Q1 = Q1_desc.load([batch_h, i * BLOCK_S, 0])
        dO0 = dO0_desc.load([batch_h, i * BLOCK_S, 0])
        dO1 = dO1_desc.load([batch_h, i * BLOCK_S, 0])
        O0 = O0_desc.load([batch_h, i * BLOCK_S, 0])
        O1 = O1_desc.load([batch_h, i * BLOCK_S, 0])
        
        row_idx = i * BLOCK_S + cols
        L_idx = batch_h * S + row_idx
        L_mask = row_idx < S
        L_val = tl.load(L_ptr + L_idx, mask=L_mask, other=0.0)
        
        D_idx = batch_h * S + row_idx
        D_mask = row_idx < S
        D_val = tl.load(dK0_desc.base + D_idx, mask=D_mask, other=0.0)
        
        S_acc = tl.zeros((BLOCK_S, BLOCK_S), tl.float32)
        S_acc = tl.dot(Q0, K0.T, S_acc)
        S_acc = tl.dot(Q1, K1.T, S_acc)
        
        P = tl.exp(S_acc * scale - L_val[:, None])
        P = P * valid_c[None, :]
        
        dP_acc = tl.zeros((BLOCK_S, BLOCK_S), tl.float32)
        dP_acc = tl.dot(dO0, V0.T, dP_acc)
        dP_acc = tl.dot(dO1, V1.T, dP_acc)
        
        dS = P * (dP_acc - D_val[:, None]) * scale
        
        dV0_acc = tl.dot(P.T, dO0, dV0_acc)
        dV1_acc = tl.dot(P.T, dO1, dV1_acc)
        
        dK0_acc = tl.dot(dS.T, Q0, dK0_acc)
        dK1_acc = tl.dot(dS.T, Q1, dK1_acc)
    
    dK0_acc = tl.where(col_idx[None, :], dK0_acc, 0.0)
    dK1_acc = tl.where(col_idx[None, :], dK1_acc, 0.0)
    dV0_acc = tl.where(col_idx[None, :], dV0_acc, 0.0)
    dV1_acc = tl.where(col_idx[None, :], dV1_acc, 0.0)
    
    dK0_out = dK0_acc.to(tl.bfloat16).unsqueeze(0)
    dK0_desc.store([batch_h, j * BLOCK_S, 0], dK0_out)
    dK1_out = dK1_acc.to(tl.bfloat16).unsqueeze(0)
    dK1_desc.store([batch_h, j * BLOCK_S, 0], dK1_out)
    
    dV0_out = dV0_acc.to(tl.bfloat16).unsqueeze(0)
    dV0_desc.store([batch_h, j * BLOCK_S, 0], dV0_out)
    dV1_out = dV1_acc.to(tl.bfloat16).unsqueeze(0)
    dV1_desc.store([batch_h, j * BLOCK_S, 0], dV1_out)


@triton.jit
def _bwd_dQ_kernel(
    Q0_desc, Q1_desc, K0_desc, K1_desc, V0_desc, V1_desc,
    dO0_desc, dO1_desc, O0_desc, O1_desc,
    L_ptr, dQ0_desc, dQ1_desc,
    S, scale,
    BLOCK_S: tl.constexpr,
):
    i = tl.program_id(0)
    batch_h = tl.program_id(1)
    
    if i * BLOCK_S >= S:
        return
    
    rows = tl.arange(0, BLOCK_S)
    cols = tl.arange(0, BLOCK_S)
    
    Q0 = Q0_desc.load([batch_h, i * BLOCK_S, 0])
    Q1 = Q1_desc.load([batch_h, i * BLOCK_S, 0])
    dO0 = dO0_desc.load([batch_h, i * BLOCK_S, 0])
    dO1 = dO1_desc.load([batch_h, i * BLOCK_S, 0])
    O0 = O0_desc.load([batch_h, i * BLOCK_S, 0])
    O1 = O1_desc.load([batch_h, i * BLOCK_S, 0])
    
    row_idx = i * BLOCK_S + rows
    L_idx = batch_h * S + row_idx
    L_mask = row_idx < S
    L_val = tl.load(L_ptr + L_idx, mask=L_mask, other=0.0)
    
    D_idx = batch_h * S + row_idx
    D_mask = row_idx < S
    D_val = tl.load(dQ0_desc.base + D_idx, mask=D_mask, other=0.0)
    
    dQ0_acc = tl.zeros((BLOCK_S, 64), tl.float32)
    dQ1_acc = tl.zeros((BLOCK_S, 64), tl.float32)
    
    num_blocks = (S + BLOCK_S - 1) // BLOCK_S
    
    row_idx_out = i * BLOCK_S + rows
    
    for j in range(num_blocks):
        K0 = K0_desc.load([batch_h, j * BLOCK_S, 0])
        K1 = K1_desc.load([batch_h, j * BLOCK_S, 0])
        V0 = V0_desc.load([batch_h, j * BLOCK_S, 0])
        V1 = V1_desc.load([batch_h, j * BLOCK_S, 0])
        
        S_acc = tl.zeros((BLOCK_S, BLOCK_S), tl.float32)
        S_acc = tl.dot(Q0, K0.T, S_acc)
        S_acc = tl.dot(Q1, K1.T, S_acc)
        
        P = tl.exp(S_acc * scale - L_val[:, None])
        
        dP_acc = tl.zeros((BLOCK_S, BLOCK_S), tl.float32)
        dP_acc = tl.dot(dO0, V0.T, dP_acc)
        dP_acc = tl.dot(dO1, V1.T, dP_acc)
        
        dS = P * (dP_acc - D_val[:, None]) * scale
        
        dQ0_acc = tl.dot(dS, K0, dQ0_acc)
        dQ1_acc = tl.dot(dS, K1, dQ1_acc)
    
    dQ0_acc = tl.where(row_idx_out[:, None], dQ0_acc, 0.0)
    dQ1_acc = tl.where(row_idx_out[:, None], dQ1_acc, 0.0)
    
    dQ0_out = dQ0_acc.to(tl.bfloat16).unsqueeze(0)
    dQ0_desc.store([batch_h, i * BLOCK_S, 0], dQ0_out)
    dQ1_out = dQ1_acc.to(tl.bfloat16).unsqueeze(0)
    dQ1_desc.store([batch_h, i * BLOCK_S, 0], dQ1_out)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    
    B_val, H_val, S_val, d_val = Q.shape
    assert d_val == 128
    
    B_total = B_val * H_val
    
    Q_3d = Q.view(B_total, S_val, d_val)
    K_3d = K.view(B_total, S_val, d_val)
    V_3d = V.view(B_total, S_val, d_val)
    O_3d = O.view(B_total, S_val, d_val)
    dO_3d = dO.view(B_total, S_val, d_val)
    dQ_3d = dQ.view(B_total, S_val, d_val)
    dK_3d = dK.view(B_total, S_val, d_val)
    dV_3d = dV.view(B_total, S_val, d_val)
    
    BLOCK_S = 64
    
    Q0_t = torch.narrow(Q_3d, 2, 0, 64)
    Q1_t = torch.narrow(Q_3d, 2, 64, 64)
    K0_t = torch.narrow(K_3d, 2, 0, 64)
    K1_t = torch.narrow(K_3d, 2, 64, 64)
    V0_t = torch.narrow(V_3d, 2, 0, 64)
    V1_t = torch.narrow(V_3d, 2, 64, 64)
    O0_t = torch.narrow(O_3d, 2, 0, 64)
    O1_t = torch.narrow(O_3d, 2, 64, 64)
    dO0_t = torch.narrow(dO_3d, 2, 0, 64)
    dO1_t = torch.narrow(dO_3d, 2, 64, 64)
    dQ0_t = torch.narrow(dQ_3d, 2, 0, 64)
    dQ1_t = torch.narrow(dQ_3d, 2, 64, 64)
    dK0_t = torch.narrow(dK_3d, 2, 0, 64)
    dK1_t = torch.narrow(dK_3d, 2, 64, 64)
    dV0_t = torch.narrow(dV_3d, 2, 0, 64)
    dV1_t = torch.narrow(dV_3d, 2, 64, 64)
    
    Q0_desc = TensorDescriptor.from_tensor(Q0_t, [1, BLOCK_S, 64])
    Q1_desc = TensorDescriptor.from_tensor(Q1_t, [1, BLOCK_S, 64])
    K0_desc = TensorDescriptor.from_tensor(K0_t, [1, BLOCK_S, 64])
    K1_desc = TensorDescriptor.from_tensor(K1_t, [1, BLOCK_S, 64])
    V0_desc = TensorDescriptor.from_tensor(V0_t, [1, BLOCK_S, 64])
    V1_desc = TensorDescriptor.from_tensor(V1_t, [1, BLOCK_S, 64])
    O0_desc = TensorDescriptor.from_tensor(O0_t, [1, BLOCK_S, 64])
    O1_desc = TensorDescriptor.from_tensor(O1_t, [1, BLOCK_S, 64])
    dO0_desc = TensorDescriptor.from_tensor(dO0_t, [1, BLOCK_S, 64])
    dO1_desc = TensorDescriptor.from_tensor(dO1_t, [1, BLOCK_S, 64])
    dQ0_desc = TensorDescriptor.from_tensor(dQ0_t, [1, BLOCK_S, 64])
    dQ1_desc = TensorDescriptor.from_tensor(dQ1_t, [1, BLOCK_S, 64])
    dK0_desc = TensorDescriptor.from_tensor(dK0_t, [1, BLOCK_S, 64])
    dK1_desc = TensorDescriptor.from_tensor(dK1_t, [1, BLOCK_S, 64])
    dV0_desc = TensorDescriptor.from_tensor(dV0_t, [1, BLOCK_S, 64])
    dV1_desc = TensorDescriptor.from_tensor(dV1_t, [1, BLOCK_S, 64])
    
    scale = 1.0 / (d_val ** 0.5)
    
    D = torch.empty((B_total, S_val), dtype=torch.float32, device=Q.device)
    grid_D = (B_total * S_val,)
    _preprocess_D_kernel[grid_D](
        dO, O, D, S_val, d_val, 
        STRIDE_B=O.stride()[0], STRIDE_S=O.stride()[1], STRIDE_D=O.stride()[2]
    )
    
    grid_bwd = (triton.cdiv(S_val, BLOCK_S), B_total)
    
    _bwd_dKdV_kernel[grid_bwd](
        Q0_desc, Q1_desc, K0_desc, K1_desc, V0_desc, V1_desc,
        dO0_desc, dO1_desc, O0_desc, O1_desc,
        L, dK0_desc, dK1_desc, dV0_desc, dV1_desc,
        S_val, scale, BLOCK_S=BLOCK_S,
        num_warps=8, num_stages=2
    )
    
    _bwd_dQ_kernel[grid_bwd](
        Q0_desc, Q1_desc, K0_desc, K1_desc, V0_desc, V1_desc,
        dO0_desc, dO1_desc, O0_desc, O1_desc,
        L, dQ0_desc, dQ1_desc,
        S_val, scale, BLOCK_S=BLOCK_S,
        num_warps=8, num_stages=2
    )