import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _bwd_dKdV_kernel(
    Q_desc, K_desc, V_desc, dO_desc, O_desc,
    L_ptr,
    dK_desc, dV_desc,
    S, scale,
    BLOCK_S: tl.constexpr,
):
    j = tl.program_id(0)
    batch_h = tl.program_id(1)
    
    if j * BLOCK_S >= S:
        return
    
    rows = tl.arange(0, BLOCK_S)
    cols = tl.arange(0, BLOCK_S)
    
    K0 = K_desc.load([batch_h, j * BLOCK_S, 0]).squeeze(0)
    K1 = K_desc.load([batch_h, j * BLOCK_S, 64]).squeeze(0)
    V0 = V_desc.load([batch_h, j * BLOCK_S, 0]).squeeze(0)
    V1 = V_desc.load([batch_h, j * BLOCK_S, 64]).squeeze(0)
    
    dK0_acc = tl.zeros((BLOCK_S, 64), tl.float32)
    dK1_acc = tl.zeros((BLOCK_S, 64), tl.float32)
    dV0_acc = tl.zeros((BLOCK_S, 64), tl.float32)
    dV1_acc = tl.zeros((BLOCK_S, 64), tl.float32)
    
    num_blocks = (S + BLOCK_S - 1) // BLOCK_S
    valid_c = ((j * BLOCK_S) + cols) < S
    
    for i in range(num_blocks):
        Q0 = Q_desc.load([batch_h, i * BLOCK_S, 0]).squeeze(0)
        Q1 = Q_desc.load([batch_h, i * BLOCK_S, 64]).squeeze(0)
        dO0 = dO_desc.load([batch_h, i * BLOCK_S, 0]).squeeze(0)
        dO1 = dO_desc.load([batch_h, i * BLOCK_S, 64]).squeeze(0)
        O0 = O_desc.load([batch_h, i * BLOCK_S, 0]).squeeze(0)
        O1 = O_desc.load([batch_h, i * BLOCK_S, 64]).squeeze(0)
        
        L_idx = batch_h * S + i * BLOCK_S + rows
        L_mask = (i * BLOCK_S + rows) < S
        L_val = tl.load(L_ptr + L_idx, mask=L_mask, other=0.0)
        
        dO0_f32 = dO0.to(tl.float32)
        dO1_f32 = dO1.to(tl.float32)
        O0_f32 = O0.to(tl.float32)
        O1_f32 = O1.to(tl.float32)
        D_val = tl.sum(dO0_f32 * O0_f32 + dO1_f32 * O1_f32, axis=1)
        
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
    
    dK0_out = dK0_acc.to(tl.bfloat16).unsqueeze(0)
    dK_desc.store([batch_h, j * BLOCK_S, 0], dK0_out)
    dK1_out = dK1_acc.to(tl.bfloat16).unsqueeze(0)
    dK_desc.store([batch_h, j * BLOCK_S, 64], dK1_out)
    
    dV0_out = dV0_acc.to(tl.bfloat16).unsqueeze(0)
    dV_desc.store([batch_h, j * BLOCK_S, 0], dV0_out)
    dV1_out = dV1_acc.to(tl.bfloat16).unsqueeze(0)
    dV_desc.store([batch_h, j * BLOCK_S, 64], dV1_out)


@triton.jit
def _bwd_dQ_kernel(
    Q_desc, K_desc, V_desc, dO_desc, O_desc,
    L_ptr,
    dQ_desc,
    S, scale,
    BLOCK_S: tl.constexpr,
):
    i = tl.program_id(0)
    batch_h = tl.program_id(1)
    
    if i * BLOCK_S >= S:
        return
    
    rows = tl.arange(0, BLOCK_S)
    cols = tl.arange(0, BLOCK_S)
    
    Q0 = Q_desc.load([batch_h, i * BLOCK_S, 0]).squeeze(0)
    Q1 = Q_desc.load([batch_h, i * BLOCK_S, 64]).squeeze(0)
    dO0 = dO_desc.load([batch_h, i * BLOCK_S, 0]).squeeze(0)
    dO1 = dO_desc.load([batch_h, i * BLOCK_S, 64]).squeeze(0)
    O0 = O_desc.load([batch_h, i * BLOCK_S, 0]).squeeze(0)
    O1 = O_desc.load([batch_h, i * BLOCK_S, 64]).squeeze(0)
    
    L_idx = batch_h * S + i * BLOCK_S + rows
    L_mask = (i * BLOCK_S + rows) < S
    L_val = tl.load(L_ptr + L_idx, mask=L_mask, other=0.0)
    
    dO0_f32 = dO0.to(tl.float32)
    dO1_f32 = dO1.to(tl.float32)
    O0_f32 = O0.to(tl.float32)
    O1_f32 = O1.to(tl.float32)
    D_val = tl.sum(dO0_f32 * O0_f32 + dO1_f32 * O1_f32, axis=1)
    
    dQ0_acc = tl.zeros((BLOCK_S, 64), tl.float32)
    dQ1_acc = tl.zeros((BLOCK_S, 64), tl.float32)
    
    num_blocks = (S + BLOCK_S - 1) // BLOCK_S
    
    for j in range(num_blocks):
        K0 = K_desc.load([batch_h, j * BLOCK_S, 0]).squeeze(0)
        K1 = K_desc.load([batch_h, j * BLOCK_S, 64]).squeeze(0)
        V0 = V_desc.load([batch_h, j * BLOCK_S, 0]).squeeze(0)
        V1 = V_desc.load([batch_h, j * BLOCK_S, 64]).squeeze(0)
        
        valid_c = ((j * BLOCK_S) + cols) < S
        
        S_acc = tl.zeros((BLOCK_S, BLOCK_S), tl.float32)
        S_acc = tl.dot(Q0, K0.T, S_acc)
        S_acc = tl.dot(Q1, K1.T, S_acc)
        
        P = tl.exp(S_acc * scale - L_val[:, None])
        P = P * valid_c[None, :]
        
        dP_acc = tl.zeros((BLOCK_S, BLOCK_S), tl.float32)
        dP_acc = tl.dot(dO0, V0.T, dP_acc)
        dP_acc = tl.dot(dO1, V1.T, dP_acc)
        
        dS = P * (dP_acc - D_val[:, None]) * scale
        
        dQ0_acc = tl.dot(dS, K0, dQ0_acc)
        dQ1_acc = tl.dot(dS, K1, dQ1_acc)
    
    dQ0_out = dQ0_acc.to(tl.bfloat16).unsqueeze(0)
    dQ_desc.store([batch_h, i * BLOCK_S, 0], dQ0_out)
    dQ1_out = dQ1_acc.to(tl.bfloat16).unsqueeze(0)
    dQ_desc.store([batch_h, i * BLOCK_S, 64], dQ1_out)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    
    B_val, H_val, S_val, d_val = Q.shape
    assert d_val == 128
    
    dQ.zero_()
    dK.zero_()
    dV.zero_()
    
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
    Q_desc = TensorDescriptor.from_tensor(Q_3d, [1, BLOCK_S, 64])
    K_desc = TensorDescriptor.from_tensor(K_3d, [1, BLOCK_S, 64])
    V_desc = TensorDescriptor.from_tensor(V_3d, [1, BLOCK_S, 64])
    O_desc = TensorDescriptor.from_tensor(O_3d, [1, BLOCK_S, 64])
    dO_desc = TensorDescriptor.from_tensor(dO_3d, [1, BLOCK_S, 64])
    dQ_desc = TensorDescriptor.from_tensor(dQ_3d, [1, BLOCK_S, 64])
    dK_desc = TensorDescriptor.from_tensor(dK_3d, [1, BLOCK_S, 64])
    dV_desc = TensorDescriptor.from_tensor(dV_3d, [1, BLOCK_S, 64])
    
    scale = 1.0 / (d_val ** 0.5)
    grid_bwd = (triton.cdiv(S_val, BLOCK_S), B_total)
    
    _bwd_dKdV_kernel[grid_bwd](
        Q_desc, K_desc, V_desc, dO_desc, O_desc, L, dK_desc, dV_desc,
        S_val, scale, BLOCK_S=BLOCK_S,
        num_warps=8, num_stages=2
    )
    
    _bwd_dQ_kernel[grid_bwd](
        Q_desc, K_desc, V_desc, dO_desc, O_desc, L, dQ_desc,
        S_val, scale, BLOCK_S=BLOCK_S,
        num_warps=8, num_stages=2
    )