import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _bwd_dKdV_kernel(
    Q0_desc, Q1_desc, K0_desc, K1_desc, V0_desc, V1_desc,
    dO0_desc, dO1_desc, O0_desc, O1_desc, L_ptr,
    dK0_desc, dK1_desc, dV0_desc, dV1_desc,
    H, S, scale,
    BLOCK_S: tl.constexpr,
):
    j = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    
    if j * BLOCK_S >= S:
        return
    
    K0 = K0_desc.load([b, h, j * BLOCK_S, 0]).squeeze(0)
    K1 = K1_desc.load([b, h, j * BLOCK_S, 0]).squeeze(0)
    V0 = V0_desc.load([b, h, j * BLOCK_S, 0]).squeeze(0)
    V1 = V1_desc.load([b, h, j * BLOCK_S, 0]).squeeze(0)
    
    dK0_acc = tl.zeros((BLOCK_S, 64), tl.float32)
    dK1_acc = tl.zeros((BLOCK_S, 64), tl.float32)
    dV0_acc = tl.zeros((BLOCK_S, 64), tl.float32)
    dV1_acc = tl.zeros((BLOCK_S, 64), tl.float32)
    
    num_blocks = (S + BLOCK_S - 1) // BLOCK_S
    
    cols = tl.arange(0, BLOCK_S)
    valid_c = (j * BLOCK_S + cols) < S
    
    for i in range(num_blocks):
        Q0 = Q0_desc.load([b, h, i * BLOCK_S, 0]).squeeze(0)
        Q1 = Q1_desc.load([b, h, i * BLOCK_S, 0]).squeeze(0)
        dO0 = dO0_desc.load([b, h, i * BLOCK_S, 0]).squeeze(0)
        dO1 = dO1_desc.load([b, h, i * BLOCK_S, 0]).squeeze(0)
        O0 = O0_desc.load([b, h, i * BLOCK_S, 0]).squeeze(0)
        O1 = O1_desc.load([b, h, i * BLOCK_S, 0]).squeeze(0)
        
        rows = tl.arange(0, BLOCK_S)
        row_idx = i * BLOCK_S + rows
        
        L_idx = b * H * S + h * S + row_idx
        L_mask = row_idx < S
        L_val = tl.load(L_ptr + L_idx, mask=L_mask, other=0.0)
        
        D_val = tl.sum(dO0.to(tl.float32) * O0.to(tl.float32) + dO1.to(tl.float32) * O1.to(tl.float32), axis=1)
        
        S_acc = tl.zeros((BLOCK_S, BLOCK_S), tl.float32)
        S_acc = tl.dot(Q0, K0.T, S_acc)
        S_acc = tl.dot(Q1, K1.T, S_acc)
        
        P = tl.exp(S_acc * scale - L_val[:, None])
        P = P * valid_c[None, :]
        
        dP_acc = tl.zeros((BLOCK_S, BLOCK_S), tl.float32)
        dP_acc = tl.dot(dO0, V0.T, dP_acc)
        dP_acc = tl.dot(dO1, V1.T, dP_acc)
        
        dS = P * (dP_acc - D_val[:, None]) * scale
        dS = dS * valid_c[None, :]
        
        dV0_acc = tl.dot(P.T.to(tl.bfloat16), dO0, dV0_acc)
        dV1_acc = tl.dot(P.T.to(tl.bfloat16), dO1, dV1_acc)
        
        dK0_acc = tl.dot(dS.T.to(tl.bfloat16), Q0, dK0_acc)
        dK1_acc = tl.dot(dS.T.to(tl.bfloat16), Q1, dK1_acc)
    
    dK0_out = dK0_acc.to(tl.bfloat16)
    dK0_desc.store([b, h, j * BLOCK_S, 0], dK0_out)
    dK1_out = dK1_acc.to(tl.bfloat16)
    dK1_desc.store([b, h, j * BLOCK_S, 0], dK1_out)
    
    dV0_out = dV0_acc.to(tl.bfloat16)
    dV0_desc.store([b, h, j * BLOCK_S, 0], dV0_out)
    dV1_out = dV1_acc.to(tl.bfloat16)
    dV1_desc.store([b, h, j * BLOCK_S, 0], dV1_out)


@triton.jit
def _bwd_dQ_kernel(
    Q0_desc, Q1_desc, K0_desc, K1_desc, V0_desc, V1_desc,
    dO0_desc, dO1_desc, O0_desc, O1_desc, L_ptr,
    dQ0_desc, dQ1_desc,
    H, S, scale,
    BLOCK_S: tl.constexpr,
):
    i = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    
    if i * BLOCK_S >= S:
        return
    
    Q0 = Q0_desc.load([b, h, i * BLOCK_S, 0]).squeeze(0)
    Q1 = Q1_desc.load([b, h, i * BLOCK_S, 0]).squeeze(0)
    dO0 = dO0_desc.load([b, h, i * BLOCK_S, 0]).squeeze(0)
    dO1 = dO1_desc.load([b, h, i * BLOCK_S, 0]).squeeze(0)
    O0 = O0_desc.load([b, h, i * BLOCK_S, 0]).squeeze(0)
    O1 = O1_desc.load([b, h, i * BLOCK_S, 0]).squeeze(0)
    
    rows = tl.arange(0, BLOCK_S)
    row_idx = i * BLOCK_S + rows
    
    L_idx = b * H * S + h * S + row_idx
    L_mask = row_idx < S
    L_val = tl.load(L_ptr + L_idx, mask=L_mask, other=0.0)
    
    D_val = tl.sum(dO0.to(tl.float32) * O0.to(tl.float32) + dO1.to(tl.float32) * O1.to(tl.float32), axis=1)
    
    dQ0_acc = tl.zeros((BLOCK_S, 64), tl.float32)
    dQ1_acc = tl.zeros((BLOCK_S, 64), tl.float32)
    
    num_blocks = (S + BLOCK_S - 1) // BLOCK_S
    
    for j in range(num_blocks):
        K0 = K0_desc.load([b, h, j * BLOCK_S, 0]).squeeze(0)
        K1 = K1_desc.load([b, h, j * BLOCK_S, 0]).squeeze(0)
        V0 = V0_desc.load([b, h, j * BLOCK_S, 0]).squeeze(0)
        V1 = V1_desc.load([b, h, j * BLOCK_S, 0]).squeeze(0)
        
        cols = tl.arange(0, BLOCK_S)
        valid_c = (j * BLOCK_S + cols) < S
        
        S_acc = tl.zeros((BLOCK_S, BLOCK_S), tl.float32)
        S_acc = tl.dot(Q0, K0.T, S_acc)
        S_acc = tl.dot(Q1, K1.T, S_acc)
        
        P = tl.exp(S_acc * scale - L_val[:, None])
        P = P * valid_c[None, :]
        
        dP_acc = tl.zeros((BLOCK_S, BLOCK_S), tl.float32)
        dP_acc = tl.dot(dO0, V0.T, dP_acc)
        dP_acc = tl.dot(dO1, V1.T, dP_acc)
        
        dS = P * (dP_acc - D_val[:, None]) * scale
        dS = dS * valid_c[None, :]
        
        dQ0_acc = tl.dot(dS.to(tl.bfloat16), K0, dQ0_acc)
        dQ1_acc = tl.dot(dS.to(tl.bfloat16), K1, dQ1_acc)
    
    dQ0_out = dQ0_acc.to(tl.bfloat16)
    dQ0_desc.store([b, h, i * BLOCK_S, 0], dQ0_out)
    dQ1_out = dQ1_acc.to(tl.bfloat16)
    dQ1_desc.store([b, h, i * BLOCK_S, 0], dQ1_out)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    
    B_val, H_val, S_val, d_val = Q.shape
    assert d_val == 128
    
    BLOCK_S = 64
    
    scale = 1.0 / (d_val ** 0.5)
    
    Q0_desc = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_S, 64])
    Q1_desc = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_S, 64])
    K0_desc = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_S, 64])
    K1_desc = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_S, 64])
    V0_desc = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_S, 64])
    V1_desc = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_S, 64])
    dO0_desc = TensorDescriptor.from_tensor(dO, [1, 1, BLOCK_S, 64])
    dO1_desc = TensorDescriptor.from_tensor(dO, [1, 1, BLOCK_S, 64])
    O0_desc = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_S, 64])
    O1_desc = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_S, 64])
    dQ0_desc = TensorDescriptor.from_tensor(dQ, [1, 1, BLOCK_S, 64])
    dQ1_desc = TensorDescriptor.from_tensor(dQ, [1, 1, BLOCK_S, 64])
    dK0_desc = TensorDescriptor.from_tensor(dK, [1, 1, BLOCK_S, 64])
    dK1_desc = TensorDescriptor.from_tensor(dK, [1, 1, BLOCK_S, 64])
    dV0_desc = TensorDescriptor.from_tensor(dV, [1, 1, BLOCK_S, 64])
    dV1_desc = TensorDescriptor.from_tensor(dV, [1, 1, BLOCK_S, 64])
    
    grid_bwd = (triton.cdiv(S_val, BLOCK_S), B_val, H_val)
    
    _bwd_dKdV_kernel[grid_bwd](
        Q0_desc, Q1_desc, K0_desc, K1_desc, V0_desc, V1_desc,
        dO0_desc, dO1_desc, O0_desc, O1_desc, L.data_ptr(),
        dK0_desc, dK1_desc, dV0_desc, dV1_desc,
        H_val, S_val, scale, BLOCK_S=BLOCK_S,
        num_warps=8, num_stages=2
    )
    
    _bwd_dQ_kernel[grid_bwd](
        Q0_desc, Q1_desc, K0_desc, K1_desc, V0_desc, V1_desc,
        dO0_desc, dO1_desc, O0_desc, O1_desc, L.data_ptr(),
        dQ0_desc, dQ1_desc,
        H_val, S_val, scale, BLOCK_S=BLOCK_S,
        num_warps=8, num_stages=2
    )