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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    j = tl.program_id(0)
    batch_h = tl.program_id(1)
    
    if j * BLOCK_N >= S:
        return
    
    b = batch_h // H
    h = batch_h % H
    
    K0 = tl.reshape(K0_desc.load([b, h, j * BLOCK_N, 0]), (BLOCK_N, 64))
    K1 = tl.reshape(K1_desc.load([b, h, j * BLOCK_N, 0]), (BLOCK_N, 64))
    V0 = tl.reshape(V0_desc.load([b, h, j * BLOCK_N, 0]), (BLOCK_N, 64))
    V1 = tl.reshape(V1_desc.load([b, h, j * BLOCK_N, 0]), (BLOCK_N, 64))
    
    dK0_acc = tl.zeros((BLOCK_N, 64), tl.float32)
    dK1_acc = tl.zeros((BLOCK_N, 64), tl.float32)
    dV0_acc = tl.zeros((BLOCK_N, 64), tl.float32)
    dV1_acc = tl.zeros((BLOCK_N, 64), tl.float32)
    
    num_blocks = (S + BLOCK_M - 1) // BLOCK_M
    
    cols = tl.arange(0, BLOCK_N)
    valid_c = (j * BLOCK_N + cols) < S
    
    for i in range(num_blocks):
        Q0 = tl.reshape(Q0_desc.load([b, h, i * BLOCK_M, 0]), (BLOCK_M, 64))
        Q1 = tl.reshape(Q1_desc.load([b, h, i * BLOCK_M, 0]), (BLOCK_M, 64))
        dO0 = tl.reshape(dO0_desc.load([b, h, i * BLOCK_M, 0]), (BLOCK_M, 64))
        dO1 = tl.reshape(dO1_desc.load([b, h, i * BLOCK_M, 0]), (BLOCK_M, 64))
        O0 = tl.reshape(O0_desc.load([b, h, i * BLOCK_M, 0]), (BLOCK_M, 64))
        O1 = tl.reshape(O1_desc.load([b, h, i * BLOCK_M, 0]), (BLOCK_M, 64))
        
        rows = tl.arange(0, BLOCK_M)
        row_idx = i * BLOCK_M + rows
        valid_r = row_idx < S
        
        L_idx = b * H * S + h * S + row_idx
        L_val = tl.load(L_ptr + L_idx, mask=valid_r, other=0.0)
        
        D_val = tl.sum(dO0.to(tl.float32) * O0.to(tl.float32) + dO1.to(tl.float32) * O1.to(tl.float32), axis=1)
        D_val = tl.where(valid_r, D_val, 0.0)
        
        S_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        S_acc = tl.dot(Q0, K0.T, S_acc)
        S_acc = tl.dot(Q1, K1.T, S_acc)
        
        S_acc = tl.where(valid_c[None, :], S_acc, -float('inf'))
        S_acc = tl.where(valid_r[:, None], S_acc, -float('inf'))
        
        P = tl.exp(S_acc * scale - L_val[:, None])
        
        dP_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        dP_acc = tl.dot(dO0, V0.T, dP_acc)
        dP_acc = tl.dot(dO1, V1.T, dP_acc)
        
        dS = P * (dP_acc - D_val[:, None]) * scale
        
        dV0_acc = tl.dot(P.T.to(tl.bfloat16), dO0, dV0_acc)
        dV1_acc = tl.dot(P.T.to(tl.bfloat16), dO1, dV1_acc)
        
        dK0_acc = tl.dot(dS.T.to(tl.bfloat16), Q0, dK0_acc)
        dK1_acc = tl.dot(dS.T.to(tl.bfloat16), Q1, dK1_acc)
    
    dK0_acc = tl.where(valid_c[:, None], dK0_acc, 0.0)
    dK1_acc = tl.where(valid_c[:, None], dK1_acc, 0.0)
    dV0_acc = tl.where(valid_c[:, None], dV0_acc, 0.0)
    dV1_acc = tl.where(valid_c[:, None], dV1_acc, 0.0)
    
    dK0_desc.store([b, h, j * BLOCK_N, 0], tl.reshape(dK0_acc.to(tl.bfloat16), (1, 1, BLOCK_N, 64)))
    dK1_desc.store([b, h, j * BLOCK_N, 0], tl.reshape(dK1_acc.to(tl.bfloat16), (1, 1, BLOCK_N, 64)))
    dV0_desc.store([b, h, j * BLOCK_N, 0], tl.reshape(dV0_acc.to(tl.bfloat16), (1, 1, BLOCK_N, 64)))
    dV1_desc.store([b, h, j * BLOCK_N, 0], tl.reshape(dV1_acc.to(tl.bfloat16), (1, 1, BLOCK_N, 64)))


@triton.jit
def _bwd_dQ_kernel(
    Q0_desc, Q1_desc, K0_desc, K1_desc, V0_desc, V1_desc,
    dO0_desc, dO1_desc, O0_desc, O1_desc, L_ptr,
    dQ0_desc, dQ1_desc,
    H, S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    i = tl.program_id(0)
    batch_h = tl.program_id(1)
    
    if i * BLOCK_M >= S:
        return
    
    b = batch_h // H
    h = batch_h % H
    
    Q0 = tl.reshape(Q0_desc.load([b, h, i * BLOCK_M, 0]), (BLOCK_M, 64))
    Q1 = tl.reshape(Q1_desc.load([b, h, i * BLOCK_M, 0]), (BLOCK_M, 64))
    dO0 = tl.reshape(dO0_desc.load([b, h, i * BLOCK_M, 0]), (BLOCK_M, 64))
    dO1 = tl.reshape(dO1_desc.load([b, h, i * BLOCK_M, 0]), (BLOCK_M, 64))
    O0 = tl.reshape(O0_desc.load([b, h, i * BLOCK_M, 0]), (BLOCK_M, 64))
    O1 = tl.reshape(O1_desc.load([b, h, i * BLOCK_M, 0]), (BLOCK_M, 64))
    
    rows = tl.arange(0, BLOCK_M)
    row_idx = i * BLOCK_M + rows
    valid_r = row_idx < S
    
    L_idx = b * H * S + h * S + row_idx
    L_val = tl.load(L_ptr + L_idx, mask=valid_r, other=0.0)
    
    D_val = tl.sum(dO0.to(tl.float32) * O0.to(tl.float32) + dO1.to(tl.float32) * O1.to(tl.float32), axis=1)
    D_val = tl.where(valid_r, D_val, 0.0)
    
    dQ0_acc = tl.zeros((BLOCK_M, 64), tl.float32)
    dQ1_acc = tl.zeros((BLOCK_M, 64), tl.float32)
    
    num_blocks = (S + BLOCK_N - 1) // BLOCK_N
    
    for j in range(num_blocks):
        K0 = tl.reshape(K0_desc.load([b, h, j * BLOCK_N, 0]), (BLOCK_N, 64))
        K1 = tl.reshape(K1_desc.load([b, h, j * BLOCK_N, 0]), (BLOCK_N, 64))
        V0 = tl.reshape(V0_desc.load([b, h, j * BLOCK_N, 0]), (BLOCK_N, 64))
        V1 = tl.reshape(V1_desc.load([b, h, j * BLOCK_N, 0]), (BLOCK_N, 64))
        
        cols = tl.arange(0, BLOCK_N)
        valid_c = (j * BLOCK_N + cols) < S
        
        S_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        S_acc = tl.dot(Q0, K0.T, S_acc)
        S_acc = tl.dot(Q1, K1.T, S_acc)
        
        S_acc = tl.where(valid_c[None, :], S_acc, -float('inf'))
        S_acc = tl.where(valid_r[:, None], S_acc, -float('inf'))
        
        P = tl.exp(S_acc * scale - L_val[:, None])
        
        dP_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        dP_acc = tl.dot(dO0, V0.T, dP_acc)
        dP_acc = tl.dot(dO1, V1.T, dP_acc)
        
        dS = P * (dP_acc - D_val[:, None]) * scale
        
        dQ0_acc = tl.dot(dS.to(tl.bfloat16), K0, dQ0_acc)
        dQ1_acc = tl.dot(dS.to(tl.bfloat16), K1, dQ1_acc)
    
    dQ0_acc = tl.where(valid_r[:, None], dQ0_acc, 0.0)
    dQ1_acc = tl.where(valid_r[:, None], dQ1_acc, 0.0)
    
    dQ0_desc.store([b, h, i * BLOCK_M, 0], tl.reshape(dQ0_acc.to(tl.bfloat16), (1, 1, BLOCK_M, 64)))
    dQ1_desc.store([b, h, i * BLOCK_M, 0], tl.reshape(dQ1_acc.to(tl.bfloat16), (1, 1, BLOCK_M, 64)))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    
    B_val, H_val, S_val, d_val = Q.shape
    assert d_val == 128
    
    B_total = B_val * H_val
    
    BLOCK_M = 64
    BLOCK_N = 64
    
    scale = 1.0 / (d_val ** 0.5)
    
    Q0_desc = TensorDescriptor.from_tensor(Q[:, :, :, 0:64], [1, 1, BLOCK_M, 64])
    Q1_desc = TensorDescriptor.from_tensor(Q[:, :, :, 64:128], [1, 1, BLOCK_M, 64])
    K0_desc = TensorDescriptor.from_tensor(K[:, :, :, 0:64], [1, 1, BLOCK_N, 64])
    K1_desc = TensorDescriptor.from_tensor(K[:, :, :, 64:128], [1, 1, BLOCK_N, 64])
    V0_desc = TensorDescriptor.from_tensor(V[:, :, :, 0:64], [1, 1, BLOCK_N, 64])
    V1_desc = TensorDescriptor.from_tensor(V[:, :, :, 64:128], [1, 1, BLOCK_N, 64])
    dO0_desc = TensorDescriptor.from_tensor(dO[:, :, :, 0:64], [1, 1, BLOCK_M, 64])
    dO1_desc = TensorDescriptor.from_tensor(dO[:, :, :, 64:128], [1, 1, BLOCK_M, 64])
    O0_desc = TensorDescriptor.from_tensor(O[:, :, :, 0:64], [1, 1, BLOCK_M, 64])
    O1_desc = TensorDescriptor.from_tensor(O[:, :, :, 64:128], [1, 1, BLOCK_M, 64])
    dQ0_desc = TensorDescriptor.from_tensor(dQ[:, :, :, 0:64], [1, 1, BLOCK_M, 64])
    dQ1_desc = TensorDescriptor.from_tensor(dQ[:, :, :, 64:128], [1, 1, BLOCK_M, 64])
    dK0_desc = TensorDescriptor.from_tensor(dK[:, :, :, 0:64], [1, 1, BLOCK_N, 64])
    dK1_desc = TensorDescriptor.from_tensor(dK[:, :, :, 64:128], [1, 1, BLOCK_N, 64])
    dV0_desc = TensorDescriptor.from_tensor(dV[:, :, :, 0:64], [1, 1, BLOCK_N, 64])
    dV1_desc = TensorDescriptor.from_tensor(dV[:, :, :, 64:128], [1, 1, BLOCK_N, 64])
    
    grid_dKdV = (triton.cdiv(S_val, BLOCK_N), B_total)
    _bwd_dKdV_kernel[grid_dKdV](
        Q0_desc, Q1_desc, K0_desc, K1_desc, V0_desc, V1_desc,
        dO0_desc, dO1_desc, O0_desc, O1_desc, L.data_ptr(),
        dK0_desc, dK1_desc, dV0_desc, dV1_desc,
        H_val, S_val, scale, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=8, num_stages=3
    )
    
    grid_dQ = (triton.cdiv(S_val, BLOCK_M), B_total)
    _bwd_dQ_kernel[grid_dQ](
        Q0_desc, Q1_desc, K0_desc, K1_desc, V0_desc, V1_desc,
        dO0_desc, dO1_desc, O0_desc, O1_desc, L.data_ptr(),
        dQ0_desc, dQ1_desc,
        H_val, S_val, scale, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=8, num_stages=3
    )