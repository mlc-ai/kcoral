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
    
    K0_tile = K0_desc.load([b, h, j * BLOCK_N, 0])
    K0 = tl.reshape(K0_tile, (BLOCK_N, 64))
    K1_tile = K1_desc.load([b, h, j * BLOCK_N, 0])
    K1 = tl.reshape(K1_tile, (BLOCK_N, 64))
    
    V0_tile = V0_desc.load([b, h, j * BLOCK_N, 0])
    V0 = tl.reshape(V0_tile, (BLOCK_N, 64))
    V1_tile = V1_desc.load([b, h, j * BLOCK_N, 0])
    V1 = tl.reshape(V1_tile, (BLOCK_N, 64))
    
    dK0_acc = tl.zeros((BLOCK_N, 64), tl.float32)
    dK1_acc = tl.zeros((BLOCK_N, 64), tl.float32)
    dV0_acc = tl.zeros((BLOCK_N, 64), tl.float32)
    dV1_acc = tl.zeros((BLOCK_N, 64), tl.float32)
    
    num_blocks = (S + BLOCK_M - 1) // BLOCK_M
    
    cols = tl.arange(0, BLOCK_N)
    valid_c = (j * BLOCK_N + cols) < S
    
    for i in range(num_blocks):
        Q0_tile = Q0_desc.load([b, h, i * BLOCK_M, 0])
        Q0 = tl.reshape(Q0_tile, (BLOCK_M, 64))
        Q1_tile = Q1_desc.load([b, h, i * BLOCK_M, 0])
        Q1 = tl.reshape(Q1_tile, (BLOCK_M, 64))
        
        dO0_tile = dO0_desc.load([b, h, i * BLOCK_M, 0])
        dO0 = tl.reshape(dO0_tile, (BLOCK_M, 64))
        dO1_tile = dO1_desc.load([b, h, i * BLOCK_M, 0])
        dO1 = tl.reshape(dO1_tile, (BLOCK_M, 64))
        
        O0_tile = O0_desc.load([b, h, i * BLOCK_M, 0])
        O0 = tl.reshape(O0_tile, (BLOCK_M, 64))
        O1_tile = O1_desc.load([b, h, i * BLOCK_M, 0])
        O1 = tl.reshape(O1_tile, (BLOCK_M, 64))
        
        rows = tl.arange(0, BLOCK_M)
        row_idx = i * BLOCK_M + rows
        valid_r = row_idx < S
        
        L_idx = batch_h * S + row_idx
        L_val = tl.load(L_ptr + L_idx, mask=valid_r, other=0.0)
        
        D_val = tl.sum(dO0.to(tl.float32) * O0.to(tl.float32) + dO1.to(tl.float32) * O1.to(tl.float32), axis=1)
        
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
    
    dK0_desc.store([b, h, j * BLOCK_N, 0], dK0_acc.to(tl.bfloat16))
    dK1_desc.store([b, h, j * BLOCK_N, 0], dK1_acc.to(tl.bfloat16))
    dV0_desc.store([b, h, j * BLOCK_N, 0], dV0_acc.to(tl.bfloat16))
    dV1_desc.store([b, h, j * BLOCK_N, 0], dV1_acc.to(tl.bfloat16))


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
    
    Q0_tile = Q0_desc.load([b, h, i * BLOCK_M, 0])
    Q0 = tl.reshape(Q0_tile, (BLOCK_M, 64))
    Q1_tile = Q1_desc.load([b, h, i * BLOCK_M, 0])
    Q1 = tl.reshape(Q1_tile, (BLOCK_M, 64))
    
    dO0_tile = dO0_desc.load([b, h, i * BLOCK_M, 0])
    dO0 = tl.reshape(dO0_tile, (BLOCK_M, 64))
    dO1_tile = dO1_desc.load([b, h, i * BLOCK_M, 0])
    dO1 = tl.reshape(dO1_tile, (BLOCK_M, 64))
    
    O0_tile = O0_desc.load([b, h, i * BLOCK_M, 0])
    O0 = tl.reshape(O0_tile, (BLOCK_M, 64))
    O1_tile = O1_desc.load([b, h, i * BLOCK_M, 0])
    O1 = tl.reshape(O1_tile, (BLOCK_M, 64))
    
    rows = tl.arange(0, BLOCK_M)
    row_idx = i * BLOCK_M + rows
    valid_r = row_idx < S
    
    L_idx = batch_h * S + row_idx
    L_val = tl.load(L_ptr + L_idx, mask=valid_r, other=0.0)
    
    D_val = tl.sum(dO0.to(tl.float32) * O0.to(tl.float32) + dO1.to(tl.float32) * O1.to(tl.float32), axis=1)
    
    dQ0_acc = tl.zeros((BLOCK_M, 64), tl.float32)
    dQ1_acc = tl.zeros((BLOCK_M, 64), tl.float32)
    
    num_blocks = (S + BLOCK_N - 1) // BLOCK_N
    
    for j in range(num_blocks):
        K0_tile = K0_desc.load([b, h, j * BLOCK_N, 0])
        K0 = tl.reshape(K0_tile, (BLOCK_N, 64))
        K1_tile = K1_desc.load([b, h, j * BLOCK_N, 0])
        K1 = tl.reshape(K1_tile, (BLOCK_N, 64))
        
        V0_tile = V0_desc.load([b, h, j * BLOCK_N, 0])
        V0 = tl.reshape(V0_tile, (BLOCK_N, 64))
        V1_tile = V1_desc.load([b, h, j * BLOCK_N, 0])
        V1 = tl.reshape(V1_tile, (BLOCK_N, 64))
        
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
    
    dQ0_desc.store([b, h, i * BLOCK_M, 0], dQ0_acc.to(tl.bfloat16))
    dQ1_desc.store([b, h, i * BLOCK_M, 0], dQ1_acc.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    
    B_val, H_val, S_val, d_val = Q.shape
    assert d_val == 128
    
    B_total = B_val * H_val
    
    BLOCK_M = 64
    BLOCK_N = 64
    
    scale = 1.0 / (d_val ** 0.5)
    
    Q0_t = Q[:, :, :, 0:64].contiguous()
    Q1_t = Q[:, :, :, 64:128].contiguous()
    K0_t = K[:, :, :, 0:64].contiguous()
    K1_t = K[:, :, :, 64:128].contiguous()
    V0_t = V[:, :, :, 0:64].contiguous()
    V1_t = V[:, :, :, 64:128].contiguous()
    O0_t = O[:, :, :, 0:64].contiguous()
    O1_t = O[:, :, :, 64:128].contiguous()
    dO0_t = dO[:, :, :, 0:64].contiguous()
    dO1_t = dO[:, :, :, 64:128].contiguous()
    dQ0_t = dQ[:, :, :, 0:64].contiguous()
    dQ1_t = dQ[:, :, :, 64:128].contiguous()
    dK0_t = dK[:, :, :, 0:64].contiguous()
    dK1_t = dK[:, :, :, 64:128].contiguous()
    dV0_t = dV[:, :, :, 0:64].contiguous()
    dV1_t = dV[:, :, :, 64:128].contiguous()
    
    Q0_desc = TensorDescriptor.from_tensor(Q0_t, [1, 1, BLOCK_M, 64])
    Q1_desc = TensorDescriptor.from_tensor(Q1_t, [1, 1, BLOCK_M, 64])
    K0_desc = TensorDescriptor.from_tensor(K0_t, [1, 1, BLOCK_N, 64])
    K1_desc = TensorDescriptor.from_tensor(K1_t, [1, 1, BLOCK_N, 64])
    V0_desc = TensorDescriptor.from_tensor(V0_t, [1, 1, BLOCK_N, 64])
    V1_desc = TensorDescriptor.from_tensor(V1_t, [1, 1, BLOCK_N, 64])
    dO0_desc = TensorDescriptor.from_tensor(dO0_t, [1, 1, BLOCK_M, 64])
    dO1_desc = TensorDescriptor.from_tensor(dO1_t, [1, 1, BLOCK_M, 64])
    O0_desc = TensorDescriptor.from_tensor(O0_t, [1, 1, BLOCK_M, 64])
    O1_desc = TensorDescriptor.from_tensor(O1_t, [1, 1, BLOCK_M, 64])
    dQ0_desc = TensorDescriptor.from_tensor(dQ0_t, [1, 1, BLOCK_M, 64])
    dQ1_desc = TensorDescriptor.from_tensor(dQ1_t, [1, 1, BLOCK_M, 64])
    dK0_desc = TensorDescriptor.from_tensor(dK0_t, [1, 1, BLOCK_N, 64])
    dK1_desc = TensorDescriptor.from_tensor(dK1_t, [1, 1, BLOCK_N, 64])
    dV0_desc = TensorDescriptor.from_tensor(dV0_t, [1, 1, BLOCK_N, 64])
    dV1_desc = TensorDescriptor.from_tensor(dV1_t, [1, 1, BLOCK_N, 64])
    
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