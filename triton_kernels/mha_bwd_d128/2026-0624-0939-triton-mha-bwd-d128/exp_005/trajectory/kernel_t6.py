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
    
    k_offset = batch_h * S + j * BLOCK_N
    
    K0_tile = K0_desc.load([k_offset, 0])
    K1_tile = K1_desc.load([k_offset, 0])
    V0_tile = V0_desc.load([k_offset, 0])
    V1_tile = V1_desc.load([k_offset, 0])
    
    dK0_acc = tl.zeros((BLOCK_N, 64), tl.float32)
    dK1_acc = tl.zeros((BLOCK_N, 64), tl.float32)
    dV0_acc = tl.zeros((BLOCK_N, 64), tl.float32)
    dV1_acc = tl.zeros((BLOCK_N, 64), tl.float32)
    
    num_blocks = (S + BLOCK_M - 1) // BLOCK_M
    
    cols = tl.arange(0, BLOCK_N)
    col_idx = j * BLOCK_N + cols
    valid_c = col_idx < S
    
    b = batch_h // H
    h = batch_h % H
    
    for i in range(num_blocks):
        q_offset = batch_h * S + i * BLOCK_M
        
        Q0_tile = Q0_desc.load([q_offset, 0])
        Q1_tile = Q1_desc.load([q_offset, 0])
        dO0_tile = dO0_desc.load([q_offset, 0])
        dO1_tile = dO1_desc.load([q_offset, 0])
        O0_tile = O0_desc.load([q_offset, 0])
        O1_tile = O1_desc.load([q_offset, 0])
        
        rows = tl.arange(0, BLOCK_M)
        row_idx = i * BLOCK_M + rows
        valid_r = row_idx < S
        
        L_idx = b * H * S + h * S + row_idx
        L_val = tl.load(L_ptr + L_idx, mask=valid_r, other=0.0)
        
        D_val = tl.sum(dO0_tile.to(tl.float32) * O0_tile.to(tl.float32) + dO1_tile.to(tl.float32) * O1_tile.to(tl.float32), axis=1)
        
        S_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        S_acc = tl.dot(Q0_tile, K0_tile.T, S_acc)
        S_acc = tl.dot(Q1_tile, K1_tile.T, S_acc)
        
        P = tl.exp(S_acc * scale - L_val[:, None])
        P = P * valid_c[None, :]
        P = P * valid_r[:, None]
        
        dP_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        dP_acc = tl.dot(dO0_tile, V0_tile.T, dP_acc)
        dP_acc = tl.dot(dO1_tile, V1_tile.T, dP_acc)
        
        dS = P * (dP_acc - D_val[:, None]) * scale
        dS = dS * valid_c[None, :]
        dS = dS * valid_r[:, None]
        
        dV0_acc = tl.dot(P.T.to(tl.bfloat16), dO0_tile, dV0_acc)
        dV1_acc = tl.dot(P.T.to(tl.bfloat16), dO1_tile, dV1_acc)
        
        dK0_acc = tl.dot(dS.T.to(tl.bfloat16), Q0_tile, dK0_acc)
        dK1_acc = tl.dot(dS.T.to(tl.bfloat16), Q1_tile, dK1_acc)
    
    valid_r_out = (j * BLOCK_N + cols) < S
    dK0_acc *= valid_r_out[:, None]
    dK1_acc *= valid_r_out[:, None]
    dV0_acc *= valid_r_out[:, None]
    dV1_acc *= valid_r_out[:, None]
    
    dK0_desc.store([k_offset, 0], dK0_acc.to(tl.bfloat16))
    dK1_desc.store([k_offset, 0], dK1_acc.to(tl.bfloat16))
    dV0_desc.store([k_offset, 0], dV0_acc.to(tl.bfloat16))
    dV1_desc.store([k_offset, 0], dV1_acc.to(tl.bfloat16))


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
    
    q_offset = batch_h * S + i * BLOCK_M
    
    Q0_tile = Q0_desc.load([q_offset, 0])
    Q1_tile = Q1_desc.load([q_offset, 0])
    dO0_tile = dO0_desc.load([q_offset, 0])
    dO1_tile = dO1_desc.load([q_offset, 0])
    O0_tile = O0_desc.load([q_offset, 0])
    O1_tile = O1_desc.load([q_offset, 0])
    
    rows = tl.arange(0, BLOCK_M)
    row_idx = i * BLOCK_M + rows
    valid_r = row_idx < S
    
    b = batch_h // H
    h = batch_h % H
    
    L_idx = b * H * S + h * S + row_idx
    L_val = tl.load(L_ptr + L_idx, mask=valid_r, other=0.0)
    
    D_val = tl.sum(dO0_tile.to(tl.float32) * O0_tile.to(tl.float32) + dO1_tile.to(tl.float32) * O1_tile.to(tl.float32), axis=1)
    
    dQ0_acc = tl.zeros((BLOCK_M, 64), tl.float32)
    dQ1_acc = tl.zeros((BLOCK_M, 64), tl.float32)
    
    num_blocks = (S + BLOCK_N - 1) // BLOCK_N
    
    for j in range(num_blocks):
        k_offset = batch_h * S + j * BLOCK_N
        
        K0_tile = K0_desc.load([k_offset, 0])
        K1_tile = K1_desc.load([k_offset, 0])
        V0_tile = V0_desc.load([k_offset, 0])
        V1_tile = V1_desc.load([k_offset, 0])
        
        cols = tl.arange(0, BLOCK_N)
        valid_c = (j * BLOCK_N + cols) < S
        
        S_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        S_acc = tl.dot(Q0_tile, K0_tile.T, S_acc)
        S_acc = tl.dot(Q1_tile, K1_tile.T, S_acc)
        
        P = tl.exp(S_acc * scale - L_val[:, None])
        P = P * valid_c[None, :]
        P = P * valid_r[:, None]
        
        dP_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        dP_acc = tl.dot(dO0_tile, V0_tile.T, dP_acc)
        dP_acc = tl.dot(dO1_tile, V1_tile.T, dP_acc)
        
        dS = P * (dP_acc - D_val[:, None]) * scale
        dS = dS * valid_c[None, :]
        dS = dS * valid_r[:, None]
        
        dQ0_acc = tl.dot(dS.to(tl.bfloat16), K0_tile, dQ0_acc)
        dQ1_acc = tl.dot(dS.to(tl.bfloat16), K1_tile, dQ1_acc)
    
    dQ0_acc *= valid_r[:, None]
    dQ1_acc *= valid_r[:, None]
    
    dQ0_desc.store([q_offset, 0], dQ0_acc.to(tl.bfloat16))
    dQ1_desc.store([q_offset, 0], dQ1_acc.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    
    B_val, H_val, S_val, d_val = Q.shape
    assert d_val == 128
    
    B_total = B_val * H_val
    
    BLOCK_M = 64
    BLOCK_N = 64
    
    scale = 1.0 / (d_val ** 0.5)
    
    Q0_2d = torch.narrow(Q, 3, 0, 64).view(B_total * S_val, 64).contiguous()
    Q1_2d = torch.narrow(Q, 3, 64, 64).view(B_total * S_val, 64).contiguous()
    K0_2d = torch.narrow(K, 3, 0, 64).view(B_total * S_val, 64).contiguous()
    K1_2d = torch.narrow(K, 3, 64, 64).view(B_total * S_val, 64).contiguous()
    V0_2d = torch.narrow(V, 3, 0, 64).view(B_total * S_val, 64).contiguous()
    V1_2d = torch.narrow(V, 3, 64, 64).view(B_total * S_val, 64).contiguous()
    O0_2d = torch.narrow(O, 3, 0, 64).view(B_total * S_val, 64).contiguous()
    O1_2d = torch.narrow(O, 3, 64, 64).view(B_total * S_val, 64).contiguous()
    dO0_2d = torch.narrow(dO, 3, 0, 64).view(B_total * S_val, 64).contiguous()
    dO1_2d = torch.narrow(dO, 3, 64, 64).view(B_total * S_val, 64).contiguous()
    dQ0_2d = torch.narrow(dQ, 3, 0, 64).view(B_total * S_val, 64).contiguous()
    dQ1_2d = torch.narrow(dQ, 3, 64, 64).view(B_total * S_val, 64).contiguous()
    dK0_2d = torch.narrow(dK, 3, 0, 64).view(B_total * S_val, 64).contiguous()
    dK1_2d = torch.narrow(dK, 3, 64, 64).view(B_total * S_val, 64).contiguous()
    dV0_2d = torch.narrow(dV, 3, 0, 64).view(B_total * S_val, 64).contiguous()
    dV1_2d = torch.narrow(dV, 3, 64, 64).view(B_total * S_val, 64).contiguous()
    
    Q0_desc = TensorDescriptor.from_tensor(Q0_2d, [BLOCK_M, 64])
    Q1_desc = TensorDescriptor.from_tensor(Q1_2d, [BLOCK_M, 64])
    K0_desc = TensorDescriptor.from_tensor(K0_2d, [BLOCK_N, 64])
    K1_desc = TensorDescriptor.from_tensor(K1_2d, [BLOCK_N, 64])
    V0_desc = TensorDescriptor.from_tensor(V0_2d, [BLOCK_N, 64])
    V1_desc = TensorDescriptor.from_tensor(V1_2d, [BLOCK_N, 64])
    dO0_desc = TensorDescriptor.from_tensor(dO0_2d, [BLOCK_M, 64])
    dO1_desc = TensorDescriptor.from_tensor(dO1_2d, [BLOCK_M, 64])
    O0_desc = TensorDescriptor.from_tensor(O0_2d, [BLOCK_M, 64])
    O1_desc = TensorDescriptor.from_tensor(O1_2d, [BLOCK_M, 64])
    dQ0_desc = TensorDescriptor.from_tensor(dQ0_2d, [BLOCK_M, 64])
    dQ1_desc = TensorDescriptor.from_tensor(dQ1_2d, [BLOCK_M, 64])
    dK0_desc = TensorDescriptor.from_tensor(dK0_2d, [BLOCK_N, 64])
    dK1_desc = TensorDescriptor.from_tensor(dK1_2d, [BLOCK_N, 64])
    dV0_desc = TensorDescriptor.from_tensor(dV0_2d, [BLOCK_N, 64])
    dV1_desc = TensorDescriptor.from_tensor(dV1_2d, [BLOCK_N, 64])
    
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