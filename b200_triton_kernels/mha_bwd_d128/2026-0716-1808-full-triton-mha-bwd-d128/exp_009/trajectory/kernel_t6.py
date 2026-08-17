import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)


@triton.jit
def bwd_dq_kernel(
    q_desc, k_desc, v_desc, o_desc, do_desc, L_flat, dq_desc,
    S, scale, BLOCK_M, BLOCK_N, BLOCK_K
):
    i = tl.program_id(0)
    bh = tl.program_id(1)
    
    start_row = bh * S + i * BLOCK_M
    
    D = tl.zeros((BLOCK_M, 1), dtype=tl.float32)
    for k in range(4):
        do_tile = do_desc.load([start_row, k * BLOCK_K])
        o_tile = o_desc.load([start_row, k * BLOCK_K])
        D += tl.sum(do_tile * o_tile, axis=1, keep_dims=True)
        
    mask_l = (i * BLOCK_M + tl.arange(0, BLOCK_M)) < S
    L_i = L_flat[bh * S + start_row + tl.arange(0, BLOCK_M)]
    
    dq_acc = [tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32) for _ in range(4)]
    
    num_blocks = tl.cdiv(S, BLOCK_N)
    for j in range(num_blocks):
        S_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(4):
            q_tile = q_desc.load([start_row, k * BLOCK_K])
            k_tile = k_desc.load([bh * S + j * BLOCK_N, k * BLOCK_K])
            S_acc = tl.dot(q_tile, k_tile.T, S_acc)
            
        dP_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(4):
            do_tile = do_desc.load([start_row, k * BLOCK_K])
            v_tile = v_desc.load([bh * S + j * BLOCK_N, k * BLOCK_K])
            dP_acc = tl.dot(do_tile, v_tile.T, dP_acc)
            
        P = tl.exp(S_acc * scale - L_i[:, None])
        
        row = i * BLOCK_M + tl.arange(0, BLOCK_M)
        col = j * BLOCK_N + tl.arange(0, BLOCK_N)
        valid = (row[:, None] < S) & (col[None, :] < S)
        P = P * valid.to(tl.float32)
        
        dS = P * (dP_acc - D) * scale
        
        for k_idx in range(4):
            k_tile_b = k_desc.load([bh * S + j * BLOCK_N, k_idx * BLOCK_K])
            dq_acc[k_idx] = tl.dot(dS, k_tile_b, dq_acc[k_idx])
            
    for k_idx in range(4):
        dq_desc.store([start_row, k_idx * BLOCK_K], dq_acc[k_idx].to(tl.bfloat16))


@triton.jit
def bwd_dkv_kernel(
    q_desc, k_desc, v_desc, o_desc, do_desc, L_flat, dk_desc, dv_desc,
    S, scale, BLOCK_M, BLOCK_N, BLOCK_K
):
    j = tl.program_id(0)
    bh = tl.program_id(1)
    
    start_row = bh * S + j * BLOCK_N
    
    dk_acc = [tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32) for _ in range(4)]
    dv_acc = [tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32) for _ in range(4)]
    
    num_blocks = tl.cdiv(S, BLOCK_M)
    for i in range(num_blocks):
        D = tl.zeros((BLOCK_M, 1), dtype=tl.float32)
        for k in range(4):
            do_tile = do_desc.load([bh * S + i * BLOCK_M, k * BLOCK_K])
            o_tile = o_desc.load([bh * S + i * BLOCK_M, k * BLOCK_K])
            D += tl.sum(do_tile * o_tile, axis=1, keep_dims=True)
            
        mask_l = (i * BLOCK_M + tl.arange(0, BLOCK_M)) < S
        L_i = L_flat[bh * S + i * BLOCK_M + tl.arange(0, BLOCK_M)]
        
        S_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(4):
            q_tile = q_desc.load([bh * S + i * BLOCK_M, k * BLOCK_K])
            k_tile = k_desc.load([start_row, k * BLOCK_K])
            S_acc = tl.dot(q_tile, k_tile.T, S_acc)
            
        dP_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(4):
            do_tile = do_desc.load([bh * S + i * BLOCK_M, k * BLOCK_K])
            v_tile = v_desc.load([start_row, k * BLOCK_K])
            dP_acc = tl.dot(do_tile, v_tile.T, dP_acc)
            
        P = tl.exp(S_acc * scale - L_i[:, None])
        
        row = i * BLOCK_M + tl.arange(0, BLOCK_M)
        col = j * BLOCK_N + tl.arange(0, BLOCK_N)
        valid = (row[:, None] < S) & (col[None, :] < S)
        P = P * valid.to(tl.float32)
        
        dS = P * (dP_acc - D) * scale
        
        for k_idx in range(4):
            q_tile_b = q_desc.load([bh * S + i * BLOCK_M, k_idx * BLOCK_K])
            dk_acc[k_idx] = tl.dot(dS.T, q_tile_b, dk_acc[k_idx])
            
            do_tile_b = do_desc.load([bh * S + i * BLOCK_M, k_idx * BLOCK_K])
            dv_acc[k_idx] = tl.dot(P.T, do_tile_b, dv_acc[k_idx])
            
    for k_idx in range(4):
        dk_desc.store([start_row, k_idx * BLOCK_K], dk_acc[k_idx].to(tl.bfloat16))
        dv_desc.store([start_row, k_idx * BLOCK_K], dv_acc[k_idx].to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute the backward pass of Multi-Head Attention."""
    B, H, S, d = Q.shape
    assert B == 4 and H == 48 and d == 128
    assert O.shape == (B, H, S, d) and dO.shape == (B, H, S, d)
    assert L.shape == (B, H, S)
    assert dQ.shape == (B, H, S, d) and dK.shape == (B, H, S, d) and dV.shape == (B, H, S, d)
    
    torch.cuda.empty_cache()
    torch.cuda.set_device(Q.device)
    
    total_S = B * H * S
    scale = 1.0 / (d ** 0.5)
    
    Q_2d = Q.view(total_S, d)
    K_2d = K.view(total_S, d)
    V_2d = V.view(total_S, d)
    O_2d = O.view(total_S, d)
    dO_2d = dO.view(total_S, d)
    dQ_2d = dQ.view(total_S, d)
    dK_2d = dK.view(total_S, d)
    dV_2d = dV.view(total_S, d)
    
    q_desc = TensorDescriptor.from_tensor(Q_2d, [64, 32])
    k_desc = TensorDescriptor.from_tensor(K_2d, [64, 32])
    v_desc = TensorDescriptor.from_tensor(V_2d, [64, 32])
    o_desc = TensorDescriptor.from_tensor(O_2d, [64, 32])
    do_desc = TensorDescriptor.from_tensor(dO_2d, [64, 32])
    dq_desc = TensorDescriptor.from_tensor(dQ_2d, [64, 32])
    dk_desc = TensorDescriptor.from_tensor(dK_2d, [64, 32])
    dv_desc = TensorDescriptor.from_tensor(dV_2d, [64, 32])
    
    num_blocks = triton.cdiv(S, 64)
    grid = (num_blocks, B * H)
    
    L_flat = L.view(B * H * S)
    
    bwd_dq_kernel[grid](q_desc, k_desc, v_desc, o_desc, do_desc, L_flat, dq_desc,
                        S, scale, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
                        num_warps=8, num_stages=3)
                        
    bwd_dkv_kernel[grid](q_desc, k_desc, v_desc, o_desc, do_desc, L_flat, dk_desc, dv_desc,
                         S, scale, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
                         num_warps=8, num_stages=3)