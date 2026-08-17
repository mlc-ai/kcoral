import torch
import triton
import triton.language as tl
import math


@triton.jit
def load_bf16_block(base_ptr, row, col, mask):
    x = base_ptr + row * 128 + col
    return tl.load(x, mask=mask, other=0.0)


@triton.jit
def store_bf16_block(base_ptr, row, col, value, mask):
    x = base_ptr + row * 128 + col
    tl.store(x, value.to(tl.bfloat16), mask=mask)


@triton.jit
def f16_to_f32(x):
    return x.to(tl.float32)


@triton.jit
def dot_128x128(A, B):
    acc = tl.zeros((128, 128), dtype=tl.float32)
    for i in range(0, 128, 16):
        a_slice = A[:, i:i+16]
        b_slice = B[i:i+16, :]
        acc += a_slice @ b_slice
    return acc


@triton.jit
def trans_a_gemm(A, B):
    acc = tl.zeros((128, 128), dtype=tl.float32)
    for i in range(0, 128, 16):
        a_slice = A[i:i+16, :]
        b_slice = B[i:i+16, :]
        acc += a_slice.T @ b_slice
    return acc


@triton.jit
def transpose_gemm(A, B):
    acc = tl.zeros((128, 128), dtype=tl.float32)
    for i in range(0, 128, 16):
        a_slice = A[i:i+16, :]
        b_slice = B[:, i:i+16]
        acc += a_slice.T @ b_slice.T
    return acc


@triton.jit
def add_gemm_128x128x128(acc, gemm_result):
    return acc + gemm_result


@triton.jit
def mha_bwd_kernel(
    Q, K, V, O, dO, L, dQ, dK, dV, P_bh_ptr, S
):
    bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    
    r_base = pid_s * 128
    c_base = pid_s * 128
    
    Q_bh = Q[bh]
    K_bh = K[bh]
    V_bh = V[bh]
    O_bh = O[bh]
    dO_bh = dO[bh]
    L_bh = L[bh]
    
    dQ_acc = tl.zeros((128, 128), dtype=tl.float32)
    dK_acc = tl.zeros((128, 128), dtype=tl.float32)
    dV_acc = tl.zeros((128, 128), dtype=tl.float32)
    
    r_idx = tl.arange(0, 128)
    c_idx = tl.arange(0, 128)
    d_idx = tl.arange(0, 128)
    
    Q_tile = load_bf16_block(Q_bh, r_base + r_idx[:, None], d_idx, ((r_base + r_idx[:, None]) < S))
    O_tile = load_bf16_block(O_bh, r_base + r_idx[:, None], d_idx, ((r_base + r_idx[:, None]) < S))
    dO_tile = load_bf16_block(dO_bh, r_base + r_idx[:, None], d_idx, ((r_base + r_idx[:, None]) < S))
    
    for c_base_inner in range(0, r_base + 128, 128):
        K_tile = load_bf16_block(K_bh, c_base_inner + r_idx[:, None], d_idx, ((c_base_inner + r_idx[:, None]) < S))
        V_tile = load_bf16_block(V_bh, c_base_inner + r_idx[:, None], d_idx, ((c_base_inner + r_idx[:, None]) < S))
        
        K_tile_T = K_tile.T
        S_h = dot_128x128(Q_tile, K_tile_T) / math.sqrt(d)
        
        L_h = L_bh[r_base + r_idx][:, None]
        exp_L_h = tl.math.exp((S_h / math.sqrt(d)) - L_h)
        mask_h = (r_base + r_idx[:, None]) >= (c_base_inner + c_idx[None, :])
        mask_h = mask_h & ((r_base + r_idx[:, None]) < S)
        mask_h = mask_h & ((c_base_inner + c_idx[None, :]) < S)
        
        P_h = exp_L_h * mask_h
        row_sum_exp_h = tl.sum(exp_L_h * mask_h, axis=1)
        P_h = P_h / row_sum_exp_h
        
        V_tile_T = V_tile.T
        dP_h = dot_128x128(dO_tile, V_tile_T)
        
        O_sum_exp_h = (dO_tile * O_tile).sum(axis=1)[:, None]
        
        D_S_h = P_h * (dP_h - O_sum_exp_h)
        
        dQ_acc = add_gemm_128x128x128(dQ_acc, dot_128x128(D_S_h, K_tile) / math.sqrt(d))
        
        D_S_h_T = D_S_h.T
        dK_acc = add_gemm_128x128x128(dK_acc, trans_a_gemm(D_S_h_T, Q_tile) / math.sqrt(d))
        
        ptr_P = P_bh_ptr + bh * S * S + (r_base//128)*128*128 + (c_base_inner//128)*128 + (r_base%128)*128
        write_float16_block(ptr_P, P_h)
    
    base = P_bh_ptr + bh * S * S
    for r_base_outer in range(c_base, S, 128):
        ptr_P = base + (r_base_outer//128)*128*128 + (c_base//128)*128 + (r_base_outer%128)*128
        P_block = read_float16_block(ptr_P, 128, 128)
        ptr_dO = dO_bh + r_base_outer * 128 * 128
        dO_block = read_bf16_block(ptr_dO, 128, 128)
        
        P_block_f32 = f16_to_f32(P_block)
        dO_block_f32 = f16_to_f32(dO_block)
        
        dV_acc = add_gemm_128x128x128(dV_acc, transpose_gemm(P_block_f32, dO_block_f32))
    
    dQ_bh = dQ[bh]
    dK_bh = dK[bh]
    dV_bh = dV[bh]
    
    store_bf16_block(dQ_bh, r_base + r_idx[:, None], d_idx, dQ_acc, ((r_base + r_idx[:, None]) < S))
    store_bf16_block(dK_bh, c_base + r_idx[:, None], d_idx, dK_acc, ((c_base + r_idx[:, None]) < S))
    store_bf16_block(dV_bh, c_base + r_idx[:, None], d_idx, dV_acc, ((c_base + r_idx[:, None]) < S))


@triton.jit
def write_float16_block(base_ptr, block_2d):
    rows, cols = block_2d.shape
    for r in range(rows):
        for c in range(cols):
            val = block_2d[r, c]
            x = (base_ptr + r * 128 + c).to(tl.pointer_ty(base_ptr.dtype.element_ty))
            tl.store(x, val.to(tl.bfloat16))

@triton.jit
def read_float16_block(base_ptr, rows, cols):
    block = []
    for r in range(rows):
        row_data = []
        for c in range(cols):
            x = (base_ptr + r * 128 + c).to(tl.pointer_ty(base_ptr.dtype.element_ty))
            val = tl.load(x)
            row_data.append(val)
        block.append(row_data)
    return block

@triton.jit
def read_bf16_block(base_ptr, rows, cols):
    block = []
    for r in range(rows):
        row_data = []
        for c in range(cols):
            x = (base_ptr + r * 128 + c).to(tl.pointer_ty(base_ptr.dtype.element_ty))
            val = tl.load(x)
            row_data.append(val)
        block.append(row_data)
    return block


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    b, h, s, d = Q.shape
    S = s
    grid = (b * h, triton.cdiv(S, 128))
    
    P_bh_ptr = torch.empty((b * h, S * S), device=Q.device, dtype=torch.float16)
    
    _mha_bwd_kernel[grid](Q, K, V, O, dO, L, dQ, dK, dV, P_bh_ptr, S)


@triton.jit
def _mha_bwd_kernel(Q, K, V, O, dO, L, dQ, dK, dV, P_bh_ptr, S):
    bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    
    r_base = pid_s * 128
    c_base = pid_s * 128
    
    Q_bh = Q[bh]
    K_bh = K[bh]
    V_bh = V[bh]
    O_bh = O[bh]
    dO_bh = dO[bh]
    L_bh = L[bh]
    
    dQ_acc = tl.zeros((128, 128), dtype=tl.float32)
    dK_acc = tl.zeros((128, 128), dtype=tl.float32)
    dV_acc = tl.zeros((128, 128), dtype=tl.float32)
    
    r_idx = tl.arange(0, 128)
    c_idx = tl.arange(0, 128)
    d_idx = tl.arange(0, 128)
    
    Q_tile = load_bf16_block(Q_bh, r_base + r_idx[:, None], d_idx, ((r_base + r_idx[:, None]) < S))
    O_tile = load_bf16_block(O_bh, r_base + r_idx[:, None], d_idx, ((r_base + r_idx[:, None]) < S))
    dO_tile = load_bf16_block(dO_bh, r_base + r_idx[:, None], d_idx, ((r_base + r_idx[:, None]) < S))
    
    for c_base_inner in range(0, r_base + 128, 128):
        K_tile = load_bf16_block(K_bh, c_base_inner + r_idx[:, None], d_idx, ((c_base_inner + r_idx[:, None]) < S))
        V_tile = load_bf16_block(V_bh, c_base_inner + r_idx[:, None], d_idx, ((c_base_inner + r_idx[:, None]) < S))
        
        K_tile_T = K_tile.T
        S_h = dot_128x128(Q_tile, K_tile_T) / math.sqrt(d)
        
        L_h = L_bh[r_base + r_idx][:, None]
        exp_L_h = tl.math.exp((S_h / math.sqrt(d)) - L_h)
        mask_h = (r_base + r_idx[:, None]) >= (c_base_inner + c_idx[None, :])
        mask_h = mask_h & ((r_base + r_idx[:, None]) < S)
        mask_h = mask_h & ((c_base_inner + c_idx[None, :]) < S)
        
        P_h = exp_L_h * mask_h
        row_sum_exp_h = tl.sum(exp_L_h * mask_h, axis=1)
        P_h = P_h / row_sum_exp_h
        
        V_tile_T = V_tile.T
        dP_h = dot_128x128(dO_tile, V_tile_T)
        
        O_sum_exp_h = (dO_tile * O_tile).sum(axis=1)[:, None]
        
        D_S_h = P_h * (dP_h - O_sum_exp_h)
        
        dQ_acc = add_gemm_128x128x128(dQ_acc, dot_128x128(D_S_h, K_tile) / math.sqrt(d))
        
        D_S_h_T = D_S_h.T
        dK_acc = add_gemm_128x128x128(dK_acc, trans_a_gemm(D_S_h_T, Q_tile) / math.sqrt(d))
        
        ptr_P = P_bh_ptr + bh * S * S + (r_base//128)*128*128 + (c_base_inner//128)*128 + (r_base%128)*128
        write_float16_block(ptr_P, P_h)
    
    base = P_bh_ptr + bh * S * S
    for r_base_outer in range(c_base, S, 128):
        ptr_P = base + (r_base_outer//128)*128*128 + (c_base//128)*128 + (r_base_outer%128)*128
        P_block = read_float16_block(ptr_P, 128, 128)
        ptr_dO = dO_bh + r_base_outer * 128 * 128
        dO_block = read_bf16_block(ptr_dO, 128, 128)
        
        P_block_f32 = f16_to_f32(P_block)
        dO_block_f32 = f16_to_f32(dO_block)
        
        dV_acc = add_gemm_128x128x128(dV_acc, transpose_gemm(P_block_f32, dO_block_f32))
    
    dQ_bh = dQ[bh]
    dK_bh = dK[bh]
    dV_bh = dV[bh]
    
    store_bf16_block(dQ_bh, r_base + r_idx[:, None], d_idx, dQ_acc, ((r_base + r_idx[:, None]) < S))
    store_bf16_block(dK_bh, c_base + r_idx[:, None], d_idx, dK_acc, ((c_base + r_idx[:, None]) < S))
    store_bf16_block(dV_bh, c_base + r_idx[:, None], d_idx, dV_acc, ((c_base + r_idx[:, None]) < S))