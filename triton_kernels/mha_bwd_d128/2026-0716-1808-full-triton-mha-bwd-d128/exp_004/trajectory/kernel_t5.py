import math

import torch
import triton
import triton.language as tl


BLOCK = 128


@triton.jit
def load_tile(base_ptr, b_h, seq_len, start_row, start_col):
    row_off = tl.arange(0, BLOCK)
    col_off = tl.arange(0, BLOCK)
    base_idx = b_h * seq_len * 128 + start_row * 128 + start_col
    ptr = base_ptr + (base_idx // 128 + row_off[:, None]) * 128 + start_col + col_off[None, :]
    valid = ((start_row + row_off[:, None]) < seq_len) & ((start_col + col_off[None, :]) < 128)
    return tl.load(ptr, mask=valid, other=0.0)


@triton.jit
def store_tile(base_ptr, value, b_h, seq_len, start_row, start_col):
    row_off = tl.arange(0, BLOCK)
    col_off = tl.arange(0, BLOCK)
    base_idx = b_h * seq_len * 128 + start_row * 128 + start_col
    ptr = base_ptr + (base_idx // 128 + row_off[:, None]) * 128 + start_col + col_off[None, :]
    valid = ((start_row + row_off[:, None]) < seq_len) & ((start_col + col_off[None, :]) < 128)
    tl.store(ptr, value.to(tl.bfloat16), mask=valid)


@triton.jit
def _bwd_dQ(
    Q, K, V, O, dO, L, dQ,
    seq_len, scale: tl.constexpr,
):
    pid = tl.program_id(0)
    b_h = tl.program_id(1)
    
    Q = load_tile(Q, b_h, seq_len, pid * BLOCK, 0)
    dO = load_tile(dO, b_h, seq_len, pid * BLOCK, 0)
    O = load_tile(O, b_h, seq_len, pid * BLOCK, 0)
    
    d_val = tl.sum(O * dO, axis=1)
    
    dQ_acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)

    num_blocks = (seq_len + BLOCK - 1) // BLOCK
    
    row_off = tl.arange(0, BLOCK)
    col_off = tl.arange(0, BLOCK)

    for k_step in range(num_blocks):
        K = load_tile(K, b_h, seq_len, k_step * BLOCK, 0)
        V = load_tile(V, b_h, seq_len, k_step * BLOCK, 0)
        
        S_acc = tl.dot(Q, K.T, input_precision="ieee")
        dP_acc = tl.dot(dO, V.T, input_precision="ieee")

        k_start_row = k_step * BLOCK
        valid = (pid * BLOCK + row_off[:, None] < seq_len) & (k_start_row + col_off[None, :] < seq_len)
        S_acc = tl.where(valid, S_acc, 0.0)
        dP_acc = tl.where(valid, dP_acc, 0.0)

        l_idx = b_h * seq_len + pid * BLOCK + row_off
        l_val = tl.load(L + l_idx, mask=(pid * BLOCK + row_off < seq_len), other=0.0)
        
        p = tl.exp(S_acc * scale - l_val[:, None])
        p = tl.where(valid, p, 0.0)

        dS = p * (dP_acc - d_val[:, None]) * scale
        
        dQ_acc += tl.dot(dS, K, acc=dQ_acc, input_precision="ieee")

    store_tile(dQ, dQ_acc, b_h, seq_len, pid * BLOCK, 0)


@triton.jit
def _bwd_dK_dV(
    Q, K, V, O, dO, L, dK, dV,
    seq_len, scale: tl.constexpr,
):
    pid = tl.program_id(0)
    b_h = tl.program_id(1)
    
    K = load_tile(K, b_h, seq_len, pid * BLOCK, 0)
    V = load_tile(V, b_h, seq_len, pid * BLOCK, 0)

    dK_acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    dV_acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)

    num_blocks = (seq_len + BLOCK - 1) // BLOCK
    
    row_off = tl.arange(0, BLOCK)
    col_off = tl.arange(0, BLOCK)

    for q_step in range(num_blocks):
        Q = load_tile(Q, b_h, seq_len, q_step * BLOCK, 0)
        dO = load_tile(dO, b_h, seq_len, q_step * BLOCK, 0)
        O = load_tile(O, b_h, seq_len, q_step * BLOCK, 0)

        d_val = tl.sum(O * dO, axis=1)

        S_acc = tl.dot(Q, K.T, input_precision="ieee")
        dP_acc = tl.dot(dO, V.T, input_precision="ieee")

        q_start_row = q_step * BLOCK
        valid = (q_start_row + row_off[:, None] < seq_len) & (pid * BLOCK + col_off[None, :] < seq_len)
        S_acc = tl.where(valid, S_acc, 0.0)
        dP_acc = tl.where(valid, dP_acc, 0.0)

        l_idx = b_h * seq_len + q_start_row + row_off
        l_val = tl.load(L + l_idx, mask=(q_start_row + row_off < seq_len), other=0.0)
        
        p = tl.exp(S_acc * scale - l_val[:, None])
        p = tl.where(valid, p, 0.0)
        
        dS = p * (dP_acc - d_val[:, None]) * scale

        dK_acc += tl.dot(dS.T, Q, acc=dK_acc, input_precision="ieee")
        dV_acc += tl.dot(p.T, dO, acc=dV_acc, input_precision="ieee")

    store_tile(dK, dK_acc, b_h, seq_len, pid * BLOCK, 0)
    store_tile(dV, dV_acc, b_h, seq_len, pid * BLOCK, 0)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute backward attention gradients dQ, dK, dV into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    device = Q.device
    b, h, seq_len, d = Q.shape
    B, H, D = b, h, d
    
    scale = 1.0 / math.sqrt(128)
    
    grid = (triton.cdiv(seq_len, BLOCK), B * H)
    
    _bwd_dQ[grid](Q, K, V, O, dO, L, dQ, seq_len, scale, num_warps=8, num_stages=3, maxnreg=255)
    _bwd_dK_dV[grid](Q, K, V, O, dO, L, dK, dV, seq_len, scale, num_warps=8, num_stages=3, maxnreg=255)