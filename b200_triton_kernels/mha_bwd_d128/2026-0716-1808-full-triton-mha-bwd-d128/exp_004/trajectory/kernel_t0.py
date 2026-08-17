import math

import torch
import triton
import triton.language as tl


BLOCK = 64


@triton.jit
def load_float_array(base_ptr, b_h, seq_len, start_row, start_col):
    row_off = tl.arange(0, BLOCK)
    col_off = tl.arange(0, BLOCK)
    base_idx = b_h * seq_len * 128 + start_row * 128 + start_col
    ptr = base_ptr + (base_idx // 128 + row_off[:, None]) * 128 + start_col + col_off[None, :]
    valid = (start_row + row_off[:, None] < seq_len) & (start_col + col_off[None, :] < 128)
    return tl.load(ptr, mask=valid, other=0.0f)


@triton.jit
def store_float_array(base_ptr, value, b_h, seq_len, start_row, start_col):
    row_off = tl.arange(0, BLOCK)
    col_off = tl.arange(0, BLOCK)
    valid = (start_row + row_off[:, None] < seq_len) & (start_col + col_off[None, :] < 128)
    idx = ((b_h * seq_len + start_row + row_off[:, None]) * 128) + (start_col + col_off[None, :])
    tl.store(base_ptr + idx, value, mask=valid)


@triton.jit
def load_L(L, b_h, seq_len, start_row):
    row_off = tl.arange(0, BLOCK)
    l_ptr = L + b_h * seq_len + start_row + row_off
    l_vals = tl.load(l_ptr)
    return l_vals


@triton.jit
def _bwd_dQ(
    Q, K, V, O, dO, L, dQ,
    seq_len, scale: tl.constexpr,
):
    pid = tl.program_id(0)
    b_h = tl.program_id(1)
    row_off = tl.arange(0, BLOCK)
    col_off = tl.arange(0, BLOCK)

    dQ0_acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    dQ1_acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)

    num_blocks = (seq_len + BLOCK - 1) // BLOCK

    for k_step in range(num_blocks):
        Q0 = load_float_array(Q, b_h, seq_len, pid * BLOCK, 0)
        Q1 = load_float_array(Q, b_h, seq_len, pid * BLOCK, 64)
        dO0 = load_float_array(dO, b_h, seq_len, pid * BLOCK, 0)
        dO1 = load_float_array(dO, b_h, seq_len, pid * BLOCK, 64)
        O0 = load_float_array(O, b_h, seq_len, pid * BLOCK, 0)
        O1 = load_float_array(O, b_h, seq_len, pid * BLOCK, 64)

        K0 = load_float_array(K, b_h, seq_len, k_step * BLOCK, 0)
        K1 = load_float_array(K, b_h, seq_len, k_step * BLOCK, 64)
        V0 = load_float_array(V, b_h, seq_len, k_step * BLOCK, 0)
        V1 = load_float_array(V, b_h, seq_len, k_step * BLOCK, 64)

        S_acc = tl.dot(Q0, K0.T, input_precision="ieee")
        S_acc = tl.dot(Q1, K1.T, acc=S_acc, input_precision="ieee")

        dP_acc = tl.dot(dO0, V0.T, input_precision="ieee")
        dP_acc = tl.dot(dO1, V1.T, acc=dP_acc, input_precision="ieee")

        start_row = pid * BLOCK
        start_col = k_step * BLOCK
        valid = (start_row + row_off[:, None] < seq_len) & (start_col + col_off[None, :] < seq_len)

        S_acc = tl.where(valid, S_acc, 0.0f)
        dP_acc = tl.where(valid, dP_acc, 0.0f)

        S_scaled = S_acc * scale

        l_idx = b_h * seq_len + k_step * BLOCK + row_off
        l_vals = tl.load(L + l_idx)

        p = tl.exp(S_scaled - l_vals)
        p = tl.where(valid, p, 0.0f)

        dS = p * (dP_acc - d_idx[row_off, 0]) * scale

        dQ0_acc += tl.dot(dS, K0, acc=dQ0_acc, input_precision="ieee")
        dQ1_acc += tl.dot(dS, K1, acc=dQ1_acc, input_precision="ieee")

    store_float_array(dQ, dQ0_acc.to(tl.bfloat16), b_h, seq_len, k_step * BLOCK, 0)
    store_float_array(dQ, dQ1_acc.to(tl.bfloat16), b_h, seq_len, k_step * BLOCK, 64)


@triton.jit
def _bwd_dK_dV(
    Q, K, V, O, dO, L, dK, dV,
    seq_len, scale: tl.constexpr,
):
    pid = tl.program_id(0)
    b_h = tl.program_id(1)
    row_off = tl.arange(0, BLOCK)
    col_off = tl.arange(0, BLOCK)

    dK0_acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    dK1_acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    dV0_acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    dV1_acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)

    num_blocks = (seq_len + BLOCK - 1) // BLOCK

    for k_step in range(num_blocks):
        Q0 = load_float_array(Q, b_h, seq_len, k_step * BLOCK, 0)
        Q1 = load_float_array(Q, b_h, seq_len, k_step * BLOCK, 64)
        dO0 = load_float_array(dO, b_h, seq_len, k_step * BLOCK, 0)
        dO1 = load_float_array(dO, b_h, seq_len, k_step * BLOCK, 64)
        O0 = load_float_array(O, b_h, seq_len, k_step * BLOCK, 0)
        O1 = load_float_array(O, b_h, seq_len, k_step * BLOCK, 64)

        K0 = load_float_array(K, b_h, seq_len, pid * BLOCK, 0)
        K1 = load_float_array(K, b_h, seq_len, pid * BLOCK, 64)
        V0 = load_float_array(V, b_h, seq_len, pid * BLOCK, 0)
        V1 = load_float_array(V, b_h, seq_len, pid * BLOCK, 64)

        S_acc = tl.dot(Q0, K0.T, input_precision="ieee")
        S_acc = tl.dot(Q1, K1.T, acc=S_acc, input_precision="ieee")

        dP_acc = tl.dot(dO0, V0.T, input_precision="ieee")
        dP_acc = tl.dot(dO1, V1.T, acc=dP_acc, input_precision="ieee")

        start_row = k_step * BLOCK
        start_col = pid * BLOCK
        valid = (start_row + row_off[:, None] < seq_len) & (start_col + col_off[None, :] < seq_len)

        S_acc = tl.where(valid, S_acc, 0.0f)
        dP_acc = tl.where(valid, dP_acc, 0.0f)

        S_scaled = S_acc * scale

        l_idx = b_h * seq_len + k_step * BLOCK + row_off
        l_vals = tl.load(L + l_idx)

        p = tl.exp(S_scaled - l_vals)
        p = tl.where(valid, p, 0.0f)

        dS = p * (dP_acc - d_idx[row_off, 0]) * scale

        dS_T = dS.T
        P_T = p.T

        dK0_acc += tl.dot(dS_T, Q0, acc=dK0_acc, input_precision="ieee")
        dK1_acc += tl.dot(dS_T, Q1, acc=dK1_acc, input_precision="ieee")
        
        dV0_acc += tl.dot(P_T, dO0, acc=dV0_acc, input_precision="ieee")
        dV1_acc += tl.dot(P_T, dO1, acc=dV1_acc, input_precision="ieee")

    store_float_array(dK, dK0_acc.to(tl.bfloat16), b_h, seq_len, pid * BLOCK, 0)
    store_float_array(dK, dK1_acc.to(tl.bfloat16), b_h, seq_len, pid * BLOCK, 64)
    store_float_array(dV, dV0_acc.to(tl.bfloat16), b_h, seq_len, pid * BLOCK, 0)
    store_float_array(dV, dV1_acc.to(tl.bfloat16), b_h, seq_len, pid * BLOCK, 64)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute backward attention gradients dQ, dK, dV into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    device = Q.device
    b, h, seq_len, d = Q.shape
    B, H, D = b, h, d
    
    scale = 1.0 / math.sqrt(128)
    
    workspace = torch.empty((B * H, seq_len), dtype=torch.float32, device=device)
    D = workspace
    
    grid_D = (B * H, triton.cdiv(seq_len, 64))
    _compute_D_kernel[grid_D](O, dO, D, seq_len, num_warps=2, num_ctas=1)
    
    grid = (triton.cdiv(seq_len, 64), B * H)
    
    _bwd_dQ[grid](Q, K, V, O, dO, L, dQ, seq_len, scale, num_warps=4, num_stages=2, num_ctas=1)
    _bwd_dK_dV[grid](Q, K, V, O, dO, L, dK, dV, seq_len, scale, num_warps=4, num_stages=2, num_ctas=1)