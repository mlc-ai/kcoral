import math

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


BLOCK = 128


@triton.jit
def load_tile(desc, b_h, start_row, start_col):
    return desc.load([b_h * seq_len + start_row + row_off[:, None], start_col + col_off[None, :]])


@triton.jit
def store_float_array(base_ptr, value, b_h, seq_len, start_row, start_col):
    row_off = tl.arange(0, BLOCK)
    col_off = tl.arange(0, BLOCK)
    valid = (start_row + row_off[:, None] < seq_len) & (start_col + col_off[None, :] < 128)
    idx = ((b_h * seq_len + start_row + row_off[:, None]) * 128) + (start_col + col_off[None, :])
    tl.store(base_ptr + idx, value, mask=valid)


@triton.jit
def _bwd_dQ(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L, D, dQ_desc,
    seq_len, scale: tl.constexpr,
):
    pid = tl.program_id(0)
    b_h = tl.program_id(1)
    row_off = tl.arange(0, BLOCK)
    col_off = tl.arange(0, BLOCK)

    Q = load_tile(Q_desc, b_h, pid * BLOCK, 0)
    dO = load_tile(dO_desc, b_h, pid * BLOCK, 0)
    O = load_tile(O_desc, b_h, pid * BLOCK, 0)

    dQ_acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)

    num_blocks = (seq_len + BLOCK - 1) // BLOCK

    for k_step in range(num_blocks):
        K = load_tile(K_desc, b_h, k_step * BLOCK, 0)
        V = load_tile(V_desc, b_h, k_step * BLOCK, 0)

        K_T = K.T
        
        S_acc = tl.dot(Q, K_T, input_precision="ieee")
        
        dP_acc = tl.dot(dO, V.T, input_precision="ieee")

        k_start_row = k_step * BLOCK
        valid = (row_off[:, None] < seq_len) & (col_off[None, :] < seq_len)
        S_acc = tl.where(valid, S_acc, 0.0)
        dP_acc = tl.where(valid, dP_acc, 0.0)

        S_scaled = S_acc * scale

        l_vals = tl.load(L + b_h * seq_len + pid * BLOCK + row_off, mask=(pid * BLOCK + row_off < seq_len), other=0.0)

        p = tl.exp(S_scaled - l_vals)
        p = tl.where(valid, p, 0.0)

        dS = p * (dP_acc - O * dO) * scale
        
        dQ_acc += tl.dot(dS, K, acc=dQ_acc, input_precision="ieee")

    store_float_array(dQ, dQ_acc.to(tl.bfloat16), b_h, seq_len, pid * BLOCK, 0)


@triton.jit
def _bwd_dK_dV(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L, D, dK_desc, dV_desc,
    seq_len, scale: tl.constexpr,
):
    pid = tl.program_id(0)
    b_h = tl.program_id(1)
    row_off = tl.arange(0, BLOCK)
    col_off = tl.arange(0, BLOCK)

    K = load_tile(K_desc, b_h, pid * BLOCK, 0)
    V = load_tile(V_desc, b_h, pid * BLOCK, 0)

    K_T = K.T
    
    dK0_acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    dV0_acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)

    num_blocks = (seq_len + BLOCK - 1) // BLOCK

    for k_step in range(num_blocks):
        Q = load_tile(Q_desc, b_h, k_step * BLOCK, 0)
        dO = load_tile(dO_desc, b_h, k_step * BLOCK, 0)
        O = load_tile(O_desc, b_h, k_step * BLOCK, 0)

        S_acc = tl.dot(Q, K_T, input_precision="ieee")
        dP_acc = tl.dot(dO, V.T, input_precision="ieee")

        q_start_row = k_step * BLOCK
        valid = (row_off[:, None] < seq_len) & (col_off[None, :] < seq_len)
        S_acc = tl.where(valid, S_acc, 0.0)
        dP_acc = tl.where(valid, dP_acc, 0.0)

        S_scaled = S_acc * scale

        l_vals = tl.load(L + b_h * seq_len + q_start_row + row_off, mask=(q_start_row + row_off < seq_len), other=0.0)

        p = tl.exp(S_scaled - l_vals)
        p = tl.where(valid, p, 0.0)

        D_local = (Q * dO).T
        d_vals = tl.load(D + q_start_row + col_off, mask=(q_start_row + col_off < seq_len), other=0.0)
        D_local = D_local * d_vals[:, None]
        
        dS = p * (dP_acc - D_local) * scale

        dS_T = dS.T
        P_T = p.T

        dK0_acc += tl.dot(dS_T, Q, acc=dK0_acc, input_precision="ieee")
        dV0_acc += tl.dot(P_T, dO, acc=dV0_acc, input_precision="ieee")

    store_float_array(dK, dK0_acc.to(tl.bfloat16), b_h, seq_len, pid * BLOCK, 0)
    store_float_array(dV, dV0_acc.to(tl.bfloat16), b_h, seq_len, pid * BLOCK, 0)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute backward attention gradients dQ, dK, dV into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    device = Q.device
    b, h, seq_len, d = Q.shape
    B, H, D = b, h, d
    
    scale = 1.0 / math.sqrt(128)
    
    # Ensure tensors are contiguous
    Q_c = Q.contiguous()
    K_c = K.contiguous()
    V_c = V.contiguous()
    O_c = O.contiguous()
    dO_c = dO.contiguous()
    
    dQ_c = dQ.contiguous()
    dK_c = dK.contiguous()
    dV_c = dV.contiguous()
    
    # Create 2D descriptors treating the tensor as [B*H*S, d]
    q_desc = TensorDescriptor.from_tensor(Q_c, shape=[B*H*seq_len, D], strides=[D, 1], block_shape=[BLOCK, BLOCK])
    k_desc = TensorDescriptor.from_tensor(K_c, shape=[B*H*seq_len, D], strides=[D, 1], block_shape=[BLOCK, BLOCK])
    v_desc = TensorDescriptor.from_tensor(V_c, shape=[B*H*seq_len, D], strides=[D, 1], block_shape=[BLOCK, BLOCK])
    o_desc = TensorDescriptor.from_tensor(O_c, shape=[B*H*seq_len, D], strides=[D, 1], block_shape=[BLOCK, BLOCK])
    do_desc = TensorDescriptor.from_tensor(dO_c, shape=[B*H*seq_len, D], strides=[D, 1], block_shape=[BLOCK, BLOCK])
    
    dq_desc = TensorDescriptor.from_tensor(dQ_c, shape=[B*H*seq_len, D], strides=[D, 1], block_shape=[BLOCK, BLOCK])
    dk_desc = TensorDescriptor.from_tensor(dK_c, shape=[B*H*seq_len, D], strides=[D, 1], block_shape=[BLOCK, BLOCK])
    dv_desc = TensorDescriptor.from_tensor(dV_c, shape=[B*H*seq_len, D], strides=[D, 1], block_shape=[BLOCK, BLOCK])
    
    workspace = torch.empty((B * H, seq_len), dtype=torch.float32, device=device)
    D = workspace
    
    grid_D = (B * H, seq_len)
    _compute_D_kernel[grid_D](O, dO, D, seq_len, num_warps=2)
    
    grid = (triton.cdiv(seq_len, BLOCK), B * H)
    
    _bwd_dQ[grid](q_desc, k_desc, v_desc, o_desc, do_desc, L, D, dq_desc, seq_len, scale, num_warps=4, num_stages=2)
    _bwd_dK_dV[grid](q_desc, k_desc, v_desc, o_desc, do_desc, L, D, dk_desc, dv_desc, seq_len, scale, num_warps=4, num_stages=2)