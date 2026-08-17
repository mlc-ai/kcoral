import math

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


BLOCK = 64


@triton.jit
def store_float_array(base_ptr, value, b_h, seq_len, start_row, start_col):
    row_off = tl.arange(0, BLOCK)
    col_off = tl.arange(0, BLOCK)
    valid = ((start_row + row_off[:, None]) < seq_len) & ((start_col + col_off[None, :]) < 128)
    idx = ((b_h * seq_len + start_row + row_off[:, None]) * 128) + (start_col + col_off[None, :])
    tl.store(base_ptr + idx, value.to(tl.bfloat16), mask=valid)


@triton.jit
def _bwd_dQ(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dQ_desc,
    seq_len, scale: tl.constexpr,
):
    pid = tl.program_id(0)
    b_h = tl.program_id(1)
    
    Q0 = Q_desc.load([b_h * seq_len + pid * BLOCK, 0])
    Q1 = Q_desc.load([b_h * seq_len + pid * BLOCK, 64])
    dO0 = dO_desc.load([b_h * seq_len + pid * BLOCK, 0])
    dO1 = dO_desc.load([b_h * seq_len + pid * BLOCK, 64])
    O0 = O_desc.load([b_h * seq_len + pid * BLOCK, 0])
    O1 = O_desc.load([b_h * seq_len + pid * BLOCK, 64])
    
    d_val0 = tl.sum(O0 * dO0, axis=1)
    d_val1 = tl.sum(O1 * dO1, axis=1)
    d_val = d_val0 + d_val1
    
    dQ0_acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    dQ1_acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)

    num_blocks = (seq_len + BLOCK - 1) // BLOCK
    
    row_off = tl.arange(0, BLOCK)
    col_off = tl.arange(0, BLOCK)

    for k_step in range(num_blocks):
        K0 = K_desc.load([b_h * seq_len + k_step * BLOCK, 0])
        K1 = K_desc.load([b_h * seq_len + k_step * BLOCK, 64])
        V0 = V_desc.load([b_h * seq_len + k_step * BLOCK, 0])
        V1 = V_desc.load([b_h * seq_len + k_step * BLOCK, 64])
        
        S_acc = tl.dot(Q0, K0.T, input_precision="ieee")
        S_acc = tl.dot(Q1, K1.T, acc=S_acc, input_precision="ieee")
        
        dP_acc = tl.dot(dO0, V0.T, input_precision="ieee")
        dP_acc = tl.dot(dO1, V1.T, acc=dP_acc, input_precision="ieee")

        k_start_row = k_step * BLOCK
        valid = (pid * BLOCK + row_off[:, None] < seq_len) & (k_start_row + col_off[None, :] < seq_len)
        S_acc = tl.where(valid, S_acc, 0.0)
        dP_acc = tl.where(valid, dP_acc, 0.0)

        l_idx = b_h * seq_len + k_step * BLOCK + row_off
        l_val = tl.load(L + l_idx, mask=(k_step * BLOCK + row_off < seq_len), other=0.0)
        
        p = tl.exp(S_acc * scale - l_val[:, None])
        p = tl.where(valid, p, 0.0)

        dS = p * (dP_acc - d_val[:, None]) * scale
        
        dQ0_acc += tl.dot(dS, K0, acc=dQ0_acc, input_precision="ieee")
        dQ1_acc += tl.dot(dS, K1, acc=dQ1_acc, input_precision="ieee")

    store_float_array(dQ_desc.base_ptr, dQ0_acc, b_h, seq_len, pid * BLOCK, 0)
    store_float_array(dQ_desc.base_ptr, dQ1_acc, b_h, seq_len, pid * BLOCK, 64)


@triton.jit
def _bwd_dK_dV(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dK_desc, dV_desc,
    seq_len, scale: tl.constexpr,
):
    pid = tl.program_id(0)
    b_h = tl.program_id(1)
    
    K0 = K_desc.load([b_h * seq_len + pid * BLOCK, 0])
    K1 = K_desc.load([b_h * seq_len + pid * BLOCK, 64])
    V0 = V_desc.load([b_h * seq_len + pid * BLOCK, 0])
    V1 = V_desc.load([b_h * seq_len + pid * BLOCK, 64])

    dK0_acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    dK1_acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    dV0_acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    dV1_acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)

    num_blocks = (seq_len + BLOCK - 1) // BLOCK
    
    row_off = tl.arange(0, BLOCK)
    col_off = tl.arange(0, BLOCK)

    for q_step in range(num_blocks):
        Q0 = Q_desc.load([b_h * seq_len + q_step * BLOCK, 0])
        Q1 = Q_desc.load([b_h * seq_len + q_step * BLOCK, 64])
        dO0 = dO_desc.load([b_h * seq_len + q_step * BLOCK, 0])
        dO1 = dO_desc.load([b_h * seq_len + q_step * BLOCK, 64])
        O0 = O_desc.load([b_h * seq_len + q_step * BLOCK, 0])
        O1 = O_desc.load([b_h * seq_len + q_step * BLOCK, 64])

        d_val0 = tl.sum(O0 * dO0, axis=1)
        d_val1 = tl.sum(O1 * dO1, axis=1)
        d_val = d_val0 + d_val1

        S_acc = tl.dot(Q0, K0.T, input_precision="ieee")
        S_acc = tl.dot(Q1, K1.T, acc=S_acc, input_precision="ieee")
        
        dP_acc = tl.dot(dO0, V0.T, input_precision="ieee")
        dP_acc = tl.dot(dO1, V1.T, acc=dP_acc, input_precision="ieee")

        q_start_row = q_step * BLOCK
        valid = (q_start_row + row_off[:, None] < seq_len) & (pid * BLOCK + col_off[None, :] < seq_len)
        S_acc = tl.where(valid, S_acc, 0.0)
        dP_acc = tl.where(valid, dP_acc, 0.0)

        l_idx = b_h * seq_len + q_start_row + row_off
        l_val = tl.load(L + l_idx, mask=(q_start_row + row_off < seq_len), other=0.0)
        
        p = tl.exp(S_acc * scale - l_val[:, None])
        p = tl.where(valid, p, 0.0)
        
        dS = p * (dP_acc - d_val[:, None]) * scale

        dS_T = dS.T
        P_T = p.T

        dK0_acc += tl.dot(dS_T, Q0, acc=dK0_acc, input_precision="ieee")
        dK1_acc += tl.dot(dS_T, Q1, acc=dK1_acc, input_precision="ieee")
        
        dV0_acc += tl.dot(P_T, dO0, acc=dV0_acc, input_precision="ieee")
        dV1_acc += tl.dot(P_T, dO1, acc=dV1_acc, input_precision="ieee")

    store_float_array(dK_desc.base_ptr, dK0_acc, b_h, seq_len, pid * BLOCK, 0)
    store_float_array(dK_desc.base_ptr, dK1_acc, b_h, seq_len, pid * BLOCK, 64)
    store_float_array(dV_desc.base_ptr, dV0_acc, b_h, seq_len, pid * BLOCK, 0)
    store_float_array(dV_desc.base_ptr, dV1_acc, b_h, seq_len, pid * BLOCK, 64)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute backward attention gradients dQ, dK, dV into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    device = Q.device
    b, h, seq_len, d = Q.shape
    B, H, D = b, h, d
    
    scale = 1.0 / math.sqrt(128)
    
    Q_c = Q.contiguous()
    K_c = K.contiguous()
    V_c = V.contiguous()
    O_c = O.contiguous()
    dO_c = dO.contiguous()
    
    dQ_c = dQ.contiguous()
    dK_c = dK.contiguous()
    dV_c = dV.contiguous()
    
    if dQ_c.stride(3) != 1:
        temp_dQ = torch.empty_like(dQ_c)
        temp_dQ.copy_(dQ_c)
        dQ_c = temp_dQ
        needs_free_dQ = True
    else:
        needs_free_dQ = False
        
    if dK_c.stride(3) != 1:
        temp_dK = torch.empty_like(dK_c)
        temp_dK.copy_(dK_c)
        dK_c = temp_dK
        needs_free_dK = True
    else:
        needs_free_dK = False
        
    if dV_c.stride(3) != 1:
        temp_dV = torch.empty_like(dV_c)
        temp_dV.copy_(dV_c)
        dV_c = temp_dV
        needs_free_dV = True
    else:
        needs_free_dV = False
    
    Q_2d = Q_c.view(-1, 128)
    K_2d = K_c.view(-1, 128)
    V_2d = V_c.view(-1, 128)
    O_2d = O_c.view(-1, 128)
    dO_2d = dO_c.view(-1, 128)
    
    dQ_2d = dQ_c.view(-1, 128)
    dK_2d = dK_c.view(-1, 128)
    dV_2d = dV_c.view(-1, 128)
    
    q_desc = TensorDescriptor.from_tensor(Q_2d, [BLOCK, BLOCK])
    k_desc = TensorDescriptor.from_tensor(K_2d, [BLOCK, BLOCK])
    v_desc = TensorDescriptor.from_tensor(V_2d, [BLOCK, BLOCK])
    o_desc = TensorDescriptor.from_tensor(O_2d, [BLOCK, BLOCK])
    do_desc = TensorDescriptor.from_tensor(dO_2d, [BLOCK, BLOCK])
    
    dq_desc = TensorDescriptor.from_tensor(dQ_2d, [BLOCK, BLOCK])
    dk_desc = TensorDescriptor.from_tensor(dK_2d, [BLOCK, BLOCK])
    dv_desc = TensorDescriptor.from_tensor(dV_2d, [BLOCK, BLOCK])
    
    grid = (triton.cdiv(seq_len, BLOCK), B * H)
    
    _bwd_dQ[grid](q_desc, k_desc, v_desc, o_desc, do_desc, L, dq_desc, seq_len, scale, num_warps=4, num_stages=2)
    _bwd_dK_dV[grid](q_desc, k_desc, v_desc, o_desc, do_desc, L, dk_desc, dv_desc, seq_len, scale, num_warps=4, num_stages=2)
    
    if needs_free_dQ:
        dQ.copy_(dQ_c)
    if needs_free_dK:
        dK.copy_(dK_c)
    if needs_free_dV:
        dV.copy_(dV_c)