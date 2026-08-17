import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _dQ_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr, o_ptr, l_ptr, dq_ptr,
    B, H, S, tau,
    BLOCK_S: tl.constexpr,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    i = tl.program_id(0)
    offset_i = i * BLOCK_S
    
    q_desc = tl.make_tensor_descriptor(
        q_ptr, shape=[B * H * S, 128], strides=[128, 1],
        block_shape=[BLOCK_S, 128], padding_option="zero",
    )
    k_desc = tl.make_tensor_descriptor(
        k_ptr, shape=[B * H * S, 128], strides=[128, 1],
        block_shape=[BLOCK_S, 128], padding_option="zero",
    )
    v_desc = tl.make_tensor_descriptor(
        v_ptr, shape=[B * H * S, 128], strides=[128, 1],
        block_shape=[BLOCK_S, 128], padding_option="zero",
    )
    do_desc = tl.make_tensor_descriptor(
        do_ptr, shape=[B * H * S, 128], strides=[128, 1],
        block_shape=[BLOCK_S, 128], padding_option="zero",
    )
    o_desc = tl.make_tensor_descriptor(
        o_ptr, shape=[B * H * S, 128], strides=[128, 1],
        block_shape=[BLOCK_S, 128], padding_option="zero",
    )
    dq_desc = tl.make_tensor_descriptor(
        dq_ptr, shape=[B * H * S, 128], strides=[128, 1],
        block_shape=[BLOCK_S, 128], padding_option="zero",
    )
    
    row_offset_i = (b * H + h) * S + offset_i
    
    Q = q_desc.load([row_offset_i, 0])
    dO = do_desc.load([row_offset_i, 0])
    O = o_desc.load([row_offset_i, 0])
    
    D = tl.sum(dO * O, axis=1)
    
    seq_idx_q = offset_i + tl.arange(0, BLOCK_S)
    l_offset_i = (b * H + h) * S + offset_i
    L = tl.load(l_ptr + l_offset_i + tl.arange(0, BLOCK_S), mask=(seq_idx_q < S), other=0.0)
    
    acc_dQ = tl.zeros((BLOCK_S, 128), tl.float32)
    
    num_k_tiles = tl.cdiv(S, BLOCK_S)
    
    for j in tl.range(num_k_tiles, num_stages=2):
        offset_j = j * BLOCK_S
        row_offset_j = (b * H + h) * S + offset_j
        
        K = k_desc.load([row_offset_j, 0])
        V = v_desc.load([row_offset_j, 0])
        
        seq_idx_kv = offset_j + tl.arange(0, BLOCK_S)
        mask_ij = (seq_idx_q[:, None] < S) & (seq_idx_kv[None, :] < S)
        
        S_mat = tl.dot(Q, K.T)
        dP = tl.dot(dO, V.T)
        
        P = tl.exp(S_mat * tau - L[:, None])
        P = P * mask_ij
        
        dS = P * (dP - D[:, None]) * tau
        dS = dS * mask_ij
        
        dS_bf16 = dS.to(tl.bfloat16)
        acc_dQ = tl.dot(dS_bf16, K, acc_dQ)
    
    dq_desc.store([row_offset_i, 0], acc_dQ.to(tl.bfloat16))


@triton.jit
def _dK_dV_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr, o_ptr, l_ptr, dk_ptr, dv_ptr,
    B, H, S, tau,
    BLOCK_S: tl.constexpr,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    j = tl.program_id(0)
    offset_j = j * BLOCK_S
    
    q_desc = tl.make_tensor_descriptor(
        q_ptr, shape=[B * H * S, 128], strides=[128, 1],
        block_shape=[BLOCK_S, 128], padding_option="zero",
    )
    k_desc = tl.make_tensor_descriptor(
        k_ptr, shape=[B * H * S, 128], strides=[128, 1],
        block_shape=[BLOCK_S, 128], padding_option="zero",
    )
    v_desc = tl.make_tensor_descriptor(
        v_ptr, shape=[B * H * S, 128], strides=[128, 1],
        block_shape=[BLOCK_S, 128], padding_option="zero",
    )
    do_desc = tl.make_tensor_descriptor(
        do_ptr, shape=[B * H * S, 128], strides=[128, 1],
        block_shape=[BLOCK_S, 128], padding_option="zero",
    )
    o_desc = tl.make_tensor_descriptor(
        o_ptr, shape=[B * H * S, 128], strides=[128, 1],
        block_shape=[BLOCK_S, 128], padding_option="zero",
    )
    dk_desc = tl.make_tensor_descriptor(
        dk_ptr, shape=[B * H * S, 128], strides=[128, 1],
        block_shape=[BLOCK_S, 128], padding_option="zero",
    )
    dv_desc = tl.make_tensor_descriptor(
        dv_ptr, shape=[B * H * S, 128], strides=[128, 1],
        block_shape=[BLOCK_S, 128], padding_option="zero",
    )
    
    row_offset_j = (b * H + h) * S + offset_j
    
    K = k_desc.load([row_offset_j, 0])
    V = v_desc.load([row_offset_j, 0])
    
    acc_dK = tl.zeros((BLOCK_S, 128), tl.float32)
    acc_dV = tl.zeros((BLOCK_S, 128), tl.float32)
    
    seq_idx_kv = offset_j + tl.arange(0, BLOCK_S)
    
    num_q_tiles = tl.cdiv(S, BLOCK_S)
    
    for i in tl.range(num_q_tiles, num_stages=2):
        offset_i = i * BLOCK_S
        row_offset_i = (b * H + h) * S + offset_i
        
        Q = q_desc.load([row_offset_i, 0])
        dO = do_desc.load([row_offset_i, 0])
        O = o_desc.load([row_offset_i, 0])
        
        D = tl.sum(dO * O, axis=1)
        
        seq_idx_q = offset_i + tl.arange(0, BLOCK_S)
        l_offset_i = (b * H + h) * S + offset_i
        L = tl.load(l_ptr + l_offset_i + tl.arange(0, BLOCK_S), mask=(seq_idx_q < S), other=0.0)
        
        mask_ij = (seq_idx_q[:, None] < S) & (seq_idx_kv[None, :] < S)
        
        S_mat = tl.dot(Q, K.T)
        dP = tl.dot(dO, V.T)
        
        P = tl.exp(S_mat * tau - L[:, None])
        P = P * mask_ij
        
        dS = P * (dP - D[:, None]) * tau
        dS = dS * mask_ij
        
        P_bf16 = P.to(tl.bfloat16)
        dS_bf16 = dS.to(tl.bfloat16)
        
        acc_dV = tl.dot(P_bf16.T, dO, acc_dV)
        acc_dK = tl.dot(dS_bf16.T, Q, acc_dK)
    
    dk_desc.store([row_offset_j, 0], acc_dK.to(tl.bfloat16))
    dv_desc.store([row_offset_j, 0], acc_dV.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    tau = 1.0 / (d ** 0.5)
    BLOCK_S = 128
    
    grid = (triton.cdiv(S, BLOCK_S), H, B)
    
    _dQ_kernel[grid](
        Q, K, V, dO, O, L, dQ,
        B, H, S, tau,
        BLOCK_S=BLOCK_S,
        num_warps=4,
        num_stages=2,
    )
    
    _dK_dV_kernel[grid](
        Q, K, V, dO, O, L, dK, dV,
        B, H, S, tau,
        BLOCK_S=BLOCK_S,
        num_warps=4,
        num_stages=2,
    )