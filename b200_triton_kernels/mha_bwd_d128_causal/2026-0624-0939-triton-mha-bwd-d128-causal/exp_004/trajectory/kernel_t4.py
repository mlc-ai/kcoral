import torch
import triton
import triton.language as tl
import math


def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)


@triton.jit
def _bwd_dKdV_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    S, H, d, tau,
    stride_b, stride_h, stride_s, stride_d,
    BLOCK_N: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    j = tl.program_id(2)
    
    T_r = tl.cdiv(S, BLOCK_N)
    if j >= T_r:
        return
    
    dK_acc = tl.zeros((BLOCK_N, 128), tl.float32)
    dV_acc = tl.zeros((BLOCK_N, 128), tl.float32)
    
    batch_off_b = b * stride_b + h * stride_h
    
    Q_desc = tl.make_tensor_descriptor(
        Q_ptr + batch_off_b, shape=[S, d], strides=[stride_s, stride_d],
        block_shape=[BLOCK_N, 128], padding_option="zero")
    
    K_desc = tl.make_tensor_descriptor(
        K_ptr + batch_off_b, shape=[S, d], strides=[stride_s, stride_d],
        block_shape=[BLOCK_N, 128], padding_option="zero")
    
    V_desc = tl.make_tensor_descriptor(
        V_ptr + batch_off_b, shape=[S, d], strides=[stride_s, stride_d],
        block_shape=[BLOCK_N, 128], padding_option="zero")
    
    O_desc = tl.make_tensor_descriptor(
        O_ptr + batch_off_b, shape=[S, d], strides=[stride_s, stride_d],
        block_shape=[BLOCK_N, 128], padding_option="zero")
    
    dO_desc = tl.make_tensor_descriptor(
        dO_ptr + batch_off_b, shape=[S, d], strides=[stride_s, stride_d],
        block_shape=[BLOCK_N, 128], padding_option="zero")
    
    dK_desc = tl.make_tensor_descriptor(
        dK_ptr + batch_off_b, shape=[S, d], strides=[stride_s, stride_d],
        block_shape=[BLOCK_N, 128])
    
    dV_desc = tl.make_tensor_descriptor(
        dV_ptr + batch_off_b, shape=[S, d], strides=[stride_s, stride_d],
        block_shape=[BLOCK_N, 128])
    
    rows_k = tl.arange(0, BLOCK_N)
    k_abs = j * BLOCK_N + rows_k
    
    K_j = K_desc.load([j * BLOCK_N, 0])
    V_j = V_desc.load([j * BLOCK_N, 0])
    
    l_offset = b * H * S + h * S
    
    for i in range(j, T_r):
        Q_i = Q_desc.load([i * BLOCK_N, 0])
        dO_i = dO_desc.load([i * BLOCK_N, 0])
        O_i = O_desc.load([i * BLOCK_N, 0])
        
        D_i = tl.sum(O_i * dO_i, axis=1)
        
        q_abs = i * BLOCK_N + rows_k
        L_i = tl.load(L_ptr + l_offset + q_abs, mask=q_abs < S, other=0.0)
        
        S_ij = tl.dot(Q_i, K_j.T) * tau
        
        if i == j:
            rows_q = tl.arange(0, BLOCK_N)
            q_abs_q = i * BLOCK_N + rows_q
            causal_mask = k_abs[None, :] <= q_abs_q[:, None]
            S_ij = tl.where(causal_mask, S_ij, -float('inf'))
        
        P_ij = tl.exp(S_ij - L_i[:, None])
        
        if i == j:
            rows_q = tl.arange(0, BLOCK_N)
            q_abs_q = i * BLOCK_N + rows_q
            causal_mask = k_abs[None, :] <= q_abs_q[:, None]
            P_ij = tl.where(causal_mask, P_ij, 0.0)
        
        dP_ij = tl.dot(dO_i, V_j.T)
        dS_ij = P_ij * (dP_ij - D_i[:, None]) * tau
        
        if i == j:
            rows_q = tl.arange(0, BLOCK_N)
            q_abs_q = i * BLOCK_N + rows_q
            causal_mask = k_abs[None, :] <= q_abs_q[:, None]
            dS_ij = tl.where(causal_mask, dS_ij, 0.0)
        
        dV_acc = tl.dot(P_ij.T, dO_i, acc=dV_acc)
        dK_acc = tl.dot(dS_ij.T, Q_i, acc=dK_acc)
    
    dK_desc.store([j * BLOCK_N, 0], dK_acc.to(tl.bfloat16))
    dV_desc.store([j * BLOCK_N, 0], dV_acc.to(tl.bfloat16))


@triton.jit
def _bwd_dQ_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
    S, H, d, tau,
    stride_b, stride_h, stride_s, stride_d,
    BLOCK_N: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    i = tl.program_id(2)
    
    T_c = tl.cdiv(S, BLOCK_N)
    if i >= T_c:
        return
    
    dQ_acc = tl.zeros((BLOCK_N, 128), tl.float32)
    
    batch_off_b = b * stride_b + h * stride_h
    
    Q_desc = tl.make_tensor_descriptor(
        Q_ptr + batch_off_b, shape=[S, d], strides=[stride_s, stride_d],
        block_shape=[BLOCK_N, 128], padding_option="zero")
    
    K_desc = tl.make_tensor_descriptor(
        K_ptr + batch_off_b, shape=[S, d], strides=[stride_s, stride_d],
        block_shape=[BLOCK_N, 128], padding_option="zero")
    
    V_desc = tl.make_tensor_descriptor(
        V_ptr + batch_off_b, shape=[S, d], strides=[stride_s, stride_d],
        block_shape=[BLOCK_N, 128], padding_option="zero")
    
    O_desc = tl.make_tensor_descriptor(
        O_ptr + batch_off_b, shape=[S, d], strides=[stride_s, stride_d],
        block_shape=[BLOCK_N, 128], padding_option="zero")
    
    dO_desc = tl.make_tensor_descriptor(
        dO_ptr + batch_off_b, shape=[S, d], strides=[stride_s, stride_d],
        block_shape=[BLOCK_N, 128], padding_option="zero")
    
    dQ_desc = tl.make_tensor_descriptor(
        dQ_ptr + batch_off_b, shape=[S, d], strides=[stride_s, stride_d],
        block_shape=[BLOCK_N, 128])
    
    rows_q = tl.arange(0, BLOCK_N)
    q_abs = i * BLOCK_N + rows_q
    
    Q_i = Q_desc.load([i * BLOCK_N, 0])
    dO_i = dO_desc.load([i * BLOCK_N, 0])
    O_i = O_desc.load([i * BLOCK_N, 0])
    
    l_offset = b * H * S + h * S
    L_i = tl.load(L_ptr + l_offset + q_abs, mask=q_abs < S, other=0.0)
    
    D_i = tl.sum(O_i * dO_i, axis=1)
    
    for j in range(0, i + 1):
        K_j = K_desc.load([j * BLOCK_N, 0])
        V_j = V_desc.load([j * BLOCK_N, 0])
        
        S_ij = tl.dot(Q_i, K_j.T) * tau
        
        if i == j:
            rows_k = tl.arange(0, BLOCK_N)
            k_abs = j * BLOCK_N + rows_k
            causal_mask = k_abs[None, :] <= q_abs[:, None]
            S_ij = tl.where(causal_mask, S_ij, -float('inf'))
        
        P_ij = tl.exp(S_ij - L_i[:, None])
        
        if i == j:
            rows_k = tl.arange(0, BLOCK_N)
            k_abs = j * BLOCK_N + rows_k
            causal_mask = k_abs[None, :] <= q_abs[:, None]
            P_ij = tl.where(causal_mask, P_ij, 0.0)
        
        dP_ij = tl.dot(dO_i, V_j.T)
        dS_ij = P_ij * (dP_ij - D_i[:, None]) * tau
        
        if i == j:
            rows_k = tl.arange(0, BLOCK_N)
            k_abs = j * BLOCK_N + rows_k
            causal_mask = k_abs[None, :] <= q_abs[:, None]
            dS_ij = tl.where(causal_mask, dS_ij, 0.0)
        
        dQ_acc = tl.dot(dS_ij, K_j, acc=dQ_acc)
    
    dQ_desc.store([i * BLOCK_N, 0], dQ_acc.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    tau = 1.0 / math.sqrt(d)
    
    T_r = triton.cdiv(S, 64)
    
    stride_b = Q.stride(0)
    stride_h = Q.stride(1)
    stride_s = Q.stride(2)
    stride_d = Q.stride(3)
    
    grid = (B, H, T_r)
    _bwd_dKdV_kernel[grid](
        Q, K, V, O, dO, L, dK, dV, S, H, d, tau,
        stride_b, stride_h, stride_s, stride_d,
        BLOCK_N=64, num_warps=4, num_stages=3
    )
    _bwd_dQ_kernel[grid](
        Q, K, V, O, dO, L, dQ, S, H, d, tau,
        stride_b, stride_h, stride_s, stride_d,
        BLOCK_N=64, num_warps=4, num_stages=3
    )