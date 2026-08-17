import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_D_kernel(O_ptr, dO_ptr, D_ptr, S, H, stride_b, stride_h, stride_s, stride_d):
    idx = tl.program_id(0)
    if idx < S * H:
        b = idx // H
        s = idx % S
        base_off = b * stride_b + s * stride_s
        o = tl.load(O_ptr + base_off + tl.arange(0, 128))
        do = tl.load(dO_ptr + base_off + tl.arange(0, 128))
        d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32))
        tl.store(D_ptr + idx, d_val)


@triton.jit
def _mha_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, D_ptr, dQ_ptr,
    S, scale, H, d,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    pid_m = tl.program_id(0)
    m_start = pid_m * BLOCK_M
    
    b_h = b * H + h
    stride_0 = S * d
    stride_1 = d
    stride_2 = 1
    
    Q_desc = tl.make_tensor_descriptor(Q_ptr, shape=[B*H, S, d], strides=[S*d, d, 1], block_shape=[1, 64, 64], padding_option="zero")
    K_desc = tl.make_tensor_descriptor(K_ptr, shape=[B*H, S, d], strides=[S*d, d, 1], block_shape=[1, 64, 64], padding_option="zero")
    V_desc = tl.make_tensor_descriptor(V_ptr, shape=[B*H, S, d], strides=[S*d, d, 1], block_shape=[1, 64, 64], padding_option="zero")
    O_desc = tl.make_tensor_descriptor(O_ptr, shape=[B*H, S, d], strides=[S*d, d, 1], block_shape=[1, 64, 64], padding_option="zero")
    dO_desc = tl.make_tensor_descriptor(dO_ptr, shape=[B*H, S, d], strides=[S*d, d, 1], block_shape=[1, 64, 64], padding_option="zero")
    dQ_desc = tl.make_tensor_descriptor(dQ_ptr, shape=[B*H, S, d], strides=[S*d, d, 1], block_shape=[1, 64, 64], padding_option="zero")
    
    Q0 = tl.reshape(Q_desc.load(b_h * stride_0 + m_start * stride_1 + 0 * stride_2), [64, 64])
    O0 = tl.reshape(O_desc.load(b_h * stride_0 + m_start * stride_1 + 0 * stride_2), [64, 64])
    dO0 = tl.reshape(dO_desc.load(b_h * stride_0 + m_start * stride_1 + 0 * stride_2), [64, 64])
    
    Q1 = tl.reshape(Q_desc.load(b_h * stride_0 + m_start * stride_1 + 64 * stride_2), [64, 64])
    O1 = tl.reshape(O_desc.load(b_h * stride_0 + m_start * stride_1 + 64 * stride_2), [64, 64])
    dO1 = tl.reshape(dO_desc.load(b_h * stride_0 + m_start * stride_1 + 64 * stride_2), [64, 64])
    
    l_idx = m_start + tl.arange(0, 64)
    L_vec = tl.load(L_ptr + b * H * S + h * S + l_idx, mask=(l_idx < S), other=0.0)
    D_vec = tl.load(D_ptr + b * H * S + h * S + l_idx, mask=(l_idx < S), other=0.0)
    
    acc_dQ0 = tl.zeros((64, 64), tl.float32)
    acc_dQ1 = tl.zeros((64, 64), tl.float32)
    
    for n_start in range(0, S, 64):
        K0 = tl.reshape(K_desc.load(b_h * stride_0 + n_start * stride_1 + 0 * stride_2), [64, 64])
        V0 = tl.reshape(V_desc.load(b_h * stride_0 + n_start * stride_1 + 0 * stride_2), [64, 64])
        K1 = tl.reshape(K_desc.load(b_h * stride_0 + n_start * stride_1 + 64 * stride_2), [64, 64])
        V1 = tl.reshape(V_desc.load(b_h * stride_0 + n_start * stride_1 + 64 * stride_2), [64, 64])
        
        S_val = tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)
        S_val *= scale
        
        P = tl.exp(S_val - L_vec[:, None])
        
        dP = tl.dot(dO0, V0.T) + tl.dot(dO1, V1.T)
        
        dS = P * (dP - D_vec[:, None]) * scale
        
        dS_bf16 = dS.to(tl.bfloat16)
        acc_dQ0 = tl.dot(dS_bf16, K0, acc_dQ0)
        acc_dQ1 = tl.dot(dS_bf16, K1, acc_dQ1)
        
    if m_start + 63 < S:
        dQ_desc.store(b_h * stride_0 + m_start * stride_1 + 0 * stride_2, tl.reshape(acc_dQ0, [1, 64, 64]))
        dQ_desc.store(b_h * stride_0 + m_start * stride_1 + 64 * stride_2, tl.reshape(acc_dQ1, [1, 64, 64]))
    else:
        mask_m = m_start + tl.arange(0, 64) < S
        dQ_desc.store(b_h * stride_0 + m_start * stride_1 + 0 * stride_2, tl.reshape(acc_dQ0, [1, 64, 64]), mask=mask_m[:, None])
        dQ_desc.store(b_h * stride_0 + m_start * stride_1 + 64 * stride_2, tl.reshape(acc_dQ1, [1, 64, 64]), mask=mask_m[:, None])


@triton.jit
def _mha_bwd_dk_dv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, D_ptr, dK_ptr, dV_ptr,
    S, scale, H, d,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    pid_n = tl.program_id(0)
    n_start = pid_n * BLOCK_N
    
    b_h = b * H + h
    stride_0 = S * d
    stride_1 = d
    stride_2 = 1
    
    Q_desc = tl.make_tensor_descriptor(Q_ptr, shape=[B*H, S, d], strides=[S*d, d, 1], block_shape=[1, 64, 64], padding_option="zero")
    K_desc = tl.make_tensor_descriptor(K_ptr, shape=[B*H, S, d], strides=[S*d, d, 1], block_shape=[1, 64, 64], padding_option="zero")
    V_desc = tl.make_tensor_descriptor(V_ptr, shape=[B*H, S, d], strides=[S*d, d, 1], block_shape=[1, 64, 64], padding_option="zero")
    O_desc = tl.make_tensor_descriptor(O_ptr, shape=[B*H, S, d], strides=[S*d, d, 1], block_shape=[1, 64, 64], padding_option="zero")
    dO_desc = tl.make_tensor_descriptor(dO_ptr, shape=[B*H, S, d], strides=[S*d, d, 1], block_shape=[1, 64, 64], padding_option="zero")
    dK_desc = tl.make_tensor_descriptor(dK_ptr, shape=[B*H, S, d], strides=[S*d, d, 1], block_shape=[1, 64, 64], padding_option="zero")
    dV_desc = tl.make_tensor_descriptor(dV_ptr, shape=[B*H, S, d], strides=[S*d, d, 1], block_shape=[1, 64, 64], padding_option="zero")

    K0 = tl.reshape(K_desc.load(b_h * stride_0 + n_start * stride_1 + 0 * stride_2), [64, 64])
    V0 = tl.reshape(V_desc.load(b_h * stride_0 + n_start * stride_1 + 0 * stride_2), [64, 64])
    K1 = tl.reshape(K_desc.load(b_h * stride_0 + n_start * stride_1 + 64 * stride_2), [64, 64])
    V1 = tl.reshape(V_desc.load(b_h * stride_0 + n_start * stride_1 + 64 * stride_2), [64, 64])
    
    acc_dK0 = tl.zeros((64, 64), tl.float32)
    acc_dK1 = tl.zeros((64, 64), tl.float32)
    acc_dV0 = tl.zeros((64, 64), tl.float32)
    acc_dV1 = tl.zeros((64, 64), tl.float32)
    
    for m_start in range(0, S, 64):
        Q0 = tl.reshape(Q_desc.load(b_h * stride_0 + m_start * stride_1 + 0 * stride_2), [64, 64])
        O0 = tl.reshape(O_desc.load(b_h * stride_0 + m_start * stride_1 + 0 * stride_2), [64, 64])
        dO0 = tl.reshape(dO_desc.load(b_h * stride_0 + m_start * stride_1 + 0 * stride_2), [64, 64])
        
        Q1 = tl.reshape(Q_desc.load(b_h * stride_0 + m_start * stride_1 + 64 * stride_2), [64, 64])
        O1 = tl.reshape(O_desc.load(b_h * stride_0 + m_start * stride_1 + 64 * stride_2), [64, 64])
        dO1 = tl.reshape(dO_desc.load(b_h * stride_0 + m_start * stride_1 + 64 * stride_2), [64, 64])
        
        l_idx = m_start + tl.arange(0, 64)
        L_vec = tl.load(L_ptr + b * H * S + h * S + l_idx, mask=(l_idx < S), other=0.0)
        D_vec = tl.load(D_ptr + b * H * S + h * S + l_idx, mask=(l_idx < S), other=0.0)
        
        S_val = tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)
        S_val *= scale
        
        P = tl.exp(S_val - L_vec[:, None])
        
        dP = tl.dot(dO0, V0.T) + tl.dot(dO1, V1.T)
        
        dS = P * (dP - D_vec[:, None]) * scale
        
        dS_bf16 = dS.to(tl.bfloat16)
        acc_dK0 = tl.dot(dS_bf16.T, Q0, acc_dK0)
        acc_dK1 = tl.dot(dS_bf16.T, Q1, acc_dK1)
        
        P_bf16 = P.to(tl.bfloat16)
        acc_dV0 = tl.dot(P_bf16.T, dO0, acc_dV0)
        acc_dV1 = tl.dot(P_bf16.T, dO1, acc_dV1)
        
    if n_start + 63 < S:
        dK_desc.store(b_h * stride_0 + n_start * stride_1 + 0 * stride_2, tl.reshape(acc_dK0, [1, 64, 64]))
        dK_desc.store(b_h * stride_0 + n_start * stride_1 + 64 * stride_2, tl.reshape(acc_dK1, [1, 64, 64]))
        dV_desc.store(b_h * stride_0 + n_start * stride_1 + 0 * stride_2, tl.reshape(acc_dV0, [1, 64, 64]))
        dV_desc.store(b_h * stride_0 + n_start * stride_1 + 64 * stride_2, tl.reshape(acc_dV1, [1, 64, 64]))
    else:
        mask_n = n_start + tl.arange(0, 64) < S
        dK_desc.store(b_h * stride_0 + n_start * stride_1 + 0 * stride_2, tl.reshape(acc_dK0, [1, 64, 64]), mask=mask_n[:, None])
        dK_desc.store(b_h * stride_0 + n_start * stride_1 + 64 * stride_2, tl.reshape(acc_dK1, [1, 64, 64]), mask=mask_n[:, None])
        dV_desc.store(b_h * stride_0 + n_start * stride_1 + 0 * stride_2, tl.reshape(acc_dV0, [1, 64, 64]), mask=mask_n[:, None])
        dV_desc.store(b_h * stride_0 + n_start * stride_1 + 64 * stride_2, tl.reshape(acc_dV1, [1, 64, 64]), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    scale = 1.0 / math.sqrt(d)
    
    D = torch.empty((B, H, S), dtype=torch.float32, device=Q.device)
    
    grid_D = (B * H * S,)
    _compute_D_kernel[grid_D](Q, K, V, O, dO, L, D, S, H, B, d)
    
    num_blocks_q = triton.cdiv(S, 64)
    grid_dq = (num_blocks_q, H, B)
    
    _mha_bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, L, D, dQ,
        S, scale, H, d,
        BLOCK_M=64, BLOCK_N=64,
        num_warps=4, num_stages=4,
    )
    
    num_blocks_k = triton.cdiv(S, 64)
    grid_dk = (num_blocks_k, H, B)
    
    _mha_bwd_dk_dv_kernel[grid_dk](
        Q, K, V, O, dO, L, D, dK, dV,
        S, scale, H, d,
        BLOCK_M=64, BLOCK_N=64,
        num_warps=4, num_stages=4,
    )