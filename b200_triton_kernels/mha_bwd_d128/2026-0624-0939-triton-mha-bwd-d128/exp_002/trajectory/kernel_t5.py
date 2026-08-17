import torch
import triton
import triton.language as tl
import math


@triton.jit
def backward_dK_dV_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    seq_len, B, H,
    q_stride_b, q_stride_h, q_stride_s, q_stride_d,
    k_stride_b, k_stride_h, k_stride_s, k_stride_d,
    v_stride_b, v_stride_h, v_stride_s, v_stride_d,
    o_stride_b, o_stride_h, o_stride_s, o_stride_d,
    do_stride_b, do_stride_h, do_stride_s, do_stride_d,
    l_stride_b, l_stride_h, l_stride_s,
    dk_stride_b, dk_stride_h, dk_stride_s, dk_stride_d,
    dv_stride_b, dv_stride_h, dv_stride_s, dv_stride_d,
    inv_sqrt_d: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    if seq_len > 0:
        assert q_stride_d == 1, f"Expected q_stride_d=1, got {q_stride_d}"
        assert q_stride_s == 128, f"Expected q_stride_s=128, got {q_stride_s}"
        assert k_stride_d == 1, f"Expected k_stride_d=1, got {k_stride_d}"
        assert v_stride_d == 1, f"Expected v_stride_d=1, got {v_stride_d}"
        assert o_stride_d == 1, f"Expected o_stride_d=1, got {o_stride_d}"
        assert do_stride_d == 1, f"Expected do_stride_d=1, got {do_stride_d}"
        assert dk_stride_d == 1, f"Expected dk_stride_d=1, got {dk_stride_d}"
        assert dv_stride_d == 1, f"Expected dv_stride_d=1, got {dv_stride_d}"

    bid_j = tl.program_id(0)
    bid_h = tl.program_id(1)
    bid_b = tl.program_id(2)
    
    j_start = bid_j * BLOCK_M
    k_row_indices = j_start + tl.arange(0, BLOCK_M)
    
    d_indices_0 = tl.arange(0, 64)
    d_indices_1 = tl.arange(0, 64)
    
    valid_k = k_row_indices[:, None] < seq_len
    
    k_ptrs_0 = K_ptr + bid_b * k_stride_b + bid_h * k_stride_h + k_row_indices[:, None] * k_stride_s + d_indices_0[None, :] * k_stride_d
    K_0_fp32 = tl.load(k_ptrs_0, mask=valid_k, other=0.0).to(tl.float32)
    
    k_ptrs_1 = K_ptr + bid_b * k_stride_b + bid_h * k_stride_h + k_row_indices[:, None] * k_stride_s + (d_indices_1[None, :] + 64) * k_stride_d
    K_1_fp32 = tl.load(k_ptrs_1, mask=valid_k, other=0.0).to(tl.float32)
    
    v_ptrs_0 = V_ptr + bid_b * v_stride_b + bid_h * v_stride_h + k_row_indices[:, None] * v_stride_s + d_indices_0[None, :] * v_stride_d
    V_0_fp32 = tl.load(v_ptrs_0, mask=valid_k, other=0.0).to(tl.float32)
    
    v_ptrs_1 = V_ptr + bid_b * v_stride_b + bid_h * v_stride_h + k_row_indices[:, None] * v_stride_s + (d_indices_1[None, :] + 64) * v_stride_d
    V_1_fp32 = tl.load(v_ptrs_1, mask=valid_k, other=0.0).to(tl.float32)
    
    dK_acc_0 = tl.zeros((BLOCK_M, 64), tl.float32)
    dK_acc_1 = tl.zeros((BLOCK_M, 64), tl.float32)
    dV_acc_0 = tl.zeros((BLOCK_M, 64), tl.float32)
    dV_acc_1 = tl.zeros((BLOCK_M, 64), tl.float32)
    
    for i in range(0, seq_len, BLOCK_M):
        q_row_indices = i + tl.arange(0, BLOCK_M)
        valid_q = q_row_indices[:, None] < seq_len
        
        q_ptrs_0 = Q_ptr + bid_b * q_stride_b + bid_h * q_stride_h + q_row_indices[:, None] * q_stride_s + d_indices_0[None, :] * q_stride_d
        Q_0_fp32 = tl.load(q_ptrs_0, mask=valid_q, other=0.0).to(tl.float32)
        
        q_ptrs_1 = Q_ptr + bid_b * q_stride_b + bid_h * q_stride_h + q_row_indices[:, None] * q_stride_s + (d_indices_1[None, :] + 64) * q_stride_d
        Q_1_fp32 = tl.load(q_ptrs_1, mask=valid_q, other=0.0).to(tl.float32)
        
        do_ptrs_0 = dO_ptr + bid_b * do_stride_b + bid_h * do_stride_h + q_row_indices[:, None] * do_stride_s + d_indices_0[None, :] * do_stride_d
        dO_0_fp32 = tl.load(do_ptrs_0, mask=valid_q, other=0.0).to(tl.float32)
        
        do_ptrs_1 = dO_ptr + bid_b * do_stride_b + bid_h * do_stride_h + q_row_indices[:, None] * do_stride_s + (d_indices_1[None, :] + 64) * do_stride_d
        dO_1_fp32 = tl.load(do_ptrs_1, mask=valid_q, other=0.0).to(tl.float32)
        
        o_ptrs_0 = O_ptr + bid_b * o_stride_b + bid_h * o_stride_h + q_row_indices[:, None] * o_stride_s + d_indices_0[None, :] * o_stride_d
        O_0_fp32 = tl.load(o_ptrs_0, mask=valid_q, other=0.0).to(tl.float32)
        
        o_ptrs_1 = O_ptr + bid_b * o_stride_b + bid_h * o_stride_h + q_row_indices[:, None] * o_stride_s + (d_indices_1[None, :] + 64) * o_stride_d
        O_1_fp32 = tl.load(o_ptrs_1, mask=valid_q, other=0.0).to(tl.float32)
        
        l_ptrs = L_ptr + bid_b * l_stride_b + bid_h * l_stride_h + q_row_indices * l_stride_s
        L_i = tl.load(l_ptrs, mask=(q_row_indices < seq_len), other=0.0)
        
        D_i = tl.sum(dO_0_fp32 * O_0_fp32 + dO_1_fp32 * O_1_fp32, axis=1)
        
        S = tl.dot(Q_0_fp32, K_0_fp32.T) + tl.dot(Q_1_fp32, K_1_fp32.T)
        P = tl.exp(S * inv_sqrt_d - L_i[:, None])
        
        dP = tl.dot(dO_0_fp32, V_0_fp32.T) + tl.dot(dO_1_fp32, V_1_fp32.T)
        dS = P * (dP - D_i[:, None])
        
        dK_acc_0 = tl.dot(dS.T, Q_0_fp32, dK_acc_0)
        dK_acc_1 = tl.dot(dS.T, Q_1_fp32, dK_acc_1)
        dV_acc_0 = tl.dot(P.T, dO_0_fp32, dV_acc_0)
        dV_acc_1 = tl.dot(P.T, dO_1_fp32, dV_acc_1)
        
    dk_ptrs_0 = dK_ptr + bid_b * dk_stride_b + bid_h * dk_stride_h + k_row_indices[:, None] * dk_stride_s + d_indices_0[None, :] * dk_stride_d
    tl.store(dk_ptrs_0, (dK_acc_0 * inv_sqrt_d).to(tl.bfloat16), mask=valid_k)
    
    dk_ptrs_1 = dK_ptr + bid_b * dk_stride_b + bid_h * dk_stride_h + k_row_indices[:, None] * dk_stride_s + (d_indices_1[None, :] + 64) * dk_stride_d
    tl.store(dk_ptrs_1, (dK_acc_1 * inv_sqrt_d).to(tl.bfloat16), mask=valid_k)
    
    dv_ptrs_0 = dV_ptr + bid_b * dv_stride_b + bid_h * dv_stride_h + k_row_indices[:, None] * dv_stride_s + d_indices_0[None, :] * dv_stride_d
    tl.store(dv_ptrs_0, dV_acc_0.to(tl.bfloat16), mask=valid_k)
    
    dv_ptrs_1 = dV_ptr + bid_b * dv_stride_b + bid_h * dv_stride_h + k_row_indices[:, None] * dv_stride_s + (d_indices_1[None, :] + 64) * dv_stride_d
    tl.store(dv_ptrs_1, dV_acc_1.to(tl.bfloat16), mask=valid_k)


@triton.jit
def backward_dQ_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
    seq_len, B, H,
    q_stride_b, q_stride_h, q_stride_s, q_stride_d,
    k_stride_b, k_stride_h, k_stride_s, k_stride_d,
    v_stride_b, v_stride_h, v_stride_s, v_stride_d,
    o_stride_b, o_stride_h, o_stride_s, o_stride_d,
    do_stride_b, do_stride_h, do_stride_s, do_stride_d,
    l_stride_b, l_stride_h, l_stride_s,
    dq_stride_b, dq_stride_h, dq_stride_s, dq_stride_d,
    inv_sqrt_d: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    if seq_len > 0:
        assert q_stride_d == 1, f"Expected q_stride_d=1, got {q_stride_d}"
        assert k_stride_d == 1, f"Expected k_stride_d=1, got {k_stride_d}"
        assert v_stride_d == 1, f"Expected v_stride_d=1, got {v_stride_d}"
        assert o_stride_d == 1, f"Expected o_stride_d=1, got {o_stride_d}"
        assert do_stride_d == 1, f"Expected do_stride_d=1, got {do_stride_d}"
        assert dq_stride_d == 1, f"Expected dq_stride_d=1, got {dq_stride_d}"

    bid_i = tl.program_id(0)
    bid_h = tl.program_id(1)
    bid_b = tl.program_id(2)
    
    i_start = bid_i * BLOCK_M
    q_row_indices = i_start + tl.arange(0, BLOCK_M)
    
    d_indices_0 = tl.arange(0, 64)
    d_indices_1 = tl.arange(0, 64)
    
    valid_q = q_row_indices[:, None] < seq_len
    
    q_ptrs_0 = Q_ptr + bid_b * q_stride_b + bid_h * q_stride_h + q_row_indices[:, None] * q_stride_s + d_indices_0[None, :] * q_stride_d
    Q_0_fp32 = tl.load(q_ptrs_0, mask=valid_q, other=0.0).to(tl.float32)
    
    q_ptrs_1 = Q_ptr + bid_b * q_stride_b + bid_h * q_stride_h + q_row_indices[:, None] * q_stride_s + (d_indices_1[None, :] + 64) * q_stride_d
    Q_1_fp32 = tl.load(q_ptrs_1, mask=valid_q, other=0.0).to(tl.float32)
    
    do_ptrs_0 = dO_ptr + bid_b * do_stride_b + bid_h * do_stride_h + q_row_indices[:, None] * do_stride_s + d_indices_0[None, :] * do_stride_d
    dO_0_fp32 = tl.load(do_ptrs_0, mask=valid_q, other=0.0).to(tl.float32)
    
    do_ptrs_1 = dO_ptr + bid_b * do_stride_b + bid_h * do_stride_h + q_row_indices[:, None] * do_stride_s + (d_indices_1[None, :] + 64) * do_stride_d
    dO_1_fp32 = tl.load(do_ptrs_1, mask=valid_q, other=0.0).to(tl.float32)
    
    o_ptrs_0 = O_ptr + bid_b * o_stride_b + bid_h * o_stride_h + q_row_indices[:, None] * o_stride_s + d_indices_0[None, :] * o_stride_d
    O_0_fp32 = tl.load(o_ptrs_0, mask=valid_q, other=0.0).to(tl.float32)
    
    o_ptrs_1 = O_ptr + bid_b * o_stride_b + bid_h * o_stride_h + q_row_indices[:, None] * o_stride_s + (d_indices_1[None, :] + 64) * o_stride_d
    O_1_fp32 = tl.load(o_ptrs_1, mask=valid_q, other=0.0).to(tl.float32)
    
    l_ptrs = L_ptr + bid_b * l_stride_b + bid_h * l_stride_h + q_row_indices * l_stride_s
    L_i = tl.load(l_ptrs, mask=(q_row_indices < seq_len), other=0.0)
    
    D_i = tl.sum(dO_0_fp32 * O_0_fp32 + dO_1_fp32 * O_1_fp32, axis=1)
    
    dQ_acc_0 = tl.zeros((BLOCK_M, 64), tl.float32)
    dQ_acc_1 = tl.zeros((BLOCK_M, 64), tl.float32)
    
    for j in range(0, seq_len, BLOCK_M):
        k_row_indices = j + tl.arange(0, BLOCK_M)
        valid_k = k_row_indices[:, None] < seq_len
        
        k_ptrs_0 = K_ptr + bid_b * k_stride_b + bid_h * k_stride_h + k_row_indices[:, None] * k_stride_s + d_indices_0[None, :] * k_stride_d
        K_0_fp32 = tl.load(k_ptrs_0, mask=valid_k, other=0.0).to(tl.float32)
        
        k_ptrs_1 = K_ptr + bid_b * k_stride_b + bid_h * k_stride_h + k_row_indices[:, None] * k_stride_s + (d_indices_1[None, :] + 64) * k_stride_d
        K_1_fp32 = tl.load(k_ptrs_1, mask=valid_k, other=0.0).to(tl.float32)
        
        v_ptrs_0 = V_ptr + bid_b * v_stride_b + bid_h * v_stride_h + k_row_indices[:, None] * v_stride_s + d_indices_0[None, :] * v_stride_d
        V_0_fp32 = tl.load(v_ptrs_0, mask=valid_k, other=0.0).to(tl.float32)
        
        v_ptrs_1 = V_ptr + bid_b * v_stride_b + bid_h * v_stride_h + k_row_indices[:, None] * v_stride_s + (d_indices_1[None, :] + 64) * v_stride_d
        V_1_fp32 = tl.load(v_ptrs_1, mask=valid_k, other=0.0).to(tl.float32)
        
        S = tl.dot(Q_0_fp32, K_0_fp32.T) + tl.dot(Q_1_fp32, K_1_fp32.T)
        P = tl.exp(S * inv_sqrt_d - L_i[:, None])
        
        dP = tl.dot(dO_0_fp32, V_0_fp32.T) + tl.dot(dO_1_fp32, V_1_fp32.T)
        dS = P * (dP - D_i[:, None])
        
        dQ_acc_0 = tl.dot(dS, K_0_fp32, dQ_acc_0)
        dQ_acc_1 = tl.dot(dS, K_1_fp32, dQ_acc_1)
        
    dq_ptrs_0 = dQ_ptr + bid_b * dq_stride_b + bid_h * dq_stride_h + q_row_indices[:, None] * dq_stride_s + d_indices_0[None, :] * dq_stride_d
    tl.store(dq_ptrs_0, (dQ_acc_0 * inv_sqrt_d).to(tl.bfloat16), mask=valid_q)
    
    dq_ptrs_1 = dQ_ptr + bid_b * dq_stride_b + bid_h * dq_stride_h + q_row_indices[:, None] * dq_stride_s + (d_indices_1[None, :] + 64) * dq_stride_d
    tl.store(dq_ptrs_1, (dQ_acc_1 * inv_sqrt_d).to(tl.bfloat16), mask=valid_q)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    inv_sqrt_d = 1.0 / math.sqrt(d)
    BLOCK_M = 64
    
    q_strides = Q.stride()
    k_strides = K.stride()
    v_strides = V.stride()
    o_strides = O.stride()
    do_strides = dO.stride()
    l_strides = L.stride()
    dq_strides = dQ.stride()
    dk_strides = dK.stride()
    dv_strides = dV.stride()
    
    grid = (triton.cdiv(S, BLOCK_M), H, B)
    
    backward_dK_dV_kernel[grid](
        Q, K, V, O, dO, L, dK, dV, S, B, H,
        q_strides[0], q_strides[1], q_strides[2], q_strides[3],
        k_strides[0], k_strides[1], k_strides[2], k_strides[3],
        v_strides[0], v_strides[1], v_strides[2], v_strides[3],
        o_strides[0], o_strides[1], o_strides[2], o_strides[3],
        do_strides[0], do_strides[1], do_strides[2], do_strides[3],
        l_strides[0], l_strides[1], l_strides[2],
        dk_strides[0], dk_strides[1], dk_strides[2], dk_strides[3],
        dv_strides[0], dv_strides[1], dv_strides[2], dv_strides[3],
        inv_sqrt_d=inv_sqrt_d, BLOCK_M=BLOCK_M, BLOCK_N=64, num_warps=4, num_stages=3
    )
    
    backward_dQ_kernel[grid](
        Q, K, V, O, dO, L, dQ, S, B, H,
        q_strides[0], q_strides[1], q_strides[2], q_strides[3],
        k_strides[0], k_strides[1], k_strides[2], k_strides[3],
        v_strides[0], v_strides[1], v_strides[2], v_strides[3],
        o_strides[0], o_strides[1], o_strides[2], o_strides[3],
        do_strides[0], do_strides[1], do_strides[2], do_strides[3],
        l_strides[0], l_strides[1], l_strides[2],
        dq_strides[0], dq_strides[1], dq_strides[2], dq_strides[3],
        inv_sqrt_d=inv_sqrt_d, BLOCK_M=BLOCK_M, BLOCK_N=64, num_warps=4, num_stages=3
    )