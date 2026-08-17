import math
import torch
import triton
import triton.language as tl


@triton.jit
def _mha_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
    S, scale,
    stride_b, stride_h, stride_s, stride_d,
    stride_l_b, stride_l_h, stride_l_s,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    i = tl.program_id(0)
    
    base_off = b * stride_b + h * stride_h
    l_base_off = b * stride_l_b + h * stride_l_h
    
    num_blocks = tl.cdiv(S, BLOCK)
    
    q_rows = i * BLOCK + tl.arange(0, BLOCK)
    cols_0 = tl.arange(0, BLOCK)
    cols_1 = BLOCK + tl.arange(0, BLOCK)
    
    q_offs_0 = base_off + q_rows[:, None] * stride_s + cols_0[None, :] * stride_d
    Q0 = tl.load(Q_ptr + q_offs_0, mask=(q_rows[:, None] < S), other=0.0)
    q_offs_1 = base_off + q_rows[:, None] * stride_s + cols_1[None, :] * stride_d
    Q1 = tl.load(Q_ptr + q_offs_1, mask=(q_rows[:, None] < S), other=0.0)
    
    o_offs_0 = base_off + q_rows[:, None] * stride_s + cols_0[None, :] * stride_d
    O0 = tl.load(O_ptr + o_offs_0, mask=(q_rows[:, None] < S), other=0.0)
    o_offs_1 = base_off + q_rows[:, None] * stride_s + cols_1[None, :] * stride_d
    O1 = tl.load(O_ptr + o_offs_1, mask=(q_rows[:, None] < S), other=0.0)
    
    do_offs_0 = base_off + q_rows[:, None] * stride_s + cols_0[None, :] * stride_d
    dO0 = tl.load(dO_ptr + do_offs_0, mask=(q_rows[:, None] < S), other=0.0)
    do_offs_1 = base_off + q_rows[:, None] * stride_s + cols_1[None, :] * stride_d
    dO1 = tl.load(dO_ptr + do_offs_1, mask=(q_rows[:, None] < S), other=0.0)
    
    l_offs = l_base_off + q_rows * stride_l_s
    L_vec = tl.load(L_ptr + l_offs, mask=(q_rows < S), other=0.0)
    
    D = tl.sum(O0.to(tl.float32) * dO0.to(tl.float32) + O1.to(tl.float32) * dO1.to(tl.float32), axis=1)
    
    dQ0_acc = tl.zeros((BLOCK, BLOCK), tl.float32)
    dQ1_acc = tl.zeros((BLOCK, BLOCK), tl.float32)
    
    for j in range(num_blocks):
        k_rows = j * BLOCK + tl.arange(0, BLOCK)
        
        k_offs_0 = base_off + k_rows[:, None] * stride_s + cols_0[None, :] * stride_d
        K0 = tl.load(K_ptr + k_offs_0, mask=(k_rows[:, None] < S), other=0.0)
        k_offs_1 = base_off + k_rows[:, None] * stride_s + cols_1[None, :] * stride_d
        K1 = tl.load(K_ptr + k_offs_1, mask=(k_rows[:, None] < S), other=0.0)
        
        v_offs_0 = base_off + k_rows[:, None] * stride_s + cols_0[None, :] * stride_d
        V0 = tl.load(V_ptr + v_offs_0, mask=(k_rows[:, None] < S), other=0.0)
        v_offs_1 = base_off + k_rows[:, None] * stride_s + cols_1[None, :] * stride_d
        V1 = tl.load(V_ptr + v_offs_1, mask=(k_rows[:, None] < S), other=0.0)
        
        S_val = (tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)) * scale
        
        P = tl.exp(S_val - L_vec[:, None])
        
        dP = tl.dot(dO0, V0.T) + tl.dot(dO1, V1.T)
        
        dS = P * (dP - D[:, None]) * scale
        
        dS_bf16 = dS.to(tl.bfloat16)
        dQ0_acc = tl.dot(dS_bf16, K0, dQ0_acc)
        dQ1_acc = tl.dot(dS_bf16, K1, dQ1_acc)
    
    dq_offs_0 = base_off + q_rows[:, None] * stride_s + cols_0[None, :] * stride_d
    tl.store(dQ_ptr + dq_offs_0, dQ0_acc.to(tl.bfloat16), mask=(q_rows[:, None] < S))
    dq_offs_1 = base_off + q_rows[:, None] * stride_s + cols_1[None, :] * stride_d
    tl.store(dQ_ptr + dq_offs_1, dQ1_acc.to(tl.bfloat16), mask=(q_rows[:, None] < S))


@triton.jit
def _mha_bwd_dk_dv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    S, scale,
    stride_b, stride_h, stride_s, stride_d,
    stride_l_b, stride_l_h, stride_l_s,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    j = tl.program_id(0)
    
    base_off = b * stride_b + h * stride_h
    l_base_off = b * stride_l_b + h * stride_l_h
    
    num_blocks = tl.cdiv(S, BLOCK)
    
    k_rows = j * BLOCK + tl.arange(0, BLOCK)
    cols_0 = tl.arange(0, BLOCK)
    cols_1 = BLOCK + tl.arange(0, BLOCK)
    
    k_offs_0 = base_off + k_rows[:, None] * stride_s + cols_0[None, :] * stride_d
    K0 = tl.load(K_ptr + k_offs_0, mask=(k_rows[:, None] < S), other=0.0)
    k_offs_1 = base_off + k_rows[:, None] * stride_s + cols_1[None, :] * stride_d
    K1 = tl.load(K_ptr + k_offs_1, mask=(k_rows[:, None] < S), other=0.0)
    
    v_offs_0 = base_off + k_rows[:, None] * stride_s + cols_0[None, :] * stride_d
    V0 = tl.load(V_ptr + v_offs_0, mask=(k_rows[:, None] < S), other=0.0)
    v_offs_1 = base_off + k_rows[:, None] * stride_s + cols_1[None, :] * stride_d
    V1 = tl.load(V_ptr + v_offs_1, mask=(k_rows[:, None] < S), other=0.0)
    
    dK0_acc = tl.zeros((BLOCK, BLOCK), tl.float32)
    dK1_acc = tl.zeros((BLOCK, BLOCK), tl.float32)
    dV0_acc = tl.zeros((BLOCK, BLOCK), tl.float32)
    dV1_acc = tl.zeros((BLOCK, BLOCK), tl.float32)
    
    for i in range(num_blocks):
        q_rows = i * BLOCK + tl.arange(0, BLOCK)
        
        q_offs_0 = base_off + q_rows[:, None] * stride_s + cols_0[None, :] * stride_d
        Q0 = tl.load(Q_ptr + q_offs_0, mask=(q_rows[:, None] < S), other=0.0)
        q_offs_1 = base_off + q_rows[:, None] * stride_s + cols_1[None, :] * stride_d
        Q1 = tl.load(Q_ptr + q_offs_1, mask=(q_rows[:, None] < S), other=0.0)
        
        o_offs_0 = base_off + q_rows[:, None] * stride_s + cols_0[None, :] * stride_d
        O0 = tl.load(O_ptr + o_offs_0, mask=(q_rows[:, None] < S), other=0.0)
        o_offs_1 = base_off + q_rows[:, None] * stride_s + cols_1[None, :] * stride_d
        O1 = tl.load(O_ptr + o_offs_1, mask=(q_rows[:, None] < S), other=0.0)
        
        do_offs_0 = base_off + q_rows[:, None] * stride_s + cols_0[None, :] * stride_d
        dO0 = tl.load(dO_ptr + do_offs_0, mask=(q_rows[:, None] < S), other=0.0)
        do_offs_1 = base_off + q_rows[:, None] * stride_s + cols_1[None, :] * stride_d
        dO1 = tl.load(dO_ptr + do_offs_1, mask=(q_rows[:, None] < S), other=0.0)
        
        l_offs = l_base_off + q_rows * stride_l_s
        L_vec = tl.load(L_ptr + l_offs, mask=(q_rows < S), other=0.0)
        
        D = tl.sum(O0.to(tl.float32) * dO0.to(tl.float32) + O1.to(tl.float32) * dO1.to(tl.float32), axis=1)
        
        S_val = (tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)) * scale
        
        P = tl.exp(S_val - L_vec[:, None])
        
        dP = tl.dot(dO0, V0.T) + tl.dot(dO1, V1.T)
        
        dS = P * (dP - D[:, None]) * scale
        
        dS_bf16 = dS.to(tl.bfloat16)
        dK0_acc = tl.dot(dS_bf16.T, Q0, dK0_acc)
        dK1_acc = tl.dot(dS_bf16.T, Q1, dK1_acc)
        
        P_bf16 = P.to(tl.bfloat16)
        dV0_acc = tl.dot(P_bf16.T, dO0, dV0_acc)
        dV1_acc = tl.dot(P_bf16.T, dO1, dV1_acc)
    
    dk_offs_0 = base_off + k_rows[:, None] * stride_s + cols_0[None, :] * stride_d
    tl.store(dK_ptr + dk_offs_0, dK0_acc.to(tl.bfloat16), mask=(k_rows[:, None] < S))
    dk_offs_1 = base_off + k_rows[:, None] * stride_s + cols_1[None, :] * stride_d
    tl.store(dK_ptr + dk_offs_1, dK1_acc.to(tl.bfloat16), mask=(k_rows[:, None] < S))
    
    dv_offs_0 = base_off + k_rows[:, None] * stride_s + cols_0[None, :] * stride_d
    tl.store(dV_ptr + dv_offs_0, dV0_acc.to(tl.bfloat16), mask=(k_rows[:, None] < S))
    dv_offs_1 = base_off + k_rows[:, None] * stride_s + cols_1[None, :] * stride_d
    tl.store(dV_ptr + dv_offs_1, dV1_acc.to(tl.bfloat16), mask=(k_rows[:, None] < S))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    scale = 1.0 / math.sqrt(d)
    
    stride_b = Q.stride(0)
    stride_h = Q.stride(1)
    stride_s = Q.stride(2)
    stride_d = Q.stride(3)
    
    stride_l_b = L.stride(0)
    stride_l_h = L.stride(1)
    stride_l_s = L.stride(2)
    
    BLOCK = 64
    num_blocks = triton.cdiv(S, BLOCK)
    
    grid = (num_blocks, H, B)
    
    _mha_bwd_dq_kernel[grid](
        Q, K, V, O, dO, L, dQ,
        S, scale,
        stride_b, stride_h, stride_s, stride_d,
        stride_l_b, stride_l_h, stride_l_s,
        BLOCK=BLOCK,
        num_warps=4,
        num_stages=3,
    )
    
    _mha_bwd_dk_dv_kernel[grid](
        Q, K, V, O, dO, L, dK, dV,
        S, scale,
        stride_b, stride_h, stride_s, stride_d,
        stride_l_b, stride_l_h, stride_l_s,
        BLOCK=BLOCK,
        num_warps=4,
        num_stages=3,
    )