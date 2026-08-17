import torch
import triton
import triton.language as tl


@triton.heuristics(values={"num_warps": 4, "num_stages": 2})
@triton.jit
def _bwd_dkv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dK_ptr, dV_ptr,
    S_len, scale, stride_l_bh, stride_bh, stride_s, stride_d,
):
    bh = tl.program_id(1)
    j = tl.program_id(0)
    
    j_rows = j * 128 + tl.arange(0, 128)
    
    k0_g = tl.load(K_ptr + bh * stride_bh + j_rows[:, None] * stride_s + tl.arange(0, 64)[None, :] * stride_d, mask=j_rows[:, None] < S_len, other=0.0)
    k1_g = tl.load(K_ptr + bh * stride_bh + j_rows[:, None] * stride_s + (64 + tl.arange(0, 64))[None, :] * stride_d, mask=j_rows[:, None] < S_len, other=0.0)
    
    v0_g = tl.load(V_ptr + bh * stride_bh + j_rows[:, None] * stride_s + tl.arange(0, 64)[None, :] * stride_d, mask=j_rows[:, None] < S_len, other=0.0)
    v1_g = tl.load(V_ptr + bh * stride_bh + j_rows[:, None] * stride_s + (64 + tl.arange(0, 64))[None, :] * stride_d, mask=j_rows[:, None] < S_len, other=0.0)
    
    dK0_acc = tl.zeros((128, 64), tl.float32)
    dK1_acc = tl.zeros((128, 64), tl.float32)
    dV0_acc = tl.zeros((128, 64), tl.float32)
    dV1_acc = tl.zeros((128, 64), tl.float32)
    
    T_r = triton.cdiv(S_len, 128)
    
    for i in range(j, T_r):
        i_rows = i * 128 + tl.arange(0, 128)
        
        q0 = tl.load(Q_ptr + bh * stride_bh + i_rows[:, None] * stride_s + tl.arange(0, 64)[None, :] * stride_d, mask=i_rows[:, None] < S_len, other=0.0)
        q1 = tl.load(Q_ptr + bh * stride_bh + i_rows[:, None] * stride_s + (64 + tl.arange(0, 64))[None, :] * stride_d, mask=i_rows[:, None] < S_len, other=0.0)
        
        do0 = tl.load(dO_ptr + bh * stride_bh + i_rows[:, None] * stride_s + tl.arange(0, 64)[None, :] * stride_d, mask=i_rows[:, None] < S_len, other=0.0)
        do1 = tl.load(dO_ptr + bh * stride_bh + i_rows[:, None] * stride_s + (64 + tl.arange(0, 64))[None, :] * stride_d, mask=i_rows[:, None] < S_len, other=0.0)
        
        o0 = tl.load(O_ptr + bh * stride_bh + i_rows[:, None] * stride_s + tl.arange(0, 64)[None, :] * stride_d, mask=i_rows[:, None] < S_len, other=0.0)
        o1 = tl.load(O_ptr + bh * stride_bh + i_rows[:, None] * stride_s + (64 + tl.arange(0, 64))[None, :] * stride_d, mask=i_rows[:, None] < S_len, other=0.0)
        
        D = (tl.sum(do0 * o0, axis=1) + tl.sum(do1 * o1, axis=1))
        D_exp = D[:, None]
        
        S = tl.dot(q0, k0_g.T, input_precision="ieee")
        S += tl.dot(q1, k1_g.T, input_precision="ieee")
        
        L_block = tl.load(L_ptr + bh * stride_l_bh + i_rows, mask=i_rows < S_len, other=0.0)
        L_exp = L_block[:, None]
        
        P = tl.exp(S * scale - L_exp)
        
        q_global = i_rows[:, None]
        k_global = j_rows[None, :]
        causal_mask = (q_global >= k_global) & (q_global < S_len) & (k_global < S_len)
        P = P * causal_mask
        
        dP = tl.dot(do0, v0_g.T, input_precision="ieee")
        dP += tl.dot(do1, v1_g.T, input_precision="ieee")
        
        dS = P * (dP - D_exp) * scale
        
        dV0_acc += tl.dot(P.T, do0, input_precision="ieee")
        dV1_acc += tl.dot(P.T, do1, input_precision="ieee")
        
        dK0_acc += tl.dot(dS.T, q0, input_precision="ieee")
        dK1_acc += tl.dot(dS.T, q1, input_precision="ieee")
    
    dK0 = dK0_acc.to(tl.bfloat16)
    dK1 = dK1_acc.to(tl.bfloat16)
    dV0 = dV0_acc.to(tl.bfloat16)
    dV1 = dV1_acc.to(tl.bfloat16)
    
    mask = j_rows[:, None] < S_len
    
    ptrs_k0 = dK_ptr + bh * stride_bh + j_rows[:, None] * stride_s + tl.arange(0, 64)[None, :] * stride_d
    tl.store(ptrs_k0, dK0, mask=mask)
    
    ptrs_k1 = dK_ptr + bh * stride_bh + j_rows[:, None] * stride_s + (64 + tl.arange(0, 64))[None, :] * stride_d
    tl.store(ptrs_k1, dK1, mask=mask)
    
    ptrs_v0 = dV_ptr + bh * stride_bh + j_rows[:, None] * stride_s + tl.arange(0, 64)[None, :] * stride_d
    tl.store(ptrs_v0, dV0, mask=mask)
    
    ptrs_v1 = dV_ptr + bh * stride_bh + j_rows[:, None] * stride_s + (64 + tl.arange(0, 64))[None, :] * stride_d
    tl.store(ptrs_v1, dV1, mask=mask)


@triton.heuristics(values={"num_warps": 4, "num_stages": 2})
@triton.jit
def _bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dQ_ptr,
    S_len, scale, stride_l_bh, stride_bh, stride_s, stride_d,
):
    bh = tl.program_id(1)
    i = tl.program_id(0)
    
    i_rows = i * 128 + tl.arange(0, 128)
    
    q0_g = tl.load(Q_ptr + bh * stride_bh + i_rows[:, None] * stride_s + tl.arange(0, 64)[None, :] * stride_d, mask=i_rows[:, None] < S_len, other=0.0)
    q1_g = tl.load(Q_ptr + bh * stride_bh + i_rows[:, None] * stride_s + (64 + tl.arange(0, 64))[None, :] * stride_d, mask=i_rows[:, None] < S_len, other=0.0)
    
    do0_g = tl.load(dO_ptr + bh * stride_bh + i_rows[:, None] * stride_s + tl.arange(0, 64)[None, :] * stride_d, mask=i_rows[:, None] < S_len, other=0.0)
    do1_g = tl.load(dO_ptr + bh * stride_bh + i_rows[:, None] * stride_s + (64 + tl.arange(0, 64))[None, :] * stride_d, mask=i_rows[:, None] < S_len, other=0.0)
    
    o0_g = tl.load(O_ptr + bh * stride_bh + i_rows[:, None] * stride_s + tl.arange(0, 64)[None, :] * stride_d, mask=i_rows[:, None] < S_len, other=0.0)
    o1_g = tl.load(O_ptr + bh * stride_bh + i_rows[:, None] * stride_s + (64 + tl.arange(0, 64))[None, :] * stride_d, mask=i_rows[:, None] < S_len, other=0.0)
    
    D = (tl.sum(do0_g * o0_g, axis=1) + tl.sum(do1_g * o1_g, axis=1))
    D_exp = D[:, None]
    
    L_block = tl.load(L_ptr + bh * stride_l_bh + i_rows, mask=i_rows < S_len, other=0.0)
    L_exp = L_block[:, None]
    
    dQ0_acc = tl.zeros((128, 64), tl.float32)
    dQ1_acc = tl.zeros((128, 64), tl.float32)
    
    for j in range(0, i + 1):
        j_rows = j * 128 + tl.arange(0, 128)
        
        k0 = tl.load(K_ptr + bh * stride_bh + j_rows[:, None] * stride_s + tl.arange(0, 64)[None, :] * stride_d, mask=j_rows[:, None] < S_len, other=0.0)
        k1 = tl.load(K_ptr + bh * stride_bh + j_rows[:, None] * stride_s + (64 + tl.arange(0, 64))[None, :] * stride_d, mask=j_rows[:, None] < S_len, other=0.0)
        
        v0 = tl.load(V_ptr + bh * stride_bh + j_rows[:, None] * stride_s + tl.arange(0, 64)[None, :] * stride_d, mask=j_rows[:, None] < S_len, other=0.0)
        v1 = tl.load(V_ptr + bh * stride_bh + j_rows[:, None] * stride_s + (64 + tl.arange(0, 64))[None, :] * stride_d, mask=j_rows[:, None] < S_len, other=0.0)
        
        S = tl.dot(q0_g, k0.T, input_precision="ieee")
        S += tl.dot(q1_g, k1.T, input_precision="ieee")
        
        P = tl.exp(S * scale - L_exp)
        
        q_global = i_rows[:, None]
        k_global = j_rows[None, :]
        causal_mask = (q_global >= k_global) & (q_global < S_len) & (k_global < S_len)
        P = P * causal_mask
        
        dP = tl.dot(do0_g, v0.T, input_precision="ieee")
        dP += tl.dot(do1_g, v1.T, input_precision="ieee")
        
        dS = P * (dP - D_exp) * scale
        
        dQ0_acc += tl.dot(dS, k0, input_precision="ieee")
        dQ1_acc += tl.dot(dS, k1, input_precision="ieee")
        
    dQ0 = dQ0_acc.to(tl.bfloat16)
    dQ1 = dQ1_acc.to(tl.bfloat16)
    
    mask = i_rows[:, None] < S_len
    
    ptrs_q0 = dQ_ptr + bh * stride_bh + i_rows[:, None] * stride_s + tl.arange(0, 64)[None, :] * stride_d
    tl.store(ptrs_q0, dQ0, mask=mask)
    
    ptrs_q1 = dQ_ptr + bh * stride_bh + i_rows[:, None] * stride_s + (64 + tl.arange(0, 64))[None, :] * stride_d
    tl.store(ptrs_q1, dQ1, mask=mask)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute backward attention gradients with destination-passing outputs."""
    B, H, S_len, d = Q.shape
    device = Q.device
    torch.cuda.set_device(device)
    
    scale = 1.0 / (d ** 0.5)
    
    # Strides mapped to standard Contiguous [B, H, S, d] memory layout mappings
    stride_bh = S_len * d
    stride_s = d
    stride_d = 1
    
    stride_l_bh = S_len
    
    T_r = triton.cdiv(S_len, 128)
    T_c = triton.cdiv(S_len, 128)
    
    grid_dkv = (T_c, B * H)
    _bwd_dkv_kernel[grid_dkv](
        Q, K, V, O, dO, L,
        dK, dV,
        S_len, scale, stride_l_bh, stride_bh, stride_s, stride_d,
    )
    
    grid_dq = (T_r, B * H)
    _bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, L,
        dQ,
        S_len, scale, stride_l_bh, stride_bh, stride_s, stride_d,
    )