import torch
import triton
import triton.language as tl

BLOCK = 128

@triton.jit
def load_g(ptr, bh, row_base, col_base, BLOCK_M, BLOCK_N, stride_bh, stride_s, stride_d, S_len):
    r_idx = tl.arange(0, BLOCK_M)
    c_idx = tl.arange(0, BLOCK_N)
    ptrs = ptr + bh * stride_bh + r_idx[:, None] * stride_s + c_idx[None, :] * stride_d
    mask = r_idx[:, None] < S_len
    return tl.load(ptrs, mask=mask, other=0.0)

@triton.jit
def _bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
    S_len, scale, stride_l_bh, stride_bh, stride_s, stride_d,
):
    bh = tl.program_id(1)
    i = tl.program_id(0)
    i_rows = i * BLOCK + tl.arange(0, BLOCK)
    
    q0 = load_g(Q_ptr, bh, i * BLOCK, 0, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
    q1 = load_g(Q_ptr, bh, i * BLOCK, 64, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
    
    do0 = load_g(dO_ptr, bh, i * BLOCK, 0, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
    do1 = load_g(dO_ptr, bh, i * BLOCK, 64, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
    
    o0 = load_g(O_ptr, bh, i * BLOCK, 0, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
    o1 = load_g(O_ptr, bh, i * BLOCK, 64, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
    
    D = tl.sum(do0 * o0, axis=1) + tl.sum(do1 * o1, axis=1)
    D_exp = D[:, None]
    
    L_block = tl.load(L_ptr + bh * stride_l_bh + i_rows, mask=i_rows < S_len, other=0.0)
    L_exp = L_block[:, None]
    
    dQ0_acc = tl.zeros((BLOCK, 64), tl.float32)
    dQ1_acc = tl.zeros((BLOCK, 64), tl.float32)
    
    for j in range(0, i + 1):
        j_rows = j * BLOCK + tl.arange(0, BLOCK)
        
        k0 = load_g(K_ptr, bh, j * BLOCK, 0, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
        k1 = load_g(K_ptr, bh, j * BLOCK, 64, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
        
        v0 = load_g(V_ptr, bh, j * BLOCK, 0, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
        v1 = load_g(V_ptr, bh, j * BLOCK, 64, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
        
        S = tl.dot(q0, k0.T, input_precision="tf32")
        S += tl.dot(q1, k1.T, input_precision="tf32")
        
        P = tl.exp(S * scale - L_exp)
        
        causal_mask = ((i_rows[:, None] >= j_rows[None, :]) & (i_rows[:, None] < S_len) & (j_rows[None, :] < S_len))
        P = P * causal_mask
        
        dP = tl.dot(do0, v0.T, input_precision="tf32")
        dP += tl.dot(do1, v1.T, input_precision="tf32")
        
        dS = P * (dP - D_exp) * scale
        
        dQ0_acc += tl.dot(dS, k0, input_precision="tf32")
        dQ1_acc += tl.dot(dS, k1, input_precision="tf32")
        
    cols0 = tl.arange(0, 64)
    cols1 = tl.arange(0, 64)
    
    ptrs_q0 = dQ_ptr + bh * stride_bh + i_rows[:, None] * stride_s + cols0[None, :] * stride_d
    ptrs_q1 = dQ_ptr + bh * stride_bh + i_rows[:, None] * stride_s + (64 + cols1[None, :] * stride_d)
    
    mask = i_rows[:, None] < S_len
    
    tl.store(ptrs_q0, dQ0_acc.to(tl.bfloat16), mask=mask)
    tl.store(ptrs_q1, dQ1_acc.to(tl.bfloat16), mask=mask)


@triton.jit
def _bwd_dkv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    S_len, scale, stride_l_bh, stride_bh, stride_s, stride_d,
):
    bh = tl.program_id(1)
    j = tl.program_id(0)
    j_rows = j * BLOCK + tl.arange(0, BLOCK)
    
    k0 = load_g(K_ptr, bh, j * BLOCK, 0, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
    k1 = load_g(K_ptr, bh, j * BLOCK, 64, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
    
    v0 = load_g(V_ptr, bh, j * BLOCK, 0, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
    v1 = load_g(V_ptr, bh, j * BLOCK, 64, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
    
    dK0_acc = tl.zeros((BLOCK, 64), tl.float32)
    dK1_acc = tl.zeros((BLOCK, 64), tl.float32)
    dV0_acc = tl.zeros((BLOCK, 64), tl.float32)
    dV1_acc = tl.zeros((BLOCK, 64), tl.float32)
    
    T_r = triton.cdiv(S_len, BLOCK)
    
    for i in range(j, T_r):
        i_rows = i * BLOCK + tl.arange(0, BLOCK)
        
        q0 = load_g(Q_ptr, bh, i * BLOCK, 0, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
        q1 = load_g(Q_ptr, bh, i * BLOCK, 64, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
        
        do0 = load_g(dO_ptr, bh, i * BLOCK, 0, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
        do1 = load_g(dO_ptr, bh, i * BLOCK, 64, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
        
        o0 = load_g(O_ptr, bh, i * BLOCK, 0, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
        o1 = load_g(O_ptr, bh, i * BLOCK, 64, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
        
        D = tl.sum(do0 * o0, axis=1) + tl.sum(do1 * o1, axis=1)
        D_exp = D[:, None]
        
        L_block = tl.load(L_ptr + bh * stride_l_bh + i_rows, mask=i_rows < S_len, other=0.0)
        L_exp = L_block[:, None]
        
        S = tl.dot(q0, k0.T, input_precision="tf32")
        S += tl.dot(q1, k1.T, input_precision="tf32")
        
        P = tl.exp(S * scale - L_exp)
        
        causal_mask = ((i_rows[:, None] >= j_rows[None, :]) & (i_rows[:, None] < S_len) & (j_rows[None, :] < S_len))
        P = P * causal_mask
        
        dP = tl.dot(do0, v0.T, input_precision="tf32")
        dP += tl.dot(do1, v1.T, input_precision="tf32")
        
        dS = P * (dP - D_exp) * scale
        
        dK0_acc += tl.dot(dS.T, q0, input_precision="tf32")
        dK1_acc += tl.dot(dS.T, q1, input_precision="tf32")
        
        dV0_acc += tl.dot(P.T, do0, input_precision="tf32")
        dV1_acc += tl.dot(P.T, do1, input_precision="tf32")
        
    cols0 = tl.arange(0, 64)
    cols1 = tl.arange(0, 64)
    
    ptrs_k0 = dK_ptr + bh * stride_bh + j_rows[:, None] * stride_s + cols0[None, :] * stride_d
    ptrs_k1 = dK_ptr + bh * stride_bh + j_rows[:, None] * stride_s + (64 + cols1[None, :] * stride_d)
    
    ptrs_v0 = dV_ptr + bh * stride_bh + j_rows[:, None] * stride_s + cols0[None, :] * stride_d
    ptrs_v1 = dV_ptr + bh * stride_bh + j_rows[:, None] * stride_s + (64 + cols1[None, :] * stride_d)
    
    mask = j_rows[:, None] < S_len
    
    tl.store(ptrs_k0, dK0_acc.to(tl.bfloat16), mask=mask)
    tl.store(ptrs_k1, dK1_acc.to(tl.bfloat16), mask=mask)
    tl.store(ptrs_v0, dV0_acc.to(tl.bfloat16), mask=mask)
    tl.store(ptrs_v1, dV1_acc.to(tl.bfloat16), mask=mask)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute backward attention gradients with destination-passing outputs."""
    B, H, S_len, d = Q.shape
    device = Q.device
    torch.cuda.set_device(device)
    
    scale = 1.0 / (d ** 0.5)
    
    stride_bh = Q.stride(1)
    stride_s = Q.stride(2)
    stride_d = Q.stride(3)
    
    stride_l_bh = L.stride(1)
    
    T_r = triton.cdiv(S_len, BLOCK)
    T_c = triton.cdiv(S_len, BLOCK)
    
    grid_dq = (T_r, B * H)
    _bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, L, dQ,
        S_len, scale, stride_l_bh, stride_bh, stride_s, stride_d,
        num_warps=4, num_stages=2
    )
    
    grid_dkv = (T_c, B * H)
    _bwd_dkv_kernel[grid_dkv](
        Q, K, V, O, dO, L, dK, dV,
        S_len, scale, stride_l_bh, stride_bh, stride_s, stride_d,
        num_warps=4, num_stages=2
    )