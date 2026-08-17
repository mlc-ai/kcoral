import torch
import triton
import triton.language as tl


@triton.heuristics(values={"num_warps": 4, "num_stages": 2})
@triton.jit
def _bwd_dkv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dK_ptr, dV_ptr,
    S_len, scale, stride_l_bh,
    stride_bh: tl.constexpr, stride_s: tl.constexpr, stride_d: tl.constexpr,
    BLOCK: tl.constexpr,
):
    bh = tl.program_id(1)
    j = tl.program_id(0)
    
    s_K0 = tl.empty((BLOCK, 64), dtype=tl.bfloat16)
    s_K1 = tl.empty((BLOCK, 64), dtype=tl.bfloat16)
    s_V0 = tl.empty((BLOCK, 64), dtype=tl.bfloat16)
    s_V1 = tl.empty((BLOCK, 64), dtype=tl.bfloat16)
    
    k0_g = load_g(K_ptr, bh, j * BLOCK, 0, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
    k1_g = load_g(K_ptr, bh, j * BLOCK, 64, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
    v0_g = load_g(V_ptr, bh, j * BLOCK, 0, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
    v1_g = load_g(V_ptr, bh, j * BLOCK, 64, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
    
    store_2(s_K0, k0_g, j * BLOCK, S_len, BLOCK)
    store_2(s_K1, k1_g, j * BLOCK, S_len, BLOCK)
    store_2(s_V0, v0_g, j * BLOCK, S_len, BLOCK)
    store_2(s_V1, v1_g, j * BLOCK, S_len, BLOCK)
    
    dK0_acc = tl.zeros((BLOCK, 64), tl.float32)
    dK1_acc = tl.zeros((BLOCK, 64), tl.float32)
    dV0_acc = tl.zeros((BLOCK, 64), tl.float32)
    dV1_acc = tl.zeros((BLOCK, 64), tl.float32)
    
    T_r = triton.cdiv(S_len, BLOCK)
    
    for i in range(j, T_r):
        q0 = load_g(Q_ptr, bh, i * BLOCK, 0, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
        q1 = load_g(Q_ptr, bh, i * BLOCK, 64, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
        do0 = load_g(dO_ptr, bh, i * BLOCK, 0, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
        do1 = load_g(dO_ptr, bh, i * BLOCK, 64, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
        o0 = load_g(O_ptr, bh, i * BLOCK, 0, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
        o1 = load_g(O_ptr, bh, i * BLOCK, 64, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
        
        D = (tl.sum(do0 * o0, axis=1) + tl.sum(do1 * o1, axis=1))
        D_exp = D[:, None]
        
        S = tl.dot(q0, load_2(s_K0, j * BLOCK, S_len, BLOCK).T, input_precision="ieee")
        S += tl.dot(q1, load_2(s_K1, j * BLOCK, S_len, BLOCK).T, input_precision="ieee")
        
        L_block = tl.load(L_ptr + bh * stride_l_bh + i * BLOCK + tl.arange(0, BLOCK), mask=(i * BLOCK + tl.arange(0, BLOCK)) < S_len, other=0.0)
        L_exp = L_block[:, None]
        
        P = tl.exp(S * scale - L_exp)
        
        causal_mask = compute_causal_mask(i * BLOCK, j * BLOCK, S_len, BLOCK)
        P = P * causal_mask
        
        dP = tl.dot(do0, load_2(s_V0, j * BLOCK, S_len, BLOCK).T, input_precision="ieee")
        dP += tl.dot(do1, load_2(s_V1, j * BLOCK, S_len, BLOCK).T, input_precision="ieee")
        
        dS = P * (dP - D_exp) * scale
        
        dV0_acc += tl.dot(P.T, do0, input_precision="ieee")
        dV1_acc += tl.dot(P.T, do1, input_precision="ieee")
        
        dK0_acc += tl.dot(dS.T, q0, input_precision="ieee")
        dK1_acc += tl.dot(dS.T, q1, input_precision="ieee")
    
    store_g(dK_ptr, bh, j * BLOCK, 0, BLOCK, 64, dK0_acc.to(tl.bfloat16), stride_bh, stride_s, stride_d, S_len)
    store_g(dK_ptr, bh, j * BLOCK, 64, BLOCK, 64, dK1_acc.to(tl.bfloat16), stride_bh, stride_s, stride_d, S_len)
    store_g(dV_ptr, bh, j * BLOCK, 0, BLOCK, 64, dV0_acc.to(tl.bfloat16), stride_bh, stride_s, stride_d, S_len)
    store_g(dV_ptr, bh, j * BLOCK, 64, BLOCK, 64, dV1_acc.to(tl.bfloat16), stride_bh, stride_s, stride_d, S_len)


@triton.heuristics(values={"num_warps": 4, "num_stages": 2})
@triton.jit
def _bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dQ_ptr,
    S_len, scale, stride_l_bh,
    stride_bh: tl.constexpr, stride_s: tl.constexpr, stride_d: tl.constexpr,
    BLOCK: tl.constexpr,
):
    bh = tl.program_id(1)
    i = tl.program_id(0)
    
    s_Q0 = tl.empty((BLOCK, 64), dtype=tl.bfloat16)
    s_Q1 = tl.empty((BLOCK, 64), dtype=tl.bfloat16)
    s_dO0 = tl.empty((BLOCK, 64), dtype=tl.bfloat16)
    s_dO1 = tl.empty((BLOCK, 64), dtype=tl.bfloat16)
    s_O0 = tl.empty((BLOCK, 64), dtype=tl.bfloat16)
    s_O1 = tl.empty((BLOCK, 64), dtype=tl.bfloat16)
    
    q0_g = load_g(Q_ptr, bh, i * BLOCK, 0, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
    q1_g = load_g(Q_ptr, bh, i * BLOCK, 64, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
    do0_g = load_g(dO_ptr, bh, i * BLOCK, 0, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
    do1_g = load_g(dO_ptr, bh, i * BLOCK, 64, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
    o0_g = load_g(O_ptr, bh, i * BLOCK, 0, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
    o1_g = load_g(O_ptr, bh, i * BLOCK, 64, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
    
    store_2(s_Q0, q0_g, i * BLOCK, S_len, BLOCK)
    store_2(s_Q1, q1_g, i * BLOCK, S_len, BLOCK)
    store_2(s_dO0, do0_g, i * BLOCK, S_len, BLOCK)
    store_2(s_dO1, do1_g, i * BLOCK, S_len, BLOCK)
    store_2(s_O0, o0_g, i * BLOCK, S_len, BLOCK)
    store_2(s_O1, o1_g, i * BLOCK, S_len, BLOCK)
    
    do0 = load_2(s_dO0, i * BLOCK, S_len, BLOCK)
    do1 = load_2(s_dO1, i * BLOCK, S_len, BLOCK)
    o0 = load_2(s_O0, i * BLOCK, S_len, BLOCK)
    o1 = load_2(s_O1, i * BLOCK, S_len, BLOCK)
    
    D = (tl.sum(do0 * o0, axis=1) + tl.sum(do1 * o1, axis=1))
    D_exp = D[:, None]
    
    L_block = tl.load(L_ptr + bh * stride_l_bh + i * BLOCK + tl.arange(0, BLOCK), mask=(i * BLOCK + tl.arange(0, BLOCK)) < S_len, other=0.0)
    L_exp = L_block[:, None]
    
    dQ0_acc = tl.zeros((BLOCK, 64), tl.float32)
    dQ1_acc = tl.zeros((BLOCK, 64), tl.float32)
    
    for j in range(0, i + 1):
        k0 = load_g(K_ptr, bh, j * BLOCK, 0, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
        k1 = load_g(K_ptr, bh, j * BLOCK, 64, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
        v0 = load_g(V_ptr, bh, j * BLOCK, 0, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
        v1 = load_g(V_ptr, bh, j * BLOCK, 64, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
        
        S = tl.dot(load_2(s_Q0, i * BLOCK, S_len, BLOCK), k0.T, input_precision="ieee")
        S += tl.dot(load_2(s_Q1, i * BLOCK, S_len, BLOCK), k1.T, input_precision="ieee")
        
        P = tl.exp(S * scale - L_exp)
        
        causal_mask = compute_causal_mask(i * BLOCK, j * BLOCK, S_len, BLOCK)
        P = P * causal_mask
        
        dP = tl.dot(load_2(s_dO0, i * BLOCK, S_len, BLOCK), v0.T, input_precision="ieee")
        dP += tl.dot(load_2(s_dO1, i * BLOCK, S_len, BLOCK), v1.T, input_precision="ieee")
        
        dS = P * (dP - D_exp) * scale
        
        dQ0_acc += tl.dot(dS, k0, input_precision="ieee")
        dQ1_acc += tl.dot(dS, k1, input_precision="ieee")
        
    store_g(dQ_ptr, bh, i * BLOCK, 0, BLOCK, 64, dQ0_acc.to(tl.bfloat16), stride_bh, stride_s, stride_d, S_len)
    store_g(dQ_ptr, bh, i * BLOCK, 64, BLOCK, 64, dQ1_acc.to(tl.bfloat16), stride_bh, stride_s, stride_d, S_len)


@triton.jit
def compute_causal_mask(q_base, k_base, S_len, BLOCK):
    q_global = q_base + tl.arange(0, BLOCK)[:, None]
    k_global = k_base + tl.arange(0, BLOCK)[None, :]
    causal_mask = (q_global >= k_global) & (q_global < S_len) & (k_global < S_len)
    return causal_mask

@triton.jit
def load_g(ptr, bh, row_base, col_base, BLOCK_M, BLOCK_N, stride_bh, stride_s, stride_d, S_len):
    rows = row_base + tl.arange(0, BLOCK_M)
    cols = col_base + tl.arange(0, BLOCK_N)
    ptrs = ptr + bh * stride_bh + rows[:, None] * stride_s + cols[None, :] * stride_d
    mask = rows[:, None] < S_len
    return tl.load(ptrs, mask=mask, other=0.0)

@triton.jit
def store_g(ptr, bh, row_base, col_base, BLOCK_M, BLOCK_N, value, stride_bh, stride_s, stride_d, S_len):
    rows = row_base + tl.arange(0, BLOCK_M)
    cols = col_base + tl.arange(0, BLOCK_N)
    ptrs = ptr + bh * stride_bh + rows[:, None] * stride_s + cols[None, :] * stride_d
    mask = rows[:, None] < S_len
    tl.store(ptrs, value, mask=mask)

@triton.jit
def store_2(s_ptr, g_data, row_base, S_len, BLOCK_M):
    rows = row_base + tl.arange(0, BLOCK_M)
    mask = rows[:, None] < S_len
    tl.store(s_ptr, g_data, mask=mask)

@triton.jit
def load_2(s_ptr, row_base, S_len, BLOCK_M):
    rows = row_base + tl.arange(0, BLOCK_M)
    mask = rows[:, None] < S_len
    return tl.load(s_ptr, mask=mask, other=0.0)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute backward attention gradients with destination-passing outputs."""
    B, H, S_len, d = Q.shape
    device = Q.device
    torch.cuda.set_device(device)
    
    scale = 1.0 / (d ** 0.5)
    
    stride_bh = Q.stride(0)
    stride_s = Q.stride(1)
    stride_d = Q.stride(2)
    
    stride_l_bh = L.stride(0)
    
    T_r = triton.cdiv(S_len, 128)
    T_c = triton.cdiv(S_len, 128)
    
    grid_dkv = (T_c, B * H)
    _bwd_dkv_kernel[grid_dkv](
        Q, K, V, O, dO, L,
        dK, dV,
        S_len, scale, stride_l_bh,
        stride_bh, stride_s, stride_d, BLOCK=128
    )
    
    grid_dq = (T_r, B * H)
    _bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, L,
        dQ,
        S_len, scale, stride_l_bh,
        stride_bh, stride_s, stride_d, BLOCK=128
    )