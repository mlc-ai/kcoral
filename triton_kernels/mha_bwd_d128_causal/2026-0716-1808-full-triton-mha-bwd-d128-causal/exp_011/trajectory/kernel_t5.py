import torch
import triton
import triton.language as tl

NUM_SMS = 132
BLOCK = 128

@triton.jit
def compute_dot(s_A, s_B, M, K, N):
    """Compute s_A @ s_B^T using shared memory inputs."""
    result = tl.zeros((M, N), dtype=tl.float32)
    for k in range(K):
        a = s_A[:, k]
        b = s_B[:, k]
        result += a[:, None] * b[None, :]
    return result

@triton.jit
def load_g(ptr, shmem_ptr, bh, row_base, col_base, BLOCK_M, BLOCK_N, stride_bh, stride_s, stride_d, S_len):
    r_idx = tl.arange(0, BLOCK_M)
    c_idx = tl.arange(0, BLOCK_N)
    ptrs = ptr + bh * stride_bh + r_idx[:, None] * stride_s + c_idx[None, :] * stride_d
    mask = r_idx[:, None] < S_len
    vals = tl.load(ptrs, mask=mask, other=0.0)
    shmem_ptr[r_idx, c_idx] = vals

@triton.jit
def store_g(ptr, shmem_ptr, bh, row_base, col_base, BLOCK_M, BLOCK_N, stride_bh, stride_s, stride_d, S_len):
    r_idx = tl.arange(0, BLOCK_M)
    c_idx = tl.arange(0, BLOCK_N)
    ptrs = ptr + bh * stride_bh + r_idx[:, None] * stride_s + c_idx[None, :] * stride_d
    mask = r_idx[:, None] < S_len
    vals = shmem_ptr[r_idx, c_idx]
    tl.store(ptrs, vals, mask=mask)

@triton.heuristics(values={"num_warps": 4, "num_stages": 2})
@triton.jit
def _bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
    S_len, scale, stride_l_bh, stride_bh, stride_s, stride_d,
):
    bh = tl.program_id(1)
    i = tl.program_id(0)
    i_rows = i * BLOCK + tl.arange(0, BLOCK)
    
    s_Q0 = torch.empty((BLOCK, 64), device=device, dtype=torch.bfloat16)
    s_Q1 = torch.empty((BLOCK, 64), device=device, dtype=torch.bfloat16)
    s_do0 = torch.empty((BLOCK, 64), device=device, dtype=torch.bfloat16)
    s_do1 = torch.empty((BLOCK, 64), device=device, dtype=torch.bfloat16)
    s_O0 = torch.empty((BLOCK, 64), device=device, dtype=torch.bfloat16)
    s_O1 = torch.empty((BLOCK, 64), device=device, dtype=torch.bfloat16)
    s_K0 = torch.empty((BLOCK, 64), device=device, dtype=torch.bfloat16)
    s_K1 = torch.empty((BLOCK, 64), device=device, dtype=torch.bfloat16)
    s_V0 = torch.empty((BLOCK, 64), device=device, dtype=torch.bfloat16)
    s_V1 = torch.empty((BLOCK, 64), device=device, dtype=torch.bfloat16)
    
    load_g(Q_ptr, s_Q0, bh, i * BLOCK, 0, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
    load_g(Q_ptr, s_Q1, bh, i * BLOCK, 64, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
    
    load_g(dO_ptr, s_do0, bh, i * BLOCK, 0, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
    load_g(dO_ptr, s_do1, bh, i * BLOCK, 64, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
    
    load_g(O_ptr, s_O0, bh, i * BLOCK, 0, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
    load_g(O_ptr, s_O1, bh, i * BLOCK, 64, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
    
    D = tl.sum(s_do0 * s_O0, axis=1) + tl.sum(s_do1 * s_O1, axis=1)
    D_exp = D[:, None]
    
    L_block = tl.load(L_ptr + bh * stride_l_bh + i_rows, mask=i_rows < S_len, other=-float('inf'))
    L_exp = L_block[:, None]
    
    dQ0_acc = tl.zeros((BLOCK, 64), tl.float32)
    dQ1_acc = tl.zeros((BLOCK, 64), tl.float32)
    
    for j in range(0, i + 1):
        j_rows = j * BLOCK + tl.arange(0, BLOCK)
        
        load_g(K_ptr, s_K0, bh, j * BLOCK, 0, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
        load_g(K_ptr, s_K1, bh, j * BLOCK, 64, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
        
        load_g(V_ptr, s_V0, bh, j * BLOCK, 0, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
        load_g(V_ptr, s_V1, bh, j * BLOCK, 64, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
        
        S = compute_dot(s_Q0, s_K0, BLOCK, 64, BLOCK) + compute_dot(s_Q1, s_K1, BLOCK, 64, BLOCK)
        
        P = tl.exp(S * scale - L_exp)
        
        causal_mask = ((i_rows[:, None] >= j_rows[None, :]) & (i_rows[:, None] < S_len) & (j_rows[None, :] < S_len))
        P = P * causal_mask
        
        dP = compute_dot(s_do0, s_V0, BLOCK, 64, BLOCK) + compute_dot(s_do1, s_V1, BLOCK, 64, BLOCK)
        
        dS = P * (dP - D_exp) * scale
        
        dQ0_acc += tl.dot(dS, s_K0, input_precision="ieee")
        dQ1_acc += tl.dot(dS, s_K1, input_precision="ieee")
        
    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, 64)
    mask = rows[:, None] < S_len
    
    s_dQ0 = torch.empty((BLOCK, 64), device=device, dtype=torch.bfloat16)
    s_dQ1 = torch.empty((BLOCK, 64), device=device, dtype=torch.bfloat16)
    
    s_dQ0[rows, cols] = dQ0_acc.to(tl.bfloat16)
    s_dQ1[rows, cols] = dQ1_acc.to(tl.bfloat16)
    
    store_g(dQ_ptr, s_dQ0, bh, i * BLOCK, 0, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
    store_g(dQ_ptr, s_dQ1, bh, i * BLOCK, 64, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)


@triton.heuristics(values={"num_warps": 4, "num_stages": 2})
@triton.jit
def _bwd_dkv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    S_len, scale, stride_l_bh, stride_bh, stride_s, stride_d,
):
    bh = tl.program_id(1)
    j = tl.program_id(0)
    j_rows = j * BLOCK + tl.arange(0, BLOCK)
    
    s_Q0 = torch.empty((BLOCK, 64), device=device, dtype=torch.bfloat16)
    s_Q1 = torch.empty((BLOCK, 64), device=device, dtype=torch.bfloat16)
    s_do0 = torch.empty((BLOCK, 64), device=device, dtype=torch.bfloat16)
    s_do1 = torch.empty((BLOCK, 64), device=device, dtype=torch.bfloat16)
    s_O0 = torch.empty((BLOCK, 64), device=device, dtype=torch.bfloat16)
    s_O1 = torch.empty((BLOCK, 64), device=device, dtype=torch.bfloat16)
    s_K0 = torch.empty((BLOCK, 64), device=device, dtype=torch.bfloat16)
    s_K1 = torch.empty((BLOCK, 64), device=device, dtype=torch.bfloat16)
    s_V0 = torch.empty((BLOCK, 64), device=device, dtype=torch.bfloat16)
    s_V1 = torch.empty((BLOCK, 64), device=device, dtype=torch.bfloat16)
    
    load_g(K_ptr, s_K0, bh, j * BLOCK, 0, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
    load_g(K_ptr, s_K1, bh, j * BLOCK, 64, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
    
    load_g(V_ptr, s_V0, bh, j * BLOCK, 0, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
    load_g(V_ptr, s_V1, bh, j * BLOCK, 64, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
    
    dK0_acc = tl.zeros((BLOCK, 64), tl.float32)
    dK1_acc = tl.zeros((BLOCK, 64), tl.float32)
    dV0_acc = tl.zeros((BLOCK, 64), tl.float32)
    dV1_acc = tl.zeros((BLOCK, 64), tl.float32)
    
    T_r = triton.cdiv(S_len, BLOCK)
    
    for i in range(j, T_r):
        i_rows = i * BLOCK + tl.arange(0, BLOCK)
        
        load_g(Q_ptr, s_Q0, bh, i * BLOCK, 0, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
        load_g(Q_ptr, s_Q1, bh, i * BLOCK, 64, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
        
        load_g(dO_ptr, s_do0, bh, i * BLOCK, 0, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
        load_g(dO_ptr, s_do1, bh, i * BLOCK, 64, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
        
        load_g(O_ptr, s_O0, bh, i * BLOCK, 0, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
        load_g(O_ptr, s_O1, bh, i * BLOCK, 64, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
        
        D = tl.sum(s_do0 * s_O0, axis=1) + tl.sum(s_do1 * s_O1, axis=1)
        D_exp = D[:, None]
        
        L_block = tl.load(L_ptr + bh * stride_l_bh + i_rows, mask=i_rows < S_len, other=-float('inf'))
        L_exp = L_block[:, None]
        
        S = compute_dot(s_Q0, s_K0, BLOCK, 64, BLOCK) + compute_dot(s_Q1, s_K1, BLOCK, 64, BLOCK)
        
        P = tl.exp(S * scale - L_exp)
        
        causal_mask = ((i_rows[:, None] >= j_rows[None, :]) & (i_rows[:, None] < S_len) & (j_rows[None, :] < S_len))
        P = P * causal_mask
        
        dP = compute_dot(s_do0, s_V0, BLOCK, 64, BLOCK) + compute_dot(s_do1, s_V1, BLOCK, 64, BLOCK)
        
        dS = P * (dP - D_exp) * scale
        
        dV0_acc += tl.dot(P.T, s_do0, input_precision="ieee")
        dV1_acc += tl.dot(P.T, s_do1, input_precision="ieee")
        
        dK0_acc += tl.dot(dS.T, s_Q0, input_precision="ieee")
        dK1_acc += tl.dot(dS.T, s_Q1, input_precision="ieee")
        
    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, 64)
    
    s_dK0 = torch.empty((BLOCK, 64), device=device, dtype=torch.bfloat16)
    s_dK1 = torch.empty((BLOCK, 64), device=device, dtype=torch.bfloat16)
    s_dV0 = torch.empty((BLOCK, 64), device=device, dtype=torch.bfloat16)
    s_dV1 = torch.empty((BLOCK, 64), device=device, dtype=torch.bfloat16)
    
    s_dK0[rows, cols] = dK0_acc.to(tl.bfloat16)
    s_dK1[rows, cols] = dK1_acc.to(tl.bfloat16)
    s_dV0[rows, cols] = dV0_acc.to(tl.bfloat16)
    s_dV1[rows, cols] = dV1_acc.to(tl.bfloat16)
    
    store_g(dK_ptr, s_dK0, bh, j * BLOCK, 0, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
    store_g(dK_ptr, s_dK1, bh, j * BLOCK, 64, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
    store_g(dV_ptr, s_dV0, bh, j * BLOCK, 0, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)
    store_g(dV_ptr, s_dV1, bh, j * BLOCK, 64, BLOCK, 64, stride_bh, stride_s, stride_d, S_len)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute backward attention gradients with destination-passing outputs."""
    B, H, S_len, d = Q.shape
    device = Q.device
    torch.cuda.set_device(device)
    
    scale = 1.0 / (d ** 0.5)
    
    s0, s1, s2, s3 = Q.stride()
    stride_bh = s0 // H
    stride_s = s2
    stride_d = s3
    
    stride_l_bh = S_len
    
    BLOCK = 128
    T_r = triton.cdiv(S_len, BLOCK)
    T_c = triton.cdiv(S_len, BLOCK)
    
    grid_dq = (min(NUM_SMS, T_r), B * H)
    _bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, L, dQ,
        S_len, scale, stride_l_bh, stride_bh, stride_s, stride_d,
        num_warps=4, num_stages=2
    )
    
    grid_dkv = (min(NUM_SMS, T_c), B * H)
    _bwd_dkv_kernel[grid_dkv](
        Q, K, V, O, dO, L, dK, dV,
        S_len, scale, stride_l_bh, stride_bh, stride_s, stride_d,
        num_warps=4, num_stages=2
    )