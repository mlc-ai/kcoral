import torch
import triton
import triton.language as tl

NUM_SMS = 132
BLOCK = 128

@triton.jit
def load_g(ptr, bh, row_base, col_base, BLOCK_M, BLOCK_N, stride_bh, stride_s, stride_d, S_len, completion_callback=None):
    r_idx = tl.arange(0, BLOCK_M)
    c_idx = tl.arange(0, BLOCK_N)
    ptrs = ptr + bh * stride_bh + r_idx[:, None] * stride_s + c_idx[None, :] * stride_d
    mask = r_idx[:, None] < S_len
    return tl.load(ptrs, mask=mask, other=0.0, completion_callback=completion_callback)

@triton.jit
def store_g(ptr, shmem_ptr, bh, row_base, col_base, BLOCK_M, BLOCK_N, stride_bh, stride_s, stride_d, S_len):
    r_idx = tl.arange(0, BLOCK_M)
    c_idx = tl.arange(0, BLOCK_N)
    ptrs = ptr + bh * stride_bh + r_idx[:, None] * stride_s + c_idx[None, :] * stride_d
    mask = r_idx[:, None] < S_len
    vals = shmem_ptr[r_idx, c_idx]
    tl.store(ptrs, vals, mask=mask)

@triton.jit
def wait_load(event):
    event.wait()

@triton.heuristics(values={"num_warps": 4, "num_stages": 2})
@triton.jit
def _bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
    S_len, scale, stride_l_bh, stride_bh, stride_s, stride_d,
):
    bh = tl.program_id(1)
    i = tl.program_id(0)
    i_rows = i * BLOCK + tl.arange(0, BLOCK)
    
    load_event = [torch.cuda.Event() for _ in range(2)]
    
    s_Q0 = tl.zeros((BLOCK, 64), dtype=tl.bfloat16)
    s_Q1 = tl.zeros((BLOCK, 64), dtype=tl.bfloat16)
    s_do0 = tl.zeros((BLOCK, 64), dtype=tl.bfloat16)
    s_do1 = tl.zeros((BLOCK, 64), dtype=tl.bfloat16)
    s_O0 = tl.zeros((BLOCK, 64), dtype=tl.bfloat16)
    s_O1 = tl.zeros((BLOCK, 64), dtype=tl.bfloat16)
    
    s_K0_0 = tl.zeros((BLOCK, 64), dtype=tl.bfloat16)
    s_K0_1 = tl.zeros((BLOCK, 64), dtype=tl.bfloat16)
    s_K1_0 = tl.zeros((BLOCK, 64), dtype=tl.bfloat16)
    s_K1_1 = tl.zeros((BLOCK, 64), dtype=tl.bfloat16)
    s_V0_0 = tl.zeros((BLOCK, 64), dtype=tl.bfloat16)
    s_V0_1 = tl.zeros((BLOCK, 64), dtype=tl.bfloat16)
    s_V1_0 = tl.zeros((BLOCK, 64), dtype=tl.bfloat16)
    s_V1_1 = tl.zeros((BLOCK, 64), dtype=tl.bfloat16)
    
    tl.load(Q_ptr + bh * stride_bh + i_rows[:, None] * stride_s + tl.arange(0, 64)[None, :] * stride_d, mask=i_rows[:, None] < S_len, other=0.0, completion_callback=wait_load(load_event[0]))
    tl.load(Q_ptr + bh * stride_bh + i_rows[:, None] * stride_s + (64 + tl.arange(0, 64))[None, :] * stride_d, mask=i_rows[:, None] < S_len, other=0.0, completion_callback=wait_load(load_event[0]))
    tl.load(dO_ptr + bh * stride_bh + i_rows[:, None] * stride_s + tl.arange(0, 64)[None, :] * stride_d, mask=i_rows[:, None] < S_len, other=0.0, completion_callback=wait_load(load_event[0]))
    tl.load(dO_ptr + bh * stride_bh + i_rows[:, None] * stride_s + (64 + tl.arange(0, 64))[None, :] * stride_d, mask=i_rows[:, None] < S_len, other=0.0, completion_callback=wait_load(load_event[0]))
    tl.load(O_ptr + bh * stride_bh + i_rows[:, None] * stride_s + tl.arange(0, 64)[None, :] * stride_d, mask=i_rows[:, None] < S_len, other=0.0, completion_callback=wait_load(load_event[0]))
    tl.load(O_ptr + bh * stride_bh + i_rows[:, None] * stride_s + (64 + tl.arange(0, 64))[None, :] * stride_d, mask=i_rows[:, None] < S_len, other=0.0, completion_callback=wait_load(load_event[0]))
    
    if 0 <= i:
        tl.load(K_ptr + bh * stride_bh + (0 * BLOCK + tl.arange(0, BLOCK))[:, None] * stride_s + tl.arange(0, 64)[None, :] * stride_d, mask=(0 * BLOCK + tl.arange(0, BLOCK))[:, None] < S_len, other=0.0, completion_callback=wait_load(load_event[0]))
        tl.load(K_ptr + bh * stride_bh + (0 * BLOCK + tl.arange(0, BLOCK))[:, None] * stride_s + (64 + tl.arange(0, 64))[None, :] * stride_d, mask=(0 * BLOCK + tl.arange(0, BLOCK))[:, None] < S_len, other=0.0, completion_callback=wait_load(load_event[0]))
        tl.load(V_ptr + bh * stride_bh + (0 * BLOCK + tl.arange(0, BLOCK))[:, None] * stride_s + tl.arange(0, 64)[None, :] * stride_d, mask=(0 * BLOCK + tl.arange(0, BLOCK))[:, None] < S_len, other=0.0, completion_callback=wait_load(load_event[0]))
        tl.load(V_ptr + bh * stride_bh + (0 * BLOCK + tl.arange(0, BLOCK))[:, None] * stride_s + (64 + tl.arange(0, 64))[None, :] * stride_d, mask=(0 * BLOCK + tl.arange(0, BLOCK))[:, None] < S_len, other=0.0, completion_callback=wait_load(load_event[0]))
    
    wait_load(load_event[0])
    
    D = tl.sum(s_do0 * s_O0, axis=1) + tl.sum(s_do1 * s_O1, axis=1)
    D_exp = D[:, None]
    
    L_block = tl.load(L_ptr + bh * stride_l_bh + i_rows, mask=i_rows < S_len, other=-float('inf'))
    L_exp = L_block[:, None]
    
    dQ0_acc = tl.zeros((BLOCK, 64), dtype=tl.float32)
    dQ1_acc = tl.zeros((BLOCK, 64), dtype=tl.float32)
    
    for j in range(0, i + 1):
        step = j
        j_rows = j * BLOCK + tl.arange(0, BLOCK)
        
        if j + 1 <= i:
            next_j = j + 1
            next_load = wait_load(load_event[1 - (step % 2)])
            tl.load(K_ptr + bh * stride_bh + (next_j * BLOCK + tl.arange(0, BLOCK))[:, None] * stride_s + tl.arange(0, 64)[None, :] * stride_d, mask=(next_j * BLOCK + tl.arange(0, BLOCK))[:, None] < S_len, other=0.0, completion_callback=next_load)
            tl.load(K_ptr + bh * stride_bh + (next_j * BLOCK + tl.arange(0, BLOCK))[:, None] * stride_s + (64 + tl.arange(0, 64))[None, :] * stride_d, mask=(next_j * BLOCK + tl.arange(0, BLOCK))[:, None] < S_len, other=0.0, completion_callback=next_load)
            tl.load(V_ptr + bh * stride_bh + (next_j * BLOCK + tl.arange(0, BLOCK))[:, None] * stride_s + tl.arange(0, 64)[None, :] * stride_d, mask=(next_j * BLOCK + tl.arange(0, BLOCK))[:, None] < S_len, other=0.0, completion_callback=next_load)
            tl.load(V_ptr + bh * stride_bh + (next_j * BLOCK + tl.arange(0, BLOCK))[:, None] * stride_s + (64 + tl.arange(0, 64))[None, :] * stride_d, mask=(next_j * BLOCK + tl.arange(0, BLOCK))[:, None] < S_len, other=0.0, completion_callback=next_load)
            
        wait_load(load_event[step % 2])
        
        buf = step % 2
        s_k0 = s_K0_0 if buf == 0 else s_K0_1
        s_k1 = s_K1_0 if buf == 0 else s_K1_1
        s_v0 = s_V0_0 if buf == 0 else s_V0_1
        s_v1 = s_V1_0 if buf == 0 else s_V1_1
        
        S = tl.dot(s_Q0, s_k0.T, input_precision="tf32")
        S += tl.dot(s_Q1, s_k1.T, input_precision="tf32")
        
        P = tl.exp(S * scale - L_exp)
        
        causal_mask = ((i_rows[:, None] >= j_rows[None, :]) & (i_rows[:, None] < S_len) & (j_rows[None, :] < S_len))
        P = P * causal_mask
        
        dP = tl.dot(s_do0, s_v0.T, input_precision="tf32")
        dP += tl.dot(s_do1, s_v1.T, input_precision="tf32")
        
        dS = P * (dP - D_exp) * scale
        
        dQ0_acc += tl.dot(dS, s_k0, input_precision="tf32")
        dQ1_acc += tl.dot(dS, s_k1, input_precision="tf32")
        
    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, 64)
    
    s_dQ0 = tl.zeros((BLOCK, 64), dtype=tl.bfloat16)
    s_dQ1 = tl.zeros((BLOCK, 64), dtype=tl.bfloat16)
    
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
    
    load_event = [torch.cuda.Event() for _ in range(2)]
    
    s_K0 = tl.zeros((BLOCK, 64), dtype=tl.bfloat16)
    s_K1 = tl.zeros((BLOCK, 64), dtype=tl.bfloat16)
    s_V0 = tl.zeros((BLOCK, 64), dtype=tl.bfloat16)
    s_V1 = tl.zeros((BLOCK, 64), dtype=tl.bfloat16)
    
    s_Q0_0 = tl.zeros((BLOCK, 64), dtype=tl.bfloat16)
    s_Q0_1 = tl.zeros((BLOCK, 64), dtype=tl.bfloat16)
    s_Q1_0 = tl.zeros((BLOCK, 64), dtype=tl.bfloat16)
    s_Q1_1 = tl.zeros((BLOCK, 64), dtype=tl.bfloat16)
    
    s_do0_0 = tl.zeros((BLOCK, 64), dtype=tl.bfloat16)
    s_do0_1 = tl.zeros((BLOCK, 64), dtype=tl.bfloat16)
    s_do1_0 = tl.zeros((BLOCK, 64), dtype=tl.bfloat16)
    s_do1_1 = tl.zeros((BLOCK, 64), dtype=tl.bfloat16)
    
    s_O0_0 = tl.zeros((BLOCK, 64), dtype=tl.bfloat16)
    s_O0_1 = tl.zeros((BLOCK, 64), dtype=tl.bfloat16)
    s_O1_0 = tl.zeros((BLOCK, 64), dtype=tl.bfloat16)
    s_O1_1 = tl.zeros((BLOCK, 64), dtype=tl.bfloat16)
    
    T_r = triton.cdiv(S_len, BLOCK)
    
    if j < T_r:
        init_i = j
        init_load = wait_load(load_event[0])
        tl.load(Q_ptr + bh * stride_bh + (init_i * BLOCK + tl.arange(0, BLOCK))[:, None] * stride_s + tl.arange(0, 64)[None, :] * stride_d, mask=(init_i * BLOCK + tl.arange(0, BLOCK))[:, None] < S_len, other=0.0, completion_callback=init_load)
        tl.load(Q_ptr + bh * stride_bh + (init_i * BLOCK + tl.arange(0, BLOCK))[:, None] * stride_s + (64 + tl.arange(0, 64))[None, :] * stride_d, mask=(init_i * BLOCK + tl.arange(0, BLOCK))[:, None] < S_len, other=0.0, completion_callback=init_load)
        
        tl.load(dO_ptr + bh * stride_bh + (init_i * BLOCK + tl.arange(0, BLOCK))[:, None] * stride_s + tl.arange(0, 64)[None, :] * stride_d, mask=(init_i * BLOCK + tl.arange(0, BLOCK))[:, None] < S_len, other=0.0, completion_callback=init_load)
        tl.load(dO_ptr + bh * stride_bh + (init_i * BLOCK + tl.arange(0, BLOCK))[:, None] * stride_s + (64 + tl.arange(0, 64))[None, :] * stride_d, mask=(init_i * BLOCK + tl.arange(0, BLOCK))[:, None] < S_len, other=0.0, completion_callback=init_load)
        
        tl.load(O_ptr + bh * stride_bh + (init_i * BLOCK + tl.arange(0, BLOCK))[:, None] * stride_s + tl.arange(0, 64)[None, :] * stride_d, mask=(init_i * BLOCK + tl.arange(0, BLOCK))[:, None] < S_len, other=0.0, completion_callback=init_load)
        tl.load(O_ptr + bh * stride_bh + (init_i * BLOCK + tl.arange(0, BLOCK))[:, None] * stride_s + (64 + tl.arange(0, 64))[None, :] * stride_d, mask=(init_i * BLOCK + tl.arange(0, BLOCK))[:, None] < S_len, other=0.0, completion_callback=init_load)
    
    wait_load(load_event[0])
    
    dK0_acc = tl.zeros((BLOCK, 64), dtype=tl.float32)
    dK1_acc = tl.zeros((BLOCK, 64), dtype=tl.float32)
    dV0_acc = tl.zeros((BLOCK, 64), dtype=tl.float32)
    dV1_acc = tl.zeros((BLOCK, 64), dtype=tl.float32)
    
    for i in range(j, T_r):
        step = i - j
        i_rows = i * BLOCK + tl.arange(0, BLOCK)
        
        if i + 1 < T_r:
            next_i = i + 1
            next_load = wait_load(load_event[1 - (step % 2)])
            tl.load(Q_ptr + bh * stride_bh + (next_i * BLOCK + tl.arange(0, BLOCK))[:, None] * stride_s + tl.arange(0, 64)[None, :] * stride_d, mask=(next_i * BLOCK + tl.arange(0, BLOCK))[:, None] < S_len, other=0.0, completion_callback=next_load)
            tl.load(Q_ptr + bh * stride_bh + (next_i * BLOCK + tl.arange(0, BLOCK))[:, None] * stride_s + (64 + tl.arange(0, 64))[None, :] * stride_d, mask=(next_i * BLOCK + tl.arange(0, BLOCK))[:, None] < S_len, other=0.0, completion_callback=next_load)
            
            tl.load(dO_ptr + bh * stride_bh + (next_i * BLOCK + tl.arange(0, BLOCK))[:, None] * stride_s + tl.arange(0, 64)[None, :] * stride_d, mask=(next_i * BLOCK + tl.arange(0, BLOCK))[:, None] < S_len, other=0.0, completion_callback=next_load)
            tl.load(dO_ptr + bh * stride_bh + (next_i * BLOCK + tl.arange(0, BLOCK))[:, None] * stride_s + (64 + tl.arange(0, 64))[None, :] * stride_d, mask=(next_i * BLOCK + tl.arange(0, BLOCK))[:, None] < S_len, other=0.0, completion_callback=next_load)
            
            tl.load(O_ptr + bh * stride_bh + (next_i * BLOCK + tl.arange(0, BLOCK))[:, None] * stride_s + tl.arange(0, 64)[None, :] * stride_d, mask=(next_i * BLOCK + tl.arange(0, BLOCK))[:, None] < S_len, other=0.0, completion_callback=next_load)
            tl.load(O_ptr + bh * stride_bh + (next_i * BLOCK + tl.arange(0, BLOCK))[:, None] * stride_s + (64 + tl.arange(0, 64))[None, :] * stride_d, mask=(next_i * BLOCK + tl.arange(0, BLOCK))[:, None] < S_len, other=0.0, completion_callback=next_load)
            
        wait_load(load_event[step % 2])
        
        buf = step % 2
        s_q0 = s_Q0_0 if buf == 0 else s_Q0_1
        s_q1 = s_Q1_0 if buf == 0 else s_Q1_1
        s_do0 = s_do0_0 if buf == 0 else s_do0_1
        s_do1 = s_do1_0 if buf == 0 else s_do1_1
        s_o0 = s_O0_0 if buf == 0 else s_O0_1
        s_o1 = s_O1_0 if buf == 0 else s_O1_1
        
        D = tl.sum(s_do0 * s_o0, axis=1) + tl.sum(s_do1 * s_o1, axis=1)
        D_exp = D[:, None]
        
        L_block = tl.load(L_ptr + bh * stride_l_bh + i_rows, mask=i_rows < S_len, other=-float('inf'))
        L_exp = L_block[:, None]
        
        S = tl.dot(s_q0, s_K0.T, input_precision="tf32")
        S += tl.dot(s_q1, s_K1.T, input_precision="tf32")
        
        P = tl.exp(S * scale - L_exp)
        
        causal_mask = ((i_rows[:, None] >= j_rows[None, :]) & (i_rows[:, None] < S_len) & (j_rows[None, :] < S_len))
        P = P * causal_mask
        
        dP = tl.dot(s_do0, s_V0.T, input_precision="tf32")
        dP += tl.dot(s_do1, s_V1.T, input_precision="tf32")
        
        dS = P * (dP - D_exp) * scale
        
        dK0_acc += tl.dot(dS.T, s_q0, input_precision="tf32")
        dK1_acc += tl.dot(dS.T, s_q1, input_precision="tf32")
        
        dV0_acc += tl.dot(P.T, s_do0, input_precision="tf32")
        dV1_acc += tl.dot(P.T, s_do1, input_precision="tf32")
        
    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, 64)
    
    s_dK0 = tl.zeros((BLOCK, 64), dtype=tl.bfloat16)
    s_dK1 = tl.zeros((BLOCK, 64), dtype=tl.bfloat16)
    s_dV0 = tl.zeros((BLOCK, 64), dtype=tl.bfloat16)
    s_dV1 = tl.zeros((BLOCK, 64), dtype=tl.bfloat16)
    
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
    
    stride_bh = S_len * d
    stride_s = d
    stride_d = 1
    
    stride_l_bh = S_len
    
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