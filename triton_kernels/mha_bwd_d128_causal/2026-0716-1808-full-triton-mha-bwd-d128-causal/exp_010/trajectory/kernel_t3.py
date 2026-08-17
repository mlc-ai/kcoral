import ctypes
import math
import torch
import triton
import triton.language as tl


@triton.jit
def init_tma_payload():
    ptr = tl.empty((1,), tl.int64)
    ptr[0] = 0
    return ptr


@triton.jit
def bwd_dkdv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    B, H, S, d,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    scale = 1.0 / math.sqrt(d)
    bh = tl.program_id(0)
    j = tl.program_id(1)
    S_len = S
    
    desc_Q = tl.make_tensor_descriptor(Q_ptr, shape=[B*H*S_len, d], strides=[d, 1], block_shape=[BLOCK_M, 64], padding_option="zero")
    desc_K = tl.make_tensor_descriptor(K_ptr, shape=[B*H*S_len, d], strides=[d, 1], block_shape=[BLOCK_N, 64], padding_option="zero")
    desc_V = tl.make_tensor_descriptor(V_ptr, shape=[B*H*S_len, d], strides=[d, 1], block_shape=[BLOCK_N, 64], padding_option="zero")
    desc_O = tl.make_tensor_descriptor(O_ptr, shape=[B*H*S_len, d], strides=[d, 1], block_shape=[BLOCK_M, 64], padding_option="zero")
    desc_dO = tl.make_tensor_descriptor(dO_ptr, shape=[B*H*S_len, d], strides=[d, 1], block_shape=[BLOCK_M, 64], padding_option="zero")
    desc_dK = tl.make_tensor_descriptor(dK_ptr, shape=[B*H*S_len, d], strides=[d, 1], block_shape=[BLOCK_N, 64], padding_option="zero")
    desc_dV = tl.make_tensor_descriptor(dV_ptr, shape=[B*H*S_len, d], strides=[d, 1], block_shape=[BLOCK_N, 64], padding_option="zero")
    
    p_K0 = init_tma_payload()
    p_K0 = p_K0.extend(desc_K.start_load(p_K0))
    p_K1 = init_tma_payload()
    p_K1 = p_K1.extend(desc_K.start_load(p_K1))
    p_V0 = init_tma_payload()
    p_V0 = p_V0.extend(desc_V.start_load(p_V0))
    p_V1 = init_tma_payload()
    p_V1 = p_V1.extend(desc_V.start_load(p_V1))
    
    p_Q0 = init_tma_payload()
    p_Q0 = p_Q0.extend(desc_Q.start_load(p_Q0))
    p_Q1 = init_tma_payload()
    p_Q1 = p_Q1.extend(desc_Q.start_load(p_Q1))
    p_O0 = init_tma_payload()
    p_O0 = p_O0.extend(desc_O.start_load(p_O0))
    p_O1 = init_tma_payload()
    p_O1 = p_O1.extend(desc_O.start_load(p_O1))
    p_dO0 = init_tma_payload()
    p_dO0 = p_dO0.extend(desc_dO.start_load(p_dO0))
    p_dO1 = init_tma_payload()
    p_dO1 = p_dO1.extend(desc_dO.start_load(p_dO1))
    
    asm_volatile_inline(f"cp_async_commit;", ...)
    asm_volatile_inline(f"tl_sem_prio_async_copy_wait;\n"
                         f"commit_group;")
    
    dK0 = tl.zeros((BLOCK_N, 64), tl.float32)
    dK1 = tl.zeros((BLOCK_N, 64), tl.float32)
    dV0 = tl.zeros((BLOCK_N, 64), tl.float32)
    dV1 = tl.zeros((BLOCK_N, 64), tl.float32)
    
    num_Q_tiles = (S_len + BLOCK_M - 1) // BLOCK_M
    
    p_cK0 = p_K0
    p_cK1 = p_K1
    p_cV0 = p_V0
    p_cV1 = p_V1
    
    p_cQ0 = p_Q0
    p_cQ1 = p_Q1
    p_cO0 = p_O0
    p_cO1 = p_O1
    p_cdO0 = p_dO0
    p_cdO1 = p_dO1
    
    p_nQ0 = init_tma_payload()
    p_nQ1 = init_tma_payload()
    p_nO0 = init_tma_payload()
    p_nO1 = init_tma_payload()
    p_ndO0 = init_tma_payload()
    p_ndO1 = init_tma_payload()
    
    cur_idx = 0
    next_idx = 1
    
    row_idx = tl.arange(0, BLOCK_M)
    col_idx = j * BLOCK_N + tl.arange(0, BLOCK_N)
    
    for i in range(j, num_Q_tiles):
        next_i = i + 1
        
        if next_i < num_Q_tiles:
            p_nQ0 = p_nQ0.extend(desc_Q.start_load(p_nQ0))
            p_nQ1 = p_nQ1.extend(desc_Q.start_load(p_nQ1))
            p_nO0 = p_nO0.extend(desc_O.start_load(p_nO0))
            p_nO1 = p_nO1.extend(desc_O.start_load(p_nO1))
            p_ndO0 = p_ndO0.extend(desc_dO.start_load(p_ndO0))
            p_ndO1 = p_ndO1.extend(desc_dO.start_load(p_ndO1))
            
            asm_volatile_inline(f"cp_async_commit;", ...)
        
        asm_volatile_inline(f"tl_sem_prio_async_copy_wait;\n"
                             f"commit_group;")
        
        c_Q0 = desc_Q.load(p_cQ0)
        c_Q1 = desc_Q.load(p_cQ1)
        c_O0 = desc_O.load(p_cO0)
        c_O1 = desc_O.load(p_cO1)
        c_dO0 = desc_dO.load(p_cdO0)
        c_dO1 = desc_dO.load(p_cdO1)
        
        D = (c_dO0 * c_O0).sum(axis=1) + (c_dO1 * c_O1).sum(axis=1)
        
        valid_rows = (i * BLOCK_M + row_idx) < S_len
        L_i = tl.load(L_ptr + bh * S_len + i * BLOCK_M + row_idx, mask=valid_rows, other=0.0)
        
        c_K0 = desc_K.load(p_cK0)
        c_K1 = desc_K.load(p_cK1)
        c_V0 = desc_V.load(p_cV0)
        c_V1 = desc_V.load(p_cV1)
        
        s = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        s = tl.dot(c_Q0, c_K0.T, acc=s)
        s = tl.dot(c_Q1, c_K1.T, acc=s)
        
        dp = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        dp = tl.dot(c_dO0, c_V0.T, acc=dp)
        dp = tl.dot(c_dO1, c_V1.T, acc=dp)
        
        p = tl.exp(s * scale - L_i[:, None])
        ds = p * (dp - D[:, None]) * scale
        
        ds = ds * ((i * BLOCK_M + row_idx[:, None]) >= (j * BLOCK_N + col_idx[None, :]))
        
        dK0 = tl.dot(ds.T, c_Q0, acc=dK0)
        dK1 = tl.dot(ds.T, c_Q1, acc=dK1)
        dV0 = tl.dot(p.T, c_dO0, acc=dV0)
        dV1 = tl.dot(p.T, c_dO1, acc=dV1)
        
        p_cQ0, p_nQ0 = p_nQ0, p_cQ0
        p_cQ1, p_nQ1 = p_nQ1, p_cQ1
        p_cO0, p_nO0 = p_nO0, p_cO0
        p_cO1, p_nO1 = p_nO1, p_cO1
        p_cdO0, p_ndO0 = p_ndO0, p_cdO0
        p_cdO1, p_ndO1 = p_ndO1, p_cdO1
        
        cur_idx = next_idx
        next_idx = 1 - next_idx
        
    desc_dK.store([bh * S_len + j * BLOCK_N, 0], dK0.to(tl.bfloat16))
    desc_dK.store([bh * S_len + j * BLOCK_N, 64], dK1.to(tl.bfloat16))
    desc_dV.store([bh * S_len + j * BLOCK_N, 0], dV0.to(tl.bfloat16))
    desc_dV.store([bh * S_len + j * BLOCK_N, 64], dV1.to(tl.bfloat16))


@triton.jit
def bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
    B, H, S, d,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    scale = 1.0 / math.sqrt(d)
    bh = tl.program_id(0)
    i = tl.program_id(1)
    S_len = S
    
    desc_Q = tl.make_tensor_descriptor(Q_ptr, shape=[B*H*S_len, d], strides=[d, 1], block_shape=[BLOCK_M, 64], padding_option="zero")
    desc_K = tl.make_tensor_descriptor(K_ptr, shape=[B*H*S_len, d], strides=[d, 1], block_shape=[BLOCK_N, 64], padding_option="zero")
    desc_V = tl.make_tensor_descriptor(V_ptr, shape=[B*H*S_len, d], strides=[d, 1], block_shape=[BLOCK_N, 64], padding_option="zero")
    desc_O = tl.make_tensor_descriptor(O_ptr, shape=[B*H*S_len, d], strides=[d, 1], block_shape=[BLOCK_M, 64], padding_option="zero")
    desc_dO = tl.make_tensor_descriptor(dO_ptr, shape=[B*H*S_len, d], strides=[d, 1], block_shape=[BLOCK_M, 64], padding_option="zero")
    desc_dQ = tl.make_tensor_descriptor(dQ_ptr, shape=[B*H*S_len, d], strides=[d, 1], block_shape=[BLOCK_M, 64], padding_option="zero")
    
    p_Q0 = init_tma_payload()
    p_Q0 = p_Q0.extend(desc_Q.start_load(p_Q0))
    p_Q1 = init_tma_payload()
    p_Q1 = p_Q1.extend(desc_Q.start_load(p_Q1))
    p_O0 = init_tma_payload()
    p_O0 = p_O0.extend(desc_O.start_load(p_O0))
    p_O1 = init_tma_payload()
    p_O1 = p_O1.extend(desc_O.start_load(p_O1))
    p_dO0 = init_tma_payload()
    p_dO0 = p_dO0.extend(desc_dO.start_load(p_dO0))
    p_dO1 = init_tma_payload()
    p_dO1 = p_dO1.extend(desc_dO.start_load(p_dO1))
    
    asm_volatile_inline(f"cp_async_commit;", ...)
    asm_volatile_inline(f"tl_sem_prio_async_copy_wait;\n"
                         f"commit_group;")
    
    c_Q0 = desc_Q.load(p_Q0)
    c_Q1 = desc_Q.load(p_Q1)
    c_O0 = desc_O.load(p_O0)
    c_O1 = desc_O.load(p_O1)
    c_dO0 = desc_dO.load(p_dO0)
    c_dO1 = desc_dO.load(p_dO1)
    
    D = (c_dO0 * c_O0).sum(axis=1) + (c_dO1 * c_O1).sum(axis=1)
    
    row_idx = tl.arange(0, BLOCK_M)
    valid_rows = (i * BLOCK_M + row_idx) < S_len
    L_i = tl.load(L_ptr + bh * S_len + i * BLOCK_M + row_idx, mask=valid_rows, other=0.0)
    
    dQ0 = tl.zeros((BLOCK_M, 64), tl.float32)
    dQ1 = tl.zeros((BLOCK_M, 64), tl.float32)
    
    p_K0 = init_tma_payload()
    p_K0 = p_K0.extend(desc_K.start_load(p_K0))
    p_K1 = init_tma_payload()
    p_K1 = p_K1.extend(desc_K.start_load(p_K1))
    p_V0 = init_tma_payload()
    p_V0 = p_V0.extend(desc_V.start_load(p_V0))
    p_V1 = init_tma_payload()
    p_V1 = p_V1.extend(desc_V.start_load(p_V1))
    
    asm_volatile_inline(f"cp_async_commit;", ...)
    
    p_cK0 = p_K0
    p_cK1 = p_K1
    p_cV0 = p_V0
    p_cV1 = p_V1
    
    p_nK0 = init_tma_payload()
    p_nK1 = init_tma_payload()
    p_nV0 = init_tma_payload()
    p_nV1 = init_tma_payload()
    
    cur_idx = 0
    next_idx = 1
    
    col_idx = j * BLOCK_N + tl.arange(0, BLOCK_N)
    
    for j in range(0, i + 1):
        next_j = j + 1
        
        if next_j <= i:
            p_nK0 = p_nK0.extend(desc_K.start_load(p_nK0))
            p_nK1 = p_nK1.extend(desc_K.start_load(p_nK1))
            p_nV0 = p_nV0.extend(desc_V.start_load(p_nV0))
            p_nV1 = p_nV1.extend(desc_V.start_load(p_nV1))
            
            asm_volatile_inline(f"cp_async_commit;", ...)
        
        asm_volatile_inline(f"tl_sem_prio_async_copy_wait;\n"
                             f"commit_group;")
        
        c_K0 = desc_K.load(p_cK0)
        c_K1 = desc_K.load(p_cK1)
        c_V0 = desc_V.load(p_cV0)
        c_V1 = desc_V.load(p_cV1)
        
        s = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        s = tl.dot(c_Q0, c_K0.T, acc=s)
        s = tl.dot(c_Q1, c_K1.T, acc=s)
        
        dp = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        dp = tl.dot(c_dO0, c_V0.T, acc=dp)
        dp = tl.dot(c_dO1, c_V1.T, acc=dp)
        
        p = tl.exp(s * scale - L_i[:, None])
        ds = p * (dp - D[:, None]) * scale
        
        ds = ds * ((i * BLOCK_M + row_idx[:, None]) >= (j * BLOCK_N + col_idx[None, :]))
        
        dQ0 = tl.dot(ds, c_K0, acc=dQ0)
        dQ1 = tl.dot(ds, c_K1, acc=dQ1)
        
        p_cK0, p_nK0 = p_nK0, p_cK0
        p_cK1, p_nK1 = p_nK1, p_cK1
        p_cV0, p_nV0 = p_nV0, p_cV0
        p_cV1, p_nV1 = p_nV1, p_cV1
        
        cur_idx = next_idx
        next_idx = 1 - next_idx
        
    desc_dQ.store([bh * S_len + i * BLOCK_M, 0], dQ0.to(tl.bfloat16))
    desc_dQ.store([bh * S_len + i * BLOCK_M, 64], dQ1.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    BLOCK_M = 128
    BLOCK_N = 128
    
    def make_desc_2d(t):
        t_flat = t.view(B * H * S, d)
        return t_flat
    
    Q_flat = make_desc_2d(Q)
    K_flat = make_desc_2d(K)
    V_flat = make_desc_2d(V)
    O_flat = make_desc_2d(O)
    dO_flat = make_desc_2d(dO)
    dQ_flat = make_desc_2d(dQ)
    dK_flat = make_desc_2d(dK)
    dV_flat = make_desc_2d(dV)
    
    grid = (B * H, triton.cdiv(S, BLOCK_M))
    
    bwd_dkdv_kernel[grid](
        Q_flat, K_flat, V_flat, O_flat, dO_flat, L, dK_flat, dV_flat,
        B, H, S, d, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4, num_stages=2)
        
    bwd_dq_kernel[grid](
        Q_flat, K_flat, V_flat, O_flat, dO_flat, L, dQ_flat,
        B, H, S, d, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4, num_stages=2)