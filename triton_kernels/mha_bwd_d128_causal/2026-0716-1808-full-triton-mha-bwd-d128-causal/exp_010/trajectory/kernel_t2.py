import ctypes
import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def load_gmem(base_ptr, start_row, S_len):
    row_idx = tl.program_id(1)  # Using program_id to get row index context implicitly via layout logic
    offsets = start_row + row_idx
    val = tl.load(base_ptr + offsets, mask=row_idx < S_len, other=0.0)
    return val


@triton.jit
def bwd_dkdv_kernel(
    Q_desc_ptr, K_desc_ptr, V_desc_ptr, O_desc_ptr, dO_desc_ptr,
    L_ptr, dK_desc_ptr, dV_desc_ptr,
    B, H, S, d, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    ext = tlt.define_shared_storage(
        tlt.var("Q0",  BLOCK_M * 64 * 2, dtype=Q_desc_ptr.dtype, align=128),
        tlt.var("Q1",  BLOCK_M * 64 * 2, dtype=Q_desc_ptr.dtype, align=128),
        tlt.var("O0",  BLOCK_M * 64 * 2, dtype=Q_desc_ptr.dtype, align=128),
        tlt.var("O1",  BLOCK_M * 64 * 2, dtype=Q_desc_ptr.dtype, align=128),
        tlt.var("dO0", BLOCK_M * 64 * 2, dtype=Q_desc_ptr.dtype, align=128),
        tlt.var("dO1", BLOCK_M * 64 * 2, dtype=Q_desc_ptr.dtype, align=128),
        tlt.var("K0",  BLOCK_N * 64,     dtype=K_desc_ptr.dtype, align=128),
        tlt.var("K1",  BLOCK_N * 64,     dtype=K_desc_ptr.dtype, align=128),
        tlt.var("V0",  BLOCK_N * 64,     dtype=V_desc_ptr.dtype, align=128),
        tlt.var("V1",  BLOCK_N * 64,     dtype=V_desc_ptr.dtype, align=128),
    )
    
    scale = 1.0 / math.sqrt(d)
    bh = tl.program_id(0)
    j = tl.program_id(1)
    S_len = S
    
    desc_Q = Q_desc_ptr[0]
    desc_K = K_desc_ptr[0]
    desc_V = V_desc_ptr[0]
    desc_O = O_desc_ptr[0]
    desc_dO = dO_desc_ptr[0]
    desc_dK = dK_desc_ptr[0]
    desc_dV = dV_desc_ptr[0]
    
    load_K0 = ext.K0
    load_K1 = ext.K1
    load_V0 = ext.V0
    load_V1 = ext.V1
    
    sem = tl.semantic_fence()
    tma.async_copy(desc_Q, load_K0, (bh * S_len + j * BLOCK_N, 0),  dependency=sem)
    tma.async_copy(desc_Q, load_K1, (bh * S_len + j * BLOCK_N, 64), dependency=sem)
    tma.async_copy(desc_Q, load_V0, (bh * S_len + j * BLOCK_N, 0),  dependency=sem)
    tma.async_copy(desc_Q, load_V1, (bh * S_len + j * BLOCK_N, 64), dependency=sem)
    sem.commit()
    tl.wait_async_group()
    
    dK0 = tl.zeros((BLOCK_N, 64), tl.float32)
    dK1 = tl.zeros((BLOCK_N, 64), tl.float32)
    dV0 = tl.zeros((BLOCK_N, 64), tl.float32)
    dV1 = tl.zeros((BLOCK_N, 64), tl.float32)
    
    num_Q_tiles = (S_len + BLOCK_M - 1) // BLOCK_M
    cur_idx = 0
    
    if j < num_Q_tiles:
        load_Q0 = ext.Q0[cur_idx * 8192]
        load_Q1 = ext.Q1[cur_idx * 8192]
        load_O0 = ext.O0[cur_idx * 8192]
        load_O1 = ext.O1[cur_idx * 8192]
        load_dO0 = ext.dO0[cur_idx * 8192]
        load_dO1 = ext.dO1[cur_idx * 8192]
        
        sem = tl.semantic_fence()
        tma.async_copy(desc_Q, load_Q0, (bh * S_len + j * BLOCK_M, 0),  dependency=sem)
        tma.async_copy(desc_Q, load_Q1, (bh * S_len + j * BLOCK_M, 64), dependency=sem)
        tma.async_copy(desc_Q, load_O0, (bh * S_len + j * BLOCK_M, 0),  dependency=sem)
        tma.async_copy(desc_Q, load_O1, (bh * S_len + j * BLOCK_M, 64), dependency=sem)
        tma.async_copy(desc_Q, load_dO0, (bh * S_len + j * BLOCK_M, 0),  dependency=sem)
        tma.async_copy(desc_Q, load_dO1, (bh * S_len + j * BLOCK_M, 64), dependency=sem)
        sem.commit()
    
    row_idx = tl.arange(0, BLOCK_M)
    
    for i in range(j, num_Q_tiles):
        next_i = i + 1
        next_idx = 1 - cur_idx
        
        if next_i < num_Q_tiles:
            load_nQ0 = ext.Q0[next_idx * 8192]
            load_nQ1 = ext.Q1[next_idx * 8192]
            load_nO0 = ext.O0[next_idx * 8192]
            load_nO1 = ext.O1[next_idx * 8192]
            load_ndO0 = ext.dO0[next_idx * 8192]
            load_ndO1 = ext.dO1[next_idx * 8192]
            
            sem = tl.semantic_fence()
            tma.async_copy(desc_Q, load_nQ0, (bh * S_len + next_i * BLOCK_M, 0),  dependency=sem)
            tma.async_copy(desc_Q, load_nQ1, (bh * S_len + next_i * BLOCK_M, 64), dependency=sem)
            tma.async_copy(desc_Q, load_nO0, (bh * S_len + next_i * BLOCK_M, 0),  dependency=sem)
            tma.async_copy(desc_Q, load_nO1, (bh * S_len + next_i * BLOCK_M, 64), dependency=sem)
            tma.async_copy(desc_Q, load_ndO0, (bh * S_len + next_i * BLOCK_M, 0),  dependency=sem)
            tma.async_copy(desc_Q, load_ndO1, (bh * S_len + next_i * BLOCK_M, 64), dependency=sem)
            sem.commit()
        
        load_cQ0 = ext.Q0[cur_idx * 8192]
        load_cQ1 = ext.Q1[cur_idx * 8192]
        load_cO0 = ext.O0[cur_idx * 8192]
        load_cO1 = ext.O1[cur_idx * 8192]
        load_cdO0 = ext.dO0[cur_idx * 8192]
        load_cdO1 = ext.dO1[cur_idx * 8192]
        
        sem = tl.semantic_fence()
        tma.wait_commit(sem)
        sem.commit()
        tl.wait_async_group()
        
        c_Q0 = tl.reshape(load_cQ0, (BLOCK_M, 64))
        c_Q1 = tl.reshape(load_cQ1, (BLOCK_M, 64))
        c_O0 = tl.reshape(load_cO0, (BLOCK_M, 64))
        c_O1 = tl.reshape(load_cO1, (BLOCK_M, 64))
        c_dO0 = tl.reshape(load_cdO0, (BLOCK_M, 64))
        c_dO1 = tl.reshape(load_cdO1, (BLOCK_M, 64))
        
        D = (c_dO0.to(tl.float32) * c_O0.to(tl.float32)).sum(axis=1) + \
            (c_dO1.to(tl.float32) * c_O1.to(tl.float32)).sum(axis=1)
        
        valid_rows = (i * BLOCK_M + row_idx) < S_len
        L_i = load_gmem(L_ptr + bh * S_len, i * BLOCK_M, S_len)
        
        s = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        s = tl.dot(c_Q0, ext.K0.T, acc=s)
        s = tl.dot(c_Q1, ext.K1.T, acc=s)
        
        dp = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        dp = tl.dot(c_dO0, ext.V0.T, acc=dp)
        dp = tl.dot(c_dO1, ext.V1.T, acc=dp)
        
        p = tl.exp(s * scale - L_i[:, None])
        ds = p * (dp - D[:, None]) * scale
        
        ds = ds * ((i * BLOCK_M + row_idx[:, None]) >= (j * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]))
        
        dK0 = tl.dot(ds.T, c_Q0, acc=dK0)
        dK1 = tl.dot(ds.T, c_Q1, acc=dK1)
        dV0 = tl.dot(p.T, c_dO0, acc=dV0)
        dV1 = tl.dot(p.T, c_dO1, acc=dV1)
        
        ext.shared_mem_barrier()
        cur_idx = next_idx
        
    store_dK0 = ext.dK0
    store_dK1 = ext.dK1
    store_dV0 = ext.dV0
    store_dV1 = ext.dV1
    
    tma.store(store_dK0, dK0.to(tl.bfloat16), (bh * S_len + j * BLOCK_N, 0))
    tma.store(store_dK1, dK1.to(tl.bfloat16), (bh * S_len + j * BLOCK_N, 64))
    tma.store(store_dV0, dV0.to(tl.bfloat16), (bh * S_len + j * BLOCK_N, 0))
    tma.store(store_dV1, dV1.to(tl.bfloat16), (bh * S_len + j * BLOCK_N, 64))


@triton.jit
def bwd_dq_kernel(
    Q_desc_ptr, K_desc_ptr, V_desc_ptr, O_desc_ptr, dO_desc_ptr,
    L_ptr, dQ_desc_ptr,
    B, H, S, d, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    ext = tlt.define_shared_storage(
        tlt.var("Q0",  BLOCK_M * 64,      dtype=Q_desc_ptr.dtype, align=128),
        tlt.var("Q1",  BLOCK_M * 64,      dtype=Q_desc_ptr.dtype, align=128),
        tlt.var("O0",  BLOCK_M * 64,      dtype=Q_desc_ptr.dtype, align=128),
        tlt.var("O1",  BLOCK_M * 64,      dtype=Q_desc_ptr.dtype, align=128),
        tlt.var("dO0", BLOCK_M * 64,      dtype=Q_desc_ptr.dtype, align=128),
        tlt.var("dO1", BLOCK_M * 64,      dtype=Q_desc_ptr.dtype, align=128),
        tlt.var("K0",  BLOCK_N * 64 * 2,  dtype=K_desc_ptr.dtype, align=128),
        tlt.var("K1",  BLOCK_N * 64 * 2,  dtype=K_desc_ptr.dtype, align=128),
        tlt.var("V0",  BLOCK_N * 64 * 2,  dtype=V_desc_ptr.dtype, align=128),
        tlt.var("V1",  BLOCK_N * 64 * 2,  dtype=V_desc_ptr.dtype, align=128),
    )
    
    scale = 1.0 / math.sqrt(d)
    bh = tl.program_id(0)
    i = tl.program_id(1)
    S_len = S
    
    desc_Q = Q_desc_ptr[0]
    desc_K = K_desc_ptr[0]
    desc_V = V_desc_ptr[0]
    desc_O = O_desc_ptr[0]
    desc_dO = dO_desc_ptr[0]
    desc_dQ = dQ_desc_ptr[0]
    
    load_Q0 = ext.Q0
    load_Q1 = ext.Q1
    load_O0 = ext.O0
    load_O1 = ext.O1
    load_dO0 = ext.dO0
    load_dO1 = ext.dO1
    
    sem = tl.semantic_fence()
    tma.async_copy(desc_Q, load_Q0, (bh * S_len + i * BLOCK_M, 0),  dependency=sem)
    tma.async_copy(desc_Q, load_Q1, (bh * S_len + i * BLOCK_M, 64), dependency=sem)
    tma.async_copy(desc_Q, load_O0, (bh * S_len + i * BLOCK_M, 0),  dependency=sem)
    tma.async_copy(desc_Q, load_O1, (bh * S_len + i * BLOCK_M, 64), dependency=sem)
    tma.async_copy(desc_Q, load_dO0, (bh * S_len + i * BLOCK_M, 0),  dependency=sem)
    tma.async_copy(desc_Q, load_dO1, (bh * S_len + i * BLOCK_M, 64), dependency=sem)
    sem.commit()
    tl.wait_async_group()
    
    c_Q0 = tl.reshape(load_Q0, (BLOCK_M, 64))
    c_Q1 = tl.reshape(load_Q1, (BLOCK_M, 64))
    c_O0 = tl.reshape(load_O0, (BLOCK_M, 64))
    c_O1 = tl.reshape(load_O1, (BLOCK_M, 64))
    c_dO0 = tl.reshape(load_dO0, (BLOCK_M, 64))
    c_dO1 = tl.reshape(load_dO1, (BLOCK_M, 64))
    
    D = (c_dO0.to(tl.float32) * c_O0.to(tl.float32)).sum(axis=1) + \
        (c_dO1.to(tl.float32) * c_O1.to(tl.float32)).sum(axis=1)
    
    row_idx = tl.arange(0, BLOCK_M)
    valid_rows = (i * BLOCK_M + row_idx) < S_len
    L_i = load_gmem(L_ptr + bh * S_len, i * BLOCK_M, S_len)
    
    dQ0 = tl.zeros((BLOCK_M, 64), tl.float32)
    dQ1 = tl.zeros((BLOCK_M, 64), tl.float32)
    
    cur_idx = 0
    if 0 <= i:
        load_cK0 = ext.K0[cur_idx * 8192]
        load_cK1 = ext.K1[cur_idx * 8192]
        load_cV0 = ext.V0[cur_idx * 8192]
        load_cV1 = ext.V1[cur_idx * 8192]
        
        sem = tl.semantic_fence()
        tma.async_copy(desc_K, load_cK0, (bh * S_len + 0 * BLOCK_N, 0),  dependency=sem)
        tma.async_copy(desc_K, load_cK1, (bh * S_len + 0 * BLOCK_N, 64), dependency=sem)
        tma.async_copy(desc_V, load_cV0, (bh * S_len + 0 * BLOCK_N, 0),  dependency=sem)
        tma.async_copy(desc_V, load_cV1, (bh * S_len + 0 * BLOCK_N, 64), dependency=sem)
        sem.commit()
    
    for j in range(0, i + 1):
        next_j = j + 1
        next_idx = 1 - cur_idx
        
        if next_j <= i:
            load_nK0 = ext.K0[next_idx * 8192]
            load_nK1 = ext.K1[next_idx * 8192]
            load_nV0 = ext.V0[next_idx * 8192]
            load_nV1 = ext.V1[next_idx * 8192]
            
            sem = tl.semantic_fence()
            tma.async_copy(desc_K, load_nK0, (bh * S_len + next_j * BLOCK_N, 0),  dependency=sem)
            tma.async_copy(desc_K, load_nK1, (bh * S_len + next_j * BLOCK_N, 64), dependency=sem)
            tma.async_copy(desc_V, load_nV0, (bh * S_len + next_j * BLOCK_N, 0),  dependency=sem)
            tma.async_copy(desc_V, load_nV1, (bh * S_len + next_j * BLOCK_N, 64), dependency=sem)
            sem.commit()
        
        load_cK0 = ext.K0[cur_idx * 8192]
        load_cK1 = ext.K1[cur_idx * 8192]
        load_cV0 = ext.V0[cur_idx * 8192]
        load_cV1 = ext.V1[cur_idx * 8192]
        
        sem = tl.semantic_fence()
        tma.wait_commit(sem)
        sem.commit()
        tl.wait_async_group()
        
        c_K0 = tl.reshape(load_cK0, (BLOCK_N, 64))
        c_K1 = tl.reshape(load_cK1, (BLOCK_N, 64))
        c_V0 = tl.reshape(load_cV0, (BLOCK_N, 64))
        c_V1 = tl.reshape(load_cV1, (BLOCK_N, 64))
        
        s = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        s = tl.dot(c_Q0, c_K0.T, acc=s)
        s = tl.dot(c_Q1, c_K1.T, acc=s)
        
        dp = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        dp = tl.dot(c_dO0, c_V0.T, acc=dp)
        dp = tl.dot(c_dO1, c_V1.T, acc=dp)
        
        p = tl.exp(s * scale - L_i[:, None])
        ds = p * (dp - D[:, None]) * scale
        
        ds = ds * ((i * BLOCK_M + row_idx[:, None]) >= (j * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]))
        
        dQ0 = tl.dot(ds, c_K0, acc=dQ0)
        dQ1 = tl.dot(ds, c_K1, acc=dQ1)
        
        ext.shared_mem_barrier()
        cur_idx = next_idx
        
    store_dQ0 = ext.dQ0
    store_dQ1 = ext.dQ1
    
    tma.store(store_dQ0, dQ0.to(tl.bfloat16), (bh * S_len + i * BLOCK_M, 0))
    tma.store(store_dQ1, dQ1.to(tl.bfloat16), (bh * S_len + i * BLOCK_M, 64))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    BLOCK_M = 128
    BLOCK_N = 128
    
    def make_desc_2d(t):
        t_flat = t.view(B * H * S, d)
        return TensorDescriptor.from_tensor(t_flat, block_shape=(BLOCK_M, 64))
    
    Q_desc = make_desc_2d(Q)
    K_desc = make_desc_2d(K)
    V_desc = make_desc_2d(V)
    O_desc = make_desc_2d(O)
    dO_desc = make_desc_2d(dO)
    dQ_desc = make_desc_2d(dQ)
    dK_desc = make_desc_2d(dK)
    dV_desc = make_desc_2d(dV)
    
    grid = (B * H, triton.cdiv(S, BLOCK_M))
    
    bwd_dkdv_kernel[grid](
        (Q_desc.base_ptr, Q_desc), (K_desc.base_ptr, K_desc), (V_desc.base_ptr, V_desc), 
        (O_desc.base_ptr, O_desc), (dO_desc.base_ptr, dO_desc),
        L, (dK_desc.base_ptr, dK_desc), (dV_desc.base_ptr, dV_desc),
        B, H, S, d, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4, num_stages=4)
        
    bwd_dq_kernel[grid](
        (Q_desc.base_ptr, Q_desc), (K_desc.base_ptr, K_desc), (V_desc.base_ptr, V_desc), 
        (O_desc.base_ptr, O_desc), (dO_desc.base_ptr, dO_desc),
        L, (dQ_desc.base_ptr, dQ_desc),
        B, H, S, d, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4, num_stages=4)