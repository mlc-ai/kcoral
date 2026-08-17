import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


# Hint for occupancy optimization given our register and shared-memory footprint
@triton.jit(__launch_bounds__(128))
def bwd_dkdv_kernel(descs, L_ptr, S, d, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    scale = 1.0 / math.sqrt(d)
    bh = tl.program_id(0)
    j = tl.program_id(1)
    
    K0 = descs['K'].load([bh * S + j * BLOCK_N, 0])
    K1 = descs['K'].load([bh * S + j * BLOCK_N, 64])
    V0 = descs['V'].load([bh * S + j * BLOCK_N, 0])
    V1 = descs['V'].load([bh * S + j * BLOCK_N, 64])
    
    dK0 = tl.zeros((BLOCK_N, 64), tl.float32)
    dK1 = tl.zeros((BLOCK_N, 64), tl.float32)
    dV0 = tl.zeros((BLOCK_N, 64), tl.float32)
    dV1 = tl.zeros((BLOCK_N, 64), tl.float32)
    
    cur_Q0 = shared::alloc("cur_Q0", BLOCK_M * 64 * 2, align=128)
    cur_Q1 = shared::alloc("cur_Q1", BLOCK_M * 64 * 2, align=128)
    cur_O0 = shared::alloc("cur_O0", BLOCK_M * 64 * 2, align=128)
    cur_O1 = shared::alloc("cur_O1", BLOCK_M * 64 * 2, align=128)
    cur_dO0 = shared::alloc("cur_dO0", BLOCK_M * 64 * 2, align=128)
    cur_dO1 = shared::alloc("cur_dO1", BLOCK_M * 64 * 2, align=128)
    next_Q0 = shared::alloc("next_Q0", BLOCK_M * 64 * 2, align=128)
    next_Q1 = shared::alloc("next_Q1", BLOCK_M * 64 * 2, align=128)
    next_O0 = shared::alloc("next_O0", BLOCK_M * 64 * 2, align=128)
    next_O1 = shared::alloc("next_O1", BLOCK_M * 64 * 2, align=128)
    next_dO0 = shared::alloc("next_dO0", BLOCK_M * 64 * 2, align=128)
    next_dO1 = shared::alloc("next_dO1", BLOCK_M * 64 * 2, align=128)
    
    num_Q_tiles = triton.cdiv(S, BLOCK_M)
    
    if j < num_Q_tiles:
        sem = tma.semantic_fence()
        tma.async_copy(descs['Q'], cur_Q0, (bh * S + j * BLOCK_M, 0), dependency=sem)
        tma.async_copy(descs['Q'], cur_Q1, (bh * S + j * BLOCK_M, 64), dependency=sem)
        tma.async_copy(descs['O'], cur_O0, (bh * S + j * BLOCK_M, 0), dependency=sem)
        tma.async_copy(descs['O'], cur_O1, (bh * S + j * BLOCK_M, 64), dependency=sem)
        tma.async_copy(descs['dO'], cur_dO0, (bh * S + j * BLOCK_M, 0), dependency=sem)
        tma.async_copy(descs['dO'], cur_dO1, (bh * S + j * BLOCK_M, 64), dependency=sem)
        sem.commit()
    
    for i in range(j, num_Q_tiles):
        next_i = i + 1
        if next_i < num_Q_tiles:
            sem = tma.semantic_fence()
            tma.async_copy(descs['Q'], next_Q0, (bh * S + next_i * BLOCK_M, 0), dependency=sem)
            tma.async_copy(descs['Q'], next_Q1, (bh * S + next_i * BLOCK_M, 64), dependency=sem)
            tma.async_copy(descs['O'], next_O0, (bh * S + next_i * BLOCK_M, 0), dependency=sem)
            tma.async_copy(descs['O'], next_O1, (bh * S + next_i * BLOCK_M, 64), dependency=sem)
            tma.async_copy(descs['dO'], next_dO0, (bh * S + next_i * BLOCK_M, 0), dependency=sem)
            tma.async_copy(descs['dO'], next_dO1, (bh * S + next_i * BLOCK_M, 64), dependency=sem)
            sem.commit()
            
        sem = tma.semantic_fence()
        tma.wait_commit(sem)
        sem.commit()
        
        c_Q0 = tl.reshape(cur_Q0, (BLOCK_M, 64))
        c_Q1 = tl.reshape(cur_Q1, (BLOCK_M, 64))
        c_O0 = tl.reshape(cur_O0, (BLOCK_M, 64))
        c_O1 = tl.reshape(cur_O1, (BLOCK_M, 64))
        c_dO0 = tl.reshape(cur_dO0, (BLOCK_M, 64))
        c_dO1 = tl.reshape(cur_dO1, (BLOCK_M, 64))
        
        D = (c_dO0.to(tl.float32) * c_O0.to(tl.float32)).sum(axis=1) + \
            (c_dO1.to(tl.float32) * c_O1.to(tl.float32)).sum(axis=1)
        
        row_idx = tl.arange(0, BLOCK_M)
        valid_rows = (i * BLOCK_M + row_idx) < S
        L_i = tl.load(L_ptr + bh * S + i * BLOCK_M + row_idx, mask=valid_rows, other=0.0)
        
        c_K0 = tl.reshape(K0, (BLOCK_N, 64))
        c_K1 = tl.reshape(K1, (BLOCK_N, 64))
        c_V0 = tl.reshape(V0, (BLOCK_N, 64))
        c_V1 = tl.reshape(V1, (BLOCK_N, 64))
        
        s = tl.dot(c_Q0.to(tl.float32), c_K0.to(tl.float32).T)
        s = tl.dot(c_Q1.to(tl.float32), c_K1.to(tl.float32).T, acc=s)
        
        dp = tl.dot(c_dO0.to(tl.float32), c_V0.to(tl.float32).T)
        dp = tl.dot(c_dO1.to(tl.float32), c_V1.to(tl.float32).T, acc=dp)
        
        p = tl.exp(s * scale - L_i[:, None])
        ds = p * (dp - D[:, None]) * scale
        
        ds = ds * ((i * BLOCK_M + row_idx[:, None]) >= (j * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]))
        
        dK0 += tl.dot(ds.T, c_Q0.to(tl.float32))
        dK1 += tl.dot(ds.T, c_Q1.to(tl.float32))
        dV0 += tl.dot(p.T, c_dO0.to(tl.float32))
        dV1 += tl.dot(p.T, c_dO1.to(tl.float32))
        
        cur_Q0, next_Q0 = next_Q0, cur_Q0
        cur_Q1, next_Q1 = next_Q1, cur_Q1
        cur_O0, next_O0 = next_O0, cur_O0
        cur_O1, next_O1 = next_O1, cur_O1
        cur_dO0, next_dO0 = next_dO0, cur_dO0
        cur_dO1, next_dO1 = next_dO1, cur_dO1
        
    sem_out = tma.semantic_fence()
    tma.store(descs['dK'], dK0.to(tl.bfloat16), (bh * S + j * BLOCK_N, 0), dependency=sem_out)
    tma.store(descs['dK'], dK1.to(tl.bfloat16), (bh * S + j * BLOCK_N, 64), dependency=sem_out)
    tma.store(descs['dV'], dV0.to(tl.bfloat16), (bh * S + j * BLOCK_N, 0), dependency=sem_out)
    tma.store(descs['dV'], dV1.to(tl.bfloat16), (bh * S + j * BLOCK_N, 64), dependency=sem_out)
    sem_out.commit()


# Hint for occupancy optimization
@triton.jit(__launch_bounds__(128))
def bwd_dq_kernel(descs, L_ptr, S, d, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    scale = 1.0 / math.sqrt(d)
    bh = tl.program_id(0)
    i = tl.program_id(1)
    
    cur_Q0 = shared::alloc("cur_Q0", BLOCK_M * 64 * 2, align=128)
    cur_Q1 = shared::alloc("cur_Q1", BLOCK_M * 64 * 2, align=128)
    cur_O0 = shared::alloc("cur_O0", BLOCK_M * 64 * 2, align=128)
    cur_O1 = shared::alloc("cur_O1", BLOCK_M * 64 * 2, align=128)
    cur_dO0 = shared::alloc("cur_dO0", BLOCK_M * 64 * 2, align=128)
    cur_dO1 = shared::alloc("cur_dO1", BLOCK_M * 64 * 2, align=128)
    
    sem = tma.semantic_fence()
    tma.async_copy(descs['Q'], cur_Q0, (bh * S + i * BLOCK_M, 0), dependency=sem)
    tma.async_copy(descs['Q'], cur_Q1, (bh * S + i * BLOCK_M, 64), dependency=sem)
    tma.async_copy(descs['O'], cur_O0, (bh * S + i * BLOCK_M, 0), dependency=sem)
    tma.async_copy(descs['O'], cur_O1, (bh * S + i * BLOCK_M, 64), dependency=sem)
    tma.async_copy(descs['dO'], cur_dO0, (bh * S + i * BLOCK_M, 0), dependency=sem)
    tma.async_copy(descs['dO'], cur_dO1, (bh * S + i * BLOCK_M, 64), dependency=sem)
    sem.commit()
    
    sem = tma.semantic_fence()
    tma.wait_commit(sem)
    sem.commit()
    
    c_Q0 = tl.reshape(cur_Q0, (BLOCK_M, 64))
    c_Q1 = tl.reshape(cur_Q1, (BLOCK_M, 64))
    c_O0 = tl.reshape(cur_O0, (BLOCK_M, 64))
    c_O1 = tl.reshape(cur_O1, (BLOCK_M, 64))
    c_dO0 = tl.reshape(cur_dO0, (BLOCK_M, 64))
    c_dO1 = tl.reshape(cur_dO1, (BLOCK_M, 64))
    
    D = (c_dO0.to(tl.float32) * c_O0.to(tl.float32)).sum(axis=1) + \
        (c_dO1.to(tl.float32) * c_O1.to(tl.float32)).sum(axis=1)
    
    row_idx = tl.arange(0, BLOCK_M)
    valid_rows = (i * BLOCK_M + row_idx) < S
    L_i = tl.load(L_ptr + bh * S + i * BLOCK_M + row_idx, mask=valid_rows, other=0.0)
    
    dQ0 = tl.zeros((BLOCK_M, 64), tl.float32)
    dQ1 = tl.zeros((BLOCK_M, 64), tl.float32)
    
    cur_K0 = shared::alloc("cur_K0", BLOCK_N * 64 * 2, align=128)
    cur_K1 = shared::alloc("cur_K1", BLOCK_N * 64 * 2, align=128)
    cur_V0 = shared::alloc("cur_V0", BLOCK_N * 64 * 2, align=128)
    cur_V1 = shared::alloc("cur_V1", BLOCK_N * 64 * 2, align=128)
    next_K0 = shared::alloc("next_K0", BLOCK_N * 64 * 2, align=128)
    next_K1 = shared::alloc("next_K1", BLOCK_N * 64 * 2, align=128)
    next_V0 = shared::alloc("next_V0", BLOCK_N * 64 * 2, align=128)
    next_V1 = shared::alloc("next_V1", BLOCK_N * 64 * 2, align=128)
    
    max_j = min(i + 1, triton.cdiv(S, BLOCK_N))
    
    if 0 < max_j:
        sem = tma.semantic_fence()
        tma.async_copy(descs['K'], cur_K0, (bh * S + 0 * BLOCK_N, 0), dependency=sem)
        tma.async_copy(descs['K'], cur_K1, (bh * S + 0 * BLOCK_N, 64), dependency=sem)
        tma.async_copy(descs['V'], cur_V0, (bh * S + 0 * BLOCK_N, 0), dependency=sem)
        tma.async_copy(descs['V'], cur_V1, (bh * S + 0 * BLOCK_N, 64), dependency=sem)
        sem.commit()
        
    for j in range(0, max_j):
        next_j = j + 1
        if next_j < max_j:
            sem = tma.semantic_fence()
            tma.async_copy(descs['K'], next_K0, (bh * S + next_j * BLOCK_N, 0), dependency=sem)
            tma.async_copy(descs['K'], next_K1, (bh * S + next_j * BLOCK_N, 64), dependency=sem)
            tma.async_copy(descs['V'], next_V0, (bh * S + next_j * BLOCK_N, 0), dependency=sem)
            tma.async_copy(descs['V'], next_V1, (bh * S + next_j * BLOCK_N, 64), dependency=sem)
            sem.commit()
            
        sem = tma.semantic_fence()
        tma.wait_commit(sem)
        sem.commit()
        
        c_K0 = tl.reshape(cur_K0, (BLOCK_N, 64))
        c_K1 = tl.reshape(cur_K1, (BLOCK_N, 64))
        c_V0 = tl.reshape(cur_V0, (BLOCK_N, 64))
        c_V1 = tl.reshape(cur_V1, (BLOCK_N, 64))
        
        s = tl.dot(c_Q0.to(tl.float32), c_K0.to(tl.float32).T)
        s = tl.dot(c_Q1.to(tl.float32), c_K1.to(tl.float32).T, acc=s)
        
        dp = tl.dot(c_dO0.to(tl.float32), c_V0.to(tl.float32).T)
        dp = tl.dot(c_dO1.to(tl.float32), c_V1.to(tl.float32).T, acc=dp)
        
        p = tl.exp(s * scale - L_i[:, None])
        ds = p * (dp - D[:, None]) * scale
        
        ds = ds * ((i * BLOCK_M + row_idx[:, None]) >= (j * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]))
        
        dQ0 += tl.dot(ds, c_K0.to(tl.float32))
        dQ1 += tl.dot(ds, c_K1.to(tl.float32))
        
        cur_K0, next_K0 = next_K0, cur_K0
        cur_K1, next_K1 = next_K1, cur_K1
        cur_V0, next_V0 = next_V0, cur_V0
        cur_V1, next_V1 = next_V1, cur_V1
        
    sem_out = tma.semantic_fence()
    tma.store(descs['dQ'], dQ0.to(tl.bfloat16), (bh * S + i * BLOCK_M, 0), dependency=sem_out)
    tma.store(descs['dQ'], dQ1.to(tl.bfloat16), (bh * S + i * BLOCK_M, 64), dependency=sem_out)
    sem_out.commit()


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
    
    descs = {
        'Q': Q_desc, 'K': K_desc, 'V': V_desc, 'O': O_desc,
        'dO': dO_desc, 'dQ': dQ_desc, 'dK': dK_desc, 'dV': dV_desc
    }
    
    grid = (B * H, triton.cdiv(S, BLOCK_M))
    
    bwd_dkdv_kernel[grid](descs, L, S, d, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4, num_stages=2)
    bwd_dq_kernel[grid](descs, L, S, d, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4, num_stages=2)