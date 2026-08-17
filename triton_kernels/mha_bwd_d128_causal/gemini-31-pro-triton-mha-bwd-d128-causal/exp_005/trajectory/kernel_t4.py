import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def bwd_dq_kernel(
    Q_desc, K_desc, V_desc, O, dO_desc, dQ_desc, L,
    seq_len, scale,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)
    
    i_start = pid_s * BLOCK_S
    if i_start >= seq_len:
        return
        
    q_4d = Q_desc.load([pid_b, pid_h, i_start, 0])
    q = tl.reshape(q_4d, (BLOCK_S, HEAD_DIM))
    
    do_4d = dO_desc.load([pid_b, pid_h, i_start, 0])
    do = tl.reshape(do_4d, (BLOCK_S, HEAD_DIM))
    
    i_offs = i_start + tl.arange(0, BLOCK_S)
    i_mask = i_offs < seq_len
    offs_d = tl.arange(0, HEAD_DIM)
    
    # Load O and L using standard pointers (bypasses SMEM and dumps into registers directly)
    o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + i_offs[:, None] * stride_os + offs_d[None, :] * stride_od
    o = tl.load(o_ptrs, mask=i_mask[:, None], other=0.0)
    
    l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + i_offs * stride_ls
    l = tl.load(l_ptrs, mask=i_mask, other=0.0)
    
    # D_i = sum(O_i * dO_i), computed efficiently outside the inner loop
    d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    
    dq = tl.zeros((BLOCK_S, HEAD_DIM), dtype=tl.float32)
    
    num_j_unmasked = min(i_start, seq_len) // BLOCK_C
    
    # 1. Unmasked Dense Loop
    for j_blk in range(0, num_j_unmasked):
        j_start = j_blk * BLOCK_C
        
        k_4d = K_desc.load([pid_b, pid_h, j_start, 0])
        k = tl.reshape(k_4d, (BLOCK_C, HEAD_DIM))
        
        v_4d = V_desc.load([pid_b, pid_h, j_start, 0])
        v = tl.reshape(v_4d, (BLOCK_C, HEAD_DIM))
        
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        
        # Valid bounds masking only (bypass divergence overhead entirely)
        s = tl.where(i_mask[:, None], s, float('-inf'))
        
        p = tl.exp(s - l[:, None])
        p = tl.where(i_mask[:, None], p, 0.0)
        
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        ds = p * (dp - d_val[:, None]) * scale
        
        dq += tl.dot(ds.to(q.dtype), k, out_dtype=tl.float32)
        
    start_j_masked = num_j_unmasked
    end_j_masked = min((i_start + BLOCK_S + BLOCK_C - 1) // BLOCK_C, (seq_len + BLOCK_C - 1) // BLOCK_C)
    
    # 2. Causal Boundary Blocks
    for j_blk in range(start_j_masked, end_j_masked):
        j_start = j_blk * BLOCK_C
        j_offs = j_start + tl.arange(0, BLOCK_C)
        j_mask = j_offs < seq_len
        
        k_4d = K_desc.load([pid_b, pid_h, j_start, 0])
        k = tl.reshape(k_4d, (BLOCK_C, HEAD_DIM))
        
        v_4d = V_desc.load([pid_b, pid_h, j_start, 0])
        v = tl.reshape(v_4d, (BLOCK_C, HEAD_DIM))
        
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        
        mask = (i_offs[:, None] >= j_offs[None, :]) & i_mask[:, None] & j_mask[None, :]
        s = tl.where(mask, s, float('-inf'))
        
        p = tl.exp(s - l[:, None])
        p = tl.where(mask, p, 0.0)
        
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        ds = p * (dp - d_val[:, None]) * scale
        
        dq += tl.dot(ds.to(q.dtype), k, out_dtype=tl.float32)
        
    dQ_desc.store([pid_b, pid_h, i_start, 0], tl.reshape(dq.to(q.dtype), (1, 1, BLOCK_S, HEAD_DIM)))


@triton.jit
def bwd_dk_dv_kernel(
    Q_desc, K_desc, V_desc, O, dO_desc, dK_desc, dV_desc, L,
    seq_len, scale,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_c = tl.program_id(2)
    
    j_start = pid_c * BLOCK_C
    if j_start >= seq_len:
        return
        
    k_4d = K_desc.load([pid_b, pid_h, j_start, 0])
    k = tl.reshape(k_4d, (BLOCK_C, HEAD_DIM))
    
    v_4d = V_desc.load([pid_b, pid_h, j_start, 0])
    v = tl.reshape(v_4d, (BLOCK_C, HEAD_DIM))
    
    dk = tl.zeros((BLOCK_C, HEAD_DIM), dtype=tl.float32)
    dv = tl.zeros((BLOCK_C, HEAD_DIM), dtype=tl.float32)
    
    offs_d = tl.arange(0, HEAD_DIM)
    
    start_i_masked = j_start // BLOCK_S
    num_i_blocks = (seq_len + BLOCK_S - 1) // BLOCK_S
    end_i_masked = min((j_start + BLOCK_C + BLOCK_S - 1) // BLOCK_S, num_i_blocks)
    
    j_offs = j_start + tl.arange(0, BLOCK_C)
    j_mask = j_offs < seq_len
    
    # 1. Causal Boundary Blocks 
    for i_blk in range(start_i_masked, end_i_masked):
        i_start = i_blk * BLOCK_S
        i_offs = i_start + tl.arange(0, BLOCK_S)
        i_mask = i_offs < seq_len
        
        q_4d = Q_desc.load([pid_b, pid_h, i_start, 0])
        q = tl.reshape(q_4d, (BLOCK_S, HEAD_DIM))
        
        do_4d = dO_desc.load([pid_b, pid_h, i_start, 0])
        do = tl.reshape(do_4d, (BLOCK_S, HEAD_DIM))
        
        o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + i_offs[:, None] * stride_os + offs_d[None, :] * stride_od
        o = tl.load(o_ptrs, mask=i_mask[:, None], other=0.0)
        
        l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + i_offs * stride_ls
        l = tl.load(l_ptrs, mask=i_mask, other=0.0)
        
        d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        
        mask = (i_offs[:, None] >= j_offs[None, :]) & i_mask[:, None] & j_mask[None, :]
        s = tl.where(mask, s, float('-inf'))
        
        p = tl.exp(s - l[:, None])
        p = tl.where(mask, p, 0.0)
        
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        ds = p * (dp - d_val[:, None]) * scale
        
        dk += tl.dot(tl.trans(ds.to(q.dtype)), q, out_dtype=tl.float32)
        dv += tl.dot(tl.trans(p.to(q.dtype)), do, out_dtype=tl.float32)
        
    # 2. Unmasked Dense Loop
    for i_blk in range(end_i_masked, num_i_blocks):
        i_start = i_blk * BLOCK_S
        i_offs = i_start + tl.arange(0, BLOCK_S)
        i_mask = i_offs < seq_len
        
        q_4d = Q_desc.load([pid_b, pid_h, i_start, 0])
        q = tl.reshape(q_4d, (BLOCK_S, HEAD_DIM))
        
        do_4d = dO_desc.load([pid_b, pid_h, i_start, 0])
        do = tl.reshape(do_4d, (BLOCK_S, HEAD_DIM))
        
        o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + i_offs[:, None] * stride_os + offs_d[None, :] * stride_od
        o = tl.load(o_ptrs, mask=i_mask[:, None], other=0.0)
        
        l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + i_offs * stride_ls
        l = tl.load(l_ptrs, mask=i_mask, other=0.0)
        
        d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        s = tl.where(i_mask[:, None], s, float('-inf'))
        
        p = tl.exp(s - l[:, None])
        p = tl.where(i_mask[:, None], p, 0.0)
        
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        ds = p * (dp - d_val[:, None]) * scale
        
        dk += tl.dot(tl.trans(ds.to(q.dtype)), q, out_dtype=tl.float32)
        dv += tl.dot(tl.trans(p.to(q.dtype)), do, out_dtype=tl.float32)
        
    dK_desc.store([pid_b, pid_h, j_start, 0], tl.reshape(dk.to(k.dtype), (1, 1, BLOCK_C, HEAD_DIM)))
    dV_desc.store([pid_b, pid_h, j_start, 0], tl.reshape(dv.to(k.dtype), (1, 1, BLOCK_C, HEAD_DIM)))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    with torch.cuda.device(Q.device):
        B, H, seq_len, HEAD_DIM = Q.shape
        scale = 1.0 / math.sqrt(HEAD_DIM)
        
        BLOCK_S = 128
        BLOCK_C = 128
        
        Q_desc = TensorDescriptor.from_tensor(Q, (1, 1, BLOCK_S, HEAD_DIM))
        K_desc = TensorDescriptor.from_tensor(K, (1, 1, BLOCK_C, HEAD_DIM))
        V_desc = TensorDescriptor.from_tensor(V, (1, 1, BLOCK_C, HEAD_DIM))
        dO_desc = TensorDescriptor.from_tensor(dO, (1, 1, BLOCK_S, HEAD_DIM))
        dQ_desc = TensorDescriptor.from_tensor(dQ, (1, 1, BLOCK_S, HEAD_DIM))
        dK_desc = TensorDescriptor.from_tensor(dK, (1, 1, BLOCK_C, HEAD_DIM))
        dV_desc = TensorDescriptor.from_tensor(dV, (1, 1, BLOCK_C, HEAD_DIM))
        
        grid_dq = (B, H, triton.cdiv(seq_len, BLOCK_S))
        bwd_dq_kernel[grid_dq](
            Q_desc, K_desc, V_desc, O, dO_desc, dQ_desc, L,
            seq_len, scale,
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_C=BLOCK_C, HEAD_DIM=HEAD_DIM,
            num_warps=8, num_stages=2
        )
        
        grid_dkdv = (B, H, triton.cdiv(seq_len, BLOCK_C))
        bwd_dk_dv_kernel[grid_dkdv](
            Q_desc, K_desc, V_desc, O, dO_desc, dK_desc, dV_desc, L,
            seq_len, scale,
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_C=BLOCK_C, HEAD_DIM=HEAD_DIM,
            num_warps=8, num_stages=2
        )
        
        return dQ, dK, dV