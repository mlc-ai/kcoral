import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


@triton.jit
def bwd_dq_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, dQ_desc, L,
    seq_len, scale,
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
        
    q_blk = Q_desc.load([pid_b, pid_h, i_start, 0])
    q = tl.reshape(q_blk, (BLOCK_S, HEAD_DIM))
    
    o_blk = O_desc.load([pid_b, pid_h, i_start, 0])
    o = tl.reshape(o_blk, (BLOCK_S, HEAD_DIM))
    
    do_blk = dO_desc.load([pid_b, pid_h, i_start, 0])
    do = tl.reshape(do_blk, (BLOCK_S, HEAD_DIM))
    
    # Compute D_i on the fly in the outer loop (very cheap in this scope)
    d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    
    i_offs = i_start + tl.arange(0, BLOCK_S)
    i_mask = i_offs < seq_len
    
    # Direct pointer load for scalar layout vectors natively into registers
    l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + i_offs * stride_ls
    l = tl.load(l_ptrs, mask=i_mask, other=0.0)
    
    dq = tl.zeros((BLOCK_S, HEAD_DIM), dtype=tl.float32)
    
    max_j = tl.minimum(seq_len, i_start + BLOCK_S)
    num_j_unmasked = i_start // BLOCK_C
    num_j_blocks = tl.cdiv(max_j, BLOCK_C)
    
    # 1. Unmasked Dense Loop
    for j_blk in range(0, num_j_unmasked):
        j_start = j_blk * BLOCK_C
        
        k_blk = K_desc.load([pid_b, pid_h, j_start, 0])
        k = tl.reshape(k_blk, (BLOCK_C, HEAD_DIM))
        
        v_blk = V_desc.load([pid_b, pid_h, j_start, 0])
        v = tl.reshape(v_blk, (BLOCK_C, HEAD_DIM))
        
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        s = tl.where(i_mask[:, None], s, float('-inf'))
        
        p = tl.exp(s - l[:, None])
        p = tl.where(i_mask[:, None], p, 0.0)
        
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        ds = p * (dp - d_val[:, None]) * scale
        
        dq = tl.dot(ds.to(q.dtype), k, dq, out_dtype=tl.float32)
        
    # 2. Causal Boundary Masked Evaluation Blocks
    for j_blk in range(num_j_unmasked, num_j_blocks):
        j_start = j_blk * BLOCK_C
        j_offs = j_start + tl.arange(0, BLOCK_C)
        j_mask = j_offs < seq_len
        
        k_blk = K_desc.load([pid_b, pid_h, j_start, 0])
        k = tl.reshape(k_blk, (BLOCK_C, HEAD_DIM))
        
        v_blk = V_desc.load([pid_b, pid_h, j_start, 0])
        v = tl.reshape(v_blk, (BLOCK_C, HEAD_DIM))
        
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        
        mask = (i_offs[:, None] >= j_offs[None, :]) & i_mask[:, None] & j_mask[None, :]
        s = tl.where(mask, s, float('-inf'))
        
        p = tl.exp(s - l[:, None])
        p = tl.where(mask, p, 0.0)
        
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        ds = p * (dp - d_val[:, None]) * scale
        
        dq = tl.dot(ds.to(q.dtype), k, dq, out_dtype=tl.float32)
        
    dQ_desc.store([pid_b, pid_h, i_start, 0], tl.reshape(dq.to(q.dtype), (1, 1, BLOCK_S, HEAD_DIM)))


@triton.jit
def bwd_dk_dv_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, dK_desc, dV_desc, L,
    seq_len, scale,
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
        
    k_blk = K_desc.load([pid_b, pid_h, j_start, 0])
    k = tl.reshape(k_blk, (BLOCK_C, HEAD_DIM))
    
    v_blk = V_desc.load([pid_b, pid_h, j_start, 0])
    v = tl.reshape(v_blk, (BLOCK_C, HEAD_DIM))
    
    dk = tl.zeros((BLOCK_C, HEAD_DIM), dtype=tl.float32)
    dv = tl.zeros((BLOCK_C, HEAD_DIM), dtype=tl.float32)
    
    num_i_blocks = tl.cdiv(seq_len, BLOCK_S)
    start_i_masked = j_start // BLOCK_S
    end_i_masked = tl.minimum((j_start + BLOCK_C + BLOCK_S - 1) // BLOCK_S, num_i_blocks)
    
    j_offs = j_start + tl.arange(0, BLOCK_C)
    j_mask = j_offs < seq_len
    
    # 1. Causal Boundary Evaluation Masked Blocks
    for i_blk in range(start_i_masked, end_i_masked):
        i_start = i_blk * BLOCK_S
        i_offs = i_start + tl.arange(0, BLOCK_S)
        i_mask = i_offs < seq_len
        
        q_blk = Q_desc.load([pid_b, pid_h, i_start, 0])
        q = tl.reshape(q_blk, (BLOCK_S, HEAD_DIM))
        
        o_blk = O_desc.load([pid_b, pid_h, i_start, 0])
        o = tl.reshape(o_blk, (BLOCK_S, HEAD_DIM))
        
        do_blk = dO_desc.load([pid_b, pid_h, i_start, 0])
        do = tl.reshape(do_blk, (BLOCK_S, HEAD_DIM))
        
        d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        
        l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + i_offs * stride_ls
        l = tl.load(l_ptrs, mask=i_mask, other=0.0)
        
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        
        mask = (i_offs[:, None] >= j_offs[None, :]) & i_mask[:, None] & j_mask[None, :]
        s = tl.where(mask, s, float('-inf'))
        
        p = tl.exp(s - l[:, None])
        p = tl.where(mask, p, 0.0)
        
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        ds = p * (dp - d_val[:, None]) * scale
        
        dk = tl.dot(tl.trans(ds.to(q.dtype)), q, dk, out_dtype=tl.float32)
        dv = tl.dot(tl.trans(p.to(q.dtype)), do, dv, out_dtype=tl.float32)
        
    # 2. Unmasked Dense Loop
    for i_blk in range(end_i_masked, num_i_blocks):
        i_start = i_blk * BLOCK_S
        i_offs = i_start + tl.arange(0, BLOCK_S)
        i_mask = i_offs < seq_len
        
        q_blk = Q_desc.load([pid_b, pid_h, i_start, 0])
        q = tl.reshape(q_blk, (BLOCK_S, HEAD_DIM))
        
        o_blk = O_desc.load([pid_b, pid_h, i_start, 0])
        o = tl.reshape(o_blk, (BLOCK_S, HEAD_DIM))
        
        do_blk = dO_desc.load([pid_b, pid_h, i_start, 0])
        do = tl.reshape(do_blk, (BLOCK_S, HEAD_DIM))
        
        d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        
        l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + i_offs * stride_ls
        l = tl.load(l_ptrs, mask=i_mask, other=0.0)
        
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        s = tl.where(i_mask[:, None] & j_mask[None, :], s, float('-inf'))
        
        p = tl.exp(s - l[:, None])
        p = tl.where(i_mask[:, None] & j_mask[None, :], p, 0.0)
        
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        ds = p * (dp - d_val[:, None]) * scale
        
        dk = tl.dot(tl.trans(ds.to(q.dtype)), q, dk, out_dtype=tl.float32)
        dv = tl.dot(tl.trans(p.to(q.dtype)), do, dv, out_dtype=tl.float32)
        
    dK_desc.store([pid_b, pid_h, j_start, 0], tl.reshape(dk.to(k.dtype), (1, 1, BLOCK_C, HEAD_DIM)))
    dV_desc.store([pid_b, pid_h, j_start, 0], tl.reshape(dv.to(k.dtype), (1, 1, BLOCK_C, HEAD_DIM)))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    # Enforces proper Triton infrastructure capabilities allocation if triggered internally 
    triton.set_allocator(alloc_fn)
    
    with torch.cuda.device(Q.device):
        B, H, seq_len, HEAD_DIM = Q.shape
        scale = 1.0 / math.sqrt(HEAD_DIM)
        
        BLOCK_S_DQ = 128
        BLOCK_C_DQ = 128
        
        Q_desc = TensorDescriptor.from_tensor(Q, (1, 1, BLOCK_S_DQ, HEAD_DIM))
        O_desc = TensorDescriptor.from_tensor(O, (1, 1, BLOCK_S_DQ, HEAD_DIM))
        dO_desc = TensorDescriptor.from_tensor(dO, (1, 1, BLOCK_S_DQ, HEAD_DIM))
        dQ_desc = TensorDescriptor.from_tensor(dQ, (1, 1, BLOCK_S_DQ, HEAD_DIM))
        
        K_desc_dq = TensorDescriptor.from_tensor(K, (1, 1, BLOCK_C_DQ, HEAD_DIM))
        V_desc_dq = TensorDescriptor.from_tensor(V, (1, 1, BLOCK_C_DQ, HEAD_DIM))
        
        grid_dq = (B, H, triton.cdiv(seq_len, BLOCK_S_DQ))
        bwd_dq_kernel[grid_dq](
            Q_desc, K_desc_dq, V_desc_dq, O_desc, dO_desc, dQ_desc, L,
            seq_len, scale,
            L.stride(0), L.stride(1), L.stride(2),
            BLOCK_S=BLOCK_S_DQ, BLOCK_C=BLOCK_C_DQ, HEAD_DIM=HEAD_DIM,
            num_warps=8, num_stages=3
        )
        
        BLOCK_S_DK = 128
        BLOCK_C_DK = 64
        
        Q_desc_dk = TensorDescriptor.from_tensor(Q, (1, 1, BLOCK_S_DK, HEAD_DIM))
        O_desc_dk = TensorDescriptor.from_tensor(O, (1, 1, BLOCK_S_DK, HEAD_DIM))
        dO_desc_dk = TensorDescriptor.from_tensor(dO, (1, 1, BLOCK_S_DK, HEAD_DIM))
        
        K_desc_dk = TensorDescriptor.from_tensor(K, (1, 1, BLOCK_C_DK, HEAD_DIM))
        V_desc_dk = TensorDescriptor.from_tensor(V, (1, 1, BLOCK_C_DK, HEAD_DIM))
        dK_desc = TensorDescriptor.from_tensor(dK, (1, 1, BLOCK_C_DK, HEAD_DIM))
        dV_desc = TensorDescriptor.from_tensor(dV, (1, 1, BLOCK_C_DK, HEAD_DIM))
        
        grid_dkdv = (B, H, triton.cdiv(seq_len, BLOCK_C_DK))
        bwd_dk_dv_kernel[grid_dkdv](
            Q_desc_dk, K_desc_dk, V_desc_dk, O_desc_dk, dO_desc_dk, dK_desc, dV_desc, L,
            seq_len, scale,
            L.stride(0), L.stride(1), L.stride(2),
            BLOCK_S=BLOCK_S_DK, BLOCK_C=BLOCK_C_DK, HEAD_DIM=HEAD_DIM,
            num_warps=8, num_stages=2
        )