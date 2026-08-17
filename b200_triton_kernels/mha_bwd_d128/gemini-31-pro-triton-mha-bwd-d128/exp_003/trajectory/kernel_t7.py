import torch
import triton
import triton.language as tl

# Set allocator for Triton to correctly create device-side TMA descriptors
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)


@triton.jit
def clear_tensor(ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements
    tl.store(ptr + idx, 0.0, mask=mask)


@triton.jit
def bwd_kernel(
    Q, K, V, dO, O, L, dQ, dK, dV,
    S, scale,
    q_st_b, q_st_h, q_st_s, q_st_d,
    k_st_b, k_st_h, k_st_s, k_st_d,
    v_st_b, v_st_h, v_st_s, v_st_d,
    do_st_b, do_st_h, do_st_s, do_st_d,
    o_st_b, o_st_h, o_st_s, o_st_d,
    l_st_b, l_st_h, l_st_s,
    dq_st_b, dq_st_h, dq_st_s, dq_st_d,
    dk_st_b, dk_st_h, dk_st_s, dk_st_d,
    dv_st_b, dv_st_h, dv_st_s, dv_st_d,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    off_b = pid_bh // H
    off_h = pid_bh % H
    
    off_n = pid_n * BLOCK_N
    
    # Base pointers offset by Batch and Head
    q_base = Q + off_b * q_st_b + off_h * q_st_h
    k_base = K + off_b * k_st_b + off_h * k_st_h
    v_base = V + off_b * v_st_b + off_h * v_st_h
    do_base = dO + off_b * do_st_b + off_h * do_st_h
    o_base = O + off_b * o_st_b + off_h * o_st_h
    
    # Hopper TMA descriptors creation
    q_desc = tl.make_tensor_descriptor(q_base, shape=[S, 128], strides=[q_st_s, q_st_d], block_shape=[BLOCK_M, 128], padding_option="zero")
    do_desc = tl.make_tensor_descriptor(do_base, shape=[S, 128], strides=[do_st_s, do_st_d], block_shape=[BLOCK_M, 128], padding_option="zero")
    o_desc = tl.make_tensor_descriptor(o_base, shape=[S, 128], strides=[o_st_s, o_st_d], block_shape=[BLOCK_M, 128], padding_option="zero")
    k_desc = tl.make_tensor_descriptor(k_base, shape=[S, 128], strides=[k_st_s, k_st_d], block_shape=[BLOCK_N, 128], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(v_base, shape=[S, 128], strides=[v_st_s, v_st_d], block_shape=[BLOCK_N, 128], padding_option="zero")
    
    # K and V are strictly loaded once for this N-block, perfectly utilizing Shared Memory WGMMA operands.
    k = tl.load(k_desc, [off_n, 0])
    v = tl.load(v_desc, [off_n, 0])
    
    dk_acc = tl.zeros((BLOCK_N, 128), tl.float32)
    dv_acc = tl.zeros((BLOCK_N, 128), tl.float32)
    
    l_base = L + off_b * l_st_b + off_h * l_st_h
    
    offs_n = off_n + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    
    dq_base = dQ + off_b * dq_st_b + off_h * dq_st_h
    off_d = tl.arange(0, 128)
    mask_d = off_d < 128
    
    num_m_blocks = tl.cdiv(S, BLOCK_M)
    
    # Loop offset purely affine monotonically to retain zero-cost software TMA pipelining mapping 
    for i in range(num_m_blocks):
        off_m = i * BLOCK_M
        
        q = tl.load(q_desc, [off_m, 0])
        do = tl.load(do_desc, [off_m, 0])
        o = tl.load(o_desc, [off_m, 0])
        
        offs_m = off_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        
        lse = tl.load(l_base + offs_m * l_st_s, mask=mask_m, other=0.0)
        
        # Q @ K.T
        s = tl.dot(q, k.T, out_dtype=tl.float32) * scale
        
        mask = mask_m[:, None] & mask_n[None, :]
        s = tl.where(mask, s, float('-inf'))
        
        p = tl.exp(s - lse[:, None])
        p = tl.where(mask, p, 0.0)
        
        # dV += P^T @ dO
        dv_acc = tl.dot(tl.trans(p).to(q.dtype), do, dv_acc, out_dtype=tl.float32)
        
        D = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
        
        # dP = dO @ V.T
        dp = tl.dot(do, v.T, out_dtype=tl.float32)
        
        ds = p * (dp - D[:, None]) * scale
        
        # dK += dS^T @ Q
        dk_acc = tl.dot(tl.trans(ds).to(q.dtype), q, dk_acc, out_dtype=tl.float32)
        
        # dQ += dS @ K 
        dq_inc = tl.dot(ds.to(q.dtype), k, out_dtype=tl.float32)
        
        # Immediate native L2 Atomic Add to dQ 
        dq_ptrs = dq_base + offs_m[:, None] * dq_st_s + off_d[None, :] * dq_st_d
        mask_dq = mask_m[:, None] & mask_d[None, :]
        tl.atomic_add(dq_ptrs, dq_inc.to(q.dtype), mask=mask_dq, sem="relaxed")
        
    dk_base = dK + off_b * dk_st_b + off_h * dk_st_h
    dv_base = dV + off_b * dv_st_b + off_h * dv_st_h
    dk_ptrs = dk_base + offs_n[:, None] * dk_st_s + off_d[None, :] * dk_st_d
    dv_ptrs = dv_base + offs_n[:, None] * dv_st_s + off_d[None, :] * dv_st_d
    
    mask_nd = mask_n[:, None] & mask_d[None, :]
    tl.store(dk_ptrs, dk_acc.to(k.dtype), mask=mask_nd)
    tl.store(dv_ptrs, dv_acc.to(v.dtype), mask=mask_nd)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes the FlashAttention backward pass natively in Triton on NVIDIA Hopper SM90/SM90a GPUs.
    
    Architecture constraints respected:
      - Strictly affine index pipelining in the inner loop avoiding pipeline break stalls
      - TMA descriptor creation cleanly executed
      - Maximum possible SMEM footprint (208KB <= 228KB limits) using 1-pass execution
      - Reduced footprint layout (BLOCK_M=64, BLOCK_N=128) halving necessary L2 dQ atomic traffic.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    scale = 1.0 / (d ** 0.5)
    
    q_st_b, q_st_h, q_st_s, q_st_d = Q.stride()
    k_st_b, k_st_h, k_st_s, k_st_d = K.stride()
    v_st_b, v_st_h, v_st_s, v_st_d = V.stride()
    do_st_b, do_st_h, do_st_s, do_st_d = dO.stride()
    o_st_b, o_st_h, o_st_s, o_st_d = O.stride()
    
    if L.dim() == 4:
        l_st_b, l_st_h, l_st_s, _ = L.stride()
    else:
        l_st_b, l_st_h, l_st_s = L.stride()
        
    dq_st_b, dq_st_h, dq_st_s, dq_st_d = dQ.stride()
    dk_st_b, dk_st_h, dk_st_s, dk_st_d = dK.stride()
    dv_st_b, dv_st_h, dv_st_s, dv_st_d = dV.stride()

    # dQ accumulates sequentially using fast native Hopper atomics; clear tensor beforehand
    n_elements = dQ.numel()
    grid_clear = (triton.cdiv(n_elements, 1024),)
    clear_tensor[grid_clear](dQ, n_elements, BLOCK_SIZE=1024)

    # BLOCK_M=64, BLOCK_N=128 is mathematically proven to incur exactly half the atomic 
    # memory writes compared to the symmetric 128x64 arrangement.
    BLOCK_M = 64
    BLOCK_N = 128
    
    grid = (triton.cdiv(S, BLOCK_N), B * H)
    
    bwd_kernel[grid](
        Q, K, V, dO, O, L, dQ, dK, dV,
        S, scale,
        q_st_b, q_st_h, q_st_s, q_st_d,
        k_st_b, k_st_h, k_st_s, k_st_d,
        v_st_b, v_st_h, v_st_s, v_st_d,
        do_st_b, do_st_h, do_st_s, do_st_d,
        o_st_b, o_st_h, o_st_s, o_st_d,
        l_st_b, l_st_h, l_st_s,
        dq_st_b, dq_st_h, dq_st_s, dq_st_d,
        dk_st_b, dk_st_h, dk_st_s, dk_st_d,
        dv_st_b, dv_st_h, dv_st_s, dv_st_d,
        H=H,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=8, num_stages=3
    )