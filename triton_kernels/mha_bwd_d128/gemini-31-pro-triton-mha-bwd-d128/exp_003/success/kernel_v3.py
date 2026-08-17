import torch
import triton
import triton.language as tl

# Set allocator for Triton to create device-side TMA descriptors if needed by `tl.make_block_ptr`
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
    
    k_base = K + off_b * k_st_b + off_h * k_st_h
    v_base = V + off_b * v_st_b + off_h * v_st_h
    
    # Outer loop variables (K, V) loaded once per block
    k_block = tl.make_block_ptr(
        base=k_base, shape=(S, 128), strides=(k_st_s, k_st_d),
        offsets=(off_n, 0), block_shape=(BLOCK_N, 128), order=(1, 0)
    )
    v_block = tl.make_block_ptr(
        base=v_base, shape=(S, 128), strides=(v_st_s, v_st_d),
        offsets=(off_n, 0), block_shape=(BLOCK_N, 128), order=(1, 0)
    )
    
    k = tl.load(k_block, boundary_check=(0, 1), padding_option="zero")
    v = tl.load(v_block, boundary_check=(0, 1), padding_option="zero")
    
    dk_acc = tl.zeros((BLOCK_N, 128), tl.float32)
    dv_acc = tl.zeros((BLOCK_N, 128), tl.float32)
    
    q_base = Q + off_b * q_st_b + off_h * q_st_h
    do_base = dO + off_b * do_st_b + off_h * do_st_h
    o_base = O + off_b * o_st_b + off_h * o_st_h
    
    # Inner loop block pointers (Q, dO, O)
    q_block = tl.make_block_ptr(
        base=q_base, shape=(S, 128), strides=(q_st_s, q_st_d),
        offsets=(0, 0), block_shape=(BLOCK_M, 128), order=(1, 0)
    )
    do_block = tl.make_block_ptr(
        base=do_base, shape=(S, 128), strides=(do_st_s, do_st_d),
        offsets=(0, 0), block_shape=(BLOCK_M, 128), order=(1, 0)
    )
    o_block = tl.make_block_ptr(
        base=o_base, shape=(S, 128), strides=(o_st_s, o_st_d),
        offsets=(0, 0), block_shape=(BLOCK_M, 128), order=(1, 0)
    )
    
    l_base = L + off_b * l_st_b + off_h * l_st_h
    
    offs_n = off_n + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    
    dq_base = dQ + off_b * dq_st_b + off_h * dq_st_h
    off_d = tl.arange(0, 128)
    mask_d = off_d < 128
    
    num_m_blocks = tl.cdiv(S, BLOCK_M)
    for i in range(num_m_blocks):
        # Load inner loop tiles. Pipelined by Triton automatically.
        q = tl.load(q_block, boundary_check=(0, 1), padding_option="zero")
        do = tl.load(do_block, boundary_check=(0, 1), padding_option="zero")
        o = tl.load(o_block, boundary_check=(0, 1), padding_option="zero")
        
        off_m = i * BLOCK_M
        offs_m = off_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        
        lse = tl.load(l_base + offs_m * l_st_s, mask=mask_m, other=0.0)
        
        # Calculate transposed attention scores: s_T = K @ Q.T
        s_T = tl.dot(k, q.T, out_dtype=tl.float32) * scale
        
        mask = mask_n[:, None] & mask_m[None, :]
        s_T = tl.where(mask, s_T, float('-inf'))
        p_T = tl.exp(s_T - lse[None, :])
        p_T = tl.where(mask, p_T, 0.0)
        
        # Accumulate dV: dV += P^T @ dO
        dv_acc = tl.dot(p_T.to(q.dtype), do, dv_acc)
        
        # Compute D on-the-fly for the loaded M block
        D = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
        
        # dP^T = V @ dO^T
        dp_T = tl.dot(v, do.T, out_dtype=tl.float32)
        
        # dS^T = P^T * (dP^T - D^T)
        ds_T = p_T * (dp_T - D[None, :]) * scale
        
        # Accumulate dK: dK += dS^T @ Q
        dk_acc = tl.dot(ds_T.to(q.dtype), q, dk_acc)
        
        # Calculate dQ_inc = dS @ K = (dS^T)^T @ K
        dq_inc = tl.dot(tl.trans(ds_T).to(q.dtype), k, out_dtype=tl.float32)
        
        # Write to dQ via atomic_add (hardware accelerated bf16 atomics on Hopper)
        dq_ptrs = dq_base + offs_m[:, None] * dq_st_s + off_d[None, :] * dq_st_d
        mask_dq = mask_m[:, None] & mask_d[None, :]
        tl.atomic_add(dq_ptrs, dq_inc.to(q.dtype), mask=mask_dq, sem="relaxed")
        
        # Advance inner loop pointers
        q_block = tl.advance(q_block, (BLOCK_M, 0))
        do_block = tl.advance(do_block, (BLOCK_M, 0))
        o_block = tl.advance(o_block, (BLOCK_M, 0))
        
    # Store fully accumulated dK and dV
    dk_block = tl.make_block_ptr(
        base=dK + off_b * dk_st_b + off_h * dk_st_h,
        shape=(S, 128), strides=(dk_st_s, dk_st_d),
        offsets=(off_n, 0), block_shape=(BLOCK_N, 128), order=(1, 0)
    )
    dv_block = tl.make_block_ptr(
        base=dV + off_b * dv_st_b + off_h * dv_st_h,
        shape=(S, 128), strides=(dv_st_s, dv_st_d),
        offsets=(off_n, 0), block_shape=(BLOCK_N, 128), order=(1, 0)
    )
    tl.store(dk_block, dk_acc.to(k.dtype), boundary_check=(0, 1))
    tl.store(dv_block, dv_acc.to(v.dtype), boundary_check=(0, 1))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes the FlashAttention backward pass natively in Triton on Hopper SM90 GPUs.
    This uses a highly optimized 1-pass algorithm where outer loop traverses K/V blocks.
    It accumulates dK and dV in registers and uses fast hardware atomics for dQ,
    halving the total mathematical operations required compared to 2-pass approaches.
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

    # Clear dQ as it will be updated via atomic adds
    n_elements = dQ.numel()
    grid_clear = (triton.cdiv(n_elements, 1024),)
    clear_tensor[grid_clear](dQ, n_elements, BLOCK_SIZE=1024)

    # We use BLOCK_N=128 and BLOCK_M=64 to halve atomic collisions on dQ compared to M=128/N=64,
    # whilst easily fitting inside the Hopper 228KB shared memory limit with pipelining.
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