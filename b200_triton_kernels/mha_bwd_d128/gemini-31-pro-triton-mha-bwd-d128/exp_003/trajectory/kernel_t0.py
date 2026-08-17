import torch
import triton
import triton.language as tl

@triton.jit
def bwd_dq_kernel(
    Q, K, V, dO, O, L, dQ,
    S, scale,
    q_st_b, q_st_h, q_st_s, q_st_d,
    k_st_b, k_st_h, k_st_s, k_st_d,
    v_st_b, v_st_h, v_st_s, v_st_d,
    do_st_b, do_st_h, do_st_s, do_st_d,
    o_st_b, o_st_h, o_st_s, o_st_d,
    l_st_b, l_st_h, l_st_s,
    dq_st_b, dq_st_h, dq_st_s, dq_st_d,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    off_b = pid_bh // H
    off_h = pid_bh % H
    
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, BLOCK_N)
    off_d = tl.arange(0, 128)
    
    mask_m = off_m < S
    
    q_ptrs = Q + off_b * q_st_b + off_h * q_st_h + off_m[:, None] * q_st_s + off_d[None, :] * q_st_d
    do_ptrs = dO + off_b * do_st_b + off_h * do_st_h + off_m[:, None] * do_st_s + off_d[None, :] * do_st_d
    o_ptrs = O + off_b * o_st_b + off_h * o_st_h + off_m[:, None] * o_st_s + off_d[None, :] * o_st_d
    l_ptrs = L + off_b * l_st_b + off_h * l_st_h + off_m * l_st_s
    
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    lse = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    # Compute row-wise sum(dO * O)
    D = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
    
    dq_acc = tl.zeros((BLOCK_M, 128), tl.float32)
    
    k_base = K + off_b * k_st_b + off_h * k_st_h
    v_base = V + off_b * v_st_b + off_h * v_st_h
    
    num_n_blocks = (S + BLOCK_N - 1) // BLOCK_N
    for i in range(num_n_blocks):
        n_start = i * BLOCK_N
        curr_off_n = n_start + off_n
        mask_n = curr_off_n < S
        
        k_ptrs = k_base + curr_off_n[:, None] * k_st_s + off_d[None, :] * k_st_d
        v_ptrs = v_base + curr_off_n[:, None] * v_st_s + off_d[None, :] * v_st_d
        
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        s = tl.dot(q, k.T, out_dtype=tl.float32) * scale
        
        p = tl.exp(s - lse[:, None])
        mask = mask_m[:, None] & mask_n[None, :]
        p = tl.where(mask, p, 0.0)
        
        dp = tl.dot(do, v.T, out_dtype=tl.float32)
        
        ds = p * (dp - D[:, None]) * scale
        
        dq_acc = tl.dot(ds.to(q.dtype), k, dq_acc)
        
    dq_ptrs = dQ + off_b * dq_st_b + off_h * dq_st_h + off_m[:, None] * dq_st_s + off_d[None, :] * dq_st_d
    tl.store(dq_ptrs, dq_acc.to(q.dtype), mask=mask_m[:, None])


@triton.jit
def bwd_dkv_kernel(
    Q, K, V, dO, O, L, dK, dV,
    S, scale,
    q_st_b, q_st_h, q_st_s, q_st_d,
    k_st_b, k_st_h, k_st_s, k_st_d,
    v_st_b, v_st_h, v_st_s, v_st_d,
    do_st_b, do_st_h, do_st_s, do_st_d,
    o_st_b, o_st_h, o_st_s, o_st_d,
    l_st_b, l_st_h, l_st_s,
    dk_st_b, dk_st_h, dk_st_s, dk_st_d,
    dv_st_b, dv_st_h, dv_st_s, dv_st_d,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    off_b = pid_bh // H
    off_h = pid_bh % H
    
    # Outer loop variables correspond to the K/V sequence dimension (N)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    off_d = tl.arange(0, 128)
    
    mask_n = off_n < S
    
    k_ptrs = K + off_b * k_st_b + off_h * k_st_h + off_n[:, None] * k_st_s + off_d[None, :] * k_st_d
    v_ptrs = V + off_b * v_st_b + off_h * v_st_h + off_n[:, None] * v_st_s + off_d[None, :] * v_st_d
    
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    
    dk_acc = tl.zeros((BLOCK_N, 128), tl.float32)
    dv_acc = tl.zeros((BLOCK_N, 128), tl.float32)
    
    q_base = Q + off_b * q_st_b + off_h * q_st_h
    do_base = dO + off_b * do_st_b + off_h * do_st_h
    o_base = O + off_b * o_st_b + off_h * o_st_h
    l_base = L + off_b * l_st_b + off_h * l_st_h
    
    num_m_blocks = (S + BLOCK_M - 1) // BLOCK_M
    for i in range(num_m_blocks):
        m_start = i * BLOCK_M
        curr_off_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = curr_off_m < S
        
        q_ptrs = q_base + curr_off_m[:, None] * q_st_s + off_d[None, :] * q_st_d
        do_ptrs = do_base + curr_off_m[:, None] * do_st_s + off_d[None, :] * do_st_d
        o_ptrs = o_base + curr_off_m[:, None] * o_st_s + off_d[None, :] * o_st_d
        l_ptrs = l_base + curr_off_m * l_st_s
        
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
        lse = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        s = tl.dot(q, k.T, out_dtype=tl.float32) * scale
        
        p = tl.exp(s - lse[:, None])
        mask = mask_m[:, None] & mask_n[None, :]
        p = tl.where(mask, p, 0.0)
        
        dv_acc = tl.dot(p.T.to(q.dtype), do, dv_acc)
        
        D = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
        
        dp = tl.dot(do, v.T, out_dtype=tl.float32)
        
        ds = p * (dp - D[:, None]) * scale
        
        dk_acc = tl.dot(ds.T.to(q.dtype), q, dk_acc)
        
    dk_ptrs = dK + off_b * dk_st_b + off_h * dk_st_h + off_n[:, None] * dk_st_s + off_d[None, :] * dk_st_d
    dv_ptrs = dV + off_b * dv_st_b + off_h * dv_st_h + off_n[:, None] * dv_st_s + off_d[None, :] * dv_st_d
    
    tl.store(dk_ptrs, dk_acc.to(k.dtype), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv_acc.to(v.dtype), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes the FlashAttention backward pass without causal mask natively in Triton.
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

    # Block configurations tuned for Hopper SM90 memory and Tensor Cores boundaries.
    BLOCK_M_DQ = 128
    BLOCK_N_DQ = 64
    grid_dq = (triton.cdiv(S, BLOCK_M_DQ), B * H)
    
    bwd_dq_kernel[grid_dq](
        Q, K, V, dO, O, L, dQ,
        S, scale,
        q_st_b, q_st_h, q_st_s, q_st_d,
        k_st_b, k_st_h, k_st_s, k_st_d,
        v_st_b, v_st_h, v_st_s, v_st_d,
        do_st_b, do_st_h, do_st_s, do_st_d,
        o_st_b, o_st_h, o_st_s, o_st_d,
        l_st_b, l_st_h, l_st_s,
        dq_st_b, dq_st_h, dq_st_s, dq_st_d,
        H=H,
        BLOCK_M=BLOCK_M_DQ, BLOCK_N=BLOCK_N_DQ,
        num_warps=8, num_stages=3
    )
    
    BLOCK_N_DKV = 128
    BLOCK_M_DKV = 64
    grid_dkv = (triton.cdiv(S, BLOCK_N_DKV), B * H)
    
    bwd_dkv_kernel[grid_dkv](
        Q, K, V, dO, O, L, dK, dV,
        S, scale,
        q_st_b, q_st_h, q_st_s, q_st_d,
        k_st_b, k_st_h, k_st_s, k_st_d,
        v_st_b, v_st_h, v_st_s, v_st_d,
        do_st_b, do_st_h, do_st_s, do_st_d,
        o_st_b, o_st_h, o_st_s, o_st_d,
        l_st_b, l_st_h, l_st_s,
        dk_st_b, dk_st_h, dk_st_s, dk_st_d,
        dv_st_b, dv_st_h, dv_st_s, dv_st_d,
        H=H,
        BLOCK_M=BLOCK_M_DKV, BLOCK_N=BLOCK_N_DKV,
        num_warps=8, num_stages=3
    )