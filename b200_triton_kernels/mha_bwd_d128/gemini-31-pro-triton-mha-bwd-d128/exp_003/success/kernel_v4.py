import torch
import triton
import triton.language as tl

# Configure Triton's allocator for device-side TensorDescriptors used by `tl.make_block_ptr`.
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)


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
    
    off_m = pid_m * BLOCK_M
    
    q_block = tl.make_block_ptr(
        base=Q + off_b * q_st_b + off_h * q_st_h,
        shape=(S, 128), strides=(q_st_s, q_st_d),
        offsets=(off_m, 0), block_shape=(BLOCK_M, 128), order=(1, 0)
    )
    do_block = tl.make_block_ptr(
        base=dO + off_b * do_st_b + off_h * do_st_h,
        shape=(S, 128), strides=(do_st_s, do_st_d),
        offsets=(off_m, 0), block_shape=(BLOCK_M, 128), order=(1, 0)
    )
    o_block = tl.make_block_ptr(
        base=O + off_b * o_st_b + off_h * o_st_h,
        shape=(S, 128), strides=(o_st_s, o_st_d),
        offsets=(off_m, 0), block_shape=(BLOCK_M, 128), order=(1, 0)
    )
    
    # Pre-load block variables (kept on register boundaries)
    q = tl.load(q_block, boundary_check=(0, 1), padding_option="zero")
    do = tl.load(do_block, boundary_check=(0, 1), padding_option="zero")
    o = tl.load(o_block, boundary_check=(0, 1), padding_option="zero")
    
    # Compute `D` term for this corresponding chunk entirely in registers, avoids inner-loop overlap
    D = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
    
    offs_m = off_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    l_ptrs = L + off_b * l_st_b + off_h * l_st_h + offs_m * l_st_s
    lse = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    dq_acc = tl.zeros((BLOCK_M, 128), tl.float32)
    
    # Inner loop TMA definitions
    k_block = tl.make_block_ptr(
        base=K + off_b * k_st_b + off_h * k_st_h,
        shape=(S, 128), strides=(k_st_s, k_st_d),
        offsets=(0, 0), block_shape=(BLOCK_N, 128), order=(1, 0)
    )
    v_block = tl.make_block_ptr(
        base=V + off_b * v_st_b + off_h * v_st_h,
        shape=(S, 128), strides=(v_st_s, v_st_d),
        offsets=(0, 0), block_shape=(BLOCK_N, 128), order=(1, 0)
    )
    
    num_n_blocks = tl.cdiv(S, BLOCK_N)
    for i in range(num_n_blocks):
        k = tl.load(k_block, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(v_block, boundary_check=(0, 1), padding_option="zero")
        
        s_val = tl.dot(q, k.T, out_dtype=tl.float32) * scale
        
        off_n = i * BLOCK_N
        offs_n = off_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        
        mask = mask_m[:, None] & mask_n[None, :]
        s_val = tl.where(mask, s_val, float('-inf'))
        p = tl.exp(s_val - lse[:, None])
        p = tl.where(mask, p, 0.0)
        
        dp = tl.dot(do, v.T, out_dtype=tl.float32)
        ds = p * (dp - D[:, None]) * scale
        
        dq_acc = tl.dot(ds.to(q.dtype), k, dq_acc)
        
        k_block = tl.advance(k_block, (BLOCK_N, 0))
        v_block = tl.advance(v_block, (BLOCK_N, 0))
        
    dq_block = tl.make_block_ptr(
        base=dQ + off_b * dq_st_b + off_h * dq_st_h,
        shape=(S, 128), strides=(dq_st_s, dq_st_d),
        offsets=(off_m, 0), block_shape=(BLOCK_M, 128), order=(1, 0)
    )
    tl.store(dq_block, dq_acc.to(q.dtype), boundary_check=(0, 1))


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
    
    off_n = pid_n * BLOCK_N
    
    k_block = tl.make_block_ptr(
        base=K + off_b * k_st_b + off_h * k_st_h,
        shape=(S, 128), strides=(k_st_s, k_st_d),
        offsets=(off_n, 0), block_shape=(BLOCK_N, 128), order=(1, 0)
    )
    v_block = tl.make_block_ptr(
        base=V + off_b * v_st_b + off_h * v_st_h,
        shape=(S, 128), strides=(v_st_s, v_st_d),
        offsets=(off_n, 0), block_shape=(BLOCK_N, 128), order=(1, 0)
    )
    
    k = tl.load(k_block, boundary_check=(0, 1), padding_option="zero")
    v = tl.load(v_block, boundary_check=(0, 1), padding_option="zero")
    
    dk_acc = tl.zeros((BLOCK_N, 128), tl.float32)
    dv_acc = tl.zeros((BLOCK_N, 128), tl.float32)
    
    q_block = tl.make_block_ptr(
        base=Q + off_b * q_st_b + off_h * q_st_h,
        shape=(S, 128), strides=(q_st_s, q_st_d),
        offsets=(0, 0), block_shape=(BLOCK_M, 128), order=(1, 0)
    )
    do_block = tl.make_block_ptr(
        base=dO + off_b * do_st_b + off_h * do_st_h,
        shape=(S, 128), strides=(do_st_s, do_st_d),
        offsets=(0, 0), block_shape=(BLOCK_M, 128), order=(1, 0)
    )
    o_block = tl.make_block_ptr(
        base=O + off_b * o_st_b + off_h * o_st_h,
        shape=(S, 128), strides=(o_st_s, o_st_d),
        offsets=(0, 0), block_shape=(BLOCK_M, 128), order=(1, 0)
    )
    
    offs_n = off_n + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    l_base = L + off_b * l_st_b + off_h * l_st_h
    
    num_m_blocks = tl.cdiv(S, BLOCK_M)
    for i in range(num_m_blocks):
        q = tl.load(q_block, boundary_check=(0, 1), padding_option="zero")
        do = tl.load(do_block, boundary_check=(0, 1), padding_option="zero")
        o = tl.load(o_block, boundary_check=(0, 1), padding_option="zero")
        
        off_m = i * BLOCK_M
        offs_m = off_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        
        lse = tl.load(l_base + offs_m * l_st_s, mask=mask_m, other=0.0)
        D_val = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
        
        # WGMMA seamlessly converts layout orientations as configured natively
        s_T = tl.dot(k, q.T, out_dtype=tl.float32) * scale
        
        mask = mask_n[:, None] & mask_m[None, :]
        s_T = tl.where(mask, s_T, float('-inf'))
        p_T = tl.exp(s_T - lse[None, :])
        p_T = tl.where(mask, p_T, 0.0)
        
        dv_acc = tl.dot(p_T.to(q.dtype), do, dv_acc)
        
        dp_T = tl.dot(v, do.T, out_dtype=tl.float32)
        ds_T = p_T * (dp_T - D_val[None, :]) * scale
        
        dk_acc = tl.dot(ds_T.to(q.dtype), q, dk_acc)
        
        q_block = tl.advance(q_block, (BLOCK_M, 0))
        do_block = tl.advance(do_block, (BLOCK_M, 0))
        o_block = tl.advance(o_block, (BLOCK_M, 0))
        
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
    Computes the FlashAttention backward pass on NVIDIA Hopper architectures.
    Implements a 2-pass deterministic layout with distinct `dQ` and `dK/dV` kernels.
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

    # Highly tuned execution matrix for H100 shared-memory throughput and cache hierarchies 
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
        H=H, BLOCK_M=BLOCK_M_DQ, BLOCK_N=BLOCK_N_DQ,
        num_warps=4, num_stages=3
    )
    
    BLOCK_M_DKV = 64
    BLOCK_N_DKV = 128
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
        H=H, BLOCK_M=BLOCK_M_DKV, BLOCK_N=BLOCK_N_DKV,
        num_warps=4, num_stages=3
    )