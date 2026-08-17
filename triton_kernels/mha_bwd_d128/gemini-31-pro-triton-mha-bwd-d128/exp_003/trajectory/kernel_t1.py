import torch
import triton
import triton.language as tl

# Configure Triton's allocator for device-side TensorDescriptors on Hopper.
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
    
    q_ptr = Q + off_b * q_st_b + off_h * q_st_h
    k_ptr = K + off_b * k_st_b + off_h * k_st_h
    v_ptr = V + off_b * v_st_b + off_h * v_st_h
    do_ptr = dO + off_b * do_st_b + off_h * do_st_h
    o_ptr = O + off_b * o_st_b + off_h * o_st_h
    
    # 2D Hopper TMA descriptors for each head
    q_desc = tl.make_tensor_descriptor(q_ptr, shape=[S, 128], strides=[q_st_s, q_st_d], block_shape=[BLOCK_M, 128], padding_option="zero")
    do_desc = tl.make_tensor_descriptor(do_ptr, shape=[S, 128], strides=[do_st_s, do_st_d], block_shape=[BLOCK_M, 128], padding_option="zero")
    o_desc = tl.make_tensor_descriptor(o_ptr, shape=[S, 128], strides=[o_st_s, o_st_d], block_shape=[BLOCK_M, 128], padding_option="zero")
    
    k_desc = tl.make_tensor_descriptor(k_ptr, shape=[S, 128], strides=[k_st_s, k_st_d], block_shape=[BLOCK_N, 128], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(v_ptr, shape=[S, 128], strides=[v_st_s, v_st_d], block_shape=[BLOCK_N, 128], padding_option="zero")
    
    off_m = pid_m * BLOCK_M
    
    # Load Q, dO, O once per block
    q = tl.load(q_desc, [off_m, 0])
    do = tl.load(do_desc, [off_m, 0])
    o = tl.load(o_desc, [off_m, 0])
    
    l_ptr_head = L + off_b * l_st_b + off_h * l_st_h
    offs_m = off_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    lse = tl.load(l_ptr_head + offs_m * l_st_s, mask=mask_m, other=0.0)
    
    # Precompute row-wise dot product of dO and O
    D = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
    
    dq_acc = tl.zeros((BLOCK_M, 128), tl.float32)
    
    num_n_blocks = tl.cdiv(S, BLOCK_N)
    for i in range(num_n_blocks):
        off_n = i * BLOCK_N
        k = tl.load(k_desc, [off_n, 0])
        v = tl.load(v_desc, [off_n, 0])
        
        s = tl.dot(q, k.T, out_dtype=tl.float32) * scale
        
        offs_n = off_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        mask = mask_m[:, None] & mask_n[None, :]
        s = tl.where(mask, s, float('-inf'))
        
        p = tl.exp(s - lse[:, None])
        
        dp = tl.dot(do, v.T, out_dtype=tl.float32)
        ds = p * (dp - D[:, None]) * scale
        
        dq_acc = tl.dot(ds.to(q.dtype), k, dq_acc)
        
    dq_ptr_head = dQ + off_b * dq_st_b + off_h * dq_st_h
    dq_ptrs = dq_ptr_head + offs_m[:, None] * dq_st_s + tl.arange(0, 128)[None, :] * dq_st_d
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
    
    q_ptr = Q + off_b * q_st_b + off_h * q_st_h
    k_ptr = K + off_b * k_st_b + off_h * k_st_h
    v_ptr = V + off_b * v_st_b + off_h * v_st_h
    do_ptr = dO + off_b * do_st_b + off_h * do_st_h
    o_ptr = O + off_b * o_st_b + off_h * o_st_h
    
    q_desc = tl.make_tensor_descriptor(q_ptr, shape=[S, 128], strides=[q_st_s, q_st_d], block_shape=[BLOCK_M, 128], padding_option="zero")
    do_desc = tl.make_tensor_descriptor(do_ptr, shape=[S, 128], strides=[do_st_s, do_st_d], block_shape=[BLOCK_M, 128], padding_option="zero")
    o_desc = tl.make_tensor_descriptor(o_ptr, shape=[S, 128], strides=[o_st_s, o_st_d], block_shape=[BLOCK_M, 128], padding_option="zero")
    k_desc = tl.make_tensor_descriptor(k_ptr, shape=[S, 128], strides=[k_st_s, k_st_d], block_shape=[BLOCK_N, 128], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(v_ptr, shape=[S, 128], strides=[v_st_s, v_st_d], block_shape=[BLOCK_N, 128], padding_option="zero")
    
    off_n = pid_n * BLOCK_N
    
    # Load K, V once per block
    k = tl.load(k_desc, [off_n, 0])
    v = tl.load(v_desc, [off_n, 0])
    
    dk_acc = tl.zeros((BLOCK_N, 128), tl.float32)
    dv_acc = tl.zeros((BLOCK_N, 128), tl.float32)
    
    l_ptr_head = L + off_b * l_st_b + off_h * l_st_h
    offs_n = off_n + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    
    num_m_blocks = tl.cdiv(S, BLOCK_M)
    for i in range(num_m_blocks):
        off_m = i * BLOCK_M
        
        q = tl.load(q_desc, [off_m, 0])
        do = tl.load(do_desc, [off_m, 0])
        o = tl.load(o_desc, [off_m, 0])
        
        offs_m = off_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        
        lse = tl.load(l_ptr_head + offs_m * l_st_s, mask=mask_m, other=0.0)
        
        s = tl.dot(q, k.T, out_dtype=tl.float32) * scale
        mask = mask_m[:, None] & mask_n[None, :]
        s = tl.where(mask, s, float('-inf'))
        p = tl.exp(s - lse[:, None])
        
        dv_acc = tl.dot((p.to(q.dtype)).T, do, dv_acc)
        
        D = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
        dp = tl.dot(do, v.T, out_dtype=tl.float32)
        ds = p * (dp - D[:, None]) * scale
        
        dk_acc = tl.dot((ds.to(q.dtype)).T, q, dk_acc)
        
    dk_ptr_head = dK + off_b * dk_st_b + off_h * dk_st_h
    dv_ptr_head = dV + off_b * dv_st_b + off_h * dv_st_h
    
    dk_ptrs = dk_ptr_head + offs_n[:, None] * dk_st_s + tl.arange(0, 128)[None, :] * dk_st_d
    dv_ptrs = dv_ptr_head + offs_n[:, None] * dv_st_s + tl.arange(0, 128)[None, :] * dv_st_d
    
    tl.store(dk_ptrs, dk_acc.to(k.dtype), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv_acc.to(v.dtype), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
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

    # Launch configuration tailored for Hopper SM90 memory and Tensor Cores boundaries using TMA.
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
        H=H,
        BLOCK_M=BLOCK_M_DKV, BLOCK_N=BLOCK_N_DKV,
        num_warps=4, num_stages=3
    )