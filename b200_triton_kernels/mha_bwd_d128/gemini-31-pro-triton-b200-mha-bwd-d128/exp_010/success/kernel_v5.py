import math
import torch
import triton
import triton.language as tl

# Standard Triton TMA descriptor memory allocator hook
def _alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(_alloc_fn)

@triton.jit
def bwd_dq_kernel(
    Q, K, V, O, dO, L, dQ,
    stride_q_b, stride_q_h, stride_q_s, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_o_b, stride_o_h, stride_o_s, stride_o_d,
    stride_do_b, stride_do_h, stride_do_s, stride_do_d,
    stride_l_b, stride_l_h, stride_l_s,
    stride_dq_b, stride_dq_h, stride_dq_s, stride_dq_d,
    B, H, S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b_idx = pid_bh // H
    h_idx = pid_bh % H
    
    m_start = pid_m * BLOCK_M
    
    # 2D logical slicing mapped directly to Blackwell TMA paths
    q_ptr = Q + b_idx * stride_q_b + h_idx * stride_q_h
    o_ptr = O + b_idx * stride_o_b + h_idx * stride_o_h
    do_ptr = dO + b_idx * stride_do_b + h_idx * stride_do_h
    dq_ptr = dQ + b_idx * stride_dq_b + h_idx * stride_dq_h
    
    q_desc = tl.make_tensor_descriptor(q_ptr, shape=[S, D], strides=[stride_q_s, 1], block_shape=[BLOCK_M, D], padding_option="zero")
    o_desc = tl.make_tensor_descriptor(o_ptr, shape=[S, D], strides=[stride_o_s, 1], block_shape=[BLOCK_M, D], padding_option="zero")
    do_desc = tl.make_tensor_descriptor(do_ptr, shape=[S, D], strides=[stride_do_s, 1], block_shape=[BLOCK_M, D], padding_option="zero")
    dq_desc = tl.make_tensor_descriptor(dq_ptr, shape=[S, D], strides=[stride_dq_s, 1], block_shape=[BLOCK_M, D], padding_option="zero")

    # Stationary descriptors loaded once per M-block outer step
    q_tile = q_desc.load([m_start, 0])
    o_tile = o_desc.load([m_start, 0])
    do_tile = do_desc.load([m_start, 0])
    
    off_m = m_start + tl.arange(0, BLOCK_M)
    mask_m = off_m < S
    
    l_ptrs = L + b_idx * stride_l_b + h_idx * stride_l_h + off_m * stride_l_s
    if EVEN_M:
        l_tile = tl.load(l_ptrs)
    else:
        l_tile = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    # Base summation metric mapping to exact forward Softmax configuration
    delta = tl.sum(do_tile.to(tl.float32) * o_tile.to(tl.float32), axis=1)
    dq_acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    
    k_ptr = K + b_idx * stride_k_b + h_idx * stride_k_h
    v_ptr = V + b_idx * stride_v_b + h_idx * stride_v_h
    
    k_desc = tl.make_tensor_descriptor(k_ptr, shape=[S, D], strides=[stride_k_s, 1], block_shape=[BLOCK_N, D], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(v_ptr, shape=[S, D], strides=[stride_v_s, 1], block_shape=[BLOCK_N, D], padding_option="zero")
    
    num_n_blocks = tl.cdiv(S, BLOCK_N)
    for n_idx in tl.range(0, num_n_blocks, num_stages=3):
        n_start = n_idx * BLOCK_N
        
        # Stream keys and values fully backed by pipelined TMEM hardware logic
        k_tile = k_desc.load([n_start, 0])
        v_tile = v_desc.load([n_start, 0])
            
        s_mat = tl.dot(q_tile, k_tile.T, out_dtype=tl.float32) * scale
        
        # Isolate conditionality out of pipeline overhead perfectly through constexpr boundaries
        if not (EVEN_M and EVEN_N):
            if EVEN_N:
                s_mat = tl.where(mask_m[:, None], s_mat, -float("inf"))
            elif EVEN_M:
                off_n = n_start + tl.arange(0, BLOCK_N)
                mask_n = off_n < S
                s_mat = tl.where(mask_n[None, :], s_mat, -float("inf"))
            else:
                off_n = n_start + tl.arange(0, BLOCK_N)
                mask_n = off_n < S
                valid = mask_m[:, None] & mask_n[None, :]
                s_mat = tl.where(valid, s_mat, -float("inf"))
            
        p_mat = tl.exp(s_mat - l_tile[:, None])
        
        dp_mat = tl.dot(do_tile, v_tile.T, out_dtype=tl.float32)
        ds_mat = p_mat * (dp_mat - delta[:, None]) * scale
        
        if not (EVEN_M and EVEN_N):
            if EVEN_N:
                ds_mat = tl.where(mask_m[:, None], ds_mat, 0.0)
            elif EVEN_M:
                ds_mat = tl.where(mask_n[None, :], ds_mat, 0.0)
            else:
                ds_mat = tl.where(valid, ds_mat, 0.0)
            
        dq_acc = tl.dot(ds_mat.to(q_tile.dtype), k_tile, acc=dq_acc)
        
    # Flush mathematically deterministic result block exactly mapping source shapes
    dq_desc.store([m_start, 0], dq_acc.to(q_tile.dtype))


@triton.jit
def bwd_dk_dv_kernel(
    Q, K, V, O, dO, L, dK, dV,
    stride_q_b, stride_q_h, stride_q_s, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_o_b, stride_o_h, stride_o_s, stride_o_d,
    stride_do_b, stride_do_h, stride_do_s, stride_do_d,
    stride_l_b, stride_l_h, stride_l_s,
    stride_dk_b, stride_dk_h, stride_dk_s, stride_dk_d,
    stride_dv_b, stride_dv_h, stride_dv_s, stride_dv_d,
    B, H, S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b_idx = pid_bh // H
    h_idx = pid_bh % H
    
    n_start = pid_n * BLOCK_N
    
    k_ptr = K + b_idx * stride_k_b + h_idx * stride_k_h
    v_ptr = V + b_idx * stride_v_b + h_idx * stride_v_h
    dk_ptr = dK + b_idx * stride_dk_b + h_idx * stride_dk_h
    dv_ptr = dV + b_idx * stride_dv_b + h_idx * stride_dv_h
    
    k_desc = tl.make_tensor_descriptor(k_ptr, shape=[S, D], strides=[stride_k_s, 1], block_shape=[BLOCK_N, D], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(v_ptr, shape=[S, D], strides=[stride_v_s, 1], block_shape=[BLOCK_N, D], padding_option="zero")
    dk_desc = tl.make_tensor_descriptor(dk_ptr, shape=[S, D], strides=[stride_dk_s, 1], block_shape=[BLOCK_N, D], padding_option="zero")
    dv_desc = tl.make_tensor_descriptor(dv_ptr, shape=[S, D], strides=[stride_dv_s, 1], block_shape=[BLOCK_N, D], padding_option="zero")
    
    k_tile = k_desc.load([n_start, 0])
    v_tile = v_desc.load([n_start, 0])
    
    off_n = n_start + tl.arange(0, BLOCK_N)
    mask_n = off_n < S
        
    dk_acc = tl.zeros((BLOCK_N, D), dtype=tl.float32)
    dv_acc = tl.zeros((BLOCK_N, D), dtype=tl.float32)
    
    q_ptr = Q + b_idx * stride_q_b + h_idx * stride_q_h
    o_ptr = O + b_idx * stride_o_b + h_idx * stride_o_h
    do_ptr = dO + b_idx * stride_do_b + h_idx * stride_do_h
    
    q_desc = tl.make_tensor_descriptor(q_ptr, shape=[S, D], strides=[stride_q_s, 1], block_shape=[BLOCK_M, D], padding_option="zero")
    o_desc = tl.make_tensor_descriptor(o_ptr, shape=[S, D], strides=[stride_o_s, 1], block_shape=[BLOCK_M, D], padding_option="zero")
    do_desc = tl.make_tensor_descriptor(do_ptr, shape=[S, D], strides=[stride_do_s, 1], block_shape=[BLOCK_M, D], padding_option="zero")
    
    num_m_blocks = tl.cdiv(S, BLOCK_M)
    for m_idx in tl.range(0, num_m_blocks, num_stages=3):
        m_start = m_idx * BLOCK_M
        
        q_tile = q_desc.load([m_start, 0])
        o_tile = o_desc.load([m_start, 0])
        do_tile = do_desc.load([m_start, 0])
        
        off_m = m_start + tl.arange(0, BLOCK_M)
        l_ptrs = L + b_idx * stride_l_b + h_idx * stride_l_h + off_m * stride_l_s
        if EVEN_M:
            l_tile = tl.load(l_ptrs)
        else:
            mask_m = off_m < S
            l_tile = tl.load(l_ptrs, mask=mask_m, other=0.0)
            
        delta = tl.sum(do_tile.to(tl.float32) * o_tile.to(tl.float32), axis=1)
        
        s_mat_T = tl.dot(k_tile, q_tile.T, out_dtype=tl.float32) * scale
        
        if not (EVEN_M and EVEN_N):
            if EVEN_M:
                s_mat_T = tl.where(mask_n[:, None], s_mat_T, -float("inf"))
            elif EVEN_N:
                mask_m_cur = off_m < S
                s_mat_T = tl.where(mask_m_cur[None, :], s_mat_T, -float("inf"))
            else:
                mask_m_cur = off_m < S
                valid = mask_n[:, None] & mask_m_cur[None, :]
                s_mat_T = tl.where(valid, s_mat_T, -float("inf"))
            
        p_mat_T = tl.exp(s_mat_T - l_tile[None, :])
        
        dp_mat_T = tl.dot(v_tile, do_tile.T, out_dtype=tl.float32)
        ds_mat_T = p_mat_T * (dp_mat_T - delta[None, :]) * scale
        
        if not (EVEN_M and EVEN_N):
            if EVEN_M:
                ds_mat_T = tl.where(mask_n[:, None], ds_mat_T, 0.0)
            elif EVEN_N:
                ds_mat_T = tl.where(mask_m_cur[None, :], ds_mat_T, 0.0)
            else:
                ds_mat_T = tl.where(valid, ds_mat_T, 0.0)
            
        dk_acc = tl.dot(ds_mat_T.to(q_tile.dtype), q_tile, acc=dk_acc)
        dv_acc = tl.dot(p_mat_T.to(q_tile.dtype), do_tile, acc=dv_acc)
        
    dk_desc.store([n_start, 0], dk_acc.to(k_tile.dtype))
    dv_desc.store([n_start, 0], dv_acc.to(v_tile.dtype))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Extremely high-throughput Split-Ownership TMA Flash-Attention Backward.
    Utilizes Blackwell's Tensor Memory Allocation (TMA) pathways via device 2D descriptors,
    providing highly optimized HBM access streams aligned logically against compute loop stages.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    scale = 1.0 / math.sqrt(D)
    
    # Kernel 1: Pipeline mapping outer loop over independent query M-blocks
    BLOCK_M_1 = 128
    BLOCK_N_1 = 64
    grid_dq = (triton.cdiv(S, BLOCK_M_1), B * H)
    
    even_m_1 = (S % BLOCK_M_1 == 0)
    even_n_1 = (S % BLOCK_N_1 == 0)
    
    bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, L, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        B, H, S, scale,
        BLOCK_M=BLOCK_M_1, BLOCK_N=BLOCK_N_1, D=D,
        EVEN_M=even_m_1, EVEN_N=even_n_1,
        num_warps=8, num_stages=3
    )
    
    # Kernel 2: Pipeline mapping outer loop symmetrically over keys/values N-blocks
    BLOCK_M_2 = 64
    BLOCK_N_2 = 128
    grid_dk_dv = (triton.cdiv(S, BLOCK_N_2), B * H)
    
    even_m_2 = (S % BLOCK_M_2 == 0)
    even_n_2 = (S % BLOCK_N_2 == 0)
    
    bwd_dk_dv_kernel[grid_dk_dv](
        Q, K, V, O, dO, L, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        B, H, S, scale,
        BLOCK_M=BLOCK_M_2, BLOCK_N=BLOCK_N_2, D=D,
        EVEN_M=even_m_2, EVEN_N=even_n_2,
        num_warps=8, num_stages=3
    )