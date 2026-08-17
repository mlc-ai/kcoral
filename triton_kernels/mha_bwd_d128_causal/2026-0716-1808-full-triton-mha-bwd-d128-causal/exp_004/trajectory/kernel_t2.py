import math
import torch
import triton
import triton.language as tl


@triton.jit
def _bwd_dQ(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
    S, stride_b, stride_h, stride_s, stride_d, B, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    """Computes the gradient w.r.t. the queries (dQ)."""
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    
    offset_m = pid_m * BLOCK_M
    if offset_m >= S:
        return
    
    base_ptr = b * stride_b + h * stride_h
    
    tl.multiple_of(Q_ptr, 128)
    tl.multiple_of(K_ptr, 128)
    tl.multiple_of(V_ptr, 128)
    tl.multiple_of(O_ptr, 128)
    tl.multiple_of(dO_ptr, 128)
    
    q_desc = tl.make_tensor_descriptor(Q_ptr, shape=[B * H * S, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_M, BLOCK_M], padding_option="zero")
    k_desc = tl.make_tensor_descriptor(K_ptr, shape=[B * H * S, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_N, BLOCK_N], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(V_ptr, shape=[B * H * S, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_N, BLOCK_N], padding_option="zero")
    o_desc = tl.make_tensor_descriptor(O_ptr, shape=[B * H * S, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_M, BLOCK_M], padding_option="zero")
    do_desc = tl.make_tensor_descriptor(dO_ptr, shape=[B * H * S, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_M, BLOCK_M], padding_option="zero")
    
    q_0 = q_desc.load([bh * S + offset_m, 0])
    q_1 = q_desc.load([bh * S + offset_m, 64])
    
    o_0 = o_desc.load([bh * S + offset_m, 0])
    o_1 = o_desc.load([bh * S + offset_m, 64])
    
    do_0 = do_desc.load([bh * S + offset_m, 0])
    do_1 = do_desc.load([bh * S + offset_m, 64])
    
    d_val = (do_0 * o_0).sum(axis=1) + (do_1 * o_1).sum(axis=1)
    
    l_desc = tl.make_tensor_descriptor(L_ptr, shape=[B * H * S], strides=[1], block_shape=[BLOCK_M], padding_option="zero")
    l_load = l_desc.load([bh * S + offset_m])
    
    acc_dQ_0 = tl.zeros((BLOCK_M, BLOCK_M), tl.float32)
    acc_dQ_1 = tl.zeros((BLOCK_M, BLOCK_M), tl.float32)
    
    scale = 1.0 / math.sqrt(HEAD_DIM)
    
    q_row = tl.arange(0, BLOCK_M)
    k_row = tl.arange(0, BLOCK_N)
    
    for j in range(pid_m + 1):
        offset_n = j * BLOCK_N
        
        k_0 = k_desc.load([bh * S + offset_n, 0])
        k_1 = k_desc.load([bh * S + offset_n, 64])
        
        v_0 = v_desc.load([bh * S + offset_n, 0])
        v_1 = v_desc.load([bh * S + offset_n, 64])
        
        acc_S = tl.dot(q_0, k_0)
        acc_S = tl.dot(q_1, k_1, acc_S)
        
        acc_dP = tl.dot(do_0, v_0)
        acc_dP = tl.dot(do_1, v_1, acc_dP)
        
        p_unmasked = tl.exp(acc_S * scale - l_load[:, None])
        
        global_q_idx = (pid_m * BLOCK_M + q_row[:, None])
        global_k_idx = (j * BLOCK_N + k_row[None, :])
        mask_2d = (global_q_idx >= global_k_idx) & (global_k_idx < S) & (global_q_idx < S)
        
        p_unmasked = tl.where(mask_2d, p_unmasked, 0.0)
        ds = tl.where(mask_2d, p_unmasked * (acc_dP - d_val[:, None]) * scale, 0.0)
        ds_0 = ds[:, :32]
        ds_1 = ds[:, 32:]
        
        acc_dQ_0 = tl.dot(ds_0, k_0, acc_dQ_0)
        acc_dQ_1 = tl.dot(ds_1, k_1, acc_dQ_1)
        
    row = tl.arange(0, BLOCK_M)
    col = tl.arange(0, BLOCK_M)
    row_mask_0 = (offset_m + row[:, None]) < S
    
    out_ptr_0 = dQ_ptr + base_ptr + (offset_m + row[:, None]) * stride_s + (0 + col[None, :]) * stride_d
    tl.store(out_ptr_0, acc_dQ_0, mask=row_mask_0)
    
    out_ptr_1 = dQ_ptr + base_ptr + (offset_m + row[:, None]) * stride_s + (64 + col[None, :]) * stride_d
    tl.store(out_ptr_1, acc_dQ_1, mask=row_mask_0)


@triton.jit
def _bwd_dK_dV(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    S, stride_b, stride_h, stride_s, stride_d, B, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    """Resolves gradients w.r.t. the memory bank keys and values (dK, dV)."""
    pid_n = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    
    offset_n = pid_n * BLOCK_N
    if offset_n >= S:
        return
    
    base_ptr = b * stride_b + h * stride_h
    
    tl.multiple_of(Q_ptr, 128)
    tl.multiple_of(K_ptr, 128)
    tl.multiple_of(V_ptr, 128)
    tl.multiple_of(O_ptr, 128)
    tl.multiple_of(dO_ptr, 128)
    
    q_desc = tl.make_tensor_descriptor(Q_ptr, shape=[B * H * S, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_M, BLOCK_M], padding_option="zero")
    k_desc = tl.make_tensor_descriptor(K_ptr, shape=[B * H * S, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_N, BLOCK_N], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(V_ptr, shape=[B * H * S, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_N, BLOCK_N], padding_option="zero")
    o_desc = tl.make_tensor_descriptor(O_ptr, shape=[B * H * S, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_M, BLOCK_M], padding_option="zero")
    do_desc = tl.make_tensor_descriptor(dO_ptr, shape=[B * H * S, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_M, BLOCK_M], padding_option="zero")
    
    k_0 = k_desc.load([bh * S + offset_n, 0])
    k_1 = k_desc.load([bh * S + offset_n, 64])
    
    v_0 = v_desc.load([bh * S + offset_n, 0])
    v_1 = v_desc.load([bh * S + offset_n, 64])
    
    acc_dK_0 = tl.zeros((BLOCK_N, BLOCK_N), tl.float32)
    acc_dK_1 = tl.zeros((BLOCK_N, BLOCK_N), tl.float32)
    acc_dV_0 = tl.zeros((BLOCK_N, BLOCK_N), tl.float32)
    acc_dV_1 = tl.zeros((BLOCK_N, BLOCK_N), tl.float32)
    
    scale = 1.0 / math.sqrt(HEAD_DIM)
    
    q_row = tl.arange(0, BLOCK_M)
    k_row = tl.arange(0, BLOCK_N)
    
    num_blocks_per_head = (S + BLOCK_M - 1) // BLOCK_M
    
    for i in range(pid_n, num_blocks_per_head):
        offset_m = i * BLOCK_M
        if offset_m >= S:
            break
            
        q_0 = q_desc.load([bh * S + offset_m, 0])
        q_1 = q_desc.load([bh * S + offset_m, 64])
        
        o_0 = o_desc.load([bh * S + offset_m, 0])
        o_1 = o_desc.load([bh * S + offset_m, 64])
        
        do_0 = do_desc.load([bh * S + offset_m, 0])
        do_1 = do_desc.load([bh * S + offset_m, 64])
        
        d_val = (do_0 * o_0).sum(axis=1) + (do_1 * o_1).sum(axis=1)
        
        l_desc = tl.make_tensor_descriptor(L_ptr, shape=[B * H * S], strides=[1], block_shape=[BLOCK_M], padding_option="zero")
        l_load = l_desc.load([bh * S + offset_m])
        
        acc_S = tl.dot(q_0, k_0)
        acc_S = tl.dot(q_1, k_1, acc_S)
        
        acc_dP = tl.dot(do_0, v_0)
        acc_dP = tl.dot(do_1, v_1, acc_dP)
        
        p_unmasked = tl.exp(acc_S * scale - l_load[:, None])
        
        global_q_idx = (i * BLOCK_M + q_row[:, None])
        global_k_idx = (pid_n * BLOCK_N + k_row[None, :])
        mask_2d = (global_q_idx >= global_k_idx) & (global_k_idx < S) & (global_q_idx < S)
        
        p_unmasked = tl.where(mask_2d, p_unmasked, 0.0)
        ds = tl.where(mask_2d, p_unmasked * (acc_dP - d_val[:, None]) * scale, 0.0)
        
        ds_T = ds.T
        p_T = p_unmasked.T
        
        ds_T_0 = ds_T[:, :32]
        ds_T_1 = ds_T[:, 32:]
        p_T_0 = p_T[:, :32]
        p_T_1 = p_T[:, 32:]
        
        acc_dK_0 = tl.dot(ds_T_0, q_0, acc_dK_0)
        acc_dK_1 = tl.dot(ds_T_1, q_1, acc_dK_1)
        
        acc_dV_0 = tl.dot(p_T_0, do_0, acc_dV_0)
        acc_dV_1 = tl.dot(p_T_1, do_1, acc_dV_1)
        
    row = tl.arange(0, BLOCK_N)
    col = tl.arange(0, BLOCK_N)
    row_mask_n = (offset_n + row[:, None]) < S
    
    out_ptr_K0 = dK_ptr + base_ptr + (offset_n + row[:, None]) * stride_s + (0 + col[None, :]) * stride_d
    tl.store(out_ptr_K0, acc_dK_0, mask=row_mask_n)
    
    out_ptr_K1 = dK_ptr + base_ptr + (offset_n + row[:, None]) * stride_s + (64 + col[None, :]) * stride_d
    tl.store(out_ptr_K1, acc_dK_1, mask=row_mask_n)
    
    out_ptr_V0 = dV_ptr + base_ptr + (offset_n + row[:, None]) * stride_s + (0 + col[None, :]) * stride_d
    tl.store(out_ptr_V0, acc_dV_0, mask=row_mask_n)
    
    out_ptr_V1 = dV_ptr + base_ptr + (offset_n + row[:, None]) * stride_s + (64 + col[None, :]) * stride_d
    tl.store(out_ptr_V1, acc_dV_1, mask=row_mask_n)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Host bridge launching the two sequential Triton device-pass routines."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    
    stride_b = H * S * d
    stride_h = S * d
    stride_s = d
    stride_d = 1
    
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)
    
    triton.set_allocator(alloc_fn)
    
    num_blocks_per_head = triton.cdiv(S, 64)
    grid = (num_blocks_per_head, B * H)
    
    print(f"[RUNTIME] Launching _bwd_dQ with grid {grid}")
    _bwd_dQ[grid](Q, K, V, O, dO, L, dQ, S, stride_b, stride_h, stride_s, stride_d, B, H, BLOCK_M=64, BLOCK_N=64, HEAD_DIM=128, num_warps=4, num_stages=2)
    
    print(f"[RUNTIME] Launching _bwd_dK_dV with grid {grid}")
    _bwd_dK_dV[grid](Q, K, V, O, dO, L, dK, dV, S, stride_b, stride_h, stride_s, stride_d, B, H, BLOCK_M=64, BLOCK_N=64, HEAD_DIM=128, num_warps=4, num_stages=2)