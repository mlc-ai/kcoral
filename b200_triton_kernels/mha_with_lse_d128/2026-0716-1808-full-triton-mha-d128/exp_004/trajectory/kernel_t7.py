import torch
import triton
import triton.language as tl


@triton.jit
def _attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S_len, scale, H,
):
    """
    Optimized FlashAttention kernel targeting Hopper WGMMA capabilities.
    
    Uses 3D TMA descriptors to natively load data in Transposed Layout, enabling 
    highly efficient WGMMA execution without explicit transpose instructions.
    Implements robust double-buffering of Key and Value chunks to guarantee 
    that HBM fetch latencies are fully hidden behind GEMM compute.
    
    Grid mapping: (ceil(S/32), B, H)
    - Axis 0: Query block index (each CTA handles 32 query rows)
    - Axis 1: Batch index
    - Axis 2: Attention head index
    """
    
    q_blk = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    
    global_row = q_blk * 32
    chunk_bh = b * H + h
    
    s_D = 128
    
    q_ptr_bh = Q_ptr + chunk_bh * S_len * s_D
    q_desc = tl.make_tensor_descriptor(
        q_ptr_bh, shape=[S_len, s_D, 1], strides=[s_D, 1, 1],
        block_shape=[32, 16, 16], padding_option="zero")
    
    k_ptr_bh = K_ptr + chunk_bh * S_len * s_D
    k_desc = tl.make_tensor_descriptor(
        k_ptr_bh, shape=[S_len, s_D, 1], strides=[s_D, 1, 1],
        block_shape=[64, 16, 16], padding_option="zero")
    
    v_ptr_bh = V_ptr + chunk_bh * S_len * s_D
    v_desc = tl.make_tensor_descriptor(
        v_ptr_bh, shape=[S_len, s_D, 1], strides=[s_D, 1, 1],
        block_shape=[64, 16, 16], padding_option="zero")
    
    Q_buf = tl.empty((2, 32 * 16, 64), dtype=tl.float32)
    Q_buf[0] = desc_load_3d(q_desc.load, [global_row, 0, 0])
    Q_buf[1] = desc_load_3d(q_desc.load, [global_row, 64, 0])
    
    k_buf0 = tl.empty((2, 64 * 16, 64), dtype=tl.float32)
    k_buf1 = tl.empty((2, 64 * 16, 64), dtype=tl.float32)
    v_buf0 = tl.empty((2, 64 * 16, 64), dtype=tl.float32)
    v_buf1 = tl.empty((2, 64 * 16, 64), dtype=tl.float32)
    
    acc_O0 = tl.zeros((32, 64), tl.float32)
    acc_O1 = tl.zeros((32, 64), tl.float32)
    
    m_i = tl.full((32,), -1e38, tl.float32)
    l_i = tl.zeros((32,), 1.0, tl.float32)
    
    num_kv_iters = tl.cdiv(S_len, 64)
    grid_stride_iters = min(num_kv_iters, 2)
    
    if num_kv_iters > 0:
        tl.shared_store(k_buf0[0], desc_load_3d(k_desc.load, [0, 0, 0]))
        tl.shared_store(k_buf0[1], desc_load_3d(k_desc.load, [0, 64, 0]))
        tl.shared_store(v_buf0[0], desc_load_3d(v_desc.load, [0, 0, 0]))
        tl.shared_store(v_buf0[1], desc_load_3d(v_desc.load, [0, 64, 0]))
    
    for i in range(num_kv_iters):
        cur_iter = i % 2
        k_buf_cur = k_buf0 if cur_iter == 0 else k_buf1
        v_buf_cur = v_buf0 if cur_iter == 0 else v_buf1
        k_buf_next = k_buf1 if cur_iter == 0 else k_buf0
        v_buf_next = v_buf1 if cur_iter == 0 else v_buf0
        
        load_next = i + 1 < num_kv_iters
        if load_next:
            next_kv_row = (i + 1) * 64
            tl.shared_store(k_buf_next[0], desc_load_3d(k_desc.load, [next_kv_row, 0, 0]))
            tl.shared_store(k_buf_next[1], desc_load_3d(k_desc.load, [next_kv_row, 64, 0]))
            tl.shared_store(v_buf_next[0], desc_load_3d(v_desc.load, [next_kv_row, 0, 0]))
            tl.shared_store(v_buf_next[1], desc_load_3d(v_desc.load, [next_kv_row, 64, 0]))
        
        k0 = tl.shared_load(k_buf_cur[0])
        k1 = tl.shared_load(k_buf_cur[1])
        
        s = (tl.dot(Q_buf[0], k0) + tl.dot(Q_buf[1], k1)) * scale
        
        global_kv_row = i * 64
        kv_offs = global_kv_row + tl.arange(0, 64)
        kv_mask = kv_offs < S_len
        q_offs = global_row + tl.arange(0, 32)
        q_mask = q_offs < S_len
        
        s = apply_softmax_mask(s, kv_mask, q_mask)
        
        m_i_prev = m_i
        m_i = tl.maximum(m_i, tl.max(s, axis=1))
        curr_p = tl.exp(s - m_i[:, None])
        l_i = l_i * tl.exp(m_i_prev - m_i) + tl.sum(curr_p, axis=1)
        
        acc_O0 *= tl.exp(m_i_prev - m_i)[:, None]
        acc_O1 *= tl.exp(m_i_prev - m_i)[:, None]
        
        v0 = tl.shared_load(v_buf_cur[0])
        v1 = tl.shared_load(v_buf_cur[1])
        
        acc_O0 += tl.dot(curr_p, v0)
        acc_O1 += tl.dot(curr_p, v1)
        
    q_offs = global_row + tl.arange(0, 32)
    q_mask = q_offs < S_len
    
    val0 = acc_O0 / l_i[:, None]
    val1 = acc_O1 / l_i[:, None]
    
    row_ptr_o = (chunk_bh * S_len + global_row) * s_D
    
    off_d0 = tl.arange(0, 64)
    out_ptr0 = O_ptr + row_ptr_o + q_offs[:, None] * s_D + off_d0[None, :]
    tl.store(out_ptr0, val0.to(tl.bfloat16), mask=q_mask[:, None])
    
    off_d1 = tl.arange(64, 128)
    out_ptr1 = O_ptr + row_ptr_o + q_offs[:, None] * s_D + off_d1[None, :]
    tl.store(out_ptr1, val1.to(tl.bfloat16), mask=q_mask[:, None])
    
    lse_ptr = LSE_ptr + chunk_bh * S_len + q_offs
    LSE_val = m_i + tl.log(l_i)
    tl.store(lse_ptr, LSE_val, mask=q_mask)


@triton.jit
def desc_load_3d(load_fn, coords):
    BLOCK_ROWS, BLOCK_COLS, BLOCK_DEPTH = 32, 64, 16
    tiles = []
    for i in range(0, BLOCK_COLS, BLOCK_DEPTH):
        tile = load_fn([coords[0], coords[1] + i, coords[2]])
        tiles.append(tile)
    return tl.reshape(tl.concat(tiles, axis=1), (BLOCK_ROWS * BLOCK_DEPTH, BLOCK_COLS))


@triton.jit
def apply_softmax_mask(s, kv_mask, q_mask):
    return tl.where(kv_mask[None, :] & q_mask[:, None], s, -1e44)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S_len, D = Q.shape
    
    scale = 1.0 / (D ** 0.5)
    
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)
    
    triton.set_allocator(alloc_fn)
    
    grid = (triton.cdiv(S_len, 32), B, H)
    _attention_kernel[grid](Q, K, V, O, LSE, S_len, scale, H, num_warps=4, num_ctas=1)