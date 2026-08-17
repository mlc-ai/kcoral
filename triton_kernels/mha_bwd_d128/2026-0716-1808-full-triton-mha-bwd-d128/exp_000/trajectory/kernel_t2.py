import torch
import triton
import triton.language as tl
import math


@triton.jit
def swizzle_8x8(row, col):
    tile_x = col // 8
    tile_y = row // 8
    new_tile_x = tile_x ^ ((tile_y % 2) * 8 + (tile_y // 2))
    return (row * 128 + new_tile_x * 8 + (col % 8)).to(tl.int64)


@triton.jit
def lane_id_to_idx(lane_id):
    idx_8 = lane_id % 8
    idx_4 = (lane_id // 8) % 4
    idx_2 = (lane_id // 32) % 2
    idx_1 = lane_id // 64
    r = idx_1 * 128 + idx_2 * 64 + (idx_4 // 2) * 8
    c = idx_8 * 8 + (idx_4 % 2) * 4
    return r, c


@triton.jit
def load_swizzled_batch(base_ptr, buf_idx, desc, offset_0, offset_1, S_len, sqrt_d):
    row_start_0 = offset_0[0]
    col_start_0 = offset_0[1]
    row_start_1 = offset_1[0]
    col_start_1 = offset_1[1]
    
    dst = tl.empty([4, 128, 64], tl.bfloat16)
    
    for tile_idx in range(4):
        for swizzle_idx in range(8):
            lane_id = tl.program_id(3)
            r, c = lane_id_to_idx(lane_id)
            
            if tile_idx < 2:
                row_idx = r % 128
                col_idx = c % 64
                desc.load_gather((row_start_0 + r, col_start_0 + c), base_ptr, buf_idx, lane_id_to_swizzled(row_idx, col_idx))
            else:
                row_idx = r % 128
                col_idx = c % 64
                desc.load_gather((row_start_1 + r, col_start_1 + c), base_ptr, buf_idx, lane_id_to_swizzled(row_idx, col_idx))


@triton.jit
def read_swizzled(src, buf_idx, shape):
    row, col = shape
    dst = tl.empty([4, row, col], tl.bfloat16)
    for tile_idx in range(4):
        for sw_idx in range(4):
            lane_id = tl.program_id(3)
            r, c = lane_id_to_idx(lane_id + sw_idx * 128)
            new_row = r % row
            new_col = c % col
            dst[tile_idx, new_row, new_col] = src[buf_idx, r, c]
    return dst


@triton.jit
def write_swizzled(shared_storage, buf_idx, src):
    for i in range(4):
        for r in range(128):
            for c in range(64):
                if i < 2:
                    shared_storage[buf_idx, r, c] = src[i, r, c]
                else:
                    shared_storage[buf_idx, r, c + 64] = src[i, r, c]


@triton.jit
def compute_S_and_d(Q_sub, dO_sub, K_sub, V_sub):
    S_acc = tl.zeros((128, 128), tl.float32)
    d_acc = tl.zeros((128, 128), tl.float32)
    for i in range(2):
        S_acc += tl.dot(Q_sub[i], K_sub[i])
        d_acc += tl.dot(dO_sub[i], V_sub[i])
    return S_acc, d_acc


@triton.jit
def compute_S_and_d_transposed(K_sub, V_sub, Q_sub, dO_sub):
    S_acc = tl.zeros((128, 256), tl.float32)
    d_acc = tl.zeros((128, 256), tl.float32)
    for i in range(2):
        S_acc += tl.dot(K_sub[i].T, Q_sub[i].T)
        d_acc += tl.dot(V_sub[i].T, dO_sub[i].T)
    return S_acc, d_acc


@triton.jit
def wait_commit(stage):
    expected = (commit_count[stage] + 1).to(tl.uint32)
    while tl.atomic_cas(mbarrier_ptr[stage], commit_count[stage], expected) != commit_count[stage]:
        pass
    commit_count[stage] = expected


@triton.jit
def signal_commit(desc, mbarrier_ptr, stage):
    desc.store_mbarrier(mbarrier_ptr, stage)


@triton.jit
def bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr_flat, dQ_ptr,
    S_len, sqrt_d,
):
    NUM_SMS: tl.constexpr = 132
    NUM_STAGES: tl.constexpr = 2
    WARP_SPECIALIZE: tl.constexpr = True
    HEAD_DIM: tl.constexpr = 128
    BLOCK_Q: tl.constexpr = 256
    BLOCK_K: tl.constexpr = 128

    q_tile = tl.program_id(0)
    b_h = tl.program_id(1)
    q_offset = q_tile * BLOCK_Q
    
    if q_offset >= S_len:
        return

    step = NUM_SMS * BLOCK_Q
    q_offset += tl.program_id(2) * step
    
    if q_offset >= S_len:
        return

    Q_buf = tl.empty([NUM_STAGES, BLOCK_Q, HEAD_DIM], tl.bfloat16)
    dO_buf = tl.empty([NUM_STAGES, BLOCK_Q, HEAD_DIM], tl.bfloat16)
    O_buf = tl.empty([NUM_STAGES, BLOCK_Q, HEAD_DIM], tl.bfloat16)
    K_buf = tl.empty([NUM_STAGES, BLOCK_K, HEAD_DIM], tl.bfloat16)
    V_buf = tl.empty([NUM_STAGES, BLOCK_K, HEAD_DIM], tl.bfloat16)
    
    mbarrier_ptr = [0] * NUM_STAGES
    commit_count = [0] * NUM_STAGES
    for i in range(NUM_STAGES):
        mbarrier_ptr[i] = tl.allocate_barrier(1)
        commit_count[i] = 0
    
    q_desc = tl.make_tensor_descriptor(Q_ptr, shape=[B * H * S_len, 128], strides=[128, 1], block_shape=[BLOCK_Q, 128], padding_option="zero")
    k_desc = tl.make_tensor_descriptor(K_ptr, shape=[B * H * S_len, 128], strides=[128, 1], block_shape=[BLOCK_K, 128], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(V_ptr, shape=[B * H * S_len, 128], strides=[128, 1], block_shape=[BLOCK_K, 128], padding_option="zero")
    do_desc = tl.make_tensor_descriptor(dO_ptr, shape=[B * H * S_len, 128], strides=[128, 1], block_shape=[BLOCK_Q, 128], padding_option="zero")
    o_desc = tl.make_tensor_descriptor(O_ptr, shape=[B * H * S_len, 128], strides=[128, 1], block_shape=[BLOCK_Q, 128], padding_option="zero")
    
    q_desc.load_mbarrier((q_offset, 0), Q_buf, 0, mbarrier_ptr[0])
    do_desc.load_mbarrier((q_offset, 0), dO_buf, 0, mbarrier_ptr[0])
    o_desc.load_mbarrier((q_offset, 0), O_buf, 0, mbarrier_ptr[0])
    
    num_k_tiles = tl.cdiv(S_len, BLOCK_K)
    k_offset_0 = 0
    if k_offset_0 < S_len:
        k_desc.load_mbarrier((k_offset_0, 0), K_buf, 0, mbarrier_ptr[0])
        v_desc.load_mbarrier((k_offset_0, 0), V_buf, 0, mbarrier_ptr[0])
    
    wait_commit(0)
    
    L_q = L_ptr_flat[(b_h * S_len + q_rows_start) : (b_h * S_len + q_rows_start + BLOCK_Q)]
    L_q_expanded = L_q[:, None]
    
    acc_dQ = tl.empty([2, BLOCK_Q, 64], tl.float32, init=0.0)
    
    for k_tile in tl.range(0, num_k_tiles, NUM_SMS, flatten=False, warp_specialize=WARP_SPECIALIZE):
        buf_idx = k_tile % 2
        next_buf_idx = (k_tile + 1) % 2
        
        k_offset = k_tile * BLOCK_K
        next_k_offset = (k_tile + 1) * BLOCK_K
        
        if next_k_offset < S_len:
            k_desc.load_mbarrier((next_k_offset, 0), K_buf, next_buf_idx, mbarrier_ptr[next_buf_idx])
            v_desc.load_mbarrier((next_k_offset, 0), V_buf, next_buf_idx, mbarrier_ptr[next_buf_idx])
        
        wait_commit(buf_idx)
        
        K_sub = read_swizzled(K_buf, k_tile % 2, (128, 64))
        V_sub = read_swizzled(V_buf, k_tile % 2, (128, 64))
        Q_sub = read_swizzled(Q_buf, k_tile % 2, (128, 64))
        dO_sub = read_swizzled(dO_buf, k_tile % 2, (128, 64))
        
        S_acc, d_acc = compute_S_and_d(Q_sub, dO_sub, K_sub, V_sub)
        
        S_scaled = S_acc * sqrt_d
        k_rows_start = k_tile * BLOCK_K
        k_rows = k_rows_start + tl.arange(0, BLOCK_K)
        k_rows_expanded = k_rows[None, :]
        valid_mask = k_rows_expanded < S_len
        S_scaled = tl.where(valid_mask, S_scaled, -1e20)
        
        P = tl.exp(S_scaled - L_q_expanded)
        P = tl.where(valid_mask, P, 0.0)
        
        ds = d_acc * P
        
        for i in range(2):
            acc_dQ[i] += tl.dot(ds, K_sub[i + 2])
            
    dQ_desc = tl.make_tensor_descriptor(dQ_ptr, shape=[B * H * S_len, 128], strides=[128, 1], block_shape=[BLOCK_Q, 128], padding_option="zero")
    
    for i in range(2):
        dQ_desc.store((q_offset, i * 64), (acc_dQ[i] * sqrt_d).to(tl.bfloat16))


@triton.jit
def bwd_dk_dv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr_flat, dK_ptr, dV_ptr,
    S_len, sqrt_d,
):
    NUM_SMS: tl.constexpr = 132
    NUM_STAGES: tl.constexpr = 2
    WARP_SPECIALIZE: tl.constexpr = True
    HEAD_DIM: tl.constexpr = 128
    BLOCK_Q: tl.constexpr = 256
    BLOCK_K: tl.constexpr = 128

    k_tile = tl.program_id(0)
    b_h = tl.program_id(1)
    k_offset = k_tile * BLOCK_K
    
    if k_offset >= S_len:
        return

    step = NUM_SMS * BLOCK_K
    k_offset += tl.program_id(2) * step
    
    if k_offset >= S_len:
        return

    K_buf = tl.empty([NUM_STAGES, BLOCK_K, HEAD_DIM], tl.bfloat16)
    V_buf = tl.empty([NUM_STAGES, BLOCK_K, HEAD_DIM], tl.bfloat16)
    Q_buf = tl.empty([NUM_STAGES, BLOCK_Q, HEAD_DIM], tl.bfloat16)
    dO_buf = tl.empty([NUM_STAGES, BLOCK_Q, HEAD_DIM], tl.bfloat16)
    
    mbarrier_ptr = [0] * NUM_STAGES
    commit_count = [0] * NUM_STAGES
    for i in range(NUM_STAGES):
        mbarrier_ptr[i] = tl.allocate_barrier(1)
        commit_count[i] = 0
    
    q_desc = tl.make_tensor_descriptor(Q_ptr, shape=[B * H * S_len, 128], strides=[128, 1], block_shape=[BLOCK_Q, 128], padding_option="zero")
    k_desc = tl.make_tensor_descriptor(K_ptr, shape=[B * H * S_len, 128], strides=[128, 1], block_shape=[BLOCK_K, 128], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(V_ptr, shape=[B * H * S_len, 128], strides=[128, 1], block_shape=[BLOCK_K, 128], padding_option="zero")
    do_desc = tl.make_tensor_descriptor(dO_ptr, shape=[B * H * S_len, 128], strides=[128, 1], block_shape=[BLOCK_Q, 128], padding_option="zero")
    
    k_desc.load_mbarrier((k_offset, 0), K_buf, 0, mbarrier_ptr[0])
    v_desc.load_mbarrier((k_offset, 0), V_buf, 0, mbarrier_ptr[0])
    
    num_q_tiles = tl.cdiv(S_len, BLOCK_Q)
    if 0 < num_q_tiles:
        q_desc.load_mbarrier((0, 0), Q_buf, 0, mbarrier_ptr[0])
        do_desc.load_mbarrier((0, 0), dO_buf, 0, mbarrier_ptr[0])
    
    wait_commit(0)
    
    L_k = L_ptr_flat[(b_h * S_len + k_offset) : (b_h * S_len + k_offset + BLOCK_K)]
    L_k_expanded = L_k[:, None]
    
    acc_dK = tl.empty([2, BLOCK_K, 64], tl.float32, init=0.0)
    acc_dV = tl.empty([2, BLOCK_K, 64], tl.float32, init=0.0)
    
    for q_tile in tl.range(0, num_q_tiles, NUM_SMS, flatten=False, warp_specialize=WARP_SPECIALIZE):
        buf_idx = q_tile % 2
        next_buf_idx = (q_tile + 1) % 2
        
        q_offset = q_tile * BLOCK_Q
        next_q_offset = (q_tile + 1) * BLOCK_Q
        
        if next_q_offset < S_len:
            q_desc.load_mbarrier((next_q_offset, 0), Q_buf, next_buf_idx, mbarrier_ptr[next_buf_idx])
            do_desc.load_mbarrier((next_q_offset, 0), dO_buf, next_buf_idx, mbarrier_ptr[next_buf_idx])
        
        wait_commit(buf_idx)
        
        Q_sub = read_swizzled(Q_buf, buf_idx, (256, 64))
        dO_sub = read_swizzled(dO_buf, buf_idx, (256, 64))
        K_sub = read_swizzled(K_buf, buf_idx, (128, 64))
        V_sub = read_swizzled(V_buf, buf_idx, (128, 64))
        
        S_acc, d_acc = compute_S_and_d_transposed(K_sub, V_sub, Q_sub, dO_sub)
        
        S_scaled = S_acc * sqrt_d
        q_rows_start = q_tile * BLOCK_Q
        q_rows = q_rows_start + tl.arange(0, BLOCK_Q)
        q_rows_expanded = q_rows[None, :]
        valid_mask = q_rows_expanded < S_len
        S_scaled = tl.where(valid_mask, S_scaled, -1e20)
        
        P = tl.exp(S_scaled - L_k_expanded)
        P = tl.where(valid_mask, P, 0.0)
        
        ds = d_acc * P
        
        for i in range(2):
            acc_dK[i] += tl.dot(ds[:, i*64:(i+1)*64].T, Q_sub[i])
            acc_dV[i] += tl.dot(P[:, i*64:(i+1)*64].T, dO_sub[i])
            
    dK_desc = tl.make_tensor_descriptor(dK_ptr, shape=[B * H * S_len, 128], strides=[128, 1], block_shape=[BLOCK_K, 128], padding_option="zero")
    dV_desc = tl.make_tensor_descriptor(dV_ptr, shape=[B * H * S_len, 128], strides=[128, 1], block_shape=[BLOCK_K, 128], padding_option="zero")
    
    for i in range(2):
        dK_desc.store((k_offset, i * 64), (acc_dK[i] * sqrt_d).to(tl.bfloat16))
        dV_desc.store((k_offset, i * 64), acc_dV[i].to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S_len, d = Q.shape
    sqrt_d = 1.0 / math.sqrt(d)
    
    Q_ptr = Q.flatten(0, 2)
    K_ptr = K.flatten(0, 2)
    V_ptr = V.flatten(0, 2)
    O_ptr = O.flatten(0, 2)
    dO_ptr = dO.flatten(0, 2)
    L_ptr = L.flatten(0, 1)
    dQ_ptr = dQ.flatten(0, 2)
    dK_ptr = dK.flatten(0, 2)
    dV_ptr = dV.flatten(0, 2)
    
    grid_dq = (triton.cdiv(S_len, 256), B * H, 1)
    grid_dk_dv = (triton.cdiv(S_len, 128), B * H, 1)

    bwd_dq_kernel[grid_dq](
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
        S_len, sqrt_d,
        num_warps=8, num_stages=2)
    
    bwd_dk_dv_kernel[grid_dk_dv](
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
        S_len, sqrt_d,
        num_warps=8, num_stages=2)