import torch
import triton
import triton.language as tl


@triton.jit
def load_chunk(ptr, bh, row_start, col_start, S):
    """Load a [16, 16] bf16 tile."""
    row_offs = row_start + tl.arange(0, 16)
    col_offs = col_start + tl.arange(0, 16)
    ptrs = ptr + (bh * S + row_offs[:, None]) * 128 + col_offs[None, :]
    mask = (row_offs[:, None] < S)
    return tl.load(ptrs, mask=mask, other=0.0)


@triton.jit
def store_chunk(ptr, bh, row_start, col_start, value, S):
    """Store a [16, 16] bf16 tile."""
    row_offs = row_start + tl.arange(0, 16)
    col_offs = col_start + tl.arange(0, 16)
    ptrs = ptr + (bh * S + row_offs[:, None]) * 128 + col_offs[None, :]
    mask = (row_offs[:, None] < S)
    tl.store(ptrs, value, mask=mask)


@triton.jit
def dKdV_kernel(Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
                S, scale):
    j = tl.program_id(0)
    bh = tl.program_id(1)
    
    k_base = load_chunk(K_ptr, bh, j * 16, 0, S)
    v_base = load_chunk(V_ptr, bh, j * 16, 0, S)
    
    k_tiles = [k_base]
    v_tiles = [v_base]
    for k_idx in range(16, 128, 16):
        k_tiles.append(load_chunk(K_ptr, bh, j * 16, k_idx, S))
        v_tiles.append(load_chunk(V_ptr, bh, j * 16, k_idx, S))
        
    acc_dv_list = [tl.zeros((16, 16), tl.float32) for _ in range(8)]
    acc_dk_list = [tl.zeros((16, 16), tl.float32) for _ in range(8)]
    
    num_blocks = tl.cdiv(S, 16)
    
    # Keep loop unroll factor moderate to prevent shared-memory overflow during load coalescing
    for i in range(num_blocks):
        q_base = load_chunk(Q_ptr, bh, i * 16, 0, S)
        do_base = load_chunk(dO_ptr, bh, i * 16, 0, S)
        o_base = load_chunk(O_ptr, bh, i * 16, 0, S)
        
        q_tiles = [q_base]
        do_tiles = [do_base]
        o_tiles = [o_base]
        for k_idx in range(16, 128, 16):
            q_tiles.append(load_chunk(Q_ptr, bh, i * 16, k_idx, S))
            do_tiles.append(load_chunk(dO_ptr, bh, i * 16, k_idx, S))
            o_tiles.append(load_chunk(O_ptr, bh, i * 16, k_idx, S))
            
        d_i = tl.sum(do_tiles[0] * o_tiles[0], axis=1)
        for k_idx in range(16, 128, 16):
            d_i += tl.sum(do_tiles[k_idx // 16] * o_tiles[k_idx // 16], axis=1)
            
        acc_s = tl.dot(q_tiles[0], k_tiles[0].T)
        acc_dp = tl.dot(do_tiles[0], v_tiles[0].T)
        
        for k_idx in range(16, 128, 16):
            acc_s += tl.dot(q_tiles[k_idx // 16], k_tiles[k_idx // 16].T)
            acc_dp += tl.dot(do_tiles[k_idx // 16], v_tiles[k_idx // 16].T)
            
        l_i = tl.load(L_ptr + bh * S + i * 16 + tl.arange(0, 16), mask=(i * 16 + tl.arange(0, 16)) < S, other=float('-inf'))
        
        s_tmp = acc_s * scale
        p_tmp = tl.exp(s_tmp - l_i[:, None])
        ds_tmp = p_tmp * (acc_dp - d_i[:, None]) * scale
        
        row_offs_i = i * 16 + tl.arange(0, 16)
        col_offs_j = j * 16 + tl.arange(0, 16)
        mask_i = (row_offs_i < S) & (row_offs_i >= i * 16)
        mask_j = (col_offs_j < S) & (col_offs_j >= j * 16)
        
        p_tmp = tl.where(mask_i[:, None] & mask_j[None, :], p_tmp, 0.0)
        ds_tmp = tl.where(mask_i[:, None] & mask_j[None, :], ds_tmp, 0.0)
        
        acc_dv_list[0] = tl.dot(p_tmp.T, do_tiles[0], acc_dv_list[0])
        acc_dk_list[0] = tl.dot(ds_tmp.T, q_tiles[0], acc_dk_list[0])
        
        for k_idx in range(16, 128, 16):
            idx = k_idx // 16
            acc_dv_list[idx] = tl.dot(p_tmp.T, do_tiles[idx], acc_dv_list[idx])
            acc_dk_list[idx] = tl.dot(ds_tmp.T, q_tiles[idx], acc_dk_list[idx])
            
        tile_loads_during_loop = False
        if tile_loads_during_loop:
            next_q_base = load_chunk(Q_ptr, bh, (i + 1) * 16, 0, S)
            next_do_base = load_chunk(dO_ptr, bh, (i + 1) * 16, 0, S)
            next_o_base = load_chunk(O_ptr, bh, (i + 1) * 16, 0, S)
            
    for k_idx in range(0, 128, 16):
        idx = k_idx // 16
        store_chunk(dK_ptr, bh, j * 16, k_idx, acc_dk_list[idx].to(tl.bfloat16), S)
        store_chunk(dV_ptr, bh, j * 16, k_idx, acc_dv_list[idx].to(tl.bfloat16), S)


@triton.jit
def dQ_kernel(Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
              S, scale):
    i = tl.program_id(0)
    bh = tl.program_id(1)
    
    q_base = load_chunk(Q_ptr, bh, i * 16, 0, S)
    do_base = load_chunk(dO_ptr, bh, i * 16, 0, S)
    o_base = load_chunk(O_ptr, bh, i * 16, 0, S)
    
    q_tiles = [q_base]
    do_tiles = [do_base]
    o_tiles = [o_base]
    for k_idx in range(16, 128, 16):
        q_tiles.append(load_chunk(Q_ptr, bh, i * 16, k_idx, S))
        do_tiles.append(load_chunk(dO_ptr, bh, i * 16, k_idx, S))
        o_tiles.append(load_chunk(O_ptr, bh, i * 16, k_idx, S))
        
    d_i = tl.sum(do_tiles[0] * o_tiles[0], axis=1)
    for k_idx in range(16, 128, 16):
        d_i += tl.sum(do_tiles[k_idx // 16] * o_tiles[k_idx // 16], axis=1)
        
    l_i = tl.load(L_ptr + bh * S + i * 16 + tl.arange(0, 16), mask=(i * 16 + tl.arange(0, 16)) < S, other=float('-inf'))
    
    acc_dq_list = [tl.zeros((16, 16), tl.float32) for _ in range(8)]
    
    num_blocks = tl.cdiv(S, 16)
    
    for j in range(num_blocks):
        k_base = load_chunk(K_ptr, bh, j * 16, 0, S)
        v_base = load_chunk(V_ptr, bh, j * 16, 0, S)
        
        k_tiles = [k_base]
        v_tiles = [v_base]
        for k_idx in range(16, 128, 16):
            k_tiles.append(load_chunk(K_ptr, bh, j * 16, k_idx, S))
            v_tiles.append(load_chunk(V_ptr, bh, j * 16, k_idx, S))
            
        acc_s = tl.dot(q_tiles[0], k_tiles[0].T)
        acc_dp = tl.dot(do_tiles[0], v_tiles[0].T)
        
        for k_idx in range(16, 128, 16):
            acc_s += tl.dot(q_tiles[k_idx // 16], k_tiles[k_idx // 16].T)
            acc_dp += tl.dot(do_tiles[k_idx // 16], v_tiles[k_idx // 16].T)
            
        s_tmp = acc_s * scale
        p_tmp = tl.exp(s_tmp - l_i[:, None])
        ds_tmp = p_tmp * (acc_dp - d_i[:, None]) * scale
        
        row_offs_i = i * 16 + tl.arange(0, 16)
        col_offs_j = j * 16 + tl.arange(0, 16)
        mask_i = (row_offs_i < S) & (row_offs_i >= i * 16)
        mask_j = (col_offs_j < S) & (col_offs_j >= j * 16)
        
        p_tmp = tl.where(mask_i[:, None] & mask_j[None, :], p_tmp, 0.0)
        ds_tmp = tl.where(mask_i[:, None] & mask_j[None, :], ds_tmp, 0.0)
        
        for k_idx in range(0, 128, 16):
            idx = k_idx // 16
            ds_chunk = ds_tmp[:, idx:(idx + 1)]
            ds_chunk = tl.reshape(ds_chunk, (16, 1))
            k_chunk = k_tiles[idx]
            acc_dq_list[idx] = tl.dot(ds_tmp, k_chunk, acc_dq_list[idx])
            
        tile_loads_during_loop = False
        if tile_loads_during_loop:
            next_k_base = load_chunk(K_ptr, bh, (j + 1) * 16, 0, S)
            next_v_base = load_chunk(V_ptr, bh, (j + 1) * 16, 0, S)

    for k_idx in range(0, 128, 16):
        idx = k_idx // 16
        store_chunk(dQ_ptr, bh, i * 16, k_idx, acc_dq_list[idx].to(tl.bfloat16), S)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Computes the backward pass of multi-head attention natively on CUDA."""
    device = torch.cuda.current_device()
    torch.cuda.set_device(device)
    
    if hasattr(torch, "distributed") and hasattr(torch.distributed, "is_initialized"):
        try:
            if torch.distributed.is_initialized():
                pass
        except Exception:
            pass

    B, H, S, d = Q.shape
    scale = 1.0 / (128 ** 0.5)
    
    grid = (triton.cdiv(S, 16), B * H)
    
    dKdV_kernel[grid](Q, K, V, O, dO, L, dK, dV, S, scale, num_warps=4, num_stages=2)
    dQ_kernel[grid](Q, K, V, O, dO, L, dQ, S, scale, num_warps=4, num_stages=2)