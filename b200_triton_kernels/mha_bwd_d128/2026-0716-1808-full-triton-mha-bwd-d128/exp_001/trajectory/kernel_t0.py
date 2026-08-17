import torch
import triton
import triton.language as tl


@triton.jit
def _bwd_dq_kernel(
    Q, K, V, O, dO, L, dQ,
    S_seq,
    stride_q_bh, stride_q_s, stride_q_d,
    stride_k_bh, stride_k_s, stride_k_d,
    stride_v_bh, stride_v_s, stride_v_d,
    stride_o_bh, stride_o_s, stride_o_d,
    stride_do_bh, stride_do_s, stride_do_d,
    stride_l_bh, stride_l_s,
    stride_dq_bh, stride_dq_s, stride_dq_d,
    BLOCK_S: tl.constexpr,
):
    i = tl.program_id(0)
    bh = tl.program_id(1)
    
    current_i_offset = bh * stride_q_bh + i * stride_q_s
    q_row = tl.load(Q + current_i_offset + tl.arange(0, 128) * stride_q_d, other=0.0)
    
    do_row = tl.load(dO + current_i_offset + tl.arange(0, 128) * stride_do_d, other=0.0)
    
    o_row = tl.load(O + current_i_offset + tl.arange(0, 128) * stride_o_d, other=0.0)
    e_val = (do_row * o_row).sum()
    
    l_val = tl.load(L + bh * stride_l_bh + (i * stride_l_s), other=0.0)
    
    dq_acc = tl.zeros((128,), tl.float32)
    
    num_blocks = tl.cdiv(S_seq, BLOCK_S)
    scale = 1.0 / tl.sqrt(128)
    
    for j_blk in range(num_blocks):
        j_start = j_blk * BLOCK_S
        
        dp_acc = tl.zeros((BLOCK_S,), tl.float32)
        
        current_j_offset_v = bh * stride_v_bh + j_start * stride_v_s
        v_block = tl.load(V + current_j_offset_v + tl.arange(0, BLOCK_S)[:, None] * stride_v_s + tl.arange(0, 128)[None, :] * stride_v_d,
                          mask=(j_start + tl.arange(0, BLOCK_S)[:, None] < S_seq), other=0.0)
        
        for r in range(0, 128, 64):
            do_row_part = do_row[r:r+64]
            v_block_part = v_block[:, r:r+64]
            dp_acc += (v_block_part * do_row_part[None, :]).sum(1)
        
        current_j_offset_k = bh * stride_k_bh + j_start * stride_k_s
        k_block = tl.load(K + current_j_offset_k + tl.arange(0, BLOCK_S)[:, None] * stride_k_s + tl.arange(0, 128)[None, :] * stride_k_d,
                          mask=(j_start + tl.arange(0, BLOCK_S)[:, None] < S_seq), other=0.0)
        
        s_row = tl.dot(q_row[None, :], k_block.T) * scale
        p_row = tl.exp(s_row - l_val)
        
        d_row = dp_acc * p_row
        
        dq_acc += tl.dot(d_row[None, :], k_block) * scale
    
    out_ptr = dQ + bh * stride_dq_bh + i * stride_dq_s + tl.arange(0, 128) * stride_dq_d
    tl.store(out_ptr, dq_acc, mask=(i < S_seq))


@triton.jit
def _bwd_dkv_kernel(
    Q, K, V, O, dO, L, dK, dV,
    S_seq,
    stride_q_bh, stride_q_s, stride_q_d,
    stride_k_bh, stride_k_s, stride_k_d,
    stride_v_bh, stride_v_s, stride_v_d,
    stride_o_bh, stride_o_s, stride_o_d,
    stride_do_bh, stride_do_s, stride_do_d,
    stride_l_bh, stride_l_s,
    stride_dk_bh, stride_dk_s, stride_dk_d,
    stride_dv_bh, stride_dv_s, stride_dv_d,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    bh = tl.program_id(1)
    
    j_blk = pid_n // BLOCK_S
    s_offs = (pid_n % BLOCK_S) + tl.arange(0, BLOCK_S)
    
    current_j_offset = bh * stride_k_bh + j_blk * BLOCK_S * stride_k_s + s_offs[:, None] * stride_k_s
    k_row = tl.load(K + current_j_offset + tl.arange(0, 128)[None, :] * stride_k_d, 
                    mask=(j_blk * BLOCK_S + s_offs[:, None] < S_seq), other=0.0)
    
    current_j_offset_v = bh * stride_v_bh + j_blk * BLOCK_S * stride_v_s + s_offs[:, None] * stride_v_s
    v_row = tl.load(V + current_j_offset_v + tl.arange(0, 128)[None, :] * stride_v_d,
                    mask=(j_blk * BLOCK_S + s_offs[:, None] < S_seq), other=0.0)
    
    dk_acc = tl.zeros((BLOCK_S, 128), tl.float32)
    dv_acc = tl.zeros((BLOCK_S, 128), tl.float32)
    
    num_blocks = tl.cdiv(S_seq, BLOCK_S)
    scale = 1.0 / tl.sqrt(128)
    
    for i_blk in range(num_blocks):
        i_start = i_blk * BLOCK_S
        
        current_i_offset_q = bh * stride_q_bh + i_start * stride_q_s
        q_block = tl.load(Q + current_i_offset_q + tl.arange(0, BLOCK_S)[:, None] * stride_q_s + tl.arange(0, 128)[None, :] * stride_q_d,
                          mask=(i_start + tl.arange(0, BLOCK_S)[:, None] < S_seq), other=0.0)
        
        current_i_offset_do = bh * stride_do_bh + i_start * stride_do_s
        do_block = tl.load(dO + current_i_offset_do + tl.arange(0, BLOCK_S)[:, None] * stride_do_s + tl.arange(0, 128)[None, :] * stride_do_d,
                           mask=(i_start + tl.arange(0, BLOCK_S)[:, None] < S_seq), other=0.0)
        
        dp_acc = tl.zeros((BLOCK_S,), tl.float32)
        for r_c in range(0, 128, 64):
            do_block_part = do_block[:, r_c:r_c+64]
            v_row_part = v_row[:, r_c:r_c+64]
            dp_acc += (do_block_part * v_row_part[None, :]).sum(1)
        
        current_i_offset_o = bh * stride_o_bh + i_start * stride_o_s
        o_block = tl.load(O + current_i_offset_o + tl.arange(0, BLOCK_S)[:, None] * stride_o_s + tl.arange(0, 128)[None, :] * stride_o_d,
                          mask=(i_start + tl.arange(0, BLOCK_S)[:, None] < S_seq), other=0.0)
        
        e_row = (do_block * o_block).sum(1)
        
        s_row = tl.dot(q_block, k_row.T) * scale
        
        current_i_offset_l = bh * stride_l_bh + (i_start + tl.arange(0, BLOCK_S)) * stride_l_s
        l_row = tl.load(L + current_i_offset_l, mask=(i_start + tl.arange(0, BLOCK_S) < S_seq), other=0.0)
        
        p_row = tl.exp(s_row - l_row)
        
        ds_row = dp_acc * p_row
        
        dk_acc += (ds_row[:, None] * q_block)
        dv_acc += (p_row[:, None] * do_block)
    
    out_ptr_dk = dK + bh * stride_dk_bh + (j_blk * BLOCK_S + s_offs[:, None]) * stride_dk_s + tl.arange(0, 128)[None, :] * stride_dk_d
    tl.store(out_ptr_dk, dk_acc * scale, mask=((j_blk * BLOCK_S + s_offs[:, None]) < S_seq))
    
    out_ptr_dv = dV + bh * stride_dv_bh + (j_blk * BLOCK_S + s_offs[:, None]) * stride_dv_s + tl.arange(0, 128)[None, :] * stride_dv_d
    tl.store(out_ptr_dv, dv_acc, mask=((j_blk * BLOCK_S + s_offs[:, None]) < S_seq))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute attention backward ``dQ, dK, dV`` in place."""
    torch.cuda.set_device(Q.device)
    S_seq = Q.shape[-2]
    block_size = 128
    
    grid_dq = (S_seq, Q.shape[0] * Q.shape[1])
    _bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, L, dQ, S_seq,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        dO.stride(0), dO.stride(1), dO.stride(2),
        L.stride(0), L.stride(1),
        dQ.stride(0), dQ.stride(1), dQ.stride(2),
        BLOCK_S=block_size,
        num_warps=4,
    )
    
    grid_dkv = (S_seq, K.shape[0] * K.shape[1])
    _bwd_dkv_kernel[grid_dkv](
        Q, K, V, O, dO, L, dK, dV, S_seq,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        dO.stride(0), dO.stride(1), dO.stride(2),
        L.stride(0), L.stride(1),
        dK.stride(0), dK.stride(1), dK.stride(2),
        dV.stride(0), dV.stride(1), dV.stride(2),
        BLOCK_S=block_size,
        num_warps=4,
    )