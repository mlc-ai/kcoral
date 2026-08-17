import torch
import triton
import triton.language as tl
import math


@triton.jit
def transpose_dot(A, B):
    return A @ B.T


@triton.jit
def backward_dK_dV_kernel(
    Q_block_ptr, K_block_ptr, V_block_ptr, O_block_ptr, dO_block_ptr, L_block_ptr, 
    dK_block_ptr, dV_block_ptr,
    seq_len, B, H,
    inv_sqrt_d: tl.constexpr,
    BLOCK: tl.constexpr,
):
    bid_j = tl.program_id(0)
    bid_h = tl.program_id(1)
    bid_b = tl.program_id(2)
    
    j_start = bid_j * BLOCK
    k_row_indices = j_start + tl.arange(0, BLOCK)
    d_indices = tl.arange(0, 128)
    
    valid_k_rows = k_row_indices[:, None] < seq_len
    valid_d_cols = d_indices[None, :] < 128
    mask_K = valid_k_rows & valid_d_cols
    
    K_block_ptr = tl.advance(K_block_ptr, [0, 0, j_start, 0])
    V_block_ptr = tl.advance(V_block_ptr, [0, 0, j_start, 0])
    dK_block_ptr = tl.advance(dK_block_ptr, [0, 0, j_start, 0])
    dV_block_ptr = tl.advance(dV_block_ptr, [0, 0, j_start, 0])
    
    K_tile = tl.load(K_block_ptr, mask=mask_K)
    V_tile = tl.load(V_block_ptr, mask=mask_K)
    
    dK_acc = tl.zeros((BLOCK, 128), tl.float32)
    dV_acc = tl.zeros((BLOCK, 128), tl.float32)
    
    for i in range(0, seq_len, BLOCK):
        q_row_indices = i + tl.arange(0, BLOCK)
        valid_q_rows = q_row_indices[:, None] < seq_len
        mask_Q = valid_q_rows & valid_d_cols
        
        Q_block_ptr_i = tl.advance(Q_block_ptr, [0, 0, i, 0])
        dO_block_ptr_i = tl.advance(dO_block_ptr, [0, 0, i, 0])
        O_block_ptr_i = tl.advance(O_block_ptr, [0, 0, i, 0])
        
        Q_tile = tl.load(Q_block_ptr_i, mask=mask_Q)
        dO_tile = tl.load(dO_block_ptr_i, mask=mask_Q)
        O_tile = tl.load(O_block_ptr_i, mask=mask_Q)
        
        L_block_ptr_i = tl.advance(L_block_ptr, [0, 0, i])
        valid_i_rows = q_row_indices[:, None] < seq_len
        L_i = tl.load(L_block_ptr_i, mask=valid_i_rows)
        
        dO_fp32 = dO_tile.to(tl.float32)
        O_fp32 = O_tile.to(tl.float32)
        D_i = tl.sum(dO_fp32 * O_fp32, axis=1)  # Shape: [BLOCK]
        
        S = transpose_dot(Q_tile, K_tile) * inv_sqrt_d
        P = tl.exp(S - L_i[:, None])
        
        dP = transpose_dot(dO_tile, V_tile)
        dS = P * (dP - D_i[:, None])
        
        dK_acc = tl.dot(dS.T, Q_tile, dK_acc)
        dV_acc = tl.dot(P.T, dO_tile, dV_acc)
        
    dK_scaled = (dK_acc * inv_sqrt_d).to(tl.bfloat16)
    dV_scaled = dV_acc.to(tl.bfloat16)
    
    tl.store(dK_block_ptr, dK_scaled, mask=mask_K)
    tl.store(dV_block_ptr, dV_scaled, mask=mask_K)


@triton.jit
def backward_dQ_kernel(
    Q_block_ptr, K_block_ptr, V_block_ptr, O_block_ptr, dO_block_ptr, L_block_ptr, dQ_block_ptr,
    seq_len, B, H,
    inv_sqrt_d: tl.constexpr,
    BLOCK: tl.constexpr,
):
    bid_i = tl.program_id(0)
    bid_h = tl.program_id(1)
    bid_b = tl.program_id(2)
    
    i_start = bid_i * BLOCK
    q_row_indices = i_start + tl.arange(0, BLOCK)
    d_indices = tl.arange(0, 128)
    
    valid_q_rows = q_row_indices[:, None] < seq_len
    valid_d_cols = d_indices[None, :] < 128
    mask_Q = valid_q_rows & valid_d_cols
    
    Q_block_ptr = tl.advance(Q_block_ptr, [0, 0, i_start, 0])
    dO_block_ptr = tl.advance(dO_block_ptr, [0, 0, i_start, 0])
    O_block_ptr = tl.advance(O_block_ptr, [0, 0, i_start, 0])
    dQ_block_ptr = tl.advance(dQ_block_ptr, [0, 0, i_start, 0])
    
    Q_tile = tl.load(Q_block_ptr, mask=mask_Q)
    dO_tile = tl.load(dO_block_ptr, mask=mask_Q)
    O_tile = tl.load(O_block_ptr, mask=mask_Q)
    
    L_block_ptr = tl.advance(L_block_ptr, [0, 0, i_start])
    valid_i_rows = q_row_indices[:, None] < seq_len
    L_i = tl.load(L_block_ptr, mask=valid_i_rows)
    
    dO_fp32 = dO_tile.to(tl.float32)
    O_fp32 = O_tile.to(tl.float32)
    D_i = tl.sum(dO_fp32 * O_fp32, axis=1)
    
    dQ_acc = tl.zeros((BLOCK, 128), tl.float32)
    
    for j in range(0, seq_len, BLOCK):
        k_row_indices = j + tl.arange(0, BLOCK)
        valid_k_rows = k_row_indices[:, None] < seq_len
        mask_K = valid_k_rows & valid_d_cols
        
        K_block_ptr_j = tl.advance(K_block_ptr, [0, 0, j, 0])
        V_block_ptr_j = tl.advance(V_block_ptr, [0, 0, j, 0])
        
        K_tile = tl.load(K_block_ptr_j, mask=mask_K)
        V_tile = tl.load(V_block_ptr_j, mask=mask_K)
        
        S = transpose_dot(Q_tile, K_tile) * inv_sqrt_d
        P = tl.exp(S - L_i[:, None])
        
        dP = transpose_dot(dO_tile, V_tile)
        dS = P * (dP - D_i[:, None])
        
        dQ_acc = tl.dot(dS, K_tile, dQ_acc)
        
    dQ_scaled = (dQ_acc * inv_sqrt_d).to(tl.bfloat16)
    tl.store(dQ_block_ptr, dQ_scaled, mask=mask_Q)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    inv_sqrt_d = 1.0 / math.sqrt(d)
    BLOCK = 128
    
    dQ.zero_()
    dK.zero_()
    dV.zero_()
    
    q_strides = Q.stride()
    Q_block_ptr = tl.make_block_ptr(
        base=Q, shape=[B, H, S, 128], 
        strides=[q_strides[0], q_strides[1], q_strides[2], q_strides[3]], 
        block_shape=[1, 1, BLOCK, 128], order=[3, 2, 1, 0]
    )
    
    k_strides = K.stride()
    K_block_ptr = tl.make_block_ptr(
        base=K, shape=[B, H, S, 128], 
        strides=[k_strides[0], k_strides[1], k_strides[2], k_strides[3]], 
        block_shape=[1, 1, BLOCK, 128], order=[3, 2, 1, 0]
    )
    
    v_strides = V.stride()
    V_block_ptr = tl.make_block_ptr(
        base=V, shape=[B, H, S, 128], 
        strides=[v_strides[0], v_strides[1], v_strides[2], v_strides[3]], 
        block_shape=[1, 1, BLOCK, 128], order=[3, 2, 1, 0]
    )
    
    o_strides = O.stride()
    O_block_ptr = tl.make_block_ptr(
        base=O, shape=[B, H, S, 128], 
        strides=[o_strides[0], o_strides[1], o_strides[2], o_strides[3]], 
        block_shape=[1, 1, BLOCK, 128], order=[3, 2, 1, 0]
    )
    
    do_strides = dO.stride()
    dO_block_ptr = tl.make_block_ptr(
        base=dO, shape=[B, H, S, 128], 
        strides=[do_strides[0], do_strides[1], do_strides[2], do_strides[3]], 
        block_shape=[1, 1, BLOCK, 128], order=[3, 2, 1, 0]
    )
    
    l_strides = L.stride()
    L_block_ptr = tl.make_block_ptr(
        base=L, shape=[B, H, S], 
        strides=[l_strides[0], l_strides[1], l_strides[2]], 
        block_shape=[1, 1, BLOCK], order=[2, 1, 0]
    )
    
    dq_strides = dQ.stride()
    dQ_block_ptr = tl.make_block_ptr(
        base=dQ, shape=[B, H, S, 128], 
        strides=[dq_strides[0], dq_strides[1], dq_strides[2], dq_strides[3]], 
        block_shape=[1, 1, BLOCK, 128], order=[3, 2, 1, 0]
    )
    
    dk_strides = dK.stride()
    dK_block_ptr = tl.make_block_ptr(
        base=dK, shape=[B, H, S, 128], 
        strides=[dk_strides[0], dk_strides[1], dk_strides[2], dk_strides[3]], 
        block_shape=[1, 1, BLOCK, 128], order=[3, 2, 1, 0]
    )
    
    dv_strides = dV.stride()
    dV_block_ptr = tl.make_block_ptr(
        base=dV, shape=[B, H, S, 128], 
        strides=[dv_strides[0], dv_strides[1], dv_strides[2], dv_strides[3]], 
        block_shape=[1, 1, BLOCK, 128], order=[3, 2, 1, 0]
    )
    
    grid = (triton.cdiv(S, BLOCK), H, B)
    backward_dK_dV_kernel[grid](
        Q_block_ptr, K_block_ptr, V_block_ptr, O_block_ptr, dO_block_ptr, L_block_ptr, 
        dK_block_ptr, dV_block_ptr, S, B, H, inv_sqrt_d=inv_sqrt_d, BLOCK=BLOCK, 
        num_warps=8, num_stages=2
    )
    
    backward_dQ_kernel[grid](
        Q_block_ptr, K_block_ptr, V_block_ptr, O_block_ptr, dO_block_ptr, L_block_ptr, dQ_block_ptr,
        S, B, H, inv_sqrt_d=inv_sqrt_d, BLOCK=BLOCK, num_warps=8, num_stages=2
    )