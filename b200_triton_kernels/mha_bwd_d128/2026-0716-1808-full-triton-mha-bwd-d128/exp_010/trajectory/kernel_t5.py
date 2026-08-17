import torch
import triton
import triton.language as tl


@triton.jit
def dKdV_kernel(
    desc_Q, desc_K, desc_V, desc_O, desc_dO, L_ptr,
    desc_dK, desc_dV,
    S, scale, BLOCK_M: tl.constexpr,
):
    j = tl.program_id(0)
    bh = tl.program_id(1)
    
    off_j = j * BLOCK_M
    
    k_tile = desc_K.load([bh * S + off_j, 0])
    v_tile = desc_V.load([bh * S + off_j, 0])
    
    acc_dK = tl.zeros((BLOCK_M, 128), tl.float32)
    acc_dV = tl.zeros((BLOCK_M, 128), tl.float32)
    
    num_blocks = S // BLOCK_M
    
    for i in range(num_blocks):
        off_i = i * BLOCK_M
        
        q_tile = desc_Q.load([bh * S + off_i, 0])
        do_tile = desc_dO.load([bh * S + off_i, 0])
        o_tile  = desc_O.load([bh * S + off_i, 0])
        
        d_i = tl.sum(do_tile * o_tile, axis=1)
        
        l_i = tl.load(L_ptr + bh * S + off_i + tl.arange(0, BLOCK_M), 
                      mask=(off_i + tl.arange(0, BLOCK_M)) < S, other=-float('inf'))
        
        acc_s = tl.dot(q_tile, k_tile.T)
        
        acc_dp = tl.dot(do_tile, v_tile.T)
        
        s_tmp = acc_s * scale
        
        p_tmp = tl.exp(s_tmp - l_i[:, None])
        
        row_offs_i = off_i + tl.arange(0, BLOCK_M)
        col_offs_j = off_j + tl.arange(0, BLOCK_M)
        pos_mask = (row_offs_i[:, None] < S) & (col_offs_j[None, :] < S)
        
        p_tmp = tl.where(pos_mask, p_tmp, 0.0)
        
        d_i_expanded = d_i[:, None]
        
        ds_tmp = p_tmp * (acc_dp - d_i_expanded) * scale
        
        ds_tmp = tl.where(pos_mask, ds_tmp, 0.0)
        
        p_tmp = p_tmp.to(tl.bfloat16)
        ds_tmp = ds_tmp.to(tl.bfloat16)
        
        acc_dV = tl.dot(p_tmp.T, do_tile, acc_dV)
        
        acc_dK = tl.dot(ds_tmp.T, q_tile, acc_dK)
        
    desc_dK.load_store([bh * S + off_j, 0], acc_dK.to(tl.bfloat16))
    desc_dV.load_store([bh * S + off_j, 0], acc_dV.to(tl.bfloat16))


@triton.jit
def dQ_kernel(
    desc_Q, desc_K, desc_V, desc_O, desc_dO, L_ptr, desc_dQ,
    S, scale, BLOCK_M: tl.constexpr,
):
    i = tl.program_id(0)
    bh = tl.program_id(1)
    
    off_i = i * BLOCK_M
    
    q_tile = desc_Q.load([bh * S + off_i, 0])
    do_tile = desc_dO.load([bh * S + off_i, 0])
    o_tile = desc_O.load([bh * S + off_i, 0])
    
    d_i = tl.sum(do_tile * o_tile, axis=1)
    
    l_i = tl.load(L_ptr + bh * S + off_i + tl.arange(0, BLOCK_M), mask=(off_i + tl.arange(0, BLOCK_M)) < S, other=-float('inf'))
    
    acc_dq = tl.zeros((BLOCK_M, 128), tl.float32)
    
    num_blocks = S // BLOCK_M
    
    for j in range(num_blocks):
        off_j = j * BLOCK_M
        
        k_tile = desc_K.load([bh * S + off_j, 0])
        v_tile = desc_V.load([bh * S + off_j, 0])
        
        acc_s = tl.dot(q_tile, k_tile.T)
        
        acc_dp = tl.dot(do_tile, v_tile.T)
        
        s_tmp = acc_s * scale
        
        p_tmp = tl.exp(s_tmp - l_i[:, None])
        
        row_offs_i = off_i + tl.arange(0, BLOCK_M)
        col_offs_j = off_j + tl.arange(0, BLOCK_M)
        pos_mask = (row_offs_i[:, None] < S) & (col_offs_j[None, :] < S)
        
        p_tmp = tl.where(pos_mask, p_tmp, 0.0)
        
        d_i_expanded = d_i[:, None]
        
        ds_tmp = p_tmp * (acc_dp - d_i_expanded) * scale
        
        ds_tmp = tl.where(pos_mask, ds_tmp, 0.0)
        
        ds_tmp = ds_tmp.to(tl.bfloat16)
        
        acc_dq = tl.dot(ds_tmp, k_tile, acc_dq)
        
    desc_dQ.load_store([bh * S + off_i, 0], acc_dq.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Computes the backward pass of multi-head attention natively on CUDA."""
    device = torch.cuda.current_device()
    torch.cuda.set_device(device)
    
    B, H, S, d = Q.shape
    
    Q = Q.view(-1, Q.shape[-1])
    K = K.view(-1, K.shape[-1])
    V = V.view(-1, V.shape[-1])
    O = O.view(-1, O.shape[-1])
    dO = dO.view(-1, dO.shape[-1])
    dQ = dQ.view(-1, dQ.shape[-1])
    dK = dK.view(-1, dK.shape[-1])
    dV = dV.view(-1, dV.shape[-1])
    
    BLOCK_M = 64
    
    desc_a_Q = TensorDescriptor.from_tensor(Q, [BLOCK_M, BLOCK_M])
    desc_a_K = TensorDescriptor.from_tensor(K, [BLOCK_M, BLOCK_M])
    desc_a_V = TensorDescriptor.from_tensor(V, [BLOCK_M, BLOCK_M])
    desc_a_O = TensorDescriptor.from_tensor(O, [BLOCK_M, BLOCK_M])
    desc_a_dO = TensorDescriptor.from_tensor(dO, [BLOCK_M, BLOCK_M])
    
    desc_c_dQ = TensorDescriptor.from_tensor(dQ, [BLOCK_M, BLOCK_M])
    desc_c_dK = TensorDescriptor.from_tensor(dK, [BLOCK_M, BLOCK_M])
    desc_c_dV = TensorDescriptor.from_tensor(dV, [BLOCK_M, BLOCK_M])
    
    scale = 1.0 / math.sqrt(d)
    
    grid = (S // BLOCK_M, B * H)
    
    dKdV_kernel[grid](
        desc_a_Q, desc_a_K, desc_a_V, desc_a_O, desc_a_dO, L,
        desc_c_dK, desc_c_dV,
        S, scale, BLOCK_M, num_warps=16, num_stages=2
    )
    dQ_kernel[grid](
        desc_a_Q, desc_a_K, desc_a_V, desc_a_O, desc_a_dO, L, desc_c_dQ,
        S, scale, BLOCK_M, num_warps=16, num_stages=2
    )