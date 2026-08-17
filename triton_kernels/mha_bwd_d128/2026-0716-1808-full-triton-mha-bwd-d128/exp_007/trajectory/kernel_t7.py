import math
import torch
import triton
import triton.language as tl


BLOCK = 64

@triton.jit
def bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
    S_len, d, total_S,
    s2, s3,
    BLOCK: tl.constexpr,
    scale: tl.constexpr,
):
    i = tl.program_id(0)
    b_h_idx = tl.program_id(1)
    offset_m = i * BLOCK
    
    q_desc = tl.make_tensor_descriptor(Q_ptr, shape=[total_S, d], strides=[s2, s3], block_shape=[BLOCK, BLOCK], padding_option="zero")
    k_desc = tl.make_tensor_descriptor(K_ptr, shape=[total_S, d], strides=[s2, s3], block_shape=[BLOCK, BLOCK], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(V_ptr, shape=[total_S, d], strides=[s2, s3], block_shape=[BLOCK, BLOCK], padding_option="zero")
    o_desc = tl.make_tensor_descriptor(O_ptr, shape=[total_S, d], strides=[s2, s3], block_shape=[BLOCK, BLOCK], padding_option="zero")
    do_desc = tl.make_tensor_descriptor(dO_ptr, shape=[total_S, d], strides=[s2, s3], block_shape=[BLOCK, BLOCK], padding_option="zero")
    dQ_desc = tl.make_tensor_descriptor(dQ_ptr, shape=[total_S, d], strides=[s2, s3], block_shape=[BLOCK, BLOCK], padding_option="zero")
    
    row_off = b_h_idx * S_len + offset_m
    
    q0 = q_desc.load([row_off, 0])
    q1 = q_desc.load([row_off, 64])
    
    o0 = o_desc.load([row_off, 0])
    o1 = o_desc.load([row_off, 64])
    
    do0 = do_desc.load([row_off, 0])
    do1 = do_desc.load([row_off, 64])
    
    D_i = tl.sum(o0 * do0, axis=-1, keep_dims=True) + tl.sum(o1 * do1, axis=-1, keep_dims=True)
    
    row = tl.arange(0, BLOCK)
    l_val = tl.load(L_ptr + b_h_idx * S_len + offset_m + row, mask=(offset_m + row < S_len), other=0.0)
    
    dQ_acc0 = tl.zeros((BLOCK, BLOCK), tl.float32)
    dQ_acc1 = tl.zeros((BLOCK, BLOCK), tl.float32)
    
    off = b_h_idx * S_len + 0
    cur_k = [k_desc.load([off, 0]), k_desc.load([off, 64])]
    cur_v = [v_desc.load([off, 0]), v_desc.load([off, 64])]
    
    for j in tl.range(0, S_len // BLOCK, num_stages=2):
        offset_n = j * BLOCK
        n_row_off = b_h_idx * S_len + offset_n
        
        if j + 1 < S_len // BLOCK:
            next_n_row_off = b_h_idx * S_len + (j + 1) * BLOCK
            next_k = [k_desc.load([next_n_row_off, 0]), k_desc.load([next_n_row_off, 64])]
            next_v = [v_desc.load([next_n_row_off, 0]), v_desc.load([next_n_row_off, 64])]
            
        s = tl.dot(q0, cur_k[0].T) + tl.dot(q1, cur_k[1].T)
        
        p = tl.exp(s * scale - l_val[:, None])
        
        dp = tl.dot(do0, cur_v[0].T) + tl.dot(do1, cur_v[1].T)
        
        ds = p * (dp - D_i) * scale
        
        dQ_acc0 = tl.dot(ds, cur_k[0], acc=dQ_acc0)
        dQ_acc1 = tl.dot(ds, cur_k[1], acc=dQ_acc1)
        
        cur_k, next_k = next_k, cur_k
        cur_v, next_v = next_v, cur_v
        
    dQ_desc.store([row_off, 0], dQ_acc0.to(tl.bfloat16))
    dQ_desc.store([row_off, 64], dQ_acc1.to(tl.bfloat16))


@triton.jit
def bwd_dkv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    S_len, d, total_S,
    s2, s3,
    BLOCK: tl.constexpr,
    scale: tl.constexpr,
):
    j = tl.program_id(0)
    b_h_idx = tl.program_id(1)
    offset_n = j * BLOCK
    
    q_desc = tl.make_tensor_descriptor(Q_ptr, shape=[total_S, d], strides=[s2, s3], block_shape=[BLOCK, BLOCK], padding_option="zero")
    k_desc = tl.make_tensor_descriptor(K_ptr, shape=[total_S, d], strides=[s2, s3], block_shape=[BLOCK, BLOCK], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(V_ptr, shape=[total_S, d], strides=[s2, s3], block_shape=[BLOCK, BLOCK], padding_option="zero")
    o_desc = tl.make_tensor_descriptor(O_ptr, shape=[total_S, d], strides=[s2, s3], block_shape=[BLOCK, BLOCK], padding_option="zero")
    do_desc = tl.make_tensor_descriptor(dO_ptr, shape=[total_S, d], strides=[s2, s3], block_shape=[BLOCK, BLOCK], padding_option="zero")
    dK_desc = tl.make_tensor_descriptor(dK_ptr, shape=[total_S, d], strides=[s2, s3], block_shape=[BLOCK, BLOCK], padding_option="zero")
    dV_desc = tl.make_tensor_descriptor(dV_ptr, shape=[total_S, d], strides=[s2, s3], block_shape=[BLOCK, BLOCK], padding_option="zero")
    
    n_row_off = b_h_idx * S_len + offset_n
    
    k0 = k_desc.load([n_row_off, 0])
    k1 = k_desc.load([n_row_off, 64])
    
    v0 = v_desc.load([n_row_off, 0])
    v1 = v_desc.load([n_row_off, 64])
    
    dK_acc0 = tl.zeros((BLOCK, BLOCK), tl.float32)
    dK_acc1 = tl.zeros((BLOCK, BLOCK), tl.float32)
    dV_acc0 = tl.zeros((BLOCK, BLOCK), tl.float32)
    dV_acc1 = tl.zeros((BLOCK, BLOCK), tl.float32)
    
    off = b_h_idx * S_len + 0
    cur_q = [q_desc.load([off, 0]), q_desc.load([off, 64])]
    cur_o = [o_desc.load([off, 0]), o_desc.load([off, 64])]
    cur_do = [do_desc.load([off, 0]), do_desc.load([off, 64])]
    
    for i in tl.range(0, S_len // BLOCK, num_stages=2):
        offset_m = i * BLOCK
        m_row_off = b_h_idx * S_len + offset_m
        
        if i + 1 < S_len // BLOCK:
            next_m_row_off = b_h_idx * S_len + (i + 1) * BLOCK
            next_q = [q_desc.load([next_m_row_off, 0]), q_desc.load([next_m_row_off, 64])]
            next_o = [o_desc.load([next_m_row_off, 0]), o_desc.load([next_m_row_off, 64])]
            next_do = [do_desc.load([next_m_row_off, 0]), do_desc.load([next_m_row_off, 64])]
            
        D_i = tl.sum(cur_o[0] * cur_do[0], axis=-1, keep_dims=True) + tl.sum(cur_o[1] * cur_do[1], axis=-1, keep_dims=True)
        
        row = tl.arange(0, BLOCK)
        l_val = tl.load(L_ptr + b_h_idx * S_len + offset_m + row, mask=(offset_m + row < S_len), other=0.0)
        
        s = tl.dot(cur_q[0], k0.T) + tl.dot(cur_q[1], k1.T)
        
        p = tl.exp(s * scale - l_val[:, None])
        
        dp = tl.dot(cur_do[0], v0.T) + tl.dot(cur_do[1], v1.T)
        
        ds = p * (dp - D_i) * scale
        
        dK_acc0 = tl.dot(ds.T, cur_q[0], acc=dK_acc0)
        dK_acc1 = tl.dot(ds.T, cur_q[1], acc=dK_acc1)
        
        dV_acc0 = tl.dot(p.T, cur_do[0], acc=dV_acc0)
        dV_acc1 = tl.dot(p.T, cur_do[1], acc=dV_acc1)
        
        cur_q, next_q = next_q, cur_q
        cur_o, next_o = next_o, cur_o
        cur_do, next_do = next_do, cur_do
        
    dK_desc.store([n_row_off, 0], dK_acc0.to(tl.bfloat16))
    dK_desc.store([n_row_off, 64], dK_acc1.to(tl.bfloat16))
    
    dV_desc.store([n_row_off, 0], dV_acc0.to(tl.bfloat16))
    dV_desc.store([n_row_off, 64], dV_acc1.to(tl.bfloat16))


NUM_WARPS = 4
NUM_STAGES = 3

def run(Q, K, V, O, dO, L, dQ, dK, dV):
    b, h, s_len, d = Q.shape
    device = Q.device
    torch.cuda.set_device(device)
    
    scale = 1.0 / math.sqrt(d)
    
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)
    triton.set_allocator(alloc_fn)
    
    s2 = Q.stride(2)
    s3 = Q.stride(3)
    
    grid = (triton.cdiv(s_len, BLOCK), b * h)
    
    bwd_dq_kernel[grid](
        Q, K, V, O, dO, L, dQ,
        s_len, d, b * h * s_len,
        s2, s3,
        BLOCK=BLOCK,
        scale=scale,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    
    bwd_dkv_kernel[grid](
        Q, K, V, O, dO, L, dK, dV,
        s_len, d, b * h * s_len,
        s2, s3,
        BLOCK=BLOCK,
        scale=scale,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )