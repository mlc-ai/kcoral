import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
    ],
    key=['seq_len'],
)
@triton.jit
def _bwd_kernel_dq(
    Q, K, V, sm_scale, Out, DO, DQ, L,
    stride_q_b, stride_q_h, stride_q_s, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_o_b, stride_o_h, stride_o_s, stride_o_d,
    stride_do_b, stride_do_h, stride_do_s, stride_do_d,
    stride_dq_b, stride_dq_h, stride_dq_s, stride_dq_d,
    stride_l_b, stride_l_h, stride_l_s,
    B, H, seq_len,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    start_m = tl.program_id(0)
    bh_id = tl.program_id(1)
    
    b_id = (bh_id // H).to(tl.int64)
    h_id = (bh_id % H).to(tl.int64)
    
    desc_q = tl.make_tensor_descriptor(Q, shape=[B, H, seq_len, BLOCK_DMODEL], strides=[stride_q_b, stride_q_h, stride_q_s, stride_q_d], block_shape=[1, 1, BLOCK_M, BLOCK_DMODEL], padding_option="zero")
    desc_k = tl.make_tensor_descriptor(K, shape=[B, H, seq_len, BLOCK_DMODEL], strides=[stride_k_b, stride_k_h, stride_k_s, stride_k_d], block_shape=[1, 1, BLOCK_N, BLOCK_DMODEL], padding_option="zero")
    desc_v = tl.make_tensor_descriptor(V, shape=[B, H, seq_len, BLOCK_DMODEL], strides=[stride_v_b, stride_v_h, stride_v_s, stride_v_d], block_shape=[1, 1, BLOCK_N, BLOCK_DMODEL], padding_option="zero")
    desc_o = tl.make_tensor_descriptor(Out, shape=[B, H, seq_len, BLOCK_DMODEL], strides=[stride_o_b, stride_o_h, stride_o_s, stride_o_d], block_shape=[1, 1, BLOCK_M, BLOCK_DMODEL], padding_option="zero")
    desc_do = tl.make_tensor_descriptor(DO, shape=[B, H, seq_len, BLOCK_DMODEL], strides=[stride_do_b, stride_do_h, stride_do_s, stride_do_d], block_shape=[1, 1, BLOCK_M, BLOCK_DMODEL], padding_option="zero")
    desc_dq = tl.make_tensor_descriptor(DQ, shape=[B, H, seq_len, BLOCK_DMODEL], strides=[stride_dq_b, stride_dq_h, stride_dq_s, stride_dq_d], block_shape=[1, 1, BLOCK_M, BLOCK_DMODEL])
    
    q_blk = desc_q.load([b_id, h_id, start_m * BLOCK_M, 0])
    do_blk = desc_do.load([b_id, h_id, start_m * BLOCK_M, 0])
    o_blk = desc_o.load([b_id, h_id, start_m * BLOCK_M, 0])
    
    q = tl.reshape(q_blk, [BLOCK_M, BLOCK_DMODEL])
    do = tl.reshape(do_blk, [BLOCK_M, BLOCK_DMODEL])
    o = tl.reshape(o_blk, [BLOCK_M, BLOCK_DMODEL])
    
    offset_l = b_id * stride_l_b + h_id * stride_l_h
    offs_m = (start_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    mask_m = offs_m < seq_len
    l_ptrs = L + offset_l + offs_m * stride_l_s
    l_i = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    d_i = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    
    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    num_n_blocks = tl.cdiv(seq_len, BLOCK_N)
    
    for start_n in range(0, num_n_blocks):
        k_blk = desc_k.load([b_id, h_id, start_n * BLOCK_N, 0])
        v_blk = desc_v.load([b_id, h_id, start_n * BLOCK_N, 0])
        
        k = tl.reshape(k_blk, [BLOCK_N, BLOCK_DMODEL])
        v = tl.reshape(v_blk, [BLOCK_N, BLOCK_DMODEL])
        
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
        qk = qk * sm_scale
        p = tl.exp(qk - l_i[:, None])
        
        offs_n = (start_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        mask = (offs_m[:, None] < seq_len) & (offs_n[None, :] < seq_len)
        p = tl.where(mask, p, 0.0)
        
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        ds = p * (dp - d_i[:, None])
        ds = ds * sm_scale
        
        ds_bf16 = ds.to(tl.bfloat16)
        dq = tl.dot(ds_bf16, k, acc=dq)
        
    dq_out = tl.reshape(dq.to(tl.bfloat16), [1, 1, BLOCK_M, BLOCK_DMODEL])
    desc_dq.store([b_id, h_id, start_m * BLOCK_M, 0], dq_out)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
    ],
    key=['seq_len'],
)
@triton.jit
def _bwd_kernel_dk_dv(
    Q, K, V, sm_scale, Out, DO, DK, DV, L,
    stride_q_b, stride_q_h, stride_q_s, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_o_b, stride_o_h, stride_o_s, stride_o_d,
    stride_do_b, stride_do_h, stride_do_s, stride_do_d,
    stride_dk_b, stride_dk_h, stride_dk_s, stride_dk_d,
    stride_dv_b, stride_dv_h, stride_dv_s, stride_dv_d,
    stride_l_b, stride_l_h, stride_l_s,
    B, H, seq_len,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    start_n = tl.program_id(0)
    bh_id = tl.program_id(1)
    
    b_id = (bh_id // H).to(tl.int64)
    h_id = (bh_id % H).to(tl.int64)
    
    desc_q = tl.make_tensor_descriptor(Q, shape=[B, H, seq_len, BLOCK_DMODEL], strides=[stride_q_b, stride_q_h, stride_q_s, stride_q_d], block_shape=[1, 1, BLOCK_M, BLOCK_DMODEL], padding_option="zero")
    desc_k = tl.make_tensor_descriptor(K, shape=[B, H, seq_len, BLOCK_DMODEL], strides=[stride_k_b, stride_k_h, stride_k_s, stride_k_d], block_shape=[1, 1, BLOCK_N, BLOCK_DMODEL], padding_option="zero")
    desc_v = tl.make_tensor_descriptor(V, shape=[B, H, seq_len, BLOCK_DMODEL], strides=[stride_v_b, stride_v_h, stride_v_s, stride_v_d], block_shape=[1, 1, BLOCK_N, BLOCK_DMODEL], padding_option="zero")
    desc_o = tl.make_tensor_descriptor(Out, shape=[B, H, seq_len, BLOCK_DMODEL], strides=[stride_o_b, stride_o_h, stride_o_s, stride_o_d], block_shape=[1, 1, BLOCK_M, BLOCK_DMODEL], padding_option="zero")
    desc_do = tl.make_tensor_descriptor(DO, shape=[B, H, seq_len, BLOCK_DMODEL], strides=[stride_do_b, stride_do_h, stride_do_s, stride_do_d], block_shape=[1, 1, BLOCK_M, BLOCK_DMODEL], padding_option="zero")
    desc_dk = tl.make_tensor_descriptor(DK, shape=[B, H, seq_len, BLOCK_DMODEL], strides=[stride_dk_b, stride_dk_h, stride_dk_s, stride_dk_d], block_shape=[1, 1, BLOCK_N, BLOCK_DMODEL])
    desc_dv = tl.make_tensor_descriptor(DV, shape=[B, H, seq_len, BLOCK_DMODEL], strides=[stride_dv_b, stride_dv_h, stride_dv_s, stride_dv_d], block_shape=[1, 1, BLOCK_N, BLOCK_DMODEL])
    
    k_blk = desc_k.load([b_id, h_id, start_n * BLOCK_N, 0])
    v_blk = desc_v.load([b_id, h_id, start_n * BLOCK_N, 0])
    
    k = tl.reshape(k_blk, [BLOCK_N, BLOCK_DMODEL])
    v = tl.reshape(v_blk, [BLOCK_N, BLOCK_DMODEL])
    
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    
    num_m_blocks = tl.cdiv(seq_len, BLOCK_M)
    
    offset_l = b_id * stride_l_b + h_id * stride_l_h
    offs_n = (start_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    
    for start_m in range(0, num_m_blocks):
        q_blk = desc_q.load([b_id, h_id, start_m * BLOCK_M, 0])
        do_blk = desc_do.load([b_id, h_id, start_m * BLOCK_M, 0])
        o_blk = desc_o.load([b_id, h_id, start_m * BLOCK_M, 0])
        
        q = tl.reshape(q_blk, [BLOCK_M, BLOCK_DMODEL])
        do = tl.reshape(do_blk, [BLOCK_M, BLOCK_DMODEL])
        o = tl.reshape(o_blk, [BLOCK_M, BLOCK_DMODEL])
        
        offs_m = (start_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        mask_m = offs_m < seq_len
        l_ptrs = L + offset_l + offs_m * stride_l_s
        l_i = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        d_i = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        
        qk_t = tl.dot(k, tl.trans(q), out_dtype=tl.float32)
        qk_t = qk_t * sm_scale
        
        p_t = tl.exp(qk_t - l_i[None, :])
        mask = (offs_n[:, None] < seq_len) & (offs_m[None, :] < seq_len)
        p_t = tl.where(mask, p_t, 0.0)
        
        dp_t = tl.dot(v, tl.trans(do), out_dtype=tl.float32)
        ds_t = p_t * (dp_t - d_i[None, :])
        ds_t = ds_t * sm_scale
        
        p_bf16_t = p_t.to(tl.bfloat16)
        ds_bf16_t = ds_t.to(tl.bfloat16)
        
        dv = tl.dot(p_bf16_t, do, acc=dv)
        dk = tl.dot(ds_bf16_t, q, acc=dk)
        
    dv_out = tl.reshape(dv.to(tl.bfloat16), [1, 1, BLOCK_N, BLOCK_DMODEL])
    dk_out = tl.reshape(dk.to(tl.bfloat16), [1, 1, BLOCK_N, BLOCK_DMODEL])
    desc_dv.store([b_id, h_id, start_n * BLOCK_N, 0], dv_out)
    desc_dk.store([b_id, h_id, start_n * BLOCK_N, 0], dk_out)


_allocator_set = False

def run(Q, K, V, O, dO, L, dQ, dK, dV):
    global _allocator_set
    if not _allocator_set:
        def alloc_fn(size: int, alignment: int, stream):
            return torch.empty(size, device="cuda", dtype=torch.int8)
        triton.set_allocator(alloc_fn)
        _allocator_set = True

    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    sm_scale = 1.0 / (d ** 0.5)
    
    if S == 0:
        return
    
    grid_dkdv = lambda META: (
        triton.cdiv(S, META['BLOCK_N']),
        B * H
    )
    _bwd_kernel_dk_dv[grid_dkdv](
        Q, K, V, sm_scale, O, dO, dK, dV, L,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        B, H, S,
        BLOCK_DMODEL=d
    )
    
    grid_dq = lambda META: (
        triton.cdiv(S, META['BLOCK_M']),
        B * H
    )
    _bwd_kernel_dq[grid_dq](
        Q, K, V, sm_scale, O, dO, dQ, L,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        B, H, S,
        BLOCK_DMODEL=d
    )