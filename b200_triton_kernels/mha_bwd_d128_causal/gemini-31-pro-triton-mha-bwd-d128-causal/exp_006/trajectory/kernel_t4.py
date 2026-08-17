import math
import torch
import triton
import triton.language as tl

def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.jit
def bwd_preprocess_kernel(
    O, DO, D_ptr,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    H, S,
    BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr
):
    """
    Computes D = sum(O * dO, axis=1) and stores it as a contiguous flat float32 array.
    We safely alias the first bytes of the preallocated dQ tensor as workspace.
    """
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    offs_d = tl.arange(0, D_HEAD)
    
    o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    do_ptrs = DO + pid_b * stride_dob + pid_h * stride_doh + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    
    o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    do_ = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    
    d = tl.sum(o * do_, axis=1)
    
    # Store D contiguously in the beginning of dQ memory
    d_f32_ptr = D_ptr.to(tl.pointer_type(tl.float32))
    flat_index = pid_b * (H * S) + pid_h * S + offs_m
    d_ptrs = d_f32_ptr + flat_index
    
    tl.store(d_ptrs, d, mask=mask_m)


def get_dkdv_configs():
    return [
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
    ]

@triton.autotune(
    configs=get_dkdv_configs(),
    key=['S'],
)
@triton.jit
def bwd_kernel_dk_dv(
    Q, K, V, DO, dK, dV, L, D_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    stride_lb, stride_lh, stride_ls,
    H, scale, S,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    EVEN_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    k_ptr = K + pid_b * stride_kb + pid_h * stride_kh
    v_ptr = V + pid_b * stride_vb + pid_h * stride_vh
    dk_ptr = dK + pid_b * stride_dkb + pid_h * stride_dkh
    dv_ptr = dV + pid_b * stride_dvb + pid_h * stride_dvh
    
    q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
    do_ptr = DO + pid_b * stride_dob + pid_h * stride_doh
    
    k_desc = tl.make_tensor_descriptor(k_ptr, shape=[S, D_HEAD], strides=[stride_ks, 1], block_shape=[BLOCK_N, D_HEAD], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(v_ptr, shape=[S, D_HEAD], strides=[stride_vs, 1], block_shape=[BLOCK_N, D_HEAD], padding_option="zero")
    dk_desc = tl.make_tensor_descriptor(dk_ptr, shape=[S, D_HEAD], strides=[stride_dks, 1], block_shape=[BLOCK_N, D_HEAD])
    dv_desc = tl.make_tensor_descriptor(dv_ptr, shape=[S, D_HEAD], strides=[stride_dvs, 1], block_shape=[BLOCK_N, D_HEAD])
    
    q_desc = tl.make_tensor_descriptor(q_ptr, shape=[S, D_HEAD], strides=[stride_qs, 1], block_shape=[BLOCK_M, D_HEAD], padding_option="zero")
    do_desc = tl.make_tensor_descriptor(do_ptr, shape=[S, D_HEAD], strides=[stride_dos, 1], block_shape=[BLOCK_M, D_HEAD], padding_option="zero")
    
    k = k_desc.load([pid_n * BLOCK_N, 0])
    v = v_desc.load([pid_n * BLOCK_N, 0])
    
    dk = tl.zeros([BLOCK_N, D_HEAD], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D_HEAD], dtype=tl.float32)
    
    start_m_start = (pid_n * BLOCK_N // BLOCK_M) * BLOCK_M
    limit_m = (S + BLOCK_M - 1) // BLOCK_M * BLOCK_M
    start_m_diag_end = ((pid_n + 1) * BLOCK_N + BLOCK_M - 1) // BLOCK_M * BLOCK_M
    start_m_diag_end = tl.minimum(start_m_diag_end, limit_m)
    
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    d_f32_ptr = D_ptr.to(tl.pointer_type(tl.float32))
    base_d_ptr = d_f32_ptr + pid_b * (H * S) + pid_h * S
    
    # Diagonal block(s) with causal masking
    for start_m_val in range(start_m_start, start_m_diag_end, BLOCK_M):
        q = q_desc.load([start_m_val, 0])
        do_ = do_desc.load([start_m_val, 0])
        
        offs_m = start_m_val + tl.arange(0, BLOCK_M)
        l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ls
        d_ptrs = base_d_ptr + offs_m
        
        if not EVEN_S:
            mask_m = offs_m < S
            l = tl.load(l_ptrs, mask=mask_m, other=0.0)
            d = tl.load(d_ptrs, mask=mask_m, other=0.0)
        else:
            l = tl.load(l_ptrs)
            d = tl.load(d_ptrs)
            
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        
        if not EVEN_S:
            mask_n = offs_n < S
            valid_mask = causal_mask & mask_m[:, None] & mask_n[None, :]
            s = tl.where(valid_mask, s, float("-inf"))
            p = tl.where(valid_mask, tl.exp(s - l[:, None]), 0.0)
        else:
            s = tl.where(causal_mask, s, float("-inf"))
            p = tl.where(causal_mask, tl.exp(s - l[:, None]), 0.0)
            
        dp = tl.dot(do_, tl.trans(v), out_dtype=tl.float32)
        ds = p * (dp - d[:, None]) * scale
        
        dk += tl.dot(tl.trans(ds.to(tl.bfloat16)), q, out_dtype=tl.float32)
        dv += tl.dot(tl.trans(p.to(tl.bfloat16)), do_, out_dtype=tl.float32)

    # Fully valid blocks (no causal mask)
    for start_m_val in range(start_m_diag_end, limit_m, BLOCK_M):
        q = q_desc.load([start_m_val, 0])
        do_ = do_desc.load([start_m_val, 0])
        
        offs_m = start_m_val + tl.arange(0, BLOCK_M)
        l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ls
        d_ptrs = base_d_ptr + offs_m
        
        if not EVEN_S:
            mask_m = offs_m < S
            l = tl.load(l_ptrs, mask=mask_m, other=0.0)
            d = tl.load(d_ptrs, mask=mask_m, other=0.0)
        else:
            l = tl.load(l_ptrs)
            d = tl.load(d_ptrs)
            
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        
        if not EVEN_S:
            mask_n = offs_n < S
            valid_mask = mask_m[:, None] & mask_n[None, :]
            s = tl.where(valid_mask, s, float("-inf"))
            p = tl.where(valid_mask, tl.exp(s - l[:, None]), 0.0)
        else:
            p = tl.exp(s - l[:, None])
            
        dp = tl.dot(do_, tl.trans(v), out_dtype=tl.float32)
        ds = p * (dp - d[:, None]) * scale
        
        dk += tl.dot(tl.trans(ds.to(tl.bfloat16)), q, out_dtype=tl.float32)
        dv += tl.dot(tl.trans(p.to(tl.bfloat16)), do_, out_dtype=tl.float32)
        
    dk_desc.store([pid_n * BLOCK_N, 0], dk.to(tl.bfloat16))
    dv_desc.store([pid_n * BLOCK_N, 0], dv.to(tl.bfloat16))


def get_dq_configs():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
    ]

@triton.autotune(
    configs=get_dq_configs(),
    key=['S'],
)
@triton.jit
def bwd_kernel_dq(
    Q, K, V, O, DO, dQ, L,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_lb, stride_lh, stride_ls,
    scale, S,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    EVEN_S: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
    o_ptr = O + pid_b * stride_ob + pid_h * stride_oh
    do_ptr = DO + pid_b * stride_dob + pid_h * stride_doh
    dq_ptr = dQ + pid_b * stride_dqb + pid_h * stride_dqh
    
    k_ptr = K + pid_b * stride_kb + pid_h * stride_kh
    v_ptr = V + pid_b * stride_vb + pid_h * stride_vh
    
    q_desc = tl.make_tensor_descriptor(q_ptr, shape=[S, D_HEAD], strides=[stride_qs, 1], block_shape=[BLOCK_M, D_HEAD], padding_option="zero")
    o_desc = tl.make_tensor_descriptor(o_ptr, shape=[S, D_HEAD], strides=[stride_os, 1], block_shape=[BLOCK_M, D_HEAD], padding_option="zero")
    do_desc = tl.make_tensor_descriptor(do_ptr, shape=[S, D_HEAD], strides=[stride_dos, 1], block_shape=[BLOCK_M, D_HEAD], padding_option="zero")
    dq_desc = tl.make_tensor_descriptor(dq_ptr, shape=[S, D_HEAD], strides=[stride_dqs, 1], block_shape=[BLOCK_M, D_HEAD])
    
    k_desc = tl.make_tensor_descriptor(k_ptr, shape=[S, D_HEAD], strides=[stride_ks, 1], block_shape=[BLOCK_N, D_HEAD], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(v_ptr, shape=[S, D_HEAD], strides=[stride_vs, 1], block_shape=[BLOCK_N, D_HEAD], padding_option="zero")
    
    q = q_desc.load([pid_m * BLOCK_M, 0])
    o = o_desc.load([pid_m * BLOCK_M, 0])
    do_ = do_desc.load([pid_m * BLOCK_M, 0])
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ls
    
    if not EVEN_S:
        mask_m = offs_m < S
        l = tl.load(l_ptrs, mask=mask_m, other=0.0)
    else:
        l = tl.load(l_ptrs)
        
    d = tl.sum(o.to(tl.float32) * do_.to(tl.float32), axis=1)
    dq = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)
    
    offs_n = tl.arange(0, BLOCK_N)
    start_n_end = (pid_m * BLOCK_M // BLOCK_N) * BLOCK_N
    
    # Fully valid blocks (no causal mask needed)
    for start_n_val in range(0, start_n_end, BLOCK_N):
        curr_n = start_n_val + offs_n
        k = k_desc.load([start_n_val, 0])
        v = v_desc.load([start_n_val, 0])
        
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        
        if not EVEN_S:
            valid_mask = (offs_m[:, None] < S) & (curr_n[None, :] < S)
            s = tl.where(valid_mask, s, float("-inf"))
            p = tl.where(valid_mask, tl.exp(s - l[:, None]), 0.0)
        else:
            p = tl.exp(s - l[:, None])
            
        dp = tl.dot(do_, tl.trans(v), out_dtype=tl.float32)
        ds = p * (dp - d[:, None]) * scale
        
        dq += tl.dot(ds.to(tl.bfloat16), k, out_dtype=tl.float32)
        
    limit_n = (S + BLOCK_N - 1) // BLOCK_N * BLOCK_N
    diag_end = ((pid_m + 1) * BLOCK_M + BLOCK_N - 1) // BLOCK_N * BLOCK_N
    diag_end = tl.minimum(diag_end, limit_n)
    
    # Diagonal block(s) with causal mask
    for start_n_val in range(start_n_end, diag_end, BLOCK_N):
        curr_n = start_n_val + offs_n
        k = k_desc.load([start_n_val, 0])
        v = v_desc.load([start_n_val, 0])
        
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        causal_mask = offs_m[:, None] >= curr_n[None, :]
        
        if not EVEN_S:
            valid_mask = causal_mask & (offs_m[:, None] < S) & (curr_n[None, :] < S)
            s = tl.where(valid_mask, s, float("-inf"))
            p = tl.where(valid_mask, tl.exp(s - l[:, None]), 0.0)
        else:
            s = tl.where(causal_mask, s, float("-inf"))
            p = tl.where(causal_mask, tl.exp(s - l[:, None]), 0.0)
            
        dp = tl.dot(do_, tl.trans(v), out_dtype=tl.float32)
        ds = p * (dp - d[:, None]) * scale
        
        dq += tl.dot(ds.to(tl.bfloat16), k, out_dtype=tl.float32)
        
    dq_desc.store([pid_m * BLOCK_M, 0], dq.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    with torch.cuda.device(Q.device):
        B, H, S, D_HEAD = Q.shape
        scale = 1.0 / math.sqrt(D_HEAD)
        EVEN_S = (S % 128 == 0)

        # 1. Precompute D = rowsum(O * dO) securely into the target dQ footprint.
        grid_pre = (triton.cdiv(S, 128), H, B)
        bwd_preprocess_kernel[grid_pre](
            O, dO, dQ,
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            H, S,
            BLOCK_M=128, D_HEAD=D_HEAD,
            num_warps=4, num_stages=2
        )

        # 2. Compute dK and dV, reading D from the aliased contiguous area in dQ memory.
        grid_dk_dv = lambda META: (triton.cdiv(S, META['BLOCK_N']), H, B)
        bwd_kernel_dk_dv[grid_dk_dv](
            Q, K, V, dO, dK, dV, L, dQ,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
            dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            H, scale, S,
            D_HEAD=D_HEAD,
            EVEN_S=EVEN_S,
        )

        # 3. Compute dQ and cleanly overwrite the dQ buffer memory.
        grid_dq = lambda META: (triton.cdiv(S, META['BLOCK_M']), H, B)
        bwd_kernel_dq[grid_dq](
            Q, K, V, O, dO, dQ, L,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            scale, S,
            D_HEAD=D_HEAD,
            EVEN_S=EVEN_S,
        )