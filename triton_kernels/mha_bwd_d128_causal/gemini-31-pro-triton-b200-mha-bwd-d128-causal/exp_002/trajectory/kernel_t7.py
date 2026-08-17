import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


def get_configs_dq():
    return [
        triton.Config({'WARP_SPECIALIZE': True, 'NUM_STAGES': 4}, num_warps=8, num_stages=4),
        triton.Config({'WARP_SPECIALIZE': True, 'NUM_STAGES': 3}, num_warps=8, num_stages=3),
        triton.Config({'WARP_SPECIALIZE': False, 'NUM_STAGES': 4}, num_warps=8, num_stages=4),
        triton.Config({'WARP_SPECIALIZE': False, 'NUM_STAGES': 3}, num_warps=8, num_stages=3),
        triton.Config({'WARP_SPECIALIZE': False, 'NUM_STAGES': 2}, num_warps=4, num_stages=2),
    ]

def get_configs_dk_dv():
    return [
        triton.Config({'WARP_SPECIALIZE': True, 'NUM_STAGES': 4}, num_warps=8, num_stages=4),
        triton.Config({'WARP_SPECIALIZE': True, 'NUM_STAGES': 3}, num_warps=8, num_stages=3),
        triton.Config({'WARP_SPECIALIZE': False, 'NUM_STAGES': 4}, num_warps=8, num_stages=4),
        triton.Config({'WARP_SPECIALIZE': False, 'NUM_STAGES': 3}, num_warps=8, num_stages=3),
        triton.Config({'WARP_SPECIALIZE': False, 'NUM_STAGES': 2}, num_warps=4, num_stages=2),
    ]

@triton.autotune(configs=get_configs_dq(), key=['S'])
@triton.jit
def bwd_dq_kernel_tma(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, dQ_desc,
    L_ptr, stride_lb, stride_lh, stride_ls,
    S, scale, H,
    S_PADDED: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr, NUM_STAGES: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    q = Q_desc.load([b, h, pid_m * BLOCK_M, 0])
    q = tl.reshape(q, (BLOCK_M, d))
    
    o = O_desc.load([b, h, pid_m * BLOCK_M, 0])
    o = tl.reshape(o, (BLOCK_M, d)).to(tl.float32)
    
    do = dO_desc.load([b, h, pid_m * BLOCK_M, 0])
    do = tl.reshape(do, (BLOCK_M, d))
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    
    l_ptrs = L_ptr + b * stride_lb + h * stride_lh + offs_m * stride_ls
    if S_PADDED:
        L = tl.load(l_ptrs)
    else:
        L = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    D = tl.sum(o * do.to(tl.float32), axis=1)
    acc_dq = tl.zeros((BLOCK_M, d), dtype=tl.float32)
    
    max_j = tl.minimum(pid_m * BLOCK_M + BLOCK_M, S)
    num_blocks = (max_j + BLOCK_N - 1) // BLOCK_N

    for n_idx in tl.range(0, num_blocks, num_stages=NUM_STAGES, warp_specialize=WARP_SPECIALIZE):
        k = K_desc.load([b, h, n_idx * BLOCK_N, 0])
        k = tl.reshape(k, (BLOCK_N, d))
        
        v = V_desc.load([b, h, n_idx * BLOCK_N, 0])
        v = tl.reshape(v, (BLOCK_N, d))
        
        s_ij = tl.dot(q, k.T, out_dtype=tl.float32) * scale
        p_ij = tl.exp(s_ij - L[:, None])
        
        current_n = n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        causal_mask = current_n[None, :] <= offs_m[:, None]
        
        if S_PADDED:
            p_ij = tl.where(causal_mask, p_ij, 0.0)
        else:
            valid_mask = causal_mask & (current_n[None, :] < S) & mask_m[:, None]
            p_ij = tl.where(valid_mask, p_ij, 0.0)
                
        dp_ij = tl.dot(do, v.T, out_dtype=tl.float32)
        
        ds_ij = p_ij * (dp_ij - D[:, None])
        ds_ij_scaled = (ds_ij * scale).to(tl.bfloat16)
        
        acc_dq = tl.dot(ds_ij_scaled, k, acc_dq)
        
    dQ_desc.store([b, h, pid_m * BLOCK_M, 0], tl.reshape(acc_dq.to(tl.bfloat16), (1, 1, BLOCK_M, d)))


@triton.autotune(configs=get_configs_dk_dv(), key=['S'])
@triton.jit
def bwd_dk_dv_kernel_tma(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, dK_desc, dV_desc,
    L_ptr, stride_lb, stride_lh, stride_ls,
    S, scale, H,
    S_PADDED: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr, NUM_STAGES: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    
    k = K_desc.load([b, h, pid_n * BLOCK_N, 0])
    k = tl.reshape(k, (BLOCK_N, d))
    
    v = V_desc.load([b, h, pid_n * BLOCK_N, 0])
    v = tl.reshape(v, (BLOCK_N, d))
    
    acc_dk = tl.zeros((BLOCK_N, d), dtype=tl.float32)
    acc_dv = tl.zeros((BLOCK_N, d), dtype=tl.float32)
    
    start_m_idx = (pid_n * BLOCK_N) // BLOCK_M
    num_m_blocks = (S + BLOCK_M - 1) // BLOCK_M
    
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S

    for m_idx in tl.range(start_m_idx, num_m_blocks, num_stages=NUM_STAGES, warp_specialize=WARP_SPECIALIZE):
        q = Q_desc.load([b, h, m_idx * BLOCK_M, 0])
        q = tl.reshape(q, (BLOCK_M, d))
        
        o = O_desc.load([b, h, m_idx * BLOCK_M, 0])
        o = tl.reshape(o, (BLOCK_M, d)).to(tl.float32)
        
        do = dO_desc.load([b, h, m_idx * BLOCK_M, 0])
        do = tl.reshape(do, (BLOCK_M, d))
        
        current_m = m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        
        l_ptrs = L_ptr + b * stride_lb + h * stride_lh + current_m * stride_ls
        if S_PADDED:
            L = tl.load(l_ptrs)
        else:
            mask_m = current_m < S
            L = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        D = tl.sum(o * do.to(tl.float32), axis=1)
        
        s_ji = tl.dot(k, q.T, out_dtype=tl.float32) * scale
        p_ji = tl.exp(s_ji - L[None, :])
        
        causal_mask = offs_n[:, None] <= current_m[None, :]
        
        if S_PADDED:
            p_ji = tl.where(causal_mask, p_ji, 0.0)
        else:
            valid_mask = causal_mask & mask_n[:, None] & mask_m[None, :]
            p_ji = tl.where(valid_mask, p_ji, 0.0)
            
        p_ji_bf16 = p_ji.to(tl.bfloat16)
        acc_dv = tl.dot(p_ji_bf16, do, acc_dv)
        
        dp_ji = tl.dot(v, do.T, out_dtype=tl.float32)
        
        ds_ji = p_ji * (dp_ji - D[None, :])
        ds_ji_scaled = (ds_ji * scale).to(tl.bfloat16)
        
        acc_dk = tl.dot(ds_ji_scaled, q, acc_dk)
        
    dK_desc.store([b, h, pid_n * BLOCK_N, 0], tl.reshape(acc_dk.to(tl.bfloat16), (1, 1, BLOCK_N, d)))
    dV_desc.store([b, h, pid_n * BLOCK_N, 0], tl.reshape(acc_dv.to(tl.bfloat16), (1, 1, BLOCK_N, d)))


@triton.autotune(configs=get_configs_dq(), key=['S'])
@triton.jit
def bwd_dq_kernel_ptr(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    S, scale, H,
    S_PADDED: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr, NUM_STAGES: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, d)
    mask_m = offs_m < S
    safe_m = tl.where(mask_m, offs_m, 0)

    q_base = Q_ptr + b * stride_qb + h * stride_qh
    o_base = O_ptr + b * stride_ob + h * stride_oh
    do_base = dO_ptr + b * stride_dob + h * stride_doh
    l_base = L_ptr + b * stride_lb + h * stride_lh
    
    q_ptrs = q_base + safe_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    o_ptrs = o_base + safe_m[:, None] * stride_os + offs_d[None, :] * stride_od
    do_ptrs = do_base + safe_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    l_ptrs = l_base + safe_m * stride_ls

    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    
    if S_PADDED:
        L = tl.load(l_ptrs)
    else:
        L = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    D = tl.sum(o * do.to(tl.float32), axis=1)
    acc_dq = tl.zeros((BLOCK_M, d), dtype=tl.float32)

    k_base = K_ptr + b * stride_kb + h * stride_kh
    v_base = V_ptr + b * stride_vb + h * stride_vh
    offs_n = tl.arange(0, BLOCK_N)
    
    max_j = tl.minimum(pid_m * BLOCK_M + BLOCK_M, S)
    num_blocks = (max_j + BLOCK_N - 1) // BLOCK_N

    for n_idx in tl.range(0, num_blocks, num_stages=NUM_STAGES, warp_specialize=WARP_SPECIALIZE):
        current_n = n_idx * BLOCK_N + offs_n
        mask_n = current_n < S
        safe_n = tl.where(mask_n, current_n, 0)
        
        k_ptrs = k_base + safe_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = v_base + safe_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        s_ij = tl.dot(q, k.T, out_dtype=tl.float32) * scale
        p_ij = tl.exp(s_ij - L[:, None])
        
        causal_mask = current_n[None, :] <= offs_m[:, None]
        if S_PADDED:
            p_ij = tl.where(causal_mask, p_ij, 0.0)
        else:
            valid_mask = causal_mask & mask_n[:, None] & mask_m[:, None]
            p_ij = tl.where(valid_mask, p_ij, 0.0)
        
        dp_ij = tl.dot(do, v.T, out_dtype=tl.float32)
        
        ds_ij = p_ij * (dp_ij - D[:, None])
        ds_ij_scaled = (ds_ij * scale).to(tl.bfloat16)
        
        acc_dq = tl.dot(ds_ij_scaled, k, acc_dq)

    dq_base = dQ_ptr + b * stride_dqb + h * stride_dqh
    dq_ptrs = dq_base + safe_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, acc_dq.to(tl.bfloat16), mask=mask_m[:, None])


@triton.autotune(configs=get_configs_dk_dv(), key=['S'])
@triton.jit
def bwd_dk_dv_kernel_ptr(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    S, scale, H,
    S_PADDED: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr, NUM_STAGES: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, d)
    mask_n = offs_n < S
    safe_n = tl.where(mask_n, offs_n, 0)
    
    k_base = K_ptr + b * stride_kb + h * stride_kh
    v_base = V_ptr + b * stride_vb + h * stride_vh
    
    k_ptrs = k_base + safe_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = v_base + safe_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    
    acc_dk = tl.zeros((BLOCK_N, d), dtype=tl.float32)
    acc_dv = tl.zeros((BLOCK_N, d), dtype=tl.float32)
    
    q_base = Q_ptr + b * stride_qb + h * stride_qh
    o_base = O_ptr + b * stride_ob + h * stride_oh
    do_base = dO_ptr + b * stride_dob + h * stride_doh
    l_base = L_ptr + b * stride_lb + h * stride_lh
    
    start_m_idx = (pid_n * BLOCK_N) // BLOCK_M
    num_m_blocks = (S + BLOCK_M - 1) // BLOCK_M
    
    for m_idx in tl.range(start_m_idx, num_m_blocks, num_stages=NUM_STAGES, warp_specialize=WARP_SPECIALIZE):
        current_m = m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = current_m < S
        safe_m = tl.where(mask_m, current_m, 0)
        
        q_ptrs = q_base + safe_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
        o_ptrs = o_base + safe_m[:, None] * stride_os + offs_d[None, :] * stride_od
        do_ptrs = do_base + safe_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
        l_ptrs = l_base + safe_m * stride_ls
        
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        
        if S_PADDED:
            L = tl.load(l_ptrs)
        else:
            L = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        D = tl.sum(o * do.to(tl.float32), axis=1) 
        
        s_ji = tl.dot(k, q.T, out_dtype=tl.float32) * scale
        p_ji = tl.exp(s_ji - L[None, :])
        
        causal_mask = offs_n[:, None] <= current_m[None, :]
        if S_PADDED:
            p_ji = tl.where(causal_mask, p_ji, 0.0)
        else:
            valid_mask = causal_mask & mask_n[:, None] & mask_m[None, :]
            p_ji = tl.where(valid_mask, p_ji, 0.0)
        
        p_ji_bf16 = p_ji.to(tl.bfloat16)
        acc_dv = tl.dot(p_ji_bf16, do, acc_dv)
        
        dp_ji = tl.dot(v, do.T, out_dtype=tl.float32)
        
        ds_ji = p_ji * (dp_ji - D[None, :])
        ds_ji_scaled = (ds_ji * scale).to(tl.bfloat16)
        
        acc_dk = tl.dot(ds_ji_scaled, q, acc_dk)

    dk_ptrs = dk_base + safe_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dv_ptrs = dv_base + safe_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
    
    tl.store(dk_ptrs, acc_dk.to(tl.bfloat16), mask=mask_n[:, None])
    tl.store(dv_ptrs, acc_dv.to(tl.bfloat16), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Destination-passing Triton entry point supporting Blackwell SMA autotuning gracefully.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    scale = 1.0 / (d ** 0.5)
    
    if L.dim() == 4:
        L = L.squeeze(-1)

    s_padded = (S % 128 == 0)
    use_tma = True
    
    try:
        Q_desc_dq = TensorDescriptor.from_tensor(Q, [1, 1, 128, 128])
        K_desc_dq = TensorDescriptor.from_tensor(K, [1, 1, 64, 128])
        V_desc_dq = TensorDescriptor.from_tensor(V, [1, 1, 64, 128])
        O_desc_dq = TensorDescriptor.from_tensor(O, [1, 1, 128, 128])
        dO_desc_dq = TensorDescriptor.from_tensor(dO, [1, 1, 128, 128])
        dQ_desc_dq = TensorDescriptor.from_tensor(dQ, [1, 1, 128, 128])

        Q_desc_dk = TensorDescriptor.from_tensor(Q, [1, 1, 64, 128])
        K_desc_dk = TensorDescriptor.from_tensor(K, [1, 1, 128, 128])
        V_desc_dk = TensorDescriptor.from_tensor(V, [1, 1, 128, 128])
        O_desc_dk = TensorDescriptor.from_tensor(O, [1, 1, 64, 128])
        dO_desc_dk = TensorDescriptor.from_tensor(dO, [1, 1, 64, 128])
        dK_desc_dk = TensorDescriptor.from_tensor(dK, [1, 1, 128, 128])
        dV_desc_dk = TensorDescriptor.from_tensor(dV, [1, 1, 128, 128])
    except Exception:
        use_tma = False
        
    if use_tma:
        grid_dq = (triton.cdiv(S, 128), B * H)
        bwd_dq_kernel_tma[grid_dq](
            Q_desc_dq, K_desc_dq, V_desc_dq, O_desc_dq, dO_desc_dq, dQ_desc_dq,
            L, L.stride(0), L.stride(1), L.stride(2),
            S, scale, H,
            S_PADDED=s_padded,
            BLOCK_M=128, BLOCK_N=64, d=128
        )

        grid_dk_dv = (triton.cdiv(S, 128), B * H)
        bwd_dk_dv_kernel_tma[grid_dk_dv](
            Q_desc_dk, K_desc_dk, V_desc_dk, O_desc_dk, dO_desc_dk, dK_desc_dk, dV_desc_dk,
            L, L.stride(0), L.stride(1), L.stride(2),
            S, scale, H,
            S_PADDED=s_padded,
            BLOCK_M=64, BLOCK_N=128, d=128
        )
    else:
        grid_dq = (triton.cdiv(S, 128), B * H)
        bwd_dq_kernel_ptr[grid_dq](
            Q, K, V, O, dO, L, dQ,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
            S, scale, H,
            S_PADDED=s_padded,
            BLOCK_M=128, BLOCK_N=64, d=128
        )

        grid_dk_dv = (triton.cdiv(S, 128), B * H)
        bwd_dk_dv_kernel_ptr[grid_dk_dv](
            Q, K, V, O, dO, L, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
            dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
            S, scale, H,
            S_PADDED=s_padded,
            BLOCK_M=64, BLOCK_N=128, d=128
        )