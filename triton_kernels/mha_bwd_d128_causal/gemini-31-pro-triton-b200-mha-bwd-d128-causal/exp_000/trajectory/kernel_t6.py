import torch
import triton
import triton.language as tl

# Configure standard TMA descriptor allocator for Blackwell
def _alloc_tma_desc(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(_alloc_tma_desc)


@triton.jit
def _bwd_kernel_dq(
    Q, K, V, O, sm_scale,
    DO, DQ, L,
    S, H,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_lb, stride_lh, stride_ls,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D_HEAD: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = (pid_bh // H).to(tl.int64)
    h = (pid_bh % H).to(tl.int64)
    
    q_base = Q + b * stride_qb + h * stride_qh
    do_base = DO + b * stride_dob + h * stride_doh
    o_base = O + b * stride_ob + h * stride_oh
    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh
    dq_base = DQ + b * stride_dqb + h * stride_dqh
    
    # Instantiate TMA descriptors handling exact shapes and async boundary padding natively
    desc_q = tl.make_tensor_descriptor(q_base, shape=[S, D_HEAD], strides=[stride_qs, stride_qd], block_shape=[BLOCK_M, D_HEAD], padding_option="zero")
    desc_do = tl.make_tensor_descriptor(do_base, shape=[S, D_HEAD], strides=[stride_dos, stride_dod], block_shape=[BLOCK_M, D_HEAD], padding_option="zero")
    desc_o = tl.make_tensor_descriptor(o_base, shape=[S, D_HEAD], strides=[stride_os, stride_od], block_shape=[BLOCK_M, D_HEAD], padding_option="zero")
    desc_k = tl.make_tensor_descriptor(k_base, shape=[S, D_HEAD], strides=[stride_ks, stride_kd], block_shape=[BLOCK_N, D_HEAD], padding_option="zero")
    desc_v = tl.make_tensor_descriptor(v_base, shape=[S, D_HEAD], strides=[stride_vs, stride_vd], block_shape=[BLOCK_N, D_HEAD], padding_option="zero")
    desc_dq = tl.make_tensor_descriptor(dq_base, shape=[S, D_HEAD], strides=[stride_dqs, stride_dqd], block_shape=[BLOCK_M, D_HEAD], padding_option="zero")
    
    offs_m_scalar = pid_m * BLOCK_M
    
    q = desc_q.load([offs_m_scalar, 0])
    do = desc_do.load([offs_m_scalar, 0])
    o = desc_o.load([offs_m_scalar, 0])
    
    idx_m = offs_m_scalar + tl.arange(0, BLOCK_M)
    l_ptrs = L + b * stride_lb + h * stride_lh + idx_m * stride_ls
    # Pad out of bounds L to infinity safely clamping probabilities to exactly 0.0 later
    lse = tl.load(l_ptrs, mask=idx_m < S, other=float("inf"))
    
    # Pre-compute Delta scale metrics inline via HBM reads caching avoiding workspace allocations
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    dq = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)
    
    n_max = tl.minimum(S, offs_m_scalar + BLOCK_M)
    n_steps = tl.cdiv(n_max, BLOCK_N)
    
    for n_idx in tl.range(0, n_steps, num_stages=3):
        offs_n_scalar = n_idx * BLOCK_N
        k = desc_k.load([offs_n_scalar, 0])
        v = desc_v.load([offs_n_scalar, 0])
        
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        p = tl.exp(qk - lse[:, None])
        
        idx_n = offs_n_scalar + tl.arange(0, BLOCK_N)
        causal_mask = (idx_m[:, None] >= idx_n[None, :]) & (idx_n[None, :] < S) & (idx_m[:, None] < S)
        p = tl.where(causal_mask, p, 0.0)
        
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        
        dq += tl.dot((ds * sm_scale).to(q.dtype), k)
        
    desc_dq.store([offs_m_scalar, 0], dq.to(q.dtype))


@triton.jit
def _bwd_kernel_dk_dv(
    Q, K, V, O, sm_scale,
    DO, DK, DV, L,
    S, H,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    stride_lb, stride_lh, stride_ls,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D_HEAD: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = (pid_bh // H).to(tl.int64)
    h = (pid_bh % H).to(tl.int64)
    
    q_base = Q + b * stride_qb + h * stride_qh
    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh
    o_base = O + b * stride_ob + h * stride_oh
    do_base = DO + b * stride_dob + h * stride_doh
    dk_base = DK + b * stride_dkb + h * stride_dkh
    dv_base = DV + b * stride_dvb + h * stride_dvh
    
    desc_q = tl.make_tensor_descriptor(q_base, shape=[S, D_HEAD], strides=[stride_qs, stride_qd], block_shape=[BLOCK_M, D_HEAD], padding_option="zero")
    desc_do = tl.make_tensor_descriptor(do_base, shape=[S, D_HEAD], strides=[stride_dos, stride_dod], block_shape=[BLOCK_M, D_HEAD], padding_option="zero")
    desc_o = tl.make_tensor_descriptor(o_base, shape=[S, D_HEAD], strides=[stride_os, stride_od], block_shape=[BLOCK_M, D_HEAD], padding_option="zero")
    desc_k = tl.make_tensor_descriptor(k_base, shape=[S, D_HEAD], strides=[stride_ks, stride_kd], block_shape=[BLOCK_N, D_HEAD], padding_option="zero")
    desc_v = tl.make_tensor_descriptor(v_base, shape=[S, D_HEAD], strides=[stride_vs, stride_vd], block_shape=[BLOCK_N, D_HEAD], padding_option="zero")
    desc_dk = tl.make_tensor_descriptor(dk_base, shape=[S, D_HEAD], strides=[stride_dks, stride_dkd], block_shape=[BLOCK_N, D_HEAD], padding_option="zero")
    desc_dv = tl.make_tensor_descriptor(dv_base, shape=[S, D_HEAD], strides=[stride_dvs, stride_dvd], block_shape=[BLOCK_N, D_HEAD], padding_option="zero")
    
    offs_n_scalar = pid_n * BLOCK_N
    
    k = desc_k.load([offs_n_scalar, 0])
    v = desc_v.load([offs_n_scalar, 0])
    
    dk = tl.zeros([BLOCK_N, D_HEAD], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D_HEAD], dtype=tl.float32)
    
    m_start = offs_n_scalar // BLOCK_M
    m_steps = tl.cdiv(S, BLOCK_M)
    
    for m_idx in tl.range(m_start, m_steps, num_stages=3):
        offs_m_scalar = m_idx * BLOCK_M
        
        q = desc_q.load([offs_m_scalar, 0])
        do = desc_do.load([offs_m_scalar, 0])
        o = desc_o.load([offs_m_scalar, 0])
        
        idx_m = offs_m_scalar + tl.arange(0, BLOCK_M)
        l_ptrs = L + b * stride_lb + h * stride_lh + idx_m * stride_ls
        lse = tl.load(l_ptrs, mask=idx_m < S, other=float("inf"))
        
        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        p = tl.exp(qk - lse[:, None])
        
        idx_n = offs_n_scalar + tl.arange(0, BLOCK_N)
        causal_mask = (idx_m[:, None] >= idx_n[None, :]) & (idx_m[:, None] < S) & (idx_n[None, :] < S)
        p = tl.where(causal_mask, p, 0.0)
        
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        
        dv += tl.dot(tl.trans(p.to(q.dtype)), do)
        dk += tl.dot(tl.trans((ds * sm_scale).to(q.dtype)), q)
        
    desc_dk.store([offs_n_scalar, 0], dk.to(q.dtype))
    desc_dv.store([offs_n_scalar, 0], dv.to(q.dtype))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Evaluates causal multi-head attention backward gradients adhering cleanly directly to input semantics.
    Targets outputs exclusively natively via destination arrays without allocating replacements.
    """
    with torch.cuda.device(Q.device):
        B, H, S, d = Q.shape
        sm_scale = 1.0 / (d ** 0.5)

        stride_lb = L.stride(0)
        stride_lh = L.stride(1)
        stride_ls = L.stride(2) if L.dim() >= 3 else 1
        
        # dQ bounds heavily utilizing high M iteration counts per block load leveraging TMA natively 
        BLOCK_M_DQ, BLOCK_N_DQ = 128, 64
        grid_dq = (triton.cdiv(S, BLOCK_M_DQ), B * H)
        
        _bwd_kernel_dq[grid_dq](
            Q, K, V, O, sm_scale,
            dO, dQ, L,
            S, H,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
            stride_lb, stride_lh, stride_ls,
            BLOCK_M=BLOCK_M_DQ, BLOCK_N=BLOCK_N_DQ, D_HEAD=d,
            num_warps=8, num_stages=3
        )

        # dK and dV targets rigorously swapped shape orientations mapping into peak compute bands
        BLOCK_M_DK, BLOCK_N_DK = 64, 128
        grid_dkdv = (triton.cdiv(S, BLOCK_N_DK), B * H)
        
        _bwd_kernel_dk_dv[grid_dkdv](
            Q, K, V, O, sm_scale,
            dO, dK, dV, L,
            S, H,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
            dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
            stride_lb, stride_lh, stride_ls,
            BLOCK_M=BLOCK_M_DK, BLOCK_N=BLOCK_N_DK, D_HEAD=d,
            num_warps=8, num_stages=3
        )