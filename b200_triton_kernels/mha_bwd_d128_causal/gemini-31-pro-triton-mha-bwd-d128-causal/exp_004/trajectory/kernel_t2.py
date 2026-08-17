import torch
import triton
import triton.language as tl

def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

_configs_dq = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
]

@triton.autotune(
    configs=_configs_dq,
    key=["S"],
)
@triton.jit
def bwd_dq_kernel(
    Q, K, V, O, dO, L,
    dQ,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    S, H: tl.constexpr, d: tl.constexpr,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    if pid_m * BLOCK_M >= S:
        return
        
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    off_b_h_q = pid_b * stride_qb + pid_h * stride_qh
    off_b_h_do = pid_b * stride_dob + pid_h * stride_doh
    off_b_h_o = pid_b * stride_ob + pid_h * stride_oh
    off_b_h_l = pid_b * stride_lb + pid_h * stride_lh
    
    Q_desc = tl.make_tensor_descriptor(
        Q + off_b_h_q, shape=[S, d], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    dO_desc = tl.make_tensor_descriptor(
        dO + off_b_h_do, shape=[S, d], strides=[stride_dos, stride_dod],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    O_desc = tl.make_tensor_descriptor(
        O + off_b_h_o, shape=[S, d], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    dQ_desc = tl.make_tensor_descriptor(
        dQ + (pid_b * stride_dqb + pid_h * stride_dqh), shape=[S, d], strides=[stride_dqs, stride_dqd],
        block_shape=[BLOCK_M, d]
    )
    
    q = tl.load(Q_desc, [pid_m * BLOCK_M, 0])
    do = tl.load(dO_desc, [pid_m * BLOCK_M, 0])
    o = tl.load(O_desc, [pid_m * BLOCK_M, 0])
    
    d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = off_m < S
    
    L_ptr = L + off_b_h_l + off_m * stride_ls
    l = tl.load(L_ptr, mask=mask_m, other=0.0)
    
    dq_acc = tl.zeros([BLOCK_M, d], dtype=tl.float32)
    
    off_b_h_k = pid_b * stride_kb + pid_h * stride_kh
    off_b_h_v = pid_b * stride_vb + pid_h * stride_vh
    
    K_desc = tl.make_tensor_descriptor(
        K + off_b_h_k, shape=[S, d], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    V_desc = tl.make_tensor_descriptor(
        V + off_b_h_v, shape=[S, d], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    
    end_n = tl.minimum((pid_m + 1) * BLOCK_M, S)
    
    for start_n in range(0, end_n, BLOCK_N):
        k = tl.load(K_desc, [start_n, 0])
        v = tl.load(V_desc, [start_n, 0])
        
        s_mat = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        
        off_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = off_n < S
        
        causal_mask = (off_m[:, None] >= off_n[None, :]) & mask_m[:, None] & mask_n[None, :]
        s_mat = tl.where(causal_mask, s_mat, float("-inf"))
        
        p = tl.exp(s_mat - l[:, None])
        p = tl.where(causal_mask, p, 0.0)
        
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        
        ds = (dp - d_val[:, None]) * p * scale
        
        dq_acc += tl.dot(ds.to(tl.bfloat16), k, out_dtype=tl.float32)
        
    tl.store(dQ_desc, [pid_m * BLOCK_M, 0], dq_acc.to(tl.bfloat16))


_configs_dkdv = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
]

@triton.autotune(
    configs=_configs_dkdv,
    key=["S"],
)
@triton.jit
def bwd_dkdv_kernel(
    Q, K, V, O, dO, L,
    dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    S, H: tl.constexpr, d: tl.constexpr,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    if pid_n * BLOCK_N >= S:
        return
        
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    off_b_h_k = pid_b * stride_kb + pid_h * stride_kh
    off_b_h_v = pid_b * stride_vb + pid_h * stride_vh
    
    K_desc = tl.make_tensor_descriptor(
        K + off_b_h_k, shape=[S, d], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    V_desc = tl.make_tensor_descriptor(
        V + off_b_h_v, shape=[S, d], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    dK_desc = tl.make_tensor_descriptor(
        dK + (pid_b * stride_dkb + pid_h * stride_dkh), shape=[S, d], strides=[stride_dks, stride_dkd],
        block_shape=[BLOCK_N, d]
    )
    dV_desc = tl.make_tensor_descriptor(
        dV + (pid_b * stride_dvb + pid_h * stride_dvh), shape=[S, d], strides=[stride_dvs, stride_dvd],
        block_shape=[BLOCK_N, d]
    )
    
    k = tl.load(K_desc, [pid_n * BLOCK_N, 0])
    v = tl.load(V_desc, [pid_n * BLOCK_N, 0])
    
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = off_n < S
    
    dk_acc = tl.zeros([BLOCK_N, d], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_N, d], dtype=tl.float32)
    
    off_b_h_q = pid_b * stride_qb + pid_h * stride_qh
    off_b_h_do = pid_b * stride_dob + pid_h * stride_doh
    off_b_h_o = pid_b * stride_ob + pid_h * stride_oh
    off_b_h_l = pid_b * stride_lb + pid_h * stride_lh
    
    Q_desc = tl.make_tensor_descriptor(
        Q + off_b_h_q, shape=[S, d], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    dO_desc = tl.make_tensor_descriptor(
        dO + off_b_h_do, shape=[S, d], strides=[stride_dos, stride_dod],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    O_desc = tl.make_tensor_descriptor(
        O + off_b_h_o, shape=[S, d], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    
    start_m = pid_n * BLOCK_N
    start_m = (start_m // BLOCK_M) * BLOCK_M
    
    for start_m_idx in range(start_m, S, BLOCK_M):
        q = tl.load(Q_desc, [start_m_idx, 0])
        do = tl.load(dO_desc, [start_m_idx, 0])
        o = tl.load(O_desc, [start_m_idx, 0])
        
        d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        
        s_mat = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        
        off_m = start_m_idx + tl.arange(0, BLOCK_M)
        mask_m = off_m < S
        
        causal_mask = (off_m[:, None] >= off_n[None, :]) & mask_m[:, None] & mask_n[None, :]
        s_mat = tl.where(causal_mask, s_mat, float("-inf"))
        
        L_ptr = L + off_b_h_l + off_m * stride_ls
        l = tl.load(L_ptr, mask=mask_m, other=0.0)
        
        p = tl.exp(s_mat - l[:, None])
        p = tl.where(causal_mask, p, 0.0)
        
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        
        ds = (dp - d_val[:, None]) * p * scale
        
        p_bf16 = p.to(tl.bfloat16)
        dv_acc += tl.dot(tl.trans(p_bf16), do, out_dtype=tl.float32)
        
        ds_bf16 = ds.to(tl.bfloat16)
        dk_acc += tl.dot(tl.trans(ds_bf16), q, out_dtype=tl.float32)
        
    tl.store(dK_desc, [pid_n * BLOCK_N, 0], dk_acc.to(tl.bfloat16))
    tl.store(dV_desc, [pid_n * BLOCK_N, 0], dv_acc.to(tl.bfloat16))

def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    scale = 1.0 / (d ** 0.5)
    
    grid_dq = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
    
    bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, L,
        dQ,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        S, H=H, d=d,
        scale=scale,
    )
    
    grid_dkdv = lambda META: (triton.cdiv(S, META['BLOCK_N']), B * H)
    
    bwd_dkdv_kernel[grid_dkdv](
        Q, K, V, O, dO, L,
        dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        S, H=H, d=d,
        scale=scale,
    )