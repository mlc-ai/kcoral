import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def bwd_kernel_dq(
    Q, K, V, O, dO, L, dQ,
    stride_qB, stride_qH, stride_qS,
    stride_kB, stride_kH, stride_kS,
    stride_vB, stride_vH, stride_vS,
    stride_oB, stride_oH, stride_oS,
    stride_doB, stride_doH, stride_doS,
    stride_lB, stride_lH, stride_lS,
    stride_dqB, stride_dqH, stride_dqS,
    H, S, scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    off_b = pid_bh // H
    off_h = pid_bh % H
    
    # Base pointers for this head
    q_ptr = Q + off_b * stride_qB + off_h * stride_qH
    k_ptr = K + off_b * stride_kB + off_h * stride_kH
    v_ptr = V + off_b * stride_vB + off_h * stride_vH
    o_ptr = O + off_b * stride_oB + off_h * stride_oH
    do_ptr = dO + off_b * stride_doB + off_h * stride_doH
    l_ptr = L + off_b * stride_lB + off_h * stride_lH
    dq_ptr = dQ + off_b * stride_dqB + off_h * stride_dqH
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    
    q_ptrs = q_ptr + offs_m[:, None] * stride_qS + offs_d[None, :]
    o_ptrs = o_ptr + offs_m[:, None] * stride_oS + offs_d[None, :]
    do_ptrs = do_ptr + offs_m[:, None] * stride_doS + offs_d[None, :]
    l_ptrs = l_ptr + offs_m * stride_lS
    
    mask_m = offs_m < S
    
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    D = tl.sum(tl.cast(do, tl.float32) * tl.cast(o, tl.float32), axis=1)
    dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    num_n_blocks = tl.cdiv(S, BLOCK_N)
    
    # Iterate over K and V
    for n in tl.range(0, num_n_blocks):
        offs_n = n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        
        k_ptrs = k_ptr + offs_n[:, None] * stride_kS + offs_d[None, :]
        v_ptrs = v_ptr + offs_n[:, None] * stride_vS + offs_d[None, :]
        
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        qk = tl.where(mask_m[:, None] & mask_n[None, :], qk, float("-inf"))
        
        p = tl.exp(qk - l[:, None])
        
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        ds = p * (dp - D[:, None])
        
        ds_cast = tl.cast(ds, q.dtype)
        dq = tl.dot(ds_cast, k, acc=dq, out_dtype=tl.float32)
        
    dq = dq * scale
    dq_ptrs = dq_ptr + offs_m[:, None] * stride_dqS + offs_d[None, :]
    tl.store(dq_ptrs, tl.cast(dq, q.dtype), mask=mask_m[:, None])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def bwd_kernel_dk_dv(
    Q, K, V, O, dO, L, dK, dV,
    stride_qB, stride_qH, stride_qS,
    stride_kB, stride_kH, stride_kS,
    stride_vB, stride_vH, stride_vS,
    stride_oB, stride_oH, stride_oS,
    stride_doB, stride_doH, stride_doS,
    stride_lB, stride_lH, stride_lS,
    stride_dkB, stride_dkH, stride_dkS,
    stride_dvB, stride_dvH, stride_dvS,
    H, S, scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    off_b = pid_bh // H
    off_h = pid_bh % H
    
    q_ptr = Q + off_b * stride_qB + off_h * stride_qH
    k_ptr = K + off_b * stride_kB + off_h * stride_kH
    v_ptr = V + off_b * stride_vB + off_h * stride_vH
    o_ptr = O + off_b * stride_oB + off_h * stride_oH
    do_ptr = dO + off_b * stride_doB + off_h * stride_doH
    l_ptr = L + off_b * stride_lB + off_h * stride_lH
    dk_ptr = dK + off_b * stride_dkB + off_h * stride_dkH
    dv_ptr = dV + off_b * stride_dvB + off_h * stride_dvH
    
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    
    k_ptrs = k_ptr + offs_n[:, None] * stride_kS + offs_d[None, :]
    v_ptrs = v_ptr + offs_n[:, None] * stride_vS + offs_d[None, :]
    
    mask_n = offs_n < S
    
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    
    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    
    num_m_blocks = tl.cdiv(S, BLOCK_M)
    
    # Iterate over Q, O, dO
    for m in tl.range(0, num_m_blocks):
        offs_m = m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        
        q_ptrs = q_ptr + offs_m[:, None] * stride_qS + offs_d[None, :]
        o_ptrs = o_ptr + offs_m[:, None] * stride_oS + offs_d[None, :]
        do_ptrs = do_ptr + offs_m[:, None] * stride_doS + offs_d[None, :]
        l_ptrs = l_ptr + offs_m * stride_lS
        
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        l = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        D = tl.sum(tl.cast(do, tl.float32) * tl.cast(o, tl.float32), axis=1)
        
        # kq = k @ q^T. Eliminates explicit transposes on local matrix P
        kq = tl.dot(k, tl.trans(q), out_dtype=tl.float32) * scale
        kq = tl.where(mask_n[:, None] & mask_m[None, :], kq, float("-inf"))
        
        p_T = tl.exp(kq - l[None, :])
        
        p_T_cast = tl.cast(p_T, q.dtype)
        # dv += p_T @ do
        dv = tl.dot(p_T_cast, do, acc=dv, out_dtype=tl.float32)
        
        # dp_T = v @ do^T
        dp_T = tl.dot(v, tl.trans(do), out_dtype=tl.float32)
        
        # ds_T = p_T * (dp_T - D_T)
        ds_T = p_T * (dp_T - D[None, :])
        
        ds_T_cast = tl.cast(ds_T, q.dtype)
        # dk += ds_T @ q
        dk = tl.dot(ds_T_cast, q, acc=dk, out_dtype=tl.float32)
        
    dk = dk * scale
    dk_ptrs = dk_ptr + offs_n[:, None] * stride_dkS + offs_d[None, :]
    dv_ptrs = dv_ptr + offs_n[:, None] * stride_dvS + offs_d[None, :]
    
    tl.store(dk_ptrs, tl.cast(dk, k.dtype), mask=mask_n[:, None])
    tl.store(dv_ptrs, tl.cast(dv, v.dtype), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes backward pass for multi-head attention on Hopper, returning updates directly in destination tensors.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    scale = 1.0 / (d ** 0.5)
    
    # Flatten optional trailing dim on L if present
    L_view = L.view(B, H, S)
    
    grid_dq = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H)
    bwd_kernel_dq[grid_dq](
        Q, K, V, O, dO, L_view, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        dO.stride(0), dO.stride(1), dO.stride(2),
        L_view.stride(0), L_view.stride(1), L_view.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2),
        H, S, scale,
        BLOCK_D=d,
    )
    
    grid_dk_dv = lambda META: (triton.cdiv(S, META["BLOCK_N"]), B * H)
    bwd_kernel_dk_dv[grid_dk_dv](
        Q, K, V, O, dO, L_view, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        dO.stride(0), dO.stride(1), dO.stride(2),
        L_view.stride(0), L_view.stride(1), L_view.stride(2),
        dK.stride(0), dK.stride(1), dK.stride(2),
        dV.stride(0), dV.stride(1), dV.stride(2),
        H, S, scale,
        BLOCK_D=d,
    )