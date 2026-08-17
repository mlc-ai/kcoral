import torch
import triton
import triton.language as tl

# Set up the descriptor allocator required for Hopper TMA (device-created descriptors)
def _descriptor_allocator(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(_descriptor_allocator)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
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
    
    q_ptr = Q + off_b * stride_qB + off_h * stride_qH
    k_ptr = K + off_b * stride_kB + off_h * stride_kH
    v_ptr = V + off_b * stride_vB + off_h * stride_vH
    o_ptr = O + off_b * stride_oB + off_h * stride_oH
    do_ptr = dO + off_b * stride_doB + off_h * stride_doH
    l_ptr = L + off_b * stride_lB + off_h * stride_lH
    dq_ptr = dQ + off_b * stride_dqB + off_h * stride_dqH
    
    q_desc = tl.make_tensor_descriptor(q_ptr, shape=[S, BLOCK_D], strides=[stride_qS, 1], block_shape=[BLOCK_M, BLOCK_D], padding_option="zero")
    o_desc = tl.make_tensor_descriptor(o_ptr, shape=[S, BLOCK_D], strides=[stride_oS, 1], block_shape=[BLOCK_M, BLOCK_D], padding_option="zero")
    do_desc = tl.make_tensor_descriptor(do_ptr, shape=[S, BLOCK_D], strides=[stride_doS, 1], block_shape=[BLOCK_M, BLOCK_D], padding_option="zero")
    k_desc = tl.make_tensor_descriptor(k_ptr, shape=[S, BLOCK_D], strides=[stride_kS, 1], block_shape=[BLOCK_N, BLOCK_D], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(v_ptr, shape=[S, BLOCK_D], strides=[stride_vS, 1], block_shape=[BLOCK_N, BLOCK_D], padding_option="zero")
    dq_desc = tl.make_tensor_descriptor(dq_ptr, shape=[S, BLOCK_D], strides=[stride_dqS, 1], block_shape=[BLOCK_M, BLOCK_D])
    
    off_m = pid_m * BLOCK_M
    q = q_desc.load([off_m, 0])
    o = o_desc.load([off_m, 0])
    do = do_desc.load([off_m, 0])
    
    offs_m = off_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    l = tl.load(l_ptr + offs_m * stride_lS, mask=mask_m, other=0.0)
    
    # Precompute row sum for stability
    D = tl.sum(tl.cast(do, tl.float32) * tl.cast(o, tl.float32), axis=1)
    
    dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    num_n_blocks = tl.cdiv(S, BLOCK_N)
    for n in range(num_n_blocks):
        off_n = n * BLOCK_N
        k = k_desc.load([off_n, 0])
        v = v_desc.load([off_n, 0])
        
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        
        offs_n = off_n + tl.arange(0, BLOCK_N)
        mask = mask_m[:, None] & (offs_n[None, :] < S)
        qk = tl.where(mask, qk, float("-inf"))
        
        p = tl.exp(qk - l[:, None])
        
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        ds = p * (dp - D[:, None])
        
        ds_cast = tl.cast(ds, q.dtype)
        dq = tl.dot(ds_cast, k, acc=dq, out_dtype=tl.float32)
        
    dq = dq * scale
    dq_desc.store([off_m, 0], tl.cast(dq, q.dtype))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=1),
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
    
    q_desc = tl.make_tensor_descriptor(q_ptr, shape=[S, BLOCK_D], strides=[stride_qS, 1], block_shape=[BLOCK_M, BLOCK_D], padding_option="zero")
    o_desc = tl.make_tensor_descriptor(o_ptr, shape=[S, BLOCK_D], strides=[stride_oS, 1], block_shape=[BLOCK_M, BLOCK_D], padding_option="zero")
    do_desc = tl.make_tensor_descriptor(do_ptr, shape=[S, BLOCK_D], strides=[stride_doS, 1], block_shape=[BLOCK_M, BLOCK_D], padding_option="zero")
    k_desc = tl.make_tensor_descriptor(k_ptr, shape=[S, BLOCK_D], strides=[stride_kS, 1], block_shape=[BLOCK_N, BLOCK_D], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(v_ptr, shape=[S, BLOCK_D], strides=[stride_vS, 1], block_shape=[BLOCK_N, BLOCK_D], padding_option="zero")
    dk_desc = tl.make_tensor_descriptor(dk_ptr, shape=[S, BLOCK_D], strides=[stride_dkS, 1], block_shape=[BLOCK_N, BLOCK_D])
    dv_desc = tl.make_tensor_descriptor(dv_ptr, shape=[S, BLOCK_D], strides=[stride_dvS, 1], block_shape=[BLOCK_N, BLOCK_D])
    
    off_n = pid_n * BLOCK_N
    k = k_desc.load([off_n, 0])
    v = v_desc.load([off_n, 0])
    
    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    
    offs_n = off_n + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    
    num_m_blocks = tl.cdiv(S, BLOCK_M)
    for m in range(num_m_blocks):
        off_m = m * BLOCK_M
        q = q_desc.load([off_m, 0])
        o = o_desc.load([off_m, 0])
        do = do_desc.load([off_m, 0])
        
        offs_m = off_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        l = tl.load(l_ptr + offs_m * stride_lS, mask=mask_m, other=0.0)
        
        D = tl.sum(tl.cast(do, tl.float32) * tl.cast(o, tl.float32), axis=1)
        
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        
        mask = mask_m[:, None] & mask_n[None, :]
        qk = tl.where(mask, qk, float("-inf"))
        
        p = tl.exp(qk - l[:, None])
        p_cast = tl.cast(p, q.dtype)
        
        dv = tl.dot(tl.trans(p_cast), do, acc=dv, out_dtype=tl.float32)
        
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        ds = p * (dp - D[:, None])
        
        ds_cast = tl.cast(ds, q.dtype)
        dk = tl.dot(tl.trans(ds_cast), q, acc=dk, out_dtype=tl.float32)
        
    dk = dk * scale
    dk_desc.store([off_n, 0], tl.cast(dk, k.dtype))
    dv_desc.store([off_n, 0], tl.cast(dv, v.dtype))


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