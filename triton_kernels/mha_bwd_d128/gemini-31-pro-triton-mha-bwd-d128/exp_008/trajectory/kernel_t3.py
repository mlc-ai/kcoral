import math
import torch
import triton
import triton.language as tl

def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.jit
def bwd_preprocess_kernel(
    O, dO, D_workspace,
    stride_ob, stride_oh, stride_os,
    stride_dob, stride_doh, stride_dos,
    stride_dqb, stride_dqh, stride_dqs,
    S, H: tl.constexpr, d: tl.constexpr, BLOCK_S: tl.constexpr
):
    """
    Computes D_i = sum(O_i * dO_i, axis=1) for all sequence elements.
    Packs each resulting float32 scalar into two bfloat16s and temporarily
    stores them in elements 0 and 1 of the dQ output tensor to avoid HBM allocations.
    """
    i_id = tl.program_id(0)
    bh_id = tl.program_id(1)
    
    b_id = bh_id // H
    h_id = bh_id % H
    
    offs_s = i_id * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < S
    
    offs_d = tl.arange(0, d)
    
    o_ptr = O + b_id * stride_ob + h_id * stride_oh + offs_s[:, None] * stride_os + offs_d[None, :]
    do_ptr = dO + b_id * stride_dob + h_id * stride_doh + offs_s[:, None] * stride_dos + offs_d[None, :]
    
    o = tl.load(o_ptr, mask=mask_s[:, None], other=0.0).to(tl.float32)
    do = tl.load(do_ptr, mask=mask_s[:, None], other=0.0).to(tl.float32)
    
    d_val = tl.sum(o * do, axis=1)
    
    # Safely pack float32 bits into two 16-bit uint segments (avoiding sign extension)
    d_uint32 = tl.cast(d_val, tl.uint32, bitcast=True)
    d_bf16_0 = tl.cast(tl.cast(d_uint32 & 0xFFFF, tl.uint16), tl.bfloat16, bitcast=True)
    d_bf16_1 = tl.cast(tl.cast(d_uint32 >> 16, tl.uint16), tl.bfloat16, bitcast=True)
    
    ptr0 = D_workspace + b_id * stride_dqb + h_id * stride_dqh + offs_s * stride_dqs + 0
    ptr1 = D_workspace + b_id * stride_dqb + h_id * stride_dqh + offs_s * stride_dqs + 1
    
    tl.store(ptr0, d_bf16_0, mask=mask_s)
    tl.store(ptr1, d_bf16_1, mask=mask_s)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_R": 128, "BLOCK_C": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_R": 128, "BLOCK_C": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_R": 64, "BLOCK_C": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_R": 64, "BLOCK_C": 64}, num_warps=4, num_stages=3),
    ],
    key=["S"]
)
@triton.jit
def bwd_dq_kernel(
    Q, K, V, dO, L, dQ,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_dob, stride_doh, stride_dos,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs,
    S, scale,
    H: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr, d: tl.constexpr
):
    i_id = tl.program_id(0)
    bh_id = tl.program_id(1)
    
    b_id = bh_id // H
    h_id = bh_id % H
    
    offs_r = i_id * BLOCK_R + tl.arange(0, BLOCK_R)
    mask_r = offs_r < S
    
    # Read precomputed D_i mapped to dQ tensor without HBM round-tripping for O
    ptr0 = dQ + b_id * stride_dqb + h_id * stride_dqh + offs_r * stride_dqs + 0
    ptr1 = dQ + b_id * stride_dqb + h_id * stride_dqh + offs_r * stride_dqs + 1
    d_bf16_0 = tl.load(ptr0, mask=mask_r, other=0.0)
    d_bf16_1 = tl.load(ptr1, mask=mask_r, other=0.0)
    
    d_uint16_0 = tl.cast(d_bf16_0, tl.uint16, bitcast=True)
    d_uint16_1 = tl.cast(d_bf16_1, tl.uint16, bitcast=True)
    d_uint32 = (tl.cast(d_uint16_1, tl.uint32) << 16) | tl.cast(d_uint16_0, tl.uint32)
    d_i = tl.cast(d_uint32, tl.float32, bitcast=True)
    
    q_desc = tl.make_tensor_descriptor(
        Q + b_id * stride_qb + h_id * stride_qh,
        shape=[S, d], strides=[stride_qs, 1],
        block_shape=[BLOCK_R, d], padding_option="zero"
    )
    do_desc = tl.make_tensor_descriptor(
        dO + b_id * stride_dob + h_id * stride_doh,
        shape=[S, d], strides=[stride_dos, 1],
        block_shape=[BLOCK_R, d], padding_option="zero"
    )
    
    q_i = q_desc.load([i_id * BLOCK_R, 0])
    do_i = do_desc.load([i_id * BLOCK_R, 0])
    
    l_ptr = L + b_id * stride_lb + h_id * stride_lh + offs_r * stride_ls
    l_i = tl.load(l_ptr, mask=mask_r, other=0.0)
    
    dq_i = tl.zeros([BLOCK_R, d], dtype=tl.float32)
    num_steps = tl.cdiv(S, BLOCK_C)
    
    k_desc = tl.make_tensor_descriptor(
        K + b_id * stride_kb + h_id * stride_kh,
        shape=[S, d], strides=[stride_ks, 1],
        block_shape=[BLOCK_C, d], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V + b_id * stride_vb + h_id * stride_vh,
        shape=[S, d], strides=[stride_vs, 1],
        block_shape=[BLOCK_C, d], padding_option="zero"
    )
    
    for j in range(0, num_steps):
        k_j = k_desc.load([j * BLOCK_C, 0])
        v_j = v_desc.load([j * BLOCK_C, 0])
        
        # Hopper native fast-paths for TN operations via WGMMA
        s_ij = tl.dot(q_i, k_j.T, out_dtype=tl.float32) * scale
        
        offs_c = j * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs_c < S
        
        p_ij = tl.exp(s_ij - l_i[:, None])
        p_ij = tl.where((mask_r[:, None]) & (mask_c[None, :]), p_ij, 0.0)
        
        dp_ij = tl.dot(do_i, v_j.T, out_dtype=tl.float32)
        
        ds_ij = p_ij * (dp_ij - d_i[:, None]) * scale
        ds_ij_bf16 = ds_ij.to(tl.bfloat16)
        
        dq_i = tl.dot(ds_ij_bf16, k_j, acc=dq_i, out_dtype=tl.float32)
        
    # The output TMA store naturally overwrites our temporary mapped D_workspace safely.
    dq_desc = tl.make_tensor_descriptor(
        dQ + b_id * stride_dqb + h_id * stride_dqh,
        shape=[S, d], strides=[stride_dqs, 1],
        block_shape=[BLOCK_R, d]
    )
    dq_desc.store([i_id * BLOCK_R, 0], dq_i.to(tl.bfloat16))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 128, "BLOCK_R": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_C": 128, "BLOCK_R": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_C": 64, "BLOCK_R": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_C": 64, "BLOCK_R": 64}, num_warps=4, num_stages=3),
    ],
    key=["S"]
)
@triton.jit
def bwd_dk_dv_kernel(
    Q, K, V, dO, L, dK, dV, D_workspace,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_dob, stride_doh, stride_dos,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks,
    stride_dvb, stride_dvh, stride_dvs,
    stride_dqb, stride_dqh, stride_dqs,
    S, scale,
    H: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_R: tl.constexpr, d: tl.constexpr
):
    j_id = tl.program_id(0)
    bh_id = tl.program_id(1)
    
    b_id = bh_id // H
    h_id = bh_id % H
    
    k_desc = tl.make_tensor_descriptor(
        K + b_id * stride_kb + h_id * stride_kh,
        shape=[S, d], strides=[stride_ks, 1],
        block_shape=[BLOCK_C, d], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V + b_id * stride_vb + h_id * stride_vh,
        shape=[S, d], strides=[stride_vs, 1],
        block_shape=[BLOCK_C, d], padding_option="zero"
    )
    
    k_j = k_desc.load([j_id * BLOCK_C, 0])
    v_j = v_desc.load([j_id * BLOCK_C, 0])
    
    dk_j = tl.zeros([BLOCK_C, d], dtype=tl.float32)
    dv_j = tl.zeros([BLOCK_C, d], dtype=tl.float32)
    
    num_steps = tl.cdiv(S, BLOCK_R)
    
    q_desc = tl.make_tensor_descriptor(
        Q + b_id * stride_qb + h_id * stride_qh,
        shape=[S, d], strides=[stride_qs, 1],
        block_shape=[BLOCK_R, d], padding_option="zero"
    )
    do_desc = tl.make_tensor_descriptor(
        dO + b_id * stride_dob + h_id * stride_doh,
        shape=[S, d], strides=[stride_dos, 1],
        block_shape=[BLOCK_R, d], padding_option="zero"
    )
    
    offs_c = j_id * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < S
    
    for i in range(0, num_steps):
        q_i = q_desc.load([i * BLOCK_R, 0])
        do_i = do_desc.load([i * BLOCK_R, 0])
        
        offs_r = i * BLOCK_R + tl.arange(0, BLOCK_R)
        mask_r = offs_r < S
        
        l_ptr = L + b_id * stride_lb + h_id * stride_lh + offs_r * stride_ls
        l_i = tl.load(l_ptr, mask=mask_r, other=0.0)
        
        # Load packed scalar D_i
        ptr0 = D_workspace + b_id * stride_dqb + h_id * stride_dqh + offs_r * stride_dqs + 0
        ptr1 = D_workspace + b_id * stride_dqb + h_id * stride_dqh + offs_r * stride_dqs + 1
        d_bf16_0 = tl.load(ptr0, mask=mask_r, other=0.0)
        d_bf16_1 = tl.load(ptr1, mask=mask_r, other=0.0)
        
        d_uint16_0 = tl.cast(d_bf16_0, tl.uint16, bitcast=True)
        d_uint16_1 = tl.cast(d_bf16_1, tl.uint16, bitcast=True)
        d_uint32 = (tl.cast(d_uint16_1, tl.uint32) << 16) | tl.cast(d_uint16_0, tl.uint32)
        d_i = tl.cast(d_uint32, tl.float32, bitcast=True)
        
        # Maintain dual states directly aligning with Hopper's optimal TMAs & native WGMMA formats
        s_trans = tl.dot(k_j, q_i.T, out_dtype=tl.float32) * scale
        
        p_trans = tl.exp(s_trans - l_i[None, :])
        p_trans = tl.where((mask_c[:, None]) & (mask_r[None, :]), p_trans, 0.0)
        
        dp_trans = tl.dot(v_j, do_i.T, out_dtype=tl.float32)
        
        ds_trans = p_trans * (dp_trans - d_i[None, :]) * scale
        
        p_trans_bf16 = p_trans.to(tl.bfloat16)
        ds_trans_bf16 = ds_trans.to(tl.bfloat16)
        
        dv_j = tl.dot(p_trans_bf16, do_i, acc=dv_j, out_dtype=tl.float32)
        dk_j = tl.dot(ds_trans_bf16, q_i, acc=dk_j, out_dtype=tl.float32)
        
    dk_desc = tl.make_tensor_descriptor(
        dK + b_id * stride_dkb + h_id * stride_dkh,
        shape=[S, d], strides=[stride_dks, 1],
        block_shape=[BLOCK_C, d]
    )
    dv_desc = tl.make_tensor_descriptor(
        dV + b_id * stride_dvb + h_id * stride_dvh,
        shape=[S, d], strides=[stride_dvs, 1],
        block_shape=[BLOCK_C, d]
    )
    
    dk_desc.store([j_id * BLOCK_C, 0], dk_j.to(tl.bfloat16))
    dv_desc.store([j_id * BLOCK_C, 0], dv_j.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes Multi-Head Attention backward pass with TMA + WGMMA optimizations.
    Uses warp specialization by isolating workloads to two pipelined math kernels.
    Completely avoids any external HBM allocations for intermediates.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    
    stride_lb = L.stride(0)
    stride_lh = L.stride(1)
    stride_ls = L.stride(2)
    
    # 1. Dispatch lightweight D projection preprocess
    grid_pre = lambda META: (triton.cdiv(S, 64), B * H)
    bwd_preprocess_kernel[grid_pre](
        O, dO, dQ,
        O.stride(0), O.stride(1), O.stride(2),
        dO.stride(0), dO.stride(1), dO.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2),
        S, H=H, d=128, BLOCK_S=64, num_warps=4
    )
    
    # 2. Dispatch dK, dV accumulation kernel
    grid_dkdv = lambda META: (triton.cdiv(S, META["BLOCK_C"]), B * H)
    bwd_dk_dv_kernel[grid_dkdv](
        Q, K, V, dO, L, dK, dV, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        dO.stride(0), dO.stride(1), dO.stride(2),
        stride_lb, stride_lh, stride_ls,
        dK.stride(0), dK.stride(1), dK.stride(2),
        dV.stride(0), dV.stride(1), dV.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2),
        S, scale, H=H, d=128
    )

    # 3. Dispatch dQ accumulation kernel
    # Naturally writes final results to dQ, implicitly clearing out our prep-workspace
    grid_dq = lambda META: (triton.cdiv(S, META["BLOCK_R"]), B * H)
    bwd_dq_kernel[grid_dq](
        Q, K, V, dO, L, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        dO.stride(0), dO.stride(1), dO.stride(2),
        stride_lb, stride_lh, stride_ls,
        dQ.stride(0), dQ.stride(1), dQ.stride(2),
        S, scale, H=H, d=128
    )