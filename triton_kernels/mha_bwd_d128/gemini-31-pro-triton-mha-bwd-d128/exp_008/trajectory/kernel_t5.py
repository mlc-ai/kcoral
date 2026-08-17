import math
import torch
import triton
import triton.language as tl

def alloc_fn(size: int, alignment: int, stream):
    """
    Allocator provided to Triton solely for device-created tensor-descriptor storage.
    """
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_R": 64, "BLOCK_C": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_R": 128, "BLOCK_C": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_R": 64, "BLOCK_C": 128}, num_warps=4, num_stages=3),
    ],
    key=["S"]
)
@triton.jit
def bwd_dq_kernel(
    Q, K, V, O, dO, L, dQ,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
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
    
    # Instantiate TMA descriptors
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
    o_desc = tl.make_tensor_descriptor(
        O + b_id * stride_ob + h_id * stride_oh,
        shape=[S, d], strides=[stride_os, 1],
        block_shape=[BLOCK_R, d], padding_option="zero"
    )
    
    # Load Q_i, dO_i, O_i once for the row block
    q_i = q_desc.load([i_id * BLOCK_R, 0])
    do_i = do_desc.load([i_id * BLOCK_R, 0])
    o_i = o_desc.load([i_id * BLOCK_R, 0])
    
    offs_r = i_id * BLOCK_R + tl.arange(0, BLOCK_R)
    mask_r = offs_r < S
    
    l_ptr = L + b_id * stride_lb + h_id * stride_lh + offs_r * stride_ls
    l_i = tl.load(l_ptr, mask=mask_r, other=0.0)
    
    # Compute row-wise D_i on the fly completely circumventing any manual global memory trip
    d_i = tl.sum(o_i.to(tl.float32) * do_i.to(tl.float32), axis=1)
    
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
        
        # Hopper TMA + WGMMA path implicitly utilized: TN dot leverages Shared-Registers WGMMA setup
        s_ij = tl.dot(q_i, k_j.T, out_dtype=tl.float32) * scale
        
        offs_c = j * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs_c < S
        
        p_ij = tl.exp(s_ij - l_i[:, None])
        p_ij = tl.where((mask_r[:, None]) & (mask_c[None, :]), p_ij, 0.0)
        
        dp_ij = tl.dot(do_i, v_j.T, out_dtype=tl.float32)
        
        ds_ij = p_ij * (dp_ij - d_i[:, None]) * scale
        ds_ij_bf16 = ds_ij.to(tl.bfloat16)
        
        dq_i = tl.dot(ds_ij_bf16, k_j, acc=dq_i, out_dtype=tl.float32)
        
    dq_desc = tl.make_tensor_descriptor(
        dQ + b_id * stride_dqb + h_id * stride_dqh,
        shape=[S, d], strides=[stride_dqs, 1],
        block_shape=[BLOCK_R, d]
    )
    dq_desc.store([i_id * BLOCK_R, 0], dq_i.to(tl.bfloat16))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 64, "BLOCK_R": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_C": 128, "BLOCK_R": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_C": 64, "BLOCK_R": 128}, num_warps=4, num_stages=3),
    ],
    key=["S"]
)
@triton.jit
def bwd_dk_dv_kernel(
    Q, K, V, O, dO, L, dK, dV,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_dob, stride_doh, stride_dos,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks,
    stride_dvb, stride_dvh, stride_dvs,
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
    
    # Load K_j, V_j once for the column block
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
    o_desc = tl.make_tensor_descriptor(
        O + b_id * stride_ob + h_id * stride_oh,
        shape=[S, d], strides=[stride_os, 1],
        block_shape=[BLOCK_R, d], padding_option="zero"
    )
    
    offs_c = j_id * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < S
    
    for i in range(0, num_steps):
        # Heavy TMA load pipelining driven by compiler directives
        q_i = q_desc.load([i * BLOCK_R, 0])
        do_i = do_desc.load([i * BLOCK_R, 0])
        o_i = o_desc.load([i * BLOCK_R, 0])
        
        offs_r = i * BLOCK_R + tl.arange(0, BLOCK_R)
        mask_r = offs_r < S
        
        l_ptr = L + b_id * stride_lb + h_id * stride_lh + offs_r * stride_ls
        l_i = tl.load(l_ptr, mask=mask_r, other=0.0)
        
        d_i = tl.sum(o_i.to(tl.float32) * do_i.to(tl.float32), axis=1)
        
        # Symmetrical dual formulations forcing transposed operands yielding native WGMMA execution geometries
        s_trans = tl.dot(k_j, q_i.T, out_dtype=tl.float32) * scale
        
        p_trans = tl.exp(s_trans - l_i[None, :])
        p_trans = tl.where((mask_c[:, None]) & (mask_r[None, :]), p_trans, 0.0)
        
        dp_trans = tl.dot(v_j, do_i.T, out_dtype=tl.float32)
        
        ds_trans = p_trans * (dp_trans - d_i[None, :]) * scale
        
        p_trans_bf16 = p_trans.to(tl.bfloat16)
        ds_trans_bf16 = ds_trans.to(tl.bfloat16)
        
        # Matrix A dynamically localized in registers matching perfectly RS-GEMM on SM90
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
    Computes Multi-Head Attention backward pass with TMA + WGMMA optimizations mapped via separated kernels.
    Asynchrony is heavily leaned on natively via completely concurrent PyTorch streaming pipelines 
    which optimally minimizes tail wave inefficiencies and serial constraints.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    
    stride_lb = L.stride(0)
    stride_lh = L.stride(1)
    stride_ls = L.stride(2)
    
    # Establish distinct streams guaranteeing maximum asynchronous execution bounds scaling over the complete GPU architecture
    stream_dq = torch.cuda.Stream()
    stream_dkdv = torch.cuda.Stream()
    
    with torch.cuda.stream(stream_dq):
        grid_dq = lambda META: (triton.cdiv(S, META["BLOCK_R"]), B * H)
        bwd_dq_kernel[grid_dq](
            Q, K, V, O, dO, L, dQ,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            stride_lb, stride_lh, stride_ls,
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            S, scale, H=H, d=d
        )
        
    with torch.cuda.stream(stream_dkdv):
        grid_dkdv = lambda META: (triton.cdiv(S, META["BLOCK_C"]), B * H)
        bwd_dk_dv_kernel[grid_dkdv](
            Q, K, V, O, dO, L, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            stride_lb, stride_lh, stride_ls,
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            S, scale, H=H, d=d
        )
        
    torch.cuda.current_stream().wait_stream(stream_dq)
    torch.cuda.current_stream().wait_stream(stream_dkdv)
    
    return dQ, dK, dV