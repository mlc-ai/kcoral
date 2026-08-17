import math
import torch
import triton
import triton.language as tl

def _alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(_alloc_fn)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
    ],
    key=['S']
)
@triton.jit
def bwd_dq_kernel(
    Q, K, V, O, dO, L, dQ,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    scale, S, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b_idx = pid_bh // H
    h_idx = pid_bh % H
    
    offset_q_bh = b_idx * stride_qb + h_idx * stride_qh
    offset_k_bh = b_idx * stride_kb + h_idx * stride_kh
    offset_v_bh = b_idx * stride_vb + h_idx * stride_vh
    offset_o_bh = b_idx * stride_ob + h_idx * stride_oh
    offset_do_bh = b_idx * stride_dob + h_idx * stride_doh
    offset_l_bh = b_idx * stride_lb + h_idx * stride_lh
    offset_dq_bh = b_idx * stride_dqb + h_idx * stride_dqh

    # Device descriptors mapped directly to hardware TMA instructions
    q_desc = tl.make_tensor_descriptor(
        Q + offset_q_bh, shape=[S, d], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        O + offset_o_bh, shape=[S, d], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    do_desc = tl.make_tensor_descriptor(
        dO + offset_do_bh, shape=[S, d], strides=[stride_dos, stride_dod],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    dq_desc = tl.make_tensor_descriptor(
        dQ + offset_dq_bh, shape=[S, d], strides=[stride_dqs, stride_dqd],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    
    k_desc = tl.make_tensor_descriptor(
        K + offset_k_bh, shape=[S, d], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V + offset_v_bh, shape=[S, d], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )

    offset_m = pid_m * BLOCK_M
    q_i = q_desc.load([offset_m, 0])
    o_i = o_desc.load([offset_m, 0])
    do_i = do_desc.load([offset_m, 0])
    
    offs_m = offset_m + tl.arange(0, BLOCK_M)
    l_ptrs = L + offset_l_bh + offs_m * stride_ls
    l_i = tl.load(l_ptrs, mask=offs_m < S, other=0.0)
    
    Di = tl.sum((do_i.to(tl.float32)) * (o_i.to(tl.float32)), axis=1)
    
    dq_i = tl.zeros([BLOCK_M, d], dtype=tl.float32)
    
    max_j = tl.minimum(S, (pid_m + 1) * BLOCK_M)
    num_j_blocks = tl.cdiv(max_j, BLOCK_N)
    
    for j_block in range(num_j_blocks):
        offset_n = j_block * BLOCK_N
        k_j = k_desc.load([offset_n, 0])
        v_j = v_desc.load([offset_n, 0])
        
        # S_ij = Q_i @ K_j^T * scale
        s_ij = tl.dot(q_i, tl.trans(k_j)) * scale
        
        offs_n = offset_n + tl.arange(0, BLOCK_N)
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        valid = causal_mask & (offs_m[:, None] < S) & (offs_n[None, :] < S)
        
        s_ij_safe = tl.where(valid, s_ij, l_i[:, None])
        p_ij = tl.exp(s_ij_safe - l_i[:, None])
        p_ij = tl.where(valid, p_ij, 0.0)
        
        # dp_ij = dO_i @ V_j^T
        dp_ij = tl.dot(do_i, tl.trans(v_j))
        
        # Compute exact scale on FP32 prior to casting for numerical accuracy
        ds_ij = p_ij * (dp_ij - Di[:, None]) * scale
        
        # dQ_i += dS_ij @ K_j
        dq_i = tl.dot(ds_ij.to(q_i.dtype), k_j, dq_i)
        
    dq_desc.store([offset_m, 0], dq_i.to(dQ.dtype.element_ty))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
    ],
    key=['S']
)
@triton.jit
def bwd_dk_dv_kernel(
    Q, K, V, O, dO, L, dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    scale, S, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b_idx = pid_bh // H
    h_idx = pid_bh % H
    
    offset_q_bh = b_idx * stride_qb + h_idx * stride_qh
    offset_k_bh = b_idx * stride_kb + h_idx * stride_kh
    offset_v_bh = b_idx * stride_vb + h_idx * stride_vh
    offset_o_bh = b_idx * stride_ob + h_idx * stride_oh
    offset_do_bh = b_idx * stride_dob + h_idx * stride_doh
    offset_l_bh = b_idx * stride_lb + h_idx * stride_lh
    offset_dk_bh = b_idx * stride_dkb + h_idx * stride_dkh
    offset_dv_bh = b_idx * stride_dvb + h_idx * stride_dvh

    q_desc = tl.make_tensor_descriptor(
        Q + offset_q_bh, shape=[S, d], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        O + offset_o_bh, shape=[S, d], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    do_desc = tl.make_tensor_descriptor(
        dO + offset_do_bh, shape=[S, d], strides=[stride_dos, stride_dod],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        K + offset_k_bh, shape=[S, d], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V + offset_v_bh, shape=[S, d], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    dk_desc = tl.make_tensor_descriptor(
        dK + offset_dk_bh, shape=[S, d], strides=[stride_dks, stride_dkd],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    dv_desc = tl.make_tensor_descriptor(
        dV + offset_dv_bh, shape=[S, d], strides=[stride_dvs, stride_dvd],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )

    offset_n = pid_n * BLOCK_N
    k_j = k_desc.load([offset_n, 0])
    v_j = v_desc.load([offset_n, 0])
    
    dk_j = tl.zeros([BLOCK_N, d], dtype=tl.float32)
    dv_j = tl.zeros([BLOCK_N, d], dtype=tl.float32)
    
    start_i_block = offset_n // BLOCK_M
    num_m_blocks = tl.cdiv(S, BLOCK_M)
    
    offs_n = offset_n + tl.arange(0, BLOCK_N)
    
    for i_block in range(start_i_block, num_m_blocks):
        offset_m = i_block * BLOCK_M
        
        q_i = q_desc.load([offset_m, 0])
        o_i = o_desc.load([offset_m, 0])
        do_i = do_desc.load([offset_m, 0])
        
        offs_m = offset_m + tl.arange(0, BLOCK_M)
        l_ptrs_i = L + offset_l_bh + offs_m * stride_ls
        l_i = tl.load(l_ptrs_i, mask=offs_m < S, other=0.0)
        
        Di = tl.sum((do_i.to(tl.float32)) * (o_i.to(tl.float32)), axis=1)
        
        s_ij = tl.dot(q_i, tl.trans(k_j)) * scale
        
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        valid = causal_mask & (offs_m[:, None] < S) & (offs_n[None, :] < S)
        
        s_ij_safe = tl.where(valid, s_ij, l_i[:, None])
        p_ij = tl.exp(s_ij_safe - l_i[:, None])
        p_ij = tl.where(valid, p_ij, 0.0)
        
        # dV_j += P_ij^T @ dO_i
        dv_j = tl.dot(tl.trans(p_ij.to(do_i.dtype)), do_i, dv_j)
        
        dp_ij = tl.dot(do_i, tl.trans(v_j))
        ds_ij = p_ij * (dp_ij - Di[:, None]) * scale
        
        # dK_j += dS_ij^T @ Q_i
        dk_j = tl.dot(tl.trans(ds_ij.to(q_i.dtype)), q_i, dk_j)
        
    dk_desc.store([offset_n, 0], dk_j.to(dK.dtype.element_ty))
    dv_desc.store([offset_n, 0], dv_j.to(dV.dtype.element_ty))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    with torch.cuda.device(Q.device):
        B, H, S, d = Q.shape
        scale = 1.0 / math.sqrt(d)
        
        # Evaluation pass for dQ
        grid_dq = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
        bwd_dq_kernel[grid_dq](
            Q, K, V, O, dO, L, dQ,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
            scale, S, H,
            d=128
        )
        
        # Evaluation pass for dK and dV
        grid_dk_dv = lambda META: (triton.cdiv(S, META['BLOCK_N']), B * H)
        bwd_dk_dv_kernel[grid_dk_dv](
            Q, K, V, O, dO, L, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
            dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
            scale, S, H,
            d=128
        )