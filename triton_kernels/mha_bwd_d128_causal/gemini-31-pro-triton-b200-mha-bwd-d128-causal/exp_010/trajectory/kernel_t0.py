import math
import torch
import triton
import triton.language as tl

@triton.jit
def _zero_tensor_3d(
    ptr,
    stride_b, stride_h, stride_s, stride_d,
    B, H, S, D,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_d = tl.arange(0, BLOCK_D)
    
    mask = (offs_s[:, None] < S) & (offs_d[None, :] < D)
    
    ptrs = ptr + pid_b * stride_b + pid_h * stride_h + offs_s[:, None] * stride_s + offs_d[None, :] * stride_d
    tl.store(ptrs, 0.0, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
    ],
    key=["seq_len"]
)
@triton.jit
def _bwd_kernel(
    q, k, v, o, do, lse, dq, dk, dv,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    num_heads, seq_len, head_dim,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    pid = tl.program_id(0) 
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    start_m = pid * BLOCK_M
    
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    
    mask_m = offs_m < seq_len
    mask_d = offs_d < head_dim
    
    q_ptrs = q + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    o_ptrs = o + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    do_ptrs = do + pid_b * stride_dob + pid_h * stride_doh + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    lse_ptrs = lse + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ls
    
    q_tile = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    o_tile = tl.load(o_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    do_tile = tl.load(do_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    lse_tile = tl.load(lse_ptrs, mask=mask_m, other=0.0)
    
    delta = tl.sum(do_tile.to(tl.float32) * o_tile.to(tl.float32), axis=1)
    
    dq_acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    
    start_n = 0
    end_n = tl.minimum(start_m + BLOCK_M, seq_len)
    
    for start_j in range(start_n, end_n, BLOCK_N):
        curr_offs_n = start_j + offs_n
        mask_n = curr_offs_n < seq_len
        
        k_ptrs = k + pid_b * stride_kb + pid_h * stride_kh + curr_offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = v + pid_b * stride_vb + pid_h * stride_vh + curr_offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        
        k_tile = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        v_tile = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        
        scores_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        scores = tl.dot(q_tile, k_tile.T, scores_acc) * sm_scale
        
        causal_mask = offs_m[:, None] >= curr_offs_n[None, :]
        valid_mask = causal_mask & mask_m[:, None] & mask_n[None, :]
        scores = tl.where(valid_mask, scores, float('-inf'))
        
        p = tl.math.exp(scores - lse_tile[:, None])
        p = tl.where(valid_mask, p, 0.0)
        
        dp_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        dp = tl.dot(do_tile, v_tile.T, dp_acc)
        
        ds = p * (dp - delta[:, None]) * sm_scale
        ds = tl.where(valid_mask, ds, 0.0)
        
        ds_cast = ds.to(q_tile.dtype)
        p_cast = p.to(q_tile.dtype)
        
        dq_acc = tl.dot(ds_cast, k_tile, dq_acc)
        
        dk_acc = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
        dk_partial = tl.dot(ds_cast.T, q_tile, dk_acc)
        
        dv_acc = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
        dv_partial = tl.dot(p_cast.T, do_tile, dv_acc)
        
        dk_ptrs = dk + pid_b * stride_dkb + pid_h * stride_dkh + curr_offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
        dv_ptrs = dv + pid_b * stride_dvb + pid_h * stride_dvh + curr_offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
        
        tl.atomic_add(dk_ptrs, dk_partial.to(q_tile.dtype), mask=mask_n[:, None] & mask_d[None, :])
        tl.atomic_add(dv_ptrs, dv_partial.to(q_tile.dtype), mask=mask_n[:, None] & mask_d[None, :])
        
    dq_ptrs = dq + pid_b * stride_dqb + pid_h * stride_dqh + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq_acc.to(q_tile.dtype), mask=mask_m[:, None] & mask_d[None, :])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes causal attention backward pass.
    Writes into preallocated tensors dQ, dK, dV.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, d_dim = Q.shape
    stride_ls = L.stride(2)
        
    grid_zero_3d = (triton.cdiv(S, 64), H, B)
    _zero_tensor_3d[grid_zero_3d](
        dK,
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        B, H, S, d_dim,
        BLOCK_S=64, BLOCK_D=128
    )
    _zero_tensor_3d[grid_zero_3d](
        dV,
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        B, H, S, d_dim,
        BLOCK_S=64, BLOCK_D=128
    )
    
    sm_scale = 1.0 / math.sqrt(d_dim)
    
    def grid(META):
        return (triton.cdiv(S, META['BLOCK_M']), H, B)
        
    _bwd_kernel[grid](
        Q, K, V, O, dO, L, dQ, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), stride_ls,
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        H, S, d_dim,
        sm_scale,
        BLOCK_D=128
    )
    return dQ, dK, dV