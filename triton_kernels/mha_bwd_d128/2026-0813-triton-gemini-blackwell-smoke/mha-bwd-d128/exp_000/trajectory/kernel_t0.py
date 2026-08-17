import torch
import triton
import triton.language as tl


@triton.jit
def _bwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, l_ptr,
    dq_ptr, dk_ptr, dv_ptr,
    S,
    stride_qz, stride_qh, stride_qs, stride_qd,
    stride_kz, stride_kh, stride_ks, stride_kd,
    stride_vz, stride_vh, stride_vs, stride_vd,
    stride_oz, stride_oh, stride_os, stride_od,
    stride_doz, stride_doh, stride_dos, stride_dod,
    stride_lz, stride_lh, stride_ls,
    stride_dqz, stride_dqh, stride_dqs, stride_dqd,
    stride_dkz, stride_dkh, stride_dks, stride_dkd,
    stride_dvz, stride_dvh, stride_dvs, stride_dvd,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    d: tl.constexpr,
    H: tl.constexpr
):
    pid_tile = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    num_kv_tiles = tl.cdiv(S, BLOCK_N)
    num_q_tiles = tl.cdiv(S, BLOCK_M)
    
    b = pid_bh // H
    h = pid_bh % H
    
    q_ptr = q_ptr + b * stride_qz + h * stride_qh
    k_ptr = k_ptr + b * stride_kz + h * stride_kh
    v_ptr = v_ptr + b * stride_vz + h * stride_vh
    o_ptr = o_ptr + b * stride_oz + h * stride_oh
    do_ptr = do_ptr + b * stride_doz + h * stride_doh
    l_ptr = l_ptr + b * stride_lz + h * stride_lh
    dq_ptr = dq_ptr + b * stride_dqz + h * stride_dqh
    dk_ptr = dk_ptr + b * stride_dkz + h * stride_dkh
    dv_ptr = dv_ptr + b * stride_dvz + h * stride_dvh
    
    if pid_tile < num_kv_tiles:
        # -------------------------------------------------------------
        # dK / dV owner region
        # exclusive ownership of one dK/dV tile, reducing over all Q tiles
        # -------------------------------------------------------------
        kv_tile = pid_tile
        offs_n = kv_tile * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, d)
        
        valid_n = offs_n < S
        offs_n_safe = tl.where(valid_n, offs_n, 0)
        
        k_ptrs = k_ptr + offs_n_safe[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = v_ptr + offs_n_safe[:, None] * stride_vs + offs_d[None, :] * stride_vd
        
        k = tl.load(k_ptrs, mask=valid_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=valid_n[:, None], other=0.0)
        
        dk = tl.zeros((BLOCK_N, d), dtype=tl.float32)
        dv = tl.zeros((BLOCK_N, d), dtype=tl.float32)
        
        for q_tile in range(0, num_q_tiles):
            offs_m = q_tile * BLOCK_M + tl.arange(0, BLOCK_M)
            valid_m = offs_m < S
            offs_m_safe = tl.where(valid_m, offs_m, 0)
            
            q_ptrs = q_ptr + offs_m_safe[:, None] * stride_qs + offs_d[None, :] * stride_qd
            do_ptrs = do_ptr + offs_m_safe[:, None] * stride_dos + offs_d[None, :] * stride_dod
            o_ptrs = o_ptr + offs_m_safe[:, None] * stride_os + offs_d[None, :] * stride_od
            l_ptrs = l_ptr + offs_m_safe * stride_ls
            
            q = tl.load(q_ptrs, mask=valid_m[:, None], other=0.0)
            do = tl.load(do_ptrs, mask=valid_m[:, None], other=0.0)
            o = tl.load(o_ptrs, mask=valid_m[:, None], other=0.0)
            l = tl.load(l_ptrs, mask=valid_m, other=0.0)
            
            # calculate inline delta for current Q tile
            delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
            
            scores_t = tl.dot(k, q.T) * scale
            scores_t = tl.where(valid_n[:, None] & valid_m[None, :], scores_t, -float('inf'))
            p_t = tl.math.exp(scores_t - l[None, :])
            
            dv += tl.dot(p_t.to(tl.bfloat16), do)
            
            dp_t = tl.dot(v, do.T)
            ds_t = tl.where(
                valid_n[:, None] & valid_m[None, :],
                p_t * (dp_t - delta[None, :]) * scale,
                0.0
            )
            
            dk += tl.dot(ds_t.to(tl.bfloat16), q)
            
        dk_ptrs = dk_ptr + offs_n_safe[:, None] * stride_dks + offs_d[None, :] * stride_dkd
        dv_ptrs = dv_ptr + offs_n_safe[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
        
        tl.store(dk_ptrs, dk.to(dk_ptr.dtype.element_ty), mask=valid_n[:, None])
        tl.store(dv_ptrs, dv.to(dv_ptr.dtype.element_ty), mask=valid_n[:, None])
        
    else:
        # -------------------------------------------------------------
        # dQ owner region
        # exclusive ownership of one dQ tile, reducing over all KV tiles
        # -------------------------------------------------------------
        q_tile = pid_tile - num_kv_tiles
        offs_m = q_tile * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, d)
        
        valid_m = offs_m < S
        offs_m_safe = tl.where(valid_m, offs_m, 0)
        
        q_ptrs = q_ptr + offs_m_safe[:, None] * stride_qs + offs_d[None, :] * stride_qd
        do_ptrs = do_ptr + offs_m_safe[:, None] * stride_dos + offs_d[None, :] * stride_dod
        o_ptrs = o_ptr + offs_m_safe[:, None] * stride_os + offs_d[None, :] * stride_od
        l_ptrs = l_ptr + offs_m_safe * stride_ls
        
        q = tl.load(q_ptrs, mask=valid_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=valid_m[:, None], other=0.0)
        o = tl.load(o_ptrs, mask=valid_m[:, None], other=0.0)
        l = tl.load(l_ptrs, mask=valid_m, other=0.0)
        
        # calculate inline delta for owned Q tile
        delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
        
        dq = tl.zeros((BLOCK_M, d), dtype=tl.float32)
        
        for kv_tile in range(0, num_kv_tiles):
            offs_n = kv_tile * BLOCK_N + tl.arange(0, BLOCK_N)
            valid_n = offs_n < S
            offs_n_safe = tl.where(valid_n, offs_n, 0)
            
            k_ptrs = k_ptr + offs_n_safe[:, None] * stride_ks + offs_d[None, :] * stride_kd
            v_ptrs = v_ptr + offs_n_safe[:, None] * stride_vs + offs_d[None, :] * stride_vd
            
            k = tl.load(k_ptrs, mask=valid_n[:, None], other=0.0)
            v = tl.load(v_ptrs, mask=valid_n[:, None], other=0.0)
            
            scores = tl.dot(q, k.T) * scale
            scores = tl.where(valid_m[:, None] & valid_n[None, :], scores, -float('inf'))
            p = tl.math.exp(scores - l[:, None])
            
            dp = tl.dot(do, v.T)
            ds = tl.where(
                valid_m[:, None] & valid_n[None, :],
                p * (dp - delta[:, None]) * scale,
                0.0
            )
            
            dq += tl.dot(ds.to(tl.bfloat16), k)
            
        dq_ptrs = dq_ptr + offs_m_safe[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
        tl.store(dq_ptrs, dq.to(dq_ptr.dtype.element_ty), mask=valid_m[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes backward gradients for multi-head attention without scaling dot product output.
    Uses destination passing logic, mapping to Blackwell-native standard Triton logic via exclusive output ownership.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 128
    
    num_kv_tiles = triton.cdiv(S, BLOCK_N)
    num_q_tiles = triton.cdiv(S, BLOCK_M)
    
    # 2D Grid: [split ownership regions per head sequence, batch * heads]
    grid = (num_kv_tiles + num_q_tiles, B * H)
    
    scale = 1.0 / (BLOCK_D ** 0.5)
    
    _bwd_kernel[grid](
        Q, K, V, O, dO, L,
        dQ, dK, dV,
        S,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        d=BLOCK_D,
        H=H,
        num_warps=8,
        num_stages=3
    )