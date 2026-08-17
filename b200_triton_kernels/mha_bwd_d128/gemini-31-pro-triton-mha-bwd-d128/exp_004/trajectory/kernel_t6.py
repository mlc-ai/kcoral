import torch
import triton
import triton.language as tl

@triton.jit
def _precompute_delta(
    O, DO, Delta,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_db, stride_dh, stride_ds,
    S, H, BLOCK_S: tl.constexpr, d: tl.constexpr
):
    pid_s = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < S
    
    offs_d = tl.arange(0, d)
    
    # Materialize pointers 
    O_ptr = O + pid_b * stride_ob + pid_h * stride_oh + offs_s[:, None] * stride_os + offs_d[None, :] * stride_od
    DO_ptr = DO + pid_b * stride_dob + pid_h * stride_doh + offs_s[:, None] * stride_dos + offs_d[None, :] * stride_dod
    
    o = tl.load(O_ptr, mask=mask_s[:, None], other=0.0)
    do = tl.load(DO_ptr, mask=mask_s[:, None], other=0.0)
    
    delta = tl.sum((o * do).to(tl.float32), axis=1)
    
    Delta_ptr = Delta + pid_b * stride_db + pid_h * stride_dh + offs_s * stride_ds
    tl.store(Delta_ptr, delta, mask=mask_s)


@triton.jit
def _bwd_dq(
    Q, K, V, DO, L, Delta, dQ,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_db, stride_dh, stride_ds,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    B, H, S, scale, d_val,
    BLOCK_S_Q: tl.constexpr,
    BLOCK_S_KV: tl.constexpr,
    d: tl.constexpr,
    EVEN_S: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr
):
    start_pid = tl.program_id(0)
    num_pid_q = tl.cdiv(S, BLOCK_S_Q)
    num_pid_bh = B * H
    num_tiles = num_pid_q * num_pid_bh
    num_kv_blocks = tl.cdiv(S, BLOCK_S_KV)

    # Warp specialized persistent SM90 loop structure 
    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        flatten=False,
        warp_specialize=WARP_SPECIALIZE,
    ):
        pid_q = tile_id % num_pid_q
        pid_bh = tile_id // num_pid_q
        
        pid_b = pid_bh // H
        pid_h = pid_bh % H
        
        offs_q = pid_q * BLOCK_S_Q
        zero_q = offs_q * 0  
        
        Q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
        DO_ptr = DO + pid_b * stride_dob + pid_h * stride_doh
        L_ptr = L + pid_b * stride_lb + pid_h * stride_lh
        Delta_ptr = Delta + pid_b * stride_db + pid_h * stride_dh
        
        # Descriptors logically created outer-loop scoped allows flawless pipelining 
        desc_Q = tl.make_tensor_descriptor(
            Q_ptr, shape=[S, d_val], strides=[stride_qs, stride_qd],
            block_shape=[BLOCK_S_Q, d], padding_option="zero"
        )
        desc_DO = tl.make_tensor_descriptor(
            DO_ptr, shape=[S, d_val], strides=[stride_dos, stride_dod],
            block_shape=[BLOCK_S_Q, d], padding_option="zero"
        )
        
        q = tl.load(desc_Q, [offs_q, zero_q])
        do = tl.load(desc_DO, [offs_q, zero_q])
        
        dq_acc = tl.zeros([BLOCK_S_Q, d], dtype=tl.float32)
        
        if not EVEN_S:
            offs_q_arr = offs_q + tl.arange(0, BLOCK_S_Q)
            mask_q = offs_q_arr < S
            l = tl.load(L_ptr + offs_q_arr * stride_ls, mask=mask_q, other=0.0)
            delta = tl.load(Delta_ptr + offs_q_arr * stride_ds, mask=mask_q, other=0.0)
        else:
            offs_q_arr = offs_q + tl.arange(0, BLOCK_S_Q)
            l = tl.load(L_ptr + offs_q_arr * stride_ls)
            delta = tl.load(Delta_ptr + offs_q_arr * stride_ds)
        
        K_ptr = K + pid_b * stride_kb + pid_h * stride_kh
        V_ptr = V + pid_b * stride_vb + pid_h * stride_vh
        
        desc_K = tl.make_tensor_descriptor(
            K_ptr, shape=[S, d_val], strides=[stride_ks, stride_kd],
            block_shape=[BLOCK_S_KV, d], padding_option="zero"
        )
        desc_V = tl.make_tensor_descriptor(
            V_ptr, shape=[S, d_val], strides=[stride_vs, stride_vd],
            block_shape=[BLOCK_S_KV, d], padding_option="zero"
        )
        
        for kv_idx in range(num_kv_blocks):
            offs_k = kv_idx * BLOCK_S_KV
            zero_k = offs_k * 0
            
            k = tl.load(desc_K, [offs_k, zero_k])
            v = tl.load(desc_V, [offs_k, zero_k])
            
            qk = tl.zeros([BLOCK_S_Q, BLOCK_S_KV], dtype=tl.float32)
            qk = tl.dot(q, tl.trans(k), qk)
            
            p = tl.exp(qk * scale - l[:, None])
            if not EVEN_S:
                offs_k_arr = offs_k + tl.arange(0, BLOCK_S_KV)
                mask_k = offs_k_arr < S
                p = tl.where(mask_q[:, None] & mask_k[None, :], p, 0.0)
                
            p_bf16 = p.to(k.dtype)
            
            dp = tl.zeros([BLOCK_S_Q, BLOCK_S_KV], dtype=tl.float32)
            dp = tl.dot(do, tl.trans(v), dp)
            
            dp_scaled = ((dp - delta[:, None]) * scale).to(k.dtype)
            ds = p_bf16 * dp_scaled
            
            dq_acc = tl.dot(ds, k, dq_acc)
            
        dQ_ptr = dQ + pid_b * stride_dqb + pid_h * stride_dqh
        desc_dQ = tl.make_tensor_descriptor(
            dQ_ptr, shape=[S, d_val], strides=[stride_dqs, stride_dqd],
            block_shape=[BLOCK_S_Q, d]
        )
        tl.store(desc_dQ, [offs_q, zero_q], dq_acc.to(q.dtype))


@triton.jit
def _bwd_dkdv(
    Q, K, V, DO, L, Delta, dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_db, stride_dh, stride_ds,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    B, H, S, scale, d_val,
    BLOCK_S_Q: tl.constexpr,
    BLOCK_S_KV: tl.constexpr,
    d: tl.constexpr,
    EVEN_S: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr
):
    start_pid = tl.program_id(0)
    num_pid_kv = tl.cdiv(S, BLOCK_S_KV)
    num_pid_bh = B * H
    num_tiles = num_pid_kv * num_pid_bh
    num_q_blocks = tl.cdiv(S, BLOCK_S_Q)

    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        flatten=False,
        warp_specialize=WARP_SPECIALIZE,
    ):
        pid_kv = tile_id % num_pid_kv
        pid_bh = tile_id // num_pid_kv
        
        pid_b = pid_bh // H
        pid_h = pid_bh % H
        
        offs_k = pid_kv * BLOCK_S_KV
        zero_k = offs_k * 0
        
        K_ptr = K + pid_b * stride_kb + pid_h * stride_kh
        V_ptr = V + pid_b * stride_vb + pid_h * stride_vh
        
        desc_K = tl.make_tensor_descriptor(
            K_ptr, shape=[S, d_val], strides=[stride_ks, stride_kd],
            block_shape=[BLOCK_S_KV, d], padding_option="zero"
        )
        desc_V = tl.make_tensor_descriptor(
            V_ptr, shape=[S, d_val], strides=[stride_vs, stride_vd],
            block_shape=[BLOCK_S_KV, d], padding_option="zero"
        )
        
        k = tl.load(desc_K, [offs_k, zero_k])
        v = tl.load(desc_V, [offs_k, zero_k])
        
        dk_acc = tl.zeros([BLOCK_S_KV, d], dtype=tl.float32)
        dv_acc = tl.zeros([BLOCK_S_KV, d], dtype=tl.float32)
        
        Q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
        DO_ptr = DO + pid_b * stride_dob + pid_h * stride_doh
        L_ptr = L + pid_b * stride_lb + pid_h * stride_lh
        Delta_ptr = Delta + pid_b * stride_db + pid_h * stride_dh
        
        desc_Q = tl.make_tensor_descriptor(
            Q_ptr, shape=[S, d_val], strides=[stride_qs, stride_qd],
            block_shape=[BLOCK_S_Q, d], padding_option="zero"
        )
        desc_DO = tl.make_tensor_descriptor(
            DO_ptr, shape=[S, d_val], strides=[stride_dos, stride_dod],
            block_shape=[BLOCK_S_Q, d], padding_option="zero"
        )
        
        if not EVEN_S:
            offs_k_arr = offs_k + tl.arange(0, BLOCK_S_KV)
            mask_k = offs_k_arr < S
            
        for q_idx in range(num_q_blocks):
            offs_q = q_idx * BLOCK_S_Q
            zero_q = offs_q * 0
            
            q = tl.load(desc_Q, [offs_q, zero_q])
            do = tl.load(desc_DO, [offs_q, zero_q])
            
            if not EVEN_S:
                offs_q_arr = offs_q + tl.arange(0, BLOCK_S_Q)
                mask_q = offs_q_arr < S
                l = tl.load(L_ptr + offs_q_arr * stride_ls, mask=mask_q, other=0.0)
                delta = tl.load(Delta_ptr + offs_q_arr * stride_ds, mask=mask_q, other=0.0)
            else:
                offs_q_arr = offs_q + tl.arange(0, BLOCK_S_Q)
                l = tl.load(L_ptr + offs_q_arr * stride_ls)
                delta = tl.load(Delta_ptr + offs_q_arr * stride_ds)
                
            # WGMMA transposed scaling natively evaluates layout-aligned results 
            qk_T = tl.zeros([BLOCK_S_KV, BLOCK_S_Q], dtype=tl.float32)
            qk_T = tl.dot(k, tl.trans(q), qk_T)
            
            p_T = tl.exp(qk_T * scale - l[None, :])
            if not EVEN_S:
                p_T = tl.where(mask_k[:, None] & mask_q[None, :], p_T, 0.0)
                
            p_T_bf16 = p_T.to(k.dtype)
            
            dp_T = tl.zeros([BLOCK_S_KV, BLOCK_S_Q], dtype=tl.float32)
            dp_T = tl.dot(v, tl.trans(do), dp_T)
            
            ds_T = (p_T * (dp_T - delta[None, :]) * scale).to(k.dtype)
            
            dv_acc = tl.dot(p_T_bf16, do, dv_acc)
            dk_acc = tl.dot(ds_T, q, dk_acc)
            
        dK_ptr = dK + pid_b * stride_dkb + pid_h * stride_dkh
        dV_ptr = dV + pid_b * stride_dvb + pid_h * stride_dvh
        
        desc_dK = tl.make_tensor_descriptor(
            dK_ptr, shape=[S, d_val], strides=[stride_dks, stride_dkd],
            block_shape=[BLOCK_S_KV, d]
        )
        desc_dV = tl.make_tensor_descriptor(
            dV_ptr, shape=[S, d_val], strides=[stride_dvs, stride_dvd],
            block_shape=[BLOCK_S_KV, d]
        )
        tl.store(desc_dK, [offs_k, zero_k], dk_acc.to(k.dtype))
        tl.store(desc_dV, [offs_k, zero_k], dv_acc.to(k.dtype))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes the multi-head attention backward pass efficiently using a full Hopper TMA Warp Specialized persistence engine.
    """
    with torch.cuda.device(Q.device):
        def alloc_fn(size: int, alignment: int, stream):
            return torch.empty(size, device="cuda", dtype=torch.int8)
        triton.set_allocator(alloc_fn)
        
        B, H, S, d_val = Q.shape
        scale = 1.0 / (d_val ** 0.5)
        
        stride_qb, stride_qh, stride_qs, stride_qd = Q.stride()
        stride_kb, stride_kh, stride_ks, stride_kd = K.stride()
        stride_vb, stride_vh, stride_vs, stride_vd = V.stride()
        stride_ob, stride_oh, stride_os, stride_od = O.stride()
        stride_dob, stride_doh, stride_dos, stride_dod = dO.stride()
        stride_lb, stride_lh, stride_ls = L.stride()
        stride_dqb, stride_dqh, stride_dqs, stride_dqd = dQ.stride()
        stride_dkb, stride_dkh, stride_dks, stride_dkd = dK.stride()
        stride_dvb, stride_dvh, stride_dvs, stride_dvd = dV.stride()
        
        # Precompute Delta = sum(O * dO, axis=-1) in a high-bandwidth prepass
        Delta = torch.empty((B, H, S), device=Q.device, dtype=torch.float32)
        stride_db, stride_dh, stride_ds = Delta.stride()
        
        grid_delta = (triton.cdiv(S, 128), B * H)
        _precompute_delta[grid_delta](
            O, dO, Delta,
            stride_ob, stride_oh, stride_os, stride_od,
            stride_dob, stride_doh, stride_dos, stride_dod,
            stride_db, stride_dh, stride_ds,
            S, H, 128, d_val,
            num_warps=4
        )
        
        # Asymmetric geometry carefully preserves WGMMA occupancy preventing 255 Reg-limit spills
        BLOCK_S_Q_DQ = 128
        BLOCK_S_KV_DQ = 64
        
        BLOCK_S_KV_DKDV = 64
        BLOCK_S_Q_DKDV = 128
        
        EVEN_S = (S % 128 == 0)
        
        NUM_SMS = 132
        WARP_SPECIALIZE = True
        
        num_tiles_dq = triton.cdiv(S, BLOCK_S_Q_DQ) * B * H
        grid_dq = (min(NUM_SMS, num_tiles_dq),)
        
        _bwd_dq[grid_dq](
            Q, K, V, dO, L, Delta, dQ,
            stride_qb, stride_qh, stride_qs, stride_qd,
            stride_kb, stride_kh, stride_ks, stride_kd,
            stride_vb, stride_vh, stride_vs, stride_vd,
            stride_dob, stride_doh, stride_dos, stride_dod,
            stride_lb, stride_lh, stride_ls,
            stride_db, stride_dh, stride_ds,
            stride_dqb, stride_dqh, stride_dqs, stride_dqd,
            B, H, S, scale, d_val,
            BLOCK_S_Q=BLOCK_S_Q_DQ, BLOCK_S_KV=BLOCK_S_KV_DQ, d=d_val,
            EVEN_S=EVEN_S,
            NUM_SMS=NUM_SMS,
            WARP_SPECIALIZE=WARP_SPECIALIZE,
            num_warps=8, num_stages=3
        )
        
        num_tiles_dkdv = triton.cdiv(S, BLOCK_S_KV_DKDV) * B * H
        grid_dkdv = (min(NUM_SMS, num_tiles_dkdv),)
        
        _bwd_dkdv[grid_dkdv](
            Q, K, V, dO, L, Delta, dK, dV,
            stride_qb, stride_qh, stride_qs, stride_qd,
            stride_kb, stride_kh, stride_ks, stride_kd,
            stride_vb, stride_vh, stride_vs, stride_vd,
            stride_dob, stride_doh, stride_dos, stride_dod,
            stride_lb, stride_lh, stride_ls,
            stride_db, stride_dh, stride_ds,
            stride_dkb, stride_dkh, stride_dks, stride_dkd,
            stride_dvb, stride_dvh, stride_dvs, stride_dvd,
            B, H, S, scale, d_val,
            BLOCK_S_Q=BLOCK_S_Q_DKDV, BLOCK_S_KV=BLOCK_S_KV_DKDV, d=d_val,
            EVEN_S=EVEN_S,
            NUM_SMS=NUM_SMS,
            WARP_SPECIALIZE=WARP_SPECIALIZE,
            num_warps=8, num_stages=3
        )
        
        return dQ, dK, dV