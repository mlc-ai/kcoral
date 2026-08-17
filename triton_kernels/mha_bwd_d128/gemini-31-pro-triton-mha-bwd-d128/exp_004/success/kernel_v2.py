import torch
import triton
import triton.language as tl

@triton.jit
def _bwd_dq(
    Q, K, V, O, DO, L, dQ,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    S, H, scale, d_val,
    BLOCK_S_Q: tl.constexpr,
    BLOCK_S_KV: tl.constexpr,
    d: tl.constexpr,
    EVEN_S: tl.constexpr
):
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    offs_q = pid_q * BLOCK_S_Q
    zero_q = offs_q * 0  
    
    Q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
    DO_ptr = DO + pid_b * stride_dob + pid_h * stride_doh
    O_ptr = O + pid_b * stride_ob + pid_h * stride_oh
    L_ptr = L + pid_b * stride_lb + pid_h * stride_lh
    
    desc_Q = tl.make_tensor_descriptor(
        Q_ptr, shape=[S, d_val], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_S_Q, d], padding_option="zero"
    )
    desc_DO = tl.make_tensor_descriptor(
        DO_ptr, shape=[S, d_val], strides=[stride_dos, stride_dod],
        block_shape=[BLOCK_S_Q, d], padding_option="zero"
    )
    desc_O = tl.make_tensor_descriptor(
        O_ptr, shape=[S, d_val], strides=[stride_os, stride_od],
        block_shape=[BLOCK_S_Q, d], padding_option="zero"
    )
    
    q = desc_Q.load([offs_q, zero_q])
    do = desc_DO.load([offs_q, zero_q])
    o = desc_O.load([offs_q, zero_q])
    
    # Compute on-chip Delta (row-wise dot product of O and dO) 
    # Multiply is kept in bf16 and widened in accumulation to heavily suppress register pressure
    delta = tl.sum((o * do).to(tl.float32), axis=1)
    
    dq_acc = tl.zeros([BLOCK_S_Q, d], dtype=tl.float32)
    
    if not EVEN_S:
        offs_q_arr = offs_q + tl.arange(0, BLOCK_S_Q)
        mask_q = offs_q_arr < S
        l = tl.load(L_ptr + offs_q_arr * stride_ls, mask=mask_q, other=0.0)
    else:
        offs_q_arr = offs_q + tl.arange(0, BLOCK_S_Q)
        l = tl.load(L_ptr + offs_q_arr * stride_ls)
    
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
    
    num_kv_blocks = tl.cdiv(S, BLOCK_S_KV)
    for kv_idx in range(num_kv_blocks):
        offs_k = kv_idx * BLOCK_S_KV
        zero_k = offs_k * 0
        
        # Top-of-loop TMA asynchronous loads for smooth WGMMA pipeline
        k = desc_K.load([offs_k, zero_k])
        v = desc_V.load([offs_k, zero_k])
        
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
        
        # Execute (dP - Delta) and scale using bf16 downstream casting, freeing dp
        dp_scaled = ((dp - delta[:, None]) * scale).to(k.dtype)
        ds = p_bf16 * dp_scaled
        
        dq_acc = tl.dot(ds, k, dq_acc)
        
    dQ_ptr = dQ + pid_b * stride_dqb + pid_h * stride_dqh
    desc_dQ = tl.make_tensor_descriptor(
        dQ_ptr, shape=[S, d_val], strides=[stride_dqs, stride_dqd],
        block_shape=[BLOCK_S_Q, d]
    )
    desc_dQ.store([offs_q, zero_q], dq_acc.to(q.dtype))


@triton.jit
def _bwd_dkdv(
    Q, K, V, O, DO, L, dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    S, H, scale, d_val,
    BLOCK_S_Q: tl.constexpr,
    BLOCK_S_KV: tl.constexpr,
    d: tl.constexpr,
    EVEN_S: tl.constexpr
):
    pid_kv = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
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
    
    k = desc_K.load([offs_k, zero_k])
    v = desc_V.load([offs_k, zero_k])
    
    dk_acc = tl.zeros([BLOCK_S_KV, d], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_S_KV, d], dtype=tl.float32)
    
    Q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
    DO_ptr = DO + pid_b * stride_dob + pid_h * stride_doh
    O_ptr = O + pid_b * stride_ob + pid_h * stride_oh
    L_ptr = L + pid_b * stride_lb + pid_h * stride_lh
    
    desc_Q = tl.make_tensor_descriptor(
        Q_ptr, shape=[S, d_val], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_S_Q, d], padding_option="zero"
    )
    desc_DO = tl.make_tensor_descriptor(
        DO_ptr, shape=[S, d_val], strides=[stride_dos, stride_dod],
        block_shape=[BLOCK_S_Q, d], padding_option="zero"
    )
    desc_O = tl.make_tensor_descriptor(
        O_ptr, shape=[S, d_val], strides=[stride_os, stride_od],
        block_shape=[BLOCK_S_Q, d], padding_option="zero"
    )
    
    if not EVEN_S:
        offs_k_arr = offs_k + tl.arange(0, BLOCK_S_KV)
        mask_k = offs_k_arr < S
        
    num_q_blocks = tl.cdiv(S, BLOCK_S_Q)
    for q_idx in range(num_q_blocks):
        offs_q = q_idx * BLOCK_S_Q
        zero_q = offs_q * 0
        
        # Top-of-loop TMA asynchronous loads ensures they are scheduled cleanly
        q = desc_Q.load([offs_q, zero_q])
        do = desc_DO.load([offs_q, zero_q])
        o = desc_O.load([offs_q, zero_q])
        
        if not EVEN_S:
            offs_q_arr = offs_q + tl.arange(0, BLOCK_S_Q)
            mask_q = offs_q_arr < S
            l = tl.load(L_ptr + offs_q_arr * stride_ls, mask=mask_q, other=0.0)
        else:
            offs_q_arr = offs_q + tl.arange(0, BLOCK_S_Q)
            l = tl.load(L_ptr + offs_q_arr * stride_ls)
            
        delta = tl.sum((o * do).to(tl.float32), axis=1)
        
        qk = tl.zeros([BLOCK_S_Q, BLOCK_S_KV], dtype=tl.float32)
        qk = tl.dot(q, tl.trans(k), qk)
        
        p = tl.exp(qk * scale - l[:, None])
        if not EVEN_S:
            p = tl.where(mask_q[:, None] & mask_k[None, :], p, 0.0)
            
        p_bf16 = p.to(k.dtype)
        
        dp = tl.zeros([BLOCK_S_Q, BLOCK_S_KV], dtype=tl.float32)
        dp = tl.dot(do, tl.trans(v), dp)
        
        dp_scaled = ((dp - delta[:, None]) * scale).to(k.dtype)
        ds = p_bf16 * dp_scaled
        
        dv_acc = tl.dot(tl.trans(p_bf16), do, dv_acc)
        dk_acc = tl.dot(tl.trans(ds), q, dk_acc)
        
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
    desc_dK.store([offs_k, zero_k], dk_acc.to(k.dtype))
    desc_dV.store([offs_k, zero_k], dv_acc.to(k.dtype))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
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
        
        # Enforcing identical tile shapes keeps register demands <192 across execution stages
        BLOCK_S_Q = 128
        BLOCK_S_KV = 64
        
        EVEN_S = (S % 128 == 0)
        
        grid_dq = (triton.cdiv(S, BLOCK_S_Q), B * H)
        _bwd_dq[grid_dq](
            Q, K, V, O, dO, L, dQ,
            stride_qb, stride_qh, stride_qs, stride_qd,
            stride_kb, stride_kh, stride_ks, stride_kd,
            stride_vb, stride_vh, stride_vs, stride_vd,
            stride_ob, stride_oh, stride_os, stride_od,
            stride_dob, stride_doh, stride_dos, stride_dod,
            stride_lb, stride_lh, stride_ls,
            stride_dqb, stride_dqh, stride_dqs, stride_dqd,
            S, H, scale, d_val,
            BLOCK_S_Q=BLOCK_S_Q, BLOCK_S_KV=BLOCK_S_KV, d=d_val,
            EVEN_S=EVEN_S,
            num_warps=8, num_stages=4
        )
        
        grid_dkdv = (triton.cdiv(S, BLOCK_S_KV), B * H)
        _bwd_dkdv[grid_dkdv](
            Q, K, V, O, dO, L, dK, dV,
            stride_qb, stride_qh, stride_qs, stride_qd,
            stride_kb, stride_kh, stride_ks, stride_kd,
            stride_vb, stride_vh, stride_vs, stride_vd,
            stride_ob, stride_oh, stride_os, stride_od,
            stride_dob, stride_doh, stride_dos, stride_dod,
            stride_lb, stride_lh, stride_ls,
            stride_dkb, stride_dkh, stride_dks, stride_dkd,
            stride_dvb, stride_dvh, stride_dvs, stride_dvd,
            S, H, scale, d_val,
            BLOCK_S_Q=BLOCK_S_Q, BLOCK_S_KV=BLOCK_S_KV, d=d_val,
            EVEN_S=EVEN_S,
            num_warps=8, num_stages=2  # Limited to 2 due to Q, DO, O (128x128) demands against 228KiB SMEM budget
        )
        
        return dQ, dK, dV