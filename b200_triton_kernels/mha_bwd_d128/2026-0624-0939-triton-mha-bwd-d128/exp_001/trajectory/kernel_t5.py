import torch
import triton
import triton.language as tl


@triton.jit
def _bwd_Q_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dQ_ptr,
    B, H, S_len, scale,
    q_s0, q_s1, q_s2, q_s3,
    k_s0, k_s1, k_s2, k_s3,
    v_s0, v_s1, v_s2, v_s3,
    do_s0, do_s1, do_s2, do_s3,
    o_s0, o_s1, o_s2, o_s3,
    BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr,
):
    b_h_idx = tl.program_id(1)
    q_idx_base = tl.program_id(0) * BLOCK_Q
    
    row_offs = tl.arange(0, 128)
    
    for b_h_idx in tl.static_range(B * H):
        s_offs = q_idx_base + row_offs
        mask_q = s_offs < S_len
        
        q_base_0 = tl.make_block_ptr(Q_ptr + b_h_idx * q_s0, 
            shape=[S, 128], strides=[q_s2, q_s3], 
            offsets=[q_idx_base, 0], 
            block_shape=[64, 64], 
            order=[1, 0])
        q_base_1 = tl.make_block_ptr(Q_ptr + b_h_idx * q_s0, 
            shape=[S, 128], strides=[q_s2, q_s3], 
            offsets=[q_idx_base, 64], 
            block_shape=[64, 64], 
            order=[1, 0])
        
        do_base_0 = tl.make_block_ptr(dO_ptr + b_h_idx * do_s0, 
            shape=[S, 128], strides=[do_s2, do_s3], 
            offsets=[q_idx_base, 0], 
            block_shape=[64, 64], 
            order=[1, 0])
        do_base_1 = tl.make_block_ptr(dO_ptr + b_h_idx * do_s0, 
            shape=[S, 128], strides=[do_s2, do_s3], 
            offsets=[q_idx_base, 64], 
            block_shape=[64, 64], 
            order=[1, 0])
            
        o_base_0 = tl.make_block_ptr(O_ptr + b_h_idx * o_s0, 
            shape=[S, 128], strides=[o_s2, o_s3], 
            offsets=[q_idx_base, 0], 
            block_shape=[64, 64], 
            order=[1, 0])
        o_base_1 = tl.make_block_ptr(O_ptr + b_h_idx * o_s0, 
            shape=[S, 128], strides=[o_s2, o_s3], 
            offsets=[q_idx_base, 64], 
            block_shape=[64, 64], 
            order=[1, 0])
        
        Q_0 = tl.load(q_base_0, boundary_check=(0, 1))
        Q_1 = tl.load(q_base_1, boundary_check=(0, 1))
        dO_0 = tl.load(do_base_0, boundary_check=(0, 1))
        dO_1 = tl.load(do_base_1, boundary_check=(0, 1))
        O_0 = tl.load(o_base_0, boundary_check=(0, 1))
        O_1 = tl.load(o_base_1, boundary_check=(0, 1))
        
        Q_0 = tl.cast(Q_0, tl.float32)
        Q_1 = tl.cast(Q_1, tl.float32)
        dO_0 = tl.cast(dO_0, tl.float32)
        dO_1 = tl.cast(dO_1, tl.float32)
        O_0 = tl.cast(O_0, tl.float32)
        O_1 = tl.cast(O_1, tl.float32)
        
        dP_0_contrib = dO_0 * O_0
        dP_1_contrib = dO_1 * O_1
        
        D_val = tl.zeros((128,), tl.float32)
        D_val += (dO_0 * O_0 + dP_0_contrib).sum(axis=1)
        D_val += (dO_1 * O_1 + dP_1_contrib).sum(axis=1)
        
        l_offs = b_h_idx * S_len + s_offs
        mask_l = s_offs < S_len
        L_val = tl.load(L_ptr + l_offs, mask=mask_l, other=0.0)
        
        acc_dQ_0 = tl.zeros((128, 64), tl.float32)
        acc_dQ_1 = tl.zeros((128, 64), tl.float32)
        
        # Unrolled iterations targeting SM coverage limits precisely
        for k_idx_base in range(0, q_idx_base, 64): 
            k_base_0 = tl.make_block_ptr(K_ptr + b_h_idx * k_s0, 
                shape=[S, 128], strides=[k_s2, k_s3], 
                offsets=[0, 0], block_shape=[64, 64], order=[1, 0])
            k_base_1 = tl.make_block_ptr(K_ptr + b_h_idx * k_s0, 
                shape=[S, 128], strides=[k_s2, k_s3], 
                offsets=[0, 64], block_shape=[64, 64], order=[1, 0])
            v_base_0 = tl.make_block_ptr(V_ptr + b_h_idx * v_s0, 
                shape=[S, 128], strides=[v_s2, v_s3], 
                offsets=[0, 0], block_shape=[64, 64], order=[1, 0])
            v_base_1 = tl.make_block_ptr(V_ptr + b_h_idx * v_s0, 
                shape=[S, 128], strides=[v_s2, v_s3], 
                offsets=[0, 64], block_shape=[64, 64], order=[1, 0])
            
            K_0 = tl.load(tl.advance(k_base_0, [k_idx_base, 0]), boundary_check=(0, 1))
            K_1 = tl.load(tl.advance(k_base_1, [k_idx_base, 0]), boundary_check=(0, 1))
            V_0 = tl.load(tl.advance(v_base_0, [k_idx_base, 0]), boundary_check=(0, 1))
            V_1 = tl.load(tl.advance(v_base_1, [k_idx_base, 0]), boundary_check=(0, 1))
            
            K_0 = tl.cast(K_0, tl.float32)
            K_1 = tl.cast(K_1, tl.float32)
            V_0 = tl.cast(V_0, tl.float32)
            V_1 = tl.cast(V_1, tl.float32)
            
            S_scores = (tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)) * scale
            P = tl.exp(S_scores - L_val[:, None])
            dP = (tl.dot(dO_0, V_0.T) + tl.dot(dO_1, V_1.T))
            dS = P * (dP - D_val[:, None]) * scale
            
            acc_dQ_0 += tl.dot(dS, K_0)
            acc_dQ_1 += tl.dot(dS, K_1)
            
        for k_idx_base in range(q_idx_base, S, 64):
            k_base_0 = tl.make_block_ptr(K_ptr + b_h_idx * k_s0, 
                shape=[S, 128], strides=[k_s2, k_s3], 
                offsets=[q_idx_base, 0], block_shape=[64, 64], order=[1, 0])
            k_base_1 = tl.make_block_ptr(K_ptr + b_h_idx * k_s0, 
                shape=[S, 128], strides=[k_s2, k_s3], 
                offsets=[q_idx_base, 64], block_shape=[64, 64], order=[1, 0])
            v_base_0 = tl.make_block_ptr(V_ptr + b_h_idx * v_s0, 
                shape=[S, 128], strides=[v_s2, v_s3], 
                offsets=[q_idx_base, 0], block_shape=[64, 64], order=[1, 0])
            v_base_1 = tl.make_block_ptr(V_ptr + b_h_idx * v_s0, 
                shape=[S, 128], strides=[v_s2, v_s3], 
                offsets=[q_idx_base, 64], block_shape=[64, 64], order=[1, 0])
            
            K_0 = tl.load(tl.advance(k_base_0, [k_idx_base - q_idx_base, 0]), boundary_check=(0, 1))
            K_1 = tl.load(tl.advance(k_base_1, [k_idx_base - q_idx_base, 0]), boundary_check=(0, 1))
            V_0 = tl.load(tl.advance(v_base_0, [k_idx_base - q_idx_base, 0]), boundary_check=(0, 1))
            V_1 = tl.load(tl.advance(v_base_1, [k_idx_base - q_idx_base, 0]), boundary_check=(0, 1))
            
            K_0 = tl.cast(K_0, tl.float32)
            K_1 = tl.cast(K_1, tl.float32)
            V_0 = tl.cast(V_0, tl.float32)
            V_1 = tl.cast(V_1, tl.float32)
            
            S_scores = (tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)) * scale
            P = tl.exp(S_scores - L_val[:, None])
            dP = (tl.dot(dO_0, V_0.T) + tl.dot(dO_1, V_1.T))
            dS = P * (dP - D_val[:, None]) * scale
            
            acc_dQ_0 += tl.dot(dS, K_0)
            acc_dQ_1 += tl.dot(dS, K_1)
            
        store_offs_0 = (b_h_idx * S_len + s_offs[:, None]) * 128 + tl.arange(0, 64)[None, :]
        tl.store(dQ_ptr + store_offs_0, acc_dQ_0.to(tl.bfloat16), mask=mask_q[:, None])
        
        store_offs_1 = (b_h_idx * S_len + s_offs[:, None]) * 128 + (64 + tl.arange(0, 64))[None, :]
        tl.store(dQ_ptr + store_offs_1, acc_dQ_1.to(tl.bfloat16), mask=mask_q[:, None])


@triton.jit
def _bwd_KV_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    B, H, S_len, scale,
    q_s0, q_s1, q_s2, q_s3,
    k_s0, k_s1, k_s2, k_s3,
    v_s0, v_s1, v_s2, v_s3,
    do_s0, do_s1, do_s2, do_s3,
    o_s0, o_s1, o_s2, o_s3,
    BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr,
):
    b_h_idx = tl.program_id(1)
    kv_idx_base = tl.program_id(0) * BLOCK_KV
    
    row_offs = tl.arange(0, 128)
    
    for b_h_idx in tl.static_range(B * H):
        s_offs = kv_idx_base + row_offs
        mask_kv = s_offs < S_len
        
        k_base_0 = tl.make_block_ptr(K_ptr + b_h_idx * k_s0, 
            shape=[S, 128], strides=[k_s2, k_s3], 
            offsets=[kv_idx_base, 0], block_shape=[64, 64], order=[1, 0])
        k_base_1 = tl.make_block_ptr(K_ptr + b_h_idx * k_s0, 
            shape=[S, 128], strides=[k_s2, k_s3], 
            offsets=[kv_idx_base, 64], block_shape=[64, 64], order=[1, 0])
        v_base_0 = tl.make_block_ptr(V_ptr + b_h_idx * v_s0, 
            shape=[S, 128], strides=[v_s2, v_s3], 
            offsets=[kv_idx_base, 0], block_shape=[64, 64], order=[1, 0])
        v_base_1 = tl.make_block_ptr(V_ptr + b_h_idx * v_s0, 
            shape=[S, 128], strides=[v_s2, v_s3], 
            offsets=[kv_idx_base, 64], block_shape=[64, 64], order=[1, 0])
        
        K_0 = tl.load(k_base_0, boundary_check=(0, 1))
        K_1 = tl.load(k_base_1, boundary_check=(0, 1))
        V_0 = tl.load(v_base_0, boundary_check=(0, 1))
        V_1 = tl.load(v_base_1, boundary_check=(0, 1))
        
        K_0 = tl.cast(K_0, tl.float32)
        K_1 = tl.cast(K_1, tl.float32)
        V_0 = tl.cast(V_0, tl.float32)
        V_1 = tl.cast(V_1, tl.float32)
        
        acc_dK_0 = tl.zeros((128, 64), tl.float32)
        acc_dK_1 = tl.zeros((128, 64), tl.float32)
        acc_dV_0 = tl.zeros((128, 64), tl.float32)
        acc_dV_1 = tl.zeros((128, 64), tl.float32)
        
        for q_idx_base in range(0, kv_idx_base, 64):
            q_base_0 = tl.make_block_ptr(Q_ptr + b_h_idx * q_s0, 
                shape=[S, 128], strides=[q_s2, q_s3], 
                offsets=[0, 0], block_shape=[64, 64], order=[1, 0])
            q_base_1 = tl.make_block_ptr(Q_ptr + b_h_idx * q_s0, 
                shape=[S, 128], strides=[q_s2, q_s3], 
                offsets=[0, 64], block_shape=[64, 64], order=[1, 0])
            do_base_0 = tl.make_block_ptr(dO_ptr + b_h_idx * do_s0, 
                shape=[S, 128], strides=[do_s2, do_s3], 
                offsets=[0, 0], block_shape=[64, 64], order=[1, 0])
            do_base_1 = tl.make_block_ptr(dO_ptr + b_h_idx * do_s0, 
                shape=[S, 128], strides=[do_s2, do_s3], 
                offsets=[0, 64], block_shape=[64, 64], order=[1, 0])
            o_base_0 = tl.make_block_ptr(O_ptr + b_h_idx * o_s0, 
                shape=[S, 128], strides=[o_s2, o_s3], 
                offsets=[0, 0], block_shape=[64, 64], order=[1, 0])
            o_base_1 = tl.make_block_ptr(O_ptr + b_h_idx * o_s0, 
                shape=[S, 128], strides=[o_s2, o_s3], 
                offsets=[0, 64], block_shape=[64, 64], order=[1, 0])
            
            Q_0 = tl.load(tl.advance(q_base_0, [q_idx_base, 0]), boundary_check=(0, 1))
            Q_1 = tl.load(tl.advance(q_base_1, [q_idx_base, 0]), boundary_check=(0, 1))
            dO_0 = tl.load(tl.advance(do_base_0, [q_idx_base, 0]), boundary_check=(0, 1))
            dO_1 = tl.load(tl.advance(do_base_1, [q_idx_base, 0]), boundary_check=(0, 1))
            O_0 = tl.load(tl.advance(o_base_0, [q_idx_base, 0]), boundary_check=(0, 1))
            O_1 = tl.load(tl.advance(o_base_1, [q_idx_base, 0]), boundary_check=(0, 1))
            
            Q_0 = tl.cast(Q_0, tl.float32)
            Q_1 = tl.cast(Q_1, tl.float32)
            dO_0 = tl.cast(dO_0, tl.float32)
            dO_1 = tl.cast(dO_1, tl.float32)
            O_0 = tl.cast(O_0, tl.float32)
            O_1 = tl.cast(O_1, tl.float32)
            
            q_offs = q_idx_base + row_offs
            mask_q = q_offs < S_len
            
            D_val = (dO_0 * O_0).sum(axis=1) + (dO_1 * O_1).sum(axis=1)
            
            l_offs = b_h_idx * S_len + q_offs
            mask_l = q_offs < S_len
            L_val = tl.load(L_ptr + l_offs, mask=mask_l, other=0.0)
            
            S_scores = (tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)) * scale
            P = tl.exp(S_scores - L_val[:, None])
            dP = (tl.dot(dO_0, V_0.T) + tl.dot(dO_1, V_1.T))
            dS = P * (dP - D_val[:, None]) * scale
            
            dS_T = dS.T
            P_T = P.T
            
            acc_dK_0 += tl.dot(dS_T, Q_0)
            acc_dK_1 += tl.dot(dS_T, Q_1)
            acc_dV_0 += tl.dot(P_T, dO_0)
            acc_dV_1 += tl.dot(P_T, dO_1)
            
        for q_idx_base in range(kv_idx_base, S, 64):
            q_base_0 = tl.make_block_ptr(Q_ptr + b_h_idx * q_s0, 
                shape=[S, 128], strides=[q_s2, q_s3], 
                offsets=[kv_idx_base, 0], block_shape=[64, 64], order=[1, 0])
            q_base_1 = tl.make_block_ptr(Q_ptr + b_h_idx * q_s0, 
                shape=[S, 128], strides=[q_s2, q_s3], 
                offsets=[kv_idx_base, 64], block_shape=[64, 64], order=[1, 0])
            do_base_0 = tl.make_block_ptr(dO_ptr + b_h_idx * do_s0, 
                shape=[S, 128], strides=[do_s2, do_s3], 
                offsets=[kv_idx_base, 0], block_shape=[64, 64], order=[1, 0])
            do_base_1 = tl.make_block_ptr(dO_ptr + b_h_idx * do_s0, 
                shape=[S, 128], strides=[do_s2, do_s3], 
                offsets=[kv_idx_base, 64], block_shape=[64, 64], order=[1, 0])
            o_base_0 = tl.make_block_ptr(O_ptr + b_h_idx * o_s0, 
                shape=[S, 128], strides=[o_s2, o_s3], 
                offsets=[kv_idx_base, 0], block_shape=[64, 64], order=[1, 0])
            o_base_1 = tl.make_block_ptr(O_ptr + b_h_idx * o_s0, 
                shape=[S, 128], strides=[o_s2, o_s3], 
                offsets=[kv_idx_base, 64], block_shape=[64, 64], order=[1, 0])
            
            Q_0 = tl.load(tl.advance(q_base_0, [q_idx_base - kv_idx_base, 0]), boundary_check=(0, 1))
            Q_1 = tl.load(tl.advance(q_base_1, [q_idx_base - kv_idx_base, 0]), boundary_check=(0, 1))
            dO_0 = tl.load(tl.advance(do_base_0, [q_idx_base - kv_idx_base, 0]), boundary_check=(0, 1))
            dO_1 = tl.load(tl.advance(do_base_1, [q_idx_base - kv_idx_base, 0]), boundary_check=(0, 1))
            O_0 = tl.load(tl.advance(o_base_0, [q_idx_base - kv_idx_base, 0]), boundary_check=(0, 1))
            O_1 = tl.load(tl.advance(o_base_1, [q_idx_base - kv_idx_base, 0]), boundary_check=(0, 1))
            
            Q_0 = tl.cast(Q_0, tl.float32)
            Q_1 = tl.cast(Q_1, tl.float32)
            dO_0 = tl.cast(dO_0, tl.float32)
            dO_1 = tl.cast(dO_1, tl.float32)
            O_0 = tl.cast(O_0, tl.float32)
            O_1 = tl.cast(O_1, tl.float32)
            
            q_offs = q_idx_base + row_offs
            mask_q = q_offs < S_len
            
            D_val = (dO_0 * O_0).sum(axis=1) + (dO_1 * O_1).sum(axis=1)
            
            l_offs = b_h_idx * S_len + q_offs
            mask_l = q_offs < S_len
            L_val = tl.load(L_ptr + l_offs, mask=mask_l, other=0.0)
            
            S_scores = (tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)) * scale
            P = tl.exp(S_scores - L_val[:, None])
            dP = (tl.dot(dO_0, V_0.T) + tl.dot(dO_1, V_1.T))
            dS = P * (dP - D_val[:, None]) * scale
            
            dS_T = dS.T
            P_T = P.T
            
            acc_dK_0 += tl.dot(dS_T, Q_0)
            acc_dK_1 += tl.dot(dS_T, Q_1)
            acc_dV_0 += tl.dot(P_T, dO_0)
            acc_dV_1 += tl.dot(P_T, dO_1)
            
        store_offs_0 = (b_h_idx * S_len + s_offs[:, None]) * 128 + tl.arange(0, 64)[None, :]
        tl.store(dK_ptr + store_offs_0, acc_dK_0.to(tl.bfloat16), mask=mask_kv[:, None])
        tl.store(dV_ptr + store_offs_0, acc_dV_0.to(tl.bfloat16), mask=mask_kv[:, None])
        
        store_offs_1 = (b_h_idx * S_len + s_offs[:, None]) * 128 + (64 + tl.arange(0, 64))[None, :]
        tl.store(dK_ptr + store_offs_1, acc_dK_1.to(tl.bfloat16), mask=mask_kv[:, None])
        tl.store(dV_ptr + store_offs_1, acc_dV_1.to(tl.bfloat16), mask=mask_kv[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Execute optimized multi-head attention backward pass."""
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    scale = 1.0 / (d ** 0.5)
    
    q_s0, q_s1, q_s2, q_s3 = Q.stride()
    k_s0, k_s1, k_s2, k_s3 = K.stride()
    v_s0, v_s1, v_s2, v_s3 = V.stride()
    do_s0, do_s1, do_s2, do_s3 = dO.stride()
    o_s0, o_s1, o_s2, o_s3 = O.stride()
    
    grid = (triton.cdiv(S, 64), B * H)
    
    _bwd_Q_kernel[grid](
        Q, K, V, dO, L, dQ,
        B, H, S, scale,
        q_s0, q_s1, q_s2, q_s3,
        k_s0, k_s1, k_s2, k_s3,
        v_s0, v_s1, v_s2, v_s3,
        do_s0, do_s1, do_s2, do_s3,
        o_s0, o_s1, o_s2, o_s3,
        BLOCK_Q=64, BLOCK_KV=64,
        num_warps=8, num_stages=3
    )
    
    _bwd_KV_kernel[grid](
        Q, K, V, dO, L, dK, dV,
        B, H, S, scale,
        q_s0, q_s1, q_s2, q_s3,
        k_s0, k_s1, k_s2, k_s3,
        v_s0, v_s1, v_s2, v_s3,
        do_s0, do_s1, do_s2, do_s3,
        o_s0, o_s1, o_s2, o_s3,
        BLOCK_Q=64, BLOCK_KV=64,
        num_warps=8, num_stages=3
    )