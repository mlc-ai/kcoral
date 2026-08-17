import torch
import triton
import triton.language as tl


@triton.jit
def dKdV_kernel(Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
                S, scale,
                stride_b_outer: tl.constexpr, stride_s_outer: tl.constexpr,
                stride_b_inner: tl.constexpr, stride_s_inner: tl.constexpr):
    j = tl.program_id(0)
    bh = tl.program_id(1)
    off_j = j * 16
    
    k_tile = tl.load(K_ptr + bh * S * 128 + off_j * 128, shape=(16, 128), mask=(off_j * 128 < S * 128), other=0.0)
    v_tile = tl.load(V_ptr + bh * S * 128 + off_j * 128, shape=(16, 128), mask=(off_j * 128 < S * 128), other=0.0)
    
    acc_dk = tl.zeros((16, 128), tl.float32)
    acc_dv = tl.zeros((16, 128), tl.float32)
    
    num_blocks = triton.cdiv(S, 16)
    
    for i in range(num_blocks):
        off_i = i * 16
        q_tile = tl.load(Q_ptr + bh * S * 128 + off_i * 128, shape=(16, 128), mask=(off_i * 128 < S * 128), other=0.0)
        do_tile = tl.load(dO_ptr + bh * S * 128 + off_i * 128, shape=(16, 128), mask=(off_i * 128 < S * 128), other=0.0)
        o_tile = tl.load(O_ptr + bh * S * 128 + off_i * 128, shape=(16, 128), mask=(off_i * 128 < S * 128), other=0.0)
        
        d_val = 0.0
        for k_idx in range(0, 128, 16):
            do_tmp = do_tile[:, k_idx:idx+16]
            o_tmp = o_tile[:, k_idx:idx+16]
            d_val += tl.sum(do_tmp * o_tmp)
        
        l_val = tl.load(L_ptr + bh * S + off_i, mask=off_i < S, other=float('-inf'))
        
        for k_idx in range(0, 128, 16):
            q_tmp = q_tile[:, k_idx:idx+16]
            do_tmp = do_tile[:, k_idx:idx+16]
            o_tmp = o_tile[:, k_idx:idx+16]
            
            acc_s = tl.zeros((16, 16), tl.float32)
            for j in range(num_blocks):
                off_j = j * 16
                k_tmp = tl.load(K_ptr + bh * S * 128 + off_j * 128 + k_idx, shape=(16, 16), mask=(off_j * 128 + k_idx < S * 128), other=0.0)
                acc_s = tl.dot(q_tmp, k_tmp.T, acc_s)
            
            s_tmp = acc_s * scale
            
            p_tmp = tl.exp(s_tmp - l_val)
            
            acc_dp = tl.zeros((16, 16), tl.float32)
            for j in range(num_blocks):
                off_j = j * 16
                v_tmp = tl.load(V_ptr + bh * S * 128 + off_j * 128 + k_idx, shape=(16, 16), mask=(off_j * 128 + k_idx < S * 128), other=0.0)
                acc_dp = tl.dot(do_tmp, v_tmp.T, acc_dp)
            
            ds_tmp = p_tmp * (acc_dp - d_val) * scale
            
            for j in range(num_blocks):
                off_j = j * 16
                do_tmp2 = tl.load(dO_ptr + bh * S * 128 + off_i * 128 + off_j * 16, shape=(16, 16), mask=(off_i * 128 + off_j * 16 < S * 128), other=0.0)
                p_t_tmp = p_tmp[None, :, k_idx:idx+16].T
                acc_dv = tl.zeros((16, 16), tl.float32)
                acc_dv = tl.dot(p_t_tmp, do_tmp2, acc_dv)
                acc_dv = tl.store(dV_ptr + bh * S * 128 + off_j * 128 + k_idx, acc_dv, mask=(off_j * 128 + k_idx < S * 128))
            
            for j in range(num_blocks):
                off_j = j * 16
                q_tmp2 = tl.load(Q_ptr + bh * S * 128 + off_i * 128 + off_j * 16, shape=(16, 16), mask=(off_i * 128 + off_j * 16 < S * 128), other=0.0)
                ds_t_tmp = ds_tmp[None, :, k_idx:idx+16].T
                acc_dk = tl.zeros((16, 16), tl.float32)
                acc_dk = tl.dot(ds_t_tmp, q_tmp2, acc_dk)
                acc_dk = tl.store(dK_ptr + bh * S * 128 + off_j * 128 + k_idx, acc_dk, mask=(off_j * 128 + k_idx < S * 128))


@triton.jit
def dQ_kernel(Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
              S, scale,
              stride_b_outer: tl.constexpr, stride_s_outer: tl.constexpr,
              stride_b_inner: tl.constexpr, stride_s_inner: tl.constexpr):
    i = tl.program_id(0)
    bh = tl.program_id(1)
    off_i = i * 16
    
    q_tile = tl.load(Q_ptr + bh * S * 128 + off_i * 128, shape=(16, 128), mask=(off_i * 128 < S * 128), other=0.0)
    do_tile = tl.load(dO_ptr + bh * S * 128 + off_i * 128, shape=(16, 128), mask=(off_i * 128 < S * 128), other=0.0)
    o_tile = tl.load(O_ptr + bh * S * 128 + off_i * 128, shape=(16, 128), mask=(off_i * 128 < S * 128), other=0.0)
    
    d_val = 0.0
    for k_idx in range(0, 128, 16):
        do_tmp = do_tile[:, k_idx:idx+16]
        o_tmp = o_tile[:, k_idx:idx+16]
        d_val += tl.sum(do_tmp * o_tmp)
    
    l_val = tl.load(L_ptr + bh * S + off_i, mask=off_i < S, other=float('-inf'))
    
    acc_dq = tl.zeros((16, 128), tl.float32)
    
    num_blocks = triton.cdiv(S, 16)
    
    for j in range(num_blocks):
        off_j = j * 16
        k_tile = tl.load(K_ptr + bh * S * 128 + off_j * 128, shape=(16, 128), mask=(off_j * 128 < S * 128), other=0.0)
        v_tile = tl.load(V_ptr + bh * S * 128 + off_j * 128, shape=(16, 128), mask=(off_j * 128 < S * 128), other=0.0)
        
        for k_idx in range(0, 128, 16):
            q_tmp = q_tile[:, k_idx:idx+16]
            do_tmp = do_tile[:, k_idx:idx+16]
            
            acc_s = tl.zeros((16, 16), tl.float32)
            for k in range(num_blocks):
                off_k = k * 16
                k_tmp = tl.load(K_ptr + bh * S * 128 + off_k * 128 + k_idx, shape=(16, 16), mask=(off_k * 128 + k_idx < S * 128), other=0.0)
                acc_s = tl.dot(q_tmp, k_tmp.T, acc_s)
            
            s_tmp = acc_s * scale
            
            p_tmp = tl.exp(s_tmp - l_val)
            
            acc_dp = tl.zeros((16, 16), tl.float32)
            for k in range(num_blocks):
                off_k = k * 16
                v_tmp = tl.load(V_ptr + bh * S * 128 + off_k * 128 + k_idx, shape=(16, 16), mask=(off_k * 128 + k_idx < S * 128), other=0.0)
                acc_dp = tl.dot(do_tmp, v_tmp.T, acc_dp)
            
            ds_tmp = p_tmp * (acc_dp - d_val) * scale
            
            k_tmp2 = tl.load(K_ptr + bh * S * 128 + off_j * 128 + k_idx, shape=(16, 16), mask=(off_j * 128 + k_idx < S * 128), other=0.0)
            
            ds_tmp_exp = tl.reshape(ds_tmp, (16, 16))
            ds_tmp_exp = ds_tmp_exp[None, :, k_idx:idx+16]
            
            acc_dq = tl.dot(ds_tmp_exp, k_tmp2, acc_dq)
            acc_dq = tl.store(dQ_ptr + bh * S * 128 + off_i * 128 + k_idx, acc_dq, mask=(off_i * 128 + k_idx < S * 128))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Computes the backward pass of multi-head attention."""
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    scale = 1.0 / (128 ** 0.5)
    
    grid = (triton.cdiv(S, 16), B * H)
    
    dKdV_kernel[grid](
        Q, K, V, O, dO, L, dK, dV,
        S, scale,
        stride_b_outer=S * 128, stride_s_outer=128,
        stride_b_inner=128, stride_s_inner=1,
        num_warps=4, num_stages=2,
    )
    
    dQ_kernel[grid](
        Q, K, V, O, dO, L, dQ,
        S, scale,
        stride_b_outer=S * 128, stride_s_outer=128,
        stride_b_inner=128, stride_s_inner=1,
        num_warps=4, num_stages=2,
    )