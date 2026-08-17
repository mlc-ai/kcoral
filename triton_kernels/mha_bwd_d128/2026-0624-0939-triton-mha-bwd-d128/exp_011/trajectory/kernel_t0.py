import torch
import triton
import triton.language as tl


def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.jit
def _compute_dKdV_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, l_ptr, d_ptr,
    dk_ptr, dv_ptr,
    s, d: tl.constexpr, grid_h,
    BLOCK_M: tl.constexpr,
):
    start_n = tl.program_id(0)
    h = tl.program_id(1)
    z = tl.program_id(2)
    
    q_ptr_base = q_ptr + (z * grid_h + h) * s * d
    k_ptr_base = k_ptr + (z * grid_h + h) * s * d
    v_ptr_base = v_ptr + (z * grid_h + h) * s * d
    o_ptr_base = o_ptr + (z * grid_h + h) * s * d
    do_ptr_base = do_ptr + (z * grid_h + h) * s * d
    
    dk_ptr_base = dk_ptr + (z * grid_h + h) * s * d
    dv_ptr_base = dv_ptr + (z * grid_h + h) * s * d
    
    q_desc_fwd = tl.make_tensor_descriptor(q_ptr_base, shape=[s, d], strides=[d, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    k_desc_fwd = tl.make_tensor_descriptor(k_ptr_base, shape=[s, d], strides=[d, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    v_desc_fwd = tl.make_tensor_descriptor(v_ptr_base, shape=[s, d], strides=[d, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    o_desc_fwd = tl.make_tensor_descriptor(o_ptr_base, shape=[s, d], strides=[d, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    do_desc_fwd = tl.make_tensor_descriptor(do_ptr_base, shape=[s, d], strides=[d, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    
    k_desc_bwd = tl.make_tensor_descriptor(dk_ptr_base, shape=[s, d], strides=[d, 1], block_shape=[BLOCK_M, d])
    v_desc_bwd = tl.make_tensor_descriptor(dv_ptr_base, shape=[s, d], strides=[d, 1], block_shape=[BLOCK_M, d])
    
    K_0 = k_desc_fwd.load([start_n * BLOCK_M, 0])
    V_0 = v_desc_fwd.load([start_n * BLOCK_M, 0])
    
    acc_dK = tl.zeros((BLOCK_M, d), dtype=tl.float32)
    acc_dV = tl.zeros((BLOCK_M, d), dtype=tl.float32)
    
    scale = 1.0 / (d ** 0.5)
    
    for start_m in range(s // BLOCK_M):
        Q_0 = q_desc_fwd.load([start_m * BLOCK_M, 0])
        O_0 = o_desc_fwd.load([start_m * BLOCK_M, 0])
        dO_0 = do_desc_fwd.load([start_m * BLOCK_M, 0])
        
        l_ptr_offset = (z * grid_h + h) * s + start_m * BLOCK_M
        D_val = tl.load(d_ptr + l_ptr_offset + tl.arange(0, BLOCK_M), other=0.0)
        L_i = tl.load(l_ptr + l_ptr_offset + tl.arange(0, BLOCK_M), other=0.0)
        
        D = tl.sum(dO_0 * O_0, axis=1, keep_dims=True)
        
        S = tl.dot(Q_0, K_0.T)
        dP = tl.dot(dO_0, V_0.T)
        
        P_exp = tl.exp(S * scale - L_i[:, None])
        dS = P_exp * (dP - D) * scale
        
        acc_dV = tl.dot(P_exp.T, dO_0, acc_dV)
        acc_dK = tl.dot(dS.T, Q_0, acc_dK)
    
    k_desc_bwd.store([start_n * BLOCK_M, 0], acc_dK.to(tl.bfloat16), mask=True)
    v_desc_bwd.store([start_n * BLOCK_M, 0], acc_dV.to(tl.bfloat16), mask=True)


@triton.jit
def _compute_dQ_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, l_ptr, d_ptr,
    dq_ptr,
    s, d: tl.constexpr, grid_h,
    BLOCK_M: tl.constexpr,
):
    start_m = tl.program_id(0)
    h = tl.program_id(1)
    z = tl.program_id(2)
    
    q_ptr_base = q_ptr + (z * grid_h + h) * s * d
    k_ptr_base = k_ptr + (z * grid_h + h) * s * d
    v_ptr_base = v_ptr + (z * grid_h + h) * s * d
    o_ptr_base = o_ptr + (z * grid_h + h) * s * d
    do_ptr_base = do_ptr + (z * grid_h + h) * s * d
    
    dq_ptr_base = dq_ptr + (z * grid_h + h) * s * d
    
    q_desc_fwd = tl.make_tensor_descriptor(q_ptr_base, shape=[s, d], strides=[d, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    k_desc_fwd = tl.make_tensor_descriptor(k_ptr_base, shape=[s, d], strides=[d, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    v_desc_fwd = tl.make_tensor_descriptor(v_ptr_base, shape=[s, d], strides=[d, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    o_desc_fwd = tl.make_tensor_descriptor(o_ptr_base, shape=[s, d], strides=[d, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    do_desc_fwd = tl.make_tensor_descriptor(do_ptr_base, shape=[s, d], strides=[d, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    
    q_desc_bwd = tl.make_tensor_descriptor(dq_ptr_base, shape=[s, d], strides=[d, 1], block_shape=[BLOCK_M, d])
    
    Q_0 = q_desc_fwd.load([start_m * BLOCK_M, 0])
    O_0 = o_desc_fwd.load([start_m * BLOCK_M, 0])
    dO_0 = do_desc_fwd.load([start_m * BLOCK_M, 0])
    
    l_ptr_offset = (z * grid_h + h) * s + start_m * BLOCK_M
    D_val = tl.load(d_ptr + l_ptr_offset + tl.arange(0, BLOCK_M), other=0.0)
    L_i = tl.load(l_ptr + l_ptr_offset + tl.arange(0, BLOCK_M), other=0.0)
    
    D = tl.sum(dO_0 * O_0, axis=1, keep_dims=True)
    
    acc_dQ = tl.zeros((BLOCK_M, d), dtype=tl.float32)
    scale = 1.0 / (d ** 0.5)
    
    for start_n in range(s // BLOCK_M):
        K_0 = k_desc_fwd.load([start_n * BLOCK_M, 0])
        V_0 = v_desc_fwd.load([start_n * BLOCK_M, 0])
        
        S = tl.dot(Q_0, K_0.T)
        dP = tl.dot(dO_0, V_0.T)
        
        P_exp = tl.exp(S * scale - L_i[:, None])
        dS = P_exp * (dP - D) * scale
        
        acc_dQ = tl.dot(dS, K_0, acc_dQ)
    
    q_desc_bwd.store([start_m * BLOCK_M, 0], acc_dQ.to(tl.bfloat16), mask=True)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    b, h, s, d = Q.shape
    
    D = torch.sum(dO * O, dim=-1, dtype=torch.float32, device=Q.device)
    
    S_padded = triton.cdiv(s, 16) * 16
    if s != S_padded:
        Q_full = torch.zeros((b, h, S_padded, d), dtype=torch.bfloat16, device=Q.device)
        Q_full[:, :, :s, :] = Q
        Q = Q_full
        
        K_full = torch.zeros((b, h, S_padded, d), dtype=torch.bfloat16, device=Q.device)
        K_full[:, :, :s, :] = K
        K = K_full
        
        V_full = torch.zeros((b, h, S_padded, d), dtype=torch.bfloat16, device=Q.device)
        V_full[:, :, :s, :] = V
        V = V_full
        
        O_full = torch.zeros((b, h, S_padded, d), dtype=torch.bfloat16, device=Q.device)
        O_full[:, :, :s, :] = O
        O = O_full
        
        dO_full = torch.zeros((b, h, S_padded, d), dtype=torch.bfloat16, device=Q.device)
        dO_full[:, :, :s, :] = dO
        dO = dO_full
        
        L_full = torch.zeros((b, h, S_padded), dtype=torch.float32, device=Q.device)
        L_full[:, :, :s] = L
        L = L_full
        
        D_full = torch.zeros((b, h, S_padded), dtype=torch.float32, device=Q.device)
        D_full[:, :, :s] = D
        D = D_full
    
    D = D.contiguous()
    
    grid = (triton.cdiv(s, 16), h, b)
    
    BLOCK_M = 16
    
    _compute_dKdV_kernel[grid](
        Q, K, V, O, dO, L, D, dK, dV, S_padded, d=d, grid_h=h, BLOCK_M=BLOCK_M, num_warps=4
    )
    
    _compute_dQ_kernel[grid](
        Q, K, V, O, dO, L, D, dQ, S_padded, d=d, grid_h=h, BLOCK_M=BLOCK_M, num_warps=4
    )