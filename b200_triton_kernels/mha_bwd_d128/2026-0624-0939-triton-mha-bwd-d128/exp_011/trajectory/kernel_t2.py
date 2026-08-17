import torch
import triton
import triton.language as tl


def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)


@triton.jit
def _compute_D_kernel(dO_ptr, O_ptr, D_ptr, N, cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    if row < N:
        offset = row * cols
        idx = tl.arange(0, BLOCK_SIZE)
        mask = idx < cols
        dO_row = tl.load(dO_ptr + offset + idx, mask=mask, other=0.0)
        O_row = tl.load(O_ptr + offset + idx, mask=mask, other=0.0)
        d = tl.sum(dO_row * O_row)
        tl.store(D_ptr + row, d)


@triton.jit
def _bwd_dKdV(
    q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, l_ptr, d_ptr,
    dk_ptr, dv_ptr,
    s_len, d, b_size,
    H: tl.constexpr,
    B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_n = tl.program_id(0)
    h = tl.program_id(1)
    z = tl.program_id(2)
    
    q_ptr_base = q_ptr + (z * H + h) * s_len * d
    k_ptr_base = k_ptr + (z * H + h) * s_len * d
    v_ptr_base = v_ptr + (z * H + h) * s_len * d
    o_ptr_base = o_ptr + (z * H + h) * s_len * d
    do_ptr_base = do_ptr + (z * H + h) * s_len * d
    dk_ptr_base = dk_ptr + (z * H + h) * s_len * d
    dv_ptr_base = dv_ptr + (z * H + h) * s_len * d
    
    q_desc = tl.make_tensor_descriptor(q_ptr_base, shape=[s_len, d], strides=[d, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    k_desc = tl.make_tensor_descriptor(k_ptr_base, shape=[s_len, d], strides=[d, 1], block_shape=[BLOCK_N, d], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(v_ptr_base, shape=[s_len, d], strides=[d, 1], block_shape=[BLOCK_N, d], padding_option="zero")
    o_desc = tl.make_tensor_descriptor(o_ptr_base, shape=[s_len, d], strides=[d, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    do_desc = tl.make_tensor_descriptor(do_ptr_base, shape=[s_len, d], strides=[d, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    dk_desc = tl.make_tensor_descriptor(dk_ptr_base, shape=[s_len, d], strides=[d, 1], block_shape=[BLOCK_N, d])
    dv_desc = tl.make_tensor_descriptor(dv_ptr_base, shape=[s_len, d], strides=[d, 1], block_shape=[BLOCK_N, d])
    
    K_0 = k_desc.load([start_n * BLOCK_N, 0])
    V_0 = v_desc.load([start_n * BLOCK_N, 0])
    
    acc_dK = tl.zeros((BLOCK_N, d), dtype=tl.float32)
    acc_dV = tl.zeros((BLOCK_N, d), dtype=tl.float32)
    
    scale = 1.0 / (d ** 0.5)
    num_m_tiles = tl.cdiv(s_len, BLOCK_M)
    
    for start_m in range(num_m_tiles):
        Q_0 = q_desc.load([start_m * BLOCK_M, 0])
        O_0 = o_desc.load([start_m * BLOCK_M, 0])
        dO_0 = do_desc.load([start_m * BLOCK_M, 0])
        
        l_idx = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        l_offset = (z * H + h) * s_len + l_idx
        m_mask = l_idx < s_len
        L_i = tl.load(l_ptr + l_offset, mask=m_mask, other=0.0)
        D_val = tl.load(d_ptr + l_offset + tl.arange(0, BLOCK_M), mask=m_mask, other=0.0)
        
        S_val = tl.dot(Q_0, K_0.T)
        dP = tl.dot(dO_0, V_0.T)
        
        P_exp = tl.exp(S_val * scale - L_i[:, None])
        dS = P_exp * (dP - D_val[:, None]) * scale
        
        acc_dV = tl.dot(P_exp.T, dO_0, acc_dV)
        acc_dK = tl.dot(dS.T, Q_0, acc_dK)
    
    dk_desc.store([start_n * BLOCK_N, 0], acc_dK.to(tl.bfloat16))
    dv_desc.store([start_n * BLOCK_N, 0], acc_dV.to(tl.bfloat16))


@triton.jit
def _bwd_dQ(
    q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, l_ptr, d_ptr,
    dq_ptr,
    s_len, d, b_size,
    H: tl.constexpr,
    B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    h = tl.program_id(1)
    z = tl.program_id(2)
    
    q_ptr_base = q_ptr + (z * H + h) * s_len * d
    k_ptr_base = k_ptr + (z * H + h) * s_len * d
    v_ptr_base = v_ptr + (z * H + h) * s_len * d
    o_ptr_base = o_ptr + (z * H + h) * s_len * d
    do_ptr_base = do_ptr + (z * H + h) * s_len * d
    dq_ptr_base = dq_ptr + (z * H + h) * s_len * d
    
    q_desc = tl.make_tensor_descriptor(q_ptr_base, shape=[s_len, d], strides=[d, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    k_desc = tl.make_tensor_descriptor(k_ptr_base, shape=[s_len, d], strides=[d, 1], block_shape=[BLOCK_N, d], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(v_ptr_base, shape=[s_len, d], strides=[d, 1], block_shape=[BLOCK_N, d], padding_option="zero")
    o_desc = tl.make_tensor_descriptor(o_ptr_base, shape=[s_len, d], strides=[d, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    do_desc = tl.make_tensor_descriptor(do_ptr_base, shape=[s_len, d], strides=[d, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    dq_desc = tl.make_tensor_descriptor(dq_ptr_base, shape=[s_len, d], strides=[d, 1], block_shape=[BLOCK_M, d])
    
    Q_0 = q_desc.load([start_m * BLOCK_M, 0])
    O_0 = o_desc.load([start_m * BLOCK_M, 0])
    dO_0 = do_desc.load([start_m * BLOCK_M, 0])
    
    l_idx = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    l_offset = (z * H + h) * s_len + l_idx
    m_mask = l_idx < s_len
    L_i = tl.load(l_ptr + l_offset, mask=m_mask, other=0.0)
    D_val = tl.load(d_ptr + l_offset + tl.arange(0, BLOCK_M), mask=m_mask, other=0.0)
    
    acc_dQ = tl.zeros((BLOCK_M, d), dtype=tl.float32)
    scale = 1.0 / (d ** 0.5)
    num_n_tiles = tl.cdiv(s_len, BLOCK_N)
    
    for start_n in range(num_n_tiles):
        K_0 = k_desc.load([start_n * BLOCK_N, 0])
        V_0 = v_desc.load([start_n * BLOCK_N, 0])
        
        S_val = tl.dot(Q_0, K_0.T)
        dP = tl.dot(dO_0, V_0.T)
        
        P_exp = tl.exp(S_val * scale - L_i[:, None])
        dS = P_exp * (dP - D_val[:, None]) * scale
        
        acc_dQ = tl.dot(dS, K_0, acc_dQ)
    
    dq_desc.store([start_m * BLOCK_M, 0], acc_dQ.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    b, h, s, d = Q.shape
    
    D = torch.empty((b, h, s), dtype=torch.float32, device=Q.device)
    
    total_rows = b * h * s
    grid_D = (triton.cdiv(total_rows, 128),)
    _compute_D_kernel[grid_D](dO, O, D, total_rows, d, BLOCK_SIZE=128)
    
    D = D.contiguous()
    
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
    
    size_assert = lambda x, msg: None
    
    H_val = h
    B_val = b
    
    BLOCK_M = 128
    BLOCK_N = 64
    
    grid_kdv = (triton.cdiv(S_padded, BLOCK_N), H_val, B_val)
    _bwd_dKdV[grid_kdv](
        Q, K, V, O, dO, L, D, dK, dV, S_padded, d, b,
        H=H_val, B=B_val, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4
    )
    
    grid_q = (triton.cdiv(S_padded, BLOCK_M), H_val, B_val)
    _bwd_dQ[grid_q](
        Q, K, V, O, dO, L, D, dQ, S_padded, d, b,
        H=H_val, B=B_val, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4
    )