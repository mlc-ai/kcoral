import torch
import triton
import triton.language as tl

triton.set_allocator(lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8))

@triton.jit
def _precompute_D_kernel(p_dO, p_O, p_D, n_elements, BLOCK: tl.constexpr):
    start_idx = tl.program_id(0) * BLOCK
    idx = start_idx + tl.arange(0, BLOCK)
    mask = idx < n_elements
    d_offs = tl.arange(0, 128)
    d_offsets = idx[:, None] * 128 + d_offs[None, :]
    dO_val = tl.load(p_dO + d_offsets, mask=mask[:, None], other=0.0)
    O_val = tl.load(p_O + d_offsets, mask=mask[:, None], other=0.0)
    D_val = tl.sum(dO_val * O_val, axis=1)
    tl.store(p_D + idx, D_val, mask=mask)

@triton.jit
def _bwd_dq_kernel(
    p_Q, p_K, p_V, p_dO, p_L, p_D, p_dQ,
    S, tau,
    c_b, c_h, c_L_b, c_L_h,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    i = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    
    off_q = i * BLOCK_M
    
    q_desc = tl.make_tensor_descriptor(p_Q + b * c_b + h * c_h, shape=[S, 128], strides=[128, 1], block_shape=[BLOCK_M, 128], padding_option="zero")
    do_desc = tl.make_tensor_descriptor(p_dO + b * c_b + h * c_h, shape=[S, 128], strides=[128, 1], block_shape=[BLOCK_M, 128], padding_option="zero")
    k_desc = tl.make_tensor_descriptor(p_K + b * c_b + h * c_h, shape=[S, 128], strides=[128, 1], block_shape=[BLOCK_N, 128], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(p_V + b * c_b + h * c_h, shape=[S, 128], strides=[128, 1], block_shape=[BLOCK_N, 128], padding_option="zero")
    dq_desc = tl.make_tensor_descriptor(p_dQ + b * c_b + h * c_h, shape=[S, 128], strides=[128, 1], block_shape=[BLOCK_M, 128])
    
    Q_i = q_desc.load([off_q, 0])
    dO_i = do_desc.load([off_q, 0])
    
    row_offs = off_q + tl.arange(0, BLOCK_M)
    mask_l = row_offs < S
    L_i = tl.load(p_L + b * c_L_b + h * c_L_h + row_offs, mask=mask_l, other=0.0)
    D_i = tl.load(p_D + b * c_L_b + h * c_L_h + row_offs, mask=mask_l, other=0.0)
    
    dQ_i = tl.zeros((BLOCK_M, 128), tl.float32)
    
    for j in range(i + 1):
        off_k = j * BLOCK_N
        K_j = k_desc.load([off_k, 0])
        V_j = v_desc.load([off_k, 0])
        
        S_mat = tl.dot(Q_i, K_j.T)
        S_mat = S_mat * tau
        dP = tl.dot(dO_i, V_j.T)
        
        P = tl.exp(S_mat - L_i[:, None])
        dS = P * (dP - D_i[:, None]) * tau
        
        q_offs = off_q + tl.arange(0, BLOCK_M)
        k_offs = off_k + tl.arange(0, BLOCK_N)
        mask = (q_offs[:, None] >= k_offs[None, :]) & (q_offs[:, None] < S) & (k_offs[None, :] < S)
        dS = dS * mask
        
        dQ_i = tl.dot(dS, K_j, acc=dQ_i)
    
    dq_desc.store([off_q, 0], dQ_i.to(tl.bfloat16))

@triton.jit
def _bwd_dkv_kernel(
    p_Q, p_K, p_V, p_dO, p_L, p_D, p_dK, p_dV,
    S, tau,
    c_b, c_h, c_L_b, c_L_h,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    j = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    
    num_blocks_r = tl.cdiv(S, BLOCK_M)
    
    off_k = j * BLOCK_N
    
    q_desc = tl.make_tensor_descriptor(p_Q + b * c_b + h * c_h, shape=[S, 128], strides=[128, 1], block_shape=[BLOCK_M, 128], padding_option="zero")
    k_desc = tl.make_tensor_descriptor(p_K + b * c_b + h * c_h, shape=[S, 128], strides=[128, 1], block_shape=[BLOCK_N, 128], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(p_V + b * c_b + h * c_h, shape=[S, 128], strides=[128, 1], block_shape=[BLOCK_N, 128], padding_option="zero")
    do_desc = tl.make_tensor_descriptor(p_dO + b * c_b + h * c_h, shape=[S, 128], strides=[128, 1], block_shape=[BLOCK_M, 128], padding_option="zero")
    dk_desc = tl.make_tensor_descriptor(p_dK + b * c_b + h * c_h, shape=[S, 128], strides=[128, 1], block_shape=[BLOCK_N, 128])
    dv_desc = tl.make_tensor_descriptor(p_dV + b * c_b + h * c_h, shape=[S, 128], strides=[128, 1], block_shape=[BLOCK_N, 128])
    
    K_j = k_desc.load([off_k, 0])
    V_j = v_desc.load([off_k, 0])
    
    dK_j = tl.zeros((BLOCK_N, 128), tl.float32)
    dV_j = tl.zeros((BLOCK_N, 128), tl.float32)
    
    for i in range(j, num_blocks_r):
        off_q = i * BLOCK_M
        Q_i = q_desc.load([off_q, 0])
        dO_i = do_desc.load([off_q, 0])
        
        row_offs = off_q + tl.arange(0, BLOCK_M)
        mask_l = row_offs < S
        L_i = tl.load(p_L + b * c_L_b + h * c_L_h + row_offs, mask=mask_l, other=0.0)
        D_i = tl.load(p_D + b * c_L_b + h * c_L_h + row_offs, mask=mask_l, other=0.0)
        
        S_mat = tl.dot(Q_i, K_j.T)
        S_mat = S_mat * tau
        dP = tl.dot(dO_i, V_j.T)
        
        P = tl.exp(S_mat - L_i[:, None])
        dS = P * (dP - D_i[:, None]) * tau
        
        q_offs = off_q + tl.arange(0, BLOCK_M)
        k_offs = off_k + tl.arange(0, BLOCK_N)
        mask = (q_offs[:, None] >= k_offs[None, :]) & (q_offs[:, None] < S) & (k_offs[None, :] < S)
        
        P_masked = P * mask
        dS_masked = dS * mask
        
        dV_j = tl.dot(P_masked.T, dO_i, acc=dV_j)
        dK_j = tl.dot(dS_masked.T, Q_i, acc=dK_j)
    
    dk_desc.store([off_k, 0], dK_j.to(tl.bfloat16))
    dv_desc.store([off_k, 0], dV_j.to(tl.bfloat16))

def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    assert d == 128, f"d must be 128, got {d}"
    n_elements = B * H * S
    tau = 1.0 / (d ** 0.5)
    
    D = torch.empty((B, H, S), dtype=torch.float32, device=Q.device)
    
    BLOCK_M = 64
    BLOCK_N = 64
    
    grid_D = (triton.cdiv(n_elements, 32),)
    _precompute_D_kernel[grid_D](
        dO.data_ptr(), O.data_ptr(), D.data_ptr(),
        n_elements,
        BLOCK=32,
    )
    
    num_blocks_r = triton.cdiv(S, BLOCK_M)
    num_blocks_c = triton.cdiv(S, BLOCK_N)
    
    grid_dQ = (num_blocks_r, B, H)
    _bwd_dq_kernel[grid_dQ](
        Q.data_ptr(), K.data_ptr(), V.data_ptr(), dO.data_ptr(), L.data_ptr(), D.data_ptr(), dQ.data_ptr(),
        S, tau,
        H * S * d, S * d, H * S, S,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_stages=3,
    )
    
    grid_dKV = (num_blocks_c, B, H)
    _bwd_dkv_kernel[grid_dKV](
        Q.data_ptr(), K.data_ptr(), V.data_ptr(), dO.data_ptr(), L.data_ptr(), D.data_ptr(), dK.data_ptr(), dV.data_ptr(),
        S, tau,
        H * S * d, S * d, H * S, S,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_stages=3,
    )