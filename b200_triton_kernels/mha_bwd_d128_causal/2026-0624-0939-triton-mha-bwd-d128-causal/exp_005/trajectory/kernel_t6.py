import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


def make_desc(tensor, block_shape):
    contiguous = tensor.contiguous()
    return TensorDescriptor.from_tensor(contiguous, block_shape)


@triton.jit
def _bwd_dKdV_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, O_ptr, L_ptr, dK_ptr, dV_ptr,
    s, tau,
    q_c, k_c, v_c, do_c, o_c, l_c,
):
    batch_head_idx = tl.program_id(1)
    j = tl.program_id(0)
    
    row_idx = tl.arange(0, 64)
    col_idx_0 = tl.arange(0, 64)
    
    k_base = (batch_head_idx * s + j * 64).to(tl.int64)
    K_j_0 = k_c.load([k_base, 0])
    K_j_1 = k_c.load([k_base, 64])
    V_j_0 = v_c.load([k_base, 0])
    V_j_1 = v_c.load([k_base, 64])
    
    dk_0 = tl.zeros((64, 64), dtype=tl.float32)
    dk_1 = tl.zeros((64, 64), dtype=tl.float32)
    dv_0 = tl.zeros((64, 64), dtype=tl.float32)
    dv_1 = tl.zeros((64, 64), dtype=tl.float32)
    
    num_s_blocks = tl.cdiv(s, 64)
    
    for i in range(j, num_s_blocks):
        q_base = (batch_head_idx * s + i * 64).to(tl.int64)
        Q_i_0 = q_c.load([q_base, 0])
        Q_i_1 = q_c.load([q_base, 64])
        dO_i_0 = do_c.load([q_base, 0])
        dO_i_1 = do_c.load([q_base, 64])
        O_i_0 = o_c.load([q_base, 0])
        O_i_1 = o_c.load([q_base, 64])
        
        D_i = tl.sum(dO_i_0 * O_i_0 + dO_i_1 * O_i_1, axis=1)
        
        l_base = (batch_head_idx * s + i * 64).to(tl.int64)
        L_i = l_c.load(l_base)
        
        S = tl.dot(Q_i_0, K_j_0.T) + tl.dot(Q_i_1, K_j_1.T)
        
        r_idx = (i * 64 + row_idx).to(tl.int32)
        c_idx = (j * 64 + row_idx).to(tl.int32)
        mask = (r_idx[None, :] >= c_idx[:, None]) & (r_idx[None, :] < s) & (c_idx[:, None] < s)
        
        P_unmasked = tl.exp(S * tau - L_i.unsqueeze(-1))
        P = tl.where(mask, P_unmasked, 0.0)
        
        dP = tl.dot(dO_i_0, V_j_0.T) + tl.dot(dO_i_1, V_j_1.T)
        
        dS = P * (dP - D_i.unsqueeze(-1)) * tau
        
        p_bf16 = P.to(tl.bfloat16)
        ds_bf16 = dS.to(tl.bfloat16)
        
        dv_0 += tl.dot(p_bf16.T, dO_i_0)
        dv_1 += tl.dot(p_bf16.T, dO_i_1)
        dk_0 += tl.dot(ds_bf16.T, Q_i_0)
        dk_1 += tl.dot(ds_bf16.T, Q_i_1)
    
    r_idx_store = (j * 64 + row_idx).to(tl.int32)
    base_dk = dK_ptr + batch_head_idx * s * 128 + j * 64 * 128
    base_dv = dV_ptr + batch_head_idx * s * 128 + j * 64 * 128
    mask_store = (r_idx_store[:, None] < s) & (col_idx_0[None, :] < 128)
    
    tl.store(base_dk + row_idx[:, None] * 128 + col_idx_0[None, :], dk_0.to(tl.bfloat16), mask=mask_store)
    tl.store(base_dk + row_idx[:, None] * 128 + col_idx_0[None, :] + 64, dk_1.to(tl.bfloat16), mask=mask_store)
    tl.store(base_dv + row_idx[:, None] * 128 + col_idx_0[None, :], dv_0.to(tl.bfloat16), mask=mask_store)
    tl.store(base_dv + row_idx[:, None] * 128 + col_idx_0[None, :] + 64, dv_1.to(tl.bfloat16), mask=mask_store)


@triton.jit
def _bwd_dQ_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, O_ptr, L_ptr, dQ_ptr,
    s, tau,
    k_c, v_c, q_c, do_c, o_c, l_c,
):
    batch_head_idx = tl.program_id(1)
    i = tl.program_id(0)
    
    row_idx = tl.arange(0, 64)
    col_idx_0 = tl.arange(0, 64)
    
    q_base = (batch_head_idx * s + i * 64).to(tl.int64)
    Q_i_0 = q_c.load([q_base, 0])
    Q_i_1 = q_c.load([q_base, 64])
    dO_i_0 = do_c.load([q_base, 0])
    dO_i_1 = do_c.load([q_base, 64])
    O_i_0 = o_c.load([q_base, 0])
    O_i_1 = o_c.load([q_base, 64])
    
    D_i = tl.sum(dO_i_0 * O_i_0 + dO_i_1 * O_i_1, axis=1)
    
    l_base = (batch_head_idx * s + i * 64).to(tl.int64)
    L_i = l_c.load(l_base)
    
    dq_0 = tl.zeros((64, 64), dtype=tl.float32)
    dq_1 = tl.zeros((64, 64), dtype=tl.float32)
    
    for j in range(0, i + 1):
        k_base = (batch_head_idx * s + j * 64).to(tl.int64)
        K_j_0 = k_c.load([k_base, 0])
        K_j_1 = k_c.load([k_base, 64])
        V_j_0 = v_c.load([k_base, 0])
        V_j_1 = v_c.load([k_base, 64])
        
        S = tl.dot(Q_i_0, K_j_0.T) + tl.dot(Q_i_1, K_j_1.T)
        
        r_idx = (i * 64 + row_idx).to(tl.int32)
        c_idx = (j * 64 + row_idx).to(tl.int32)
        mask = (r_idx[None, :] >= c_idx[:, None]) & (r_idx[None, :] < s) & (c_idx[:, None] < s)
        
        P_unmasked = tl.exp(S * tau - L_i.unsqueeze(-1))
        P = tl.where(mask, P_unmasked, 0.0)
        
        dP = tl.dot(dO_i_0, V_j_0.T) + tl.dot(dO_i_1, V_j_1.T)
        
        dS = P * (dP - D_i.unsqueeze(-1)) * tau
        
        ds_bf16 = dS.to(tl.bfloat16)
        
        dq_0 += tl.dot(ds_bf16, K_j_0)
        dq_1 += tl.dot(ds_bf16, K_j_1)
    
    r_idx_store = (i * 64 + row_idx).to(tl.int32)
    base_dq = dQ_ptr + batch_head_idx * s * 128 + i * 64 * 128
    mask_store = (r_idx_store[:, None] < s) & (col_idx_0[None, :] < 128)
    
    tl.store(base_dq + row_idx[:, None] * 128 + col_idx_0[None, :], dq_0.to(tl.bfloat16), mask=mask_store)
    tl.store(base_dq + row_idx[:, None] * 128 + col_idx_0[None, :] + 64, dq_1.to(tl.bfloat16), mask=mask_store)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    b, h, s, d = Q.shape
    
    if s == 0:
        return
    
    tau = 1.0 / (d ** 0.5)
    
    q_c = make_desc(Q.view(b * h * s, 128), [64, 64])
    k_c = make_desc(K.view(b * h * s, 128), [64, 64])
    v_c = make_desc(V.view(b * h * s, 128), [64, 64])
    o_c = make_desc(O.view(b * h * s, 128), [64, 64])
    do_c = make_desc(dO.view(b * h * s, 128), [64, 64])
    l_c = make_desc(L.view(b * h * s), [64])
    
    grid_KV = (triton.cdiv(s, 64), b * h)
    _bwd_dKdV_kernel[grid_KV](
        Q, K, V, dO, O, L, dK, dV,
        s, tau,
        q_c, k_c, v_c, do_c, o_c, l_c,
        num_warps=8,
        num_stages=2,
    )
    
    grid_Q = (triton.cdiv(s, 64), b * h)
    _bwd_dQ_kernel[grid_Q](
        Q, K, V, dO, O, L, dQ,
        s, tau,
        k_c, v_c, q_c, do_c, o_c, l_c,
        num_warps=8,
        num_stages=2,
    )