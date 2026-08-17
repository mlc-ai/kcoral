import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


def make_desc(tensor, block_shape):
    contiguous = tensor.contiguous()
    return TensorDescriptor.from_tensor(contiguous, block_shape, padding_option="zero")


@triton.jit
def _bwd_dKdV_kernel(
    L_ptr, dK_ptr, dV_ptr,
    s, d, tau,
    q_c, k_c, v_c, do_c, o_c,
):
    batch_head_idx = tl.program_id(1)
    j = tl.program_id(0)
    
    k_base = (batch_head_idx * s + j * 128).to(tl.int64)
    K_j = k_c.load([k_base, 0])
    V_j = v_c.load([k_base, 0])
    
    dk = tl.zeros((128, 128), dtype=tl.float32)
    dv = tl.zeros((128, 128), dtype=tl.float32)
    
    num_s_blocks = tl.cdiv(s, 128)
    row_idx = tl.arange(0, 128)
    
    for i in range(j, num_s_blocks):
        q_base = (batch_head_idx * s + i * 128).to(tl.int64)
        Q_i = q_c.load([q_base, 0])
        dO_i = do_c.load([q_base, 0])
        O_i = o_c.load([q_base, 0])
        
        D_val = tl.sum(dO_i * O_i, axis=1)
        
        r_idx = (i * 128 + row_idx).to(tl.int32)
        L_i = tl.load(L_ptr + batch_head_idx * s + i * 128 + row_idx, mask=(r_idx < s), other=0.0)
        
        S = tl.dot(Q_i, K_j.T)
        
        c_idx = (j * 128 + row_idx).to(tl.int32)
        mask = (r_idx[None, :] >= c_idx[:, None]) & (r_idx[None, :] < s) & (c_idx[:, None] < s)
        
        P_unmasked = tl.exp(S * tau - L_i.unsqueeze(-1))
        P = tl.where(mask, P_unmasked, 0.0)
        
        dP = tl.dot(dO_i, V_j.T)
        
        dS = P * (dP - D_val.unsqueeze(-1)) * tau
        
        p_bf16 = P.to(tl.bfloat16)
        ds_bf16 = dS.to(tl.bfloat16)
        
        dv += tl.dot(p_bf16.T, dO_i)
        dk += tl.dot(ds_bf16.T, Q_i)

    r_idx_store = (j * 128 + row_idx).to(tl.int32)
    col_idx = tl.arange(0, 128)
    dk_bf16 = dk.to(tl.bfloat16)
    dv_bf16 = dv.to(tl.bfloat16)
    
    base_dk = dK_ptr + batch_head_idx * s * 128 + j * 128 * 128
    base_dv = dV_ptr + batch_head_idx * s * 128 + j * 128 * 128
    
    mask_store = (r_idx_store[:, None] < s)
    tl.store(base_dk + row_idx[:, None] * 128 + col_idx[None, :], dk_bf16, mask=mask_store)
    tl.store(base_dv + row_idx[:, None] * 128 + col_idx[None, :], dv_bf16, mask=mask_store)


@triton.jit
def _bwd_dQ_kernel(
    L_ptr, dQ_ptr,
    s, d, tau,
    k_c, v_c, q_c, do_c, o_c,
):
    batch_head_idx = tl.program_id(1)
    i = tl.program_id(0)
    
    q_base = (batch_head_idx * s + i * 128).to(tl.int64)
    Q_i = q_c.load([q_base, 0])
    dO_i = do_c.load([q_base, 0])
    O_i = o_c.load([q_base, 0])
    
    D_val = tl.sum(dO_i * O_i, axis=1)
    
    row_idx = tl.arange(0, 128)
    r_idx = (i * 128 + row_idx).to(tl.int32)
    L_i = tl.load(L_ptr + batch_head_idx * s + i * 128 + row_idx, mask=(r_idx < s), other=0.0)
    
    dq = tl.zeros((128, 128), dtype=tl.float32)
    
    for j in range(0, i + 1):
        k_base = (batch_head_idx * s + j * 128).to(tl.int64)
        K_j = k_c.load([k_base, 0])
        V_j = v_c.load([k_base, 0])
        
        S = tl.dot(Q_i, K_j.T)
        
        c_idx = (j * 128 + row_idx).to(tl.int32)
        mask = (r_idx[None, :] >= c_idx[:, None]) & (r_idx[None, :] < s) & (c_idx[:, None] < s)
        
        P_unmasked = tl.exp(S * tau - L_i.unsqueeze(-1))
        P = tl.where(mask, P_unmasked, 0.0)
        
        dP = tl.dot(dO_i, V_j.T)
        
        dS = P * (dP - D_val.unsqueeze(-1)) * tau
        
        ds_bf16 = dS.to(tl.bfloat16)
        
        dq += tl.dot(ds_bf16, K_j)

    col_idx = tl.arange(0, 128)
    dq_bf16 = dq.to(tl.bfloat16)
    
    base_dq = dQ_ptr + batch_head_idx * s * 128 + i * 128 * 128
    mask_store = (r_idx[:, None] < s)
    tl.store(base_dq + row_idx[:, None] * 128 + col_idx[None, :], dq_bf16, mask=mask_store)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    b, h, s, d = Q.shape
    
    if s == 0:
        return

    q_c = make_desc(Q.view(b * h * s, 128), [128, 128])
    k_c = make_desc(K.view(b * h * s, 128), [128, 128])
    v_c = make_desc(V.view(b * h * s, 128), [128, 128])
    o_c = make_desc(O.view(b * h * s, 128), [128, 128])
    do_c = make_desc(dO.view(b * h * s, 128), [128, 128])
    
    tau = 1.0 / (d ** 0.5)
    
    grid_KV = (triton.cdiv(s, 128), b * h)
    _bwd_dKdV_kernel[grid_KV](
        L, dK, dV,
        s, d, tau,
        q_c, k_c, v_c, do_c, o_c,
        num_warps=8,
    )
    
    grid_Q = (triton.cdiv(s, 128), b * h)
    _bwd_dQ_kernel[grid_Q](
        L, dQ,
        s, d, tau,
        k_c, v_c, q_c, do_c, o_c,
        num_warps=8,
    )