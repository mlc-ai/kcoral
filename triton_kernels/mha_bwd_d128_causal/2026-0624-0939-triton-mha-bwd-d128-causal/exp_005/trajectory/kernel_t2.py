import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


def make_desc(tensor, block_shape):
    contiguous = torch.ascontiguousarray(tensor)
    return TensorDescriptor.from_tensor(contiguous, block_shape, padding_option="zero")


@triton.jit
def _preprocess_kernel(dO_ptr, O_ptr, D_ptr, n_elements, d_dim, BLOCK_D: tl.constexpr):
    idx = tl.program_id(0)
    if idx < n_elements:
        base = (idx * d_dim).to(tl.int64)
        d_offsets = tl.arange(0, BLOCK_D)
        do_vals = tl.load(dO_ptr + base + d_offsets)
        o_vals = tl.load(O_ptr + base + d_offsets)
        d = tl.sum(do_vals * o_vals)
        tl.store(D_ptr + idx, d)


@triton.jit
def _bwd_dKdV_kernel(
    Q_ptr, dO_ptr, L_ptr, D_ptr, dK_ptr, dV_ptr,
    s, d, tau,
    NUM_SMS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    q_c, k_c, v_c, do_c, o_c, l_c, d_c,
):
    batch_head_idx = tl.program_id(1)
    pid_n = tl.program_id(0)
    
    dk_0 = tl.zeros((64, 64), dtype=tl.float32)
    dk_1 = tl.zeros((64, 64), dtype=tl.float32)
    dv_0 = tl.zeros((64, 64), dtype=tl.float32)
    dv_1 = tl.zeros((64, 64), dtype=tl.float32)

    k_base_offset = (batch_head_idx * s + pid_n * 64).to(tl.int64)
    K_j_0 = k_c.load([k_base_offset, 0])
    K_j_1 = k_c.load([k_base_offset, 64])
    V_j_0 = v_c.load([k_base_offset, 0])
    V_j_1 = v_c.load([k_base_offset, 64])

    num_s_blocks = tl.cdiv(s, 64)

    for i in range(pid_n, num_s_blocks):
        q_base_offset = (batch_head_idx * s + i * 64).to(tl.int64)
        Q_i_0 = q_c.load([q_base_offset, 0])
        Q_i_1 = q_c.load([q_base_offset, 64])
        
        dO_i_0 = do_c.load([q_base_offset, 0])
        dO_i_1 = do_c.load([q_base_offset, 64])
        
        O_i_0 = o_c.load([q_base_offset, 0])
        O_i_1 = o_c.load([q_base_offset, 64])
        
        D_val = tl.sum(dO_i_0 * O_i_0 + dO_i_1 * O_i_1, axis=1)
        d_i = D_val.unsqueeze(-1)
        
        l_base_offset = (batch_head_idx * s + i * 64).to(tl.int64)
        L_i = l_c.load(l_base_offset)
        D_i = d_c.load(l_base_offset)
        d_i_scalar = D_i.unsqueeze(-1)
        l_i = L_i.unsqueeze(-1)
        
        s_0 = tl.dot(Q_i_0, K_j_0.T)
        s_1 = tl.dot(Q_i_1, K_j_1.T)
        s_val = s_0 + s_1
        
        r_idx = (i * 64 + tl.arange(0, 64)).to(tl.int32)
        c_idx = (pid_n * 64 + tl.arange(0, 64)).to(tl.int32)
        mask = (r_idx[None, :] >= c_idx[:, None]) & (r_idx[None, :] < s) & (c_idx[:, None] < s)
        
        p_unmasked = tl.exp(s_val * tau - l_i)
        p = tl.where(mask, p_unmasked, 0.0)
        
        dp_0 = tl.dot(dO_i_0, V_j_0.T)
        dp_1 = tl.dot(dO_i_1, V_j_1.T)
        dp = dp_0 + dp_1
        
        ds = p * (dp - d_i) * tau
        
        p_bf16 = p.to(tl.bfloat16)
        ds_bf16 = ds.to(tl.bfloat16)
        
        dv_0 += tl.dot(p_bf16.T, dO_i_0)
        dv_1 += tl.dot(p_bf16.T, dO_i_1)
        
        dk_0 += tl.dot(ds_bf16.T, Q_i_0)
        dk_1 += tl.dot(ds_bf16.T, Q_i_1)

        D_val = tl.sum(do_i_0 * o_i_0 + do_i_1 * o_i_1, axis=1)
        d_i = D_val.unsqueeze(-1)

    q_c.store(dK_ptr, [batch_head_idx * s + pid_n * 64, 0], dk_0.to(tl.bfloat16))
    q_c.store(dK_ptr, [batch_head_idx * s + pid_n * 64, 64], dk_1.to(tl.bfloat16))
    q_c.store(dV_ptr, [batch_head_idx * s + pid_n * 64, 0], dv_0.to(tl.bfloat16))
    q_c.store(dV_ptr, [batch_head_idx * s + pid_n * 64, 64], dv_1.to(tl.bfloat16))


@triton.jit
def _bwd_dQ_kernel(
    K_ptr, V_ptr, L_ptr, D_ptr, Q_ptr, dO_ptr, dQ_ptr,
    s, d, tau,
    NUM_SMS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    k_c, v_c, q_c, do_c, o_c, l_c, d_c,
):
    batch_head_idx = tl.program_id(1)
    pid_m = tl.program_id(0)
    
    dq_0 = tl.zeros((64, 64), dtype=tl.float32)
    dq_1 = tl.zeros((64, 64), dtype=tl.float32)

    q_base_offset = (batch_head_idx * s + pid_m * 64).to(tl.int64)
    Q_i_0 = q_c.load([q_base_offset, 0])
    Q_i_1 = q_c.load([q_base_offset, 64])
    
    dO_i_0 = do_c.load([q_base_offset, 0])
    dO_i_1 = do_c.load([q_base_offset, 64])
    
    O_i_0 = o_c.load([q_base_offset, 0])
    O_i_1 = o_c.load([q_base_offset, 64])
    
    D_val = tl.sum(dO_i_0 * O_i_0 + dO_i_1 * O_i_1, axis=1)
    d_i = D_val.unsqueeze(-1)

    l_base_offset = (batch_head_idx * s + pid_m * 64).to(tl.int64)
    L_i = l_c.load(l_base_offset)
    D_i = d_c.load(l_base_offset)
    d_i_scalar = D_i.unsqueeze(-1)
    l_i = L_i.unsqueeze(-1)

    num_s_blocks = tl.cdiv(s, 64)

    for j in range(0, pid_m + 1):
        k_base_offset = (batch_head_idx * s + j * 64).to(tl.int64)
        K_j_0 = k_c.load([k_base_offset, 0])
        K_j_1 = k_c.load([k_base_offset, 64])
        
        V_j_0 = v_c.load([k_base_offset, 0])
        V_j_1 = v_c.load([k_base_offset, 64])
        
        s_0 = tl.dot(Q_i_0, K_j_0.T)
        s_1 = tl.dot(Q_i_1, K_j_1.T)
        s_val = s_0 + s_1
        
        r_idx = (pid_m * 64 + tl.arange(0, 64)).to(tl.int32)
        c_idx = (j * 64 + tl.arange(0, 64)).to(tl.int32)
        mask = (r_idx[None, :] >= c_idx[:, None]) & (r_idx[None, :] < s) & (c_idx[:, None] < s)
        
        p_unmasked = tl.exp(s_val * tau - l_i)
        p = tl.where(mask, p_unmasked, 0.0)
        
        dp_0 = tl.dot(dO_i_0, V_j_0.T)
        dp_1 = tl.dot(dO_i_1, V_j_1.T)
        dp = dp_0 + dp_1
        
        ds = p * (dp - d_i) * tau
        
        ds_bf16 = ds.to(tl.bfloat16)
        
        dq_0 += tl.dot(ds_bf16, K_j_0)
        dq_1 += tl.dot(ds_bf16, K_j_1)

    q_c.store(dQ_ptr, [batch_head_idx * s + pid_m * 64, 0], dq_0.to(tl.bfloat16))
    q_c.store(dQ_ptr, [batch_head_idx * s + pid_m * 64, 64], dq_1.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    b, h, s, d = Q.shape
    n_elements = b * h * s
    
    if s == 0:
        return

    D_buf = torch.empty((b, h, s), dtype=torch.float32, device=Q.device)
    
    grid_pre = (n_elements,)
    _preprocess_kernel[grid_pre](dO, O, D_buf, n_elements, d, BLOCK_D=128)
    
    q_c = make_desc(Q.view(b * h * s, 128), [64, 64])
    k_c = make_desc(K.view(b * h * s, 128), [64, 64])
    v_c = make_desc(V.view(b * h * s, 128), [64, 64])
    o_c = make_desc(O.view(b * h * s, 128), [64, 64])
    do_c = make_desc(dO.view(b * h * s, 128), [64, 64])
    
    l_c = make_desc(L.view(b * h * s), [64])
    d_c = make_desc(D_buf.view(b * h * s), [64])
    
    num_sms = torch.cuda.get_device_properties(Q.device).multi_processor_count
    tau = 1.0 / (d ** 0.5)
    
    grid_KV = (triton.cdiv(s, 64), b * h)
    _bwd_dKdV_kernel[grid_KV](
        Q, dO, L, D_buf, dK, dV,
        s, d, tau,
        NUM_SMS=num_sms,
        BLOCK_N=64,
        q_c=q_c, k_c=k_c, v_c=v_c, do_c=do_c, o_c=o_c, l_c=l_c, d_c=d_c,
        num_warps=8,
        maxnreg=128,
    )
    
    grid_Q = (triton.cdiv(s, 64), b * h)
    _bwd_dQ_kernel[grid_Q](
        K, V, L, D_buf, Q, dO, dQ,
        s, d, tau,
        NUM_SMS=num_sms,
        BLOCK_N=64,
        k_c=k_c, v_c=v_c, q_c=q_c, do_c=do_c, o_c=o_c, l_c=l_c, d_c=d_c,
        num_warps=8,
        maxnreg=128,
    )