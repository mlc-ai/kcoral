import torch
import triton
import triton.language as tl


@triton.jit
def _bwd_dKdV_kernel(
    Q_ptr, dO_ptr, L_ptr, D_ptr, dK_ptr, dV_ptr,
    s, d, tau,
    NUM_SMS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch_head_idx = tl.program_id(1)
    pid_n = tl.program_id(0)
    
    dk_0 = tl.zeros((128, 64), dtype=tl.float32)
    dk_1 = tl.zeros((128, 64), dtype=tl.float32)
    dv_0 = tl.zeros((128, 64), dtype=tl.float32)
    dv_1 = tl.zeros((128, 64), dtype=tl.float32)

    row_idx = tl.arange(0, 128)
    col_idx_0 = tl.arange(0, 64)
    col_idx_1 = tl.arange(64, 128)
    
    k_base = (batch_head_idx * s * 128 + pid_n * 128 * 128).to(tl.int64)
    K_j_0 = tl.load(K_ptr + k_base + row_idx[:, None] * 128 + col_idx_0[None, :], mask=(row_idx[:, None] < s), other=0.0)
    K_j_1 = tl.load(K_ptr + k_base + row_idx[:, None] * 128 + col_idx_1[None, :], mask=(row_idx[:, None] < s), other=0.0)
    V_j_0 = tl.load(V_ptr + k_base + row_idx[:, None] * 128 + col_idx_0[None, :], mask=(row_idx[:, None] < s), other=0.0)
    V_j_1 = tl.load(V_ptr + k_base + row_idx[:, None] * 128 + col_idx_1[None, :], mask=(row_idx[:, None] < s), other=0.0)

    num_blocks = tl.cdiv(s, 128)

    for i in range(pid_n, num_blocks):
        q_base = (batch_head_idx * s * 128 + i * 128 * 128).to(tl.int64)
        Q_i_0 = tl.load(Q_ptr + q_base + row_idx[:, None] * 128 + col_idx_0[None, :], mask=(row_idx[:, None] < s), other=0.0)
        Q_i_1 = tl.load(Q_ptr + q_base + row_idx[:, None] * 128 + col_idx_1[None, :], mask=(row_idx[:, None] < s), other=0.0)
        
        dO_i_0 = tl.load(dO_ptr + q_base + row_idx[:, None] * 128 + col_idx_0[None, :], mask=(row_idx[:, None] < s), other=0.0)
        dO_i_1 = tl.load(dO_ptr + q_base + row_idx[:, None] * 128 + col_idx_1[None, :], mask=(row_idx[:, None] < s), other=0.0)
        
        O_i_0 = tl.load(O_ptr + q_base + row_idx[:, None] * 128 + col_idx_0[None, :], mask=(row_idx[:, None] < s), other=0.0)
        O_i_1 = tl.load(O_ptr + q_base + row_idx[:, None] * 128 + col_idx_1[None, :], mask=(row_idx[:, None] < s), other=0.0)
        
        D_i = tl.sum(dO_i_0 * O_i_0 + dO_i_1 * O_i_1, axis=1)
        D_i = D_i.unsqueeze(-1)

        r = (i * 128 + row_idx).to(tl.int32)
        l_i = tl.load(L_ptr + batch_head_idx * s + r, mask=(r < s), other=0.0)
        l_i = l_i.unsqueeze(-1)
        
        s_0 = tl.dot(Q_i_0, K_j_0.T)
        s_1 = tl.dot(Q_i_1, K_j_1.T)
        s_val = s_0 + s_1
        
        r_idx = (i * 128 + row_idx).to(tl.int32)
        c_idx = (pid_n * 128 + row_idx).to(tl.int32)
        mask = (r_idx[None, :] >= c_idx[:, None]) & (r_idx[None, :] < s) & (c_idx[:, None] < s)
        
        p = tl.exp(s_val * tau - l_i)
        
        dp_0 = tl.dot(dO_i_0, V_j_0.T)
        dp_1 = tl.dot(dO_i_1, V_j_1.T)
        dp = dp_0 + dp_1
        
        ds = p * (dp - D_i) * tau
        
        dv_0 += tl.dot(p.T, dO_i_0)
        dv_1 += tl.dot(p.T, dO_i_1)
        
        dk_0 += tl.dot(ds.T, Q_i_0)
        dk_1 += tl.dot(ds.T, Q_i_1)

    out = dk_0.to(tl.bfloat16)
    tl.store(dK_ptr + batch_head_idx * s * 128 + pid_n * 128 * 128 + row_idx[:, None] * 128 + col_idx_0[None, :], out, mask=((row_idx[:, None] < s)))
    out = dk_1.to(tl.bfloat16)
    tl.store(dK_ptr + batch_head_idx * s * 128 + pid_n * 128 * 128 + row_idx[:, None] * 128 + col_idx_1[None, :], out, mask=((row_idx[:, None] < s)))
    
    out = dv_0.to(tl.bfloat16)
    tl.store(dV_ptr + batch_head_idx * s * 128 + pid_n * 128 * 128 + row_idx[:, None] * 128 + col_idx_0[None, :], out, mask=((row_idx[:, None] < s)))
    out = dv_1.to(tl.bfloat16)
    tl.store(dV_ptr + batch_head_idx * s * 128 + pid_n * 128 * 128 + row_idx[:, None] * 128 + col_idx_1[None, :], out, mask=((row_idx[:, None] < s)))


@triton.jit
def _bwd_dQ_kernel(
    K_ptr, V_ptr, L_ptr, D_ptr, Q_ptr, dO_ptr, O_ptr, dQ_ptr,
    s, d, tau,
    NUM_SMS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch_head_idx = tl.program_id(1)
    pid_m = tl.program_id(0)
    
    dq_0 = tl.zeros((128, 64), dtype=tl.float32)
    dq_1 = tl.zeros((128, 64), dtype=tl.float32)

    row_idx = tl.arange(0, 128)
    col_idx_0 = tl.arange(0, 64)
    col_idx_1 = tl.arange(64, 128)
    
    q_base = (batch_head_idx * s * 128 + pid_m * 128 * 128).to(tl.int64)
    Q_i_0 = tl.load(Q_ptr + q_base + row_idx[:, None] * 128 + col_idx_0[None, :], mask=(row_idx[:, None] < s), other=0.0)
    Q_i_1 = tl.load(Q_ptr + q_base + row_idx[:, None] * 128 + col_idx_1[None, :], mask=(row_idx[:, None] < s), other=0.0)
    
    dO_i_0 = tl.load(dO_ptr + q_base + row_idx[:, None] * 128 + col_idx_0[None, :], mask=(row_idx[:, None] < s), other=0.0)
    dO_i_1 = tl.load(dO_ptr + q_base + row_idx[:, None] * 128 + col_idx_1[None, :], mask=(row_idx[:, None] < s), other=0.0)
    
    O_i_0 = tl.load(O_ptr + q_base + row_idx[:, None] * 128 + col_idx_0[None, :], mask=(row_idx[:, None] < s), other=0.0)
    O_i_1 = tl.load(O_ptr + q_base + row_idx[:, None] * 128 + col_idx_1[None, :], mask=(row_idx[:, None] < s), other=0.0)
    
    D_i = tl.sum(dO_i_0 * O_i_0 + dO_i_1 * O_i_1, axis=1)
    D_i = D_i.unsqueeze(-1)

    r = (pid_m * 128 + row_idx).to(tl.int32)
    l_i = tl.load(L_ptr + batch_head_idx * s + r, mask=(r < s), other=0.0)
    l_i = l_i.unsqueeze(-1)

    num_blocks = tl.cdiv(s, 128)

    for j in range(0, pid_m + 1):
        k_base = (batch_head_idx * s * 128 + j * 128 * 128).to(tl.int64)
        K_j_0 = tl.load(K_ptr + k_base + row_idx[:, None] * 128 + col_idx_0[None, :], mask=(row_idx[:, None] < s), other=0.0)
        K_j_1 = tl.load(K_ptr + k_base + row_idx[:, None] * 128 + col_idx_1[None, :], mask=(row_idx[:, None] < s), other=0.0)
        
        V_j_0 = tl.load(V_ptr + k_base + row_idx[:, None] * 128 + col_idx_0[None, :], mask=(row_idx[:, None] < s), other=0.0)
        V_j_1 = tl.load(V_ptr + k_base + row_idx[:, None] * 128 + col_idx_1[None, :], mask=(row_idx[:, None] < s), other=0.0)
        
        s_0 = tl.dot(Q_i_0, K_j_0.T)
        s_1 = tl.dot(Q_i_1, K_j_1.T)
        s_val = s_0 + s_1
        
        r_idx = (pid_m * 128 + row_idx).to(tl.int32)
        c_idx = (j * 128 + row_idx).to(tl.int32)
        mask = (r_idx[None, :] >= c_idx[:, None]) & (r_idx[None, :] < s) & (c_idx[:, None] < s)
        
        p = tl.exp(s_val * tau - l_i)
        
        dp_0 = tl.dot(dO_i_0, V_j_0.T)
        dp_1 = tl.dot(dO_i_1, V_j_1.T)
        dp = dp_0 + dp_1
        
        ds = p * (dp - D_i) * tau
        
        dq_0 += tl.dot(ds, K_j_0)
        dq_1 += tl.dot(ds, K_j_1)

    out = dq_0.to(tl.bfloat16)
    tl.store(dQ_ptr + batch_head_idx * s * 128 + pid_m * 128 * 128 + row_idx[:, None] * 128 + col_idx_0[None, :], out, mask=((row_idx[:, None] < s)))
    out = dq_1.to(tl.bfloat16)
    tl.store(dQ_ptr + batch_head_idx * s * 128 + pid_m * 128 * 128 + row_idx[:, None] * 128 + col_idx_1[None, :], out, mask=((row_idx[:, None] < s)))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    b, h, s, d = Q.shape
    
    num_sms = torch.cuda.get_device_properties(Q.device).multi_processor_count
    tau = 1.0 / (d ** 0.5)
    
    grid_KV = (triton.cdiv(s, 128), b * h)
    _bwd_dKdV_kernel[grid_KV](
        Q, dO, L, O, dK, dV,
        s, d, tau,
        NUM_SMS=num_sms,
        BLOCK_N=128,
        num_warps=8,
        maxnreg=128,
    )
    
    grid_Q = (triton.cdiv(s, 128), b * h)
    _bwd_dQ_kernel[grid_Q](
        K, V, L, O, Q, dO, O, dQ,
        s, d, tau,
        NUM_SMS=num_sms,
        BLOCK_N=128,
        num_warps=8,
        maxnreg=128,
    )