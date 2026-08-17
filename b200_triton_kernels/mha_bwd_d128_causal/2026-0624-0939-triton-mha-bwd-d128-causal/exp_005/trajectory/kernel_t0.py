import torch
import triton
import triton.language as tl


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
    d_offset: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch_head_idx = tl.program_id(1)
    num_programs_n = tl.num_programs(0)
    if (num_programs_n > NUM_SMS) and (tl.program_id(0) >= NUM_SMS):
        return

    pid_n = tl.program_id(0)
    if pid_n >= tl.cdiv(s, BLOCK_N):
        return

    dk_0 = tl.shared_memory.make_block((BLOCK_N, 64), dtype=tl.bfloat16)
    dk_1 = tl.shared_memory.make_block((BLOCK_N, 64), dtype=tl.bfloat16)
    dv_0 = tl.shared_memory.make_block((BLOCK_N, 64), dtype=tl.bfloat16)
    dv_1 = tl.shared_memory.make_block((BLOCK_N, 64), dtype=tl.bfloat16)
    dk_0.fill(0.0)
    dk_1.fill(0.0)
    dv_0.fill(0.0)
    dv_1.fill(0.0)

    col_idx = tl.arange(0, 64)
    k_base_offset = (batch_head_idx * s * d + d_offset).to(tl.int64)
    
    k_0 = tl.load(dK_ptr + k_base_offset + pid_n * BLOCK_N * d + col_idx[None, :] * d, mask=(col_idx[None, :] < s), other=0.0)
    k_1 = tl.load(dK_ptr + k_base_offset + pid_n * BLOCK_N * d + 64 + col_idx[None, :] * d, mask=(col_idx[None, :] < s), other=0.0)
    v_0 = tl.load(dV_ptr + k_base_offset + pid_n * BLOCK_N * d + col_idx[None, :] * d, mask=(col_idx[None, :] < s), other=0.0)
    v_1 = tl.load(dV_ptr + k_base_offset + pid_n * BLOCK_N * d + 64 + col_idx[None, :] * d, mask=(col_idx[None, :] < s), other=0.0)

    shared_k_0 = tl.shared_memory.make_block((BLOCK_N, 64), dtype=tl.bfloat16)
    shared_k_1 = tl.shared_memory.make_block((BLOCK_N, 64), dtype=tl.bfloat16)
    shared_v_0 = tl.shared_memory.make_block((BLOCK_N, 64), dtype=tl.bfloat16)
    shared_v_1 = tl.shared_memory.make_block((BLOCK_N, 64), dtype=tl.bfloat16)
    shared_k_0[0:64, 0:64] = k_0
    shared_k_1[0:64, 0:64] = k_1
    shared_v_0[0:64, 0:64] = v_0
    shared_v_1[0:64, 0:64] = v_1

    shared_pt = tl.shared_memory.make_block((BLOCK_N, 64), dtype=tl.bfloat16)
    shared_ds_t = tl.shared_memory.make_block((BLOCK_N, 64), dtype=tl.bfloat16)

    k_start = pid_n
    k_end = tl.cdiv(s, BLOCK_N)

    for i in range(k_start, k_end):
        q_base_offset = (batch_head_idx * s * d + d_offset).to(tl.int64)
        q_0 = tl.load(Q_ptr + q_base_offset + i * BLOCK_N * d + col_idx[None, :] * d, mask=(col_idx[None, :] < s), other=0.0)
        q_1 = tl.load(Q_ptr + q_base_offset + i * BLOCK_N * d + 64 + col_idx[None, :] * d, mask=(col_idx[None, :] < s), other=0.0)
        do_0 = tl.load(dO_ptr + q_base_offset + i * BLOCK_N * d + col_idx[None, :] * d, mask=(col_idx[None, :] < s), other=0.0)
        do_1 = tl.load(dO_ptr + q_base_offset + i * BLOCK_N * d + 64 + col_idx[None, :] * d, mask=(col_idx[None, :] < s), other=0.0)

        shared_q_0 = tl.shared_memory.make_block((BLOCK_N, 64), dtype=tl.bfloat16)
        shared_q_1 = tl.shared_memory.make_block((BLOCK_N, 64), dtype=tl.bfloat16)
        shared_do_0 = tl.shared_memory.make_block((BLOCK_N, 64), dtype=tl.bfloat16)
        shared_do_1 = tl.shared_memory.make_block((BLOCK_N, 64), dtype=tl.bfloat16)
        shared_q_0[0:64, 0:64] = q_0
        shared_q_1[0:64, 0:64] = q_1
        shared_do_0[0:64, 0:64] = do_0
        shared_do_1[0:64, 0:64] = do_1

        row_idx = i * BLOCK_N + tl.arange(0, BLOCK_N)
        l_i = tl.load(L_ptr + batch_head_idx * s + row_idx)
        l_i = l_i.unsqueeze(-1)
        dl_i = tl.load(D_ptr + batch_head_idx * s + row_idx)
        dl_i = dl_i.unsqueeze(-1)

        s_0 = tl.dot(shared_q_0, shared_k_0.T)
        s_1 = tl.dot(shared_q_1, shared_k_1.T)
        s = s_0 + s_1

        r = (i * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int32)
        c = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int32)
        mask = (r[None, :] >= c[:, None]) & (r[None, :] < s) & (c[:, None] < s)

        s_masked = s * mask + (-float('inf')) * (1 - mask)
        p = tl.exp(s_masked * tau - l_i)

        dp_0 = tl.dot(shared_do_0, shared_v_0.T)
        dp_1 = tl.dot(shared_do_1, shared_v_1.T)
        dp = dp_0 + dp_1

        ds = p * (dp - dl_i) * tau

        pt = (p.T).to(tl.bfloat16)
        shared_pt[0:64, 0:64] = pt
        ds_t = (ds.T).to(tl.bfloat16)
        shared_ds_t[0:64, 0:64] = ds_t

        dv_0 += tl.dot(shared_pt, shared_do_0)
        dv_1 += tl.dot(shared_pt, shared_do_1)
        dk_0 += tl.dot(shared_ds_t, shared_q_0)
        dk_1 += tl.dot(shared_ds_t, shared_q_1)

    row_idx = tl.arange(0, BLOCK_N)
    col_idx = tl.arange(0, 64)
    dk_out_0 = dK_ptr + batch_head_idx * s*d + pid_n * BLOCK_N * d + 0
    dk_out_1 = dK_ptr + batch_head_idx * s*d + pid_n * BLOCK_N * d + 64
    tl.store(dk_out_0 + row_idx[:, None] * d + col_idx[None, :], dk_0, mask=((row_idx[:, None] < s) & (col_idx[None, :] < s)))
    tl.store(dk_out_1 + row_idx[:, None] * d + col_idx[None, :], dk_1, mask=((row_idx[:, None] < s) & (col_idx[None, :] < s)))

    dv_out_0 = dV_ptr + batch_head_idx * s*d + pid_n * BLOCK_N * d + 0
    dv_out_1 = dV_ptr + batch_head_idx * s*d + pid_n * BLOCK_N * d + 64
    tl.store(dv_out_0 + row_idx[:, None] * d + col_idx[None, :], dv_0, mask=((row_idx[:, None] < s) & (col_idx[None, :] < s)))
    tl.store(dv_out_1 + row_idx[:, None] * d + col_idx[None, :], dv_1, mask=((row_idx[:, None] < s) & (col_idx[None, :] < s)))


@triton.jit
def _bwd_dQ_kernel(
    K_ptr, V_ptr, L_ptr, D_ptr, Q_ptr, dO_ptr, dQ_ptr,
    s, d, tau,
    d_offset: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch_head_idx = tl.program_id(1)
    num_programs_m = tl.num_programs(0)
    if (num_programs_m > NUM_SMS) and (tl.program_id(0) >= NUM_SMS):
        return

    pid_m = tl.program_id(0)
    if pid_m >= tl.cdiv(s, BLOCK_N):
        return

    dq_0 = tl.shared_memory.make_block((BLOCK_N, 64), dtype=tl.bfloat16)
    dq_1 = tl.shared_memory.make_block((BLOCK_N, 64), dtype=tl.bfloat16)
    dq_0.fill(0.0)
    dq_1.fill(0.0)

    col_idx = tl.arange(0, 64)
    q_base_offset = (batch_head_idx * s * d + d_offset).to(tl.int64)
    
    q_0 = tl.load(Q_ptr + q_base_offset + pid_m * BLOCK_N * d + col_idx[None, :] * d, mask=(col_idx[None, :] < s), other=0.0)
    q_1 = tl.load(Q_ptr + q_base_offset + pid_m * BLOCK_N * d + 64 + col_idx[None, :] * d, mask=(col_idx[None, :] < s), other=0.0)
    do_0 = tl.load(dO_ptr + q_base_offset + pid_m * BLOCK_N * d + col_idx[None, :] * d, mask=(col_idx[None, :] < s), other=0.0)
    do_1 = tl.load(dO_ptr + q_base_offset + pid_m * BLOCK_N * d + 64 + col_idx[None, :] * d, mask=(col_idx[None, :] < s), other=0.0)

    shared_q_0 = tl.shared_memory.make_block((BLOCK_N, 64), dtype=tl.bfloat16)
    shared_q_1 = tl.shared_memory.make_block((BLOCK_N, 64), dtype=tl.bfloat16)
    shared_do_0 = tl.shared_memory.make_block((BLOCK_N, 64), dtype=tl.bfloat16)
    shared_do_1 = tl.shared_memory.make_block((BLOCK_N, 64), dtype=tl.bfloat16)
    shared_q_0[0:64, 0:64] = q_0
    shared_q_1[0:64, 0:64] = q_1
    shared_do_0[0:64, 0:64] = do_0
    shared_do_1[0:64, 0:64] = do_1

    shared_ds = tl.shared_memory.make_block((BLOCK_N, 64), dtype=tl.bfloat16)

    row_idx = pid_m * BLOCK_N + tl.arange(0, BLOCK_N)
    l_i = tl.load(L_ptr + batch_head_idx * s + row_idx)
    l_i = l_i.unsqueeze(-1)
    dl_i = tl.load(D_ptr + batch_head_idx * s + row_idx)
    dl_i = dl_i.unsqueeze(-1)

    for j in range(0, pid_m + 1):
        k_base_offset = (batch_head_idx * s * d + d_offset).to(tl.int64)
        k_0 = tl.load(K_ptr + k_base_offset + j * BLOCK_N * d + col_idx[None, :] * d, mask=(col_idx[None, :] < s), other=0.0)
        k_1 = tl.load(K_ptr + k_base_offset + j * BLOCK_N * d + 64 + col_idx[None, :] * d, mask=(col_idx[None, :] < s), other=0.0)
        v_0 = tl.load(V_ptr + k_base_offset + j * BLOCK_N * d + col_idx[None, :] * d, mask=(col_idx[None, :] < s), other=0.0)
        v_1 = tl.load(V_ptr + k_base_offset + j * BLOCK_N * d + 64 + col_idx[None, :] * d, mask=(col_idx[None, :] < s), other=0.0)

        shared_k_0 = tl.shared_memory.make_block((BLOCK_N, 64), dtype=tl.bfloat16)
        shared_k_1 = tl.shared_memory.make_block((BLOCK_N, 64), dtype=tl.bfloat16)
        shared_v_0 = tl.shared_memory.make_block((BLOCK_N, 64), dtype=tl.bfloat16)
        shared_v_1 = tl.shared_memory.make_block((BLOCK_N, 64), dtype=tl.bfloat16)
        shared_k_0[0:64, 0:64] = k_0
        shared_k_1[0:64, 0:64] = k_1
        shared_v_0[0:64, 0:64] = v_0
        shared_v_1[0:64, 0:64] = v_1

        s_0 = tl.dot(shared_q_0, shared_k_0.T)
        s_1 = tl.dot(shared_q_1, shared_k_1.T)
        s = s_0 + s_1

        r = (pid_m * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int32)
        c = (j * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int32)
        mask = (r[None, :] >= c[:, None]) & (r[None, :] < s) & (c[:, None] < s)

        s_masked = s * mask + (-float('inf')) * (1 - mask)
        p = tl.exp(s_masked * tau - l_i)

        dp_0 = tl.dot(shared_do_0, shared_v_0.T)
        dp_1 = tl.dot(shared_do_1, shared_v_1.T)
        dp = dp_0 + dp_1

        ds = p * (dp - dl_i) * tau

        ds_bf16 = ds.to(tl.bfloat16)
        shared_ds[0:64, 0:64] = ds_bf16

        dq_0 += tl.dot(shared_ds, shared_k_0)
        dq_1 += tl.dot(shared_ds, shared_k_1)

    row_idx = tl.arange(0, BLOCK_N)
    col_idx = tl.arange(0, 64)
    dq_out_0 = dQ_ptr + batch_head_idx * s*d + pid_m * BLOCK_N * d + 0
    dq_out_1 = dQ_ptr + batch_head_idx * s*d + pid_m * BLOCK_N * d + 64
    tl.store(dq_out_0 + row_idx[:, None] * d + col_idx[None, :], dq_0, mask=((row_idx[:, None] < s) & (col_idx[None, :] < s)))
    tl.store(dq_out_1 + row_idx[:, None] * d + col_idx[None, :], dq_1, mask=((row_idx[:, None] < s) & (col_idx[None, :] < s)))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    b, h, s, d = Q.shape
    n_elements = b * h * s
    
    if s == 0:
        return

    D_buf = torch.empty((b, h, s), dtype=torch.float32, device=Q.device)
    
    grid_pre = (n_elements,)
    _preprocess_kernel[grid_pre](dO, O, D_buf, n_elements, d, BLOCK_D=128)
    
    num_sms = torch.cuda.get_device_properties(Q.device).multi_processor_count
    tau = 1.0 / (d ** 0.5)
    
    grid_KV = (min(num_sms, triton.cdiv(s, 64)), b * h)
    _bwd_dKdV_kernel[grid_KV](
        Q, dO, L, D_buf, dK, dV,
        s, d, tau,
        d_offset=0,
        NUM_SMS=num_sms,
        BLOCK_N=64,
        num_warps=8,
        num_stages=3,
    )
    
    grid_Q = (min(num_sms, triton.cdiv(s, 64)), b * h)
    _bwd_dQ_kernel[grid_Q](
        K, V, L, D_buf, Q, dO, dQ,
        s, d, tau,
        d_offset=0,
        NUM_SMS=num_sms,
        BLOCK_N=64,
        num_warps=8,
        num_stages=3,
    )