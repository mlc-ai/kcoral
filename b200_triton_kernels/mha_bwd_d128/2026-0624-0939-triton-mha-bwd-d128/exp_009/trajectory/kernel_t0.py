import math
import torch
import triton
import triton.language as tl


@triton.jit
def _precompute_D_kernel(O_ptr, dO_ptr, D_ptr, S, H, B):
    b = tl.program_id(2)
    h = tl.program_id(1)
    row_idx = tl.program_id(0) * 256 + tl.arange(0, 256)
    mask = row_idx < S
    off = (b * H + h) * S * 128 + row_idx * 128
    o_row = tl.load(O_ptr + off, mask=mask[:, None], other=0.0)
    do_row = tl.load(dO_ptr + off, mask=mask[:, None], other=0.0)
    d_val = tl.sum(o_row * do_row, axis=1)
    d_off = (b * H + h) * S + row_idx
    tl.store(D_ptr + d_off, d_val, mask=mask)


@triton.jit
def _bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, L_ptr, D_ptr, dO_ptr, dQ,
    S, H, B,
    sdpa_scale,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    col_idx = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)

    batch_head_offset = (b * H + h) * S * 128
    batch_head_lse_offset = (b * H + h) * S
    
    dtype_q = Q_ptr.dtype.element_ty
    rows_1d = tl.arange(0, BLOCK_N)
    seq_lse = col_idx * BLOCK_N + rows_1d
    
    lse = tl.load(L_ptr + batch_head_lse_offset + seq_lse, mask=seq_lse < S, other=0.0)
    d = tl.load(D_ptr + batch_head_lse_offset + seq_lse, mask=seq_lse < S, other=0.0)

    stride_seq = 128
    stride_chunk = 1
    rows = tl.arange(0, BLOCK_N)
    seq_q = col_idx * BLOCK_N + rows
    mask_q = col_idx + rows // 64 < S
    
    q_base_offs = batch_head_offset + seq_q[:, None] * stride_seq + 0 * stride_chunk
    do_base_offs = batch_head_offset + seq_q[:, None] * stride_seq + 0 * stride_chunk

    Q_tile = []
    dO_tile = []
    for chunk in range(NUM_CHUNKS):
        q_offs = q_base_offs + chunk * stride_chunk
        Q_tile.append(tl.load(Q_ptr + q_offs, mask=mask_q[:, None], other=0.0))
        do_offs = do_base_offs + chunk * stride_chunk
        dO_tile.append(tl.load(dO_ptr + do_offs, mask=mask_q[:, None], other=0.0))

    dQ_acc = [tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32) for _ in range(NUM_CHUNKS)]

    for j in range(tl.cdiv(S, BLOCK_K)):
        K_tile = []
        V_tile = []
        seq_k = j * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = col_idx + rows // 64 < S
        k_base_offs = batch_head_offset + seq_k[:, None] * stride_seq + 0 * stride_chunk
        
        for chunk in range(NUM_CHUNKS):
            k_offs = k_base_offs + chunk * stride_chunk
            K_tile.append(tl.load(K_ptr + k_offs, mask=mask_k[:, None], other=0.0))
            v_offs = k_base_offs + chunk * stride_chunk
            V_tile.append(tl.load(V_ptr + v_offs, mask=mask_k[:, None], other=0.0))

        S = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for c_idx in range(NUM_CHUNKS):
            S = tl.dot(Q_tile[c_idx], K_tile[c_idx].T, acc=S)

        S_scaled = S * sdpa_scale
        p = tl.exp(S_scaled - lse)
        
        cols_k = j * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_p = (cols_k < S) & (seq_q >= 0)
        p = p * mask_p
        
        dp = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for c_idx in range(NUM_CHUNKS):
            dp = tl.dot(dO_tile[c_idx], V_tile[c_idx].T, acc=dp)
            
        ds = p * (dp - d) * sdpa_scale

        for c_idx in range(NUM_CHUNKS):
            dQ_acc[c_idx] = tl.dot(ds, K_tile[c_idx], acc=dQ_acc[c_idx])

    out_q0 = dQ + q_base_offs
    out_q1 = dQ + q_base_offs + 64
    dq_out0 = dQ_acc[0].to(dtype_q)
    dq_out1 = dQ_acc[1].to(dtype_q)
    store_mask = (col_idx + rows // 64 < S)[..., None]
    tl.store(out_q0, dq_out0, mask=store_mask)
    tl.store(out_q1, dq_out1, mask=store_mask)


@triton.jit
def _bwd_dk_dv_kernel(
    Q_ptr, K_ptr, V_ptr, L_ptr, D_ptr, dO_ptr, dK, dV,
    S, H, B,
    sdpa_scale,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    col_idx = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)

    batch_head_offset = (b * H + h) * S * 128
    
    rows = tl.arange(0, BLOCK_K)
    seq_k = col_idx * BLOCK_K + rows
    mask_k = col_idx + rows // 64 < S
    
    stride_seq = 128
    stride_chunk = 1

    k_base_offs = batch_head_offset + seq_k[:, None] * stride_seq + 0 * stride_chunk
    K_tile = []
    V_tile = []
    for chunk in range(NUM_CHUNKS):
        k_offs = k_base_offs + chunk * stride_chunk
        K_tile.append(tl.load(K_ptr + k_offs, mask=mask_k[:, None], other=0.0))
        v_offs = k_base_offs + chunk * stride_chunk
        V_tile.append(tl.load(V_ptr + v_offs, mask=mask_k[:, None], other=0.0))

    dK_acc = [tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32) for _ in range(NUM_CHUNKS)]
    dV_acc = [tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32) for _ in range(NUM_CHUNKS)]

    for i in range(tl.cdiv(S, BLOCK_N)):
        Q_tile = []
        dO_tile = []
        q_base_offs = batch_head_offset + i * BLOCK_N * stride_seq + 0 * stride_chunk
        
        rows_1d = tl.arange(0, BLOCK_N)
        seq_lse = i * BLOCK_N + rows_1d
        lse = tl.load(L_ptr + (b * H + h) * S + seq_lse, mask=seq_lse < S, other=0.0)
        d = tl.load(D_ptr + (b * H + h) * S + seq_lse, mask=seq_lse < S, other=0.0)
        
        rows_q = tl.arange(0, BLOCK_N)
        seq_q = i * BLOCK_N + rows_q
        mask_q = i + rows_q // 64 < S

        for chunk in range(NUM_CHUNKS):
            q_offs = q_base_offs + seq_q[:, None] * stride_seq + chunk * stride_chunk
            Q_tile.append(tl.load(Q_ptr + q_offs, mask=mask_q[:, None], other=0.0))
            do_offs = q_base_offs + seq_q[:, None] * stride_seq + chunk * stride_chunk
            dO_tile.append(tl.load(dO_ptr + do_offs, mask=mask_q[:, None], other=0.0))

        S = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for c_idx in range(NUM_CHUNKS):
            S = tl.dot(Q_tile[c_idx], K_tile[c_idx].T, acc=S)

        S_scaled = S * sdpa_scale
        p = tl.exp(S_scaled - lse) 
        
        cols_k = col_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_p = (cols_k < S) & (seq_q >= 0)
        p = p * mask_p
        
        dp = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for c_idx in range(NUM_CHUNKS):
            dp = tl.dot(dO_tile[c_idx], V_tile[c_idx].T, acc=dp)

        ds = p * (dp - d) * sdpa_scale
        
        for c_idx in range(NUM_CHUNKS):
            dV_acc[c_idx] = tl.dot(p.T, dO_tile[c_idx], acc=dV_acc[c_idx])
            dK_acc[c_idx] = tl.dot(ds.T, Q_tile[c_idx], acc=dK_acc[c_idx])

    out_k0 = dK + k_base_offs
    out_k1 = dK + k_base_offs + 64
    dk_out0 = dK_acc[0].to(dtype_q)
    dk_out1 = dK_acc[1].to(dtype_q)
    store_mask_k = (col_idx + rows // 64 < S)[..., None]
    tl.store(out_k0, dk_out0, mask=store_mask_k)
    tl.store(out_k1, dk_out1, mask=store_mask_k)

    out_v0 = dV + k_base_offs
    out_v1 = dV + k_base_offs + 64
    dv_out0 = dV_acc[0].to(dtype_q)
    dv_out1 = dV_acc[1].to(dtype_q)
    tl.store(out_v0, dv_out0, mask=store_mask_k)
    tl.store(out_v1, dv_out1, mask=store_mask_k)


def precompute_D(O, dO, S, H, B):
    device = O.device
    D_ptr = torch.empty((B, H, S), dtype=torch.float32, device=device)
    grid_D = ((S + 255) // 256, H, B)
    _precompute_D_kernel[grid_D](O, dO, D_ptr, S, H, B)
    return D_ptr


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute the backward pass for multi-head attention."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    sdpa_scale = 0.08838834764831843 
    
    L = L.contiguous()
    D_ptr = precompute_D(O, dO, S, H, B)

    grid_dq = (triton.cdiv(S, 64), H, B)
    _bwd_dq_kernel[grid_dq](
        Q, K, V, L, D_ptr, dO, dQ, S, H, B, sdpa_scale,
        NUM_CHUNKS=2, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=2
    )

    grid_dk_dv = (triton.cdiv(S, 64), H, B)
    _bwd_dk_dv_kernel[grid_dk_dv](
        Q, K, V, L, D_ptr, dO, dK, dV, S, H, B, sdpa_scale,
        NUM_CHUNKS=2, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=2
    )