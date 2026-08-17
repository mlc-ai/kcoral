import math
import torch
import triton
import triton.language as tl


@triton.jit
def D_kernel(O, dO, D_buf, s_len, d, H_STRIDE, S_STRIDE):
    """
    Precompute D = rowsum(O * dO) for all batches, heads, and sequences.
    """
    idx = tl.program_id(0)
    batch_idx = idx // (H_STRIDE // (s_len * d))
    head_idx = (idx % (H_STRIDE // (s_len * d))) // (s_len * d // d)
    seq_idx = idx % s_len
    
    O_ptr = O + batch_idx * H_STRIDE + head_idx * s_len * d + seq_idx * d
    dO_ptr = dO + batch_idx * H_STRIDE + head_idx * s_len * d + seq_idx * d
    
    o_row = tl.load(O_ptr + tl.arange(0, d), mask=(tl.arange(0, d) < d), other=0)
    do_row = tl.load(dO_ptr + tl.arange(0, d), mask=(tl.arange(0, d) < d), other=0)
    
    d_val = tl.sum(o_row * do_row)
    
    D_buf[idx] = d_val


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128}, num_warps=4, num_stages=3),
    ],
    key=["S_len"],
)
@triton.jit(launch_bounds=min_ctas=1, max_ctas=2)
def bwd_dq_kernel(
    Q, K, V, O, dO, L, dQ, D_buf,
    S_len, d, H,
    H_STRIDE: tl.constexpr,
    S_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    scale: tl.constexpr,
):
    i_start = tl.program_id(0) * BLOCK_M
    head_idx = tl.program_id(1)
    batch_idx = head_idx // H
    
    row_offsets_64 = tl.arange(0, 64)
    col_offsets_64 = tl.arange(0, 64)
    
    Q_ptr = Q + batch_idx * H_STRIDE
    K_ptr = K + batch_idx * H_STRIDE
    V_ptr = V + batch_idx * H_STRIDE
    O_ptr = O + batch_idx * H_STRIDE
    dO_ptr = dO + batch_idx * H_STRIDE
    dQ_ptr = dQ + batch_idx * H_STRIDE
    L_ptr = L + batch_idx * (h * s_len)
    
    r_idx = (i_start + row_offsets_64)[:, None]
    c_idx0 = (0 + col_offsets_64)[None, :]
    c_idx1 = (64 + col_offsets_64)[None, :]
    mask0 = (r_idx < S_len) & (c_idx0 < d)
    mask1 = (r_idx < S_len) & (c_idx1 < d)
    
    q0 = tl.load(Q_ptr + i_start * S_STRIDE + 0 * 1 + row_offsets_64[:, None] * S_STRIDE + col_offsets_64[None, :] * 1, mask=mask0, other=0)
    q1 = tl.load(Q_ptr + i_start * S_STRIDE + 64 * 1 + row_offsets_64[:, None] * S_STRIDE + col_offsets_64[None, :] * 1, mask=mask1, other=0)
    
    o0 = tl.load(O_ptr + i_start * S_STRIDE + 0 * 1 + row_offsets_64[:, None] * S_STRIDE + col_offsets_64[None, :] * 1, mask=mask0, other=0)
    o1 = tl.load(O_ptr + i_start * S_STRIDE + 64 * 1 + row_offsets_64[:, None] * S_STRIDE + col_offsets_64[None, :] * 1, mask=mask1, other=0)
    
    do0 = tl.load(dO_ptr + i_start * S_STRIDE + 0 * 1 + row_offsets_64[:, None] * S_STRIDE + col_offsets_64[None, :] * 1, mask=mask0, other=0)
    do1 = tl.load(dO_ptr + i_start * S_STRIDE + 64 * 1 + row_offsets_64[:, None] * S_STRIDE + col_offsets_64[None, :] * 1, mask=mask1, other=0)
    
    d_sum = tl.sum(o0 * do0, axis=1, keep_dims=True) + tl.sum(o1 * do1, axis=1, keep_dims=True)
    
    l_val = tl.load(L_ptr + i_start + row_offsets_64, mask=(i_start + row_offsets_64 < S_len), other=0.0)
    
    D_i = D_buf + batch_idx * s_len + i_start + row_offsets_64
    d_val = tl.load(D_i, mask=(i_start + row_offsets_64 < S_len), other=0.0)
    
    dQ0_acc = tl.zeros((BLOCK_M, 64), tl.float32)
    dQ1_acc = tl.zeros((BLOCK_M, 64), tl.float32)
    
    for j_start in range(0, S_len, BLOCK_N):
        r_idx_j = (j_start + row_offsets_64)[:, None]
        mask_j0 = (r_idx_j < S_len) & (c_idx0 < d)
        mask_j1 = (r_idx_j < S_len) & (c_idx1 < d)
        
        k0 = tl.load(K_ptr + j_start * S_STRIDE + 0 * 1 + row_offsets_64[:, None] * S_STRIDE + col_offsets_64[None, :] * 1, mask=mask_j0, other=0)
        k1 = tl.load(K_ptr + j_start * S_STRIDE + 64 * 1 + row_offsets_64[:, None] * S_STRIDE + col_offsets_64[None, :] * 1, mask=mask_j1, other=0)
        
        v0 = tl.load(V_ptr + j_start * S_STRIDE + 0 * 1 + row_offsets_64[:, None] * S_STRIDE + col_offsets_64[None, :] * 1, mask=mask_j0, other=0)
        v1 = tl.load(V_ptr + j_start * S_STRIDE + 64 * 1 + row_offsets_64[:, None] * S_STRIDE + col_offsets_64[None, :] * 1, mask=mask_j1, other=0)
        
        S = tl.dot(q0, k0, layout_nhwc=True) + tl.dot(q1, k1, layout_nhwc=True)
        P = tl.exp(S * scale - l_val[:, None])
        
        dP = tl.dot(do0, v0, layout_nhwc=True) + tl.dot(do1, v1, layout_nhwc=True)
        
        dS = P * (dP - d_val) * scale
        
        dQ0_acc = tl.dot(dS, k0.T, acc=dQ0_acc, layout_nhwc=True)
        dQ1_acc = tl.dot(dS, k1.T, acc=dQ1_acc, layout_nhwc=True)
        
    if i_start < S_len:
        valid = (i_start + row_offsets_64) < S_len
        ptr_dQ0 = dQ_ptr + i_start * S_STRIDE
        ptr_dQ1 = ptr_dQ0 + 64
        tl.store(ptr_dQ0 + row_offsets_64[:, None] * S_STRIDE + col_offsets_64[None, :] * 1, dQ0_acc.to(tl.bfloat16), mask=valid[:, None])
        tl.store(ptr_dQ1 + row_offsets_64[:, None] * S_STRIDE + col_offsets_64[None, :] * 1, dQ1_acc.to(tl.bfloat16), mask=valid[:, None])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128}, num_warps=4, num_stages=3),
    ],
    key=["S_len"],
)
@triton.jit(launch_bounds=min_ctas=1, max_ctas=2)
def bwd_dkv_kernel(
    Q, K, V, O, dO, L, dK, dV, D_buf,
    S_len, d, H,
    H_STRIDE: tl.constexpr,
    S_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    scale: tl.constexpr,
):
    j_start = tl.program_id(0) * BLOCK_N
    head_idx = tl.program_id(1)
    batch_idx = head_idx // H
    
    row_offsets_64 = tl.arange(0, 64)
    col_offsets_64 = tl.arange(0, 64)
    
    Q_ptr = Q + batch_idx * H_STRIDE
    K_ptr = K + batch_idx * H_STRIDE
    V_ptr = V + batch_idx * H_STRIDE
    O_ptr = O + batch_idx * H_STRIDE
    dO_ptr = dO + batch_idx * H_STRIDE
    dK_ptr = dK + batch_idx * H_STRIDE
    dV_ptr = dV + batch_idx * H_STRIDE
    L_ptr = L + batch_idx * (h * s_len)
    
    r_idx_j = (j_start + row_offsets_64)[:, None]
    c_idx0 = (0 + col_offsets_64)[None, :]
    c_idx1 = (64 + col_offsets_64)[None, :]
    mask_j0 = (r_idx_j < S_len) & (c_idx0 < d)
    mask_j1 = (r_idx_j < S_len) & (c_idx1 < d)
    
    k0 = tl.load(K_ptr + j_start * S_STRIDE + 0 * 1 + row_offsets_64[:, None] * S_STRIDE + col_offsets_64[None, :] * 1, mask=mask_j0, other=0)
    k1 = tl.load(K_ptr + j_start * S_STRIDE + 64 * 1 + row_offsets_64[:, None] * S_STRIDE + col_offsets_64[None, :] * 1, mask=mask_j1, other=0)
    
    v0 = tl.load(V_ptr + j_start * S_STRIDE + 0 * 1 + row_offsets_64[:, None] * S_STRIDE + col_offsets_64[None, :] * 1, mask=mask_j0, other=0)
    v1 = tl.load(V_ptr + j_start * S_STRIDE + 64 * 1 + row_offsets_64[:, None] * S_STRIDE + col_offsets_64[None, :] * 1, mask=mask_j1, other=0)
    
    dK0_acc = tl.zeros((BLOCK_N, 64), tl.float32)
    dK1_acc = tl.zeros((BLOCK_N, 64), tl.float32)
    dV0_acc = tl.zeros((BLOCK_N, 64), tl.float32)
    dV1_acc = tl.zeros((BLOCK_N, 64), tl.float32)
    
    for i_start in range(0, S_len, BLOCK_M):
        r_idx_i = (i_start + row_offsets_64)[:, None]
        mask_i0 = (r_idx_i < S_len) & (c_idx0 < d)
        mask_i1 = (r_idx_i < S_len) & (c_idx1 < d)
        
        q0 = tl.load(Q_ptr + i_start * S_STRIDE + 0 * 1 + row_offsets_64[:, None] * S_STRIDE + col_offsets_64[None, :] * 1, mask=mask_i0, other=0)
        q1 = tl.load(Q_ptr + i_start * S_STRIDE + 64 * 1 + row_offsets_64[:, None] * S_STRIDE + col_offsets_64[None, :] * 1, mask=mask_i1, other=0)
        
        o0 = tl.load(O_ptr + i_start * S_STRIDE + 0 * 1 + row_offsets_64[:, None] * S_STRIDE + col_offsets_64[None, :] * 1, mask=mask_i0, other=0)
        o1 = tl.load(O_ptr + i_start * S_STRIDE + 64 * 1 + row_offsets_64[:, None] * S_STRIDE + col_offsets_64[None, :] * 1, mask=mask_i1, other=0)
        
        do0 = tl.load(dO_ptr + i_start * S_STRIDE + 0 * 1 + row_offsets_64[:, None] * S_STRIDE + col_offsets_64[None, :] * 1, mask=mask_i0, other=0)
        do1 = tl.load(dO_ptr + i_start * S_STRIDE + 64 * 1 + row_offsets_64[:, None] * S_STRIDE + col_offsets_64[None, :] * 1, mask=mask_i1, other=0)
        
        d_sum = tl.sum(o0 * do0, axis=1, keep_dims=True) + tl.sum(o1 * do1, axis=1, keep_dims=True)
        
        l_val = tl.load(L_ptr + i_start + row_offsets_64, mask=(i_start + row_offsets_64 < S_len), other=0.0)
        
        D_i = D_buf + batch_idx * s_len + i_start + row_offsets_64
        d_val = tl.load(D_i, mask=(i_start + row_offsets_64 < S_len), other=0.0)
        
        S = tl.dot(q0, k0, layout_nhwc=True) + tl.dot(q1, k1, layout_nhwc=True)
        P = tl.exp(S * scale - l_val[:, None])
        
        dP = tl.dot(do0, v0, layout_nhwc=True) + tl.dot(do1, v1, layout_nhwc=True)
        
        dS = P * (dP - d_val) * scale
        
        dK0_acc = tl.dot(dS.T, q0.T, acc=dK0_acc, layout_nhwc=True)
        dK1_acc = tl.dot(dS.T, q1.T, acc=dK1_acc, layout_nhwc=True)
        
        dV0_acc = tl.dot(P.T, do0.T, acc=dV0_acc, layout_nhwc=True)
        dV1_acc = tl.dot(P.T, do1.T, acc=dV1_acc, layout_nhwc=True)
        
    if j_start < S_len:
        valid = (j_start + row_offsets_64) < S_len
        ptr_dK0 = dK_ptr + j_start * S_STRIDE
        ptr_dK1 = ptr_dK0 + 64
        tl.store(ptr_dK0 + row_offsets_64[:, None] * S_STRIDE + col_offsets_64[None, :] * 1, dK0_acc.to(tl.bfloat16), mask=valid[:, None])
        tl.store(ptr_dK1 + row_offsets_64[:, None] * S_STRIDE + col_offsets_64[None, :] * 1, dK1_acc.to(tl.bfloat16), mask=valid[:, None])
        
        ptr_dV0 = dV_ptr + j_start * S_STRIDE
        ptr_dV1 = ptr_dV0 + 64
        tl.store(ptr_dV0 + row_offsets_64[:, None] * S_STRIDE + col_offsets_64[None, :] * 1, dV0_acc.to(tl.bfloat16), mask=valid[:, None])
        tl.store(ptr_dV1 + row_offsets_64[:, None] * S_STRIDE + col_offsets_64[None, :] * 1, dV1_acc.to(tl.bfloat16), mask=valid[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    b, h, s_len, d = Q.shape
    device = Q.device
    
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)
    triton.set_allocator(alloc_fn)
    
    D_buf = torch.empty(b * h * s_len, dtype=torch.float32, device=device)
    
    grid_D = (b * h * s_len,)
    D_kernel[grid_D](
        O, dO, D_buf, s_len, d, h * s_len * d, d,
        num_warps=1,
    )
    
    scale = 1.0 / math.sqrt(d)
    
    grid = (triton.cdiv(s_len, 64), b * h)
    
    bwd_dq_kernel[grid](
        Q, K, V, O, dO, L, dQ, D_buf,
        s_len, d, h,
        H_STRIDE = h * s_len * d,
        S_STRIDE = d,
        scale=scale,
    )
    
    bwd_dkv_kernel[grid](
        Q, K, V, O, dO, L, dK, dV, D_buf,
        s_len, d, h,
        H_STRIDE = h * s_len * d,
        S_STRIDE = d,
        scale=scale,
    )