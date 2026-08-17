import torch
import triton
import triton.language as tl


@triton.jit
def _mha_fwd(
    Q, K, V, O, LSE,
    B, H, S,
    scale,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    bh_idx = tl.program_id(1)
    b_idx = bh_idx // H
    h_idx = bh_idx % H
    
    start_m = pid_m * BLOCK_M
    
    batch_head_offset = b_idx * H * S * D + h_idx * S * D
    
    row_m = tl.arange(0, BLOCK_M)
    q_idx = start_m + row_m
    mask_q = (start_m + row_m < S)[:, None]
    
    col_d = tl.arange(0, D)
    
    Q_tile = tl.load(Q + batch_head_offset + q_idx[:, None] * D + col_d[None, :], mask=mask_q, other=0.0)
    
    o_acc = tl.zeros((BLOCK_M, D), tl.float32)
    running_max = tl.full((BLOCK_M,), -1e20, tl.float32)
    running_sum = tl.full((BLOCK_M,), 0.0, tl.float32)
    
    n_end = (start_m + BLOCK_M + BLOCK_N - 1) // BLOCK_N
    if n_end > (S + BLOCK_N - 1) // BLOCK_N:
        n_end = (S + BLOCK_N - 1) // BLOCK_N
    
    for n in range(n_end):
        start_n = n * BLOCK_N
        row_n = tl.arange(0, BLOCK_N)
        k_idx = start_n + row_n
        mask_k = (start_n + row_n < S)[:, None]
        
        K_tile = tl.load(K + batch_head_offset + k_idx[:, None] * D + col_d[None, :], mask=mask_k, other=0.0)
        V_tile = tl.load(V + batch_head_offset + k_idx[:, None] * D + col_d[None, :], mask=mask_k, other=0.0)
        
        p = tl.dot(Q_tile, K_tile.T) * scale
        
        causal_mask = q_idx[:, None] >= k_idx[None, :]
        valid_mask = causal_mask & (q_idx[:, None] < S) & (k_idx[None, :] < S)
        p = tl.where(valid_mask, p, -1e20)
        
        m_local = tl.max(p, axis=1)
        new_max = tl.maximum(running_max, m_local)
        
        scale_prev = tl.math.exp(running_max - new_max)
        
        s = tl.math.exp(p - new_max)
        sum_s = tl.sum(s, axis=1)
        
        running_sum = running_sum * scale_prev + sum_s
        
        o_acc = o_acc * scale_prev[:, None]
        
        o_acc += tl.dot(s, V_tile)
        
        running_max = new_max
        
    inv_sum = 1.0 / running_sum
    o_acc = o_acc * inv_sum[:, None]
    
    seq_idx = start_m + row_m
    mask_o = (seq_idx < S)[:, None]
    
    tl.store(O + batch_head_offset + seq_idx[:, None] * D + col_d[None, :], o_acc, mask=mask_o)
    
    lse = running_max + tl.math.log(running_sum)
    tl.store(LSE + b_idx * H * S + h_idx * S + seq_idx, lse, mask=seq_idx < S)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    grid = ((S + 63) // 64, B * H)
    scale = 1.0 / (D ** 0.5)
    
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    O = O.contiguous()
    LSE = LSE.contiguous()

    _mha_fwd[grid](
        Q, K, V, O, LSE, B, H, S, scale, D=D, BLOCK_M=64, BLOCK_N=64
    )