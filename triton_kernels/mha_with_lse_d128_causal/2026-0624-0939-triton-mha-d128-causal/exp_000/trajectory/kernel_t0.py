import torch
import triton
import triton.language as tl


@triton.jit
def _mha_fwd(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S, H,
    scale,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    bid = tl.program_id(0)
    b_idx = bid // (H * (S // BLOCK_M))
    h_idx = (bid // (S // BLOCK_M)) % H
    start_m = (bid % (S // BLOCK_M)) * BLOCK_M
    
    base = Q_ptr + b_idx * H * S * D + h_idx * S * D
    
    row = tl.arange(0, BLOCK_M)
    col_left = tl.arange(0, 64)
    col_right = tl.arange(0, 64) + 64
    
    q_left = tl.load(base + (start_m + row)[:, None] * D + col_left[None, :], mask=(start_m + row)[:, None] < S, other=0.0)
    q_right = tl.load(base + (start_m + row)[:, None] * D + col_right[None, :], mask=(start_m + row)[:, None] < S, other=0.0)
    
    o_left = tl.zeros((BLOCK_M, 64), tl.float32)
    o_right = tl.zeros((BLOCK_M, 64), tl.float32)
    running_max = tl.full((BLOCK_M,), -1e20, tl.float32)
    running_sum = tl.full((BLOCK_M,), 0.0, tl.float32)
    
    query_seq_ids = start_m + row
    
    with triton.persistent_buffer(k_left):
        with triton.persistent_buffer(k_right):
            with triton.persistent_buffer(v_left):
                with triton.persistent_buffer(v_right):
                    for n_idx in range(0, (start_m // BLOCK_N) + 1):
                        if n_idx * BLOCK_N >= S:
                            break
                        
                        k_left = tl.load(base + (n_idx * BLOCK_N + row)[:, None] * D + col_left[None, :], mask=(n_idx * BLOCK_N + row)[:, None] < S, other=0.0)
                        k_right = tl.load(base + (n_idx * BLOCK_N + row)[:, None] * D + col_right[None, :], mask=(n_idx * BLOCK_N + row)[:, None] < S, other=0.0)
                        v_left = tl.load(base + (n_idx * BLOCK_N + row)[:, None] * D + col_left[None, :], mask=(n_idx * BLOCK_N + row)[:, None] < S, other=0.0)
                        v_right = tl.load(base + (n_idx * BLOCK_N + row)[:, None] * D + col_right[None, :], mask=(n_idx * BLOCK_N + row)[:, None] < S, other=0.0)
                        
                        if n_idx == 0:
                            o_left = tl.zeros((BLOCK_M, 64), tl.float32)
                            o_right = tl.zeros((BLOCK_M, 64), tl.float32)
                        
                        key_seq_ids = n_idx * BLOCK_N + row
                        
                        p = tl.dot(q_left, k_left.T) + tl.dot(q_right, k_right.T)
                        p = p * scale
                        
                        causal_mask = query_seq_ids[:, None] >= key_seq_ids[None, :]
                        mask = causal_mask & (query_seq_ids[:, None] < S) & (key_seq_ids[None, :] < S)
                        p = tl.where(mask, p, -1e20)
                        
                        m_local = tl.max(p, axis=1)
                        new_max = tl.maximum(running_max, m_local)
                        
                        exp_scale = tl.math.exp(p - new_max)
                        sum_local = tl.sum(exp_scale, axis=1)
                        
                        running_sum = running_sum * tl.math.exp(running_max - new_max) + sum_local
                        running_max = new_max
                        
                        inv_sum = 1.0 / sum_local
                        s = exp_scale * inv_sum[:, None]
                        
                        o_left = o_left * tl.math.exp(running_max - new_max)[:, None]
                        o_right = o_right * tl.math.exp(running_max - new_max)[:, None]
                        
                        o_left += tl.dot(s, v_left)
                        o_right += tl.dot(s, v_right)
    
    seq_idx = start_m + row
    tl.store(O_ptr + b_idx * H * S * D + h_idx * S * D + seq_idx[:, None] * D + col_left[None, :], o_left, mask=seq_idx[:, None] < S)
    tl.store(O_ptr + b_idx * H * S * D + h_idx * S * D + seq_idx[:, None] * D + col_right[None, :], o_right, mask=seq_idx[:, None] < S)
    
    lse = running_max + tl.math.log(running_sum)
    tl.store(LSE_ptr + b_idx * H * S + h_idx * S + seq_idx, lse, mask=seq_idx < S)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    grid = ((S + 31) // 32, H, B)
    scale = 1.0 / (D ** 0.5)
    _mha_fwd[grid](
        Q, K, V, O, LSE, S, H, scale, D=D, BLOCK_M=32, BLOCK_N=32,
        num_warps=4, num_stages=4
    )