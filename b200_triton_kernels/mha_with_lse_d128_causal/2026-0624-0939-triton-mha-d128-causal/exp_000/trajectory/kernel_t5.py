import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {},
            num_warps=4,
            num_stages=3,
            pre_hook=lambda args: {
                "D": 128,
                "BLOCK_M": 64,
                "BLOCK_N": 64
            },
            prune_configs_by="resources"
        )
    ],
    key=["B", "H", "S"]
)
@triton.jit
def _mha_fwd(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    B, H, S,
    scale,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    D_val = 128
    
    pid_m = tl.program_id(0)
    bh_idx = tl.program_id(1)
    b_idx = bh_idx // H
    h_idx = bh_idx % H
    
    start_m = pid_m * BLOCK_M
    
    batch_head_offset = b_idx * H * S * D_val + h_idx * S * D_val
    
    row_m = tl.arange(0, BLOCK_M)
    q_idx = start_m + row_m
    mask_q = (q_idx < S)[:, None]
    
    c_0 = tl.arange(0, 64)
    c_64 = tl.arange(0, 64) + 64
    
    q0 = tl.load(Q_ptr + batch_head_offset + q_idx[:, None] * D_val + c_0[None, :], mask=mask_q, other=0.0)
    q1 = tl.load(Q_ptr + batch_head_offset + q_idx[:, None] * D_val + c_64[None, :], mask=mask_q, other=0.0)
    
    o0 = tl.zeros((BLOCK_M, 64), tl.float32)
    o1 = tl.zeros((BLOCK_M, 64), tl.float32)
    running_max = tl.full((BLOCK_M,), -1e20, tl.float32)
    running_sum = tl.full((BLOCK_M,), 0.0, tl.float32)
    
    n_end = tl.minimum((start_m + BLOCK_M + BLOCK_N - 1) // BLOCK_N, (S + BLOCK_N - 1) // BLOCK_N)
    
    for n in range(n_end):
        start_n = n * BLOCK_N
        row_n = tl.arange(0, BLOCK_N)
        k_idx = start_n + row_n
        mask_k = (k_idx < S)[:, None]
        
        k0 = tl.load(K_ptr + batch_head_offset + k_idx[:, None] * D_val + c_0[None, :], mask=mask_k, other=0.0)
        k1 = tl.load(K_ptr + batch_head_offset + k_idx[:, None] * D_val + c_64[None, :], mask=mask_k, other=0.0)
        v0 = tl.load(V_ptr + batch_head_offset + k_idx[:, None] * D_val + c_0[None, :], mask=mask_k, other=0.0)
        v1 = tl.load(V_ptr + batch_head_offset + k_idx[:, None] * D_val + c_64[None, :], mask=mask_k, other=0.0)
        
        p = (tl.dot(q0, k0.T) + tl.dot(q1, k1.T)) * scale
        
        causal_mask = q_idx[:, None] >= k_idx[None, :]
        valid_mask = causal_mask & (q_idx[:, None] < S) & (k_idx[None, :] < S)
        p = tl.where(valid_mask, p, -1e20)
        
        m_local = tl.max(p, axis=1)
        new_max = tl.maximum(running_max, m_local)
        
        scale_prev = tl.math.exp(running_max - new_max)
        
        s = tl.math.exp(p - new_max)
        sum_s = tl.sum(s, axis=1)
        
        running_sum = running_sum * scale_prev + sum_s
        
        o0 = o0 * scale_prev[:, None]
        o1 = o1 * scale_prev[:, None]
        
        o0 += tl.dot(s, v0)
        o1 += tl.dot(s, v1)
        
        running_max = new_max
        
    inv_sum = 1.0 / running_sum
    o0 = o0 * inv_sum[:, None]
    o1 = o1 * inv_sum[:, None]
    
    seq_idx = start_m + row_m
    mask_o = (seq_idx < S)[:, None]
    
    tl.store(O_ptr + batch_head_offset + seq_idx[:, None] * D_val + c_0[None, :], o0, mask=mask_o)
    tl.store(O_ptr + batch_head_offset + seq_idx[:, None] * D_val + c_64[None, :], o1, mask=mask_o)
    
    lse = running_max + tl.math.log(running_sum)
    tl.store(LSE_ptr + b_idx * H * S + h_idx * S + seq_idx, lse, mask=seq_idx < S)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    grid = ((S + 63) // 64, B * H)
    scale = 1.0 / (D ** 0.5)
    
    _mha_fwd[grid](
        Q.data_ptr(), K.data_ptr(), V.data_ptr(), O.data_ptr(), LSE.data_ptr(), 
        B, H, S, scale
    )