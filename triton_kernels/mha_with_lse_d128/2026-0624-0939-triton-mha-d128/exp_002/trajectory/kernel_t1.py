import torch
import triton
import triton.language as tl


@triton.jit
def _mha_kernel(
    Q, K, V, O, LSE,
    S_len, H,
    scale,
):
    start_m = tl.program_id(0) * 64
    b = tl.program_id(1)
    h = tl.program_id(2)
    
    base = b * H * S_len * 128 + h * S_len * 128
    
    q_rows = tl.arange(0, 64)
    cols_d = tl.arange(0, 128)
    
    q_off = base + (start_m + q_rows[:, None]) * 128 + cols_d[None, :]
    mask_m = (start_m + q_rows[:, None]) < S_len
    Q_tile = tl.load(Q + q_off, mask=mask_m, other=0.0)
    
    m_local = tl.full((64,), -float('inf'), dtype=tl.float32)
    s_local = tl.full((64,), 0.0, dtype=tl.float32)
    acc = tl.zeros((64, 128), dtype=tl.float32)
    
    for start_k in range(0, S_len, 64):
        k_rows = tl.arange(0, 64)
        
        k_off = base + (start_k + k_rows[:, None]) * 128 + cols_d[None, :]
        mask_k = (start_k + k_rows[:, None]) < S_len
        
        K_tile = tl.load(K + k_off, mask=mask_k, other=0.0)
        V_tile = tl.load(V + k_off, mask=mask_k, other=0.0)
        
        S_qk = tl.dot(Q_tile, K_tile.T) * scale
        
        k_idx = start_k + k_rows
        valid_k = k_idx < S_len
        S_qk = tl.where(valid_k[None, :], S_qk, -float('inf'))
        
        m_curr = tl.max(S_qk, axis=1)
        m_new = tl.maximum(m_local, m_curr)
        
        p = tl.exp(S_qk - m_new[:, None])
        
        s_curr = tl.sum(p, axis=1)
        s_local = s_local * tl.exp(m_local - m_new) + s_curr
        
        acc = acc * tl.exp(m_local - m_new)[:, None] + tl.dot(p, V_tile)
        
        m_local = m_new
    
    valid_m = q_rows < (S_len - start_m)
    d = 1.0 / s_local
    O_tile = acc * d[:, None]
    
    o_off = base + (start_m + q_rows[:, None]) * 128 + cols_d[None, :]
    tl.store(O + o_off, O_tile, mask=valid_m[:, None])
    
    lse = m_local + tl.log(s_local)
    lse_off = b * H * S_len + h * S_len + start_m + q_rows
    tl.store(LSE + lse_off, lse, mask=valid_m)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S_len, D = Q.shape
    scale = 1.0 / (D ** 0.5)
    
    grid = (triton.cdiv(S_len, 64), B, H)
    _mha_kernel[grid](
        Q, K, V, O, LSE,
        S_len, H,
        scale,
        num_warps=4, num_stages=3,
    )