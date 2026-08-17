import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attention(
    Q, K, V, O, LSE,
    S_len,
    stride_hq, stride_sq, stride_hk, stride_sk, stride_hv, stride_sv,
    stride_ho, stride_so, stride_hl, stride_sl,
    BLOCK_S: tl.constexpr,
):
    bh = tl.program_id(0)
    block_row = tl.program_id(1)
    
    row = tl.arange(0, BLOCK_S)
    col = tl.arange(0, 128)
    
    scale = 1.0 / tl.sqrt(128.0)
    
    q_ptr = Q + bh * stride_hq + block_row * BLOCK_S * stride_sq
    q = tl.load(q_ptr + row[:, None] * stride_sq + col[None, :], mask=(block_row * BLOCK_S + row[:, None]) < S_len, other=0.0)
    
    out = tl.zeros((BLOCK_S, 128), dtype=tl.float32)
    curr_max = tl.full((BLOCK_S,), -1e20, dtype=tl.float32)
    curr_sum = tl.full((BLOCK_S,), 0.0, dtype=tl.float32)
    
    for kb in range(0, block_row + 1):
        k_ptr = K + bh * stride_hk + kb * BLOCK_S * stride_sk
        k = tl.load(k_ptr + row[:, None] * stride_sk + col[None, :], mask=(kb * BLOCK_S + row[:, None]) < S_len, other=0.0)
        
        p = tl.dot(q, k.T)
        p *= scale
        
        valid = (kb * BLOCK_S + col[None, :]) <= (block_row * BLOCK_S + row[:, None])
        p = tl.where(valid, p, -1e20)
        
        block_max = tl.max(p, axis=1)
        prev_max = curr_max
        curr_max = tl.maximum(curr_max, block_max)
        
        curr_sum = curr_sum * tl.exp2(prev_max - curr_max)
        
        p_exp = tl.exp2(p - curr_max[:, None])
        curr_sum += tl.sum(p_exp, axis=1)
        
        out *= tl.exp2(prev_max[:, None] - curr_max[:, None])
        
        v_ptr = V + bh * stride_hv + kb * BLOCK_S * stride_sv
        v = tl.load(v_ptr + row[:, None] * stride_sv + col[None, :], mask=(kb * BLOCK_S + row[:, None]) < S_len, other=0.0)
        
        out = tl.dot(p_exp, v, out)
    
    out = out / curr_sum[:, None]
    
    o_ptr = O + bh * stride_ho + block_row * BLOCK_S * stride_so
    tl.store(o_ptr + row[:, None] * stride_so + col[None, :], out.to(tl.bfloat16), mask=(block_row * BLOCK_S + row[:, None]) < S_len)
    
    lse = curr_max + tl.log(curr_sum)
    
    lse_ptr = LSE + bh * stride_hl + block_row * BLOCK_S * stride_sl
    tl.store(lse_ptr + row, lse.to(tl.float32), mask=(block_row * BLOCK_S + row) < S_len)


def run(Q, K, V, O, LSE):
    """Compute causal multi-head attention with Log-Sum-Exp."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    assert K.shape == (B, H, S, D)
    assert V.shape == (B, H, S, D)
    assert O.shape == (B, H, S, D)
    assert LSE.shape == (B, H, S)
    
    BLOCK_S = 128
    
    sbq, shq, ssq, sdq = Q.stride()
    sbk, shk, ssk, sdk = K.stride()
    sbv, shv, ssv, sdv = V.stride()
    sbo, sho, sso, sdo = O.stride()
    sbl, shl, ssl = LSE.stride()
    
    grid = (B * H, triton.cdiv(S, BLOCK_S))
    
    _flash_attention[grid](
        Q, K, V, O, LSE,
        S,
        shq, ssq, shk, ssk, shv, ssv,
        sho, sso, shl, ssl,
        BLOCK_S=BLOCK_S,
        num_warps=4,
        num_stages=1,
    )