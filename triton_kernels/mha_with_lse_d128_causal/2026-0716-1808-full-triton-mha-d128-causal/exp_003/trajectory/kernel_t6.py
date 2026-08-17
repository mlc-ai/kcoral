import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attention(
    Q, K, V, O, LSE,
    S_len,
    stride_hq, stride_sq, stride_dq,
    stride_hk, stride_sk, stride_dk,
    stride_hv, stride_sv, stride_dv,
    stride_ho, stride_so, stride_do,
    stride_hl, stride_sl,
):
    bh = tl.program_id(0)
    block_row = tl.program_id(1)
    
    row_c = tl.arange(0, 128)
    col_c = tl.arange(0, 64)
    
    scale = 1.0 / (8.0 * 1.4142135623730951) 
    
    q_base = Q + bh * stride_hq + block_row * 128 * stride_sq
    q0 = tl.load(q_base + row_c[:, None] * stride_sq + col_c[None, :] * stride_dq, mask=(block_row * 128 + row_c[:, None]) < S_len, other=0.0)
    q1 = tl.load(q_base + row_c[:, None] * stride_sq + (col_c[None, :] + 64) * stride_dq, mask=(block_row * 128 + row_c[:, None]) < S_len, other=0.0)
    
    acc_o0 = tl.zeros((128, 64), dtype=tl.float32)
    acc_o1 = tl.zeros((128, 64), dtype=tl.float32)
    
    curr_max = tl.full((128,), -1e20, dtype=tl.float32)
    curr_sum = tl.full((128,), 0.0, dtype=tl.float32)
    
    total_k_iters = triton.cdiv(S_len, 64)
    
    for kb in range(0, total_k_iters):
        k_base = K + bh * stride_hk + kb * 64 * stride_sk
        k0 = tl.load(k_base + col_c[:, None] * stride_sk + col_c[None, :] * stride_dk, mask=(kb * 64 + col_c[:, None]) < S_len, other=0.0)
        k1 = tl.load(k_base + col_c[:, None] * stride_sk + (col_c[None, :] + 64) * stride_dk, mask=(kb * 64 + col_c[:, None]) < S_len, other=0.0)
        
        acc_p = tl.dot(q0, k0.T)
        acc_p = tl.dot(q1, k1.T, acc_p)
        
        p = acc_p * scale
        
        valid = (kb * 64 + col_c[None, :]) <= (block_row * 128 + row_c[:, None])
        p = tl.where(valid, p, -1e20)
        
        block_max = tl.max(p, axis=1)
        prev_max = curr_max
        curr_max = tl.maximum(curr_max, block_max)
        
        curr_sum = curr_sum * tl.exp(prev_max - curr_max)
        
        p_exp = tl.exp(p - curr_max[:, None])
        curr_sum += tl.sum(p_exp, axis=1)
        
        acc_o0 *= tl.exp(prev_max[:, None] - curr_max[:, None])
        acc_o1 *= tl.exp(prev_max[:, None] - curr_max[:, None])
        
        v_base = V + bh * stride_hv + kb * 64 * stride_sv
        v0 = tl.load(v_base + col_c[:, None] * stride_sv + col_c[None, :] * stride_dv, mask=(kb * 64 + col_c[:, None]) < S_len, other=0.0)
        v1 = tl.load(v_base + col_c[:, None] * stride_sv + (col_c[None, :] + 64) * stride_dv, mask=(kb * 64 + col_c[:, None]) < S_len, other=0.0)
        
        acc_o0 = tl.dot(p_exp.to(tl.bfloat16), v0, acc_o0)
        acc_o1 = tl.dot(p_exp.to(tl.bfloat16), v1, acc_o1)
        
    out0 = acc_o0 / curr_sum[:, None]
    out1 = acc_o1 / curr_sum[:, None]
    
    o_base = O + bh * stride_ho + block_row * 128 * stride_so
    tl.store(o_base + row_c[:, None] * stride_so + col_c[None, :] * stride_do, out0.to(tl.bfloat16), mask=(block_row * 128 + row_c[:, None]) < S_len)
    tl.store(o_base + row_c[:, None] * stride_so + (col_c[None, :] + 64) * stride_do, out1.to(tl.bfloat16), mask=(block_row * 128 + row_c[:, None]) < S_len)
    
    lse = curr_max + tl.log(curr_sum)
    
    lse_ptr = LSE + bh * stride_hl + block_row * 128 * stride_sl
    tl.store(lse_ptr + row_c, lse.to(tl.float32), mask=(block_row * 128 + row_c) < S_len)


def run(Q, K, V, O, LSE):
    """Compute causal multi-head attention with Log-Sum-Exp."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    assert K.shape == (B, H, S, D)
    assert V.shape == (B, H, S, D)
    assert O.shape == (B, H, S, D)
    assert LSE.shape == (B, H, S)
    
    sbq, shq, ssq, sdq = Q.stride()
    sbk, shk, ssk, sdk = K.stride()
    sbv, shv, ssv, sdv = V.stride()
    sbo, sho, sso, sdo = O.stride()
    sbl, shl, ssl = LSE.stride()
    
    grid = (B * H, triton.cdiv(S, 128))
    
    _flash_attention[grid](
        Q, K, V, O, LSE,
        S,
        shq, ssq, sdq,
        shk, ssk, sdk,
        shv, ssv, sdv,
        sho, sso, sdo,
        shl, ssl,
        num_warps=8,
        num_stages=3,
    )