import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attention(
    Q, K, V, O, LSE,
    S_len,
    stride_bq, stride_hq, stride_sq, stride_dq,
    stride_bk, stride_hk, stride_sk, stride_dk,
    stride_bv, stride_hv, stride_sv, stride_dv,
    stride_bo, stride_ho, stride_so, stride_do,
    stride_bl, stride_hl, stride_sl,
    BLOCK_S: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
):
    bh = tl.program_id(0)
    block_row = tl.program_id(1)
    
    i = tl.arange(0, BLOCK_HEAD)
    j = tl.arange(0, BLOCK_S)
    
    scale = 1.0 / tl.sqrt(128.0)
    
    q_base_offset = bh * stride_bq + block_row * BLOCK_S * stride_sq
    q0_ptr = Q + q_base_offset
    q1_ptr = Q + q_base_offset + BLOCK_HEAD * stride_dq
    
    mask_q = ((block_row * BLOCK_S + j[:, None]) < S_len) & (i[None, :] < 128)
    q0 = tl.load(q0_ptr + j*stride_sq[:, None] + i*stride_dq[None, :], mask=mask_q, other=0.0)
    q1 = tl.load(q1_ptr + j*stride_sq[:, None] + i*stride_dq[None, :], mask=mask_q, other=0.0)
    
    acc_o0 = tl.zeros((BLOCK_S, BLOCK_HEAD), dtype=tl.float32)
    acc_o1 = tl.zeros((BLOCK_S, BLOCK_HEAD), dtype=tl.float32)
    
    curr_max = tl.full((BLOCK_S,), -1e20, dtype=tl.float32)
    curr_sum = tl.full((BLOCK_S,), 0.0, dtype=tl.float32)
    
    for kb in range(0, block_row + 1):
        k_base_offset = bh * stride_bk + kb * BLOCK_S * stride_sk
        k0_ptr = K + k_base_offset
        k1_ptr = K + k_base_offset + BLOCK_HEAD * stride_dk
        
        mask_k = ((kb * BLOCK_S + j[:, None]) < S_len) & (i[None, :] < 128)
        k0 = tl.load(k0_ptr + j*stride_sk[:, None] + i*stride_dk[None, :], mask=mask_k, other=0.0)
        k1 = tl.load(k1_ptr + j*stride_sk[:, None] + i*stride_dk[None, :], mask=mask_k, other=0.0)
        
        p = tl.dot(q0, k0)
        p = tl.dot(q1, k1, p)
        p *= scale
        
        causal_mask = (kb * BLOCK_S + i[None, :]) <= (block_row * BLOCK_S + j[:, None])
        p = tl.where(causal_mask, p, -1e20)
        
        block_max = tl.max(p, axis=1)
        prev_max = curr_max
        curr_max = tl.maximum(curr_max, block_max)
        
        curr_sum = curr_sum * tl.exp2(prev_max - curr_max)
        
        p_exp = tl.exp2(p - curr_max[:, None])
        curr_sum += tl.sum(p_exp, axis=1)
        
        acc_o0 *= tl.exp2(prev_max[:, None] - curr_max[:, None])
        acc_o1 *= tl.exp2(prev_max[:, None] - curr_max[:, None])
        
        v_base_offset = bh * stride_bv + kb * BLOCK_S * stride_sv
        v0_ptr = V + v_base_offset
        v1_ptr = V + v_base_offset + BLOCK_HEAD * stride_dv
        
        mask_v = ((kb * BLOCK_S + j[None, :]) < S_len) & (i[:, None] < 128)
        v0_T = tl.load(v0_ptr + i*stride_dv[:, None] + j*stride_sv[None, :], mask=mask_v, other=0.0)
        v1_T = tl.load(v1_ptr + i*stride_dv[:, None] + j*stride_sv[None, :], mask=mask_v, other=0.0)
        
        acc_o0 = tl.dot(p_exp, v0_T, acc_o0)
        acc_o1 = tl.dot(p_exp, v1_T, acc_o1)
    
    acc_o0 /= curr_sum[:, None]
    acc_o1 /= curr_sum[:, None]
    
    lse = curr_max + tl.log2(curr_sum)
    
    out_mask_row = (block_row * BLOCK_S + j[:, None]) < S_len
    out_mask_col0 = i[None, :] < BLOCK_HEAD
    out_mask_col1 = (i[None, :] + BLOCK_HEAD) < 128
    
    o_base_offset = bh * stride_bo + block_row * BLOCK_S * stride_so
    o0_ptr = O + o_base_offset
    o1_ptr = O + o_base_offset + BLOCK_HEAD * stride_do
    
    tl.store(o0_ptr + j*stride_so[:, None] + i*stride_do[None, :], acc_o0.to(tl.bfloat16), mask=out_mask_row & out_mask_col0)
    tl.store(o1_ptr + j*stride_so[:, None] + i*stride_do[None, :], acc_o1.to(tl.bfloat16), mask=out_mask_row & out_mask_col1)
    
    if LSE is not None:
        lse_base_offset = bh * stride_bl
        lse_ptr = LSE + lse_base_offset
        tl.store(lse_ptr + j*stride_sl, lse.to(tl.float32), mask=(block_row * BLOCK_S + j) < S_len)


def run(Q, K, V, O, LSE):
    """Compute causal multi-head attention with Log-Sum-Exp."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    assert K.shape == (B, H, S, D)
    assert V.shape == (B, H, S, D)
    assert O.shape == (B, H, S, D)
    assert LSE.shape == (B, H, S)
    
    BLOCK_S = 64
    BLOCK_HEAD = 64
    
    sbq, shq, ssq, sdq = Q.stride()
    sbk, shk, ssk, sdk = K.stride()
    sbv, shv, ssv, sdv = V.stride()
    sbo, sho, sso, sdo = O.stride()
    sbl, shl, ssl = LSE.stride()
    
    grid = (B * H, triton.cdiv(S, BLOCK_S))
    
    _flash_attention[grid](
        Q, K, V, O, LSE,
        S,
        sbq, shq, ssq, sdq,
        sbk, shk, ssk, sdk,
        sbv, shv, ssv, sdv,
        sbo, sho, sso, sdo,
        sbl, shl, ssl,
        BLOCK_S=BLOCK_S,
        BLOCK_HEAD=BLOCK_HEAD,
        num_warps=4,
        num_stages=3,
    )