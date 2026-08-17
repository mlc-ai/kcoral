import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _attention_kernel(
    desc_Q,
    desc_K,
    desc_V,
    O,
    LSE,
    S_len,
    stride_ho,
    stride_so,
    stride_hl,
    stride_sl,
    BLOCK_ROWS: tl.constexpr,
):
    bh = tl.program_id(0)
    block_row = tl.program_id(1)
    
    row_c = tl.arange(0, BLOCK_ROWS)
    
    q = desc_Q.load([bh, block_row * BLOCK_ROWS, 0])
    
    acc_o = tl.zeros((BLOCK_ROWS, 128), dtype=tl.float32)
    curr_max = tl.full((BLOCK_ROWS,), -1e20, dtype=tl.float32)
    curr_sum = tl.full((BLOCK_ROWS,), 0.0, dtype=tl.float32)
    
    scale = 1.0 / (8.0 * 1.4142135623730951) 
    
    for kb in tl.range(0, block_row + 1, num_stages=4):
        k = desc_K.load([bh, kb * BLOCK_ROWS, 0])
        acc_p = tl.dot(q, k.T)
        p = acc_p * scale
        
        if kb == block_row:
            valid = (kb * BLOCK_ROWS + row_c[None, :]) <= (block_row * BLOCK_ROWS + row_c[:, None])
            p = tl.where(valid, p, -1e20)
        
        new_max = tl.maximum(curr_max, tl.max(p, axis=1))
        old_scale = tl.exp(curr_max - new_max)
        curr_sum *= old_scale
        curr_max = new_max
        
        p_exp = tl.exp(p - curr_max[:, None])
        curr_sum += tl.sum(p_exp, axis=1)
        
        acc_o *= old_scale[:, None]
        
        v = desc_V.load([bh, kb * BLOCK_ROWS, 0])
        acc_o = tl.dot(p_exp, v, acc_o)
        
    out = acc_o / curr_sum[:, None]
    
    o_ptr = O + bh * stride_ho + block_row * BLOCK_ROWS * stride_so
    tl.store(o_ptr + row_c[:, None] * stride_so + tl.arange(0, 128)[None, :], out.to(tl.bfloat16), mask=(block_row * BLOCK_ROWS + row_c[:, None]) < S_len)
    
    if stride_hl is not None:
        lse = curr_max + tl.log(curr_sum)
        lse_ptr = LSE + bh * stride_hl + block_row * BLOCK_ROWS * stride_sl
        tl.store(lse_ptr + row_c, lse.to(tl.float32), mask=(block_row * BLOCK_ROWS + row_c) < S_len)


def run(Q, K, V, O, LSE):
    """Compute causal multi-head attention with Log-Sum-Exp."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    assert K.shape == (B, H, S, D)
    assert V.shape == (B, H, S, D)
    assert O.shape == (B, H, S, D)
    assert LSE.shape == (B, H, S)
    
    BLOCK_ROWS = 64
    
    Q_bh = Q.view(B * H, S, D)
    K_bh = K.view(B * H, S, D)
    V_bh = V.view(B * H, S, D)
    
    desc_Q = TensorDescriptor.from_tensor(Q_bh, [BLOCK_ROWS, 128])
    desc_K = TensorDescriptor.from_tensor(K_bh, [BLOCK_ROWS, 128])
    desc_V = TensorDescriptor.from_tensor(V_bh, [BLOCK_ROWS, 128])
    
    sbq, shq, ssq, sdq = Q.stride()
    sbk, shk, ssk, sdk = K.stride()
    sbv, shv, ssv, sdv = V.stride()
    sbo, sho, sso, sdo = O.stride()
    sbl, shl, ssl = LSE.stride()
    
    grid = (B * H, triton.cdiv(S, BLOCK_ROWS))
    
    _attention_kernel[grid](
        desc_Q, desc_K, desc_V,
        O, LSE,
        S,
        sho, sso,
        shl, ssl,
        BLOCK_ROWS=BLOCK_ROWS,
        num_warps=8,
        num_stages=4,
    )