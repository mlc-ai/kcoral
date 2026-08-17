import torch
import triton
import triton.language as tl
import math
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _attention_kernel(
    q_desc, k_desc, v_desc, o_desc,
    LSE, total_rows, S_length, lse_stride,
    BM: tl.constexpr, BN: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    bh = pid_bh
    S = S_length
    q_start_row = bh * S + pid_m * BM
    
    q0 = q_desc.load([q_start_row, 0])
    q1 = q_desc.load([q_start_row, 64])
    
    acc_o0 = tl.zeros((BM, BN), tl.float32)
    acc_o1 = tl.zeros((BM, BN), tl.float32)
    m = tl.full((BM,), float('-inf'), tl.float32)
    sum_p = tl.zeros((BM,), tl.float32)
    
    num_k_tiles = S // BN
    
    scale = 1.0 / math.sqrt(128.0)
    
    for kv_idx in range(num_k_tiles):
        k_start_row = bh * S + kv_idx * BN
        
        k0 = k_desc.load([k_start_row, 0])
        k1 = k_desc.load([k_start_row, 64])
        v0 = v_desc.load([k_start_row, 0])
        v1 = v_desc.load([k_start_row, 64])
        
        s = tl.dot(q0, k0.T, out_dtype=tl.float32) + tl.dot(q1, k1.T, out_dtype=tl.float32)
        s *= scale
        
        cols = tl.arange(0, BN)
        col_valid = kv_idx * BN + cols < S_length
        s = tl.where(col_valid[None, :], s, float('-inf'))
        
        m_new = tl.maximum(m, tl.max(s, axis=1))
        alpha = tl.exp(m - m_new)
        p = tl.exp(s - m_new[:, None])
        
        sum_p = sum_p * alpha + tl.sum(p, axis=1)
        m = m_new
        
        acc_o0 = acc_o0 * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v0, out_dtype=tl.float32)
        acc_o1 = acc_o1 * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v1, out_dtype=tl.float32)
        
    denom = sum_p
    
    acc_o0 = acc_o0 / denom[:, None]
    acc_o1 = acc_o1 / denom[:, None]
    
    o_desc.store([q_start_row, 0], acc_o0.to(tl.bfloat16))
    o_desc.store([q_start_row, 64], acc_o1.to(tl.bfloat16))
    
    rows = tl.arange(0, BM)
    lse = m + tl.log(denom)
    
    lse_ptr_curr = LSE + q_start_row + rows
    row_valid = q_start_row + rows < (bh + 1) * S
    tl.store(lse_ptr_curr, lse, mask=row_valid)


def run(Q, K, V, O, LSE):
    """Compute Non-causal multi-head attention forward returning O and LSE."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    if S == 0:
        return
    
    Q = Q.reshape(B * H * S, D).contiguous()
    K = K.reshape(B * H * S, D).contiguous()
    V = V.reshape(B * H * S, D).contiguous()
    O = O.reshape(B * H * S, D).contiguous()
    
    total_rows = B * H * S
    
    BLOCK_M, BLOCK_N = 64, 64
    
    q_desc = TensorDescriptor.from_tensor(Q, [BLOCK_M, 64])
    k_desc = TensorDescriptor.from_tensor(K, [BLOCK_N, 64])
    v_desc = TensorDescriptor.from_tensor(V, [BLOCK_N, 64])
    o_desc = TensorDescriptor.from_tensor(O, [BLOCK_M, 64])
    
    grid = (triton.cdiv(S, BLOCK_M), B * H)
    
    _attention_kernel[grid](
        q_desc, k_desc, v_desc, o_desc, LSE,
        total_rows, S, LSE.stride(0),
        BM=BLOCK_M, BN=BLOCK_N,
        num_warps=4, num_stages=5
    )