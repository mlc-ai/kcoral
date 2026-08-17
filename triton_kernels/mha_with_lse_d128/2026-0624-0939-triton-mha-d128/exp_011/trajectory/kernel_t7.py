import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_kernel(
    q_desc,
    k_desc,
    v_desc,
    o_desc,
    lse_ptr,
    seq_len,
    head_dim,
    SCALE,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    m_block_idx = tl.program_id(1)
    
    start_m = m_block_idx * BLOCK_M
    num_n_blocks = tl.cdiv(seq_len, BLOCK_N)
    
    q_left = q_desc.load([batch_idx * seq_len + start_m, 0])
    q_right = q_desc.load([batch_idx * seq_len + start_m, 64])
    
    m = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    acc_O_left = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    acc_O_right = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    
    for j in range(num_n_blocks):
        start_n = j * BLOCK_N
        
        k_left = k_desc.load([batch_idx * seq_len + start_n, 0])
        k_right = k_desc.load([batch_idx * seq_len + start_n, 64])
        
        v_left = v_desc.load([batch_idx * seq_len + start_n, 0])
        v_right = v_desc.load([batch_idx * seq_len + start_n, 64])
        
        acc_S = tl.dot(q_left, k_left.T)
        acc_S = tl.dot(q_right, k_right.T, acc_S)
        
        S_scaled = acc_S * SCALE
        
        m_old = m
        m_new = tl.maximum(m, tl.max(S_scaled, axis=1))
        
        exp_diff = tl.exp(m_old - m_new)
        
        P = tl.exp(S_scaled - m_new[:, None])
        
        valid_k = (start_n + tl.arange(0, BLOCK_N)) < seq_len
        P = P * valid_k[None, :]
        
        l = l * exp_diff + tl.sum(P, axis=1)
        m = m_new
        
        acc_O_left = acc_O_left * exp_diff[:, None] + tl.dot(P.to(tl.bfloat16), v_left)
        acc_O_right = acc_O_right * exp_diff[:, None] + tl.dot(P.to(tl.bfloat16), v_right)
        
    inv_l = 1.0 / tl.maximum(l, 1e-30)[:, None]
    o_left = (acc_O_left * inv_l).to(tl.bfloat16)
    o_right = (acc_O_right * inv_l).to(tl.bfloat16)
    
    o_desc.store([batch_idx * seq_len + start_m, 0], o_left)
    o_desc.store([batch_idx * seq_len + start_m, 64], o_right)
    
    seq_idx = start_m + tl.arange(0, BLOCK_M)
    valid_m = seq_idx < seq_len
    lse_val = m + tl.log(l)
    lse_val = lse_val * valid_m + 0.0 * (1 - valid_m)
    base_lse = batch_idx * seq_len
    tl.store(lse_ptr + base_lse + seq_idx, lse_val, mask=valid_m)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    if S == 0:
        return

    q_desc = TensorDescriptor.from_tensor(Q, [128, 64], padding_option="zero")
    k_desc = TensorDescriptor.from_tensor(K, [128, 64], padding_option="zero")
    v_desc = TensorDescriptor.from_tensor(V, [128, 64], padding_option="zero")
    o_desc = TensorDescriptor.from_tensor(O, [128, 64])
    
    SCALE = 1.0 / (128.0 ** 0.5)
    
    grid = (B * H, triton.cdiv(S, 128))
    _mha_kernel[grid](
        q_desc, k_desc, v_desc, o_desc, LSE, S, D,
        SCALE,
        BLOCK_M=128, BLOCK_N=128,
        num_warps=8, num_stages=3
    )