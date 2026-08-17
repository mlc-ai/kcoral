import torch
import triton
import triton.language as tl


def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)


@triton.jit
def _mha_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    H, S, scale,
    BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr, D: tl.constexpr,
):
    bid_q = tl.program_id(0)
    bid_h = tl.program_id(1)
    bid_b = tl.program_id(2)
    
    qo = bid_q * BLOCK_Q
    
    batch_offset_elements = bid_b * H * S * D + bid_h * S * D
    
    q_desc = tl.make_tensor_descriptor(
        Q_ptr + batch_offset_elements,
        shape=[S, D], strides=[D, 1],
        block_shape=[BLOCK_Q, D], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        K_ptr + batch_offset_elements,
        shape=[S, D], strides=[D, 1],
        block_shape=[BLOCK_KV, D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V_ptr + batch_offset_elements,
        shape=[S, D], strides=[D, 1],
        block_shape=[BLOCK_KV, D], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        O_ptr + batch_offset_elements,
        shape=[S, D], strides=[D, 1],
        block_shape=[BLOCK_Q, D]
    )
    
    Q = q_desc.load([qo, 0])
    
    num_kv = tl.cdiv(S, BLOCK_KV)
    
    acc_o = tl.zeros((BLOCK_Q, D), tl.float32)
    m_i = tl.full((BLOCK_Q, 1), -float('inf'), tl.float32)
    l_i = tl.full((BLOCK_Q, 1), 0.0, tl.float32)
    
    row_indices = tl.arange(0, BLOCK_Q)
    col_indices = tl.arange(0, BLOCK_KV)
    q_seq_o = bid_q * BLOCK_Q + row_indices
    
    for j in tl.range(num_kv, num_stages=3):
        ko = j * BLOCK_KV
        
        K = k_desc.load([ko, 0])
        S_tile = tl.dot(Q, K.T) * scale
        
        k_seq_o = j * BLOCK_KV + col_indices
        valid_mask = (q_seq_o[:, None] < S) & (k_seq_o[None, :] < S)
        S_tile = tl.where(valid_mask, S_tile, -float('inf'))
        
        m_old = m_i
        m_new = tl.maximum(m_i, tl.max(S_tile, axis=1, keep_dims=True))
        
        P = tl.where(valid_mask, tl.exp(S_tile - m_new), 0.0)
        
        exp_diff = tl.where(m_new > -float('inf'), tl.exp(m_old - m_new), 0.0)
        l_i = tl.fma(exp_diff, l_i, tl.sum(P, axis=1, keep_dims=True))
        m_i = m_new
        
        V = v_desc.load([ko, 0])
        acc_o = tl.fma(exp_diff, acc_o, tl.dot(P, V))
    
    O = acc_o / (l_i + 1e-30)
    o_desc.store([qo, 0], O.to(tl.bfloat16))
    
    lse = m_i + tl.log(l_i + 1e-30)
    lse_row_indices = tl.arange(0, BLOCK_Q)
    q_seq_o_lse = bid_q * BLOCK_Q + lse_row_indices
    valid_mask_lse = q_seq_o_lse < S
    lse_base = bid_b * H * S + bid_h * S + qo
    tl.store(LSE_ptr + lse_base + lse_row_indices, lse.squeeze(-1), mask=valid_mask_lse)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    scale = 1.0 / (D ** 0.5)
    
    BLOCK_Q = 64
    BLOCK_KV = 64
    
    num_blocks = triton.cdiv(S, BLOCK_Q)
    grid = (num_blocks, H, B)
    
    _mha_kernel[grid](
        Q, K, V, O, LSE,
        H, S, scale,
        BLOCK_Q=BLOCK_Q, BLOCK_KV=BLOCK_KV, D=D,
        num_warps=8, num_stages=3,
    )