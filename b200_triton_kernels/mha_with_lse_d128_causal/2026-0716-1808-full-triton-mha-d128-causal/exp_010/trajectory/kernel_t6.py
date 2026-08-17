import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _attention_kernel(
    q_desc,
    k_desc,
    v_desc,
    o_desc,
    lse_ptr,
    M,
    sqrt_D,
    stride_bh_lse,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    tid = tl.program_id(0)
    num_blocks_per_bh = triton.cdiv(M, BLOCK_M)
    batch_head_idx = tid // num_blocks_per_bh
    query_block_idx = tid % num_blocks_per_bh

    q_row_idx = tl.arange(0, BLOCK_M)
    
    Q_0 = q_desc.load([batch_head_idx * M + query_block_idx * BLOCK_M, 0])
    Q_1 = q_desc.load([batch_head_idx * M + query_block_idx * BLOCK_M, BLOCK_D])
    
    O_0 = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    O_1 = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    m = tl.full((BLOCK_M,), -1e20, tl.float32)
    l = tl.full((BLOCK_M,), 0.0, tl.float32)

    for j in range(query_block_idx + 1):
        K_0 = k_desc.load([batch_head_idx * M + j * BLOCK_N, 0])
        K_1 = k_desc.load([batch_head_idx * M + j * BLOCK_N, BLOCK_D])
        
        V_0 = v_desc.load([batch_head_idx * M + j * BLOCK_N, 0])
        V_1 = v_desc.load([batch_head_idx * M + j * BLOCK_N, BLOCK_D])
        
        S = tl.dot(Q_0, K_0.T)
        S += tl.dot(Q_1, K_1.T)
        S = S * (1.0 / sqrt_D)
        
        apply_mask = (j == query_block_idx)
        mask_ok = (query_block_idx * BLOCK_M + q_row_idx[:, None]) >= (j * BLOCK_N + tl.arange(0, BLOCK_N)[None, :])
        S = tl.where(apply_mask, tl.where(mask_ok, S, -1e20), S)
            
        row_max = tl.max(S, axis=1)
        m_old = m
        m = tl.maximum(m_old, row_max)
        
        exp_m_diff = tl.exp(m_old - m)
        l = l * exp_m_diff
        
        P = tl.exp(S - m[:, None])
        l += tl.sum(P, axis=1)
        
        O_0 = O_0 * exp_m_diff[:, None]
        O_1 = O_1 * exp_m_diff[:, None]
        
        O_0 += tl.dot(P.to(tl.bfloat16), V_0)
        O_1 += tl.dot(P.to(tl.bfloat16), V_1)
        
    O_0 = O_0 / l[:, None]
    O_1 = O_1 / l[:, None]
    
    o_desc.store([batch_head_idx * M + query_block_idx * BLOCK_M, 0], O_0.to(tl.bfloat16))
    o_desc.store([batch_head_idx * M + query_block_idx * BLOCK_M, BLOCK_D], O_1.to(tl.bfloat16))
    
    lse_ptr_bh = lse_ptr + batch_head_idx * stride_bh_lse + query_block_idx * BLOCK_M + q_row_idx
    tl.store(lse_ptr_bh, m + tl.log(l), mask=((query_block_idx * BLOCK_M + q_row_idx) < M))


def run(Q, K, V, O, LSE):
    """Compute causal multi-head attention O and LSE into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    B, H, M, D = Q.shape
    sqrt_D = D ** 0.5
    
    stride_bh_lse = M
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_D = 64
    
    q_desc = TensorDescriptor.from_tensor(Q, [BLOCK_M, BLOCK_D])
    k_desc = TensorDescriptor.from_tensor(K, [BLOCK_N, BLOCK_D])
    v_desc = TensorDescriptor.from_tensor(V, [BLOCK_N, BLOCK_D])
    o_desc = TensorDescriptor.from_tensor(O, [BLOCK_M, BLOCK_D])
    
    num_blocks = triton.cdiv(M, BLOCK_M)
    grid = (B * H * num_blocks,)
    
    _attention_kernel[grid](
        q_desc, k_desc, v_desc, o_desc,
        LSE, M, sqrt_D, stride_bh_lse,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2,
    )