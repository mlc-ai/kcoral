import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_kernel(
    q_desc,
    k_desc,
    v_desc,
    out_desc,
    lse_desc,
    seq_len,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    m_block_idx = tl.program_id(1)
    
    num_k_tiles = HEAD_DIM // BLOCK_D
    
    start_m = m_block_idx * BLOCK_M
    row_idx = start_m + tl.arange(0, BLOCK_M)
    
    global_m_off = batch_idx * seq_len + start_m
    
    acc_O = []
    for i in range(num_k_tiles):
        d_off = i * BLOCK_D
        q = q_desc.load([global_m_off, d_off])
        acc_O.append(tl.zeros((BLOCK_M, BLOCK_N), tl.float32))
    
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    m = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    
    num_n_blocks = tl.cdiv(seq_len, BLOCK_N)
    SCALE = 0.08838834764831843 
    
    for j in range(num_n_blocks):
        start_n = j * BLOCK_N
        global_n_off = batch_idx * seq_len + start_n
        
        k_tiles = []
        for i in range(num_k_tiles):
            d_off = i * BLOCK_D
            k = k_desc.load([global_n_off, d_off])
            k_tiles.append(k)
            
        v_tiles = []
        for i in range(num_k_tiles):
            d_off = i * BLOCK_D
            v = v_desc.load([global_n_off, d_off])
            v_tiles.append(v)
            
        acc_S = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for i in range(num_k_tiles):
            d_off = i * BLOCK_D
            q = q_desc.load([global_m_off, d_off])
            k = k_tiles[i]
            acc_S = tl.dot(q, k.T, acc_S)
            
        cols = tl.arange(0, BLOCK_N)
        k_seq_idx = start_n + cols
        valid_k = k_seq_idx < seq_len
        
        m_old = m
        S_scaled = acc_S * SCALE
        m_new = tl.maximum(m, tl.max(S_scaled, axis=1))
        
        P = tl.exp(S_scaled - m_new[:, None])
        P = P * valid_k[None, :]
        
        l = l * tl.exp(m_old - m_new) + tl.sum(P, axis=1)
        m = m_new
        
        exp_scale = tl.exp(m_old - m_new)[:, None]
        for i in range(num_k_tiles):
            acc_O[i] = acc_O[i] * exp_scale
            
        P_bf16 = P.to(tl.bfloat16)
        for i in range(num_k_tiles):
            v = v_tiles[i]
            acc_O[i] = tl.dot(P_bf16, v, acc_O[i])
            
    cols = tl.arange(0, BLOCK_N)
    seq_idx = start_m + cols
    valid_m = seq_idx < seq_len
    
    inv_l = 1.0 / l[:, None]
    for i in range(num_k_tiles):
        d_off = i * BLOCK_D
        O = (acc_O[i] * inv_l).to(tl.bfloat16)
        out_desc.store([global_m_off, d_off], O)
        
    LSE = m + tl.log(l)
    LSE = LSE * valid_m
    lse_desc.store([global_m_off, 0], LSE[:, None])


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    if S == 0:
        return

    q_desc = TensorDescriptor.from_tensor(Q, [64, 64], contiguous_or_permutable=True)
    k_desc = TensorDescriptor.from_tensor(K, [64, 64], contiguous_or_permutable=True)
    v_desc = TensorDescriptor.from_tensor(V, [64, 64], contiguous_or_permutable=True)
    out_desc = TensorDescriptor.from_tensor(O, [64, 64], contiguous_or_permutable=True)
    lse_desc = TensorDescriptor.from_tensor(LSE, [64, 1], contiguous_or_permutable=True)
    
    grid = (B * H, triton.cdiv(S, 64))
    _mha_kernel[grid](
        q_desc, k_desc, v_desc, out_desc, lse_desc, S,
        HEAD_DIM=128, BLOCK_M=64, BLOCK_N=64, BLOCK_D=64,
        num_warps=8, num_stages=2
    )