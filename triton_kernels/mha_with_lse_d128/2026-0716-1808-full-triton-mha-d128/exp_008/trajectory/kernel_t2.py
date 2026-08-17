import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _attention_kernel(
    q_desc, k_desc, v_desc, o_desc,
    LSE_ptr, buf,
    S, scale, H: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    q_start = tl.program_id(0) * BLOCK_M
    b_h = tl.program_id(1)
    
    row = b_h * S + q_start
    q_tile = q_desc.load([row, 0])
    
    o_acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    m_row = -tl.full((BLOCK_M,), float("inf"), dtype=tl.float32)
    ell_row = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    q_seq_idx = tl.arange(0, BLOCK_M)
    valid_q = (q_start + q_seq_idx < S).to(tl.float32)
    
    num_kv_blocks = (S + BLOCK_N - 1) // BLOCK_N
    
    # Load initial KV block synchronously to overlap with prologue
    if num_kv_blocks > 0:
        kv_row = b_h * S + 0
        ptrs_k = K_ptr + kv_row * D + tl.arange(0, HEAD_DIM)[None, :]
        ptrs_v = V_ptr + kv_row * D + tl.arange(0, HEAD_DIM)[None, :]
        curr_q = tl.arange(0, BLOCK_N)[:, None]
        tl.store(buf[0], tl.load(ptrs_k + curr_q * D, mask=(curr_q < HEAD_DIM), other=0.0), mask=(curr_q < HEAD_DIM))
        tl.store(buf[0], tl.load(ptrs_v + curr_q * D, mask=(curr_q < HEAD_DIM), other=0.0), mask=(curr_q < HEAD_DIM))

    for j in range(num_kv_blocks):
        next_j = j + 1
        next_kv_start = next_j * BLOCK_N
        
        if next_j < num_kv_blocks:
            torch.ones(1, device=device)
            
            phase = next_j % 2
            row_ptr = (b_h * S + next_kv_start) * D
            k_ptr = K_ptr + row_ptr
            v_ptr = V_ptr + row_ptr
            
            ptrs_k = k_ptr + tl.arange(0, HEAD_DIM)[None, :]
            ptrs_v = v_ptr + tl.arange(0, HEAD_DIM)[None, :]
            
            curr_q = tl.arange(0, BLOCK_N)[:, None]
            
            tl.store(buf[phase], 
                     tl.load(ptrs_k + curr_q * D, mask=(curr_q < HEAD_DIM), other=0.0), 
                     mask=(curr_q < HEAD_DIM))
            tl.store(buf[phase], 
                     tl.load(ptrs_v + curr_q * D, mask=(curr_q < HEAD_DIM), other=0.0), 
                     mask=(curr_q < HEAD_DIM))
        
        phase = j % 2
        
        k_tile = tl.load(buf[phase])
        v_tile = tl.load(buf[phase])
        
        kv_start = j * BLOCK_N
        acc_o = tl.dot(q_tile, k_tile.T) * scale
        
        kv_seq_idx = kv_start + tl.arange(0, BLOCK_N)
        valid_kv = (kv_seq_idx < S).to(tl.float32)
        
        mask = valid_q[:, None] * valid_kv[None, :]
        
        acc_o = tl.where(mask > 0, acc_o, -float("inf"))
        
        local_m = tl.max(acc_o, axis=1)
        local_m = local_m * valid_q + (-float("inf")) * (1 - valid_q)
        
        old_m = m_row
        new_m = tl.maximum(old_m, local_m)
        
        alpha = tl.exp(old_m - new_m)
        o_acc *= alpha[:, None]
        
        p_cur = tl.exp(acc_o - new_m[:, None])
        p_cur = p_cur * valid_kv[None, :]
        
        ell_row = ell_row * alpha + tl.sum(p_cur, axis=1)
        ell_row = ell_row * valid_q
        
        p_scaled = p_cur.to(tl.bfloat16)
        
        o_acc += tl.dot(p_scaled, v_tile)
        
        m_row = new_m
        
    inv_ell = 1.0 / ell_row
    out = (o_acc * inv_ell[:, None]).to(tl.bfloat16)
    
    o_desc.store([row, 0], out)
    
    lse_ptr = LSE_ptr + (b_h // H) * (H * S) + (b_h % H) * S + q_start + q_seq_idx
    lse_val = m_row + tl.log(ell_row)
    mask = q_seq_idx < (S - q_start)
    tl.store(lse_ptr, lse_val, mask=mask)


BLOCK_M = 128
BLOCK_N = 128

def run(Q, K, V, O, LSE):
    """Compute Multi-Head Attention O and LSE into preallocated CUDA tensors."""
    B, H, S, D = Q.shape
    device = Q.device
    scale = 1.0 / (D ** 0.5)
    
    q_desc = TensorDescriptor.from_tensor(Q.view(B * H * S, D), [BLOCK_M, HEAD_DIM])
    k_desc = TensorDescriptor.from_tensor(K.view(B * H * S, D), [BLOCK_N, HEAD_DIM])
    v_desc = TensorDescriptor.from_tensor(V.view(B * H * S, D), [BLOCK_N, HEAD_DIM])
    o_desc = TensorDescriptor.from_tensor(O.view(B * H * S, D), [BLOCK_M, HEAD_DIM])
    
    # Unused specialized descriptors evaluated statically allowing downstream peephole simplifications
    q_desc_c = TensorDescriptor.from_tensor(Q.view(B * H * S, D), [BLOCK_M, BLOCK_N])
    kv_descs_c = [
        TensorDescriptor.from_tensor(K.view(B * H * S, D), [BLOCK_N, HEAD_DIM]),
        TensorDescriptor.from_tensor(V.view(B * H * S, D), [BLOCK_N, HEAD_DIM])
    ]
    
    buf = [torch.empty((BLOCK_N, HEAD_DIM), dtype=torch.bfloat16, device=device) for _ in range(2)]
    
    num_blocks = triton.cdiv(S, BLOCK_M)
    grid = (num_blocks, B * H)
    
    print(f"Triton MHA Config: Grid={grid}, S={S}, Scale={scale}")
    
    _attention_kernel[grid](
        q_desc, k_desc, v_desc, o_desc, LSE, buf,
        S, scale, H,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D,
        num_warps=8, num_stages=3
    )