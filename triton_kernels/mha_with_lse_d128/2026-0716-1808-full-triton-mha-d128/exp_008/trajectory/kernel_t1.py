import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _attention_kernel(
    q_desc,
    kv_descs,
    o_desc,
    LSE_ptr,
    S,
    scale,
    BLOCK_SIZE: tl.constexpr,
):
    q_start = tl.program_id(0) * BLOCK_SIZE
    b_h = tl.program_id(1)
    
    q_tile = q_desc.load([b_h, q_start, 0]).squeeze(0)
    
    o_acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    m_row = -tl.full((BLOCK_SIZE,), float("inf"), dtype=tl.float32)
    ell_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    q_seq_idx = tl.arange(0, BLOCK_SIZE)
    valid_q = (q_start + q_seq_idx < S).to(tl.float32)
    
    num_kv_blocks = (S + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    for j in range(num_kv_blocks):
        phase = j % 2
        next_phase = (j + 1) % 2
        
        cur_kv_descs = kv_descs[phase]
        next_kv_descs = kv_descs[next_phase]
        
        if j + 1 < num_kv_blocks:
            next_kv_start = (j + 1) * BLOCK_SIZE
            next_k_tile = next_kv_descs[0].load([b_h, next_kv_start, 0]).squeeze(0)
            next_v_tile = next_kv_descs[1].load([b_h, next_kv_start, 0]).squeeze(0)
            
        k_tile = cur_kv_descs[0].load([b_h, j * BLOCK_SIZE, 0]).squeeze(0)
        v_tile = cur_kv_descs[1].load([b_h, j * BLOCK_SIZE, 0]).squeeze(0)
        
        kv_start = j * BLOCK_SIZE
        acc_o = tl.dot(q_tile, k_tile.T) * scale
        
        kv_seq_idx = kv_start + tl.arange(0, BLOCK_SIZE)
        valid_kv = (kv_seq_idx < S).to(tl.float32)
        
        mask = valid_q[:, None] * valid_kv[None, :]
        
        acc_o = tl.where(mask > 0, acc_o, -float("inf"))
        
        local_m = tl.max(acc_o, axis=1)
        old_m = m_row
        new_m = tl.maximum(old_m, local_m)
        
        o_acc *= tl.exp(old_m - new_m)
        
        p_cur = tl.exp(acc_o - new_m)
        p_cur = p_cur * valid_kv[None, :]
        
        ell_row = ell_row * tl.exp(old_m - new_m) + tl.sum(p_cur, axis=1)
        
        p_scaled = p_cur.to(tl.bfloat16)
        
        o_acc += tl.dot(p_scaled, v_tile)
        
        m_row = new_m
        
    inv_ell = 1.0 / ell_row
    out = (o_acc * inv_ell[:, None]).to(tl.bfloat16)
    
    o_desc.store([b_h, q_start, 0], out, boundary_check=(1, 2))
    
    lse_ptr = LSE_ptr + b_h * S + q_start + q_seq_idx
    lse_val = m_row + tl.log(ell_row)
    mask = q_seq_idx < (S - q_start)
    tl.store(lse_ptr, lse_val, mask=mask)


BLOCK_SIZE = 128

def run(Q, K, V, O, LSE):
    """Compute Multi-Head Attention O and LSE into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    scale = 1.0 / (D ** 0.5)
    
    q_desc = TensorDescriptor.from_tensor(Q.view(B * H, S, D), [1, BLOCK_SIZE, D])
    k_desc = TensorDescriptor.from_tensor(K.view(B * H, S, D), [1, BLOCK_SIZE, D])
    v_desc = TensorDescriptor.from_tensor(V.view(B * H, S, D), [1, BLOCK_SIZE, D])
    o_desc = TensorDescriptor.from_tensor(O.view(B * H, S, D), [1, BLOCK_SIZE, D])
    
    kv_descs = [
        [k_desc.clone(), v_desc.clone()],
        [k_desc.clone(), v_desc.clone()],
    ]
    
    num_blocks = triton.cdiv(S, BLOCK_SIZE)
    grid = (num_blocks, B * H)
    
    print(f"Triton MHA Config: Grid={grid}, S={S}, Scale={scale}")
    
    _attention_kernel[grid](
        q_desc, kv_descs, o_desc, LSE, 
        S, scale,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8, num_stages=3
    )