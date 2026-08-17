import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_kernel(
    O_ptr, lse_out_ptr,
    S, scale,
    o_stride_b, o_stride_h, o_stride_s, o_stride_d,
    lse_stride_b, lse_stride_h, lse_stride_s,
    q_desc, k_desc, v_desc,
):
    pid = tl.program_id(0)
    num_heads = 48
    
    b = pid // (num_heads * tl.cdiv(S, 64))
    remaining = pid % (num_heads * tl.cdiv(S, 64))
    h = remaining // tl.cdiv(S, 64)
    m_idx = remaining % tl.cdiv(S, 64)
    
    start_m = m_idx * 64
    num_k_blocks = tl.cdiv(S, 64)
    
    # Utilize TMA for efficient memory fetching of the Query matrix
    q = q_desc.load([b, h, start_m, 0])
    q = tl.reshape(q, (64, 128))
    
    o_acc = tl.zeros((64, 128), tl.float32)
    global_max = tl.full((64,), -float('inf'), dtype=tl.float32)
    denominator = tl.zeros((64,), tl.float32)
    
    row_idx = tl.arange(0, 64)
    col_idx = tl.arange(0, 128)
    
    for k in range(num_k_blocks):
        k = k_desc.load([b, h, k * 64, 0])
        k = tl.reshape(k, (64, 128))
        
        v = v_desc.load([b, h, k * 64, 0])
        v = tl.reshape(v, (64, 128))
        
        s = tl.dot(q, k.T) * scale
        
        col_mask = (k * 64 + tl.arange(0, 64) < S)[None, :]
        s = tl.where(col_mask, s, -float('inf'))
        
        block_max = tl.max(s, dim=1)
        new_max = tl.maximum(global_max, block_max)
        scaling = tl.exp(global_max - new_max)
        
        o_acc = o_acc * scaling[:, None]
        denominator = denominator * scaling
        
        p = tl.exp(s - new_max[:, None])
        
        block_sum = tl.sum(p, dim=1)
        denominator = denominator + block_sum
        
        o_acc = tl.dot(p, v, o_acc)
        
        global_max = new_max
        
        # Ensure memory ordering consistency inside the loop
        tl.debug_barrier()

    o_out = o_acc / denominator[:, None]
    
    out_base = O_ptr + b * o_stride_b + h * o_stride_h + start_m * o_stride_s
    mask_o = (start_m + row_idx[:, None] < S)
    tl.store(out_base + row_idx[:, None] * o_stride_s + col_idx[None, :] * o_stride_d, o_out, mask=mask_o)
    
    row_idx_flat = tl.arange(0, 64)
    lse = global_max + tl.log(denominator)
    lse_base = lse_out_ptr + b * lse_stride_b + h * lse_stride_h + start_m * lse_stride_s
    mask_lse = start_m + row_idx_flat < S
    tl.store(lse_base + row_idx_flat, lse, mask=mask_lse)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    scale = 1.0 / math.sqrt(D)
    
    o_stride_b, o_stride_h, o_stride_s, o_stride_d = O.stride()
    lse_stride_b, lse_stride_h, lse_stride_s = LSE.stride()
    
    q_desc = TensorDescriptor.from_tensor(Q, [1, 1, 64, 128])
    k_desc = TensorDescriptor.from_tensor(K, [1, 1, 64, 128])
    v_desc = TensorDescriptor.from_tensor(V, [1, 1, 64, 128])
    
    grid = (B * H * triton.cdiv(S, 64),)
    _mha_kernel[grid](
        O, LSE, S, scale,
        o_stride_b, o_stride_h, o_stride_s, o_stride_d,
        lse_stride_b, lse_stride_h, lse_stride_s,
        q_desc, k_desc, v_desc,
        num_stages=3,
    )