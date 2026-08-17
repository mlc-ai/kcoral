import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_kernel(
    O_ptr, lse_out_ptr,
    S, scale,
    o_stride_b, o_stride_h, o_stride_s,
    lse_stride_b, lse_stride_h, lse_stride_s,
    q_desc, k_desc, v_desc,
):
    pid = tl.program_id(0)
    num_heads = 48
    
    b = pid // (num_heads * tl.cdiv(S, 128))
    remaining = pid % (num_heads * tl.cdiv(S, 128))
    h = remaining // tl.cdiv(S, 128)
    n_idx = remaining % tl.cdiv(S, 128)
    
    start_n = n_idx * 128
    num_k_blocks = tl.cdiv(S, 64)
    
    q = q_desc.load([b, h, start_n, 0])
    q = tl.reshape(q, (128, 128))
    
    q0, q1 = split_tensor(q, 1)
    
    o_acc_left = tl.zeros((128, 64), tl.float32)
    o_acc_right = tl.zeros((128, 64), tl.float32)
    global_max = tl.full((1,), -float('inf'), dtype=tl.float32)
    denominator = tl.zeros((1,), tl.float32)
    
    for k in range(num_k_blocks):
        k_tile = k_desc.load([b, h, k * 64, 0])
        k = tl.reshape(k_tile, (64, 128))
        
        v_tile = v_desc.load([b, h, k * 64, 0])
        v = tl.reshape(v_tile, (64, 128))
        
        k0, k1 = split_tensor(k, 1)
        v0, v1 = split_tensor(v, 1)
        
        s_0 = tl.dot(q0, k0.T)
        s_1 = tl.dot(q1, k1.T)
        s = (s_0 + s_1) * scale
        
        col_mask = k * 64 + tl.arange(0, 64)[None, :] < S
        s = tl.where(col_mask, s, -float('inf'))
        
        block_max = tl.max(s, dim=1)
        new_max = tl.maximum(global_max, block_max)
        scaling = tl.exp(global_max - new_max)
        
        o_acc_left = o_acc_left * scaling[:, None]
        o_acc_right = o_acc_right * scaling[:, None]
        denominator = denominator * scaling
        
        p = tl.exp(s - new_max[:, None])
        
        block_sum = tl.sum(p)
        denominator = denominator + block_sum
        
        o_acc_left = tl.dot(p, v0, o_acc_left)
        o_acc_right = tl.dot(p, v1, o_acc_right)
        
        global_max = new_max
        
        tl.debug_barrier()

    o_left = o_acc_left / denominator[:, None]
    o_right = o_acc_right / denominator[:, None]
    
    row_idx = tl.arange(0, 128)[:, None]
    col_left = tl.arange(0, 64)[None, :]
    col_right = 64 + tl.arange(0, 64)[None, :]
    
    out_base = O_ptr + b * o_stride_b + h * o_stride_h + start_n * o_stride_s
    mask_o = (start_n + row_idx < S)
    tl.store(out_base + row_idx * o_stride_s + col_left, o_left, mask=mask_o)
    tl.store(out_base + row_idx * o_stride_s + col_right, o_right, mask=mask_o)
    
    row_idx_flat = tl.arange(0, 128)
    lse_base = lse_out_ptr + b * lse_stride_b + h * lse_stride_h + start_n * lse_stride_s
    mask_lse = (start_n + row_idx_flat < S)
    lse = global_max + tl.log(denominator)
    tl.store(lse_base + row_idx_flat, lse, mask=mask_lse)


@triton.jit
def split_tensor(tensor, dim):
    flat = tl.reshape(tensor, (-1,))
    mid = flat.numel() // 2
    left = flat[:mid]
    right = flat[mid:]
    orig_shape = list(tensor.shape)
    new_dim = orig_shape[dim] // 2
    orig_shape[dim] = new_dim
    left = tl.reshape(left, orig_shape)
    right = tl.reshape(right, orig_shape)
    return left, right


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    scale = 1.0 / math.sqrt(D)
    
    o_stride_b = O.stride(0)
    o_stride_h = O.stride(1)
    o_stride_s = O.stride(2)
    
    lse_stride_b = LSE.stride(0)
    lse_stride_h = LSE.stride(1)
    lse_stride_s = LSE.stride(2)
    
    q_desc = TensorDescriptor.from_tensor(Q, [1, 1, 128, 128])
    k_desc = TensorDescriptor.from_tensor(K, [1, 1, 64, 128])
    v_desc = TensorDescriptor.from_tensor(V, [1, 1, 64, 128])
    
    grid = (B * H * triton.cdiv(S, 128),)
    _mha_kernel[grid](
        O, LSE, S, scale,
        o_stride_b, o_stride_h, o_stride_s,
        lse_stride_b, lse_stride_h, lse_stride_s,
        q_desc, k_desc, v_desc,
        num_stages=2,
    )