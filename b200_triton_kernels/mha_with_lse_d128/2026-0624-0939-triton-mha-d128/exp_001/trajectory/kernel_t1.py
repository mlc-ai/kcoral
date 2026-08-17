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
    num_batches = 4
    num_heads = 48
    
    b = pid // (num_heads * tl.cdiv(S, 128))
    remaining = pid % (num_heads * tl.cdiv(S, 128))
    h = remaining // tl.cdiv(S, 128)
    m_idx = remaining % tl.cdiv(S, 128)
    
    start_m = m_idx * 128
    num_k = tl.cdiv(S, 64)
    
    # Load Q tile 
    if start_m < S:
        q = q_desc.load([b, h, start_m, 0])
        q = tl.reshape(q, (128, 128))
        q0, q1 = tl.split(q, 2)
    else:
        q0 = tl.zeros((128, 64), tl.bfloat16)
        q1 = tl.zeros((128, 64), tl.bfloat16)
    
    o_acc_left = tl.zeros((128, 64), tl.float32)
    o_acc_right = tl.zeros((128, 64), tl.float32)
    global_max = tl.full((128,), -float('inf'), dtype=tl.float32)
    denominator = tl.zeros((128,), tl.float32)
    
    start_k = 0
    
    if start_k < num_k:
        k_0 = k_desc.load([b, h, start_k * 64, 0])
        v_0 = v_desc.load([b, h, start_k * 64, 0])
    
    for i in range(start_k, num_k):
        idx = (i - start_k) % 2
        next_i = i + 1
        
        if next_i < num_k:
            if idx == 0:
                k_1 = k_desc.load([b, h, next_i * 64, 0])
                v_1 = v_desc.load([b, h, next_i * 64, 0])
            else:
                k_0 = k_desc.load([b, h, next_i * 64, 0])
                v_0 = v_desc.load([b, h, next_i * 64, 0])
        else:
            if idx == 0:
                k_1 = tl.zeros((1, 1, 64, 128), tl.bfloat16)
                v_1 = tl.zeros((1, 1, 64, 128), tl.bfloat16)
            else:
                k_0 = tl.zeros((1, 1, 64, 128), tl.bfloat16)
                v_0 = tl.zeros((1, 1, 64, 128), tl.bfloat16)
        
        if idx == 0:
            k_curr = k_0
            v_curr = v_0
        else:
            k_curr = k_1
            v_curr = v_1
        
        k_curr = tl.reshape(k_curr, (64, 128))
        v_curr = tl.reshape(v_curr, (64, 128))
        k0_split, k1_split = tl.split(k_curr, 2)
        v0_split, v1_split = tl.split(v_curr, 2)
        
        s = (tl.dot(q0, k0_split.T) + tl.dot(q1, k1_split.T)) * scale
        
        block_max = tl.max(s, dim=1)
        new_max = tl.maximum(global_max, block_max)
        scaling = tl.exp(global_max - new_max)
        
        o_acc_left = o_acc_left * scaling[:, None]
        o_acc_right = o_acc_right * scaling[:, None]
        denominator = denominator * scaling
        
        p = tl.exp(s - new_max[:, None])
        
        block_sum = tl.sum(p, dim=1)
        denominator = denominator + block_sum
        
        o_acc_left = tl.dot(p, v0_split, o_acc_left)
        o_acc_right = tl.dot(p, v1_split, o_acc_right)
        
        global_max = new_max
        
        tl.debug_barrier()
    
    o_left = o_acc_left / denominator[:, None]
    o_right = o_acc_right / denominator[:, None]
    
    row_idx = tl.arange(0, 128)[:, None]
    col_0 = tl.arange(0, 64)[None, :]
    col_1 = 64 + tl.arange(0, 64)[None, :]
    
    out_base = O_ptr + b * o_stride_b + h * o_stride_h + start_m * o_stride_s
    mask_o = (start_m + row_idx < S)
    tl.store(out_base + row_idx * o_stride_s + col_0, o_left, mask=mask_o)
    tl.store(out_base + row_idx * o_stride_s + col_1, o_right, mask=mask_o)
    
    row_idx_flat = tl.arange(0, 128)
    lse_base = lse_out_ptr + b * lse_stride_b + h * lse_stride_h + start_m * lse_stride_s
    mask_lse = (start_m + row_idx_flat < S)
    lse = global_max + tl.log(denominator)
    tl.store(lse_base + row_idx_flat, lse, mask=mask_lse)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    scale = 1.0 / math.sqrt(D)
    
    q_strides = Q.stride()
    k_strides = K.stride()
    v_strides = V.stride()
    o_strides = O.stride()
    lse_strides = LSE.stride()
    
    k_desc = TensorDescriptor.from_tensor(K, [1, 1, 64, 128])
    v_desc = TensorDescriptor.from_tensor(V, [1, 1, 64, 128])
    q_desc = TensorDescriptor.from_tensor(Q, [1, 1, 128, 128])
    
    grid = (B * H * triton.cdiv(S, 128),)
    _mha_kernel[grid](
        Q, K, V, O, LSE, S, scale,
        q_strides, k_strides, v_strides, o_strides, lse_strides,
        q_desc, k_desc, v_desc,
        num_stages=2,
    )