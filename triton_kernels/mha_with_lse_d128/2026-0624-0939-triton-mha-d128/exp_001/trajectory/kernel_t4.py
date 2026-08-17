import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_kernel(
    O_ptr, lse_out_ptr, S, scale,
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
    
    start_m = n_idx * 128
    num_k_blocks = tl.cdiv(S, 64)
    bh = b * num_heads + h
    
    q0_3d = q_desc.load([bh, start_m, 0])
    q0 = tl.reshape(q0_3d, (128, 64))
    q1_3d = q_desc.load([bh, start_m, 64])
    q1 = tl.reshape(q1_3d, (128, 64))
    
    o_acc_left = tl.zeros((128, 64), tl.float32)
    o_acc_right = tl.zeros((128, 64), tl.float32)
    global_max = tl.full((128,), -float('inf'), dtype=tl.float32)
    denominator = tl.zeros((128,), tl.float32)
    
    for k in range(num_k_blocks):
        k0_3d = k_desc.load([bh, k * 64, 0])
        k0 = tl.reshape(k0_3d, (64, 64))
        
        k1_3d = k_desc.load([bh, k * 64, 64])
        k1 = tl.reshape(k1_3d, (64, 64))
        
        v0_3d = v_desc.load([bh, k * 64, 0])
        v0 = tl.reshape(v0_3d, (64, 64))
        
        v1_3d = v_desc.load([bh, k * 64, 64])
        v1 = tl.reshape(v1_3d, (64, 64))
        
        s = (tl.dot(q0, k0.T) + tl.dot(q1, k1.T)) * scale
        
        col_mask = (k * 64 + tl.arange(0, 64)[None, :] < S)
        s = tl.where(col_mask, s, -10000.0)
        
        block_max = tl.max(s, dim=1)
        new_max = tl.maximum(global_max, block_max)
        scaling = tl.exp(global_max - new_max)
        
        o_acc_left = o_acc_left * scaling[:, None]
        o_acc_right = o_acc_right * scaling[:, None]
        denominator = denominator * scaling
        
        p = tl.exp(s - new_max[:, None])
        
        block_sum = tl.sum(p, dim=1)
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
    
    out_base = O_ptr + b * o_stride_b + h * o_stride_h + start_m * o_stride_s
    mask_o = (start_m + row_idx < S)
    tl.store(out_base + row_idx * o_stride_s + col_left, o_left, mask=mask_o)
    tl.store(out_base + row_idx * o_stride_s + col_right, o_right, mask=mask_o)
    
    row_idx_flat = tl.arange(0, 128)
    lse_base = lse_out_ptr + b * lse_stride_b + h * lse_stride_h + start_m * lse_stride_s
    mask_lse = (start_m + row_idx_flat < S)
    lse = global_max + tl.log(denominator)
    tl.store(lse_base + row_idx_flat, lse, mask=mask_lse)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    scale = 1.0 / math.sqrt(D)
    
    o_stride_b = torch.tensor(O.stride(0), device='cuda')
    o_stride_h = torch.tensor(O.stride(1), device='cuda')
    o_stride_s = torch.tensor(O.stride(2), device='cuda')
    
    lse_stride_b = torch.tensor(LSE.stride(0), device='cuda')
    lse_stride_h = torch.tensor(LSE.stride(1), device='cuda')
    lse_stride_s = torch.tensor(LSE.stride(2), device='cuda')
    
    q_desc = TensorDescriptor.from_tensor(Q, [B*H, S, 128], block_shape=[1, 128, 64])
    k_desc = TensorDescriptor.from_tensor(K, [B*H, S, 128], block_shape=[1, 64, 64])
    v_desc = TensorDescriptor.from_tensor(V, [B*H, S, 128], block_shape=[1, 64, 64])
    
    grid = (B * H * triton.cdiv(S, 128),)
    _mha_kernel[grid](
        O, LSE, S, scale,
        o_stride_b, o_stride_h, o_stride_s,
        lse_stride_b, lse_stride_h, lse_stride_s,
        q_desc, k_desc, v_desc,
        num_stages=3,
    )