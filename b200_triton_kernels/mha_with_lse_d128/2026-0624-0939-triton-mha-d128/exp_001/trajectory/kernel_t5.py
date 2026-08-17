import math
import torch
import triton
import triton.language as tl


@triton.jit
def _mha_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, lse_out_ptr,
    S, scale,
    q_stride_b, q_stride_h, q_stride_s, q_stride_d,
    k_stride_b, k_stride_h, k_stride_s, k_stride_d,
    v_stride_b, v_stride_h, v_stride_s, v_stride_d,
    o_stride_b, o_stride_h, o_stride_s, o_stride_d,
    lse_stride_b, lse_stride_h, lse_stride_s,
):
    pid = tl.program_id(0)
    num_heads = 48
    
    b = pid // (num_heads * tl.cdiv(S, 128))
    remaining = pid % (num_heads * tl.cdiv(S, 128))
    h = remaining // tl.cdiv(S, 128)
    m_idx = remaining % tl.cdiv(S, 128)
    
    start_m = m_idx * 128
    num_k_blocks = tl.cdiv(S, 64)
    
    # Pre-calculate base pointers to avoid repeatedly adding the same offsets
    q_base_ptr = Q_ptr + b * q_stride_b + h * q_stride_h + start_m * q_stride_s
    row_idx = tl.arange(0, 128)[:, None]
    col_idx_0 = tl.arange(0, 64)[None, :]
    col_idx_1 = 64 + tl.arange(0, 64)[None, :]
    
    mask_q_row = (start_m + row_idx < S)
    
    q0 = tl.load(q_base_ptr + row_idx * q_stride_s + col_idx_0 * q_stride_d, mask=mask_q_row, other=0.0)
    q1 = tl.load(q_base_ptr + row_idx * q_stride_s + col_idx_1 * q_stride_d, mask=mask_q_row, other=0.0)
    
    o_acc_left = tl.zeros((128, 64), tl.float32)
    o_acc_right = tl.zeros((128, 64), tl.float32)
    global_max = tl.full((128,), -float('inf'), dtype=tl.float32)
    denominator = tl.zeros((128,), tl.float32)
    
    k_row_idx = tl.arange(0, 64)[:, None]
    
    for k in range(num_k_blocks):
        k_base_ptr = K_ptr + b * k_stride_b + h * k_stride_h + k * 64 * k_stride_s
        v_base_ptr = V_ptr + b * v_stride_b + h * v_stride_h + k * 64 * v_stride_s
        
        mask_k_row = (k * 64 + tl.arange(0, 64) < S)
        
        k0 = tl.load(k_base_ptr + k_row_idx * k_stride_s + col_idx_0 * k_stride_d, mask=mask_k_row, other=0.0)
        k1 = tl.load(k_base_ptr + k_row_idx * k_stride_s + col_idx_1 * k_stride_d, mask=mask_k_row, other=0.0)
        
        v0 = tl.load(v_base_ptr + k_row_idx * v_stride_s + col_idx_0 * v_stride_d, mask=mask_k_row, other=0.0)
        v1 = tl.load(v_base_ptr + k_row_idx * v_stride_s + col_idx_1 * v_stride_d, mask=mask_k_row, other=0.0)
        
        s = (tl.dot(q0, k0.T) + tl.dot(q1, k1.T)) * scale
        
        mask_k_col = (k * 64 + tl.arange(0, 64) < S)[None, :]
        s = tl.where(mask_k_col, s, -float('inf'))
        
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
    
    out_base = O_ptr + b * o_stride_b + h * o_stride_h + start_m * o_stride_s
    mask_o = (start_m + row_idx < S)
    tl.store(out_base + row_idx * o_stride_s + col_idx_0 * o_stride_d, o_left, mask=mask_o)
    tl.store(out_base + row_idx * o_stride_s + col_idx_1 * o_stride_d, o_right, mask=mask_o)
    
    row_idx_flat = tl.arange(0, 128)
    lse_base = lse_out_ptr + b * lse_stride_b + h * lse_stride_h + start_m * lse_stride_s
    mask_lse = (start_m + row_idx_flat < S)
    lse = global_max + tl.log(denominator)
    tl.store(lse_base + row_idx_flat, lse, mask=mask_lse)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    scale = 1.0 / math.sqrt(D)
    
    q_stride_b, q_stride_h, q_stride_s, q_stride_d = Q.stride()
    k_stride_b, k_stride_h, k_stride_s, k_stride_d = K.stride()
    v_stride_b, v_stride_h, v_stride_s, v_stride_d = V.stride()
    o_stride_b, o_stride_h, o_stride_s, o_stride_d = O.stride()
    lse_stride_b, lse_stride_h, lse_stride_s = LSE.stride()
    
    grid = (B * H * triton.cdiv(S, 128),)
    _mha_kernel[grid](
        Q, K, V, O, LSE, S, scale,
        q_stride_b, q_stride_h, q_stride_s, q_stride_d,
        k_stride_b, k_stride_h, k_stride_s, k_stride_d,
        v_stride_b, v_stride_h, v_stride_s, v_stride_d,
        o_stride_b, o_stride_h, o_stride_s, o_stride_d,
        lse_stride_b, lse_stride_h, lse_stride_s,
        num_stages=3,
    )