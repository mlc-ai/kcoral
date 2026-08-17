import torch
import triton
import triton.language as tl
import math

B = 4
H = 48
d = 128
BLOCK = 128

@triton.jit
def bwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    S, scale,
    stride_b, stride_h, stride_s, stride_d,
    BLOCK: tl.constexpr
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    
    s_off = tl.arange(0, BLOCK)
    d_off = tl.arange(0, BLOCK // 2)
    
    ptrs_Q = Q_ptr + (b * stride_b + h * stride_h) + s_off[:, None] * stride_s
    ptrs_K = K_ptr + (b * stride_b + h * stride_h) + s_off[:, None] * stride_s
    ptrs_V = V_ptr + (b * stride_b + h * stride_h) + s_off[:, None] * stride_s
    ptrs_O = O_ptr + (b * stride_b + h * stride_h) + s_off[:, None] * stride_s
    ptrs_dO = dO_ptr + (b * stride_b + h * stride_h) + s_off[:, None] * stride_s
    
    num_tiles = tl.cdiv(S, BLOCK)
    total_pairs = num_tiles * (num_tiles + 1) // 2
    
    dQ_acc_0 = tl.zeros((BLOCK, BLOCK), tl.float32)
    dQ_acc_1 = tl.zeros((BLOCK, BLOCK), tl.float32)
    dK_acc_0 = tl.zeros((BLOCK, BLOCK), tl.float32)
    dK_acc_1 = tl.zeros((BLOCK, BLOCK), tl.float32)
    dV_acc_0 = tl.zeros((BLOCK, BLOCK), tl.float32)
    dV_acc_1 = tl.zeros((BLOCK, BLOCK), tl.float32)
    
    computed_DO_or_i = [False] * num_tiles
    
    for t_idx in range(total_pairs):
        i_tile = (math.isqrt(1 + 8 * t_idx) - 1) // 2
        j_tile = t_idx - i_tile * (i_tile + 1) // 2
        
        i_start = i_tile * BLOCK
        j_start = j_tile * BLOCK
        
        s_mask_i = i_start + s_off < S
        s_mask_j = j_start + s_off < S
        
        Q_0 = tl.load(ptrs_Q + (i_start + s_off[:, None]) * stride_s + d_off[None, :] * stride_d, mask=s_mask_i[:, None], other=0.0, padding_option="zero")
        Q_1 = tl.load(ptrs_Q + (i_start + s_off[:, None]) * stride_s + (d_off[None, :] + 64) * stride_d, mask=s_mask_i[:, None], other=0.0, padding_option="zero")
        
        K_0 = tl.load(ptrs_K + (j_start + s_off[:, None]) * stride_s + d_off[None, :] * stride_d, mask=s_mask_j[:, None], other=0.0, padding_option="zero")
        K_1 = tl.load(ptrs_K + (j_start + s_off[:, None]) * stride_s + (d_off[None, :] + 64) * stride_d, mask=s_mask_j[:, None], other=0.0, padding_option="zero")
        
        V_0 = tl.load(ptrs_V + (j_start + s_off[:, None]) * stride_s + d_off[None, :] * stride_d, mask=s_mask_j[:, None], other=0.0, padding_option="zero")
        V_1 = tl.load(ptrs_V + (j_start + s_off[:, None]) * stride_s + (d_off[None, :] + 64) * stride_d, mask=s_mask_j[:, None], other=0.0, padding_option="zero")
        
        dO_0 = tl.load(ptrs_dO + (i_start + s_off[:, None]) * stride_s + d_off[None, :] * stride_d, mask=s_mask_i[:, None], other=0.0, padding_option="zero")
        dO_1 = tl.load(ptrs_dO + (i_start + s_off[:, None]) * stride_s + (d_off[None, :] + 64) * stride_d, mask=s_mask_i[:, None], other=0.0, padding_option="zero")
        
        S_val = tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)
        
        mask = ((i_start + s_off[:, None]) >= (j_start + s_off[None, :])) & s_mask_i[:, None]
        
        L_expanded = tl.load(L_ptr + off_L(b, h, i_start + s_off), mask=s_mask_i, other=0.0)
        A = tl.where(mask, tl.exp(S_val * scale - L_expanded[:, None]), 0.0)
        
        if not computed_DO_or_i[i_tile]:
            DO_OT_0 = tl.load(ptrs_dO + (i_start + s_off[:, None]) * stride_s + d_off[None, :] * stride_d, mask=s_mask_i[:, None], other=0.0, padding_option="zero")
            DO_OT_1 = tl.load(ptrs_dO + (i_start + s_off[:, None]) * stride_s + (d_off[None, :] + 64) * stride_d, mask=s_mask_i[:, None], other=0.0, padding_option="zero")
            D_O_O = DO_OT_0 @ DO_OT_1.T
            computed_DO_or_i[i_tile] = True
            
        dP = dO_0 @ V_0.T + dO_1 @ V_1.T
        dA = A * (dP - D_O_O)
        
        dQ_acc_0 = dQ_acc_0 + dA_0 @ K_0
        dQ_acc_1 = dQ_acc_1 + dA_1 @ K_1
        dK_acc_0 = dK_acc_0 + dA_0.T @ Q_0
        dK_acc_1 = dK_acc_1 + dA_1.T @ Q_1
        dV_acc_0 = dV_acc_0 + A_0.T @ dO_0
        dV_acc_1 = dV_acc_1 + A_1.T @ dO_1
        
    dQ_acc_0 = dQ_acc_0 * scale
    dQ_acc_1 = dQ_acc_1 * scale
    dK_acc_0 = dK_acc_0 * scale
    dK_acc_1 = dK_acc_1 * scale
    
    ptrs_dQ = dQ_ptr + (b * stride_b + h * stride_h) + s_off[:, None] * stride_s
    ptrs_dK = dK_ptr + (b * stride_b + h * stride_h) + s_off[:, None] * stride_s
    ptrs_dV = dV_ptr + (b * stride_b + h * stride_h) + s_off[:, None] * stride_s
    
    s_mask = s_off < S
    
    tl.store(ptrs_dQ + d_off[None, :] * stride_d, dQ_acc_0, mask=s_mask[:, None])
    tl.store(ptrs_dQ + (d_off[None, :] + 64) * stride_d, dQ_acc_1, mask=s_mask[:, None])
    tl.store(ptrs_dK + d_off[None, :] * stride_d, dK_acc_0, mask=s_mask[:, None])
    tl.store(ptrs_dK + (d_off[None, :] + 64) * stride_d, dK_acc_1, mask=s_mask[:, None])
    tl.store(ptrs_dV + d_off[None, :] * stride_d, dV_acc_0, mask=s_mask[:, None])
    tl.store(ptrs_dV + (d_off[None, :] + 64) * stride_d, dV_acc_1, mask=s_mask[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    S = Q.shape[2]
    
    dQ = dQ.to(torch.bfloat16)
    dK = dK.to(torch.bfloat16)
    dV = dV.to(torch.bfloat16)
    
    scale = 1.0 / math.sqrt(128)
    
    grid = (B, H)
    stride_b = H * S * d
    stride_h = S * d
    stride_s = d
    stride_d = 1
    
    bwd_kernel[grid](
        Q, K, V, O, dO, L,
        dQ, dK, dV,
        S, scale,
        stride_b, stride_h, stride_s, stride_d,
        BLOCK=BLOCK,
        num_warps=4,
    )