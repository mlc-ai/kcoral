import torch
import triton
import triton.language as tl
import math

NUM_SMS = 132
BLOCK = 128
D_DIM = 128

@triton.heuristics(values={"num_warps": 4, "num_stages": 2})
@triton.jit
def mha_bwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    S_len, scale, mode: tl.constexpr, T_r: int,
):
    total_bh = 192
    bh = tl.program_id(1)
    s_base = bh * S_len
    
    if mode == 0:  # dQ mode
        i = tl.program_id(0)
        q_row_base = s_base + i * BLOCK
        q_row = q_row_base + tl.arange(0, BLOCK)
        q_mask = q_row < (s_base + S_len)
        
        dQ0_acc = tl.zeros((BLOCK, 64), tl.float32)
        dQ1_acc = tl.zeros((BLOCK, 64), tl.float32)
        
        half0_d = tl.arange(0, 64)
        half1_d = tl.arange(0, 64)
        stride_Q_row = 128
        
        for j in range(i + 1):
            j_row_base = s_base + j * BLOCK
            j_row = j_row_base + tl.arange(0, BLOCK)
            j_mask = j_row < (s_base + S_len)
            
            q0_block = tl.load(Q_ptr + q_row[:, None] * stride_Q_row + half0_d[None, :], mask=q_mask[:, None], other=0.0)
            q1_block = tl.load(Q_ptr + q_row[:, None] * stride_Q_row + half1_d[None, :] + 64, mask=q_mask[:, None], other=0.0)
            
            do0_block = tl.load(dO_ptr + q_row[:, None] * stride_Q_row + half0_d[None, :], mask=q_mask[:, None], other=0.0)
            do1_block = tl.load(dO_ptr + q_row[:, None] * stride_Q_row + half1_d[None, :] + 64, mask=q_mask[:, None], other=0.0)
            
            o0_block = tl.load(O_ptr + q_row[:, None] * stride_Q_row + half0_d[None, :], mask=q_mask[:, None], other=0.0)
            o1_block = tl.load(O_ptr + q_row[:, None] * stride_Q_row + half1_d[None, :] + 64, mask=q_mask[:, None], other=0.0)
            
            D = (tl.sum(do0_block * o0_block, axis=1) + tl.sum(do1_block * o1_block, axis=1))
            D_exp = D[:, None]
            
            k0_block = tl.load(K_ptr + j_row[:, None] * stride_Q_row + half0_d[None, :], mask=j_mask[:, None], other=0.0)
            k1_block = tl.load(K_ptr + j_row[:, None] * stride_Q_row + half1_d[None, :] + 64, mask=j_mask[:, None], other=0.0)
            
            v0_block = tl.load(V_ptr + j_row[:, None] * stride_Q_row + half0_d[None, :], mask=j_mask[:, None], other=0.0)
            v1_block = tl.load(V_ptr + j_row[:, None] * stride_Q_row + half1_d[None, :] + 64, mask=j_mask[:, None], other=0.0)
            
            S = tl.dot(q0_block, k0_block, input_precision="ieee")
            S += tl.dot(q1_block, k1_block, input_precision="ieee")
            
            L_block = tl.load(L_ptr + q_row, mask=q_mask, other=0.0)
            L_exp = L_block[:, None]
            
            P = exp(S * scale - L_exp)
            
            q_global = q_row_base + tl.arange(0, BLOCK)
            j_global = j_row_base + tl.arange(0, BLOCK)
            causal_mask = q_global[:, None] >= j_global[None, :]
            causal_mask = causal_mask & (q_global[:, None] < s_base + S_len) & (j_global[None, :] < s_base + S_len)
            
            P = P * causal_mask
            
            dP = tl.dot(do0_block, v0_block, input_precision="ieee")
            dP += tl.dot(do1_block, v1_block, input_precision="ieee")
            
            dS = P * (dP - D_exp) * scale
            
            dQ0_acc += tl.dot(dS, k0_block, input_precision="ieee")
            dQ1_acc += tl.dot(dS, k1_block, input_precision="ieee")
        
        dQ0 = dQ0_acc.to(tl.bfloat16)
        dQ1 = dQ1_acc.to(tl.bfloat16)
        
        q_mask_s = q_row < (s_base + S_len)
        
        tl.store(dQ_ptr + q_row[:, None] * stride_Q_row + half0_d[None, :], dQ0, mask=q_mask_s[:, None])
        tl.store(dQ_ptr + q_row[:, None] * stride_Q_row + half1_d[None, :] + 64, dQ1, mask=q_mask_s[:, None])
    
    else:  # mode == 1, dKV mode
        j = tl.program_id(0)
        j_row_base = s_base + j * BLOCK
        j_row = j_row_base + tl.arange(0, BLOCK)
        j_mask = j_row < (s_base + S_len)
        
        dK0_acc = tl.zeros((BLOCK, 64), tl.float32)
        dK1_acc = tl.zeros((BLOCK, 64), tl.float32)
        dV0_acc = tl.zeros((BLOCK, 64), tl.float32)
        dV1_acc = tl.zeros((BLOCK, 64), tl.float32)
        
        half0_d = tl.arange(0, 64)
        half1_d = tl.arange(0, 64)
        stride_Q_row = 128
        
        k0_p = tl.load(K_ptr + j_row[:, None] * stride_Q_row + half0_d[None, :], mask=j_mask[:, None], other=0.0)
        k1_p = tl.load(K_ptr + j_row[:, None] * stride_Q_row + half1_d[None, :] + 64, mask=j_mask[:, None], other=0.0)
        v0_p = tl.load(V_ptr + j_row[:, None] * stride_Q_row + half0_d[None, :], mask=j_mask[:, None], other=0.0)
        v1_p = tl.load(V_ptr + j_row[:, None] * stride_Q_row + half1_d[None, :] + 64, mask=j_mask[:, None], other=0.0)
        
        for i in range(j, T_r):
            i_row_base = s_base + i * BLOCK
            i_row = i_row_base + tl.arange(0, BLOCK)
            i_mask = i_row < (s_base + S_len)
            
            q0_i = tl.load(Q_ptr + i_row[:, None] * stride_Q_row + half0_d[None, :], mask=i_mask[:, None], other=0.0)
            q1_i = tl.load(Q_ptr + i_row[:, None] * stride_Q_row + half1_d[None, :] + 64, mask=i_mask[:, None], other=0.0)
            
            do0_i = tl.load(dO_ptr + i_row[:, None] * stride_Q_row + half0_d[None, :], mask=i_mask[:, None], other=0.0)
            do1_i = tl.load(dO_ptr + i_row[:, None] * stride_Q_row + half1_d[None, :] + 64, mask=i_mask[:, None], other=0.0)
            
            o0_i = tl.load(O_ptr + i_row[:, None] * stride_Q_row + half0_d[None, :], mask=i_mask[:, None], other=0.0)
            o1_i = tl.load(O_ptr + i_row[:, None] * stride_Q_row + half1_d[None, :] + 64, mask=i_mask[:, None], other=0.0)
            
            D = (tl.sum(do0_i * o0_i, axis=1) + tl.sum(do1_i * o1_i, axis=1))
            D_exp = D[:, None]
            
            S = tl.dot(q0_i, k0_p, input_precision="ieee")
            S += tl.dot(q1_i, k1_p, input_precision="ieee")
            
            L_block = tl.load(L_ptr + i_row, mask=i_mask, other=0.0)
            L_exp = L_block[:, None]
            
            P = exp(S * scale - L_exp)
            
            i_global = i_row_base + tl.arange(0, BLOCK)
            j_global = j_row_base + tl.arange(0, BLOCK)
            causal_mask = i_global[:, None] >= j_global[None, :]
            causal_mask = causal_mask & (i_global[:, None] < s_base + S_len) & (j_global[None, :] < s_base + S_len)
            
            P = P * causal_mask
            
            dP = tl.dot(do0_i, v0_p, input_precision="ieee")
            dP += tl.dot(do1_i, v1_p, input_precision="ieee")
            
            dS = P * (dP - D_exp) * scale
            
            dV0_acc += tl.dot(P, do0_i, input_precision="ieee")
            dV1_acc += tl.dot(P, do1_i, input_precision="ieee")
            
            dK0_acc += tl.dot(dS, q0_i, input_precision="ieee")
            dK1_acc += tl.dot(dS, q1_i, input_precision="ieee")
        
        dK0 = dK0_acc.to(tl.bfloat16)
        dK1 = dK1_acc.to(tl.bfloat16)
        dV0 = dV0_acc.to(tl.bfloat16)
        dV1 = dV1_acc.to(tl.bfloat16)
        
        j_mask_s = j_row < (s_base + S_len)
        
        tl.store(dK_ptr + j_row[:, None] * stride_Q_row + half0_d[None, :], dK0, mask=j_mask_s[:, None])
        tl.store(dK_ptr + j_row[:, None] * stride_Q_row + half1_d[None, :] + 64, dK1, mask=j_mask_s[:, None])
        
        tl.store(dV_ptr + j_row[:, None] * stride_Q_row + half0_d[None, :], dV0, mask=j_mask_s[:, None])
        tl.store(dV_ptr + j_row[:, None] * stride_Q_row + half1_d[None, :] + 64, dV1, mask=j_mask_s[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute backward attention gradients with destination-passing outputs."""
    B, H, S_len, d = Q.shape
    device = Q.device
    torch.cuda.set_device(device)
    
    scale = 1.0 / math.sqrt(D_DIM)
    
    Q_ptr = Q.contiguous().view(B * H * S_len, 2, 64).cuda()
    K_ptr = K.contiguous().view(B * H * S_len, 2, 64).cuda()
    V_ptr = V.contiguous().view(B * H * S_len, 2, 64).cuda()
    O_ptr = O.contiguous().view(B * H * S_len, 2, 64).cuda()
    dO_ptr = dO.contiguous().view(B * H * S_len, 2, 64).cuda()
    L_ptr = L.view(B * H * S_len).cuda()
    
    dQ_ptr = dQ.contiguous().view(B * H * S_len, 2, 64).cuda()
    dK_ptr = dK.contiguous().view(B * H * S_len, 2, 64).cuda()
    dV_ptr = dV.contiguous().view(B * H * S_len, 2, 64).cuda()
    
    T_r = triton.cdiv(S_len, BLOCK)
    T_c = triton.cdiv(S_len, BLOCK)
    
    grid_dQ = (min(NUM_SMS, T_r), B * H)
    mha_bwd_kernel[grid_dQ](
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
        dQ_ptr, dK_ptr, dV_ptr,
        S_len, scale, mode=0, T_r=T_r
    )
    
    grid_dKV = (min(NUM_SMS, T_c), B * H)
    mha_bwd_kernel[grid_dKV](
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
        dQ_ptr, dK_ptr, dV_ptr,
        S_len, scale, mode=1, T_r=T_r
    )