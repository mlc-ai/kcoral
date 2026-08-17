import torch
import triton
import triton.language as tl
import math
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _dKdV_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc,
    L_ptr, dK_desc, dV_desc,
    S, H, scale, BLOCK: tl.constexpr
):
    """
    Iterates over Key/Value blocks (j) and accumulates over valid Query blocks (i >= j).
    Computes gradients dK and dV.
    """
    tid = tl.program_id(0)
    pid_y = tl.program_id(1)
    
    b = pid_y // H
    h = pid_y % H
    
    j = tid
    T_r = triton.cdiv(S, BLOCK)
    
    s_arange = tl.arange(0, BLOCK)
    row_offset_j = (b * H + h) * S + j * BLOCK
    
    K_0 = K_desc.load([row_offset_j, 0])
    K_1 = K_desc.load([row_offset_j, 64])
    V_0 = V_desc.load([row_offset_j, 0])
    V_1 = V_desc.load([row_offset_j, 64])
    
    dK_0_acc = tl.zeros((BLOCK, 64), dtype=tl.float32)
    dK_1_acc = tl.zeros((BLOCK, 64), dtype=tl.float32)
    dV_0_acc = tl.zeros((BLOCK, 64), dtype=tl.float32)
    dV_1_acc = tl.zeros((BLOCK, 64), dtype=tl.float32)
    
    for i in range(j, T_r):
        row_offset_i = (b * H + h) * S + i * BLOCK
        
        Q_0 = Q_desc.load([row_offset_i, 0])
        Q_1 = Q_desc.load([row_offset_i, 64])
        
        dO_0 = dO_desc.load([row_offset_i, 0])
        dO_1 = dO_desc.load([row_offset_i, 64])
        
        O_0 = O_desc.load([row_offset_i, 0])
        O_1 = O_desc.load([row_offset_i, 64])
        
        bh = b * H + h
        L_i = tl.load(L_ptr + bh * S + i * BLOCK + s_arange, 
                       mask=((i * BLOCK + s_arange) < S).to(tl.int1), other=0.0)
                       
        D_i = ((O_0 * dO_0).to(tl.float32)).sum(1, keep_dims=True) + ((O_1 * dO_1).to(tl.float32)).sum(1, keep_dims=True)
        
        S_acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
        S_reg = tl.dot(Q_0, K_0.T, S_acc)
        S_reg += tl.dot(Q_1, K_1.T, S_reg)
        
        dP_acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
        dP_reg = tl.dot(dO_0, V_0.T, dP_acc)
        dP_reg += tl.dot(dO_1, V_1.T, dP_reg)
        
        i_s = (i * BLOCK + s_arange[:, None]).to(tl.int32)
        j_s = (j * BLOCK + s_arange[None, :]).to(tl.int32)
        
        S_reg = tl.where(i_s >= j_s, S_reg * scale, 0.0)
        P_reg = tl.where(i_s >= j_s, tl.exp(S_reg - L_i[:, None]), 0.0)
        dS_reg = P_reg * (dP_reg - D_i) * scale
        
        dS_T = dS_reg.T
        P_reg_T = P_reg.T
        
        dK_0_acc = tl.dot(dS_T, Q_0, dK_0_acc)
        dK_1_acc = tl.dot(dS_T, Q_1, dK_1_acc)
        dV_0_acc = tl.dot(P_reg_T, dO_0, dV_0_acc)
        dV_1_acc = tl.dot(P_reg_T, dO_1, dV_1_acc)
            
    dK_desc.store([row_offset_j, 0], dK_0_acc.to(tl.bfloat16), boundary_check=(0,))
    dK_desc.store([row_offset_j, 64], dK_1_acc.to(tl.bfloat16), boundary_check=(0,))
    dV_desc.store([row_offset_j, 0], dV_0_acc.to(tl.bfloat16), boundary_check=(0,))
    dV_desc.store([row_offset_j, 64], dV_1_acc.to(tl.bfloat16), boundary_check=(0,))


@triton.jit
def _dQ_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc,
    L_ptr, dQ_desc,
    S, H, scale, BLOCK: tl.constexpr
):
    """
    Iterates over Query blocks (i) and accumulates over valid Key/Value blocks (j <= i).
    Computes gradient dQ.
    """
    tid = tl.program_id(0)
    pid_y = tl.program_id(1)
    
    b = pid_y // H
    h = pid_y % H
    
    i = tid
    T_c = triton.cdiv(S, BLOCK)
    
    s_arange = tl.arange(0, BLOCK)
    row_offset_i = (b * H + h) * S + i * BLOCK
    
    Q_0 = Q_desc.load([row_offset_i, 0])
    Q_1 = Q_desc.load([row_offset_i, 64])
    
    dO_0 = dO_desc.load([row_offset_i, 0])
    dO_1 = dO_desc.load([row_offset_i, 64])
    
    O_0 = O_desc.load([row_offset_i, 0])
    O_1 = O_desc.load([row_offset_i, 64])
    
    bh = b * H + h
    L_i = tl.load(L_ptr + bh * S + i * BLOCK + s_arange, 
                   mask=((i * BLOCK + s_arange) < S).to(tl.int1), other=0.0)
                   
    D_i = ((O_0 * dO_0).to(tl.float32)).sum(1, keep_dims=True) + ((O_1 * dO_1).to(tl.float32)).sum(1, keep_dims=True)
    
    dQ_0_acc = tl.zeros((BLOCK, 64), dtype=tl.float32)
    dQ_1_acc = tl.zeros((BLOCK, 64), dtype=tl.float32)
    
    for j in range(0, min(i + 1, T_c)):
        row_offset_j = (b * H + h) * S + j * BLOCK
        
        K_0 = K_desc.load([row_offset_j, 0])
        K_1 = K_desc.load([row_offset_j, 64])
        
        V_0 = V_desc.load([row_offset_j, 0])
        V_1 = V_desc.load([row_offset_j, 64])
        
        S_acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
        S_reg = tl.dot(Q_0, K_0.T, S_acc)
        S_reg += tl.dot(Q_1, K_1.T, S_reg)
        
        dP_acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
        dP_reg = tl.dot(dO_0, V_0.T, dP_acc)
        dP_reg += tl.dot(dO_1, V_1.T, dP_reg)
        
        i_s = (i * BLOCK + s_arange[:, None]).to(tl.int32)
        j_s = (j * BLOCK + s_arange[None, :]).to(tl.int32)
        
        S_reg = tl.where(i_s >= j_s, S_reg * scale, 0.0)
        P_reg = tl.where(i_s >= j_s, tl.exp(S_reg - L_i[:, None]), 0.0)
        dS_reg = P_reg * (dP_reg - D_i) * scale
        
        dQ_0_acc = tl.dot(dS_reg, K_0, dQ_0_acc)
        dQ_1_acc = tl.dot(dS_reg, K_1, dQ_1_acc)
            
    dQ_desc.store([row_offset_i, 0], dQ_0_acc.to(tl.bfloat16), boundary_check=(0,))
    dQ_desc.store([row_offset_i, 64], dQ_1_acc.to(tl.bfloat16), boundary_check=(0,))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Implements the 2-pass backward for causal Multi-Head Attention.
    Matches reference semantics exactly while avoiding heavy memory usage.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    assert d == 128
    assert Q.dtype == torch.bfloat16
    
    scale = 1.0 / math.sqrt(d)
    
    BLOCK = 128
    T_c = triton.cdiv(S, BLOCK)
    grid = (T_c, B * H)
    
    Q_c = Q.view(B * H * S, d).contiguous()
    K_c = K.view(B * H * S, d).contiguous()
    V_c = V.view(B * H * S, d).contiguous()
    O_c = O.view(B * H * S, d).contiguous()
    dO_c = dO.view(B * H * S, d).contiguous()
    dQ_c = dQ.view(B * H * S, d).contiguous()
    dK_c = dK.view(B * H * S, d).contiguous()
    dV_c = dV.view(B * H * S, d).contiguous()
    
    Q_desc = TensorDescriptor.from_tensor(Q_c, [BLOCK, 128])
    K_desc = TensorDescriptor.from_tensor(K_c, [BLOCK, 128])
    V_desc = TensorDescriptor.from_tensor(V_c, [BLOCK, 128])
    O_desc = TensorDescriptor.from_tensor(O_c, [BLOCK, 128])
    dO_desc = TensorDescriptor.from_tensor(dO_c, [BLOCK, 128])
    
    dQ_desc = TensorDescriptor.from_tensor(dQ_c, [BLOCK, 128])
    dK_desc = TensorDescriptor.from_tensor(dK_c, [BLOCK, 128])
    dV_desc = TensorDescriptor.from_tensor(dV_c, [BLOCK, 128])
    
    Q_desc_T = TensorDescriptor.from_tensor(Q_c, [128, BLOCK])
    K_desc_T = TensorDescriptor.from_tensor(K_c, [128, BLOCK])
    V_desc_T = TensorDescriptor.from_tensor(V_c, [128, BLOCK])
    O_desc_T = TensorDescriptor.from_tensor(O_c, [128, BLOCK])
    dO_desc_T = TensorDescriptor.from_tensor(dO_c, [128, BLOCK])
    
    dQ_desc_T = TensorDescriptor.from_tensor(dQ_c, [128, BLOCK])
    dK_desc_T = TensorDescriptor.from_tensor(dK_c, [128, BLOCK])
    dV_desc_T = TensorDescriptor.from_tensor(dV_c, [128, BLOCK])
    
    _dKdV_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc,
        L.data_ptr(), dK_desc, dV_desc,
        S, H, scale, BLOCK=BLOCK, num_warps=4, num_stages=2
    )
    
    _dQ_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc,
        L.data_ptr(), dQ_desc,
        S, H, scale, BLOCK=BLOCK, num_warps=4, num_stages=2
    )