import torch
import triton
import triton.language as tl
import math


@triton.jit
def _dKdV_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    total_S, d, B, H, scale, stride_s, stride_bh, BLOCK: tl.constexpr
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
    S = total_S // (B * H)
    T_r = triton.cdiv(S, BLOCK)
    
    s_arange = tl.arange(0, BLOCK)
    d_arange_0 = tl.arange(0, 64)
    d_arange_1 = tl.arange(64, 128)
    
    sh_Q_0 = extern.shared_array("Q_0", BLOCK * 64)
    sh_Q_1 = extern.shared_array("Q_1", BLOCK * 64)
    sh_K_0 = extern.shared_array("K_0", BLOCK * 64)
    sh_K_1 = extern.shared_array("K_1", BLOCK * 64)
    sh_V_0 = extern.shared_array("V_0", BLOCK * 64)
    sh_V_1 = extern.shared_array("V_1", BLOCK * 64)
    sh_dO_0 = extern.shared_array("dO_0", BLOCK * 64)
    sh_dO_1 = extern.shared_array("dO_1", BLOCK * 64)
    sh_O_0 = extern.shared_array("O_0", BLOCK * 64)
    sh_O_1 = extern.shared_array("O_1", BLOCK * 64)
    sh_L = extern.shared_array("L", BLOCK)
    sh_D = extern.shared_array("D", BLOCK)
    
    row_offset_j = (b * H + h) * S + j * BLOCK
    row_mask_j = ((j * BLOCK + s_arange) < S).to(tl.int1)[:, None]
    
    K_0 = tl.load(K_ptr + row_offset_j * stride_s + s_arange[:, None] * stride_s + d_arange_0[None, :] * 1, mask=row_mask_j, other=0.0)
    K_1 = tl.load(K_ptr + row_offset_j * stride_s + s_arange[:, None] * stride_s + d_arange_1[None, :] * 1, mask=row_mask_j, other=0.0)
    V_0 = tl.load(V_ptr + row_offset_j * stride_s + s_arange[:, None] * stride_s + d_arange_0[None, :] * 1, mask=row_mask_j, other=0.0)
    V_1 = tl.load(V_ptr + row_offset_j * stride_s + s_arange[:, None] * stride_s + d_arange_1[None, :] * 1, mask=row_mask_j, other=0.0)
    
    tl.store(sh_K_0 + s_arange[:, None] * 64 + d_arange_0[None, :] * 1, K_0)
    tl.store(sh_K_1 + s_arange[:, None] * 64 + d_arange_1[None, :] * 1, K_1)
    tl.store(sh_V_0 + s_arange[:, None] * 64 + d_arange_0[None, :] * 1, V_0)
    tl.store(sh_V_1 + s_arange[:, None] * 64 + d_arange_1[None, :] * 1, V_1)
    
    dK_0_acc = tl.zeros((BLOCK, 64), dtype=tl.float32)
    dK_1_acc = tl.zeros((BLOCK, 64), dtype=tl.float32)
    dV_0_acc = tl.zeros((BLOCK, 64), dtype=tl.float32)
    dV_1_acc = tl.zeros((BLOCK, 64), dtype=tl.float32)
    
    for i in range(j, T_r):
        row_offset_i = (b * H + h) * S + i * BLOCK
        row_mask_i = ((i * BLOCK + s_arange) < S).to(tl.int1)[:, None]
        
        Q_0 = tl.load(Q_ptr + row_offset_i * stride_s + s_arange[:, None] * stride_s + d_arange_0[None, :] * 1, mask=row_mask_i, other=0.0)
        Q_1 = tl.load(Q_ptr + row_offset_i * stride_s + s_arange[:, None] * stride_s + d_arange_1[None, :] * 1, mask=row_mask_i, other=0.0)
        
        dO_0 = tl.load(dO_ptr + row_offset_i * stride_s + s_arange[:, None] * stride_s + d_arange_0[None, :] * 1, mask=row_mask_i, other=0.0)
        dO_1 = tl.load(dO_ptr + row_offset_i * stride_s + s_arange[:, None] * stride_s + d_arange_1[None, :] * 1, mask=row_mask_i, other=0.0)
        
        O_0 = tl.load(O_ptr + row_offset_i * stride_s + s_arange[:, None] * stride_s + d_arange_0[None, :] * 1, mask=row_mask_i, other=0.0)
        O_1 = tl.load(O_ptr + row_offset_i * stride_s + s_arange[:, None] * stride_s + d_arange_1[None, :] * 1, mask=row_mask_i, other=0.0)
        
        L_i = tl.load(L_ptr + row_offset_i + s_arange, 
                       mask=((i * BLOCK + s_arange) < S).to(tl.int1), other=0.0)
                       
        tl.store(sh_Q_0 + s_arange[:, None] * 64 + d_arange_0[None, :] * 1, Q_0)
        tl.store(sh_Q_1 + s_arange[:, None] * 64 + d_arange_1[None, :] * 1, Q_1)
        tl.store(sh_dO_0 + s_arange[:, None] * 64 + d_arange_0[None, :] * 1, dO_0)
        tl.store(sh_dO_1 + s_arange[:, None] * 64 + d_arange_1[None, :] * 1, dO_1)
        tl.store(sh_O_0 + s_arange[:, None] * 64 + d_arange_0[None, :] * 1, O_0)
        tl.store(sh_O_1 + s_arange[:, None] * 64 + d_arange_1[None, :] * 1, O_1)
        tl.store(sh_L + s_arange, L_i)
        
        tl.sync()
        
        Q_0 = tl.load(sh_Q_0 + s_arange[:, None] * 64 + d_arange_0[None, :] * 1)
        Q_1 = tl.load(sh_Q_1 + s_arange[:, None] * 64 + d_arange_1[None, :] * 1)
        dO_0 = tl.load(sh_dO_0 + s_arange[:, None] * 64 + d_arange_0[None, :] * 1)
        dO_1 = tl.load(sh_dO_1 + s_arange[:, None] * 64 + d_arange_1[None, :] * 1)
        O_0 = tl.load(sh_O_0 + s_arange[:, None] * 64 + d_arange_0[None, :] * 1)
        O_1 = tl.load(sh_O_1 + s_arange[:, None] * 64 + d_arange_1[None, :] * 1)
        L_i = tl.load(sh_L + s_arange)
        
        D_i = ((O_0 * dO_0).to(tl.float32)).sum(1, keep_dims=True) + ((O_1 * dO_1).to(tl.float32)).sum(1, keep_dims=True)
        tl.store(sh_D + s_arange, D_i[:, 0])
        
        K_0 = tl.load(sh_K_0 + s_arange[:, None] * 64 + d_arange_0[None, :] * 1)
        K_1 = tl.load(sh_K_1 + s_arange[:, None] * 64 + d_arange_1[None, :] * 1)
        V_0 = tl.load(sh_V_0 + s_arange[:, None] * 64 + d_arange_0[None, :] * 1)
        V_1 = tl.load(sh_V_1 + s_arange[:, None] * 64 + d_arange_1[None, :] * 1)
        
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
        
        tl.sync()
            
    tl.store((dK_ptr + row_offset_j * stride_s + s_arange[:, None] * stride_s + d_arange_0[None, :] * 1).to(tl.pointer_ty(tl.bfloat16)), dK_0_acc.to(tl.bfloat16), mask=row_mask_j)
    tl.store((dK_ptr + row_offset_j * stride_s + s_arange[:, None] * stride_s + d_arange_1[None, :] * 1).to(tl.pointer_ty(tl.bfloat16)), dK_1_acc.to(tl.bfloat16), mask=row_mask_j)
    
    tl.store((dV_ptr + row_offset_j * stride_s + s_arange[:, None] * stride_s + d_arange_0[None, :] * 1).to(tl.pointer_ty(tl.bfloat16)), dV_0_acc.to(tl.bfloat16), mask=row_mask_j)
    tl.store((dV_ptr + row_offset_j * stride_s + s_arange[:, None] * stride_s + d_arange_1[None, :] * 1).to(tl.pointer_ty(tl.bfloat16)), dV_1_acc.to(tl.bfloat16), mask=row_mask_j)


@triton.jit
def _dQ_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
    total_S, d, B, H, scale, stride_s, stride_bh, BLOCK: tl.constexpr
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
    S = total_S // (B * H)
    T_c = triton.cdiv(S, BLOCK)
    
    s_arange = tl.arange(0, BLOCK)
    d_arange_0 = tl.arange(0, 64)
    d_arange_1 = tl.arange(64, 128)
    
    sh_Q_0 = extern.shared_array("Q_0", BLOCK * 64)
    sh_Q_1 = extern.shared_array("Q_1", BLOCK * 64)
    sh_K_0 = extern.shared_array("K_0", BLOCK * 64)
    sh_K_1 = extern.shared_array("K_1", BLOCK * 64)
    sh_V_0 = extern.shared_array("V_0", BLOCK * 64)
    sh_V_1 = extern.shared_array("V_1", BLOCK * 64)
    sh_dO_0 = extern.shared_array("dO_0", BLOCK * 64)
    sh_dO_1 = extern.shared_array("dO_1", BLOCK * 64)
    sh_O_0 = extern.shared_array("O_0", BLOCK * 64)
    sh_O_1 = extern.shared_array("O_1", BLOCK * 64)
    sh_L = extern.shared_array("L", BLOCK)
    sh_D = extern.shared_array("D", BLOCK)
    
    row_offset_i = (b * H + h) * S + i * BLOCK
    row_mask_i = ((i * BLOCK + s_arange) < S).to(tl.int1)[:, None]
    
    Q_0 = tl.load(Q_ptr + row_offset_i * stride_s + s_arange[:, None] * stride_s + d_arange_0[None, :] * 1, mask=row_mask_i, other=0.0)
    Q_1 = tl.load(Q_ptr + row_offset_i * stride_s + s_arange[:, None] * stride_s + d_arange_1[None, :] * 1, mask=row_mask_i, other=0.0)
    
    dO_0 = tl.load(dO_ptr + row_offset_i * stride_s + s_arange[:, None] * stride_s + d_arange_0[None, :] * 1, mask=row_mask_i, other=0.0)
    dO_1 = tl.load(dO_ptr + row_offset_i * stride_s + s_arange[:, None] * stride_s + d_arange_1[None, :] * 1, mask=row_mask_i, other=0.0)
    
    O_0 = tl.load(O_ptr + row_offset_i * stride_s + s_arange[:, None] * stride_s + d_arange_0[None, :] * 1, mask=row_mask_i, other=0.0)
    O_1 = tl.load(O_ptr + row_offset_i * stride_s + s_arange[:, None] * stride_s + d_arange_1[None, :] * 1, mask=row_mask_i, other=0.0)
    
    L_i = tl.load(L_ptr + row_offset_i + s_arange, 
                   mask=((i * BLOCK + s_arange) < S).to(tl.int1), other=0.0)
                   
    tl.store(sh_Q_0 + s_arange[:, None] * 64 + d_arange_0[None, :] * 1, Q_0)
    tl.store(sh_Q_1 + s_arange[:, None] * 64 + d_arange_1[None, :] * 1, Q_1)
    tl.store(sh_dO_0 + s_arange[:, None] * 64 + d_arange_0[None, :] * 1, dO_0)
    tl.store(sh_dO_1 + s_arange[:, None] * 64 + d_arange_1[None, :] * 1, dO_1)
    tl.store(sh_O_0 + s_arange[:, None] * 64 + d_arange_0[None, :] * 1, O_0)
    tl.store(sh_O_1 + s_arange[:, None] * 64 + d_arange_1[None, :] * 1, O_1)
    tl.store(sh_L + s_arange, L_i)
    
    tl.sync()
    
    Q_0 = tl.load(sh_Q_0 + s_arange[:, None] * 64 + d_arange_0[None, :] * 1)
    Q_1 = tl.load(sh_Q_1 + s_arange[:, None] * 64 + d_arange_1[None, :] * 1)
    dO_0 = tl.load(sh_dO_0 + s_arange[:, None] * 64 + d_arange_0[None, :] * 1)
    dO_1 = tl.load(sh_dO_1 + s_arange[:, None] * 64 + d_arange_1[None, :] * 1)
    O_0 = tl.load(sh_O_0 + s_arange[:, None] * 64 + d_arange_0[None, :] * 1)
    O_1 = tl.load(sh_O_1 + s_arange[:, None] * 64 + d_arange_1[None, :] * 1)
    L_i = tl.load(sh_L + s_arange)
    
    D_i = ((O_0 * dO_0).to(tl.float32)).sum(1, keep_dims=True) + ((O_1 * dO_1).to(tl.float32)).sum(1, keep_dims=True)
    tl.store(sh_D + s_arange, D_i[:, 0])
    
    dQ_0_acc = tl.zeros((BLOCK, 64), dtype=tl.float32)
    dQ_1_acc = tl.zeros((BLOCK, 64), dtype=tl.float32)
    
    for j in range(0, min(i + 1, T_c)):
        row_offset_j = (b * H + h) * S + j * BLOCK
        row_mask_j = ((j * BLOCK + s_arange) < S).to(tl.int1)[:, None]
        
        K_0 = tl.load(K_ptr + row_offset_j * stride_s + s_arange[:, None] * stride_s + d_arange_0[None, :] * 1, mask=row_mask_j, other=0.0)
        K_1 = tl.load(K_ptr + row_offset_j * stride_s + s_arange[:, None] * stride_s + d_arange_1[None, :] * 1, mask=row_mask_j, other=0.0)
        
        V_0 = tl.load(V_ptr + row_offset_j * stride_s + s_arange[:, None] * stride_s + d_arange_0[None, :] * 1, mask=row_mask_j, other=0.0)
        V_1 = tl.load(V_ptr + row_offset_j * stride_s + s_arange[:, None] * stride_s + d_arange_1[None, :] * 1, mask=row_mask_j, other=0.0)
        
        tl.store(sh_K_0 + s_arange[:, None] * 64 + d_arange_0[None, :] * 1, K_0)
        tl.store(sh_K_1 + s_arange[:, None] * 64 + d_arange_1[None, :] * 1, K_1)
        tl.store(sh_V_0 + s_arange[:, None] * 64 + d_arange_0[None, :] * 1, V_0)
        tl.store(sh_V_1 + s_arange[:, None] * 64 + d_arange_1[None, :] * 1, V_1)
        
        tl.sync()
        
        K_0 = tl.load(sh_K_0 + s_arange[:, None] * 64 + d_arange_0[None, :] * 1)
        K_1 = tl.load(sh_K_1 + s_arange[:, None] * 64 + d_arange_1[None, :] * 1)
        V_0 = tl.load(sh_V_0 + s_arange[:, None] * 64 + d_arange_0[None, :] * 1)
        V_1 = tl.load(sh_V_1 + s_arange[:, None] * 64 + d_arange_1[None, :] * 1)
        
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
        
        tl.sync()
            
    tl.store((dQ_ptr + row_offset_i * stride_s + s_arange[:, None] * stride_s + d_arange_0[None, :] * 1).to(tl.pointer_ty(tl.bfloat16)), dQ_0_acc.to(tl.bfloat16), mask=row_mask_i)
    tl.store((dQ_ptr + row_offset_i * stride_s + s_arange[:, None] * stride_s + d_arange_1[None, :] * 1).to(tl.pointer_ty(tl.bfloat16)), dQ_1_acc.to(tl.bfloat16), mask=row_mask_i)


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
    
    stride_s = d
    stride_bh = d * S
    
    _dKdV_kernel[grid](
        Q.data_ptr(), K.data_ptr(), V.data_ptr(), O.data_ptr(), dO.data_ptr(), 
        L.data_ptr(), dK.data_ptr(), dV.data_ptr(),
        B * H * S, d, B, H, scale, stride_s, stride_bh, BLOCK=BLOCK, num_warps=4, num_stages=2
    )
    
    _dQ_kernel[grid](
        Q.data_ptr(), K.data_ptr(), V.data_ptr(), O.data_ptr(), dO.data_ptr(), 
        L.data_ptr(), dQ.data_ptr(),
        B * H * S, d, B, H, scale, stride_s, stride_bh, BLOCK=BLOCK, num_warps=4, num_stages=2
    )