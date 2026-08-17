import torch
import triton
import triton.language as tl
import math


@triton.jit
def _dKdV_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dK_ptr, dV_ptr,
    S, scale,
    s_stride, d_stride, l_stride,
    BLOCK: tl.constexpr,
):
    """
    Attention Backward Pass Kernel 1: Computes dK and dV.
    
    Logic:
    - Iterates over Key/Value blocks (j) in the outer loop.
    - Iterates over Query blocks (i >= j) in the inner loop to respect causal masking.
    - Accumulates gradients for K and V across all valid Q blocks.
    """
    tid = tl.program_id(0)
    pid_y = tl.program_id(1)
    
    b = pid_y // 48
    h = pid_y % 48
    
    j = tid
    T_r = (S + BLOCK - 1) // BLOCK
    
    q_ptr = Q_ptr + (b * s_stride + h * s_stride)
    k_ptr = K_ptr + (b * s_stride + h * s_stride)
    v_ptr = V_ptr + (b * s_stride + h * s_stride)
    o_ptr = O_ptr + (b * s_stride + h * s_stride)
    do_ptr = dO_ptr + (b * s_stride + h * s_stride)
    
    dK_ptr = dK_ptr + (b * s_stride + h * s_stride)
    dV_ptr = dV_ptr + (b * s_stride + h * s_stride)
    
    smem_ptr = extern.shared.data()
    sh_Q_0 = smem_ptr + 0 * 16384
    sh_Q_1 = smem_ptr + 1 * 16384
    sh_K_0 = smem_ptr + 2 * 16384
    sh_K_1 = smem_ptr + 3 * 16384
    sh_V_0 = smem_ptr + 4 * 16384
    sh_V_1 = smem_ptr + 5 * 16384
    sh_dO_0 = smem_ptr + 6 * 16384
    sh_dO_1 = smem_ptr + 7 * 16384
    sh_O_0 = smem_ptr + 8 * 16384
    sh_O_1 = smem_ptr + 9 * 16384
    sh_L = smem_ptr + 10 * 16384
    
    s_arange = tl.arange(0, BLOCK)
    d_arange_0 = tl.arange(0, BLOCK // 2)
    d_arange_1 = tl.arange(BLOCK // 2, BLOCK)
    
    row_mask_j = ((j * BLOCK + s_arange) < S).to(tl.int32)[:, None]
    
    K_0 = tl.load((k_ptr + j * BLOCK * s_stride + s_arange[:, None] * s_stride + d_arange_0[None, :] * d_stride).to(tl.pointer_ty(tl.bfloat16)), 
                  mask=(row_mask_j[:, None]).to(tl.bfloat16), other=0.0)
    K_1 = tl.load((k_ptr + j * BLOCK * s_stride + s_arange[:, None] * s_stride + d_arange_1[None, :] * d_stride).to(tl.pointer_ty(tl.bfloat16)), 
                  mask=(row_mask_j[:, None]).to(tl.bfloat16), other=0.0)
    V_0 = tl.load((v_ptr + j * BLOCK * s_stride + s_arange[:, None] * s_stride + d_arange_0[None, :] * d_stride).to(tl.pointer_ty(tl.bfloat16)), 
                  mask=(row_mask_j[:, None]).to(tl.bfloat16), other=0.0)
    V_1 = tl.load((v_ptr + j * BLOCK * s_stride + s_arange[:, None] * s_stride + d_arange_1[None, :] * d_stride).to(tl.pointer_ty(tl.bfloat16)), 
                  mask=(row_mask_j[:, None]).to(tl.bfloat16), other=0.0)
    
    tl.store((sh_K_0 + s_arange[:, None] * 128 + d_arange_0[None, :] * 2), K_0)
    tl.store((sh_K_1 + s_arange[:, None] * 128 + d_arange_1[None, :] * 2), K_1)
    tl.store((sh_V_0 + s_arange[:, None] * 128 + d_arange_0[None, :] * 2), V_0)
    tl.store((sh_V_1 + s_arange[:, None] * 128 + d_arange_1[None, :] * 2), V_1)
    tl.enforce_seq()
    tl.sync()
    
    dK_0_acc = tl.zeros((BLOCK, BLOCK // 2), dtype=tl.float32)
    dK_1_acc = tl.zeros((BLOCK, BLOCK // 2), dtype=tl.float32)
    dV_0_acc = tl.zeros((BLOCK, BLOCK // 2), dtype=tl.float32)
    dV_1_acc = tl.zeros((BLOCK, BLOCK // 2), dtype=tl.float32)
    
    for i in range(j, T_r):
        row_mask_i = ((i * BLOCK + s_arange) < S).to(tl.int32)[:, None]
        
        Q_0 = tl.load((q_ptr + i * BLOCK * s_stride + s_arange[:, None] * s_stride + d_arange_0[None, :] * d_stride).to(tl.pointer_ty(tl.bfloat16)), 
                      mask=(row_mask_i).to(tl.bfloat16), other=0.0)
        Q_1 = tl.load((q_ptr + i * BLOCK * s_stride + s_arange[:, None] * s_stride + d_arange_1[None, :] * d_stride).to(tl.pointer_ty(tl.bfloat16)), 
                      mask=(row_mask_i).to(tl.bfloat16), other=0.0)
        
        tl.store((sh_Q_0 + s_arange[:, None] * 128 + d_arange_0[None, :] * 2), Q_0)
        tl.store((sh_Q_1 + s_arange[:, None] * 128 + d_arange_1[None, :] * 2), Q_1)
        
        dO_0 = tl.load((do_ptr + i * BLOCK * s_stride + s_arange[:, None] * s_stride + d_arange_0[None, :] * d_stride).to(tl.pointer_ty(tl.bfloat16)), 
                       mask=(row_mask_i).to(tl.bfloat16), other=0.0)
        dO_1 = tl.load((do_ptr + i * BLOCK * s_stride + s_arange[:, None] * s_stride + d_arange_1[None, :] * d_stride).to(tl.pointer_ty(tl.bfloat16)), 
                       mask=(row_mask_i).to(tl.bfloat16), other=0.0)
        
        tl.store((sh_dO_0 + s_arange[:, None] * 128 + d_arange_0[None, :] * 2), dO_0)
        tl.store((sh_dO_1 + s_arange[:, None] * 128 + d_arange_1[None, :] * 2), dO_1)
        
        O_0 = tl.load((o_ptr + i * BLOCK * s_stride + s_arange[:, None] * s_stride + d_arange_0[None, :] * d_stride).to(tl.pointer_ty(tl.bfloat16)), 
                      mask=(row_mask_i).to(tl.bfloat16), other=0.0)
        O_1 = tl.load((o_ptr + i * BLOCK * s_stride + s_arange[:, None] * s_stride + d_arange_1[None, :] * d_stride).to(tl.pointer_ty(tl.bfloat16)), 
                      mask=(row_mask_i).to(tl.bfloat16), other=0.0)
        
        tl.store((sh_O_0 + s_arange[:, None] * 128 + d_arange_0[None, :] * 2), O_0)
        tl.store((sh_O_1 + s_arange[:, None] * 128 + d_arange_1[None, :] * 2), O_1)
        
        L_i = tl.load((L_ptr + b * S + h * S + i * BLOCK + s_arange).to(tl.pointer_ty(tl.float32)), 
                       mask=((i * BLOCK + s_arange) < S).to(tl.int32), other=0.0)
        tl.store((sh_L + s_arange), L_i)
        
        tl.enforce_seq()
        tl.sync()
        
        Q_0 = tl.load((sh_Q_0 + s_arange[:, None] * 128 + d_arange_0[None, :] * 2).to(tl.pointer_ty(tl.bfloat16)))
        Q_1 = tl.load((sh_Q_1 + s_arange[:, None] * 128 + d_arange_1[None, :] * 2).to(tl.pointer_ty(tl.bfloat16)))
        dO_0 = tl.load((sh_dO_0 + s_arange[:, None] * 128 + d_arange_0[None, :] * 2).to(tl.pointer_ty(tl.bfloat16)))
        dO_1 = tl.load((sh_dO_1 + s_arange[:, None] * 128 + d_arange_1[None, :] * 2).to(tl.pointer_ty(tl.bfloat16)))
        O_0 = tl.load((sh_O_0 + s_arange[:, None] * 128 + d_arange_0[None, :] * 2).to(tl.pointer_ty(tl.bfloat16)))
        O_1 = tl.load((sh_O_1 + s_arange[:, None] * 128 + d_arange_1[None, :] * 2).to(tl.pointer_ty(tl.bfloat16)))
        L_i = tl.load((sh_L + s_arange).to(tl.pointer_ty(tl.float32)), mask=((i * BLOCK + s_arange) < S).to(tl.int32), other=0.0)
        
        D_i = ((O_0 * dO_0).to(tl.float32)).sum(1, keep_dims=True) + ((O_1 * dO_1).to(tl.float32)).sum(1, keep_dims=True)
        
        K_0 = tl.load((sh_K_0 + s_arange[:, None] * 128 + d_arange_0[None, :] * 2).to(tl.pointer_ty(tl.bfloat16)))
        K_1 = tl.load((sh_K_1 + s_arange[:, None] * 128 + d_arange_1[None, :] * 2).to(tl.pointer_ty(tl.bfloat16)))
        V_0 = tl.load((sh_V_0 + s_arange[:, None] * 128 + d_arange_0[None, :] * 2).to(tl.pointer_ty(tl.bfloat16)))
        V_1 = tl.load((sh_V_1 + s_arange[:, None] * 128 + d_arange_1[None, :] * 2).to(tl.pointer_ty(tl.bfloat16)))
        
        S_acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
        S_reg = tl.dot(Q_0, K_0.T, S_acc)
        S_reg += tl.dot(Q_1, K_1.T, S_reg)
        
        dP_acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
        dP_reg = tl.dot(dO_0, V_0.T, dP_acc)
        dP_reg += tl.dot(dO_1, V_1.T, dP_reg)
        
        i_s = (i * BLOCK + s_arange[:, None]).to(tl.int32)
        j_s = (j * BLOCK + s_arange[None, :]).to(tl.int32)
        
        S_reg = tl.where(i_s >= j_s, S_reg * scale, 0.0)
        P_reg = tl.where(i_s >= j_s, tl.exp(S_reg - L_i), 0.0)
        dS_reg = P_reg * (dP_reg - D_i) * scale
        
        dS_T = dS_reg.T
        P_reg_T = P_reg.T
        
        dK_0_acc = tl.dot(dS_T, Q_0, dK_0_acc)
        dK_1_acc = tl.dot(dS_T, Q_1, dK_1_acc)
        dV_0_acc = tl.dot(P_reg_T, dO_0, dV_0_acc)
        dV_1_acc = tl.dot(P_reg_T, dO_1, dV_1_acc)
        
        if i + 1 < T_r:
            tl.sync()
            
    dK_ptr_0 = (dK_ptr + j * BLOCK * s_stride + s_arange[:, None] * s_stride + d_arange_0[None, :] * d_stride).to(tl.pointer_ty(tl.bfloat16))
    dK_ptr_1 = (dK_ptr + j * BLOCK * s_stride + s_arange[:, None] * s_stride + d_arange_1[None, :] * d_stride).to(tl.pointer_ty(tl.bfloat16))
    tl.store(dK_ptr_0, dK_0_acc.to(tl.bfloat16), mask=(row_mask_j).to(tl.bfloat16))
    tl.store(dK_ptr_1, dK_1_acc.to(tl.bfloat16), mask=(row_mask_j).to(tl.bfloat16))
    
    dV_ptr_0 = (dV_ptr + j * BLOCK * s_stride + s_arange[:, None] * s_stride + d_arange_0[None, :] * d_stride).to(tl.pointer_ty(tl.bfloat16))
    dV_ptr_1 = (dV_ptr + j * BLOCK * s_stride + s_arange[:, None] * s_stride + d_arange_1[None, :] * d_stride).to(tl.pointer_ty(tl.bfloat16))
    tl.store(dV_ptr_0, dV_0_acc.to(tl.bfloat16), mask=(row_mask_j).to(tl.bfloat16))
    tl.store(dV_ptr_1, dV_1_acc.to(tl.bfloat16), mask=(row_mask_j).to(tl.bfloat16))


@triton.jit
def _dQ_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dQ_ptr,
    S, scale,
    s_stride, d_stride, l_stride,
    BLOCK: tl.constexpr,
):
    """
    Attention Backward Pass Kernel 2: Computes dQ.
    
    Logic:
    - Iterates over Query blocks (i) in the outer loop.
    - Iterates over Key/Value blocks (j <= i) in the inner loop.
    - Accumulates gradients for Q across all valid KV blocks.
    """
    tid = tl.program_id(0)
    pid_y = tl.program_id(1)
    
    b = pid_y // 48
    h = pid_y % 48
    
    i = tid
    T_c = (S + BLOCK - 1) // BLOCK
    
    q_ptr = Q_ptr + (b * s_stride + h * s_stride)
    k_ptr = K_ptr + (b * s_stride + h * s_stride)
    v_ptr = V_ptr + (b * s_stride + h * s_stride)
    o_ptr = O_ptr + (b * s_stride + h * s_stride)
    do_ptr = dO_ptr + (b * s_stride + h * s_stride)
    
    dQ_ptr = dQ_ptr + (b * s_stride + h * s_stride)
    
    smem_ptr = extern.shared.data()
    sh_Q_0 = smem_ptr + 0 * 16384
    sh_Q_1 = smem_ptr + 1 * 16384
    sh_K_0 = smem_ptr + 2 * 16384
    sh_K_1 = smem_ptr + 3 * 16384
    sh_V_0 = smem_ptr + 4 * 16384
    sh_V_1 = smem_ptr + 5 * 16384
    sh_dO_0 = smem_ptr + 6 * 16384
    sh_dO_1 = smem_ptr + 7 * 16384
    sh_O_0 = smem_ptr + 8 * 16384
    sh_O_1 = smem_ptr + 9 * 16384
    sh_L = smem_ptr + 10 * 16384
    
    s_arange = tl.arange(0, BLOCK)
    d_arange_0 = tl.arange(0, BLOCK // 2)
    d_arange_1 = tl.arange(BLOCK // 2, BLOCK)
    
    row_mask_i = ((i * BLOCK + s_arange) < S).to(tl.int32)[:, None]
    
    Q_0 = tl.load((q_ptr + i * BLOCK * s_stride + s_arange[:, None] * s_stride + d_arange_0[None, :] * d_stride).to(tl.pointer_ty(tl.bfloat16)), 
                  mask=(row_mask_i).to(tl.bfloat16), other=0.0)
    Q_1 = tl.load((q_ptr + i * BLOCK * s_stride + s_arange[:, None] * s_stride + d_arange_1[None, :] * d_stride).to(tl.pointer_ty(tl.bfloat16)), 
                  mask=(row_mask_i).to(tl.bfloat16), other=0.0)
    
    tl.store((sh_Q_0 + s_arange[:, None] * 128 + d_arange_0[None, :] * 2), Q_0)
    tl.store((sh_Q_1 + s_arange[:, None] * 128 + d_arange_1[None, :] * 2), Q_1)
    
    dO_0 = tl.load((do_ptr + i * BLOCK * s_stride + s_arange[:, None] * s_stride + d_arange_0[None, :] * d_stride).to(tl.pointer_ty(tl.bfloat16)), 
                   mask=(row_mask_i).to(tl.bfloat16), other=0.0)
    dO_1 = tl.load((do_ptr + i * BLOCK * s_stride + s_arange[:, None] * s_stride + d_arange_1[None, :] * d_stride).to(tl.pointer_ty(tl.bfloat16)), 
                   mask=(row_mask_i).to(tl.bfloat16), other=0.0)
    
    tl.store((sh_dO_0 + s_arange[:, None] * 128 + d_arange_0[None, :] * 2), dO_0)
    tl.store((sh_dO_1 + s_arange[:, None] * 128 + d_arange_1[None, :] * 2), dO_1)
    
    O_0 = tl.load((o_ptr + i * BLOCK * s_stride + s_arange[:, None] * s_stride + d_arange_0[None, :] * d_stride).to(tl.pointer_ty(tl.bfloat16)), 
                  mask=(row_mask_i).to(tl.bfloat16), other=0.0)
    O_1 = tl.load((o_ptr + i * BLOCK * s_stride + s_arange[:, None] * s_stride + d_arange_1[None, :] * d_stride).to(tl.pointer_ty(tl.bfloat16)), 
                  mask=(row_mask_i).to(tl.bfloat16), other=0.0)
    
    tl.store((sh_O_0 + s_arange[:, None] * 128 + d_arange_0[None, :] * 2), O_0)
    tl.store((sh_O_1 + s_arange[:, None] * 128 + d_arange_1[None, :] * 2), O_1)
    
    L_i = tl.load((L_ptr + b * S + h * S + i * BLOCK + s_arange).to(tl.pointer_ty(tl.float32)), 
                   mask=((i * BLOCK + s_arange) < S).to(tl.int32), other=0.0)
    tl.store((sh_L + s_arange), L_i)
    
    tl.enforce_seq()
    tl.sync()
    
    Q_0 = tl.load((sh_Q_0 + s_arange[:, None] * 128 + d_arange_0[None, :] * 2).to(tl.pointer_ty(tl.bfloat16)))
    Q_1 = tl.load((sh_Q_1 + s_arange[:, None] * 128 + d_arange_1[None, :] * 2).to(tl.pointer_ty(tl.bfloat16)))
    dO_0 = tl.load((sh_dO_0 + s_arange[:, None] * 128 + d_arange_0[None, :] * 2).to(tl.pointer_ty(tl.bfloat16)))
    dO_1 = tl.load((sh_dO_1 + s_arange[:, None] * 128 + d_arange_1[None, :] * 2).to(tl.pointer_ty(tl.bfloat16)))
    O_0 = tl.load((sh_O_0 + s_arange[:, None] * 128 + d_arange_0[None, :] * 2).to(tl.pointer_ty(tl.bfloat16)))
    O_1 = tl.load((sh_O_1 + s_arange[:, None] * 128 + d_arange_1[None, :] * 2).to(tl.pointer_ty(tl.bfloat16)))
    L_i = tl.load((sh_L + s_arange).to(tl.pointer_ty(tl.float32)), mask=((i * BLOCK + s_arange) < S).to(tl.int32), other=0.0)
    
    D_i = ((O_0 * dO_0).to(tl.float32)).sum(1, keep_dims=True) + ((O_1 * dO_1).to(tl.float32)).sum(1, keep_dims=True)
    
    dQ_0_acc = tl.zeros((BLOCK, BLOCK // 2), dtype=tl.float32)
    dQ_1_acc = tl.zeros((BLOCK, BLOCK // 2), dtype=tl.float32)
    
    for j in range(0, i + 1):
        row_mask_j = ((j * BLOCK + s_arange) < S).to(tl.int32)[:, None]
        
        K_0 = tl.load((k_ptr + j * BLOCK * s_stride + s_arange[:, None] * s_stride + d_arange_0[None, :] * d_stride).to(tl.pointer_ty(tl.bfloat16)), 
                      mask=(row_mask_j).to(tl.bfloat16), other=0.0)
        K_1 = tl.load((k_ptr + j * BLOCK * s_stride + s_arange[:, None] * s_stride + d_arange_1[None, :] * d_stride).to(tl.pointer_ty(tl.bfloat16)), 
                      mask=(row_mask_j).to(tl.bfloat16), other=0.0)
        
        tl.store((sh_K_0 + s_arange[:, None] * 128 + d_arange_0[None, :] * 2), K_0)
        tl.store((sh_K_1 + s_arange[:, None] * 128 + d_arange_1[None, :] * 2), K_1)
        
        V_0 = tl.load((v_ptr + j * BLOCK * s_stride + s_arange[:, None] * s_stride + d_arange_0[None, :] * d_stride).to(tl.pointer_ty(tl.bfloat16)), 
                      mask=(row_mask_j).to(tl.bfloat16), other=0.0)
        V_1 = tl.load((v_ptr + j * BLOCK * s_stride + s_arange[:, None] * s_stride + d_arange_1[None, :] * d_stride).to(tl.pointer_ty(tl.bfloat16)), 
                      mask=(row_mask_j).to(tl.bfloat16), other=0.0)
        
        tl.store((sh_V_0 + s_arange[:, None] * 128 + d_arange_0[None, :] * 2), V_0)
        tl.store((sh_V_1 + s_arange[:, None] * 128 + d_arange_1[None, :] * 2), V_1)
        
        tl.enforce_seq()
        tl.sync()
        
        K_0 = tl.load((sh_K_0 + s_arange[:, None] * 128 + d_arange_0[None, :] * 2).to(tl.pointer_ty(tl.bfloat16)))
        K_1 = tl.load((sh_K_1 + s_arange[:, None] * 128 + d_arange_1[None, :] * 2).to(tl.pointer_ty(tl.bfloat16)))
        V_0 = tl.load((sh_V_0 + s_arange[:, None] * 128 + d_arange_0[None, :] * 2).to(tl.pointer_ty(tl.bfloat16)))
        V_1 = tl.load((sh_V_1 + s_arange[:, None] * 128 + d_arange_1[None, :] * 2).to(tl.pointer_ty(tl.bfloat16)))
        
        S_acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
        S_reg = tl.dot(Q_0, K_0.T, S_acc)
        S_reg += tl.dot(Q_1, K_1.T, S_reg)
        
        dP_acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
        dP_reg = tl.dot(dO_0, V_0.T, dP_acc)
        dP_reg += tl.dot(dO_1, V_1.T, dP_reg)
        
        i_s = (i * BLOCK + s_arange[:, None]).to(tl.int32)
        j_s = (j * BLOCK + s_arange[None, :]).to(tl.int32)
        
        S_reg = tl.where(i_s >= j_s, S_reg * scale, 0.0)
        P_reg = tl.where(i_s >= j_s, tl.exp(S_reg - L_i), 0.0)
        dS_reg = P_reg * (dP_reg - D_i) * scale
        
        dQ_0_acc = tl.dot(dS_reg, K_0, dQ_0_acc)
        dQ_1_acc = tl.dot(dS_reg, K_1, dQ_1_acc)
        
        if j + 1 <= i:
            tl.sync()
            
    dQ_ptr_0 = (dQ_ptr + i * BLOCK * s_stride + s_arange[:, None] * s_stride + d_arange_0[None, :] * d_stride).to(tl.pointer_ty(tl.bfloat16))
    dQ_ptr_1 = (dQ_ptr + i * BLOCK * s_stride + s_arange[:, None] * s_stride + d_arange_1[None, :] * d_stride).to(tl.pointer_ty(tl.bfloat16))
    tl.store(dQ_ptr_0, dQ_0_acc.to(tl.bfloat16), mask=(row_mask_i).to(tl.bfloat16))
    tl.store(dQ_ptr_1, dQ_1_acc.to(tl.bfloat16), mask=(row_mask_i).to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Implements the 2-pass backward for causal Multi-Head Attention.
    Matches reference semantics exactly while avoiding heavy memory usage.
    """
    torch.cuda.set_device(Q.device)
    
    b, h, s, d = Q.shape
    assert d == 128
    assert Q.dtype == torch.bfloat16
    
    scale = 1.0 / math.sqrt(d)
    
    s_stride = s.stride(2)
    d_stride = s.stride(3)
    l_stride = L.stride(2) if L.dim() == 3 else 1
    
    BLOCK = 128
    T_c = triton.cdiv(s, BLOCK)
    grid = (T_c, b * h)
    
    print(f"Launching dKdV with grid {grid}")
    _dKdV_kernel[grid](
        Q, K, V, O, dO, L, dK, dV,
        s, scale, s_stride, d_stride, l_stride, BLOCK=BLOCK, num_warps=4
    )
    print("Finished dKdV")
    
    print(f"Launching dQ with grid {grid}")
    _dQ_kernel[grid](
        Q, K, V, O, dO, L, dQ,
        s, scale, s_stride, d_stride, l_stride, BLOCK=BLOCK, num_warps=4
    )
    print("Finished dQ")