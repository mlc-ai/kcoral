import torch
import triton
import triton.language as tl
import math


@triton.jit
def _dKdV_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dK_ptr, dV_ptr,
    S, scale, d,
    l_mult, head_stride, batch_head_stride,
    BLOCK: tl.constexpr,
):
    """
    Attention Backward Pass Kernel 1: Computes dK and dV.
    Iterates over Key/Value blocks (j) and accumulates over valid Query blocks (i >= j).
    """
    tid = tl.program_id(0)
    pid_y = tl.program_id(1)
    
    b = pid_y // 48
    h = pid_y % 48
    
    j = tid
    T_r = (S + BLOCK - 1) // BLOCK
    
    q_ptr = Q_ptr + b * batch_head_stride + h * head_stride
    k_ptr = K_ptr + b * batch_head_stride + h * head_stride
    v_ptr = V_ptr + b * batch_head_stride + h * head_stride
    o_ptr = O_ptr + b * batch_head_stride + h * head_stride
    do_ptr = dO_ptr + b * batch_head_stride + h * head_stride
    
    dK_ptr = dK_ptr + b * batch_head_stride + h * head_stride
    dV_ptr = dV_ptr + b * batch_head_stride + h * head_stride
    
    sh_Q_0 = extern.shared_array("Q_0", BLOCK * BLOCK // 2)
    sh_Q_1 = extern.shared_array("Q_1", BLOCK * BLOCK // 2)
    sh_K_0 = extern.shared_array("K_0", BLOCK * BLOCK // 2)
    sh_K_1 = extern.shared_array("K_1", BLOCK * BLOCK // 2)
    sh_V_0 = extern.shared_array("V_0", BLOCK * BLOCK // 2)
    sh_V_1 = extern.shared_array("V_1", BLOCK * BLOCK // 2)
    sh_dO_0 = extern.shared_array("dO_0", BLOCK * BLOCK // 2)
    sh_dO_1 = extern.shared_array("dO_1", BLOCK * BLOCK // 2)
    sh_O_0 = extern.shared_array("O_0", BLOCK * BLOCK // 2)
    sh_O_1 = extern.shared_array("O_1", BLOCK * BLOCK // 2)
    sh_L = extern.shared_array("L", BLOCK)
    
    s_arange = tl.arange(0, BLOCK)
    d_arange = tl.arange(0, 128)
    d_arange_0 = tl.arange(0, 64)
    d_arange_1 = tl.arange(64, 128)
    
    row_mask_j = ((j * BLOCK + s_arange) < S).to(tl.int1)[:, None]
    
    K = tl.load(k_ptr + j * BLOCK * d + s_arange[:, None] * d + d_arange[None, :] * 1, mask=row_mask_j, other=0.0)
    K_0, K_1 = tl.split(K, 2, dim=1)
    
    V = tl.load(v_ptr + j * BLOCK * d + s_arange[:, None] * d + d_arange[None, :] * 1, mask=row_mask_j, other=0.0)
    V_0, V_1 = tl.split(V, 2, dim=1)
    
    tl.store(sh_K_0 + s_arange[:, None] * 64 + d_arange_0[None, :] * 1, K_0)
    tl.store(sh_K_1 + s_arange[:, None] * 64 + d_arange_1[None, :] * 1, K_1)
    tl.store(sh_V_0 + s_arange[:, None] * 64 + d_arange_0[None, :] * 1, V_0)
    tl.store(sh_V_1 + s_arange[:, None] * 64 + d_arange_1[None, :] * 1, V_1)
    
    dK_0_acc = tl.zeros((BLOCK, BLOCK // 2), dtype=tl.float32)
    dK_1_acc = tl.zeros((BLOCK, BLOCK // 2), dtype=tl.float32)
    dV_0_acc = tl.zeros((BLOCK, BLOCK // 2), dtype=tl.float32)
    dV_1_acc = tl.zeros((BLOCK, BLOCK // 2), dtype=tl.float32)
    
    for i in range(j, T_r):
        row_mask_i = ((i * BLOCK + s_arange) < S).to(tl.int1)[:, None]
        
        Q = tl.load(q_ptr + i * BLOCK * d + s_arange[:, None] * d + d_arange[None, :] * 1, mask=row_mask_i, other=0.0)
        Q_0, Q_1 = tl.split(Q, 2, dim=1)
        
        dO = tl.load(do_ptr + i * BLOCK * d + s_arange[:, None] * d + d_arange[None, :] * 1, mask=row_mask_i, other=0.0)
        dO_0, dO_1 = tl.split(dO, 2, dim=1)
        
        O = tl.load(o_ptr + i * BLOCK * d + s_arange[:, None] * d + d_arange[None, :] * 1, mask=row_mask_i, other=0.0)
        O_0, O_1 = tl.split(O, 2, dim=1)
        
        L_i = tl.load(L_ptr + b * l_mult + h * S + i * BLOCK + s_arange, mask=((i * BLOCK + s_arange) < S).to(tl.int1), other=0.0)
        
        tl.store(sh_Q_0 + s_arange[:, None] * 64 + d_arange_0[None, :] * 1, Q_0)
        tl.store(sh_Q_1 + s_arange[:, None] * 64 + d_arange_1[None, :] * 1, Q_1)
        tl.store(sh_dO_0 + s_arange[:, None] * 64 + d_arange_0[None, :] * 1, dO_0)
        tl.store(sh_dO_1 + s_arange[:, None] * 64 + d_arange_1[None, :] * 1, dO_1)
        tl.store(sh_O_0 + s_arange[:, None] * 64 + d_arange_0[None, :] * 1, O_0)
        tl.store(sh_O_1 + s_arange[:, None] * 64 + d_arange_1[None, :] * 1, O_1)
        tl.store(sh_L + s_arange, L_i)
        
        Q_0 = tl.load(sh_Q_0 + s_arange[:, None] * 64 + d_arange_0[None, :] * 1)
        Q_1 = tl.load(sh_Q_1 + s_arange[:, None] * 64 + d_arange_1[None, :] * 1)
        dO_0 = tl.load(sh_dO_0 + s_arange[:, None] * 64 + d_arange_0[None, :] * 1)
        dO_1 = tl.load(sh_dO_1 + s_arange[:, None] * 64 + d_arange_1[None, :] * 1)
        O_0 = tl.load(sh_O_0 + s_arange[:, None] * 64 + d_arange_0[None, :] * 1)
        O_1 = tl.load(sh_O_1 + s_arange[:, None] * 64 + d_arange_1[None, :] * 1)
        L_i = tl.load(sh_L + s_arange)
        
        D_i = ((O_0 * dO_0).to(tl.float32)).sum(1, keep_dims=True) + ((O_1 * dO_1).to(tl.float32)).sum(1, keep_dims=True)
        
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
        P_reg = tl.where(i_s >= j_s, tl.exp(S_reg - L_i), 0.0)
        dS_reg = P_reg * (dP_reg - D_i) * scale
        
        dS_T = dS_reg.T
        P_reg_T = P_reg.T
        
        dK_0_acc = tl.dot(dS_T, Q_0, dK_0_acc)
        dK_1_acc = tl.dot(dS_T, Q_1, dK_1_acc)
        dV_0_acc = tl.dot(P_reg_T, dO_0, dV_0_acc)
        dV_1_acc = tl.dot(P_reg_T, dO_1, dV_1_acc)
            
    dK_ptr_0 = (dK_ptr + j * BLOCK * d + s_arange[:, None] * d + d_arange_0[None, :] * 1).to(tl.pointer_ty(tl.bfloat16))
    dK_ptr_1 = (dK_ptr + j * BLOCK * d + s_arange[:, None] * d + d_arange_1[None, :] * 1).to(tl.pointer_ty(tl.bfloat16))
    tl.store(dK_ptr_0, dK_0_acc.to(tl.bfloat16), mask=(row_mask_j).to(tl.bfloat16))
    tl.store(dK_ptr_1, dK_1_acc.to(tl.bfloat16), mask=(row_mask_j).to(tl.bfloat16))
    
    dV_ptr_0 = (dV_ptr + j * BLOCK * d + s_arange[:, None] * d + d_arange_0[None, :] * 1).to(tl.pointer_ty(tl.bfloat16))
    dV_ptr_1 = (dV_ptr + j * BLOCK * d + s_arange[:, None] * d + d_arange_1[None, :] * 1).to(tl.pointer_ty(tl.bfloat16))
    tl.store(dV_ptr_0, dV_0_acc.to(tl.bfloat16), mask=(row_mask_j).to(tl.bfloat16))
    tl.store(dV_ptr_1, dV_1_acc.to(tl.bfloat16), mask=(row_mask_j).to(tl.bfloat16))


@triton.jit
def _dQ_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dQ_ptr,
    S, scale, d,
    l_mult, head_stride, batch_head_stride,
    BLOCK: tl.constexpr,
):
    """
    Attention Backward Pass Kernel 2: Computes dQ.
    Iterates over Query blocks (i) and accumulates over valid Key/Value blocks (j <= i).
    """
    tid = tl.program_id(0)
    pid_y = tl.program_id(1)
    
    b = pid_y // 48
    h = pid_y % 48
    
    i = tid
    T_c = (S + BLOCK - 1) // BLOCK
    
    q_ptr = Q_ptr + b * batch_head_stride + h * head_stride
    k_ptr = K_ptr + b * batch_head_stride + h * head_stride
    v_ptr = V_ptr + b * batch_head_stride + h * head_stride
    o_ptr = O_ptr + b * batch_head_stride + h * head_stride
    do_ptr = dO_ptr + b * batch_head_stride + h * head_stride
    
    dQ_ptr = dQ_ptr + b * batch_head_stride + h * head_stride
    
    sh_Q_0 = extern.shared_array("Q_0", BLOCK * BLOCK // 2)
    sh_Q_1 = extern.shared_array("Q_1", BLOCK * BLOCK // 2)
    sh_K_0 = extern.shared_array("K_0", BLOCK * BLOCK // 2)
    sh_K_1 = extern.shared_array("K_1", BLOCK * BLOCK // 2)
    sh_V_0 = extern.shared_array("V_0", BLOCK * BLOCK // 2)
    sh_V_1 = extern.shared_array("V_1", BLOCK * BLOCK // 2)
    sh_dO_0 = extern.shared_array("dO_0", BLOCK * BLOCK // 2)
    sh_dO_1 = extern.shared_array("dO_1", BLOCK * BLOCK // 2)
    sh_O_0 = extern.shared_array("O_0", BLOCK * BLOCK // 2)
    sh_O_1 = extern.shared_array("O_1", BLOCK * BLOCK // 2)
    sh_L = extern.shared_array("L", BLOCK)
    
    s_arange = tl.arange(0, BLOCK)
    d_arange = tl.arange(0, 128)
    d_arange_0 = tl.arange(0, 64)
    d_arange_1 = tl.arange(64, 128)
    
    row_mask_i = ((i * BLOCK + s_arange) < S).to(tl.int1)[:, None]
    
    Q = tl.load(q_ptr + i * BLOCK * d + s_arange[:, None] * d + d_arange[None, :] * 1, mask=row_mask_i, other=0.0)
    Q_0, Q_1 = tl.split(Q, 2, dim=1)
    
    dO = tl.load(do_ptr + i * BLOCK * d + s_arange[:, None] * d + d_arange[None, :] * 1, mask=row_mask_i, other=0.0)
    dO_0, dO_1 = tl.split(dO, 2, dim=1)
    
    O = tl.load(o_ptr + i * BLOCK * d + s_arange[:, None] * d + d_arange[None, :] * 1, mask=row_mask_i, other=0.0)
    O_0, O_1 = tl.split(O, 2, dim=1)
    
    L_i = tl.load(L_ptr + b * l_mult + h * S + i * BLOCK + s_arange, mask=((i * BLOCK + s_arange) < S).to(tl.int1), other=0.0)
    
    tl.store(sh_Q_0 + s_arange[:, None] * 64 + d_arange_0[None, :] * 1, Q_0)
    tl.store(sh_Q_1 + s_arange[:, None] * 64 + d_arange_1[None, :] * 1, Q_1)
    tl.store(sh_dO_0 + s_arange[:, None] * 64 + d_arange_0[None, :] * 1, dO_0)
    tl.store(sh_dO_1 + s_arange[:, None] * 64 + d_arange_1[None, :] * 1, dO_1)
    tl.store(sh_O_0 + s_arange[:, None] * 64 + d_arange_0[None, :] * 1, O_0)
    tl.store(sh_O_1 + s_arange[:, None] * 64 + d_arange_1[None, :] * 1, O_1)
    tl.store(sh_L + s_arange, L_i)
    
    Q_0 = tl.load(sh_Q_0 + s_arange[:, None] * 64 + d_arange_0[None, :] * 1)
    Q_1 = tl.load(sh_Q_1 + s_arange[:, None] * 64 + d_arange_1[None, :] * 1)
    dO_0 = tl.load(sh_dO_0 + s_arange[:, None] * 64 + d_arange_0[None, :] * 1)
    dO_1 = tl.load(sh_dO_1 + s_arange[:, None] * 64 + d_arange_1[None, :] * 1)
    O_0 = tl.load(sh_O_0 + s_arange[:, None] * 64 + d_arange_0[None, :] * 1)
    O_1 = tl.load(sh_O_1 + s_arange[:, None] * 64 + d_arange_1[None, :] * 1)
    L_i = tl.load(sh_L + s_arange)
    
    D_i = ((O_0 * dO_0).to(tl.float32)).sum(1, keep_dims=True) + ((O_1 * dO_1).to(tl.float32)).sum(1, keep_dims=True)
    
    dQ_0_acc = tl.zeros((BLOCK, BLOCK // 2), dtype=tl.float32)
    dQ_1_acc = tl.zeros((BLOCK, BLOCK // 2), dtype=tl.float32)
    
    for j in range(0, i + 1):
        row_mask_j = ((j * BLOCK + s_arange) < S).to(tl.int1)[:, None]
        
        K = tl.load(k_ptr + j * BLOCK * d + s_arange[:, None] * d + d_arange[None, :] * 1, mask=row_mask_j, other=0.0)
        K_0, K_1 = tl.split(K, 2, dim=1)
        
        V = tl.load(v_ptr + j * BLOCK * d + s_arange[:, None] * d + d_arange[None, :] * 1, mask=row_mask_j, other=0.0)
        V_0, V_1 = tl.split(V, 2, dim=1)
        
        tl.store(sh_K_0 + s_arange[:, None] * 64 + d_arange_0[None, :] * 1, K_0)
        tl.store(sh_K_1 + s_arange[:, None] * 64 + d_arange_1[None, :] * 1, K_1)
        tl.store(sh_V_0 + s_arange[:, None] * 64 + d_arange_0[None, :] * 1, V_0)
        tl.store(sh_V_1 + s_arange[:, None] * 64 + d_arange_1[None, :] * 1, V_1)
        
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
        P_reg = tl.where(i_s >= j_s, tl.exp(S_reg - L_i), 0.0)
        dS_reg = P_reg * (dP_reg - D_i) * scale
        
        dQ_0_acc = tl.dot(dS_reg, K_0, dQ_0_acc)
        dQ_1_acc = tl.dot(dS_reg, K_1, dQ_1_acc)
            
    dQ_ptr_0 = (dQ_ptr + i * BLOCK * d + s_arange[:, None] * d + d_arange_0[None, :] * 1).to(tl.pointer_ty(tl.bfloat16))
    dQ_ptr_1 = (dQ_ptr + i * BLOCK * d + s_arange[:, None] * d + d_arange_1[None, :] * 1).to(tl.pointer_ty(tl.bfloat16))
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
    
    s_stride = Q.stride(2)
    d_stride = Q.stride(3)
    
    # Handle L being 3D (B, H, S) or 4D (B, H, S, 1)
    if L.dim() == 3:
        l_mult = L.shape[0] * L.shape[1]
    else:
        l_mult = L.shape[0] * L.shape[1]
        
    head_stride = s * d
    batch_head_stride = h * head_stride
    
    BLOCK = 128
    T_c = triton.cdiv(s, BLOCK)
    grid = (T_c, b * h)
    
    _dKdV_kernel[grid](
        Q, K, V, O, dO, L, dK, dV,
        s, scale, d, l_mult, head_stride, batch_head_stride,
        BLOCK=BLOCK, num_warps=8, num_stages=2
    )
    
    _dQ_kernel[grid](
        Q, K, V, O, dO, L, dQ,
        s, scale, d, l_mult, head_stride, batch_head_stride,
        BLOCK=BLOCK, num_warps=8, num_stages=2
    )