import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _dQ_kernel(
    q_desc, k_desc, v_desc, do_desc, o_desc, l_ptr, dq_desc,
    B, H, S, tau,
    BLOCK_S: tl.constexpr,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    i = tl.program_id(0)
    offset_i = i * BLOCK_S
    
    Q = q_desc.load([b, h, offset_i, 0])
    dO = do_desc.load([b, h, offset_i, 0])
    O = o_desc.load([b, h, offset_i, 0])
    
    seq_idx_q = offset_i + tl.arange(0, 128)
    mask_seq_q = (seq_idx_q < S)
    
    Q_zero = tl.zeros((128, 128), tl.bfloat16)
    dO_masked = tl.where(mask_seq_q[:, None], dO, Q_zero)
    O_masked = tl.where(mask_seq_q[:, None], O, Q_zero)
    D = tl.sum(dO_masked * O_masked, axis=1)
    
    l_offset_i = b * H * S + h * S + offset_i
    L = tl.load(l_ptr + l_offset_i + tl.arange(0, 128), mask=(seq_idx_q < S), other=0.0)
    
    acc_dQ = tl.zeros((128, 128), tl.float32)
    num_k_tiles = tl.cdiv(S, 128)
    
    seq_idx_kv = tl.arange(0, 128)
    
    K_zero = tl.zeros((128, 128), tl.bfloat16)
    V_zero = tl.zeros((128, 128), tl.bfloat16)
    
    J = 0
    K = k_desc.load([b, h, J * 128, 0])
    V = v_desc.load([b, h, J * 128, 0])
    mask_seq_kv = (J * 128 + seq_idx_kv < S)
    
    for j in tl.range(num_k_tiles, num_stages=3):
        if j + 1 < num_k_tiles:
            J_next = j + 1
            K_next = k_desc.load([b, h, J_next * 128, 0])
            V_next = v_desc.load([b, h, J_next * 128, 0])
            mask_seq_kv_next = (J_next * 128 + seq_idx_kv < S)
            
            K = K_next
            V = V_next
            mask_seq_kv = mask_seq_kv_next
        
        K_masked = tl.where(mask_seq_kv[None, :], K, K_zero)
        V_masked = tl.where(mask_seq_kv[None, :], V, V_zero)
        
        S_mat = tl.dot(Q, K_masked.T)
        dP = tl.dot(dO, V_masked.T)
        
        mask_QK = (seq_idx_q[:, None] < S) & (mask_seq_kv[None, :])
        S_mat = tl.where(mask_QK, S_mat, 0.0)
        dP = tl.where(mask_QK, dP, 0.0)
        
        P = tl.exp(S_mat * tau - L[:, None])
        dS = P * (dP - D[:, None]) * tau
        
        dS_bf16 = dS.to(tl.bfloat16)
        acc_dQ = tl.dot(dS_bf16, K_masked, acc_dQ)
    
    mask_dQ = (seq_idx_q < S)[:, None]
    dq_desc.store([b, h, offset_i, 0], acc_dQ.to(tl.bfloat16), mask=mask_dQ)


@triton.jit
def _dK_dV_kernel(
    q_desc, k_desc, v_desc, do_desc, o_desc, l_ptr, dk_desc, dv_desc,
    B, H, S, tau,
    BLOCK_S: tl.constexpr,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    j = tl.program_id(0)
    offset_j = j * BLOCK_S
    
    K = k_desc.load([b, h, offset_j, 0])
    V = v_desc.load([b, h, offset_j, 0])
    
    seq_idx_kv = offset_j + tl.arange(0, 128)
    mask_seq_kv = (seq_idx_kv < S)
    
    K_zero = tl.zeros((128, 128), tl.bfloat16)
    V_zero = tl.zeros((128, 128), tl.bfloat16)
    K_masked = tl.where(mask_seq_kv[None, :], K, K_zero)
    V_masked = tl.where(mask_seq_kv[None, :], V, V_zero)
    
    acc_dK = tl.zeros((128, 128), tl.float32)
    acc_dV = tl.zeros((128, 128), tl.float32)
    
    num_q_tiles = tl.cdiv(S, 128)
    
    seq_idx_q = tl.arange(0, 128)
    
    Q_zero = tl.zeros((128, 128), tl.bfloat16)
    
    I = 0
    Q = q_desc.load([b, h, I * 128, 0])
    dO = do_desc.load([b, h, I * 128, 0])
    O = o_desc.load([b, h, I * 128, 0])
    mask_seq_q = (I * 128 + seq_idx_q < S)
    
    l_offset_i = b * H * S + h * S + I * 128
    L = tl.load(l_ptr + l_offset_i + tl.arange(0, 128), mask=(I * 128 + seq_idx_q < S), other=0.0)
    
    for i in tl.range(num_q_tiles, num_stages=3):
        if i + 1 < num_q_tiles:
            I_next = i + 1
            Q_next = q_desc.load([b, h, I_next * 128, 0])
            dO_next = do_desc.load([b, h, I_next * 128, 0])
            O_next = o_desc.load([b, h, I_next * 128, 0])
            mask_seq_q_next = (I_next * 128 + seq_idx_q < S)
            
            Q = Q_next
            dO = dO_next
            O = O_next
            mask_seq_q = mask_seq_q_next
            
            l_offset_i = b * H * S + h * S + I_next * 128
            L = tl.load(l_ptr + l_offset_i + tl.arange(0, 128), mask=(I_next * 128 + seq_idx_q < S), other=0.0)
        
        dO_masked = tl.where(mask_seq_q[:, None], dO, Q_zero)
        O_masked = tl.where(mask_seq_q[:, None], O, Q_zero)
        D = tl.sum(dO_masked * O_masked, axis=1)
        
        S_mat = tl.dot(Q, K_masked.T)
        dP = tl.dot(dO, V_masked.T)
        
        mask_QK = (mask_seq_q[:, None] < True) & (mask_seq_kv[None, :] < True)
        actual_mask_QK = ((I * 128 + seq_idx_q)[:, None] < S) & (mask_seq_kv[None, :])
        S_mat = tl.where(actual_mask_QK, S_mat, 0.0)
        dP = tl.where(actual_mask_QK, dP, 0.0)
        
        P = tl.exp(S_mat * tau - L[:, None])
        dS = P * (dP - D[:, None]) * tau
        
        P_bf16 = P.to(tl.bfloat16)
        dS_bf16 = dS.to(tl.bfloat16)
        
        acc_dV = tl.dot(P_bf16.T, dO_masked, acc_dV)
        acc_dK = tl.dot(dS_bf16.T, Q_masked, acc_dK)
        
        Q_masked = tl.where(mask_seq_q[:, None], Q, Q_zero)
    
    mask_dK = (seq_idx_kv < S)[:, None]
    mask_dV = (seq_idx_kv < S)[:, None]
    dk_desc.store([b, h, offset_j, 0], acc_dK.to(tl.bfloat16), mask=mask_dK)
    dv_desc.store([b, h, offset_j, 0], acc_dV.to(tl.bfloat16), mask=mask_dV)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    tau = 1.0 / (d ** 0.5)
    BLOCK_S = 128
    
    q_desc = TensorDescriptor.from_tensor(Q, [128, 128])
    k_desc = TensorDescriptor.from_tensor(K, [128, 128])
    v_desc = TensorDescriptor.from_tensor(V, [128, 128])
    o_desc = TensorDescriptor.from_tensor(O, [128, 128])
    do_desc = TensorDescriptor.from_tensor(dO, [128, 128])
    
    dq_desc = TensorDescriptor.from_tensor(dQ, [128, 128])
    dk_desc = TensorDescriptor.from_tensor(dK, [128, 128])
    dv_desc = TensorDescriptor.from_tensor(dV, [128, 128])
    
    grid = (triton.cdiv(S, BLOCK_S), H, B)
    
    _dQ_kernel[grid](
        q_desc, k_desc, v_desc, do_desc, o_desc, L, dq_desc,
        B, H, S, tau,
        BLOCK_S=BLOCK_S,
        num_warps=8,
        num_stages=3,
        maxnreg=255,
    )
    
    _dK_dV_kernel[grid](
        q_desc, k_desc, v_desc, do_desc, o_desc, L, dk_desc, dv_desc,
        B, H, S, tau,
        BLOCK_S=BLOCK_S,
        num_warps=8,
        num_stages=3,
        maxnreg=255,
    )