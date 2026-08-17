import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def offset_i(idx):
    return idx * 128


@triton.jit
def row_offset(b, h, s):
    return (b * H + h) * S + s


@triton.jit
def _dQ_kernel(
    q_desc, k_desc, v_desc, do_desc, o_desc, l_ptr, dq_desc,
    B, H, S, tau,
    BLOCK_S: tl.constexpr,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    i = tl.program_id(0)
    
    Q = q_desc.load([row_offset(b, h, offset_i(i)), 0])
    dO = do_desc.load([row_offset(b, h, offset_i(i)), 0])
    O = o_desc.load([row_offset(b, h, offset_i(i)), 0])
    
    seq_idx_q = tl.arange(0, 128)
    
    D = tl.sum(dO * O, axis=1)
    
    l_offset_i = (b * H + h) * S + offset_i(i)
    L = tl.load(l_ptr + l_offset_i + tl.arange(0, 128), mask=(offset_i(i) + seq_idx_q < S), other=0.0)
    
    acc_dQ = tl.zeros((128, 128), tl.float32)
    num_k_tiles = tl.cdiv(S, 128)
    
    seq_idx_kv = tl.arange(0, 128)
    
    J = 0
    K = k_desc.load([row_offset(J), 0])
    V = v_desc.load([row_offset(J), 0])
    
    for j in tl.range(num_k_tiles, num_stages=3):
        if j + 1 < num_k_tiles:
            J_next = j + 1
            K = k_desc.load([row_offset(J_next), 0])
            V = v_desc.load([row_offset(J_next), 0])
        
        S_mat = tl.dot(Q, K.T)
        dP = tl.dot(dO, V.T)
        
        P = tl.exp(S_mat * tau - L[:, None])
        dS = P * (dP - D[:, None]) * tau
        
        mask_QK = ((offset_i(i) + seq_idx_q)[:, None] < S) & ((offset_i(J) + seq_idx_kv)[None, :] < S)
        dS = tl.where(mask_QK, dS, 0.0)
        
        dS_bf16 = dS.to(tl.bfloat16)
        acc_dQ = tl.dot(dS_bf16, K, acc_dQ)
        
        J = j + 1
    
    dq_desc.store([row_offset(b, h, offset_i(i)), 0], acc_dQ.to(tl.bfloat16))


@triton.jit
def _dK_dV_kernel(
    q_desc, k_desc, v_desc, do_desc, o_desc, l_ptr, dk_desc, dv_desc,
    B, H, S, tau,
    BLOCK_S: tl.constexpr,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    j = tl.program_id(0)
    
    K = k_desc.load([row_offset(b, h, offset_i(j)), 0])
    V = v_desc.load([row_offset(b, h, offset_i(j)), 0])
    
    acc_dK = tl.zeros((128, 128), tl.float32)
    acc_dV = tl.zeros((128, 128), tl.float32)
    
    num_q_tiles = tl.cdiv(S, 128)
    
    seq_idx_q = tl.arange(0, 128)
    seq_idx_kv = tl.arange(0, 128)
    
    I = 0
    Q = q_desc.load([row_offset(I), 0])
    dO = do_desc.load([row_offset(I), 0])
    O = o_desc.load([row_offset(I), 0])
    
    l_offset_i = (b * H + h) * S + offset_i(I)
    L = tl.load(l_ptr + l_offset_i + tl.arange(0, 128), mask=(offset_i(I) + seq_idx_q < S), other=0.0)
    
    for i in tl.range(num_q_tiles, num_stages=3):
        if i + 1 < num_q_tiles:
            I_next = i + 1
            Q = q_desc.load([row_offset(I_next), 0])
            dO = do_desc.load([row_offset(I_next), 0])
            O = o_desc.load([row_offset(I_next), 0])
            
            l_offset_i = (b * H + h) * S + offset_i(I_next)
            L = tl.load(l_ptr + l_offset_i + tl.arange(0, 128), mask=(offset_i(I_next) + seq_idx_q < S), other=0.0)
        
        D = tl.sum(dO * O, axis=1)
        
        S_mat = tl.dot(Q, K.T)
        dP = tl.dot(dO, V.T)
        
        P = tl.exp(S_mat * tau - L[:, None])
        dS = P * (dP - D[:, None]) * tau
        
        mask_QK = ((offset_i(I) + seq_idx_q)[:, None] < S) & ((offset_i(j) + seq_idx_kv)[None, :] < S)
        P = tl.where(mask_QK, P, 0.0)
        dS = tl.where(mask_QK, dS, 0.0)
        
        P_bf16 = P.to(tl.bfloat16)
        dS_bf16 = dS.to(tl.bfloat16)
        
        acc_dV = tl.dot(P_bf16.T, dO, acc_dV)
        acc_dK = tl.dot(dS_bf16.T, Q, acc_dK)
        
        I = i + 1
    
    dk_desc.store([row_offset(b, h, offset_i(j)), 0], acc_dK.to(tl.bfloat16))
    dv_desc.store([row_offset(b, h, offset_i(j)), 0], acc_dV.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    tau = 1.0 / (d ** 0.5)
    BLOCK_S = 128
    
    q_2d = Q.reshape(-1, 128)
    k_2d = K.reshape(-1, 128)
    v_2d = V.reshape(-1, 128)
    o_2d = O.reshape(-1, 128)
    do_2d = dO.reshape(-1, 128)
    
    dq_2d = dQ.reshape(-1, 128)
    dk_2d = dK.reshape(-1, 128)
    dv_2d = dV.reshape(-1, 128)
    
    q_desc = TensorDescriptor.from_tensor(q_2d, [128, 128])
    k_desc = TensorDescriptor.from_tensor(k_2d, [128, 128])
    v_desc = TensorDescriptor.from_tensor(v_2d, [128, 128])
    o_desc = TensorDescriptor.from_tensor(o_2d, [128, 128])
    do_desc = TensorDescriptor.from_tensor(do_2d, [128, 128])
    
    dq_desc = TensorDescriptor.from_tensor(dq_2d, [128, 128])
    dk_desc = TensorDescriptor.from_tensor(dk_2d, [128, 128])
    dv_desc = TensorDescriptor.from_tensor(dv_2d, [128, 128])
    
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