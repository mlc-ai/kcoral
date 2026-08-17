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
    
    Q = q_desc.load([b, h, i * 128, 0])
    dO = do_desc.load([b, h, i * 128, 0])
    O = o_desc.load([b, h, i * 128, 0])
    
    seq_idx_q = tl.arange(0, 128)
    
    D = tl.sum(dO * O, axis=1)
    
    l_offset_i = (b * H + h) * S + seq_idx_q
    L = tl.load(l_ptr + l_offset_i, mask=(seq_idx_q < S), other=0.0)
    
    acc_dQ = tl.zeros((128, 128), tl.float32)
    num_k_tiles = tl.cdiv(S, 128)
    
    J = 0
    K = k_desc.load([b, h, J * 128, 0])
    V = v_desc.load([b, h, J * 128, 0])
    
    seq_idx_kv = tl.arange(0, 128)
    
    for j in tl.range(num_k_tiles, num_stages=3):
        if j + 1 < num_k_tiles:
            K = k_desc.load([b, h, (j + 1) * 128, 0])
            V = v_desc.load([b, h, (j + 1) * 128, 0])
        
        S_mat = tl.dot(Q, K.T)
        dP = tl.dot(dO, V.T)
        
        P = tl.exp(S_mat * tau - L[:, None])
        dS = P * (dP - D[:, None]) * tau
        
        mask_QK = ((i * 128 + seq_idx_q)[:, None] < S) & ((J * 128 + seq_idx_kv)[None, :] < S)
        dS = tl.where(mask_QK, dS, 0.0)
        
        dS_bf16 = dS.to(tl.bfloat16)
        acc_dQ = tl.dot(dS_bf16, K, acc_dQ)
        
        J = j + 1
    
    dq_desc.store([b, h, i * 128, 0], acc_dQ.to(tl.bfloat16))


@triton.jit
def _dK_dV_kernel(
    q_desc, k_desc, v_desc, do_desc, o_desc, l_ptr, dk_desc, dv_desc,
    B, H, S, tau,
    BLOCK_S: tl.constexpr,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    j = tl.program_id(0)
    
    K = k_desc.load([b, h, j * 128, 0])
    V = v_desc.load([b, h, j * 128, 0])
    
    acc_dK = tl.zeros((128, 128), tl.float32)
    acc_dV = tl.zeros((128, 128), tl.float32)
    
    num_q_tiles = tl.cdiv(S, 128)
    
    seq_idx_q = tl.arange(0, 128)
    seq_idx_kv = tl.arange(0, 128)
    
    I = 0
    Q = q_desc.load([b, h, I * 128, 0])
    dO = do_desc.load([b, h, I * 128, 0])
    O = o_desc.load([b, h, I * 128, 0])
    
    l_offset_i = (b * H + h) * S + seq_idx_q
    L = tl.load(l_ptr + l_offset_i, mask=(seq_idx_q < S), other=0.0)
    
    for i in tl.range(num_q_tiles, num_stages=3):
        if i + 1 < num_q_tiles:
            Q = q_desc.load([b, h, (i + 1) * 128, 0])
            dO = do_desc.load([b, h, (i + 1) * 128, 0])
            O = o_desc.load([b, h, (i + 1) * 128, 0])
            l_offset_i = (b * H + h) * S + (i + 1) * 128 + seq_idx_q
            L = tl.load(l_ptr + l_offset_i, mask=((i + 1) * 128 + seq_idx_q < S), other=0.0)
        
        D = tl.sum(dO * O, axis=1)
        
        S_mat = tl.dot(Q, K.T)
        dP = tl.dot(dO, V.T)
        
        P = tl.exp(S_mat * tau - L[:, None])
        dS = P * (dP - D[:, None]) * tau
        
        mask_QK = ((I * 128 + seq_idx_q)[:, None] < S) & ((j * 128 + seq_idx_kv)[None, :] < S)
        P = tl.where(mask_QK, P, 0.0)
        dS = tl.where(mask_QK, dS, 0.0)
        
        P_bf16 = P.to(tl.bfloat16)
        dS_bf16 = dS.to(tl.bfloat16)
        
        acc_dV = tl.dot(P_bf16.T, dO, acc_dV)
        acc_dK = tl.dot(dS_bf16.T, Q, acc_dK)
        
        I = i + 1
    
    dk_desc.store([b, h, j * 128, 0], acc_dK.to(tl.bfloat16))
    dv_desc.store([b, h, j * 128, 0], acc_dV.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    tau = 1.0 / (d ** 0.5)
    BLOCK_S = 128
    
    q_desc = TensorDescriptor.from_tensor(Q, [1, 1, 128, 128])
    k_desc = TensorDescriptor.from_tensor(K, [1, 1, 128, 128])
    v_desc = TensorDescriptor.from_tensor(V, [1, 1, 128, 128])
    o_desc = TensorDescriptor.from_tensor(O, [1, 1, 128, 128])
    do_desc = TensorDescriptor.from_tensor(dO, [1, 1, 128, 128])
    
    dq_desc = TensorDescriptor.from_tensor(dQ, [1, 1, 128, 128])
    dk_desc = TensorDescriptor.from_tensor(dK, [1, 1, 128, 128])
    dv_desc = TensorDescriptor.from_tensor(dV, [1, 1, 128, 128])
    
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