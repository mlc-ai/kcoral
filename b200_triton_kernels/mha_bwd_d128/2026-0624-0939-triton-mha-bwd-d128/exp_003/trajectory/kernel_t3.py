import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_bwd_dq_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L_ptr, dQ_desc,
    S_len, d, scale, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    b_h_idx = tl.program_id(0)
    m_idx = tl.program_id(1)
    
    b = b_h_idx // H
    h = b_h_idx % H
    
    offset_m = m_idx * BLOCK_M
    mask_m = (offset_m + tl.arange(0, BLOCK_M)) < S_len
    
    Q_m_0 = Q_desc.load([b, h, offset_m, 0])
    Q_m_1 = Q_desc.load([b, h, offset_m, 64])
    dO_m_0 = dO_desc.load([b, h, offset_m, 0])
    dO_m_1 = dO_desc.load([b, h, offset_m, 64])
    O_m_0 = O_desc.load([b, h, offset_m, 0])
    O_m_1 = O_desc.load([b, h, offset_m, 64])
    
    Q_m_0 = tl.reshape(Q_m_0.to(tl.float32), [BLOCK_M, 64])
    Q_m_1 = tl.reshape(Q_m_1.to(tl.float32), [BLOCK_M, 64])
    dO_m_0 = tl.reshape(dO_m_0.to(tl.float32), [BLOCK_M, 64])
    dO_m_1 = tl.reshape(dO_m_1.to(tl.float32), [BLOCK_M, 64])
    O_m_0 = tl.reshape(O_m_0.to(tl.float32), [BLOCK_M, 64])
    O_m_1 = tl.reshape(O_m_1.to(tl.float32), [BLOCK_M, 64])
    
    D_m = tl.sum(dO_m_0 * O_m_0, axis=1) + tl.sum(dO_m_1 * O_m_1, axis=1)
    D_m = D_m[:, None]
    
    L_base = L_ptr + b_h_idx * S_len + offset_m
    L_m = tl.load(L_base + tl.arange(0, BLOCK_M), mask=mask_m, other=0.0)
    L_m = L_m[:, None]
    
    acc_dQ_0 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    acc_dQ_1 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    
    for n_idx in tl.range(0, tl.cdiv(S_len, BLOCK_N), 1, flatten=True):
        offset_n = n_idx * BLOCK_N
        
        K_n_0 = tl.reshape(K_desc.load([b, h, offset_n, 0]).to(tl.float32), [BLOCK_N, 64])
        K_n_1 = tl.reshape(K_desc.load([b, h, offset_n, 64]).to(tl.float32), [BLOCK_N, 64])
        V_n_0 = tl.reshape(V_desc.load([b, h, offset_n, 0]).to(tl.float32), [BLOCK_N, 64])
        V_n_1 = tl.reshape(V_desc.load([b, h, offset_n, 64]).to(tl.float32), [BLOCK_N, 64])
        
        S_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        S_acc = tl.dot(Q_m_0, K_n_0.T, S_acc)
        S_acc = tl.dot(Q_m_1, K_n_1.T, S_acc)
        S = S_acc * scale
        
        P = tl.exp(S - L_m)
        
        dP_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        dP_acc = tl.dot(dO_m_0, V_n_0.T, dP_acc)
        dP_acc = tl.dot(dO_m_1, V_n_1.T, dP_acc)
        
        dS = P * (dP_acc - D_m) * scale
        
        acc_dQ_0 = tl.dot(dS, K_n_0, acc_dQ_0)
        acc_dQ_1 = tl.dot(dS, K_n_1, acc_dQ_1)
    
    dq_out_0 = acc_dQ_0.to(tl.bfloat16)
    dq_out_1 = acc_dQ_1.to(tl.bfloat16)
    
    dQ_desc.store([b, h, offset_m, 0], dq_out_0)
    dQ_desc.store([b, h, offset_m, 64], dq_out_1)


@triton.jit
def _mha_bwd_dkv_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L_ptr, dK_desc, dV_desc,
    S_len, d, scale, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    b_h_idx = tl.program_id(0)
    n_idx = tl.program_id(1)
    
    b = b_h_idx // H
    h = b_h_idx % H
    
    offset_n = n_idx * BLOCK_N
    
    K_n_0 = tl.reshape(K_desc.load([b, h, offset_n, 0]).to(tl.float32), [BLOCK_N, 64])
    K_n_1 = tl.reshape(K_desc.load([b, h, offset_n, 64]).to(tl.float32), [BLOCK_N, 64])
    V_n_0 = tl.reshape(V_desc.load([b, h, offset_n, 0]).to(tl.float32), [BLOCK_N, 64])
    V_n_1 = tl.reshape(V_desc.load([b, h, offset_n, 64]).to(tl.float32), [BLOCK_N, 64])
    
    acc_dK_0 = tl.zeros((BLOCK_N, 64), dtype=tl.float32)
    acc_dK_1 = tl.zeros((BLOCK_N, 64), dtype=tl.float32)
    acc_dV_0 = tl.zeros((BLOCK_N, 64), dtype=tl.float32)
    acc_dV_1 = tl.zeros((BLOCK_N, 64), dtype=tl.float32)
    
    for m_idx in range(0, S_len, BLOCK_M):
        offset_m = m_idx * BLOCK_M
        mask_m = (offset_m + tl.arange(0, BLOCK_M)) < S_len
        
        Q_m_0 = tl.reshape(Q_desc.load([b, h, offset_m, 0]).to(tl.float32), [BLOCK_M, 64])
        Q_m_1 = tl.reshape(Q_desc.load([b, h, offset_m, 64]).to(tl.float32), [BLOCK_M, 64])
        dO_m_0 = tl.reshape(dO_desc.load([b, h, offset_m, 0]).to(tl.float32), [BLOCK_M, 64])
        dO_m_1 = tl.reshape(dO_desc.load([b, h, offset_m, 64]).to(tl.float32), [BLOCK_M, 64])
        O_m_0 = tl.reshape(O_desc.load([b, h, offset_m, 0]).to(tl.float32), [BLOCK_M, 64])
        O_m_1 = tl.reshape(O_desc.load([b, h, offset_m, 64]).to(tl.float32), [BLOCK_M, 64])
        
        D_m = tl.sum(dO_m_0 * O_m_0, axis=1) + tl.sum(dO_m_1 * O_m_1, axis=1)
        D_m = D_m[:, None]
        
        L_base = L_ptr + b_h_idx * S_len + offset_m
        L_m = tl.load(L_base + tl.arange(0, BLOCK_M), mask=mask_m, other=0.0)
        L_m = L_m[:, None]
        
        S_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        S_acc = tl.dot(Q_m_0, K_n_0.T, S_acc)
        S_acc = tl.dot(Q_m_1, K_n_1.T, S_acc)
        S = S_acc * scale
        
        P = tl.exp(S - L_m)
        
        dP_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        dP_acc = tl.dot(dO_m_0, V_n_0.T, dP_acc)
        dP_acc = tl.dot(dO_m_1, V_n_1.T, dP_acc)
        
        dS = P * (dP_acc - D_m) * scale
        
        acc_dV_0 = tl.dot(P.T, dO_m_0, acc_dV_0)
        acc_dV_1 = tl.dot(P.T, dO_m_1, acc_dV_1)
        
        acc_dK_0 = tl.dot(dS.T, Q_m_0, acc_dK_0)
        acc_dK_1 = tl.dot(dS.T, Q_m_1, acc_dK_1)
    
    dk_out_0 = acc_dK_0.to(tl.bfloat16)
    dk_out_1 = acc_dK_1.to(tl.bfloat16)
    dv_out_0 = acc_dV_0.to(tl.bfloat16)
    dv_out_1 = acc_dV_1.to(tl.bfloat16)
    
    dK_desc.store([b, h, offset_n, 0], dk_out_0)
    dK_desc.store([b, h, offset_n, 64], dk_out_1)
    dV_desc.store([b, h, offset_n, 0], dv_out_0)
    dV_desc.store([b, h, offset_n, 64], dv_out_1)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    
    assert Q.is_contiguous()
    assert K.is_contiguous()
    assert V.is_contiguous()
    assert O.is_contiguous()
    assert dO.is_contiguous()
    assert dQ.is_contiguous()
    assert dK.is_contiguous()
    assert dV.is_contiguous()
    
    for t in [Q, K, V, O, dO, dQ, dK, dV]:
        for s in t.stride():
            assert s == 0 or (s * t.element_size()) % 16 == 0
    
    BLOCK_M = 64
    Q_desc = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M, 64])
    K_desc = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_M, 64])
    V_desc = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_M, 64])
    O_desc = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M, 64])
    dO_desc = TensorDescriptor.from_tensor(dO, [1, 1, BLOCK_M, 64])
    dQ_desc = TensorDescriptor.from_tensor(dQ, [1, 1, BLOCK_M, 64])
    dK_desc = TensorDescriptor.from_tensor(dK, [1, 1, BLOCK_M, 64])
    dV_desc = TensorDescriptor.from_tensor(dV, [1, 1, BLOCK_M, 64])
    
    grid_dq = (B * H, triton.cdiv(S, BLOCK_M))
    _mha_bwd_dq_kernel[grid_dq](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dQ_desc,
        S, d, scale, H,
        BLOCK_M=BLOCK_M, BLOCK_N=64,
        num_warps=8, num_stages=3,
    )
    
    grid_dkv = (B * H, triton.cdiv(S, 64))
    _mha_bwd_dkv_kernel[grid_dkv](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dK_desc, dV_desc,
        S, d, scale, H,
        BLOCK_M=64, BLOCK_N=64,
        num_warps=8, num_stages=3,
    )