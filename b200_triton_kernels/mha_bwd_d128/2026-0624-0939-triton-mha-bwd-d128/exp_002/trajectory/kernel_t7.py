import torch
import triton
import triton.language as tl
import math
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_bwd_dK_dV(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L_desc, dK_desc, dV_desc,
    S, inv_sqrt_d: tl.constexpr, BLOCK: tl.constexpr
):
    bid_h = tl.program_id(0)
    bid_b = tl.program_id(1)
    
    for j in range(0, S, BLOCK):
        K_0 = K_desc.load([bid_b, bid_h, j, 0])
        K_1 = K_desc.load([bid_b, bid_h, j, 64])
        V_0 = V_desc.load([bid_b, bid_h, j, 0])
        V_1 = V_desc.load([bid_b, bid_h, j, 64])
        
        dK_0 = tl.zeros((1, 1, BLOCK, 64), tl.float32)
        dK_1 = tl.zeros((1, 1, BLOCK, 64), tl.float32)
        dV_0 = tl.zeros((1, 1, BLOCK, 64), tl.float32)
        dV_1 = tl.zeros((1, 1, BLOCK, 64), tl.float32)
        
        for i in range(0, S, BLOCK):
            Q_0 = Q_desc.load([bid_b, bid_h, i, 0])
            Q_1 = Q_desc.load([bid_b, bid_h, i, 64])
            dO_0 = dO_desc.load([bid_b, bid_h, i, 0])
            dO_1 = dO_desc.load([bid_b, bid_h, i, 64])
            O_0 = O_desc.load([bid_b, bid_h, i, 0])
            O_1 = O_desc.load([bid_b, bid_h, i, 64])
            
            L_i = L_desc.load([bid_b, bid_h, i])
            L_i = tl.unsqueeze(L_i, dim=-1)
            
            D_i = tl.sum(dO_0.to(tl.float32) * O_0.to(tl.float32) + dO_1.to(tl.float32) * O_1.to(tl.float32), axis=-1, keep_dims=True)
            
            S_mat = tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)
            P = tl.exp(S_mat * inv_sqrt_d - L_i)
            
            dP = tl.dot(dO_0, V_0.T) + tl.dot(dO_1, V_1.T)
            dS = P * (dP - D_i)
            
            dS_bf16 = dS.to(tl.bfloat16)
            P_bf16 = P.to(tl.bfloat16)
            
            dK_0 = tl.dot(dS_bf16.T, Q_0, dK_0)
            dK_1 = tl.dot(dS_bf16.T, Q_1, dK_1)
            dV_0 = tl.dot(P_bf16.T, dO_0, dV_0)
            dV_1 = tl.dot(P_bf16.T, dO_1, dV_1)
            
        dK_0_scaled = (dK_0 * inv_sqrt_d).to(tl.bfloat16)
        dK_1_scaled = (dK_1 * inv_sqrt_d).to(tl.bfloat16)
        dV_0_scaled = dV_0.to(tl.bfloat16)
        dV_1_scaled = dV_1.to(tl.bfloat16)
        
        dK_desc.store([bid_b, bid_h, j, 0], dK_0_scaled)
        dK_desc.store([bid_b, bid_h, j, 64], dK_1_scaled)
        dV_desc.store([bid_b, bid_h, j, 0], dV_0_scaled)
        dV_desc.store([bid_b, bid_h, j, 64], dV_1_scaled)


@triton.jit
def _mha_bwd_dQ(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L_desc, dQ_desc,
    S, inv_sqrt_d: tl.constexpr, BLOCK: tl.constexpr
):
    bid_h = tl.program_id(0)
    bid_b = tl.program_id(1)
    
    for i in range(0, S, BLOCK):
        Q_0 = Q_desc.load([bid_b, bid_h, i, 0])
        Q_1 = Q_desc.load([bid_b, bid_h, i, 64])
        dO_0 = dO_desc.load([bid_b, bid_h, i, 0])
        dO_1 = dO_desc.load([bid_b, bid_h, i, 64])
        O_0 = O_desc.load([bid_b, bid_h, i, 0])
        O_1 = O_desc.load([bid_b, bid_h, i, 64])
        
        L_i = L_desc.load([bid_b, bid_h, i])
        L_i = tl.unsqueeze(L_i, dim=-1)
        
        D_i = tl.sum(dO_0.to(tl.float32) * O_0.to(tl.float32) + dO_1.to(tl.float32) * O_1.to(tl.float32), axis=-1, keep_dims=True)
        
        dQ_0 = tl.zeros((1, 1, BLOCK, 64), tl.float32)
        dQ_1 = tl.zeros((1, 1, BLOCK, 64), tl.float32)
        
        for j in range(0, S, BLOCK):
            K_0 = K_desc.load([bid_b, bid_h, j, 0])
            K_1 = K_desc.load([bid_b, bid_h, j, 64])
            V_0 = V_desc.load([bid_b, bid_h, j, 0])
            V_1 = V_desc.load([bid_b, bid_h, j, 64])
            
            S_mat = tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)
            P = tl.exp(S_mat * inv_sqrt_d - L_i)
            
            dP = tl.dot(dO_0, V_0.T) + tl.dot(dO_1, V_1.T)
            dS = P * (dP - D_i)
            
            dS_bf16 = dS.to(tl.bfloat16)
            
            dQ_0 = tl.dot(dS_bf16, K_0, dQ_0)
            dQ_1 = tl.dot(dS_bf16, K_1, dQ_1)
            
        dQ_0_scaled = (dQ_0 * inv_sqrt_d).to(tl.bfloat16)
        dQ_1_scaled = (dQ_1 * inv_sqrt_d).to(tl.bfloat16)
        
        dQ_desc.store([bid_b, bid_h, i, 0], dQ_0_scaled)
        dQ_desc.store([bid_b, bid_h, i, 64], dQ_1_scaled)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    inv_sqrt_d = 1.0 / math.sqrt(d)
    BLOCK = 128
    
    Q_desc = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK, 128])
    K_desc = TensorDescriptor.from_tensor(K, [1, 1, BLOCK, 128])
    V_desc = TensorDescriptor.from_tensor(V, [1, 1, BLOCK, 128])
    O_desc = TensorDescriptor.from_tensor(O, [1, 1, BLOCK, 128])
    dO_desc = TensorDescriptor.from_tensor(dO, [1, 1, BLOCK, 128])
    L_desc = TensorDescriptor.from_tensor(L, [1, 1, BLOCK])
    dQ_desc = TensorDescriptor.from_tensor(dQ, [1, 1, BLOCK, 128])
    dK_desc = TensorDescriptor.from_tensor(dK, [1, 1, BLOCK, 128])
    dV_desc = TensorDescriptor.from_tensor(dV, [1, 1, BLOCK, 128])
    
    grid = (H, B)
    
    _mha_bwd_dK_dV[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L_desc, dK_desc, dV_desc,
        S, inv_sqrt_d=inv_sqrt_d, BLOCK=BLOCK, num_warps=8, num_stages=2, num_ctas=1
    )
    
    _mha_bwd_dQ[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L_desc, dQ_desc,
        S, inv_sqrt_d=inv_sqrt_d, BLOCK=BLOCK, num_warps=8, num_stages=2, num_ctas=1
    )