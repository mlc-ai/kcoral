import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def rowsum(x):
    return tl.sum(x, axis=1)


@triton.jit
def _mha_bwd_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, dQ_desc, dK_desc, dV_desc, L,
    S_len, tau, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    num_blocks_q: tl.constexpr,
):
    idx = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    bh = b * H + h

    if idx < num_blocks_q:
        i = idx
        q_base = bh * S_len + i * BLOCK_M
        
        Q_i = Q_desc.load([q_base, 0])
        dO_i = dO_desc.load([q_base, 0])
        O_i = O_desc.load([q_base, 0])
        
        D = rowsum(dO_i * O_i)
        
        q_offsets_local = tl.arange(0, BLOCK_M)
        mask_L = (i * BLOCK_M + q_offsets_local) < S_len
        L_i = tl.load(L + q_base + q_offsets_local, mask=mask_L, other=0.0)
        
        dQ_i = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
        
        for j in range(tl.cdiv(S_len, BLOCK_N)):
            k_base = bh * S_len + j * BLOCK_N
            
            K_j = K_desc.load([k_base, 0])
            V_j = V_desc.load([k_base, 0])
            
            S = tl.dot(Q_i, K_j.T)
            
            P = tl.exp(S * tau - L_i[:, None])
            
            dP = tl.dot(dO_i, V_j.T)
            
            dS = P * (dP - D[:, None]) * tau
            
            dQ_i = tl.dot(dS.to(tl.bfloat16), K_j, dQ_i)
        
        dQ_desc.store([q_base, 0], dQ_i.to(tl.bfloat16))
        
    else:
        j = idx - num_blocks_q
        k_base = bh * S_len + j * BLOCK_N
        
        K_j = K_desc.load([k_base, 0])
        V_j = V_desc.load([k_base, 0])
        
        dK_j = tl.zeros((BLOCK_N, 128), dtype=tl.float32)
        dV_j = tl.zeros((BLOCK_N, 128), dtype=tl.float32)
        
        for i in range(tl.cdiv(S_len, BLOCK_M)):
            q_base = bh * S_len + i * BLOCK_M
            
            Q_i = Q_desc.load([q_base, 0])
            dO_i = dO_desc.load([q_base, 0])
            O_i = O_desc.load([q_base, 0])
            
            D = rowsum(dO_i * O_i)
            
            q_offsets_local = tl.arange(0, BLOCK_M)
            mask_L = (i * BLOCK_M + q_offsets_local) < S_len
            L_i = tl.load(L + q_base + q_offsets_local, mask=mask_L, other=0.0)
            
            S = tl.dot(Q_i, K_j.T)
            
            P = tl.exp(S * tau - L_i[:, None])
            
            dP = tl.dot(dO_i, V_j.T)
            
            dS = P * (dP - D[:, None]) * tau
            
            dV_j = tl.dot(P.to(tl.bfloat16).T, dO_i, dV_j)
            dK_j = tl.dot(dS.to(tl.bfloat16).T, Q_i, dK_j)
        
        dK_desc.store([k_base, 0], dK_j.to(tl.bfloat16))
        dV_desc.store([k_base, 0], dV_j.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S_len = Q.shape[0], Q.shape[1], Q.shape[2]
    d = Q.shape[3]
    tau = 1.0 / (d ** 0.5)
    
    Q_desc = TensorDescriptor.from_tensor(Q.reshape(B * H * S_len, 128), [128, 128])
    K_desc = TensorDescriptor.from_tensor(K.reshape(B * H * S_len, 128), [128, 128])
    V_desc = TensorDescriptor.from_tensor(V.reshape(B * H * S_len, 128), [128, 128])
    O_desc = TensorDescriptor.from_tensor(O.reshape(B * H * S_len, 128), [128, 128])
    dO_desc = TensorDescriptor.from_tensor(dO.reshape(B * H * S_len, 128), [128, 128])
    dQ_desc = TensorDescriptor.from_tensor(dQ.reshape(B * H * S_len, 128), [128, 128])
    dK_desc = TensorDescriptor.from_tensor(dK.reshape(B * H * S_len, 128), [128, 128])
    dV_desc = TensorDescriptor.from_tensor(dV.reshape(B * H * S_len, 128), [128, 128])
    
    num_blocks_q = triton.cdiv(S_len, 128)
    num_blocks_k = triton.cdiv(S_len, 128)
    grid = (num_blocks_q + num_blocks_k, B, H)
    
    _mha_bwd_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, dQ_desc, dK_desc, dV_desc, L,
        S_len, tau, H,
        BLOCK_M=128, BLOCK_N=128,
        num_blocks_q=num_blocks_q,
        num_warps=8,
        num_stages=3,
    )