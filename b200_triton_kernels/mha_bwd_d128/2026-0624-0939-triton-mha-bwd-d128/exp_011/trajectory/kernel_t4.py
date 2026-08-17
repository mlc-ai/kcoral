import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _bwd_dKdV(
    q_desc, k_desc, v_desc, o_desc, do_desc, l_ptr, d_ptr,
    dk_desc, dv_desc,
    S, H, B, scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_n = tl.program_id(0)
    h = tl.program_id(1)
    z = tl.program_id(2)
    
    base_row = (z * H + h) * S
    
    K_0 = k_desc.load([base_row + start_n * BLOCK_N, 0])
    V_0 = v_desc.load([base_row + start_n * BLOCK_N, 0])
    
    acc_dK = tl.zeros((BLOCK_N, 128), dtype=tl.float32)
    acc_dV = tl.zeros((BLOCK_N, 128), dtype=tl.float32)
    
    num_m_tiles = tl.cdiv(S, BLOCK_M)
    
    for start_m in range(num_m_tiles):
        m_row = base_row + start_m * BLOCK_M
        Q_0 = q_desc.load([m_row, 0])
        O_0 = o_desc.load([m_row, 0])
        dO_0 = do_desc.load([m_row, 0])
        
        l_idx = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        l_offset = (z * H + h) * S + l_idx
        m_mask = l_idx < S
        L_i = tl.load(l_ptr + l_offset, mask=m_mask, other=0.0)
        D_i = tl.load(d_ptr + l_offset, mask=m_mask, other=0.0)
        
        S_val = tl.dot(Q_0, K_0.T)
        dP = tl.dot(dO_0, V_0.T)
        
        P_exp = tl.exp(S_val * scale - L_i[:, None])
        
        valid_row = m_mask[:, None]
        P_exp = tl.where(valid_row, P_exp, 0.0)
        
        dS = P_exp * (dP - D_i[:, None]) * scale
        
        acc_dV = tl.dot(P_exp.T, dO_0, acc_dV)
        acc_dK = tl.dot(dS.T, Q_0, acc_dK)
    
    dk_desc.store([base_row + start_n * BLOCK_N, 0], acc_dK.to(tl.bfloat16))
    dv_desc.store([base_row + start_n * BLOCK_N, 0], acc_dV.to(tl.bfloat16))


@triton.jit
def _bwd_dQ(
    q_desc, k_desc, v_desc, o_desc, do_desc, l_ptr, d_ptr,
    dq_desc,
    S, H, B, scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    h = tl.program_id(1)
    z = tl.program_id(2)
    
    base_row = (z * H + h) * S
    m_row = base_row + start_m * BLOCK_M
    
    Q_0 = q_desc.load([m_row, 0])
    O_0 = o_desc.load([m_row, 0])
    dO_0 = do_desc.load([m_row, 0])
    
    l_idx = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    l_offset = (z * H + h) * S + l_idx
    m_mask = l_idx < S
    L_i = tl.load(l_ptr + l_offset, mask=m_mask, other=0.0)
    D_i = tl.load(d_ptr + l_offset, mask=m_mask, other=0.0)
    
    acc_dQ = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    num_n_tiles = tl.cdiv(S, BLOCK_N)
    
    for start_n in range(num_n_tiles):
        n_row = base_row + start_n * BLOCK_N
        K_0 = k_desc.load([n_row, 0])
        V_0 = v_desc.load([n_row, 0])
        
        S_val = tl.dot(Q_0, K_0.T)
        dP = tl.dot(dO_0, V_0.T)
        
        P_exp = tl.exp(S_val * scale - L_i[:, None])
        
        valid_row = m_mask[:, None]
        P_exp = tl.where(valid_row, P_exp, 0.0)
        
        dS = P_exp * (dP - D_i[:, None]) * scale
        
        acc_dQ = tl.dot(dS, K_0, acc_dQ)
    
    dq_desc.store([m_row, 0], acc_dQ.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    b, h, s, d = Q.shape
    
    D = torch.sum(dO * O, dim=-1, dtype=torch.float32)
    L = L.contiguous()
    D = D.contiguous()
    
    scale = 1.0 / (d ** 0.5)
    
    Q_2d = Q.contiguous().view(-1, 128)
    K_2d = K.contiguous().view(-1, 128)
    V_2d = V.contiguous().view(-1, 128)
    O_2d = O.contiguous().view(-1, 128)
    dO_2d = dO.contiguous().view(-1, 128)
    dQ_2d = dQ.contiguous().view(-1, 128)
    dK_2d = dK.contiguous().view(-1, 128)
    dV_2d = dV.contiguous().view(-1, 128)
    
    BLOCK_M = 64
    BLOCK_N = 64
    
    q_desc = TensorDescriptor.from_tensor(Q_2d, [BLOCK_M, 128])
    k_desc = TensorDescriptor.from_tensor(K_2d, [BLOCK_N, 128])
    v_desc = TensorDescriptor.from_tensor(V_2d, [BLOCK_N, 128])
    o_desc = TensorDescriptor.from_tensor(O_2d, [BLOCK_M, 128])
    do_desc = TensorDescriptor.from_tensor(dO_2d, [BLOCK_M, 128])
    dq_desc = TensorDescriptor.from_tensor(dQ_2d, [BLOCK_M, 128])
    dk_desc = TensorDescriptor.from_tensor(dK_2d, [BLOCK_N, 128])
    dv_desc = TensorDescriptor.from_tensor(dV_2d, [BLOCK_N, 128])
    
    grid_kdv = (triton.cdiv(s, BLOCK_N), h, b)
    _bwd_dKdV[grid_kdv](
        q_desc, k_desc, v_desc, o_desc, do_desc, L, D,
        dk_desc, dv_desc,
        s, h, b, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4
    )
    
    grid_q = (triton.cdiv(s, BLOCK_M), h, b)
    _bwd_dQ[grid_q](
        q_desc, k_desc, v_desc, o_desc, do_desc, L, D,
        dq_desc,
        s, h, b, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4
    )