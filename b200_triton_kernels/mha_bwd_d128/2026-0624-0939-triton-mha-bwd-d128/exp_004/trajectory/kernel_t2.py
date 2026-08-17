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
    
    row_offset_i = (b * H + h) * S + offset_i
    
    Q0 = q_desc.load([row_offset_i, 0])
    Q1 = q_desc.load([row_offset_i, 64])
    dO0 = do_desc.load([row_offset_i, 0])
    dO1 = do_desc.load([row_offset_i, 64])
    O0 = o_desc.load([row_offset_i, 0])
    O1 = o_desc.load([row_offset_i, 64])
    
    D = tl.sum(dO0 * O0 + dO1 * O1, axis=1)
    
    seq_idx_q = offset_i + tl.arange(0, BLOCK_S)
    l_offset_i = (b * H + h) * S + offset_i
    L = tl.load(l_ptr + l_offset_i + tl.arange(0, BLOCK_S), mask=(seq_idx_q < S), other=0.0)
    
    is_valid_row = seq_idx_q < S
    D = tl.where(is_valid_row[:, None], D, 0.0)
    
    acc_dQ0 = tl.zeros((BLOCK_S, 64), tl.float32)
    acc_dQ1 = tl.zeros((BLOCK_S, 64), tl.float32)
    
    num_k_tiles = tl.cdiv(S, BLOCK_S)
    
    for j in tl.range(num_k_tiles, num_stages=2):
        offset_j = j * BLOCK_S
        row_offset_j = (b * H + h) * S + offset_j
        
        K0_col = k_desc.load([row_offset_j, 0])
        K1_col = k_desc.load([row_offset_j, 64])
        V0_col = v_desc.load([row_offset_j, 0])
        V1_col = v_desc.load([row_offset_j, 64])
        
        seq_idx_kv = offset_j + tl.arange(0, BLOCK_S)
        mask_ij = (seq_idx_q[:, None] < S) & (seq_idx_kv[None, :] < S)
        
        S_mat = tl.dot(Q0, K0_col.T) + tl.dot(Q1, K1_col.T)
        dP = tl.dot(dO0, V0_col.T) + tl.dot(dO1, V1_col.T)
        
        P = tl.exp(S_mat * tau - L[:, None])
        dS = P * (dP - D[:, None]) * tau
        
        dS = tl.where(mask_ij, dS, 0.0)
        P = tl.where(mask_ij, P, 0.0)
        
        dS_bf16 = dS.to(tl.bfloat16)
        
        acc_dQ0 = tl.dot(dS_bf16, K0_col, acc_dQ0)
        acc_dQ1 = tl.dot(dS_bf16, K1_col, acc_dQ1)
    
    dq_desc.store([row_offset_i, 0], acc_dQ0.to(tl.bfloat16))
    dq_desc.store([row_offset_i, 64], acc_dQ1.to(tl.bfloat16))


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
    
    row_offset_j = (b * H + h) * S + offset_j
    
    K0 = k_desc.load([row_offset_j, 0])
    K1 = k_desc.load([row_offset_j, 64])
    V0 = v_desc.load([row_offset_j, 0])
    V1 = v_desc.load([row_offset_j, 64])
    
    acc_dK0 = tl.zeros((BLOCK_S, 64), tl.float32)
    acc_dK1 = tl.zeros((BLOCK_S, 64), tl.float32)
    acc_dV0 = tl.zeros((BLOCK_S, 64), tl.float32)
    acc_dV1 = tl.zeros((BLOCK_S, 64), tl.float32)
    
    seq_idx_kv = offset_j + tl.arange(0, BLOCK_S)
    
    num_q_tiles = tl.cdiv(S, BLOCK_S)
    
    for i in tl.range(num_q_tiles, num_stages=2):
        offset_i = i * BLOCK_S
        row_offset_i = (b * H + h) * S + offset_i
        
        Q0_col = q_desc.load([row_offset_i, 0])
        Q1_col = q_desc.load([row_offset_i, 64])
        dO0_col = do_desc.load([row_offset_i, 0])
        dO1_col = do_desc.load([row_offset_i, 64])
        O0_col = o_desc.load([row_offset_i, 0])
        O1_col = o_desc.load([row_offset_i, 64])
        
        D = tl.sum(dO0_col * O0_col + dO1_col * O1_col, axis=1)
        
        seq_idx_q = offset_i + tl.arange(0, BLOCK_S)
        l_offset_i = (b * H + h) * S + offset_i
        L = tl.load(l_ptr + l_offset_i + tl.arange(0, BLOCK_S), mask=(seq_idx_q < S), other=0.0)
        
        is_valid_row = seq_idx_q < S
        D = tl.where(is_valid_row[:, None], D, 0.0)
        
        mask_ij = (seq_idx_q[:, None] < S) & (seq_idx_kv[None, :] < S)
        
        S_mat = tl.dot(Q0_col, K0.T) + tl.dot(Q1_col, K1.T)
        dP = tl.dot(dO0_col, V0.T) + tl.dot(dO1_col, V1.T)
        
        P = tl.exp(S_mat * tau - L[:, None])
        dS = P * (dP - D[:, None]) * tau
        
        dS = tl.where(mask_ij, dS, 0.0)
        P = tl.where(mask_ij, P, 0.0)
        
        P_bf16 = P.to(tl.bfloat16)
        dS_bf16 = dS.to(tl.bfloat16)
        
        acc_dV0 = tl.dot(P_bf16.T, dO0_col, acc_dV0)
        acc_dV1 = tl.dot(P_bf16.T, dO1_col, acc_dV1)
        acc_dK0 = tl.dot(dS_bf16.T, Q0_col, acc_dK0)
        acc_dK1 = tl.dot(dS_bf16.T, Q1_col, acc_dK1)
    
    dk_desc.store([row_offset_j, 0], acc_dK0.to(tl.bfloat16))
    dk_desc.store([row_offset_j, 64], acc_dK1.to(tl.bfloat16))
    dv_desc.store([row_offset_j, 0], acc_dV0.to(tl.bfloat16))
    dv_desc.store([row_offset_j, 64], acc_dV1.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    tau = 1.0 / (d ** 0.5)
    BLOCK_S = 64
    
    q_2d = Q.reshape(-1, 128)
    k_2d = K.reshape(-1, 128)
    v_2d = V.reshape(-1, 128)
    o_2d = O.reshape(-1, 128)
    do_2d = dO.reshape(-1, 128)
    
    dq_2d = dQ.reshape(-1, 128)
    dk_2d = dK.reshape(-1, 128)
    dv_2d = dV.reshape(-1, 128)
    
    q_desc = TensorDescriptor.from_tensor(q_2d, [BLOCK_S, 64], padding_option="nan")
    k_desc = TensorDescriptor.from_tensor(k_2d, [BLOCK_S, 64], padding_option="zero")
    v_desc = TensorDescriptor.from_tensor(v_2d, [BLOCK_S, 64], padding_option="zero")
    o_desc = TensorDescriptor.from_tensor(o_2d, [BLOCK_S, 64], padding_option="nan")
    do_desc = TensorDescriptor.from_tensor(do_2d, [BLOCK_S, 64], padding_option="nan")
    
    dq_desc = TensorDescriptor.from_tensor(dq_2d, [BLOCK_S, 64])
    dk_desc = TensorDescriptor.from_tensor(dk_2d, [BLOCK_S, 64])
    dv_desc = TensorDescriptor.from_tensor(dv_2d, [BLOCK_S, 64])
    
    grid = (triton.cdiv(S, BLOCK_S), H, B)
    
    _dQ_kernel[grid](
        q_desc, k_desc, v_desc, do_desc, o_desc, L, dq_desc,
        B, H, S, tau,
        BLOCK_S=BLOCK_S,
        num_warps=4,
        num_stages=2,
    )
    
    _dK_dV_kernel[grid](
        q_desc, k_desc, v_desc, do_desc, o_desc, L, dk_desc, dv_desc,
        B, H, S, tau,
        BLOCK_S=BLOCK_S,
        num_warps=4,
        num_stages=2,
    )