import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def bwd_dq_kernel(
    q_desc, k_desc, v_desc, o_desc, do_desc, l_ptr, dq_desc,
    S, scale, BLOCK_M, BLOCK_N, BLOCK_K
):
    i = tl.program_id(0)
    bh = tl.program_id(1)
    
    row_off_i = bh * S + i * BLOCK_M
    
    q_0 = q_desc.load([row_off_i, 0])
    q_1 = q_desc.load([row_off_i, BLOCK_K])
    do_0 = do_desc.load([row_off_i, 0])
    do_1 = do_desc.load([row_off_i, BLOCK_K])
    o_0 = o_desc.load([row_off_i, 0])
    o_1 = o_desc.load([row_off_i, BLOCK_K])
    
    D = tl.zeros((BLOCK_M, 1), dtype=tl.float32)
    D += tl.sum(do_0 * o_0, axis=1, keep_dims=True)
    D += tl.sum(do_1 * o_1, axis=1, keep_dims=True)
    
    mask_l = (i * BLOCK_M + tl.arange(0, BLOCK_M)) < S
    L_i = tl.load(l_ptr + bh * S + i * BLOCK_M + tl.arange(0, BLOCK_M), mask=mask_l, other=0.0)
    
    dQ_acc_0 = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    dQ_acc_1 = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    
    num_blocks = tl.cdiv(S, BLOCK_N)
    for j in range(num_blocks):
        row_off_j = bh * S + j * BLOCK_N
        
        k_0 = k_desc.load([row_off_j, 0])
        k_1 = k_desc.load([row_off_j, BLOCK_K])
        v_0 = v_desc.load([row_off_j, 0])
        v_1 = v_desc.load([row_off_j, BLOCK_K])
        
        S_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        S_acc = tl.dot(q_0, k_0.T, S_acc)
        S_acc = tl.dot(q_1, k_1.T, S_acc)
            
        dP_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        dP_acc = tl.dot(do_0, v_0.T, dP_acc)
        dP_acc = tl.dot(do_1, v_1.T, dP_acc)
            
        P = tl.exp(S_acc * scale - L_i[:, None])
        
        row = i * BLOCK_M + tl.arange(0, BLOCK_M)
        col = j * BLOCK_N + tl.arange(0, BLOCK_N)
        valid = (row[:, None] < S) & (col[None, :] < S)
        P = P * valid.to(tl.float32)
        
        dS = P * (dP_acc - D) * scale
        
        dQ_acc_0 = tl.dot(dS, k_0, dQ_acc_0)
        dQ_acc_1 = tl.dot(dS, k_1, dQ_acc_1)
            
    dq_desc.store([row_off_i, 0], dQ_acc_0.to(tl.bfloat16))
    dq_desc.store([row_off_i, BLOCK_K], dQ_acc_1.to(tl.bfloat16))


@triton.jit
def bwd_dkv_kernel(
    q_desc, k_desc, v_desc, o_desc, do_desc, l_ptr, dk_desc, dv_desc,
    S, scale, BLOCK_M, BLOCK_N, BLOCK_K
):
    j = tl.program_id(0)
    bh = tl.program_id(1)
    
    row_off_j = bh * S + j * BLOCK_N
    
    k_0 = k_desc.load([row_off_j, 0])
    k_1 = k_desc.load([row_off_j, BLOCK_K])
    v_0 = v_desc.load([row_off_j, 0])
    v_1 = v_desc.load([row_off_j, BLOCK_K])
    
    dk_acc_0 = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    dk_acc_1 = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    dv_acc_0 = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    dv_acc_1 = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    
    num_blocks = tl.cdiv(S, BLOCK_M)
    for i in range(num_blocks):
        row_off_i = bh * S + i * BLOCK_M
        
        q_0 = q_desc.load([row_off_i, 0])
        q_1 = q_desc.load([row_off_i, BLOCK_K])
        do_0 = do_desc.load([row_off_i, 0])
        do_1 = do_desc.load([row_off_i, BLOCK_K])
        o_0 = o_desc.load([row_off_i, 0])
        o_1 = o_desc.load([row_off_i, BLOCK_K])
        
        D = tl.zeros((BLOCK_M, 1), dtype=tl.float32)
        D += tl.sum(do_0 * o_0, axis=1, keep_dims=True)
        D += tl.sum(do_1 * o_1, axis=1, keep_dims=True)
            
        mask_l = (i * BLOCK_M + tl.arange(0, BLOCK_M)) < S
        L_i = tl.load(l_ptr + bh * S + i * BLOCK_M + tl.arange(0, BLOCK_M), mask=mask_l, other=0.0)
        
        S_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        S_acc = tl.dot(q_0, k_0.T, S_acc)
        S_acc = tl.dot(q_1, k_1.T, S_acc)
            
        dP_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        dP_acc = tl.dot(do_0, v_0.T, dP_acc)
        dP_acc = tl.dot(do_1, v_1.T, dP_acc)
            
        P = tl.exp(S_acc * scale - L_i[:, None])
        
        row = i * BLOCK_M + tl.arange(0, BLOCK_M)
        col = j * BLOCK_N + tl.arange(0, BLOCK_N)
        valid = (row[:, None] < S) & (col[None, :] < S)
        P = P * valid.to(tl.float32)
        
        dS = P * (dP_acc - D) * scale
        
        dS_T = dS.T
        P_T = P.T
            
        dk_acc_0 = tl.dot(dS_T, q_0, dk_acc_0)
        dk_acc_1 = tl.dot(dS_T, q_1, dk_acc_1)
            
        dv_acc_0 = tl.dot(P_T, do_0, dv_acc_0)
        dv_acc_1 = tl.dot(P_T, do_1, dv_acc_1)
            
    dk_desc.store([row_off_j, 0], dk_acc_0.to(tl.bfloat16))
    dk_desc.store([row_off_j, BLOCK_K], dk_acc_1.to(tl.bfloat16))
    dv_desc.store([row_off_j, 0], dv_acc_0.to(tl.bfloat16))
    dv_desc.store([row_off_j, BLOCK_K], dv_acc_1.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute the backward pass of Multi-Head Attention."""
    B, H, S, d = Q.shape
    assert B == 4 and H == 48 and d == 128
    assert O.shape == (B, H, S, d) and dO.shape == (B, H, S, d)
    assert L.shape == (B, H, S)
    assert dQ.shape == (B, H, S, d) and dK.shape == (B, H, S, d) and dV.shape == (B, H, S, d)
    
    torch.cuda.set_device(Q.device)
    
    total_S = B * H * S
    scale = 1.0 / (d ** 0.5)
    
    Q_2d = Q.view(total_S, d)
    K_2d = K.view(total_S, d)
    V_2d = V.view(total_S, d)
    O_2d = O.view(total_S, d)
    dO_2d = dO.view(total_S, d)
    dQ_2d = dQ.view(total_S, d)
    dK_2d = dK.view(total_S, d)
    dV_2d = dV.view(total_S, d)
    
    q_desc = TensorDescriptor.from_tensor(Q_2d, [128, 64])
    k_desc = TensorDescriptor.from_tensor(K_2d, [128, 64])
    v_desc = TensorDescriptor.from_tensor(V_2d, [128, 64])
    o_desc = TensorDescriptor.from_tensor(O_2d, [128, 64])
    do_desc = TensorDescriptor.from_tensor(dO_2d, [128, 64])
    dq_desc = TensorDescriptor.from_tensor(dQ_2d, [128, 64])
    dk_desc = TensorDescriptor.from_tensor(dK_2d, [128, 64])
    dv_desc = TensorDescriptor.from_tensor(dV_2d, [128, 64])
    
    num_blocks_q = triton.cdiv(S, 128)
    num_blocks_kv = triton.cdiv(S, 64)
    grid_dq = (num_blocks_q, B * H)
    grid_dkv = (num_blocks_kv, B * H)
    
    L_flat = L.view(B * H * S)
    
    bwd_dq_kernel[grid_dq](q_desc, k_desc, v_desc, o_desc, do_desc, L_flat, dq_desc,
                        S, scale, BLOCK_M=128, BLOCK_N=64, BLOCK_K=64,
                        num_warps=8, num_stages=3)
                        
    bwd_dkv_kernel[grid_dkv](q_desc, k_desc, v_desc, o_desc, do_desc, L_flat, dk_desc, dv_desc,
                         S, scale, BLOCK_M=128, BLOCK_N=64, BLOCK_K=64,
                         num_warps=8, num_stages=3)