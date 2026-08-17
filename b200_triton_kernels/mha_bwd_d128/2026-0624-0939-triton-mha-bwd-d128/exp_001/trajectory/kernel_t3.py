import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _bwd_Q_kernel(
    desc_Q, desc_K, desc_V, desc_dO, desc_O, desc_dQ,
    L_ptr, B, H, S_len, scale, NUM_SMS: tl.constexpr,
):
    num_pid_s = tl.cdiv(S_len, 128)
    total_tiles = B * H * num_pid_s
    grid_size = min(NUM_SMS, total_tiles)
    
    for i in tl.range(tl.program_id(0), total_tiles, grid_size):
        b_h_idx = i // num_pid_s
        q_idx = i % num_pid_s
        q_idx_base = q_idx * 128
        
        s_offs = q_idx_base + tl.arange(0, 128)
        mask_q = s_offs < S_len
        
        Q_0 = tl.cast(desc_Q.load([b_h_idx * S_len + q_idx_base, 0]), tl.float32)
        Q_1 = tl.cast(desc_Q.load([b_h_idx * S_len + q_idx_base, 64]), tl.float32)
        dO_0 = tl.cast(desc_dO.load([b_h_idx * S_len + q_idx_base, 0]), tl.float32)
        dO_1 = tl.cast(desc_dO.load([b_h_idx * S_len + q_idx_base, 64]), tl.float32)
        
        O_0 = tl.cast(desc_O.load([b_h_idx * S_len + q_idx_base, 0]), tl.float32)
        O_1 = tl.cast(desc_O.load([b_h_idx * S_len + q_idx_base, 64]), tl.float32)
        D_val = (dO_0 * O_0 + dO_1 * O_1).sum(axis=1)
        
        l_offs = b_h_idx * S_len + s_offs
        L_val = tl.load(L_ptr + l_offs, mask=mask_q, other=0.0)
        
        acc_dQ_0 = tl.zeros((128, 64), tl.float32)
        acc_dQ_1 = tl.zeros((128, 64), tl.float32)
        
        for kv_idx in range(num_pid_s):
            kv_idx_base = kv_idx * 128
            
            K_0 = tl.cast(desc_K.load([b_h_idx * S_len + kv_idx_base, 0]), tl.float32)
            K_1 = tl.cast(desc_K.load([b_h_idx * S_len + kv_idx_base, 64]), tl.float32)
            V_0 = tl.cast(desc_V.load([b_h_idx * S_len + kv_idx_base, 0]), tl.float32)
            V_1 = tl.cast(desc_V.load([b_h_idx * S_len + kv_idx_base, 64]), tl.float32)
            
            S_scores = (tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)) * scale
            P = tl.exp(S_scores - L_val[:, None])
            
            dP = tl.dot(dO_0, V_0.T) + tl.dot(dO_1, V_1.T)
            dS = P * (dP - D_val[:, None]) * scale
            
            acc_dQ_0 += tl.dot(dS, K_0)
            acc_dQ_1 += tl.dot(dS, K_1)
        
        desc_dQ.store([b_h_idx * S_len + q_idx_base, 0], acc_dQ_0.to(tl.bfloat16))
        desc_dQ.store([b_h_idx * S_len + q_idx_base, 64], acc_dQ_1.to(tl.bfloat16))


@triton.jit
def _bwd_KV_kernel(
    desc_Q, desc_K, desc_V, desc_dO, desc_O, desc_dK, desc_dV,
    L_ptr, B, H, S_len, scale, NUM_SMS: tl.constexpr,
):
    num_pid_s = tl.cdiv(S_len, 128)
    total_tiles = B * H * num_pid_s
    grid_size = min(NUM_SMS, total_tiles)
    
    for i in tl.range(tl.program_id(0), total_tiles, grid_size):
        kv_idx = i // (B * H)
        b_h_idx = i % (B * H)
        kv_idx_base = kv_idx * 128
        
        s_offs = kv_idx_base + tl.arange(0, 128)
        mask_kv = s_offs < S_len
        
        K_0 = tl.cast(desc_K.load([b_h_idx * S_len + kv_idx_base, 0]), tl.float32)
        K_1 = tl.cast(desc_K.load([b_h_idx * S_len + kv_idx_base, 64]), tl.float32)
        V_0 = tl.cast(desc_V.load([b_h_idx * S_len + kv_idx_base, 0]), tl.float32)
        V_1 = tl.cast(desc_V.load([b_h_idx * S_len + kv_idx_base, 64]), tl.float32)
        
        acc_dK_0 = tl.zeros((128, 64), tl.float32)
        acc_dK_1 = tl.zeros((128, 64), tl.float32)
        acc_dV_0 = tl.zeros((128, 64), tl.float32)
        acc_dV_1 = tl.zeros((128, 64), tl.float32)
        
        for q_idx in range(num_pid_s):
            q_idx_base = q_idx * 128
            q_offs = q_idx_base + tl.arange(0, 128)
            mask_q = q_offs < S_len
            
            Q_0 = tl.cast(desc_Q.load([b_h_idx * S_len + q_idx_base, 0]), tl.float32)
            Q_1 = tl.cast(desc_Q.load([b_h_idx * S_len + q_idx_base, 64]), tl.float32)
            dO_0 = tl.cast(desc_dO.load([b_h_idx * S_len + q_idx_base, 0]), tl.float32)
            dO_1 = tl.cast(desc_dO.load([b_h_idx * S_len + q_idx_base, 64]), tl.float32)
            
            O_0 = tl.cast(desc_O.load([b_h_idx * S_len + q_idx_base, 0]), tl.float32)
            O_1 = tl.cast(desc_O.load([b_h_idx * S_len + q_idx_base, 64]), tl.float32)
            D_val = (dO_0 * O_0 + dO_1 * O_1).sum(axis=1)
            
            l_offs = b_h_idx * S_len + q_offs
            L_val = tl.load(L_ptr + l_offs, mask=mask_q, other=0.0)
            
            S_scores = (tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)) * scale
            P = tl.exp(S_scores - L_val[:, None])
            
            dP = tl.dot(dO_0, V_0.T) + tl.dot(dO_1, V_1.T)
            dS = P * (dP - D_val[:, None]) * scale
            
            dS_T = dS.T
            P_T = P.T
            
            acc_dK_0 += tl.dot(dS_T, Q_0)
            acc_dK_1 += tl.dot(dS_T, Q_1)
            acc_dV_0 += tl.dot(P_T, dO_0)
            acc_dV_1 += tl.dot(P_T, dO_1)
        
        desc_dK.store([b_h_idx * S_len + kv_idx_base, 0], acc_dK_0.to(tl.bfloat16))
        desc_dK.store([b_h_idx * S_len + kv_idx_base, 64], acc_dK_1.to(tl.bfloat16))
        desc_dV.store([b_h_idx * S_len + kv_idx_base, 0], acc_dV_0.to(tl.bfloat16))
        desc_dV.store([b_h_idx * S_len + kv_idx_base, 64], acc_dV_1.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Execute optimized multi-head attention backward pass."""
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    scale = 1.0 / (d ** 0.5)
    
    desc_Q = TensorDescriptor.from_tensor(Q, [128, 64])
    desc_K = TensorDescriptor.from_tensor(K, [128, 64])
    desc_V = TensorDescriptor.from_tensor(V, [128, 64])
    desc_dO = TensorDescriptor.from_tensor(dO, [128, 64])
    desc_O = TensorDescriptor.from_tensor(O, [128, 64])
    desc_dQ = TensorDescriptor.from_tensor(dQ, [128, 64])
    desc_dK = TensorDescriptor.from_tensor(dK, [128, 64])
    desc_dV = TensorDescriptor.from_tensor(dV, [128, 64])
    
    num_pid_s = triton.cdiv(S, 128)
    total_tiles = B * H * num_pid_s
    grid_size = min(132, total_tiles)
    grid = (grid_size,)
    
    _bwd_Q_kernel[grid](
        desc_Q, desc_K, desc_V, desc_dO, desc_O, desc_dQ, L,
        B, H, S, scale, NUM_SMS=132,
        num_warps=8, num_stages=3
    )
    
    _bwd_KV_kernel[grid](
        desc_Q, desc_K, desc_V, desc_dO, desc_O, desc_dK, desc_dV, L,
        B, H, S, scale, NUM_SMS=132,
        num_warps=8, num_stages=3
    )