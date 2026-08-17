import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_fwd(
    Q_desc, K_desc, V_desc, O_ptr, LSE_ptr,
    B, H, S,
    scale,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    bh_idx = tl.program_id(1)
    b_idx = bh_idx // H
    h_idx = bh_idx % H
    
    start_m = pid_m * BLOCK_M
    batch_head_offset = (b_idx * H + h_idx) * S
    
    Q_tile = Q_desc.load([batch_head_offset + start_m, 0])
    
    o0 = tl.zeros((BLOCK_M, 64), tl.float32)
    o1 = tl.zeros((BLOCK_M, 64), tl.float32)
    running_max = tl.full((BLOCK_M,), -1e20, tl.float32)
    running_sum = tl.full((BLOCK_M,), 0.0, tl.float32)
    
    q_idx = start_m + tl.arange(0, BLOCK_M)
    
    n_end = min((start_m + BLOCK_M + BLOCK_N - 1) // BLOCK_N, (S + BLOCK_N - 1) // BLOCK_N)
    
    for n in range(n_end):
        start_n = n * BLOCK_N
        
        K_tile = K_desc.load([batch_head_offset + start_n, 0])
        V_tile = V_desc.load([batch_head_offset + start_n, 0])
        
        k_left = K_tile[:, :64]
        k_right = K_tile[:, 64:]
        v_left = V_tile[:, :64]
        v_right = V_tile[:, 64:]
        
        q_left = Q_tile[:, :64]
        q_right = Q_tile[:, 64:]
        
        p = (tl.dot(q_left, k_left.T) + tl.dot(q_right, k_right.T)) * scale
        
        k_idx = start_n + tl.arange(0, BLOCK_N)
        causal_mask = q_idx[:, None] >= k_idx[None, :]
        valid_mask = causal_mask & (q_idx[:, None] < S) & (k_idx[None, :] < S)
        p = tl.where(valid_mask, p, -1e20)
        
        m_local = tl.max(p, axis=1)
        new_max = tl.maximum(running_max, m_local)
        
        scale_prev = tl.math.exp(running_max - new_max)
        
        s = tl.math.exp(p - new_max)
        sum_s = tl.sum(s, axis=1)
        
        running_sum = running_sum * scale_prev + sum_s
        
        o0 = o0 * scale_prev[:, None]
        o1 = o1 * scale_prev[:, None]
        
        s = s / (sum_s + 1e-30)[:, None]
        
        o0 += tl.dot(s, v_left)
        o1 += tl.dot(s, v_right)
        
        running_max = new_max
        
    seq_idx = start_m + tl.arange(0, BLOCK_M)
    mask_o = seq_idx[:, None] < S
    
    c_0 = tl.arange(0, 64)
    c_64 = tl.arange(0, 64) + 64
    
    batch_head_offset_O = b_idx * H * S * D + h_idx * S * D
    tl.store(O_ptr + batch_head_offset_O + seq_idx[:, None] * D + c_0[None, :], o0, mask=mask_o)
    tl.store(O_ptr + batch_head_offset_O + seq_idx[:, None] * D + c_64[None, :], o1, mask=mask_o)
    
    lse = running_max + tl.math.log(running_sum)
    tl.store(LSE_ptr + b_idx * H * S + h_idx * S + seq_idx, lse, mask=seq_idx < S)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    Q_flat = Q.contiguous().view(B * H * S, D)
    K_flat = K.contiguous().view(B * H * S, D)
    V_flat = V.contiguous().view(B * H * S, D)
    
    Q_desc = TensorDescriptor.from_tensor(Q_flat, [64, D])
    K_desc = TensorDescriptor.from_tensor(K_flat, [64, D])
    V_desc = TensorDescriptor.from_tensor(V_flat, [64, D])
    
    grid = ((S + 63) // 64, B * H)
    scale = 1.0 / (D ** 0.5)
    
    _mha_fwd[grid](
        Q_desc, K_desc, V_desc, O.data_ptr(), LSE.data_ptr(), B, H, S, scale,
        D=D, BLOCK_M=64, BLOCK_N=64,
        num_warps=4, num_stages=2
    )