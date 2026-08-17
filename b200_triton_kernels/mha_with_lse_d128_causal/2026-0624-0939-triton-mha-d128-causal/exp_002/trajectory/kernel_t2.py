import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_fwd(
    Q_desc, K_desc, V_desc, O_ptr, LSE_ptr,
    S_len, H,
    stride_b, stride_h, stride_s, stride_d,
    lse_stride_b, lse_stride_h,
    SCALE,
):
    b_idx = tl.program_id(2)
    h_idx = tl.program_id(1)
    q_idx = tl.program_id(0)
    seq_q = q_idx * 64
    
    # Load the full Q tile for the batch, head, and query sequence chunk
    Q_0 = Q_desc.load([b_idx, h_idx, seq_q, 0])
    Q_1 = Q_desc.load([b_idx, h_idx, seq_q, 64])
    Q_local = tl.stack([Q_0, Q_1])
    
    old_max = tl.full((64,), -float('inf'), dtype=tl.float32)
    old_l = tl.zeros((64,), dtype=tl.float32)
    O_acc = tl.zeros((64, 64), dtype=tl.float32)
    
    num_k_tiles = q_idx + 1
    for k_idx in range(num_k_tiles):
        seq_k = k_idx * 64
        K_0 = K_desc.load([b_idx, h_idx, seq_k, 0])
        K_1 = K_desc.load([b_idx, h_idx, seq_k, 64])
        K_local = tl.stack([K_0, K_1])
        
        V_0 = V_desc.load([b_idx, h_idx, seq_k, 0])
        V_1 = V_desc.load([b_idx, h_idx, seq_k, 64])
        V_local = tl.stack([V_0, V_1])
        
        # Element-wise multiplication over D chunks followed by reduction over features
        S_local = Q_local[:, None, :, :] * K_local[None, :, :, :]
        S_local = S_local.sum(axis=-1)
        S_local = S_local * SCALE
        
        q_seq = seq_q + tl.arange(0, 64)
        k_seq = seq_k + tl.arange(0, 64)
        mask = k_seq[None, :] <= q_seq[:, None]
        S_local = S_local * mask
        
        m_new = tl.max(S_local, axis=1)
        new_max = tl.maximum(old_max[None, :], m_new)
        corr = tl.exp(old_max[None, :] - new_max)
        
        exp_S = tl.exp(S_local - new_max[:, None])
        sum_exp = tl.sum(exp_S, axis=1)
        
        old_l = old_l * corr + sum_exp
        old_max = new_max
        
        P = exp_S / old_l[:, None]
        
        O_acc = O_acc + P[:, :, :, None] * V_local[:, None, :, :]
    
    lse_out = old_max + tl.log(old_l)
    
    q_row = seq_q + tl.arange(0, 64)
    lse_ptr = LSE_ptr + b_idx * lse_stride_b + h_idx * lse_stride_h + q_row
    tl.store(lse_ptr, lse_out, mask=(q_row < S_len))
    
    for i in range(2):
        out_ptr = O_ptr + b_idx * stride_b + h_idx * stride_h + \
                  q_row[:, None] * stride_s + (tl.arange(0, 64) + i * 64)[None, :] * stride_d
        tl.store(out_ptr, O_acc[i, :, :], mask=(q_row[:, None] < S_len))


def run(Q, K, V, O, LSE):
    """Compute causal multi-head attention forward returning O and LSE."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    grid = (triton.cdiv(S, 64), H, B)
    
    c_dim = 64
    Q_desc = TensorDescriptor.from_tensor(Q, [c_dim, c_dim])
    K_desc = TensorDescriptor.from_tensor(K, [c_dim, c_dim])
    V_desc = TensorDescriptor.from_tensor(V, [c_dim, c_dim])
    
    q_strides = torch.stride(Q)
    stride_b, stride_h, stride_s, stride_d = q_strides[0], q_strides[1], q_strides[2], q_strides[3]
    
    lse_strides = torch.stride(LSE)
    lse_stride_b, lse_stride_h, lse_stride_s = lse_strides[0], lse_strides[1], lse_strides[2]
    
    scale = 1.0 / (D ** 0.5)
    
    _mha_fwd[grid](
        Q_desc, K_desc, V_desc, O, LSE,
        S, H,
        stride_b, stride_h, stride_s, stride_d,
        lse_stride_b, lse_stride_h,
        scale,
        num_stages=3,
    )