import math
import torch
import triton
import triton.language as tl

from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _bwd_dq_kernel(
    Q_desc, K_desc, V_desc, dO_desc, L_desc, dQ_desc,
    S, alpha: tl.constexpr,
):
    q_tile = tl.program_id(0)
    kv_tile = tl.program_id(1)
    bh_idx = tl.program_id(2)
    
    q_seq_idx = q_tile * 32
    kv_seq_idx = kv_tile * 32
    
    q0 = Q_desc.load([bh_idx, q_seq_idx, 0]).squeeze(0)
    q1 = Q_desc.load([bh_idx, q_seq_idx, 32]).squeeze(0)
    q2 = Q_desc.load([bh_idx, q_seq_idx, 64]).squeeze(0)
    q3 = Q_desc.load([bh_idx, q_seq_idx, 96]).squeeze(0)
    
    do0 = dO_desc.load([bh_idx, q_seq_idx, 0]).squeeze(0)
    do1 = dO_desc.load([bh_idx, q_seq_idx, 32]).squeeze(0)
    do2 = dO_desc.load([bh_idx, q_seq_idx, 64]).squeeze(0)
    do3 = dO_desc.load([bh_idx, q_seq_idx, 96]).squeeze(0)
    
    lse = L_desc.load([bh_idx, q_seq_idx]).squeeze(0)
    
    acc_dQ0 = tl.zeros((32, 32), tl.float32)
    acc_dQ1 = tl.zeros((32, 32), tl.float32)
    acc_dQ2 = tl.zeros((32, 32), tl.float32)
    acc_dQ3 = tl.zeros((32, 32), tl.float32)
    
    k0 = K_desc.load([bh_idx, kv_seq_idx, 0]).squeeze(0)
    k1 = K_desc.load([bh_idx, kv_seq_idx, 32]).squeeze(0)
    k2 = K_desc.load([bh_idx, kv_seq_idx, 64]).squeeze(0)
    k3 = K_desc.load([bh_idx, kv_seq_idx, 96]).squeeze(0)
    
    v0 = V_desc.load([bh_idx, kv_seq_idx, 0]).squeeze(0)
    v1 = V_desc.load([bh_idx, kv_seq_idx, 32]).squeeze(0)
    v2 = V_desc.load([bh_idx, kv_seq_idx, 64]).squeeze(0)
    v3 = V_desc.load([bh_idx, kv_seq_idx, 96]).squeeze(0)
    
    acc_S = tl.zeros((32, 32), tl.float32)
    acc_S = tl.dot(q0, k0.T, acc_S)
    acc_S = tl.dot(q1, k1.T, acc_S)
    acc_S = tl.dot(q2, k2.T, acc_S)
    acc_S = tl.dot(q3, k3.T, acc_S)
    
    s = acc_S * alpha
    exp_s = tl.exp(s)
    
    p = exp_s - lse[:, None]
    
    acc_dP = tl.zeros((32, 32), tl.float32)
    acc_dP = tl.dot(do0, v0.T, acc_dP)
    acc_dP = tl.dot(do1, v1.T, acc_dP)
    acc_dP = tl.dot(do2, v2.T, acc_dP)
    acc_dP = tl.dot(do3, v3.T, acc_dP)
    
    ds = p * acc_dP
    
    acc_dQ0 = tl.dot(ds, k0, acc_dQ0)
    acc_dQ1 = tl.dot(ds, k1, acc_dQ1)
    acc_dQ2 = tl.dot(ds, k2, acc_dQ2)
    acc_dQ3 = tl.dot(ds, k3, acc_dQ3)
    
    row_offsets = q_seq_idx + tl.arange(0, 32)
    
    col_offsets = 0 + tl.arange(0, 32)
    ptr = dQ_desc.store([bh_idx, row_offsets, col_offsets], acc_dQ0.to(tl.bfloat16))
    
    col_offsets = 32 + tl.arange(0, 32)
    ptr = dQ_desc.store([bh_idx, row_offsets, col_offsets], acc_dQ1.to(tl.bfloat16))
    
    col_offsets = 64 + tl.arange(0, 32)
    ptr = dQ_desc.store([bh_idx, row_offsets, col_offsets], acc_dQ2.to(tl.bfloat16))
    
    col_offsets = 96 + tl.arange(0, 32)
    ptr = dQ_desc.store([bh_idx, row_offsets, col_offsets], acc_dQ3.to(tl.bfloat16))


@triton.jit
def _bwd_dkv_kernel(
    Q_desc, K_desc, V_desc, dO_desc, L_desc, dK_desc, dV_desc,
    S, alpha: tl.constexpr,
):
    q_tile = tl.program_id(0)
    kv_tile = tl.program_id(1)
    bh_idx = tl.program_id(2)
    
    q_seq_idx = q_tile * 32
    kv_seq_idx = kv_tile * 32
    
    q0 = Q_desc.load([bh_idx, q_seq_idx, 0]).squeeze(0)
    q1 = Q_desc.load([bh_idx, q_seq_idx, 32]).squeeze(0)
    q2 = Q_desc.load([bh_idx, q_seq_idx, 64]).squeeze(0)
    q3 = Q_desc.load([bh_idx, q_seq_idx, 96]).squeeze(0)
    
    do0 = dO_desc.load([bh_idx, q_seq_idx, 0]).squeeze(0)
    do1 = dO_desc.load([bh_idx, q_seq_idx, 32]).squeeze(0)
    do2 = dO_desc.load([bh_idx, q_seq_idx, 64]).squeeze(0)
    do3 = dO_desc.load([bh_idx, q_seq_idx, 96]).squeeze(0)
    
    lse = L_desc.load([bh_idx, q_seq_idx]).squeeze(0)
    
    k0 = K_desc.load([bh_idx, kv_seq_idx, 0]).squeeze(0)
    k1 = K_desc.load([bh_idx, kv_seq_idx, 32]).squeeze(0)
    k2 = K_desc.load([bh_idx, kv_seq_idx, 64]).squeeze(0)
    k3 = K_desc.load([bh_idx, kv_seq_idx, 96]).squeeze(0)
    
    v0 = V_desc.load([bh_idx, kv_seq_idx, 0]).squeeze(0)
    v1 = V_desc.load([bh_idx, kv_seq_idx, 32]).squeeze(0)
    v2 = V_desc.load([bh_idx, kv_seq_idx, 64]).squeeze(0)
    v3 = V_desc.load([bh_idx, kv_seq_idx, 96]).squeeze(0)
    
    acc_dK0 = tl.zeros((32, 32), tl.float32)
    acc_dK1 = tl.zeros((32, 32), tl.float32)
    acc_dK2 = tl.zeros((32, 32), tl.float32)
    acc_dK3 = tl.zeros((32, 32), tl.float32)
    
    acc_dV0 = tl.zeros((32, 32), tl.float32)
    acc_dV1 = tl.zeros((32, 32), tl.float32)
    acc_dV2 = tl.zeros((32, 32), tl.float32)
    acc_dV3 = tl.zeros((32, 32), tl.float32)
    
    acc_S = tl.zeros((32, 32), tl.float32)
    acc_S = tl.dot(q0, k0.T, acc_S)
    acc_S = tl.dot(q1, k1.T, acc_S)
    acc_S = tl.dot(q2, k2.T, acc_S)
    acc_S = tl.dot(q3, k3.T, acc_S)
    
    s = acc_S * alpha
    exp_s = tl.exp(s)
    
    p = exp_s - lse[:, None]
    
    acc_dP = tl.zeros((32, 32), tl.float32)
    acc_dP = tl.dot(do0, v0.T, acc_dP)
    acc_dP = tl.dot(do1, v1.T, acc_dP)
    acc_dP = tl.dot(do2, v2.T, acc_dP)
    acc_dP = tl.dot(do3, v3.T, acc_dP)
    
    ds = p * acc_dP
    
    ds_T = ds.T
    p_T = p.T
    
    acc_dV0 = tl.dot(p_T, do0, acc_dV0)
    acc_dV1 = tl.dot(p_T, do1, acc_dV1)
    acc_dV2 = tl.dot(p_T, do2, acc_dV2)
    acc_dV3 = tl.dot(p_T, do3, acc_dV3)
    
    acc_dK0 = tl.dot(ds_T, q0, acc_dK0)
    acc_dK1 = tl.dot(ds_T, q1, acc_dK1)
    acc_dK2 = tl.dot(ds_T, q2, acc_dK2)
    acc_dK3 = tl.dot(ds_T, q3, acc_dK3)
    
    row_offsets = kv_seq_idx + tl.arange(0, 32)
    
    col_offsets = 0 + tl.arange(0, 32)
    ptr = dK_desc.store([bh_idx, row_offsets, col_offsets], acc_dK0.to(tl.bfloat16))
    ptr = dV_desc.store([bh_idx, row_offsets, col_offsets], acc_dV0.to(tl.bfloat16))
    
    col_offsets = 32 + tl.arange(0, 32)
    ptr = dK_desc.store([bh_idx, row_offsets, col_offsets], acc_dK1.to(tl.bfloat16))
    ptr = dV_desc.store([bh_idx, row_offsets, col_offsets], acc_dV1.to(tl.bfloat16))
    
    col_offsets = 64 + tl.arange(0, 32)
    ptr = dK_desc.store([bh_idx, row_offsets, col_offsets], acc_dK2.to(tl.bfloat16))
    ptr = dV_desc.store([bh_idx, row_offsets, col_offsets], acc_dV2.to(tl.bfloat16))
    
    col_offsets = 96 + tl.arange(0, 32)
    ptr = dK_desc.store([bh_idx, row_offsets, col_offsets], acc_dK3.to(tl.bfloat16))
    ptr = dV_desc.store([bh_idx, row_offsets, col_offsets], acc_dV3.to(tl.bfloat16))


@triton.jit
def _convert_to_bf16(in_ptr, out_ptr, n_elements):
    idx = tl.program_id(0) * tl.arange(0, 128)
    mask = idx < n_elements
    vals = tl.load(in_ptr + idx, mask=mask)
    tl.store(out_ptr + idx, vals.to(tl.bfloat16), mask=mask)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute backward pass of Multi-Head Attention."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    alpha = 1.0 / math.sqrt(d)
    
    Q_bh = Q.view(B * H, S, d)
    K_bh = K.view(B * H, S, d)
    V_bh = V.view(B * H, S, d)
    O_bh = O.view(B * H, S, d)
    dO_bh = dO.view(B * H, S, d)
    dQ_bh = dQ.view(B * H, S, d)
    dK_bh = dK.view(B * H, S, d)
    dV_bh = dV.view(B * H, S, d)
    
    Q_desc = TensorDescriptor.from_tensor(Q_bh, block_shape=[1, 32, 32])
    K_desc = TensorDescriptor.from_tensor(K_bh, block_shape=[1, 32, 32])
    V_desc = TensorDescriptor.from_tensor(V_bh, block_shape=[1, 32, 32])
    O_desc = TensorDescriptor.from_tensor(O_bh, block_shape=[1, 32, 32])
    dO_desc = TensorDescriptor.from_tensor(dO_bh, block_shape=[1, 32, 32])
    dQ_desc = TensorDescriptor.from_tensor(dQ_bh, block_shape=[1, 32, 32])
    dK_desc = TensorDescriptor.from_tensor(dK_bh, block_shape=[1, 32, 32])
    dV_desc = TensorDescriptor.from_tensor(dV_bh, block_shape=[1, 32, 32])
    
    L_2d = L.view(B * H, S)
    L_desc = TensorDescriptor.from_tensor(L_2d, block_shape=[1, 32])
    
    dQ_fp32 = torch.zeros_like(Q, dtype=torch.float32)
    dK_fp32 = torch.zeros_like(K, dtype=torch.float32)
    dV_fp32 = torch.zeros_like(V, dtype=torch.float32)
    
    dQ_fp32_bh = dQ_fp32.view(B * H, S, d)
    dK_fp32_bh = dK_fp32.view(B * H, S, d)
    dV_fp32_bh = dV_fp32.view(B * H, S, d)
    
    dQ_fp32_desc = TensorDescriptor.from_tensor(dQ_fp32_bh, block_shape=[1, 32, 32])
    dK_fp32_desc = TensorDescriptor.from_tensor(dK_fp32_bh, block_shape=[1, 32, 32])
    dV_fp32_desc = TensorDescriptor.from_tensor(dV_fp32_bh, block_shape=[1, 32, 32])
    
    num_tiles = triton.cdiv(S, 32)
    grid = (num_tiles, num_tiles, B * H)
    
    print(f"Grid: {grid}, S={S}, B*H={B*H}")
    
    _bwd_dq_kernel[grid](
        Q_desc, K_desc, V_desc, dO_desc, L_desc, dQ_fp32_desc,
        S, alpha=alpha,
        num_warps=4, num_stages=3,
    )
    
    _bwd_dkv_kernel[grid](
        Q_desc, K_desc, V_desc, dO_desc, L_desc, dK_fp32_desc, dV_fp32_desc,
        S, alpha=alpha,
        num_warps=4, num_stages=3,
    )
    
    convert_grid = (triton.cdiv(dQ.numel(), 128),)
    _convert_to_bf16[convert_grid](dQ_fp32.data_ptr(), dQ.data_ptr(), dQ.numel())
    _convert_to_bf16[convert_grid](dK_fp32.data_ptr(), dK.data_ptr(), dK.numel())
    _convert_to_bf16[convert_grid](dV_fp32.data_ptr(), dV.data_ptr(), dV.numel())