import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _compute_D_kernel(
    dO_ptr,
    O_ptr,
    D_ptr,
    N,  # total B*H*S elements
    BLOCK: tl.constexpr,
    HEAD_DIM: tl.constexpr,  # 128
):
    base_idx = tl.program_id(0) * BLOCK
    idx = base_idx + tl.arange(0, BLOCK)
    mask = idx < N
    
    sum_val = tl.zeros((BLOCK,), tl.float32)
    # Iterate over the 128-dimensional head chunk by chunk
    for chunk in range(0, HEAD_DIM, 64):
        off = idx[:, None] * HEAD_DIM + chunk + tl.arange(0, 64)[None, :]
        val_do = tl.load(dO_ptr + off, mask=(idx[:, None] < N), other=0.0)
        val_o = tl.load(O_ptr + off, mask=(idx[:, None] < N), other=0.0)
        sum_val += (val_do * val_o).sum(axis=1)
    
    tl.store(D_ptr + idx, sum_val, mask=mask)


@triton.jit
def _bwd_Q_kernel(
    desc_Q, desc_K, desc_V, desc_dO, desc_dQ,
    L_ptr, D_ptr,
    B, H, S, scale,
    BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr,
):
    start_pid = tl.program_id(0)
    num_pid_s = tl.cdiv(S, BLOCK_Q)
    total_tiles = num_pid_s * B * H
    
    # Coordinate mapping distributing batches and heads across the grid
    pid = start_pid + tl.program_id(1) * total_tiles
    q_idx_base = (pid % num_pid_s) * BLOCK_Q
    b_h_idx = pid // num_pid_s
    
    # Load query representation and corresponding metrics
    Q_half_a = desc_Q.load([b_h_idx * S + q_idx_base, 0])
    Q_half_b = desc_Q.load([b_h_idx * S + q_idx_base, 64])
    dO_half_a = desc_dO.load([b_h_idx * S + q_idx_base, 0])
    dO_half_b = desc_dO.load([b_h_idx * S + q_idx_base, 64])
    
    l_offs = b_h_idx * S + q_idx_base + tl.arange(0, BLOCK_Q)
    mask_l = l_offs < (b_h_idx * S + S)
    L_vals = tl.load(L_ptr + l_offs, mask=mask_l, other=0.0)
    D_vals = tl.load(D_ptr + l_offs, mask=mask_l, other=0.0)
    
    acc_dQ_a = tl.zeros((BLOCK_Q, 64), tl.float32)
    acc_dQ_b = tl.zeros((BLOCK_Q, 64), tl.float32)
    
    # Traverse the entire sequence length processing key/value blocks
    for kv_idx_base in range(0, S, BLOCK_KV):
        b_K_half_a = desc_K.load([b_h_idx * S + kv_idx_base, 0])
        b_K_half_b = desc_K.load([b_h_idx * S + kv_idx_base, 64])
        b_V_half_a = desc_V.load([b_h_idx * S + kv_idx_base, 0])
        b_V_half_b = desc_V.load([b_h_idx * S + kv_idx_base, 64])
        
        K_half_a_T = tl.permute(b_K_half_a, [1, 0])
        K_half_b_T = tl.permute(b_K_half_b, [1, 0])
        V_half_a_T = tl.permute(b_V_half_a, [1, 0])
        V_half_b_T = tl.permute(b_V_half_b, [1, 0])
        
        S = (tl.dot(Q_half_a, K_half_a_T) + tl.dot(Q_half_b, K_half_b_T)) * scale
        P = tl.exp(S - L_vals[:, None])
        
        dP = tl.dot(dO_half_a, V_half_a_T) + tl.dot(dO_half_b, V_half_b_T)
        
        # Softmax backward reformulated projecting onto orthogonal residuals
        dS = P * (dP - D_vals[:, None]) * scale
        
        # Accumulate exact gradients resolving into respective head subspaces
        acc_dQ_a += tl.dot(dS, b_K_half_a)
        acc_dQ_b += tl.dot(dS, b_K_half_b)
        
    desc_dQ.store([b_h_idx * S + q_idx_base, 0], acc_dQ_a.to(tl.bfloat16))
    desc_dQ.store([b_h_idx * S + q_idx_base, 64], acc_dQ_b.to(tl.bfloat16))


@triton.jit
def _bwd_KV_kernel(
    desc_Q, desc_K, desc_V, desc_dO, desc_dK, desc_dV,
    L_ptr, D_ptr,
    B, H, S, scale,
    BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr,
):
    start_pid = tl.program_id(0)
    num_pid_s = tl.cdiv(S, BLOCK_KV)
    total_tiles = num_pid_s * B * H
    
    # Map linear execution index into logical coordinate space
    pid = start_pid + tl.program_id(1) * total_tiles
    kv_idx_base = (pid % num_pid_s) * BLOCK_KV
    b_h_idx = pid // num_pid_s
    
    # Materialize key and value fragments spanning the full feature extent
    K_half_a = desc_K.load([b_h_idx * S + kv_idx_base, 0])
    K_half_b = desc_K.load([b_h_idx * S + kv_idx_base, 64])
    V_half_a = desc_V.load([b_h_idx * S + kv_idx_base, 0])
    V_half_b = desc_V.load([b_h_idx * S + kv_idx_base, 64])
    
    K_half_a_T = tl.permute(K_half_a, [1, 0])
    K_half_b_T = tl.permute(K_half_b, [1, 0])
    V_half_a_T = tl.permute(V_half_a, [1, 0])
    V_half_b_T = tl.permute(V_half_b, [1, 0])
    
    acc_dK_a = tl.zeros((BLOCK_KV, 64), tl.float32)
    acc_dK_b = tl.zeros((BLOCK_KV, 64), tl.float32)
    acc_dV_a = tl.zeros((BLOCK_KV, 64), tl.float32)
    acc_dV_b = tl.zeros((BLOCK_KV, 64), tl.float32)
    
    # Iterate systematically over all queries contributing to this key/value block
    for q_idx_base in range(0, S, BLOCK_Q):
        q_Q_half_a = desc_Q.load([b_h_idx * S + q_idx_base, 0])
        q_Q_half_b = desc_Q.load([b_h_idx * S + q_idx_base, 64])
        q_dO_half_a = desc_dO.load([b_h_idx * S + q_idx_base, 0])
        q_dO_half_b = desc_dO.load([b_h_idx * S + q_idx_base, 64])
        
        l_offs = b_h_idx * S + q_idx_base + tl.arange(0, BLOCK_Q)
        mask_l = l_offs < (b_h_idx * S + S)
        l_vals = tl.load(L_ptr + l_offs, mask=mask_l, other=0.0)
        d_vals = tl.load(D_ptr + l_offs, mask=mask_l, other=0.0)
        
        S = (tl.dot(q_Q_half_a, K_half_a_T) + tl.dot(q_Q_half_b, K_half_b_T)) * scale
        P = tl.exp(S - l_vals[:, None])
        
        dP = tl.dot(q_dO_half_a, V_half_a_T) + tl.dot(q_dO_half_b, V_half_b_T)
        
        dS = P * (dP - d_vals[:, None]) * scale
        
        dS_T = tl.permute(dS, [1, 0])
        P_T = tl.permute(P, [1, 0])
        
        acc_dK_a += tl.dot(dS_T, q_Q_half_a)
        acc_dK_b += tl.dot(dS_T, q_Q_half_b)
        acc_dV_a += tl.dot(P_T, q_dO_half_a)
        acc_dV_b += tl.dot(P_T, q_dO_half_b)
        
    desc_dK.store([b_h_idx * S + kv_idx_base, 0], acc_dK_a.to(tl.bfloat16))
    desc_dK.store([b_h_idx * S + kv_idx_base, 64], acc_dK_b.to(tl.bfloat16))
    desc_dV.store([b_h_idx * S + kv_idx_base, 0], acc_dV_a.to(tl.bfloat16))
    desc_dV.store([b_h_idx * S + kv_idx_base, 64], acc_dV_b.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Execute optimized multi-head attention backward pass."""
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    # Inverse square root scaling matching standard attention formulations
    scale = 1.0 / (d ** 0.5)
    
    # Configure unified memory access descriptors enabling hardware TMA acceleration
    desc_Q = TensorDescriptor.from_tensor(Q, [64, 64])
    desc_K = TensorDescriptor.from_tensor(K, [64, 64])
    desc_V = TensorDescriptor.from_tensor(V, [64, 64])
    desc_dO = TensorDescriptor.from_tensor(dO, [64, 64])
    desc_dQ = TensorDescriptor.from_tensor(dQ, [64, 64])
    desc_dK = TensorDescriptor.from_tensor(dK, [64, 64])
    desc_dV = TensorDescriptor.from_tensor(dV, [64, 64])
    
    D = torch.empty((B, H, S), dtype=torch.float32, device=Q.device)
    
    # Phase 1: Compute per-position diagonal scaling factors
    grid_D = (triton.cdiv(B * H * S, 64),)
    _compute_D_kernel[grid_D](
        dO, O, D, B * H * S, BLOCK=64, HEAD_DIM=128
    )
    
    # Phase 2: Compute query gradients traversing the sequence dimension
    grid = lambda META: (1, B * H, triton.cdiv(S, 64))
    _bwd_Q_kernel[grid](
        desc_Q, desc_K, desc_V, desc_dO, desc_dQ, L, D,
        B, H, S, scale,
        BLOCK_Q=64, BLOCK_KV=64,
        num_warps=8, num_stages=3, num_ctas=1
    )
    
    # Phase 3: Compute key and value gradients symmetrically iterating over queries
    _bwd_KV_kernel[grid](
        desc_Q, desc_K, desc_V, desc_dO, desc_dK, desc_dV, L, D,
        B, H, S, scale,
        BLOCK_Q=64, BLOCK_KV=64,
        num_warps=8, num_stages=3, num_ctas=1
    )