import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def bwd_dq_kernel(
    q_desc, k_desc, v_desc, o_desc, do_desc, dq_desc,
    L, stride_l_b, stride_l_h, stride_l_s,
    B, H, S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr
):
    pid = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b_idx = pid_bh // H
    h_idx = pid_bh % H
    
    # Swizzle program instances for better L2 data reuse
    grid_m = tl.cdiv(S, BLOCK_M)
    GROUP_SIZE = 8
    group_id = pid // GROUP_SIZE
    first_pid_m = group_id * GROUP_SIZE
    group_size_m = min(grid_m - first_pid_m, GROUP_SIZE)
    pid_m = first_pid_m + (pid % group_size_m)
    
    m_start = pid_m * BLOCK_M
    is_last_m = m_start + BLOCK_M > S
    
    # Pre-calculate M bounds for masking logic
    off_m = m_start + tl.arange(0, BLOCK_M)
    valid_m = off_m < S
    
    # Clamp starting coordinates to ensure TMA hardware does not trap on un-padded boundaries
    safe_m = tl.minimum(m_start, S - 1)
    
    # Load stationary M-block tiles via TMA 
    q_tile = tl.reshape(q_desc.load([b_idx, h_idx, safe_m, 0]), (BLOCK_M, D))
    o_tile = tl.reshape(o_desc.load([b_idx, h_idx, safe_m, 0]), (BLOCK_M, D))
    do_tile = tl.reshape(do_desc.load([b_idx, h_idx, safe_m, 0]), (BLOCK_M, D))
    
    # Load 1D LSE log values
    l_ptr = L + b_idx * stride_l_b + h_idx * stride_l_h + off_m * stride_l_s
    l_tile = tl.load(l_ptr, mask=valid_m, other=0.0)
    
    # Precompute rowwise sum(dO * O)
    delta = tl.sum(do_tile.to(tl.float32) * o_tile.to(tl.float32), axis=1)
    
    dq_acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    
    num_n_blocks = tl.cdiv(S, BLOCK_N)
    for n_idx in tl.range(0, num_n_blocks, num_stages=3):
        n_start = n_idx * BLOCK_N
        is_last_n = n_start + BLOCK_N > S
        need_mask = is_last_m or is_last_n
        
        # Prevent TMA out-of-bounds hardware trap in pipeline prefetch stages
        safe_n = tl.minimum(n_start, S - 1)
        
        # Pipelined loading of stream K and V tiles
        k_tile = tl.reshape(k_desc.load([b_idx, h_idx, safe_n, 0]), (BLOCK_N, D))
        v_tile = tl.reshape(v_desc.load([b_idx, h_idx, safe_n, 0]), (BLOCK_N, D))
        
        # S = Q @ K^T
        s_mat = tl.dot(q_tile, k_tile.T, out_dtype=tl.float32) * scale
        
        # Conditional bounds masking fully bypasses overhead for inner full-blocks
        if need_mask:
            off_n = n_start + tl.arange(0, BLOCK_N)
            valid_n = off_n < S
            valid = valid_m[:, None] & valid_n[None, :]
            s_mat = tl.where(valid, s_mat, -float("inf"))
        
        # P = softmax(S)
        p_mat = tl.exp(s_mat - l_tile[:, None])
        
        # dP = dO @ V^T
        dp_mat = tl.dot(do_tile, v_tile.T, out_dtype=tl.float32)
        
        # dS = P * (dP - delta)
        ds_mat = p_mat * (dp_mat - delta[:, None]) * scale
        
        if need_mask:
            ds_mat = tl.where(valid, ds_mat, 0.0)
        
        # dQ = dS @ K
        dq_acc = tl.dot(ds_mat.to(q_tile.dtype), k_tile, acc=dq_acc)
        
    # Store finalized dQ block relying directly on un-clamped tracking coordinates
    dq_desc.store([b_idx, h_idx, m_start, 0], tl.reshape(dq_acc.to(q_tile.dtype), (1, 1, BLOCK_M, D)))


@triton.jit
def bwd_dk_dv_kernel(
    q_desc, k_desc, v_desc, o_desc, do_desc, dk_desc, dv_desc,
    L, stride_l_b, stride_l_h, stride_l_s,
    B, H, S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr
):
    pid = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b_idx = pid_bh // H
    h_idx = pid_bh % H
    
    grid_n = tl.cdiv(S, BLOCK_N)
    GROUP_SIZE = 8
    group_id = pid // GROUP_SIZE
    first_pid_n = group_id * GROUP_SIZE
    group_size_n = min(grid_n - first_pid_n, GROUP_SIZE)
    pid_n = first_pid_n + (pid % group_size_n)
    
    n_start = pid_n * BLOCK_N
    is_last_n = n_start + BLOCK_N > S
    
    off_n = n_start + tl.arange(0, BLOCK_N)
    valid_n = off_n < S
    safe_n = tl.minimum(n_start, S - 1)
    
    # Load stationary N-block tiles via TMA 
    k_tile = tl.reshape(k_desc.load([b_idx, h_idx, safe_n, 0]), (BLOCK_N, D))
    v_tile = tl.reshape(v_desc.load([b_idx, h_idx, safe_n, 0]), (BLOCK_N, D))
    
    dk_acc = tl.zeros((BLOCK_N, D), dtype=tl.float32)
    dv_acc = tl.zeros((BLOCK_N, D), dtype=tl.float32)
    
    num_m_blocks = tl.cdiv(S, BLOCK_M)
    for m_idx in tl.range(0, num_m_blocks, num_stages=3):
        m_start = m_idx * BLOCK_M
        is_last_m = m_start + BLOCK_M > S
        need_mask = is_last_n or is_last_m
        
        # Guard pipeline prefetches from TMA bounds trapping
        safe_m = tl.minimum(m_start, S - 1)
        
        # Pipelined loading of stream Q, O, dO tiles 
        q_tile = tl.reshape(q_desc.load([b_idx, h_idx, safe_m, 0]), (BLOCK_M, D))
        o_tile = tl.reshape(o_desc.load([b_idx, h_idx, safe_m, 0]), (BLOCK_M, D))
        do_tile = tl.reshape(do_desc.load([b_idx, h_idx, safe_m, 0]), (BLOCK_M, D))
        
        off_m = m_start + tl.arange(0, BLOCK_M)
        valid_m = off_m < S
        
        l_ptr = L + b_idx * stride_l_b + h_idx * stride_l_h + off_m * stride_l_s
        l_tile = tl.load(l_ptr, mask=valid_m, other=0.0)
        
        # Precompute rowwise sum(dO * O) inline per block
        delta = tl.sum(do_tile.to(tl.float32) * o_tile.to(tl.float32), axis=1)
        
        # S^T = K @ Q^T
        s_mat_T = tl.dot(k_tile, q_tile.T, out_dtype=tl.float32) * scale
        
        if need_mask:
            valid = valid_n[:, None] & valid_m[None, :]
            s_mat_T = tl.where(valid, s_mat_T, -float("inf"))
        
        # P^T = softmax(S^T)
        p_mat_T = tl.exp(s_mat_T - l_tile[None, :])
        
        # dP^T = V @ dO^T
        dp_mat_T = tl.dot(v_tile, do_tile.T, out_dtype=tl.float32)
        
        # dS^T = P^T * (dP^T - delta)
        ds_mat_T = p_mat_T * (dp_mat_T - delta[None, :]) * scale
        
        if need_mask:
            ds_mat_T = tl.where(valid, ds_mat_T, 0.0)
        
        # dK = dS^T @ Q
        dk_acc = tl.dot(ds_mat_T.to(q_tile.dtype), q_tile, acc=dk_acc)
        # dV = P^T @ dO
        dv_acc = tl.dot(p_mat_T.to(q_tile.dtype), do_tile, acc=dv_acc)
        
    # Store finalized deterministic N-blocks locally
    dk_desc.store([b_idx, h_idx, n_start, 0], tl.reshape(dk_acc.to(k_tile.dtype), (1, 1, BLOCK_N, D)))
    dv_desc.store([b_idx, h_idx, n_start, 0], tl.reshape(dv_acc.to(v_tile.dtype), (1, 1, BLOCK_N, D)))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Standard Split-Ownership Attention Backward pass using hardware TMA paths
    optimized for Blackwell architectures safely clamping loop coordinates.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    scale = 1.0 / math.sqrt(D)
    
    # Stage 1: Compute dQ. Outer-loop streams over M (queries)
    BLOCK_M_1 = 128
    BLOCK_N_1 = 64
    
    q_desc_1 = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M_1, D])
    k_desc_1 = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N_1, D])
    v_desc_1 = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N_1, D])
    o_desc_1 = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M_1, D])
    do_desc_1 = TensorDescriptor.from_tensor(dO, [1, 1, BLOCK_M_1, D])
    dq_desc_1 = TensorDescriptor.from_tensor(dQ, [1, 1, BLOCK_M_1, D])
    
    grid_dq = (triton.cdiv(S, BLOCK_M_1), B * H)
    bwd_dq_kernel[grid_dq](
        q_desc_1, k_desc_1, v_desc_1, o_desc_1, do_desc_1, dq_desc_1,
        L, L.stride(0), L.stride(1), L.stride(2),
        B, H, S, scale,
        BLOCK_M=BLOCK_M_1, BLOCK_N=BLOCK_N_1, D=D,
        num_warps=8, num_stages=3
    )
    
    # Stage 2: Compute dK and dV. Outer-loop streams over N (keys/values)
    BLOCK_M_2 = 64
    BLOCK_N_2 = 128
    
    q_desc_2 = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M_2, D])
    k_desc_2 = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N_2, D])
    v_desc_2 = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N_2, D])
    o_desc_2 = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M_2, D])
    do_desc_2 = TensorDescriptor.from_tensor(dO, [1, 1, BLOCK_M_2, D])
    dk_desc_2 = TensorDescriptor.from_tensor(dK, [1, 1, BLOCK_N_2, D])
    dv_desc_2 = TensorDescriptor.from_tensor(dV, [1, 1, BLOCK_N_2, D])
    
    grid_dk_dv = (triton.cdiv(S, BLOCK_N_2), B * H)
    bwd_dk_dv_kernel[grid_dk_dv](
        q_desc_2, k_desc_2, v_desc_2, o_desc_2, do_desc_2, dk_desc_2, dv_desc_2,
        L, L.stride(0), L.stride(1), L.stride(2),
        B, H, S, scale,
        BLOCK_M=BLOCK_M_2, BLOCK_N=BLOCK_N_2, D=D,
        num_warps=8, num_stages=3
    )