import torch
import triton
import triton.language as tl
import math


@triton.jit
def backward_dK_dV_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    seq_len,
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    o_stride_b, o_stride_h, o_stride_s,
    do_stride_b, do_stride_h, do_stride_s,
    l_stride_b, l_stride_h,
    dk_stride_b, dk_stride_h, dk_stride_s,
    dv_stride_b, dv_stride_h, dv_stride_s,
    inv_sqrt_d: tl.constexpr,
    BLOCK: tl.constexpr,
):
    bid_j = tl.program_id(0)
    bid_h = tl.program_id(1)
    bid_b = tl.program_id(2)
    
    j_start = bid_j * BLOCK
    k_row_indices = j_start + tl.arange(0, BLOCK)
    d_indices = tl.arange(0, 128)
    
    # Load K and V tiles (persistent across the i-loop)
    k_ptrs = K_ptr + bid_b * k_stride_b + bid_h * k_stride_h + k_row_indices[:, None] * k_stride_s + d_indices[None, :]
    K_tile = tl.load(k_ptrs, mask=(k_row_indices[:, None] < seq_len), other=0.0)
    
    v_ptrs = V_ptr + bid_b * v_stride_b + bid_h * v_stride_h + k_row_indices[:, None] * v_stride_s + d_indices[None, :]
    V_tile = tl.load(v_ptrs, mask=(k_row_indices[:, None] < seq_len), other=0.0)
    
    dK_acc = tl.zeros((BLOCK, 128), tl.float32)
    dV_acc = tl.zeros((BLOCK, 128), tl.float32)
    
    # Loop over all query blocks
    for i in range(0, seq_len, BLOCK):
        q_row_indices = i + tl.arange(0, BLOCK)
        
        # Load Q, dO, O
        q_ptrs = Q_ptr + bid_b * q_stride_b + bid_h * q_stride_h + q_row_indices[:, None] * q_stride_s + d_indices[None, :]
        Q_tile = tl.load(q_ptrs, mask=(q_row_indices[:, None] < seq_len), other=0.0)
        
        do_ptrs = dO_ptr + bid_b * do_stride_b + bid_h * do_stride_h + q_row_indices[:, None] * do_stride_s + d_indices[None, :]
        dO_tile = tl.load(do_ptrs, mask=(q_row_indices[:, None] < seq_len), other=0.0)
        
        o_ptrs = O_ptr + bid_b * o_stride_b + bid_h * o_stride_h + q_row_indices[:, None] * o_stride_s + d_indices[None, :]
        O_tile = tl.load(o_ptrs, mask=(q_row_indices[:, None] < seq_len), other=0.0)
        
        # Load L for these specific query positions
        l_ptrs = L_ptr + bid_b * l_stride_b + bid_h * l_stride_h + q_row_indices
        L_tile = tl.load(l_ptrs, mask=(q_row_indices < seq_len), other=0.0)
        
        # Compute pre-softmax gradients scaling factor D_i = dO_i \cdot O_i
        D_i = tl.sum(dO_tile.to(tl.float32) * O_tile.to(tl.float32), axis=1)  # Shape: [BLOCK]
        
        # Forward pass inside the loop
        S = tl.dot(Q_tile, K_tile.T)                      # Shape: [BLOCK, BLOCK]
        P = tl.exp(S * inv_sqrt_d - L_tile[:, None])      # Shape: [BLOCK, BLOCK]
        
        # Backward pass inside the loop
        dP = tl.dot(dO_tile, V_tile.T)                    # Shape: [BLOCK, BLOCK]
        dS = P * (dP - D_i[:, None])                      # Shape: [BLOCK, BLOCK]
        
        # Accumulate gradients for K and V
        dK_acc = tl.dot(dS.T, Q_tile, dK_acc)             # dK += dS^T @ Q
        dV_acc = tl.dot(P.T, dO_tile, dV_acc)             # dV += P^T @ dO
        
    # Scale dK appropriately and store
    dk_ptrs = dK_ptr + bid_b * dk_stride_b + bid_h * dk_stride_h + k_row_indices[:, None] * dk_stride_s + d_indices[None, :]
    tl.store(dk_ptrs, (dK_acc * inv_sqrt_d).to(tl.bfloat16), mask=(k_row_indices[:, None] < seq_len))
    
    # Store dV
    dv_ptrs = dV_ptr + bid_b * dv_stride_b + bid_h * dv_stride_h + k_row_indices[:, None] * dv_stride_s + d_indices[None, :]
    tl.store(dv_ptrs, dV_acc.to(tl.bfloat16), mask=(k_row_indices[:, None] < seq_len))


@triton.jit
def backward_dQ_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
    seq_len,
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    o_stride_b, o_stride_h, o_stride_s,
    do_stride_b, do_stride_h, do_stride_s,
    l_stride_b, l_stride_h,
    dq_stride_b, dq_stride_h, dq_stride_s,
    inv_sqrt_d: tl.constexpr,
    BLOCK: tl.constexpr,
):
    bid_i = tl.program_id(0)
    bid_h = tl.program_id(1)
    bid_b = tl.program_id(2)
    
    i_start = bid_i * BLOCK
    q_row_indices = i_start + tl.arange(0, BLOCK)
    d_indices = tl.arange(0, 128)
    
    # Load Q, dO, O (persistent across the j-loop)
    q_ptrs = Q_ptr + bid_b * q_stride_b + bid_h * q_stride_h + q_row_indices[:, None] * q_stride_s + d_indices[None, :]
    Q_tile = tl.load(q_ptrs, mask=(q_row_indices[:, None] < seq_len), other=0.0)
    
    do_ptrs = dO_ptr + bid_b * do_stride_b + bid_h * do_stride_h + q_row_indices[:, None] * do_stride_s + d_indices[None, :]
    dO_tile = tl.load(do_ptrs, mask=(q_row_indices[:, None] < seq_len), other=0.0)
    
    o_ptrs = O_ptr + bid_b * o_stride_b + bid_h * o_stride_h + q_row_indices[:, None] * o_stride_s + d_indices[None, :]
    O_tile = tl.load(o_ptrs, mask=(q_row_indices[:, None] < seq_len), other=0.0)
    
    # Load L for the fixed query block
    l_ptrs = L_ptr + bid_b * l_stride_b + bid_h * l_stride_h + q_row_indices
    L_tile = tl.load(l_ptrs, mask=(q_row_indices < seq_len), other=0.0)
    
    # Compute pre-softmax gradients scaling factor D_i = dO_i \cdot O_i
    D_i = tl.sum(dO_tile.to(tl.float32) * O_tile.to(tl.float32), axis=1)  # Shape: [BLOCK]
    
    dQ_acc = tl.zeros((BLOCK, 128), tl.float32)
    
    # Loop over all key/value blocks
    for j in range(0, seq_len, BLOCK):
        k_row_indices = j + tl.arange(0, BLOCK)
        
        # Load K and V tiles
        k_ptrs = K_ptr + bid_b * k_stride_b + bid_h * k_stride_h + k_row_indices[:, None] * k_stride_s + d_indices[None, :]
        K_tile = tl.load(k_ptrs, mask=(k_row_indices[:, None] < seq_len), other=0.0)
        
        v_ptrs = V_ptr + bid_b * v_stride_b + bid_h * v_stride_h + k_row_indices[:, None] * v_stride_s + d_indices[None, :]
        V_tile = tl.load(v_ptrs, mask=(k_row_indices[:, None] < seq_len), other=0.0)
        
        # Forward pass inside the loop
        S = tl.dot(Q_tile, K_tile.T)                      # Shape: [BLOCK, BLOCK]
        P = tl.exp(S * inv_sqrt_d - L_tile[:, None])      # Shape: [BLOCK, BLOCK]
        
        # Backward pass inside the loop
        dP = tl.dot(dO_tile, V_tile.T)                    # Shape: [BLOCK, BLOCK]
        dS = P * (dP - D_i[:, None])                      # Shape: [BLOCK, BLOCK]
        
        # Accumulate gradients for Q
        dQ_acc = tl.dot(dS, K_tile, dQ_acc)               # dQ += dS @ K
        
    # Scale dQ appropriately and store
    dq_ptrs = dQ_ptr + bid_b * dq_stride_b + bid_h * dq_stride_h + q_row_indices[:, None] * dq_stride_s + d_indices[None, :]
    tl.store(dq_ptrs, (dQ_acc * inv_sqrt_d).to(tl.bfloat16), mask=(q_row_indices[:, None] < seq_len))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Execute the multi-head attention backward pass.
    
    Computes dQ, dK, dV utilizing the provided forward outputs (O) 
    and log-sum-exp statistics (L). Emits outputs into preassigned 
    CUDA storage passed via the destination-passing contract.
    """
    torch.cuda.set_device(Q.device)
    
    # Fetch logical sequence length `S` derived directly from input dimensions.
    B, H, S, d = Q.shape
    inv_sqrt_d = 1.0 / math.sqrt(d)
    BLOCK = 64
    
    # Capture memory strides natively using PyTorch stride inspection API
    q_strides = Q.stride()
    k_strides = K.stride()
    v_strides = V.stride()
    o_strides = O.stride()
    do_strides = dO.stride()
    l_strides = L.stride()
    dq_strides = dQ.stride()
    dk_strides = dK.stride()
    dv_strides = dV.stride()
    
    grid_dK_dV = (triton.cdiv(S, BLOCK), H, B)
    backward_dK_dV_kernel[grid_dK_dV](
        Q, K, V, O, dO, L, dK, dV, S,
        q_strides[0], q_strides[1], q_strides[2],
        k_strides[0], k_strides[1], k_strides[2],
        v_strides[0], v_strides[1], v_strides[2],
        o_strides[0], o_strides[1], o_strides[2],
        do_strides[0], do_strides[1], do_strides[2],
        l_strides[0], l_strides[1],
        dk_strides[0], dk_strides[1], dk_strides[2],
        dv_strides[0], dv_strides[1], dv_strides[2],
        inv_sqrt_d=inv_sqrt_d, BLOCK=BLOCK, num_warps=4, num_stages=2
    )
    
    grid_dQ = (triton.cdiv(S, BLOCK), H, B)
    backward_dQ_kernel[grid_dQ](
        Q, K, V, O, dO, L, dQ, S,
        q_strides[0], q_strides[1], q_strides[2],
        k_strides[0], k_strides[1], k_strides[2],
        v_strides[0], v_strides[1], v_strides[2],
        o_strides[0], o_strides[1], o_strides[2],
        do_strides[0], do_strides[1], do_strides[2],
        l_strides[0], l_strides[1],
        dq_strides[0], dq_strides[1], dq_strides[2],
        inv_sqrt_d=inv_sqrt_d, BLOCK=BLOCK, num_warps=4, num_stages=2
    )