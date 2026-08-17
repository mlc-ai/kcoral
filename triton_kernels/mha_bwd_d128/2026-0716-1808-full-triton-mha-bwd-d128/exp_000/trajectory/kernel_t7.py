import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math


@triton.jit
def load_l(L_ptr, b_h, offset, length, S_len):
    """Load logsumexp values."""
    base_offset = b_h * S_len + offset
    return tl.load(L_ptr + base_offset + tl.arange(0, length), 
                   mask=(offset + tl.arange(0, length)) < S_len, other=0.0)


@triton.jit
def bwd_dq_kernel(
    Q_desc, K_desc, V_desc, dO_desc, L_ptr, dQ_desc,
    S_len, sqrt_d,
):
    """Compute dQ for one query tile."""
    q_tile = tl.program_id(0)
    b_h = tl.program_id(1)
    
    q_offset = q_tile * 256
    if q_offset >= S_len:
        return
    
    flat_q_offset = b_h * S_len + q_offset
    
    # Load Q, dO tiles (reused across all K iterations)
    q0 = Q_desc.load([flat_q_offset, 0])     # [256, 64]
    q1 = Q_desc.load([flat_q_offset, 64])    # [256, 64]
    do0 = dO_desc.load([flat_q_offset, 0])   # [256, 64]
    do1 = dO_desc.load([flat_q_offset, 64])  # [256, 64]
    
    # Load natural-log logsumexp row vector
    L_q = load_l(L_ptr, b_h, q_offset, 256, S_len)  # [256]
    
    # Accumulators
    acc_dQ0 = tl.zeros((256, 64), tl.float32)
    acc_dQ1 = tl.zeros((256, 64), tl.float32)
    
    # Iterate over K tiles
    num_k_tiles = triton.cdiv(S_len, 256)
    for k_tile in range(num_k_tiles):
        k_offset = k_tile * 256
        if k_offset >= S_len:
            break
        
        flat_k_offset = b_h * S_len + k_offset
        
        # Load K, V tiles for this stage
        k0 = K_desc.load([flat_k_offset, 0])   # [256, 64]
        k1 = K_desc.load([flat_k_offset, 64])  # [256, 64]
        v0 = V_desc.load([flat_k_offset, 0])   # [256, 64]
        v1 = V_desc.load([flat_k_offset, 64])  # [256, 64]
        
        # Core Attention Backward Computations
        s = tl.dot(q0, k0.T) + tl.dot(q1, k1.T)    # S = Q @ K^T
        dp = tl.dot(do0, v0.T) + tl.dot(do1, v1.T) # dP = dO @ V^T
        
        # Calculate probabilities P and gradient ds = dP * P
        p = tl.exp(s * sqrt_d - L_q[:, None])
        
        # Boundary Handling Masking 
        valid_q = (q_offset + tl.arange(0, 256))[:, None] < S_len
        valid_k = (k_offset + tl.arange(0, 256))[None, :] < S_len
        p = tl.where(valid_q & valid_k, p, 0.0)
        
        ds = dp * p
        
        # Accumulate unscaled gradients
        acc_dQ0 += tl.dot(ds, k0) 
        acc_dQ1 += tl.dot(ds, k1) 
    
    # Write fully reduced outputs incorporating scaling
    dQ_desc.store([flat_q_offset, 0], (acc_dQ0 * sqrt_d).to(tl.bfloat16))
    dQ_desc.store([flat_q_offset, 64], (acc_dQ1 * sqrt_d).to(tl.bfloat16))


@triton.jit
def bwd_dk_dv_kernel(
    Q_desc, K_desc, V_desc, dO_desc, L_ptr, dK_desc, dV_desc,
    S_len, sqrt_d,
):
    """Compute dK, dV for one key tile."""
    k_tile = tl.program_id(0)
    b_h = tl.program_id(1)
    
    k_offset = k_tile * 256
    if k_offset >= S_len:
        return
    
    flat_k_offset = b_h * S_len + k_offset
    
    # Load K, V tiles (reused across all Q iterations)
    k0 = K_desc.load([flat_k_offset, 0])     # [256, 64]
    k1 = K_desc.load([flat_k_offset, 64])    # [256, 64]
    v0 = V_desc.load([flat_k_offset, 0])     # [256, 64]
    v1 = V_desc.load([flat_k_offset, 64])    # [256, 64]
    
    # Accumulators
    acc_dK0 = tl.zeros((256, 64), tl.float32)
    acc_dK1 = tl.zeros((256, 64), tl.float32)
    acc_dV0 = tl.zeros((256, 64), tl.float32)
    acc_dV1 = tl.zeros((256, 64), tl.float32)
    
    # Iterate over Q tiles
    num_q_tiles = triton.cdiv(S_len, 256)
    for q_tile in range(num_q_tiles):
        q_offset = q_tile * 256
        if q_offset >= S_len:
            break
        
        flat_q_offset = b_h * S_len + q_offset
        
        # Load Q, dO tiles for this stage
        q0 = Q_desc.load([flat_q_offset, 0])     # [256, 64]
        q1 = Q_desc.load([flat_q_offset, 64])    # [256, 64]
        do0 = dO_desc.load([flat_q_offset, 0])   # [256, 64]
        do1 = dO_desc.load([flat_q_offset, 64])  # [256, 64]
        
        L_q = load_l(L_ptr, b_h, q_offset, 256, S_len)  # [256]
        
        # Core Attention Backward Computations
        s = tl.dot(q0, k0.T) + tl.dot(q1, k1.T)    # S = Q @ K^T
        dp = tl.dot(do0, v0.T) + tl.dot(do1, v1.T) # dP = dO @ V^T
        
        # Calculate probabilities P and gradient ds = dP * P
        p = tl.exp(s * sqrt_d - L_q[:, None])
        
        # Boundary Handling Masking
        valid_q = (q_offset + tl.arange(0, 256))[:, None] < S_len
        valid_k = (k_offset + tl.arange(0, 256))[None, :] < S_len
        p = tl.where(valid_q & valid_k, p, 0.0)
        
        ds = dp * p
        
        # Accumulate unscaled gradients
        acc_dK0 += tl.dot(ds.T, q0) 
        acc_dK1 += tl.dot(ds.T, q1) 
        acc_dV0 += tl.dot(p.T, do0) 
        acc_dV1 += tl.dot(p.T, do1) 
    
    # Write fully reduced outputs incorporating scaling
    dK_desc.store([flat_k_offset, 0], (acc_dK0 * sqrt_d).to(tl.bfloat16))
    dK_desc.store([flat_k_offset, 64], (acc_dK1 * sqrt_d).to(tl.bfloat16))
    dV_desc.store([flat_k_offset, 0], acc_dV0.to(tl.bfloat16))
    dV_desc.store([flat_k_offset, 64], acc_dV1.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S_len, d = Q.shape
    sqrt_d = 1.0 / math.sqrt(d)
    
    # Flatten all inputs and outputs to 2-Dimensional space enabling uniform TMA Stride application mapping
    Q_f = Q.flatten(0, 2)
    K_f = K.flatten(0, 2)
    V_f = V.flatten(0, 2)
    dO_f = dO.flatten(0, 2)
    L_f = L.flatten(0, 1)
    dQ_f = dQ.flatten(0, 2)
    dK_f = dK.flatten(0, 2)
    dV_f = dV.flatten(0, 2)
    
    Q_desc = TensorDescriptor.from_tensor(Q_f, [256, 64])
    K_desc = TensorDescriptor.from_tensor(K_f, [256, 64])
    V_desc = TensorDescriptor.from_tensor(V_f, [256, 64])
    dO_desc = TensorDescriptor.from_tensor(dO_f, [256, 64])
    dQ_desc = TensorDescriptor.from_tensor(dQ_f, [256, 64])
    dK_desc = TensorDescriptor.from_tensor(dK_f, [256, 64])
    dV_desc = TensorDescriptor.from_tensor(dV_f, [256, 64])
    
    # Launch specialized independent backward kernels
    grid_dq = (triton.cdiv(S_len, 256), B * H)
    grid_dk_dv = (triton.cdiv(S_len, 256), B * H)

    bwd_dq_kernel[grid_dq](
        Q_desc, K_desc, V_desc, dO_desc, L_f, dQ_desc,
        S_len, sqrt_d,
        num_warps=8, num_stages=3)
    
    bwd_dk_dv_kernel[grid_dk_dv](
        Q_desc, K_desc, V_desc, dO_desc, L_f, dK_desc, dV_desc,
        S_len, sqrt_d,
        num_warps=8, num_stages=3)