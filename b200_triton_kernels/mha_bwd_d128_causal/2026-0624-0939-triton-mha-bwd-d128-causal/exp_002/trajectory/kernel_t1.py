import torch
import triton
import triton.language as tl
import math
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def dv_dk_kernel(
    q_desc, k_desc, v_desc, o_desc, do_desc,
    l_ptr, dv_desc, dk_desc,
    S, H, B,
    stride_l,
    scale,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b_idx = tl.program_id(2)
    h_idx = tl.program_id(1)
    j = tl.program_id(0)
    
    b_h_idx = b_idx * H + h_idx
    
    # Load invariant K and V for block j (split d into two 64-wide chunks)
    K_j0 = k_desc.load([b_h_idx, j * BLOCK_S, 0])
    K_j1 = k_desc.load([b_h_idx, j * BLOCK_S, BLOCK_D])
    V_j0 = v_desc.load([b_h_idx, j * BLOCK_S, 0])
    V_j1 = v_desc.load([b_h_idx, j * BLOCK_S, BLOCK_D])
    
    dV_acc0 = tl.zeros((BLOCK_S, BLOCK_D), tl.float32)
    dV_acc1 = tl.zeros((BLOCK_S, BLOCK_D), tl.float32)
    dK_acc0 = tl.zeros((BLOCK_S, BLOCK_D), tl.float32)
    dK_acc1 = tl.zeros((BLOCK_S, BLOCK_D), tl.float32)
    
    num_blocks = tl.cdiv(S, BLOCK_S)
    
    # Iterate over all query blocks causally following or overlapping with key block j
    for i in range(j, num_blocks):
        # Load Q, O, dO for block i (split d)
        Q_i0 = q_desc.load([b_h_idx, i * BLOCK_S, 0])
        Q_i1 = q_desc.load([b_h_idx, i * BLOCK_S, BLOCK_D])
        O_i0 = o_desc.load([b_h_idx, i * BLOCK_S, 0])
        O_i1 = o_desc.load([b_h_idx, i * BLOCK_S, BLOCK_D])
        dO_i0 = do_desc.load([b_h_idx, i * BLOCK_S, 0])
        dO_i1 = do_desc.load([b_h_idx, i * BLOCK_S, BLOCK_D])
        
        # Load LogSumExp for block i
        q_idx_base = i * BLOCK_S + tl.arange(0, BLOCK_S)
        L_i = tl.load(l_ptr + b_h_idx * stride_l + q_idx_base, mask=q_idx_base < S, other=0.0)
        
        # Forward logic to get P
        s = tl.dot(Q_i0, K_j0.T) + tl.dot(Q_i1, K_j1.T)
        
        q_idx_2d = q_idx_base[:, None]
        k_idx_base = j * BLOCK_S + tl.arange(0, BLOCK_S)
        k_idx_2d = k_idx_base[None, :]
        causal_mask = (q_idx_2d >= k_idx_2d) & (q_idx_2d < S) & (k_idx_2d < S)
        
        p = tl.exp(s * scale - L_i[:, None])
        p = tl.where(causal_mask, p, 0.0)
        
        # Backward logic components
        D = tl.sum(O_i0.to(tl.float32) * dO_i0.to(tl.float32), axis=1) + \
            tl.sum(O_i1.to(tl.float32) * dO_i1.to(tl.float32), axis=1)
        
        dp = tl.dot(dO_i0, V_j0.T) + tl.dot(dO_i1, V_j1.T)
        
        ds = p * (dp - D[:, None]) * scale
        
        # Accumulate dV and dK
        dV_acc0 = tl.dot(p.T, dO_i0.to(tl.float32), dV_acc0)
        dV_acc1 = tl.dot(p.T, dO_i1.to(tl.float32), dV_acc1)
        dK_acc0 = tl.dot(ds.T, Q_i0.to(tl.float32), dK_acc0)
        dK_acc1 = tl.dot(ds.T, Q_i1.to(tl.float32), dK_acc1)
    
    # Flush accumulated gradients
    dv_desc.store([b_h_idx, j * BLOCK_S, 0], dV_acc0.to(tl.bfloat16))
    dv_desc.store([b_h_idx, j * BLOCK_S, BLOCK_D], dV_acc1.to(tl.bfloat16))
    
    dk_desc.store([b_h_idx, j * BLOCK_S, 0], dK_acc0.to(tl.bfloat16))
    dk_desc.store([b_h_idx, j * BLOCK_S, BLOCK_D], dK_acc1.to(tl.bfloat16))


@triton.jit
def dq_kernel(
    q_desc, k_desc, v_desc, o_desc, do_desc,
    l_ptr, dq_desc,
    S, H, B,
    stride_l,
    scale,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b_idx = tl.program_id(2)
    h_idx = tl.program_id(1)
    i = tl.program_id(0)
    
    b_h_idx = b_idx * H + h_idx
    
    # Load invariant Query information (split d)
    Q_i0 = q_desc.load([b_h_idx, i * BLOCK_S, 0])
    Q_i1 = q_desc.load([b_h_idx, i * BLOCK_S, BLOCK_D])
    O_i0 = o_desc.load([b_h_idx, i * BLOCK_S, 0])
    O_i1 = o_desc.load([b_h_idx, i * BLOCK_S, BLOCK_D])
    dO_i0 = do_desc.load([b_h_idx, i * BLOCK_S, 0])
    dO_i1 = do_desc.load([b_h_idx, i * BLOCK_S, BLOCK_D])
    
    # Load LogSumExp for block i
    q_idx_base = i * BLOCK_S + tl.arange(0, BLOCK_S)
    L_i = tl.load(l_ptr + b_h_idx * stride_l + q_idx_base, mask=q_idx_base < S, other=0.0)
    
    # Precompute D invariant for this query block
    D = tl.sum(O_i0.to(tl.float32) * dO_i0.to(tl.float32), axis=1) + \
        tl.sum(O_i1.to(tl.float32) * dO_i1.to(tl.float32), axis=1)
    
    dQ_acc0 = tl.zeros((BLOCK_S, BLOCK_D), tl.float32)
    dQ_acc1 = tl.zeros((BLOCK_S, BLOCK_D), tl.float32)
    
    # Iterate over key blocks causally preceding or overlapping with query block i
    for j in range(0, i + 1):
        K_j0 = k_desc.load([b_h_idx, j * BLOCK_S, 0])
        K_j1 = k_desc.load([b_h_idx, j * BLOCK_S, BLOCK_D])
        V_j0 = v_desc.load([b_h_idx, j * BLOCK_S, 0])
        V_j1 = v_desc.load([b_h_idx, j * BLOCK_S, BLOCK_D])
        
        s = tl.dot(Q_i0, K_j0.T) + tl.dot(Q_i1, K_j1.T)
        
        q_idx_2d = q_idx_base[:, None]
        k_idx_base = j * BLOCK_S + tl.arange(0, BLOCK_S)
        k_idx_2d = k_idx_base[None, :]
        causal_mask = (q_idx_2d >= k_idx_2d) & (q_idx_2d < S) & (k_idx_2d < S)
        
        p = tl.exp(s * scale - L_i[:, None])
        p = tl.where(causal_mask, p, 0.0)
        
        dp = tl.dot(dO_i0, V_j0.T) + tl.dot(dO_i1, V_j1.T)
        
        ds = p * (dp - D[:, None]) * scale
        
        dQ_acc0 = tl.dot(ds, K_j0.to(tl.float32), dQ_acc0)
        dQ_acc1 = tl.dot(ds, K_j1.to(tl.float32), dQ_acc1)
    
    # Flush accumulated gradient
    dq_desc.store([b_h_idx, i * BLOCK_S, 0], dQ_acc0.to(tl.bfloat16))
    dq_desc.store([b_h_idx, i * BLOCK_S, BLOCK_D], dQ_acc1.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute causal multi-head attention backward pass."""
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    scale = 1.0 / math.sqrt(float(d))
    
    BLOCK_S = 64
    BLOCK_D = 64
    
    Q_3d = Q.view(B * H, S, d)
    K_3d = K.view(B * H, S, d)
    V_3d = V.view(B * H, S, d)
    O_3d = O.view(B * H, S, d)
    dO_3d = dO.view(B * H, S, d)
    dQ_3d = dQ.view(B * H, S, d)
    dK_3d = dK.view(B * H, S, d)
    dV_3d = dV.view(B * H, S, d)
    
    q_desc = TensorDescriptor.from_tensor(Q_3d, [BLOCK_S, BLOCK_D])
    k_desc = TensorDescriptor.from_tensor(K_3d, [BLOCK_S, BLOCK_D])
    v_desc = TensorDescriptor.from_tensor(V_3d, [BLOCK_S, BLOCK_D])
    o_desc = TensorDescriptor.from_tensor(O_3d, [BLOCK_S, BLOCK_D])
    do_desc = TensorDescriptor.from_tensor(dO_3d, [BLOCK_S, BLOCK_D])
    dq_desc = TensorDescriptor.from_tensor(dQ_3d, [BLOCK_S, BLOCK_D])
    dk_desc = TensorDescriptor.from_tensor(dK_3d, [BLOCK_S, BLOCK_D])
    dv_desc = TensorDescriptor.from_tensor(dV_3d, [BLOCK_S, BLOCK_D])
    
    stride_l = S
    
    grid = (triton.cdiv(S, BLOCK_S), H, B)
    
    dv_dk_kernel[grid](
        q_desc, k_desc, v_desc, o_desc, do_desc,
        L, dv_desc, dk_desc,
        S, H, B,
        stride_l,
        scale,
        BLOCK_S=BLOCK_S,
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )
    
    dq_kernel[grid](
        q_desc, k_desc, v_desc, o_desc, do_desc,
        L, dq_desc,
        S, H, B,
        stride_l,
        scale,
        BLOCK_S=BLOCK_S,
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )