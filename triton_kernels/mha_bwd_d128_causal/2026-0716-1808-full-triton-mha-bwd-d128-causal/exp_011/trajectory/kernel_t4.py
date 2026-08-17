import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

NUM_SMS = 132
BLOCK = 128

@triton.jit
def _bwd_dq_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, dQ_desc, L_ptr,
    S_len, scale, stride_l_bh,
):
    bh = tl.program_id(1)
    i = tl.program_id(0)
    i_rows = i * BLOCK + tl.arange(0, BLOCK)
    row_base = bh * S_len + i * BLOCK
    
    q = Q_desc.load([row_base, 0])
    do = dO_desc.load([row_base, 0])
    o = O_desc.load([row_base, 0])
    
    D = tl.sum(do * o, axis=1)
    D_exp = D[:, None]
    
    L_block = tl.load(L_ptr + bh * stride_l_bh + i_rows, mask=i_rows < S_len, other=-float('inf'))
    L_exp = L_block[:, None]
    
    dQ_acc = tl.zeros((BLOCK, BLOCK), tl.float32)
    
    j_row_base = bh * S_len + 0 * BLOCK
    k = K_desc.load([j_row_base, 0])
    v = V_desc.load([j_row_base, 0])
    
    for j in range(0, i + 1):
        j_rows = j * BLOCK + tl.arange(0, BLOCK)
        
        if j < i:
            next_j = j + 1
            next_j_row_base = bh * S_len + next_j * BLOCK
            k_next = K_desc.load([next_j_row_base, 0])
            v_next = V_desc.load([next_j_row_base, 0])
            
        S = tl.dot(q, k.T, input_precision="tf32")
        
        P = tl.exp(S * scale - L_exp)
        
        causal_mask = ((i_rows[:, None] >= j_rows[None, :]) & (i_rows[:, None] < S_len) & (j_rows[None, :] < S_len))
        P = P * causal_mask
        
        dP = tl.dot(do, v.T, input_precision="tf32")
        
        dS = P * (dP - D_exp) * scale
        
        dQ_acc += tl.dot(dS, k, input_precision="tf32")
        
        if j < i:
            k = k_next
            v = v_next
            
    dQ_desc.store([row_base, 0], dQ_acc.to(tl.bfloat16))


@triton.jit
def _bwd_dkv_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, dK_desc, dV_desc, L_ptr,
    S_len, scale, stride_l_bh,
):
    bh = tl.program_id(1)
    j = tl.program_id(0)
    j_rows = j * BLOCK + tl.arange(0, BLOCK)
    j_row_base = bh * S_len + j * BLOCK
    
    k = K_desc.load([j_row_base, 0])
    v = V_desc.load([j_row_base, 0])
    
    dK_acc = tl.zeros((BLOCK, BLOCK), tl.float32)
    dV_acc = tl.zeros((BLOCK, BLOCK), tl.float32)
    
    T_r = triton.cdiv(S_len, BLOCK)
    
    for i in range(j, T_r):
        i_rows = i * BLOCK + tl.arange(0, BLOCK)
        i_row_base = bh * S_len + i * BLOCK
        
        q = Q_desc.load([i_row_base, 0])
        do = dO_desc.load([i_row_base, 0])
        o = O_desc.load([i_row_base, 0])
        
        D = tl.sum(do * o, axis=1)
        D_exp = D[:, None]
        
        S = tl.dot(q, k.T, input_precision="tf32")
        
        L_block = tl.load(L_ptr + bh * stride_l_bh + i_rows, mask=i_rows < S_len, other=-float('inf'))
        L_exp = L_block[:, None]
        
        P = tl.exp(S * scale - L_exp)
        
        causal_mask = ((i_rows[:, None] >= j_rows[None, :]) & (i_rows[:, None] < S_len) & (j_rows[None, :] < S_len))
        P = P * causal_mask
        
        dP = tl.dot(do, v.T, input_precision="tf32")
        
        dS = P * (dP - D_exp) * scale
        
        dK_acc += tl.dot(dS.T, q, input_precision="tf32")
        
        dV_acc += tl.dot(P.T, do, input_precision="tf32")
        
    dK_desc.store([j_row_base, 0], dK_acc.to(tl.bfloat16))
    dV_desc.store([j_row_base, 0], dV_acc.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute backward attention gradients with destination-passing outputs."""
    B, H, S_len, d = Q.shape
    device = Q.device
    torch.cuda.set_device(device)
    
    scale = 1.0 / (d ** 0.5)
    
    Q_2d = Q.view(B * H * S_len, d)
    K_2d = K.view(B * H * S_len, d)
    V_2d = V.view(B * H * S_len, d)
    O_2d = O.view(B * H * S_len, d)
    dO_2d = dO.view(B * H * S_len, d)
    dQ_2d = dQ.view(B * H * S_len, d)
    dK_2d = dK.view(B * H * S_len, d)
    dV_2d = dV.view(B * H * S_len, d)
    
    Q_desc = TensorDescriptor.from_tensor(Q_2d, [BLOCK, d])
    K_desc = TensorDescriptor.from_tensor(K_2d, [BLOCK, d])
    V_desc = TensorDescriptor.from_tensor(V_2d, [BLOCK, d])
    O_desc = TensorDescriptor.from_tensor(O_2d, [BLOCK, d])
    dO_desc = TensorDescriptor.from_tensor(dO_2d, [BLOCK, d])
    dQ_desc = TensorDescriptor.from_tensor(dQ_2d, [BLOCK, d])
    dK_desc = TensorDescriptor.from_tensor(dK_2d, [BLOCK, d])
    dV_desc = TensorDescriptor.from_tensor(dV_2d, [BLOCK, d])
    
    stride_l_bh = S_len
    
    T_r = triton.cdiv(S_len, BLOCK)
    T_c = triton.cdiv(S_len, BLOCK)
    
    grid_dq = (min(NUM_SMS, T_r), B * H)
    _bwd_dq_kernel[grid_dq](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, dQ_desc, L,
        S_len, scale, stride_l_bh,
        num_warps=4, num_stages=2
    )
    
    grid_dkv = (min(NUM_SMS, T_c), B * H)
    _bwd_dkv_kernel[grid_dkv](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, dK_desc, dV_desc, L,
        S_len, scale, stride_l_bh,
        num_warps=4, num_stages=2
    )