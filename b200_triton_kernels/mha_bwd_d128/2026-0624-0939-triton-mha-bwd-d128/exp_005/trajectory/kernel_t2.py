import torch
import triton
import triton.language as tl


@triton.jit
def _bwd_dKdV_kernel(
    Q0_base_ptr, Q1_base_ptr, K0_base_ptr, K1_base_ptr, V0_base_ptr, V1_base_ptr,
    dO0_base_ptr, dO1_base_ptr, O0_base_ptr, O1_base_ptr,
    L_ptr,
    dK0_base_ptr, dK1_base_ptr, dV0_base_ptr, dV1_base_ptr,
    S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    j = tl.program_id(0)
    batch_h = tl.program_id(1)
    
    if j * BLOCK_N >= S:
        return
    
    q_base = Q0_base_ptr + batch_h * S * 128
    k_base = K0_base_ptr + batch_h * S * 128
    v_base = V0_base_ptr + batch_h * S * 128
    do_base = dO0_base_ptr + batch_h * S * 128
    o_base = O0_base_ptr + batch_h * S * 128
    dk_base = dK0_base_ptr + batch_h * S * 128
    dv_base = dV0_base_ptr + batch_h * S * 128
    
    Q0_desc = tl.make_tensor_descriptor(q_base + 0, [S, 64], [128, 1], [BLOCK_M, 64], "zero")
    Q1_desc = tl.make_tensor_descriptor(q_base + 64, [S, 64], [128, 1], [BLOCK_M, 64], "zero")
    K0_desc = tl.make_tensor_descriptor(k_base + 0, [S, 64], [128, 1], [BLOCK_N, 64], "zero")
    K1_desc = tl.make_tensor_descriptor(k_base + 64, [S, 64], [128, 1], [BLOCK_N, 64], "zero")
    V0_desc = tl.make_tensor_descriptor(v_base + 0, [S, 64], [128, 1], [BLOCK_N, 64], "zero")
    V1_desc = tl.make_tensor_descriptor(v_base + 64, [S, 64], [128, 1], [BLOCK_N, 64], "zero")
    dO0_desc = tl.make_tensor_descriptor(do_base + 0, [S, 64], [128, 1], [BLOCK_M, 64], "zero")
    dO1_desc = tl.make_tensor_descriptor(do_base + 64, [S, 64], [128, 1], [BLOCK_M, 64], "zero")
    O0_desc = tl.make_tensor_descriptor(o_base + 0, [S, 64], [128, 1], [BLOCK_M, 64], "zero")
    O1_desc = tl.make_tensor_descriptor(o_base + 64, [S, 64], [128, 1], [BLOCK_M, 64], "zero")
    dK0_desc = tl.make_tensor_descriptor(dk_base + 0, [S, 64], [128, 1], [BLOCK_N, 64], "zero")
    dK1_desc = tl.make_tensor_descriptor(dk_base + 64, [S, 64], [128, 1], [BLOCK_N, 64], "zero")
    dV0_desc = tl.make_tensor_descriptor(dv_base + 0, [S, 64], [128, 1], [BLOCK_N, 64], "zero")
    dV1_desc = tl.make_tensor_descriptor(dv_base + 64, [S, 64], [128, 1], [BLOCK_N, 64], "zero")

    K0 = K0_desc.load([j * BLOCK_N, 0])
    K1 = K1_desc.load([j * BLOCK_N, 0])
    V0 = V0_desc.load([j * BLOCK_N, 0])
    V1 = V1_desc.load([j * BLOCK_N, 0])
    
    dK0_acc = tl.zeros((BLOCK_N, 64), tl.float32)
    dK1_acc = tl.zeros((BLOCK_N, 64), tl.float32)
    dV0_acc = tl.zeros((BLOCK_N, 64), tl.float32)
    dV1_acc = tl.zeros((BLOCK_N, 64), tl.float32)
    
    num_blocks = (S + BLOCK_M - 1) // BLOCK_M
    
    cols = tl.arange(0, BLOCK_N)
    valid_c = (j * BLOCK_N + cols) < S
    
    for i in range(num_blocks):
        Q0 = Q0_desc.load([i * BLOCK_M, 0])
        Q1 = Q1_desc.load([i * BLOCK_M, 0])
        dO0 = dO0_desc.load([i * BLOCK_M, 0])
        dO1 = dO1_desc.load([i * BLOCK_M, 0])
        O0 = O0_desc.load([i * BLOCK_M, 0])
        O1 = O1_desc.load([i * BLOCK_M, 0])
        
        rows = tl.arange(0, BLOCK_M)
        row_idx = i * BLOCK_M + rows
        
        L_idx = batch_h * S + row_idx
        L_mask = row_idx < S
        L_val = tl.load(L_ptr + L_idx, mask=L_mask, other=0.0)
        
        valid_r = row_idx < S
        
        dO0_f32 = dO0.to(tl.float32)
        dO1_f32 = dO1.to(tl.float32)
        O0_f32 = O0.to(tl.float32)
        O1_f32 = O1.to(tl.float32)
        D_val = tl.sum(dO0_f32 * O0_f32 + dO1_f32 * O1_f32, axis=1)
        
        S_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        S_acc = tl.dot(Q0, K0.T, S_acc)
        S_acc = tl.dot(Q1, K1.T, S_acc)
        
        S_acc = tl.where(valid_c[None, :], S_acc, -float('inf'))
        
        P = tl.exp(S_acc * scale - L_val[:, None])
        P = P * valid_c[None, :]
        P = P * valid_r[:, None]
        
        dP_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        dP_acc = tl.dot(dO0, V0.T, dP_acc)
        dP_acc = tl.dot(dO1, V1.T, dP_acc)
        
        dS = P * (dP_acc - D_val[:, None]) * scale
        dS = dS * valid_c[None, :]
        dS = dS * valid_r[:, None]
        
        dS_bf16 = dS.to(tl.bfloat16)
        P_bf16 = P.to(tl.bfloat16)
        
        dV0_acc = tl.dot(P_bf16.T, dO0, dV0_acc)
        dV1_acc = tl.dot(P_bf16.T, dO1, dV1_acc)
        
        dK0_acc = tl.dot(dS_bf16.T, Q0, dK0_acc)
        dK1_acc = tl.dot(dS_bf16.T, Q1, dK1_acc)
    
    dK0_out = dK0_acc.to(tl.bfloat16)
    dK0_desc.store([j * BLOCK_N, 0], dK0_out)
    dK1_out = dK1_acc.to(tl.bfloat16)
    dK1_desc.store([j * BLOCK_N, 0], dK1_out)
    
    dV0_out = dV0_acc.to(tl.bfloat16)
    dV0_desc.store([j * BLOCK_N, 0], dV0_out)
    dV1_out = dV1_acc.to(tl.bfloat16)
    dV1_desc.store([j * BLOCK_N, 0], dV1_out)


@triton.jit
def _bwd_dQ_kernel(
    Q0_base_ptr, Q1_base_ptr, K0_base_ptr, K1_base_ptr, V0_base_ptr, V1_base_ptr,
    dO0_base_ptr, dO1_base_ptr, O0_base_ptr, O1_base_ptr,
    L_ptr,
    dQ0_base_ptr, dQ1_base_ptr,
    S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    i = tl.program_id(0)
    batch_h = tl.program_id(1)
    
    if i * BLOCK_M >= S:
        return
    
    q_base = Q0_base_ptr + batch_h * S * 128
    k_base = K0_base_ptr + batch_h * S * 128
    v_base = V0_base_ptr + batch_h * S * 128
    do_base = dO0_base_ptr + batch_h * S * 128
    o_base = O0_base_ptr + batch_h * S * 128
    dq_base = dQ0_base_ptr + batch_h * S * 128
    
    Q0_desc = tl.make_tensor_descriptor(q_base + 0, [S, 64], [128, 1], [BLOCK_M, 64], "zero")
    Q1_desc = tl.make_tensor_descriptor(q_base + 64, [S, 64], [128, 1], [BLOCK_M, 64], "zero")
    K0_desc = tl.make_tensor_descriptor(k_base + 0, [S, 64], [128, 1], [BLOCK_N, 64], "zero")
    K1_desc = tl.make_tensor_descriptor(k_base + 64, [S, 64], [128, 1], [BLOCK_N, 64], "zero")
    V0_desc = tl.make_tensor_descriptor(v_base + 0, [S, 64], [128, 1], [BLOCK_N, 64], "zero")
    V1_desc = tl.make_tensor_descriptor(v_base + 64, [S, 64], [128, 1], [BLOCK_N, 64], "zero")
    dO0_desc = tl.make_tensor_descriptor(do_base + 0, [S, 64], [128, 1], [BLOCK_M, 64], "zero")
    dO1_desc = tl.make_tensor_descriptor(do_base + 64, [S, 64], [128, 1], [BLOCK_M, 64], "zero")
    O0_desc = tl.make_tensor_descriptor(o_base + 0, [S, 64], [128, 1], [BLOCK_M, 64], "zero")
    O1_desc = tl.make_tensor_descriptor(o_base + 64, [S, 64], [128, 1], [BLOCK_M, 64], "zero")
    dQ0_desc = tl.make_tensor_descriptor(dq_base + 0, [S, 64], [128, 1], [BLOCK_M, 64], "zero")
    dQ1_desc = tl.make_tensor_descriptor(dq_base + 64, [S, 64], [128, 1], [BLOCK_M, 64], "zero")

    Q0 = Q0_desc.load([i * BLOCK_M, 0])
    Q1 = Q1_desc.load([i * BLOCK_M, 0])
    dO0 = dO0_desc.load([i * BLOCK_M, 0])
    dO1 = dO1_desc.load([i * BLOCK_M, 0])
    O0 = O0_desc.load([i * BLOCK_M, 0])
    O1 = O1_desc.load([i * BLOCK_M, 0])
    
    rows = tl.arange(0, BLOCK_M)
    row_idx = i * BLOCK_M + rows
    
    L_idx = batch_h * S + row_idx
    L_mask = row_idx < S
    L_val = tl.load(L_ptr + L_idx, mask=L_mask, other=0.0)
    
    dO0_f32 = dO0.to(tl.float32)
    dO1_f32 = dO1.to(tl.float32)
    O0_f32 = O0.to(tl.float32)
    O1_f32 = O1.to(tl.float32)
    D_val = tl.sum(dO0_f32 * O0_f32 + dO1_f32 * O1_f32, axis=1)
    
    dQ0_acc = tl.zeros((BLOCK_M, 64), tl.float32)
    dQ1_acc = tl.zeros((BLOCK_M, 64), tl.float32)
    
    num_blocks = (S + BLOCK_N - 1) // BLOCK_N
    
    for j in range(num_blocks):
        K0 = K0_desc.load([j * BLOCK_N, 0])
        K1 = K1_desc.load([j * BLOCK_N, 0])
        V0 = V0_desc.load([j * BLOCK_N, 0])
        V1 = V1_desc.load([j * BLOCK_N, 0])
        
        cols = tl.arange(0, BLOCK_N)
        valid_c = (j * BLOCK_N + cols) < S
        
        S_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        S_acc = tl.dot(Q0, K0.T, S_acc)
        S_acc = tl.dot(Q1, K1.T, S_acc)
        
        S_acc = tl.where(valid_c[None, :], S_acc, -float('inf'))
        
        P = tl.exp(S_acc * scale - L_val[:, None])
        P = P * valid_c[None, :]
        
        dP_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        dP_acc = tl.dot(dO0, V0.T, dP_acc)
        dP_acc = tl.dot(dO1, V1.T, dP_acc)
        
        dS = P * (dP_acc - D_val[:, None]) * scale
        dS = dS * valid_c[None, :]
        
        dS_bf16 = dS.to(tl.bfloat16)
        
        dQ0_acc = tl.dot(dS_bf16, K0, dQ0_acc)
        dQ1_acc = tl.dot(dS_bf16, K1, dQ1_acc)
    
    dQ0_out = dQ0_acc.to(tl.bfloat16)
    dQ0_desc.store([i * BLOCK_M, 0], dQ0_out)
    dQ1_out = dQ1_acc.to(tl.bfloat16)
    dQ1_desc.store([i * BLOCK_M, 0], dQ1_out)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    
    B_val, H_val, S_val, d_val = Q.shape
    assert d_val == 128
    
    B_total = B_val * H_val
    
    BLOCK_M = 64
    BLOCK_N = 64
    
    scale = 1.0 / (d_val ** 0.5)
    
    grid_bwd = (triton.cdiv(S_val, BLOCK_N), B_total)
    
    _bwd_dKdV_kernel[grid_bwd](
        Q.element(), K.element(), V.element(), dO.element(), O.element(),
        L.element(), dK.element(), dV.element(),
        S_val, scale, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=8, num_stages=2
    )
    
    _bwd_dQ_kernel[grid_bwd](
        Q.element(), K.element(), V.element(), dO.element(), O.element(),
        L.element(), dQ.element(),
        S_val, scale, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=8, num_stages=2
    )