import math
import torch
import triton
import triton.language as tl


@triton.jit
def bwd_dkdv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dK_ptr, dV_ptr,
    B, H, S, d,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    scale = 1.0 / math.sqrt(d)
    bh = tl.program_id(0)
    j = tl.program_id(1)
    
    Q_desc = tl.make_tensor_descriptor(Q_ptr, shape=[B*H*S, d], strides=[d, 1], block_shape=[BLOCK_M, 64], padding_option="zero")
    K_desc = tl.make_tensor_descriptor(K_ptr, shape=[B*H*S, d], strides=[d, 1], block_shape=[BLOCK_N, 64], padding_option="zero")
    V_desc = tl.make_tensor_descriptor(V_ptr, shape=[B*H*S, d], strides=[d, 1], block_shape=[BLOCK_N, 64], padding_option="zero")
    O_desc = tl.make_tensor_descriptor(O_ptr, shape=[B*H*S, d], strides=[d, 1], block_shape=[BLOCK_M, 64], padding_option="zero")
    dO_desc = tl.make_tensor_descriptor(dO_ptr, shape=[B*H*S, d], strides=[d, 1], block_shape=[BLOCK_M, 64], padding_option="zero")
    dK_desc = tl.make_tensor_descriptor(dK_ptr, shape=[B*H*S, d], strides=[d, 1], block_shape=[BLOCK_N, 64], padding_option="zero")
    dV_desc = tl.make_tensor_descriptor(dV_ptr, shape=[B*H*S, d], strides=[d, 1], block_shape=[BLOCK_N, 64], padding_option="zero")
    
    K0 = K_desc.load([bh * S + j * BLOCK_N, 0])
    K1 = K_desc.load([bh * S + j * BLOCK_N, 64])
    V0 = V_desc.load([bh * S + j * BLOCK_N, 0])
    V1 = V_desc.load([bh * S + j * BLOCK_N, 64])
    
    dK0 = tl.zeros((BLOCK_N, 64), tl.float32)
    dK1 = tl.zeros((BLOCK_N, 64), tl.float32)
    dV0 = tl.zeros((BLOCK_N, 64), tl.float32)
    dV1 = tl.zeros((BLOCK_N, 64), tl.float32)
    
    row_idx = j * BLOCK_N + tl.arange(0, BLOCK_N)
    col_idx = tl.arange(0, BLOCK_M)
    
    for i in range(j, (S + BLOCK_M - 1) // BLOCK_M):
        Q0 = Q_desc.load([bh * S + i * BLOCK_M, 0])
        Q1 = Q_desc.load([bh * S + i * BLOCK_M, 64])
        O0 = O_desc.load([bh * S + i * BLOCK_M, 0])
        O1 = O_desc.load([bh * S + i * BLOCK_M, 64])
        dO0 = dO_desc.load([bh * S + i * BLOCK_M, 0])
        dO1 = dO_desc.load([bh * S + i * BLOCK_M, 64])
        
        D = tl.sum(dO0.to(tl.float32) * O0.to(tl.float32), axis=1) + tl.sum(dO1.to(tl.float32) * O1.to(tl.float32), axis=1)
        
        valid_rows = (i * BLOCK_M + tl.arange(0, BLOCK_M)) < S
        L_i = tl.load(L_ptr + bh * S + i * BLOCK_M + tl.arange(0, BLOCK_M), mask=valid_rows, other=0.0)
        
        s = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        s = tl.dot(Q0.to(tl.float32), K0.to(tl.float32).T, acc=s)
        s = tl.dot(Q1.to(tl.float32), K1.to(tl.float32).T, acc=s)
        
        dp = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        dp = tl.dot(dO0.to(tl.float32), V0.to(tl.float32).T, acc=dp)
        dp = tl.dot(dO1.to(tl.float32), V1.to(tl.float32).T, acc=dp)
        
        p = tl.exp(s * scale - L_i[:, None])
        ds = p * (dp - D[:, None]) * scale
        
        ds = ds * ((i * BLOCK_M + col_idx[:, None]) >= (j * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]))
        
        dK0 = tl.dot(ds.T, Q0.to(tl.float32), acc=dK0)
        dK1 = tl.dot(ds.T, Q1.to(tl.float32), acc=dK1)
        dV0 = tl.dot(p.T, dO0.to(tl.float32), acc=dV0)
        dV1 = tl.dot(p.T, dO1.to(tl.float32), acc=dV1)
        
    dK_desc.store([bh * S + j * BLOCK_N, 0], dK0.to(tl.bfloat16))
    dK_desc.store([bh * S + j * BLOCK_N, 64], dK1.to(tl.bfloat16))
    dV_desc.store([bh * S + j * BLOCK_N, 0], dV0.to(tl.bfloat16))
    dV_desc.store([bh * S + j * BLOCK_N, 64], dV1.to(tl.bfloat16))


@triton.jit
def bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dQ_ptr,
    B, H, S, d,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    scale = 1.0 / math.sqrt(d)
    bh = tl.program_id(0)
    i = tl.program_id(1)
    
    Q_desc = tl.make_tensor_descriptor(Q_ptr, shape=[B*H*S, d], strides=[d, 1], block_shape=[BLOCK_M, 64], padding_option="zero")
    K_desc = tl.make_tensor_descriptor(K_ptr, shape=[B*H*S, d], strides=[d, 1], block_shape=[BLOCK_N, 64], padding_option="zero")
    V_desc = tl.make_tensor_descriptor(V_ptr, shape=[B*H*S, d], strides=[d, 1], block_shape=[BLOCK_N, 64], padding_option="zero")
    O_desc = tl.make_tensor_descriptor(O_ptr, shape=[B*H*S, d], strides=[d, 1], block_shape=[BLOCK_M, 64], padding_option="zero")
    dO_desc = tl.make_tensor_descriptor(dO_ptr, shape=[B*H*S, d], strides=[d, 1], block_shape=[BLOCK_M, 64], padding_option="zero")
    dQ_desc = tl.make_tensor_descriptor(dQ_ptr, shape=[B*H*S, d], strides=[d, 1], block_shape=[BLOCK_M, 64], padding_option="zero")
    
    Q0 = Q_desc.load([bh * S + i * BLOCK_M, 0])
    Q1 = Q_desc.load([bh * S + i * BLOCK_M, 64])
    O0 = O_desc.load([bh * S + i * BLOCK_M, 0])
    O1 = O_desc.load([bh * S + i * BLOCK_M, 64])
    dO0 = dO_desc.load([bh * S + i * BLOCK_M, 0])
    dO1 = dO_desc.load([bh * S + i * BLOCK_M, 64])
    
    D = tl.sum(dO0.to(tl.float32) * O0.to(tl.float32), axis=1) + tl.sum(dO1.to(tl.float32) * O1.to(tl.float32), axis=1)
    
    valid_rows = (i * BLOCK_M + tl.arange(0, BLOCK_M)) < S
    L_i = tl.load(L_ptr + bh * S + i * BLOCK_M + tl.arange(0, BLOCK_M), mask=valid_rows, other=0.0)
    
    dQ0 = tl.zeros((BLOCK_M, 64), tl.float32)
    dQ1 = tl.zeros((BLOCK_M, 64), tl.float32)
    
    row_idx = i * BLOCK_M + tl.arange(0, BLOCK_M)
    col_idx = tl.arange(0, BLOCK_N)
    
    for j in range(0, i + 1):
        K0 = K_desc.load([bh * S + j * BLOCK_N, 0])
        K1 = K_desc.load([bh * S + j * BLOCK_N, 64])
        V0 = V_desc.load([bh * S + j * BLOCK_N, 0])
        V1 = V_desc.load([bh * S + j * BLOCK_N, 64])
        
        s = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        s = tl.dot(Q0.to(tl.float32), K0.to(tl.float32).T, acc=s)
        s = tl.dot(Q1.to(tl.float32), K1.to(tl.float32).T, acc=s)
        
        dp = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        dp = tl.dot(dO0.to(tl.float32), V0.to(tl.float32).T, acc=dp)
        dp = tl.dot(dO1.to(tl.float32), V1.to(tl.float32).T, acc=dp)
        
        p = tl.exp(s * scale - L_i[:, None])
        ds = p * (dp - D[:, None]) * scale
        
        ds = ds * ((i * BLOCK_M + col_idx[:, None]) >= (j * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]))
        
        dQ0 = tl.dot(ds, K0.to(tl.float32), acc=dQ0)
        dQ1 = tl.dot(ds, K1.to(tl.float32), acc=dQ1)
        
    dQ_desc.store([bh * S + i * BLOCK_M, 0], dQ0.to(tl.bfloat16))
    dQ_desc.store([bh * S + i * BLOCK_M, 64], dQ1.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    BLOCK_SIZE = 128
    grid = (B * H, (S + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    bwd_dkdv_kernel[grid](Q, K, V, O, dO, L, dK, dV, B, H, S, d, BLOCK_M=BLOCK_SIZE, BLOCK_N=BLOCK_SIZE, num_warps=4, num_stages=2)
    bwd_dq_kernel[grid](Q, K, V, O, dO, L, dQ, B, H, S, d, BLOCK_M=BLOCK_SIZE, BLOCK_N=BLOCK_SIZE, num_warps=4, num_stages=2)