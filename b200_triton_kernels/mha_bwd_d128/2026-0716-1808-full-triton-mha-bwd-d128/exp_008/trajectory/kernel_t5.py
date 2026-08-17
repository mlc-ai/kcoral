import math
import torch
import triton
import triton.language as tl


@triton.jit
def make_swizzled_desc(ptr, shape, strides):
    desc = tl.make_tensor_descriptor(
        ptr, shape=shape, strides=strides,
        block_shape=[BLOCK_Q, BLOCK_D // 2],
        padding_option="zero",
        elem_ty=tl.bfloat16,
    )
    return desc

@triton.jit
def make_L_desc(ptr, shape):
    desc = tl.make_tensor_descriptor(
        ptr, shape=shape, strides=[shape[1], 1],
        block_shape=[1, 1],
        padding_option="zero",
        elem_ty=tl.float32,
    )
    return desc

@triton.jit
def load_swizzled(desc, smem, offset):
    bar = desc.tile.create_load(smem, offset)
    desc.tile.do_swizzle = True
    return bar

@triton.jit
def store_swizzled(ptr, tensor, mask, desc):
    desc.tile.do_swizzle = True
    desc.tile.create_store(ptr, tensor, mask)

@triton.jit
def dot(a, b, acc):
    return tl.dot(a, b, acc)

@triton.jit
def split(tensor, num_splits, dim):
    shape = list(tensor.shape)
    shape[dim] = 1
    tensor = tensor.reshape(shape, can_reorder=True)
    split_tensors = []
    for i in range(num_splits):
        s = list(shape)
        s[dim] = s[dim] * (tensor.shape[dim] // num_splits)
        split_tensors.append(tl.split(tensor, num_splits, dim)[i])
    return tuple(split_tensors) if num_splits > 1 else tensor[0]


BLOCK_Q = 64
BLOCK_D = 128


@triton.jit
def dKdV_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dK_ptr, dV_ptr,
    S,
    SCALE: tl.constexpr,
):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    bh = tl.program_id(2)
    
    l_desc = make_L_desc(L_ptr, [192, S])
    
    q_desc = make_swizzled_desc(Q_ptr, [192, S, 2, 64], [S * 128, 128, 64, 1])
    k_desc = make_swizzled_desc(K_ptr, [192, S, 2, 64], [S * 128, 128, 64, 1])
    v_desc = make_swizzled_desc(V_ptr, [192, S, 2, 64], [S * 128, 128, 64, 1])
    o_desc = make_swizzled_desc(O_ptr, [192, S, 2, 64], [S * 128, 128, 64, 1])
    do_desc = make_swizzled_desc(dO_ptr, [192, S, 2, 64], [S * 128, 128, 64, 1])
    dK_desc = make_swizzled_desc(dK_ptr, [192, S, 2, 64], [S * 128, 128, 64, 1])
    dV_desc = make_swizzled_desc(dV_ptr, [192, S, 2, 64], [S * 128, 128, 64, 1])
    
    bar_init = tl.named_barrier_sync_fn("init")
    bar_pipe = [
        tl.named_barrier_sync_fn(f"pipe_{i}_0"),
        tl.named_barrier_sync_fn(f"pipe_{i}_1")
    ]
    
    q_smem_buf = [
        tl.allocate_shared_memory(tl.int8, 16384),
        tl.allocate_shared_memory(tl.int8, 16384),
    ]
    k_smem_buf = [
        tl.allocate_shared_memory(tl.int8, 16384),
        tl.allocate_shared_memory(tl.int8, 16384),
    ]
    v_smem_buf = [
        tl.allocate_shared_memory(tl.int8, 16384),
        tl.allocate_shared_memory(tl.int8, 16384),
    ]
    o_smem_buf = [
        tl.allocate_shared_memory(tl.int8, 16384),
        tl.allocate_shared_memory(tl.int8, 16384),
    ]
    do_smem_buf = [
        tl.allocate_shared_memory(tl.int8, 16384),
        tl.allocate_shared_memory(tl.int8, 16384),
    ]
    
    for stage in range(2):
        load_swizzled(k_desc, k_smem_buf[stage], [bh, pid_j * BLOCK_Q, 0, 0])
        load_swizzled(v_desc, v_smem_buf[stage], [bh, pid_j * BLOCK_Q, 0, 0])
        
        buffer_idx = stage
        j_start_val = buffer_idx * 64
        if j_start_val < S:
            load_swizzled(q_desc, q_smem_buf[stage], [bh, j_start_val, 0, 0])
            load_swizzled(o_desc, o_smem_buf[stage], [bh, j_start_val, 0, 0])
            load_swizzled(do_desc, do_smem_buf[stage], [bh, j_start_val, 0, 0])
            
            bar_pipe[stage] = bar_init
            
    tl.wait(bar_init)
    
    dK_0_acc = tl.zeros((BLOCK_Q, BLOCK_Q), tl.float32)
    dK_1_acc = tl.zeros((BLOCK_Q, BLOCK_Q), tl.float32)
    dV_0_acc = tl.zeros((BLOCK_Q, BLOCK_Q), tl.float32)
    dV_1_acc = tl.zeros((BLOCK_Q, BLOCK_Q), tl.float32)
    
    loop_cnt = tl.cdiv(S, BLOCK_Q) - 1
    
    k0, k1 = split(k_smem_buf[0], 2, dim=1)
    v0, v1 = split(v_smem_buf[0], 2, dim=1)
    
    j_start = 0
    for i in range(0, loop_cnt, 2):
        j0 = j_start + 0
        j1 = j_start + 1
        
        bar0 = bar_pipe[0]
        bar1 = bar_pipe[1]
        
        if j1 < S:
            load_swizzled(q_desc, q_smem_buf[1], [bh, j1 * BLOCK_Q, 0, 0])
            load_swizzled(o_desc, o_smem_buf[1], [bh, j1 * BLOCK_Q, 0, 0])
            load_swizzled(do_desc, do_smem_buf[1], [bh, j1 * BLOCK_Q, 0, 0])
            bar_pipe[1] = tl.named_barrier_sync_fn(f"pipe_{i}_{1}")
        
        load_swizzled(q_desc, q_smem_buf[0], [bh, j0 * BLOCK_Q, 0, 0])
        load_swizzled(o_desc, o_smem_buf[0], [bh, j0 * BLOCK_Q, 0, 0])
        load_swizzled(do_desc, do_smem_buf[0], [bh, j0 * BLOCK_Q, 0, 0])
        bar_pipe[0] = tl.named_barrier_sync_fn(f"pipe_{i}_{0}")
        
        j_start += 2
        
        tl.commit(bar0)
        tl.commit(bar1)
        tl.wait(bar0)
        tl.wait(bar1)
        
        q0_0, q0_1 = split(q_smem_buf[0], 2, dim=1)
        o0_0, o0_1 = split(o_smem_buf[0], 2, dim=1)
        do0_0, do0_1 = split(do_smem_buf[0], 2, dim=1)
        
        D0 = tl.sum(o0_0 * do0_0 + o0_1 * do0_1, axis=1)
        D0 = D0[:, None]
        
        S0_acc = tl.zeros((BLOCK_Q, BLOCK_Q), tl.float32)
        S0_acc = dot(q0_0, k0.T, S0_acc)
        S0_acc = dot(q0_1, k1.T, S0_acc)
        
        dP0_acc = tl.zeros((BLOCK_Q, BLOCK_Q), tl.float32)
        dP0_acc = dot(do0_0, v0.T, dP0_acc)
        dP0_acc = dot(do0_1, v1.T, dP0_acc)
        
        L0 = tl.load(l_desc, [bh + j0 * BLOCK_Q + tl.arange(0, BLOCK_Q)], mask=(j0 * BLOCK_Q + tl.arange(0, BLOCK_Q)) < S, other=float('inf'))
        
        P0 = tl.exp(S0_acc * SCALE - L0[:, None])
        dS0 = P0 * (dP0_acc - D0) * SCALE
        
        dS0_T = dS0.T
        P0_T = P0.T
        
        dK_0_acc = dot(dS0_T, q0_0, dK_0_acc)
        dK_1_acc = dot(dS0_T, q0_1, dK_1_acc)
        dV_0_acc = dot(P0_T, do0_0, dV_0_acc)
        dV_1_acc = dot(P0_T, do0_1, dV_1_acc)
        
        q1_0, q1_1 = split(q_smem_buf[1], 2, dim=1)
        o1_0, o1_1 = split(o_smem_buf[1], 2, dim=1)
        do1_0, do1_1 = split(do_smem_buf[1], 2, dim=1)
        
        D1 = tl.sum(o1_0 * do1_0 + o1_1 * do1_1, axis=1)
        D1 = D1[:, None]
        
        S1_acc = tl.zeros((BLOCK_Q, BLOCK_Q), tl.float32)
        S1_acc = dot(q1_0, k0.T, S1_acc)
        S1_acc = dot(q1_1, k1.T, S1_acc)
        
        dP1_acc = tl.zeros((BLOCK_Q, BLOCK_Q), tl.float32)
        dP1_acc = dot(do1_0, v0.T, dP1_acc)
        dP1_acc = dot(do1_1, v1.T, dP1_acc)
        
        L1 = tl.load(l_desc, [bh + j1 * BLOCK_Q + tl.arange(0, BLOCK_Q)], mask=(j1 * BLOCK_Q + tl.arange(0, BLOCK_Q)) < S, other=float('inf'))
        
        P1 = tl.exp(S1_acc * SCALE - L1[:, None])
        dS1 = P1 * (dP1_acc - D1) * SCALE
        
        dS1_T = dS1.T
        P1_T = P1.T
        
        dK_0_acc = dot(dS1_T, q1_0, dK_0_acc)
        dK_1_acc = dot(dS1_T, q1_1, dK_1_acc)
        dV_0_acc = dot(P1_T, do1_0, dV_0_acc)
        dV_1_acc = dot(P1_T, do1_1, dV_1_acc)
        
    if j_start < S:
        load_swizzled(q_desc, q_smem_buf[0], [bh, j_start * BLOCK_Q, 0, 0])
        load_swizzled(o_desc, o_smem_buf[0], [bh, j_start * BLOCK_Q, 0, 0])
        load_swizzled(do_desc, do_smem_buf[0], [bh, j_start * BLOCK_Q, 0, 0])
        
        bar0 = bar_pipe[0]
        tl.commit(bar0)
        tl.wait(bar0)
        
        q0_0, q0_1 = split(q_smem_buf[0], 2, dim=1)
        o0_0, o0_1 = split(o_smem_buf[0], 2, dim=1)
        do0_0, do0_1 = split(do_smem_buf[0], 2, dim=1)
        
        D0 = tl.sum(o0_0 * do0_0 + o0_1 * do0_1, axis=1)
        D0 = D0[:, None]
        
        S0_acc = tl.zeros((BLOCK_Q, BLOCK_Q), tl.float32)
        S0_acc = dot(q0_0, k0.T, S0_acc)
        S0_acc = dot(q0_1, k1.T, S0_acc)
        
        dP0_acc = tl.zeros((BLOCK_Q, BLOCK_Q), tl.float32)
        dP0_acc = dot(do0_0, v0.T, dP0_acc)
        dP0_acc = dot(do0_1, v1.T, dP0_acc)
        
        L0 = tl.load(l_desc, [bh + j_start * BLOCK_Q + tl.arange(0, BLOCK_Q)], mask=(j_start * BLOCK_Q + tl.arange(0, BLOCK_Q)) < S, other=float('inf'))
        
        P0 = tl.exp(S0_acc * SCALE - L0[:, None])
        dS0 = P0 * (dP0_acc - D0) * SCALE
        
        dS0_T = dS0.T
        P0_T = P0.T
        
        dK_0_acc = dot(dS0_T, q0_0, dK_0_acc)
        dK_1_acc = dot(dS0_T, q0_1, dK_1_acc)
        dV_0_acc = dot(P0_T, do0_0, dV_0_acc)
        dV_1_acc = dot(P0_T, do0_1, dV_1_acc)
        
    row_idx = pid_j * BLOCK_Q + tl.arange(0, BLOCK_Q)
    mask = (row_idx < S)[:, None]
    
    store_swizzled(dK_ptr + bh * S * 128 + pid_j * BLOCK_Q * 128 + 0, dK_0_acc, mask, dK_desc)
    store_swizzled(dK_ptr + bh * S * 128 + pid_j * BLOCK_Q * 128 + 64, dK_1_acc, mask, dK_desc)
    store_swizzled(dV_ptr + bh * S * 128 + pid_j * BLOCK_Q * 128 + 0, dV_0_acc, mask, dV_desc)
    store_swizzled(dV_ptr + bh * S * 128 + pid_j * BLOCK_Q * 128 + 64, dV_1_acc, mask, dV_desc)


@triton.jit
def dQ_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dQ_ptr,
    S,
    SCALE: tl.constexpr,
):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    bh = tl.program_id(2)
    
    l_desc = make_L_desc(L_ptr, [192, S])
    
    q_desc = make_swizzled_desc(Q_ptr, [192, S, 2, 64], [S * 128, 128, 64, 1])
    k_desc = make_swizzled_desc(K_ptr, [192, S, 2, 64], [S * 128, 128, 64, 1])
    v_desc = make_swizzled_desc(V_ptr, [192, S, 2, 64], [S * 128, 128, 64, 1])
    o_desc = make_swizzled_desc(O_ptr, [192, S, 2, 64], [S * 128, 128, 64, 1])
    do_desc = make_swizzled_desc(dO_ptr, [192, S, 2, 64], [S * 128, 128, 64, 1])
    dQ_desc = make_swizzled_desc(dQ_ptr, [192, S, 2, 64], [S * 128, 128, 64, 1])
    
    bar_init = tl.named_barrier_sync_fn("init")
    bar_pipe = [
        tl.named_barrier_sync_fn(f"pipe_{i}_0"),
        tl.named_barrier_sync_fn(f"pipe_{i}_1")
    ]
    
    q_smem_buf = [
        tl.allocate_shared_memory(tl.int8, 16384),
        tl.allocate_shared_memory(tl.int8, 16384),
    ]
    k_smem_buf = [
        tl.allocate_shared_memory(tl.int8, 16384),
        tl.allocate_shared_memory(tl.int8, 16384),
    ]
    v_smem_buf = [
        tl.allocate_shared_memory(tl.int8, 16384),
        tl.allocate_shared_memory(tl.int8, 16384),
    ]
    o_smem_buf = [
        tl.allocate_shared_memory(tl.int8, 16384),
        tl.allocate_shared_memory(tl.int8, 16384),
    ]
    do_smem_buf = [
        tl.allocate_shared_memory(tl.int8, 16384),
        tl.allocate_shared_memory(tl.int8, 16384),
    ]
    
    for stage in range(2):
        load_swizzled(q_desc, q_smem_buf[stage], [bh, pid_i * BLOCK_Q, 0, 0])
        load_swizzled(o_desc, o_smem_buf[stage], [bh, pid_i * BLOCK_Q, 0, 0])
        load_swizzled(do_desc, do_smem_buf[stage], [bh, pid_i * BLOCK_Q, 0, 0])
        
        buffer_idx = stage
        j_start_val = buffer_idx * 64
        if j_start_val < S:
            load_swizzled(k_desc, k_smem_buf[stage], [bh, j_start_val, 0, 0])
            load_swizzled(v_desc, v_smem_buf[stage], [bh, j_start_val, 0, 0])
            
            bar_pipe[stage] = bar_init
            
    tl.wait(bar_init)
    
    q0, q1 = split(q_smem_buf[0], 2, dim=1)
    o0, o1 = split(o_smem_buf[0], 2, dim=1)
    do0, do1 = split(do_smem_buf[0], 2, dim=1)
    
    D = tl.sum(o0 * do0 + o1 * do1, axis=1)
    D = D[:, None]
    
    L_i = tl.load(l_desc, [bh + pid_i * BLOCK_Q + tl.arange(0, BLOCK_Q)], mask=(pid_i * BLOCK_Q + tl.arange(0, BLOCK_Q)) < S, other=float('inf'))
    
    dQ_0_acc = tl.zeros((BLOCK_Q, BLOCK_Q), tl.float32)
    dQ_1_acc = tl.zeros((BLOCK_Q, BLOCK_Q), tl.float32)
    
    loop_cnt = tl.cdiv(S, BLOCK_Q) - 1
    
    j_start = 0
    for i in range(0, loop_cnt, 2):
        j0 = j_start + 0
        j1 = j_start + 1
        
        bar0 = bar_pipe[0]
        bar1 = bar_pipe[1]
        
        if j1 < S:
            load_swizzled(k_desc, k_smem_buf[1], [bh, j1 * BLOCK_Q, 0, 0])
            load_swizzled(v_desc, v_smem_buf[1], [bh, j1 * BLOCK_Q, 0, 0])
            bar_pipe[1] = tl.named_barrier_sync_fn(f"pipe_{i}_{1}")
            
        load_swizzled(k_desc, k_smem_buf[0], [bh, j0 * BLOCK_Q, 0, 0])
        load_swizzled(v_desc, v_smem_buf[0], [bh, j0 * BLOCK_Q, 0, 0])
        bar_pipe[0] = tl.named_barrier_sync_fn(f"pipe_{i}_{0}")
        
        j_start += 2
        
        tl.commit(bar0)
        tl.commit(bar1)
        tl.wait(bar0)
        tl.wait(bar1)
        
        k0_0, k0_1 = split(k_smem_buf[0], 2, dim=1)
        v0_0, v0_1 = split(v_smem_buf[0], 2, dim=1)
        
        S0_acc = tl.zeros((BLOCK_Q, BLOCK_Q), tl.float32)
        S0_acc = dot(q0, k0_0.T, S0_acc)
        S0_acc = dot(q1, k0_1.T, S0_acc)
        
        dP0_acc = tl.zeros((BLOCK_Q, BLOCK_Q), tl.float32)
        dP0_acc = dot(do0, v0_0.T, dP0_acc)
        dP0_acc = dot(do1, v0_1.T, dP0_acc)
        
        P0 = tl.exp(S0_acc * SCALE - L_i[:, None])
        dS0 = P0 * (dP0_acc - D) * SCALE
        
        dQ_0_acc = dot(dS0, k0_0, dQ_0_acc)
        dQ_1_acc = dot(dS0, k0_1, dQ_1_acc)
        
        k1_0, k1_1 = split(k_smem_buf[1], 2, dim=1)
        v1_0, v1_1 = split(v_smem_buf[1], 2, dim=1)
        
        S1_acc = tl.zeros((BLOCK_Q, BLOCK_Q), tl.float32)
        S1_acc = dot(q0, k1_0.T, S1_acc)
        S1_acc = dot(q1, k1_1.T, S1_acc)
        
        dP1_acc = tl.zeros((BLOCK_Q, BLOCK_Q), tl.float32)
        dP1_acc = dot(do0, v1_0.T, dP1_acc)
        dP1_acc = dot(do1, v1_1.T, dP1_acc)
        
        P1 = tl.exp(S1_acc * SCALE - L_i[:, None])
        dS1 = P1 * (dP1_acc - D) * SCALE
        
        dQ_0_acc = dot(dS1, k1_0, dQ_0_acc)
        dQ_1_acc = dot(dS1, k1_1, dQ_1_acc)
        
    if j_start < S:
        load_swizzled(k_desc, k_smem_buf[0], [bh, j_start * BLOCK_Q, 0, 0])
        load_swizzled(v_desc, v_smem_buf[0], [bh, j_start * BLOCK_Q, 0, 0])
        
        bar0 = bar_pipe[0]
        tl.commit(bar0)
        tl.wait(bar0)
        
        k0_0, k0_1 = split(k_smem_buf[0], 2, dim=1)
        v0_0, v0_1 = split(v_smem_buf[0], 2, dim=1)
        
        S0_acc = tl.zeros((BLOCK_Q, BLOCK_Q), tl.float32)
        S0_acc = dot(q0, k0_0.T, S0_acc)
        S0_acc = dot(q1, k0_1.T, S0_acc)
        
        dP0_acc = tl.zeros((BLOCK_Q, BLOCK_Q), tl.float32)
        dP0_acc = dot(do0, v0_0.T, dP0_acc)
        dP0_acc = dot(do1, v0_1.T, dP0_acc)
        
        P0 = tl.exp(S0_acc * SCALE - L_i[:, None])
        dS0 = P0 * (dP0_acc - D) * SCALE
        
        dQ_0_acc = dot(dS0, k0_0, dQ_0_acc)
        dQ_1_acc = dot(dS0, k0_1, dQ_1_acc)
        
    row_idx = pid_i * BLOCK_Q + tl.arange(0, BLOCK_Q)
    mask = (row_idx < S)[:, None]
    
    store_swizzled(dQ_ptr + bh * S * 128 + pid_i * BLOCK_Q * 128 + 0, dQ_0_acc, mask, dQ_desc)
    store_swizzled(dQ_ptr + bh * S * 128 + pid_i * BLOCK_Q * 128 + 64, dQ_1_acc, mask, dQ_desc)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute backward pass for Multi-Head Attention."""
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    scale = 1.0 / math.sqrt(d)
    
    Q_flat = Q.view(-1, S, d)
    K_flat = K.view(-1, S, d)
    V_flat = V.view(-1, S, d)
    O_flat = O.view(-1, S, d)
    dO_flat = dO.view(-1, S, d)
    dQ_flat = dQ.view(-1, S, d)
    dK_flat = dK.view(-1, S, d)
    dV_flat = dV.view(-1, S, d)
    
    num_blocks_S = triton.cdiv(S, 64)
    num_blocks_d = triton.cdiv(128, 64)
    grid = (num_blocks_S, num_blocks_d, B * H)
    
    dKdV_kernel[grid](
        Q_flat.data_ptr(), K_flat.data_ptr(), V_flat.data_ptr(), O_flat.data_ptr(), 
        dO_flat.data_ptr(), L.data_ptr(),
        dK_flat.data_ptr(), dV_flat.data_ptr(),
        S,
        SCALE=scale,
        num_warps=4, num_stages=2
    )
    
    dQ_kernel[grid](
        Q_flat.data_ptr(), K_flat.data_ptr(), V_flat.data_ptr(), O_flat.data_ptr(), 
        dO_flat.data_ptr(), L.data_ptr(),
        dQ_flat.data_ptr(),
        S,
        SCALE=scale,
        num_warps=4, num_stages=2
    )