import torch
import triton
import triton.language as tl


@triton.jit
def create_desc_2d(ptr, shape, strides, D):
    return tl.make_tensor_descriptor(
        ptr,
        shape=shape,
        strides=strides,
        block_shape=[64, D],
        padding_option="zero",
    )


@triton.jit
def load_strided(ptr, base_ptr, shape, strides, padding_option):
    S = shape[0]
    D = shape[1]
    desc = create_desc_2d(base_ptr, shape, strides, D)
    row_idx = shape[2] if len(shape) > 2 else 0
    col_idx = shape[3] if len(shape) > 3 else 0
    block_ptr = tl.make_block_ptr(
        base_ptr,
        shape=[S, D],
        strides=[strides[0], 1],
        offsets=[row_idx, col_idx],
        block_shape=[64, D],
        padding_option=padding_option,
        boundary_check=(1,),
    )
    return tl.load(block_ptr)


@triton.jit
def pad_kv(desc, kv_start, S):
    BLOCK_SIZE = 64
    remaining = S - kv_start
    if remaining >= BLOCK_SIZE:
        return desc.load([kv_start, 0])
    else:
        loaded = desc.load([kv_start, 0])
        row_idx = tl.arange(0, BLOCK_SIZE)
        loaded[row_idx, :] = tl.where(row_idx < remaining, loaded[row_idx, :], 0.0)
        return loaded


@triton.jit
def store_strided(ptr, base_ptr, acc, shape, strides):
    D = shape[1]
    desc = create_desc_2d(base_ptr, shape, strides, D)
    row_ptr = 0
    desc.store([row_ptr, 0], acc)


@triton.jit
def _attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S, scale, H: tl.constexpr, BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    q_start = pid * BLOCK_SIZE
    
    b_h = tl.program_id(1)
    b = b_h // H
    h = b_h % H
    
    row_ptr = (b_h * S + q_start) * D
    q_tile = load_strided(Q_ptr, Q_ptr + row_ptr, [BLOCK_SIZE, HEAD_DIM], [HEAD_DIM, 1], "zero")
    
    o_acc = tl.zeros((BLOCK_SIZE, HEAD_DIM), dtype=tl.float32)
    m_row = -tl.full((BLOCK_SIZE,), float("inf"), dtype=tl.float32)
    ell_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    num_kv_blocks = (S + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    for j in range(num_kv_blocks):
        kv_start = j * BLOCK_SIZE
        
        kv_row_ptr = (b_h * S + kv_start) * D
        
        key_tile = load_strided(K_ptr, K_ptr + kv_row_ptr, [BLOCK_SIZE, HEAD_DIM], [HEAD_DIM, 1], "zero")
        value_tile = load_strided(V_ptr, V_ptr + kv_row_ptr, [BLOCK_SIZE, HEAD_DIM], [HEAD_DIM, 1], "zero")
        
        acc_o = tl.dot(q_tile, key_tile.T) * scale
        
        kv_idx = kv_start + tl.arange(0, BLOCK_SIZE)
        mask = kv_idx < S
        acc_o = tl.where(mask, acc_o, -float("inf"))
        
        old_m = m_row
        new_m = tl.maximum(old_m, tl.max(acc_o, axis=1))
        
        o_acc *= tl.exp(old_m - new_m)
        
        p_cur = tl.exp(acc_o - new_m)
        ell_row = ell_row * tl.exp(old_m - new_m) + tl.sum(p_cur, axis=1)
        
        p_scaled = p_cur.to(tl.bfloat16)
        
        o_acc += tl.dot(p_scaled, value_tile)
        
        m_row = new_m
        
    inv_ell = 1.0 / ell_row
    out = (o_acc * inv_ell[:, None]).to(tl.bfloat16)
    
    base_ptr = O_ptr + (b_h * S + q_start) * D
    store_strided(O_ptr, base_ptr, out, [BLOCK_SIZE, HEAD_DIM], [HEAD_DIM, 1])
    
    q_seq_idx = tl.arange(0, BLOCK_SIZE)
    lse_ptr = LSE_ptr + (b * H + h) * S + q_start + q_seq_idx
    mask = q_seq_idx < BLOCK_SIZE
    tl.store(lse_ptr, m_row + tl.log(ell_row), mask=mask)


D = 128

def run(Q, K, V, O, LSE):
    """Compute Multi-Head Attention O and LSE into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    scale = 1.0 / (D ** 0.5)
    
    num_blocks = triton.cdiv(S, 64)
    grid = (num_blocks, B * H)
    
    print(f"Triton MHA Config: Grid={grid}, S={S}, Scale={scale}")
    
    _attention_kernel[grid](
        Q.view(B * H, S, D), K.view(B * H, S, D), V.view(B * H, S, D), 
        O, LSE, 
        S, scale, H, BLOCK_SIZE=64, HEAD_DIM=D,
        num_warps=8,
    )