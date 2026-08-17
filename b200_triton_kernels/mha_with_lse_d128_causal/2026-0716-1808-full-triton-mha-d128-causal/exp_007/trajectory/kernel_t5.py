import torch
import triton
import triton.language as tl


@triton.jit
def _attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S_len,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Optimized FlashAttention kernel leveraging shared memory and software pipelining.
    
    Computes causal multi-head attention forward pass with Log-Sum-Exp outputs.
    Uses explicit shared memory staging to overlap HBM loads with Tensor Core compute.
    """
    scale = 1.0 / (BLOCK_D ** 0.5)
    
    bh_id = tl.program_id(0)
    q_blk = tl.program_id(1)
    
    valid_q = (q_blk * BLOCK_M + tl.arange(0, BLOCK_M)) < S_len
    if not valid_q[0]:
        return
    
    # Explicitly allocated shared memory space for zero-latency re-access during looping
    smem_Q = tl_shared_array((BLOCK_M, BLOCK_D), layout="row_major", elem_dtype=tl.float32)
    smem_K = tl_shared_array((2, BLOCK_N, BLOCK_D), layout="row_major", elem_dtype=tl.bfloat16)
    smem_V = tl_shared_array((2, BLOCK_N, BLOCK_D), layout="row_major", elem_dtype=tl.bfloat16)
    
    O_acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    m = tl.full((BLOCK_M,), -1e38, tl.float32)
    l = tl.full((BLOCK_M,), 0.0, tl.float32)
    
    max_k_blk = min(q_blk, (S_len - 1) // BLOCK_N)
    
    # Pipeline staging control
    barriers = [tl.named_barrier(f"stage_{i}") for i in range(2)]
    
    # Initial Prime: Load Q and the initial KV slice (Block 0)
    col_offsets = tl.arange(0, BLOCK_D)
    q_row_offsets = (bh_id * S_len + q_blk * BLOCK_M) + tl.arange(0, BLOCK_M)
    
    Q = tl.load(Q_ptr + q_row_offsets[:, None] * BLOCK_D + col_offsets[None, :], 
                mask=valid_q[:, None], other=0.0)
    smem_Q[:] = Q
    
    if 0 <= max_k_blk:
        load_2d_async(K_ptr, bh_id * S_len + 0 * BLOCK_N, 0, (S_len, BLOCK_D), BLOCK_D, smem_K, 0, barriers[0])
        load_2d_async(V_ptr, bh_id * S_len + 0 * BLOCK_N, 0, (S_len, BLOCK_D), BLOCK_D, smem_V, 0, barriers[0])
        
    for k_blk in range(0, max_k_blk + 1):
        stage_idx = k_blk % 2
        
        # Software pipelining: issue load for next block while processing current
        if k_blk + 1 <= max_k_blk:
            next_kv_offset = bh_id * S_len + (k_blk + 1) * BLOCK_N
            next_stage = (k_blk + 1) % 2
            load_2d_async(K_ptr, next_kv_offset, 0, (S_len, BLOCK_D), BLOCK_D, smem_K, next_stage, barriers[next_stage])
            load_2d_async(V_ptr, next_kv_offset, 0, (S_len, BLOCK_D), BLOCK_D, smem_V, next_stage, barriers[next_stage])
            
        # Wait for the current block to be fully loaded
        tl.wait_barrier(barriers[stage_idx])
        
        K = smem_K[stage_idx]
        V = smem_V[stage_idx]
        
        S = tl.dot(Q, K.T) * scale
        
        # Causal Masking & Online Softmax
        r_idx = q_blk * BLOCK_M + tl.arange(0, BLOCK_M)
        c_idx = k_blk * BLOCK_N + tl.arange(0, BLOCK_N)
        valid = (c_idx[None, :] <= r_idx[:, None]) & (c_idx[None, :] < S_len)
        
        # Mask out fully invalid rows to avoid NaN propagation in downstream exp/log/sum reductions
        valid_q_expanded = valid_q[:, None].to(tl.bool)
        S = tl.where(valid_q_expanded & valid, S, -1e38)
        
        m_prev = m
        m = tl.maximum(m, tl.max(S, axis=1))
        P = tl.exp(S - m[:, None])
        
        l = l * tl.exp(m_prev - m) + tl.sum(P, axis=1)
        
        # Rescale accumulated features and execute PV GEMM
        O_acc *= tl.exp(m_prev - m)[:, None]
        O_acc = tl.dot(P, V, acc=O_acc)
        
        # Manual cross-stage synchronization fence
        if stage_idx == 0:
            tl.wait_barrier(barriers[1])
        else:
            tl.wait_barrier(barriers[0])
            
    # Epilogue calculations
    O_acc = O_acc / l[:, None]
    lse_val = m + tl.log(l)
    
    valid_q = (q_blk * BLOCK_M + tl.arange(0, BLOCK_M)) < S_len
    
    # Write outputs observing boundaries
    out_ptr = O_ptr + bh_id * S_len * BLOCK_D + q_blk * BLOCK_M * BLOCK_D
    row_idx = tl.arange(0, BLOCK_M)[:, None] * BLOCK_D
    col_idx = tl.arange(0, BLOCK_D)[None, :]
    tl.store(out_ptr + row_idx + col_idx, O_acc.to(tl.bfloat16), mask=valid_q[:, None])
    
    lse_ptr = LSE_ptr + bh_id * S_len + q_blk * BLOCK_M
    tl.store(lse_ptr + tl.arange(0, BLOCK_M), lse_val, mask=valid_q)


@triton.jit
def load_2d_async(base_ptr, row, col, shape, stride, smem, idx, commit_barrier):
    row = row.to(tl.int64)
    col = col.to(tl.int64)
    valid = (row < shape[0]).to(tl.int1)
    ptr = base_ptr + row * stride + col
    ptr = ptr.to(tl.int64)
    smem_ptr = smem.base_ptr + idx
    smem_ptr = smem_ptr.to(tl.int64)
    asm = "cp.async.ca.shared.global {valid}, {ptr}, 1;"
    args = (valid, ptr)
    tl.inline_asm_elementwise(asm, "{reg}, {reg}, {num}", args, dtype=tl.uint8, is_pure=False, pack=1)
    if commit_barrier is not None:
        tl.commit_barrier(commit_barrier)


def run(Q, K, V, O, LSE):
    """Compute causal attention O and LSE into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    B, H, S_len, D = Q.shape
    
    # Allocate infrastructure storage space on device for descriptors
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)
    
    triton.set_allocator(alloc_fn)
    
    Q_ptr = Q.contiguous()
    K_ptr = K.contiguous()
    V_ptr = V.contiguous()
    
    grid = (B * H, triton.cdiv(S_len, 128))
    
    _attention_kernel[grid](
        Q_ptr, K_ptr, V_ptr, O, LSE,
        S_len,
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_D=128,
        num_warps=4,
        num_stages=2,
    )