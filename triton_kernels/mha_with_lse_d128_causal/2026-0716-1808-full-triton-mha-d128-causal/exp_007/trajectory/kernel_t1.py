import torch
import triton
import triton.language as tl


@triton.jit
def _attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S_len, B, H, D,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """
    Persistent warp-specialized FlashAttention kernel.
    
    Computes causal multi-head attention forward pass with Log-Sum-Exp outputs.
    Q/K/V are expected to be contiguous tensors of shape [B, H, S_len, D].
    """
    scale = 1.0 / (D ** 0.5)
    
    # Create device-side tensor descriptors for TMA loads
    q_desc = tl.make_tensor_descriptor(
        Q_ptr, shape=[B * H * S_len, D], strides=[D, 1],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero")
    k_desc = tl.make_tensor_descriptor(
        K_ptr, shape=[B * H * S_len, D], strides=[D, 1],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(
        V_ptr, shape=[B * H * S_len, D], strides=[D, 1],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero")
    
    # Distribute work across the minimal set of CTAs
    total_ctas = B * H * triton.cdiv(S_len, BLOCK_M)
    start_pid = tl.program_id(0)
    num_pid = total_ctas // NUM_SMS
    extra = total_ctas % NUM_SMS
    my_pids = []
    for i in range(num_pid):
        my_pids.append(start_pid * num_pid + i)
    if start_pid < extra:
        my_pids.append(num_pid * NUM_SMS + start_pid)
    
    num_q_blks = triton.cdiv(S_len, BLOCK_M)
    
    for pid in my_pids:
        bh_id = pid // num_q_blks
        q_blk = pid % num_q_blks
        
        valid_q = (q_blk * BLOCK_M + tl.arange(0, BLOCK_M)) < S_len
        if not valid_q[0]:
            continue
        
        # Load Q tile and initialize accumulators
        Q = q_desc.load([bh_id * S_len + q_blk * BLOCK_M, 0])
        
        O_acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
        m = tl.full((BLOCK_M,), -1e38, tl.float32)
        l = tl.full((BLOCK_M,), 0.0, tl.float32)
        
        # Causal attention limits the search to the current query block
        k_start = 0
        k_end = min(q_blk + 1, triton.cdiv(S_len, BLOCK_N))
        
        kv_base = bh_id * S_len
        
        # Initialize double buffer barriers
        barriers = [tl.named_barrier(f"stage_{i}") for i in range(2)]
        
        # Prime the pump with the initial fetch
        if k_start < k_end:
            k_desc.load([kv_base + k_start * BLOCK_N, 0], stage=0, commit_barrier=barriers[0])
            v_desc.load([kv_base + k_start * BLOCK_N, 0], stage=0, commit_barrier=barriers[1])
            
        for k_blk in tl.range(k_start, k_end, 1, flatten=False, warp_specialize=True):
            stage_idx = k_blk % 2
            
            # Software pipelining: issue load for next block
            if k_blk + 1 < k_end:
                next_kv_offset = kv_base + (k_blk + 1) * BLOCK_N
                next_stage = (k_blk + 1) % 2
                k_desc.load([next_kv_offset, 0], stage=next_stage, commit_barrier=barriers[next_stage])
                v_desc.load([next_kv_offset, 0], stage=next_stage, commit_barrier=barriers[next_stage])
                
            # Wait for the current block to be fully loaded
            tl.wait_barrier(barriers[stage_idx])
            
            K = k_desc.load([kv_base + k_blk * BLOCK_N, 0])
            V = v_desc.load([kv_base + k_blk * BLOCK_N, 0])
            
            S = tl.dot(Q, K.T) * scale
            
            # Causal Masking & Online Softmax
            if k_blk == q_blk:
                r_idx = q_blk * BLOCK_M + tl.arange(0, BLOCK_M)
                c_idx = k_blk * BLOCK_N + tl.arange(0, BLOCK_N)
                valid = c_idx[None, :] <= r_idx[:, None]
                S = tl.where(valid, S, -1e38)
            
            m_prev = m
            m = tl.maximum(m, tl.max(S, axis=1))
            P = tl.exp(S - m[:, None])
            
            l = l * tl.exp(m_prev - m) + tl.sum(P, axis=1)
            
            # Rescale accumulated features and execute PV GEMM
            O_acc *= tl.exp(m_prev - m)[:, None]
            O_acc = tl.dot(P, V, O_acc)
        
        # Epilogue calculations
        O_acc = O_acc / l[:, None]
        lse_val = m + tl.log(l)
        
        valid_q = (q_blk * BLOCK_M + tl.arange(0, BLOCK_M)) < S_len
        
        # Write outputs observing boundaries
        out_ptr = O_ptr + bh_id * S_len * D + q_blk * BLOCK_M * D
        row_idx = tl.arange(0, BLOCK_M)[:, None] * D
        col_idx = tl.arange(0, BLOCK_D)[None, :]
        tl.store(out_ptr + row_idx + col_idx, O_acc.to(tl.bfloat16), mask=valid_q[:, None])
        
        lse_ptr = LSE_ptr + bh_id * S_len + q_blk * BLOCK_M
        tl.store(lse_ptr + tl.arange(0, BLOCK_M), lse_val, mask=valid_q)


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
    
    NUM_SMS = 132 
    total_ctas = B * H * triton.cdiv(S_len, 128)
    grid_size = min(NUM_SMS, total_ctas)
    
    grid = (grid_size,)
    _attention_kernel[grid](
        Q_ptr, K_ptr, V_ptr, O, LSE,
        S_len, B, H, D,
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_D=128,
        NUM_SMS=NUM_SMS,
        num_warps=4,
        num_stages=2,
    )