import torch
import triton
import triton.language as tl


def alloc_fn(size: int, alignment: int, stream):
    """Allocator for device-created descriptor infrastructure storage."""
    return torch.empty(size, device="cuda", dtype=torch.int8)


@triton.jit
def _gemm(
    A,
    B,
    C,
    M,
    K,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    """
    Optimized GEMM kernel targeting Hopper WGMMA capabilities via device TMA descriptors.
    Computes C = A @ B.T using persistent software pipelining.
    """
    a_desc = tl.make_tensor_descriptor(
        A,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_M, BLOCK_K],
        padding_option="zero",
    )
    
    b_desc = tl.make_tensor_descriptor(
        B,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_N, BLOCK_K],
        padding_option="zero",
    )
    
    c_desc = tl.make_tensor_descriptor(
        C,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )
    
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    
    start_pid = tl.program_id(0)
    tile_stride = tl.num_programs(0)
    
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    
    for tile_id in tl.range(
        start_pid,
        num_tiles,
        tile_stride,
        flatten=False,
        warp_specialize=WARP_SPECIALIZE,
    ):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        
        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        for k_iter in range(num_k_tiles):
            offset_k = k_iter * BLOCK_K
            
            if offset_k + BLOCK_K <= K:
                a = a_desc.load([offset_m, offset_k])
                b = b_desc.load([offset_n, offset_k])
                
                # Transposing B locally aligns it strictly for consuming via standard WGMMA tensors.
                acc = tl.dot(a, b.T, acc)
            else:
                break
        
        # Defer casting until after summing to prevent losing accumulated lower bits.
        out_val = acc.to(tl.bfloat16)
        c_desc.store([offset_m, offset_n], out_val)


def run(A, B, C):
    """
    Efficient destination-passing wrapper launching our TMA GEMM. 
    
    Takes preallocated tensors explicitly matching definition sequence constraints.
    """
    torch.cuda.set_device(A.device)
    
    M, K = A.shape
    N, _ = B.shape
    
    # Allocate necessary space for internal device descriptor bookkeeping.
    triton.set_allocator(alloc_fn)
    
    # Limit launch grid expansion to available physical SM capacity bounds.
    num_sms = min(132, (M + 127) // 128 * ((N + 127) // 128))
    
    grid = (num_sms,)
    _gemm[grid](
        A, B, C,
        M, K, N,
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=128,
        NUM_SMS=num_sms, WARP_SPECIALIZE=False,
        num_warps=8, num_stages=4,
    )