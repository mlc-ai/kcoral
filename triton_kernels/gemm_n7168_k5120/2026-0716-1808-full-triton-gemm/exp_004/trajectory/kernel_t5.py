import torch
import triton
import triton.language as tl


def alloc_fn(size: int, alignment: int, stream):
    """Allocator for device-created descriptor infrastructure storage."""
    return torch.empty(size, device="cuda", dtype=torch.int8)


@triton.jit
def _grouped_tile_coordinates(
    tile_id,
    num_pid_m,
    num_pid_n,
    GROUP_M: tl.constexpr,
):
    """
    Decode flat tile ID into logically grouped coordinates to cluster contiguous
    blocks of work that heavily share underlying operand subblocks (typically B matrices).
    """
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    
    pid_in_group = tile_id % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m
    return pid_m, pid_n


@triton.jit
def _gemm(
    A,
    B,
    C,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    """
    Computes C = A @ B.T using device-created TMA descriptors and static layout knowledge.
    Utilizes a persistent loop over grouped tiles to drastically increase L2 residency probability.
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
    
    for tile_id in tl.range(
        start_pid,
        num_tiles,
        tile_stride,
        flatten=False,
        warp_specialize=WARP_SPECIALIZE,
    ):
        
        GROUP_M: tl.constexpr = 4
        pid_m, pid_n = _grouped_tile_coordinates(
            tile_id, num_pid_m, num_pid_n, GROUP_M
        )
        
        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        num_k_tiles = tl.cdiv(K, BLOCK_K)
        for k_iter in range(num_k_tiles):
            offset_k = k_iter * BLOCK_K
            
            a = a_desc.load([offset_m, offset_k])
            b = b_desc.load([offset_n, offset_k])
            
            # B acts conceptually as [BLOCK_N, BLOCK_K] internally, 
            # transposing locally aligns it structurally matching standard expectations.
            acc = tl.dot(a, b.T, acc)
            
        # Defer casting until after summing to prevent losing accumulated lower bits.
        out_val = acc.to(tl.bfloat16)
        c_desc.store([offset_m, offset_n], out_val)


def run(A, B, C):
    """
    Efficient destination-passing wrapper launching our highly specialized persistent TMA GEMM. 
    
    Takes preallocated tensors explicitly matching definition sequence constraints.
    """
    torch.cuda.set_device(A.device)
    
    M, K = A.shape
    N, _ = B.shape
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128
    
    num_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    NUM_SMS = 132
    
    # Limit initial grid expansion to available SM capacity to avoid scheduling overhead bottlenecks 
    num_sms = min(NUM_SMS, num_tiles)
    grid = (num_sms,)
    
    triton.set_allocator(alloc_fn)
    
    _gemm[grid](
        A, B, C,
        M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        NUM_SMS=NUM_SMS, WARP_SPECIALIZE=False,
        num_warps=8,
    )