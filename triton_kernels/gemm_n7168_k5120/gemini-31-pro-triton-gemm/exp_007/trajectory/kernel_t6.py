import torch
import triton
import triton.language as tl

# Standard Triton descriptor allocator for device-created TensorDescriptors.
# This is infrastructure storage mapped on the host, required to utilize the
# Hopper TMA warp-specialized persistent contract safely.
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def get_configs():
    """
    Extensive set of configurations specifically tuned for Hopper SM90 TMA/WGMMA limits.
    Filters out unsupported layouts that could crash WGMMA lowering (e.g. K=128 with N<128).
    Includes multiple PROGS_PER_SM targets to maximize hardware occupancy.
    """
    configs = []
    
    shapes = [
        # Large tiles for max Tensor Core utilization
        (256, 128, 64, 8, 3, 1),
        (128, 256, 64, 8, 3, 1),
        (128, 128, 128, 8, 3, 1),
        
        # 128x128 standard robust paths
        (128, 128, 64, 4, 3, 1),
        (128, 128, 64, 4, 4, 1),
        (128, 128, 64, 8, 4, 1),
        (128, 128, 64, 4, 5, 1),
        
        # Paths allowing 2 CTAs per SM (requires total shared memory < 114KB)
        (128, 128, 64, 4, 2, 2),
        (128, 128, 64, 8, 2, 2),
        (64, 128, 64, 4, 3, 2),
        (128, 64, 64, 4, 3, 2),
        
        # Skinny latency-hiding tiles
        (64, 256, 64, 4, 3, 1),
        (256, 64, 64, 4, 3, 1),
        (64, 256, 64, 8, 4, 1),
        (256, 64, 64, 8, 4, 1),
    ]
    
    for m, n, k, w, s, p in shapes:
        # Avoid K=128 with N<128 to strictly prevent WGMMA layout lowering failures.
        if k == 128 and n < 128:
            continue
            
        # Heavy L2 cache reuse relies on GROUP_M logic
        for group in [8, 16]:
            for warp_spec in [True, False]:
                configs.append(triton.Config(
                    {'BLOCK_M': m, 'BLOCK_N': n, 'BLOCK_K': k, 
                     'GROUP_M': group, 'WARP_SPECIALIZE': warp_spec, 'PROGS_PER_SM': p},
                    num_warps=w, num_stages=s
                ))
    return configs

@triton.autotune(
    configs=get_configs(),
    key=['M', 'N', 'K'],
)
@triton.jit
def _gemm_tma_persistent_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_bn, stride_cm,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr, PROGS_PER_SM: tl.constexpr, WARP_SPECIALIZE: tl.constexpr
):
    """
    Standard Hopper warp-specialized descriptor loop.
    Maps an entire GEMM over persistently scheduled thread blocks across all SMs.
    TMA loads directly from HBM to SMEM, autonomously driving Hopper WGMMA via dot(a, b.T).
    """
    dtype = C_ptr.dtype.element_ty
    
    a_desc = tl.make_tensor_descriptor(
        A_ptr,
        shape=[M, K],
        strides=[stride_am, 1],
        block_shape=[BLOCK_M, BLOCK_K],
        padding_option="zero"
    )
    b_desc = tl.make_tensor_descriptor(
        B_ptr,
        shape=[N, K],
        strides=[stride_bn, 1],
        block_shape=[BLOCK_N, BLOCK_K],
        padding_option="zero"
    )
    c_desc = tl.make_tensor_descriptor(
        C_ptr,
        shape=[M, N],
        strides=[stride_cm, 1],
        block_shape=[BLOCK_M, BLOCK_N]
    )

    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    
    TOTAL_PROGRAMS = NUM_SMS * PROGS_PER_SM
    
    for tile_id in tl.range(
        start_pid,
        num_tiles,
        TOTAL_PROGRAMS,
        flatten=False,
        warp_specialize=WARP_SPECIALIZE
    ):
        # L2-Aware Grouped Tile Ordering
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        
        pid_in_group = tile_id % num_pid_in_group
        pid_m = first_pid_m + (pid_in_group % group_size_m)
        pid_n = pid_in_group // group_size_m
        
        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        
        # Accumulate natively on Tensor Cores in FP32
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        for k_tile in range(num_k_tiles):
            offset_k = k_tile * BLOCK_K
            a = a_desc.load([offset_m, offset_k])
            b = b_desc.load([offset_n, offset_k])
            
            # B layout physically stored as [N, K]. Passing b.T here constructs a
            # logically correct view triggering the fastest Hopper non-transposed WGMMA fast path.
            acc = tl.dot(a, b.T, acc)
            
        c_desc.store([offset_m, offset_n], acc.to(dtype))

def run(A, B, C):
    """
    Destination-passing execution entry point.
    Computes C = A @ B.T.
    """
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    if M == 0 or N == 0 or K == 0:
        return
        
    num_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
    
    def grid_fn(META):
        num_tiles = triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"])
        total_programs = num_sms * META["PROGS_PER_SM"]
        # Limit the initial grid launch purely to necessary CTAs to utilize hardware persistence perfectly
        return (min(total_programs, num_tiles),)
    
    _gemm_tma_persistent_kernel[grid_fn](
        A, B, C,
        M, N, K,
        A.stride(0), B.stride(0), C.stride(0),
        NUM_SMS=num_sms
    )