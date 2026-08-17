import torch
import triton
import triton.language as tl

# Install Triton's descriptor allocator to support device-created descriptors for TMA WGMMA
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def get_autotune_config():
    configs = []
    # Exhaustive search across high-efficiency configurations, utilizing
    # threadblock clusters (num_ctas) for L2 cache locality and TMA multicast.
    shapes = [
        # (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, num_ctas)
        (256, 128, 64, 8, 3, 1),
        (256, 128, 64, 8, 4, 1),
        (128, 256, 64, 8, 3, 1),
        (128, 256, 64, 8, 4, 1),
        (128, 128, 128, 8, 3, 1),
        (128, 128, 64, 4, 4, 1),
        (128, 128, 64, 4, 5, 1),
        
        # Clustered configurations for massive L2 throughput
        (256, 128, 64, 8, 3, 2),
        (128, 256, 64, 8, 3, 2),
        (256, 128, 64, 8, 3, 4),
        (128, 256, 64, 8, 3, 4),
        (128, 128, 128, 8, 3, 2),
    ]
    
    for m, n, k, w, s, c in shapes:
        # Guarantee we don't violate the ~228KB Shared Memory limit per SM on Hopper
        bytes_per_stage = (m * k + n * k) * 2
        if bytes_per_stage * s > 225000:
            continue
            
        for ws in [True, False]:
            configs.append(triton.Config({
                'BLOCK_M': m, 'BLOCK_N': n, 'BLOCK_K': k,
                'GROUP_M': 8, 'WARP_SPECIALIZE': ws
            }, num_stages=s, num_warps=w, num_ctas=c))
            
    return configs

@triton.autotune(
    configs=get_autotune_config(),
    key=['M'],
)
@triton.jit
def _descriptor_persistent_matmul(
    a_ptr, b_ptr, c_ptr,
    M,
    stride_am, stride_bn, stride_cm,
    N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    dtype = c_ptr.dtype.element_ty
    
    # Create TMA descriptors inherently mapped to natively optimized layouts. 
    # Explicitly hardcoding the last stride to `1` ensures TMA 16-byte alignment conditions are met safely.
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[stride_am, 1],
        block_shape=[BLOCK_M, BLOCK_K], padding_option="zero"
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[N, K], strides=[stride_bn, 1],
        block_shape=[BLOCK_N, BLOCK_K], padding_option="zero"
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[stride_cm, 1],
        block_shape=[BLOCK_M, BLOCK_N]
    )

    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    num_k_tiles = tl.cdiv(K, BLOCK_K)

    # Automatically matches step to the effectively deployed CTAs taking `num_ctas` into account.
    step = tl.num_programs(0)
    
    for tile_id in tl.range(
        start_pid, num_tiles, step,
        flatten=False, warp_specialize=WARP_SPECIALIZE
    ):
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        
        pid_in_group = tile_id % num_pid_in_group
        pid_m = first_pid_m + (pid_in_group % group_size_m)
        pid_n = pid_in_group // group_size_m

        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        for k_tile in range(num_k_tiles):
            offset_k = k_tile * BLOCK_K
            
            # Utilizing TMA loads removes inline masking/bounds check overhead natively
            a = a_desc.load([offset_m, offset_k])
            b = b_desc.load([offset_n, offset_k])
            
            # Syntactic transposition logically maps `b_ptr` storage format directly to the Right-Operand
            # WGMMA Fast Path structure expected by Hopper matrix math engines. 
            acc = tl.dot(a, b.T, acc)

        c_desc.store([offset_m, offset_n], acc.to(dtype))

def run(A, B, C):
    """
    General matrix multiply C = A @ B.T.
    A: [M, K]
    B: [N, K]
    C: [M, N]
    Both N (7168) and K (5120) strictly maintained as specified compile-time constants.
    """
    if A.shape[0] == 0:
        return
        
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    
    # Constants from the problem definition structurally injected as constexpr values
    N = 7168
    K = 5120
    
    num_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
    
    def grid_fn(META):
        num_ctas = META.get('num_ctas', 1)
        # Guarantees the distributed total CTAs correctly scales with clustered architectures
        grid_size = (num_sms // num_ctas) * num_ctas
        return (grid_size, )
    
    _descriptor_persistent_matmul[grid_fn](
        A, B, C,
        M, 
        A.stride(0), B.stride(0), C.stride(0),
        N=N, K=K
    )