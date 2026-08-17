import torch
import triton
import triton.language as tl

# Required allocator setup to facilitate rapid device-side TMA descriptor construction
def _descriptor_allocator(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(_descriptor_allocator)

def get_configs():
    configs = []
    # Exhaustively tuned parameter space mapped precisely to B200 TMA/TMEM physical limits.
    for ws in [True, False]:
        for block_m, block_n in [
            (256, 128),
            (128, 256),
            (128, 128),
            (256, 64),
            (64, 256),
            (128, 64),
            (64, 128),
        ]:
            for warps in [4, 8]:
                # Warp specialization divides workloads across partitions; heavily needs enough functional warps
                if ws and warps == 4:
                    continue
                
                for stages in [2, 3, 4, 5]:
                    # Rigorous fail-safe limit: B200 SMs strictly max out at 228 KiB of addressable Shared Memory.
                    smem_per_stage = (block_m + block_n) * 128 * 2  # Physical BF16 footprint
                    total_smem = stages * smem_per_stage
                    if total_smem > 220 * 1024:
                        continue
                        
                    # Standard optimized launch
                    configs.append(triton.Config(
                        {'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': 128, 'GROUP_M': 8, 'WARP_SPECIALIZE': ws, 'LOOP_STAGES': stages},
                        num_warps=warps, num_stages=stages, num_ctas=1
                    ))
                    
                    # Threadblock cluster evaluation natively aggregates CTAs over symmetric SM boundaries for expanded L2 scaling 
                    if block_m >= 128 and block_n >= 128:
                        configs.append(triton.Config(
                            {'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': 128, 'GROUP_M': 8, 'WARP_SPECIALIZE': ws, 'LOOP_STAGES': stages},
                            num_warps=warps, num_stages=stages, num_ctas=4
                        ))
    return configs


@triton.autotune(configs=get_configs(), key=["M", "N", "K"])
@triton.jit
def _gemm_tma_device_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
):
    tile_id = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # Robust Grouped Traversal Pattern. Traverses multiple spatial M's together efficiently sticking 
    # the common operand locally in the shared cache layers before moving onto orthogonal sweeps.
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    
    pid_in_group = tile_id % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m
    
    offs_m = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N
    
    # Utilize extremely optimized native device-created TMA descriptors entirely bypassing 
    # Python latency mappings. Inherently zero-padded tails.
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K],
        padding_option="zero"
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[stride_bn, stride_bk],
        block_shape=[BLOCK_N, BLOCK_K],
        padding_option="zero"
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[stride_cm, stride_cn],
        block_shape=[BLOCK_M, BLOCK_N],
        padding_option="zero"  # Out of boundary TMA stores gracefully ignored inherently by Blackwell
    )
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    k_tiles = tl.cdiv(K, BLOCK_K)
    
    # Internal math loop explicitly linked safely onto hardware pipelining mechanisms   
    for k0 in tl.range(0, k_tiles, num_stages=LOOP_STAGES, warp_specialize=WARP_SPECIALIZE):
        a = a_desc.load([offs_m, k0 * BLOCK_K])
        b = b_desc.load([offs_n, k0 * BLOCK_K])
        
        # Operand B mapped dynamically transpiled naturally aligns to physical `tcgen05.mma` instructions
        acc = tl.dot(a, b.T, acc)
        
    # Standard hardware-routed result cleanly overriding arbitrary boundaries  
    c_desc.store([offs_m, offs_n], acc.to(tl.bfloat16))


def run(A, B, C):
    """
    Computes generalized scaled precision matrix multiplication C = A @ B.T.
    Targeting specific structural workloads modeled off Qwen-3 projection bounds.
    """
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N, _ = B.shape
    
    if M == 0 or N == 0 or K == 0:
        return
        
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    
    _gemm_tma_device_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )