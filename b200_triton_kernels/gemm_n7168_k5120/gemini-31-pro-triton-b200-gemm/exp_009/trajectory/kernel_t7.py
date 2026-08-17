import torch
import triton
import triton.language as tl

# Configure Triton descriptor allocator for Blackwell TMA descriptors
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        # 128x256x128 requires 96KB shared mem per stage -> max 2 stages (<227KB limit).
        # When WARP_SPECIALIZE=True, MMA warps hold the accumulator.
        # num_warps=16 provides 512 threads, ensuring MMA warps have enough registers
        # to hold the 128x256 FP32 accumulator without spilling.
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8, "WARP_SPECIALIZE": True}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8, "WARP_SPECIALIZE": True}, num_warps=16, num_stages=2),

        # 128x256x64 requires 48KB shared mem per stage -> max 4 stages.
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8, "WARP_SPECIALIZE": True}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8, "WARP_SPECIALIZE": True}, num_warps=16, num_stages=4),
        
        # 128x128x128 requires 64KB shared mem per stage -> max 3 stages.
        # num_warps=8 (256 threads) is sufficient for a 128x128 accumulator.
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8, "WARP_SPECIALIZE": True}, num_warps=8, num_stages=3),
        
        # 128x128x64 requires 32KB shared mem per stage -> max 4 stages.
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8, "WARP_SPECIALIZE": True}, num_warps=8, num_stages=4),
        
        # Clustered fallback for smaller tiles
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8, "WARP_SPECIALIZE": True}, num_warps=8, num_stages=3, num_ctas=2),

        # Fallbacks with WARP_SPECIALIZE=False (all warps participate in MMA, easing register bounds)
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8, "WARP_SPECIALIZE": False}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8, "WARP_SPECIALIZE": False}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8, "WARP_SPECIALIZE": False}, num_warps=8, num_stages=3),
    ],
    key=["M"], 
)
@triton.jit
def _tma_gemm_kernel(
    A, B, C,
    M,
    stride_am, stride_bn, stride_cm,
    N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Grid logic
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    
    # L2-Grouped CTA Swizzling
    num_pid_in_group = GROUP_M * grid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(grid_m - first_pid_m, GROUP_M)
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offset_m = (pid_m * BLOCK_M).to(tl.int32)
    offset_n = (pid_n * BLOCK_N).to(tl.int32)
    
    # Setup Device descriptors for TMA backed loads/stores
    # Note: the inner physical strides are exactly 1, meeting standard alignment checks
    a_desc = tl.make_tensor_descriptor(
        A, shape=[M, K], strides=[stride_am, 1], block_shape=[BLOCK_M, BLOCK_K], padding_option="zero"
    )
    b_desc = tl.make_tensor_descriptor(
        B, shape=[N, K], strides=[stride_bn, 1], block_shape=[BLOCK_N, BLOCK_K], padding_option="zero"
    )
    c_desc = tl.make_tensor_descriptor(
        C, shape=[M, N], strides=[stride_cm, 1], block_shape=[BLOCK_M, BLOCK_N]
    )
    
    # Initialize the FP32 accumulator safely
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    # Core inner loop pipelined cleanly 
    # warp_specialize isolates TMA tracking out to producer warps while letting MMA execute natively in consumers
    for k_tile in tl.range(0, K // BLOCK_K, warp_specialize=WARP_SPECIALIZE):
        offset_k = (k_tile * BLOCK_K).to(tl.int32)
        
        # Asynchronous decoupled descriptor loads
        a_tile = a_desc.load([offset_m, offset_k])
        b_tile = b_desc.load([offset_n, offset_k])
        
        # Core TCgen05 Dot
        acc = tl.dot(a_tile, b_tile.T, acc)
        
    # Standard descriptor layout bound-handles bounds safely implicitly
    c_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))


def run(A, B, C):
    """
    Dest-passing entrypoint targeting robust Blackwell tuning constraints via TMA descriptors.
    Computes C = A @ B.T where outputs match bfloat16 implicitly.
    """
    if A.numel() == 0 or B.numel() == 0 or C.numel() == 0:
        return
        
    torch.cuda.set_device(A.device)
    
    M, K = A.shape
    N = C.shape[1]
    
    # Set flat 1D grid layout to efficiently hand off L2 clustering mathematically in the kernel
    def grid_fn(META):
        return (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]), )
    
    _tma_gemm_kernel[grid_fn](
        A, B, C,
        M,
        A.stride(0), B.stride(0), C.stride(0),
        N=N, K=K
    )