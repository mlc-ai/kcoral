import torch
import triton
import triton.language as tl

# Triton 3.7.1 global allocator setup for device-side descriptor creation.
# This provides infrastructure storage for the TMA descriptors.
def _descriptor_alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device=torch.cuda.current_device(), dtype=torch.int8)

triton.set_allocator(_descriptor_alloc_fn)

@triton.autotune(
    configs=[
        # Large rectangular tiles: Excellent for minimizing memory bandwidth while 
        # ensuring the 32K fp32 accumulators fit perfectly in 8 warps (128 registers/thread).
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64,  "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64,  "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64,  "GROUP_M": 4}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64,  "GROUP_M": 4}, num_warps=8, num_stages=4),
        
        # Balanced 128x128 tiles with larger K to increase arithmetic intensity per loop iteration
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 4}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=4, num_stages=3),
        
        # Extremely deep pipeline config to completely mask instruction & memory latencies
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64,  "GROUP_M": 8}, num_warps=4, num_stages=6),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _tma_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Device-side descriptors for Hopper TMA Hardware.
    # Out-of-bounds reads are safely padded to zero by TMA hardware natively.
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[stride_am, stride_ak], block_shape=[BLOCK_M, BLOCK_K], padding_option="zero"
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[N, K], strides=[stride_bn, stride_bk], block_shape=[BLOCK_N, BLOCK_K], padding_option="zero"
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[stride_cm, stride_cn], block_shape=[BLOCK_M, BLOCK_N]
    )

    # Standard non-persistent loop leveraging hardware cluster-scheduling mapping
    tile_id = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # L2-Aware Grouped Tile Ordering logic
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    
    pid_in_group = tile_id % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    # High precision float32 matrix accumulation
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    num_k_tiles = tl.cdiv(K, BLOCK_K)
    for k_tile in tl.range(0, num_k_tiles):
        offset_k = k_tile * BLOCK_K
        
        a = a_desc.load([offset_m, offset_k])
        b = b_desc.load([offset_n, offset_k])
        
        # b was logically loaded exactly as physical layout [BLOCK_N, BLOCK_K] to preserve continuous memory reads.
        # Transposing before standard `tl.dot` gives the correct [BLOCK_K, BLOCK_N] configuration for GEMM.
        acc = tl.dot(a, b.T, acc, out_dtype=tl.float32)

    # Cast purely at the end for accuracy
    c_desc.store([offset_m, offset_n], acc.to(c_ptr.dtype.element_ty))


def run(A, B, C):
    """
    Compute C = A @ B.T into a preallocated CUDA tensor using standard Triton
    Tensor Descriptors mapped to Hopper TMA instructions.
    
    A: [M, K]
    B: [N, K]
    C: [M, N]
    """
    torch.cuda.set_device(A.device)
    if C.numel() == 0:
        return
        
    M, K = A.shape
    N, _ = B.shape
    
    def grid_fn(META):
        num_pid_m = triton.cdiv(M, META['BLOCK_M'])
        num_pid_n = triton.cdiv(N, META['BLOCK_N'])
        return (num_pid_m * num_pid_n,)

    _tma_gemm_kernel[grid_fn](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )