import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

# Configure Triton allocator for device-side tensor descriptors on Blackwell.
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.jit
def _grouped_tile_coordinates(
    tile_id,
    num_pid_m,
    num_pid_n,
    GROUP_M: tl.constexpr,
):
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)

    pid_in_group = tile_id % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m
    return pid_m, pid_n

def get_configs():
    configs = []
    # Explore configurations across both standard loops and Blackwell warp-specialized loops
    for ws in [False, True]:
        for num_stages in [3, 4]:
            for block_m, block_n, block_k, num_warps in [
                (128, 128, 64, 4),
                (128, 256, 64, 8),
                (256, 128, 64, 8),
                (128, 128, 128, 8),
                (64, 128, 64, 4),
            ]:
                configs.append(
                    triton.Config(
                        {
                            "BLOCK_M": block_m,
                            "BLOCK_N": block_n,
                            "BLOCK_K": block_k,
                            "GROUP_M": 8,
                            "WARP_SPECIALIZE": ws,
                        },
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                )
    return configs

@triton.autotune(
    configs=get_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def gemm_kernel(
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
):
    tile_id = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    pid_m, pid_n = _grouped_tile_coordinates(
        tile_id,
        num_pid_m,
        num_pid_n,
        GROUP_M,
    )
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    # Device-created descriptors lower to TMA hardware acceleration on Blackwell
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K], padding_option="zero"
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[N, K], strides=[stride_bn, stride_bk],
        block_shape=[BLOCK_N, BLOCK_K], padding_option="zero"
    )
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_tiles = tl.cdiv(K, BLOCK_K)
    
    # Simple descriptor-load and MMA dot loop optionally targeted for warp specialization
    for k0 in tl.range(0, k_tiles, warp_specialize=WARP_SPECIALIZE):
        a_tile = a_desc.load([offset_m, k0 * BLOCK_K])
        b_tile = b_desc.load([offset_n, k0 * BLOCK_K])
        
        # B is loaded as [BLOCK_N, BLOCK_K]. Its transpose prepares it for the [K, N] RHS operand.
        acc = tl.dot(a_tile, b_tile.T, acc)
        
    offs_m = offset_m + tl.arange(0, BLOCK_M)
    offs_n = offset_n + tl.arange(0, BLOCK_N)
    
    # Using pointer loads for Epilogue storing as we require robust sub-tile boundary checking 
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    # Low-precision convert after the complete reduction to avoid truncating FP32 accumulators
    acc_cast = acc.to(tl.bfloat16)
    tl.store(c_ptrs, acc_cast, mask=mask_c)

def run(A, B, C):
    """
    Computes generalized matrix multiplication C = A @ B.T.
    Follows destination-passing style; results are written directly into C.
    """
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N, _ = B.shape
    
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    
    gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )