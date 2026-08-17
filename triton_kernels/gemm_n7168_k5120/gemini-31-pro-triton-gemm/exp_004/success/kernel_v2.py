import torch
import triton
import triton.language as tl

# Install Triton's descriptor allocator for device-created descriptors (TMA)
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def get_autotune_config():
    configs = []
    # Broad Hopper tile shape search list optimized for SM90 registers and WGMMA
    shapes = [
        # block_m, block_n, block_k, num_warps
        (256, 128, 64, 8),
        (128, 256, 64, 8),
        (128, 128, 64, 8),
        (128, 128, 64, 4),
        (256, 64, 64, 8),
        (64, 256, 64, 8),
        (64, 128, 64, 4),
        (128, 64, 64, 4),
        # Larger K blocks significantly reduce main-loop iteration count overhead
        (128, 128, 128, 8),
        (256, 64, 128, 8),
        (64, 256, 128, 8),
        (64, 128, 128, 4),
        (128, 64, 128, 4),
    ]
    for m, n, k, w in shapes:
        # Compute stage limits dynamically based on the shared memory capacity (~228 KB max per H100 SM)
        # Element dtype is bfloat16 (2 bytes per element)
        bytes_per_stage = (m * k + n * k) * 2
        max_stages = min(5, 225000 // bytes_per_stage)
        
        # Software pipelining requires at least 2 stages 
        if max_stages < 2:
            continue
            
        for stages in range(2, max_stages + 1):
            for group_m in [4, 8]:
                configs.append(triton.Config(
                    {'BLOCK_M': m, 'BLOCK_N': n, 'BLOCK_K': k, 'GROUP_M': group_m},
                    num_stages=stages, num_warps=w
                ))
    return configs

@triton.autotune(
    configs=get_autotune_config(),
    key=['M'],
)
@triton.jit
def _tma_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    dtype = c_ptr.dtype.element_ty
    
    # Construct memory mappings defining multidimensional layout parameters safely for TMA instructions
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
        block_shape=[BLOCK_M, BLOCK_N]
    )

    tile_id = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # L2-aware grouped output-tile ordering for improved operand cache reuse
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    
    pid_in_group = tile_id % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    # Low-precision floating point GEMMs should internally accumulate in FP32 format 
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    num_k_tiles = tl.cdiv(K, BLOCK_K)
    for k_tile in tl.range(0, num_k_tiles):
        offset_k = k_tile * BLOCK_K
        
        # Automatic bounds-checked asynchronous global memory fetch by TMA hardware mechanism
        a = a_desc.load([offset_m, offset_k])
        b = b_desc.load([offset_n, offset_k])
        
        # Logical `.T` transposition on right operand informs Triton/SM90 compiler of WGMMA 
        # naturally-transposed operand layout match for peak tensor core scaling
        acc = tl.dot(a, b.T, acc)

    # Hardware handled boundary checking and data formatting writes
    c_desc.store([offset_m, offset_n], acc.to(dtype))

def run(A, B, C):
    """
    General matrix multiply (GEMM) C = A @ B.T.
    A: [M, K]
    B: [N, K]
    C: [M, N]
    Captured from Qwen3 14B qkv_proj (combined Q+K+V, N=7168, K=5120).
    """
    if A.shape[0] == 0:
        return
        
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    
    # Extract structural dimensions provided from problem definition
    N = 7168
    K = 5120
    
    # 1D flattened grouping index grid representation
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )
    
    _tma_gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1)
    )