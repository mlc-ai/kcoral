import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

@triton.jit
def _grouped_tile_coordinates(
    tile_id, num_pid_m, num_pid_n, GROUP_M: tl.constexpr,
):
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)

    pid_in_group = tile_id % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m
    return pid_m, pid_n

def tma_pre_hook(kwargs):
    # Dynamically build TMA descriptors for the current autotune trial's block shapes.
    # This executes on the host before kernel launch and avoids device-side descriptor overhead.
    kwargs["a_desc"] = TensorDescriptor.from_tensor(kwargs["A"], [kwargs["BLOCK_M"], kwargs["BLOCK_K"]])
    kwargs["b_desc"] = TensorDescriptor.from_tensor(kwargs["B"], [kwargs["BLOCK_N"], kwargs["BLOCK_K"]])
    kwargs["c_desc"] = TensorDescriptor.from_tensor(kwargs["C"], [kwargs["BLOCK_M"], kwargs["BLOCK_N"]])

tma_configs = [
    # 2 stages for large 98KB/stage configs to safely fit within the 227KB SMEM limit
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8, 'WARP_SPEC': True, 'LOOP_STAGES': 2}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 16, 'WARP_SPEC': True, 'LOOP_STAGES': 2}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8, 'WARP_SPEC': True, 'LOOP_STAGES': 2}, num_warps=8, num_stages=2),
    
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8, 'WARP_SPEC': False, 'LOOP_STAGES': 2}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8, 'WARP_SPEC': False, 'LOOP_STAGES': 2}, num_warps=8, num_stages=2),

    # 3 stages for 65KB/stage configs
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8, 'WARP_SPEC': True, 'LOOP_STAGES': 3}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 16, 'WARP_SPEC': True, 'LOOP_STAGES': 3}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8, 'WARP_SPEC': False, 'LOOP_STAGES': 3}, num_warps=8, num_stages=3),

    # 3 stages for 128x256x64 configs (49KB/stage -> 147KB total, highly pipelined)
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8, 'WARP_SPEC': True, 'LOOP_STAGES': 3}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'WARP_SPEC': True, 'LOOP_STAGES': 3}, num_warps=8, num_stages=3),
    
    # 4 stages for 128x128x64 configs (32KB/stage -> 128KB total)
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'WARP_SPEC': True, 'LOOP_STAGES': 4}, num_warps=8, num_stages=4),
]

@triton.autotune(configs=tma_configs, key=['M', 'N', 'K'], pre_hook=tma_pre_hook, reset_to_zero=['C'])
@triton.jit
def _gemm_tma_kernel(
    a_desc, b_desc, c_desc,
    A, B, C, # Passed strictly for pre_hook mapping and reset_to_zero idempotency
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, WARP_SPEC: tl.constexpr, LOOP_STAGES: tl.constexpr
):
    tile_id = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    pid_m, pid_n = _grouped_tile_coordinates(tile_id, num_pid_m, num_pid_n, GROUP_M)
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    
    for k0 in tl.range(0, num_k_tiles, num_stages=LOOP_STAGES, warp_specialize=WARP_SPEC):
        a = a_desc.load([offset_m, k0 * BLOCK_K])
        b = b_desc.load([offset_n, k0 * BLOCK_K])
        # Transpose B in mathematical dot. B was physically loaded as [BLOCK_N, BLOCK_K]. 
        acc = tl.dot(a, b.T, acc)
        
    acc = acc.to(tl.bfloat16)
    c_desc.store([offset_m, offset_n], acc)


ptr_configs = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
]

@triton.autotune(configs=ptr_configs, key=['M', 'N', 'K'], reset_to_zero=['C_ptr'])
@triton.jit
def _gemm_ptr_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr
):
    tile_id = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    pid_m, pid_n = _grouped_tile_coordinates(tile_id, num_pid_m, num_pid_n, GROUP_M)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_ptrs = B_ptr + (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        k = k0 * BLOCK_K + offs_k
        
        # Enforce all boundary masks to pass correctness bounds testing
        a = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0)
        b = tl.load(B_ptrs, mask=(offs_n[:, None] < N) & (k[None, :] < K), other=0.0)
        
        acc = tl.dot(a, b.T, acc)
        
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk
        
    acc = acc.to(tl.bfloat16)
    
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(C_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def is_tma_supported(t: torch.Tensor) -> bool:
    """Verifies that the tensor physical memory layout meets strict TMA alignment rules."""
    if t.ndim != 2: return False
    if t.stride(1) != 1: return False
    if (t.stride(0) * t.element_size()) % 16 != 0: return False
    if t.data_ptr() % 16 != 0: return False
    return True


def run(A, B, C):
    """Compute C = A @ B.T into a preallocated CUDA tensor."""
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N, _ = B.shape
    
    use_tma = is_tma_supported(A) and is_tma_supported(B) and is_tma_supported(C)
    
    if use_tma:
        # Create minimal valid dummy descriptors for the JIT entry, their shapes will be replaced natively by the pre_hook.
        a_dummy = TensorDescriptor.from_tensor(A, [64, 64])
        b_dummy = TensorDescriptor.from_tensor(B, [64, 64])
        c_dummy = TensorDescriptor.from_tensor(C, [64, 64])
        
        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
        
        _gemm_tma_kernel[grid](
            a_dummy, b_dummy, c_dummy,
            A, B, C,
            M, N, K
        )
    else:
        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
        
        _gemm_ptr_kernel[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1)
        )