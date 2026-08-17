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
    BM = kwargs["BLOCK_M"]
    BN = kwargs["BLOCK_N"]
    BK = kwargs["BLOCK_K"]
    kwargs["a_desc"] = TensorDescriptor.from_tensor(kwargs["A"], [BM, BK])
    kwargs["b_desc"] = TensorDescriptor.from_tensor(kwargs["B"], [BN, BK])
    kwargs["c_desc"] = TensorDescriptor.from_tensor(kwargs["C"], [BM, BN])

tma_configs = [
    # WARP_SPEC = True (partitions logical warps into async producer/consumer roles)
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8, 'WARP_SPEC': True}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8, 'WARP_SPEC': True}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8, 'WARP_SPEC': True}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'WARP_SPEC': True}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8, 'WARP_SPEC': True}, num_warps=8, num_stages=3),
    
    # WARP_SPEC = False
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8, 'WARP_SPEC': False}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8, 'WARP_SPEC': False}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8, 'WARP_SPEC': False}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'WARP_SPEC': False}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8, 'WARP_SPEC': False}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'WARP_SPEC': False}, num_warps=4, num_stages=4),
]

@triton.autotune(configs=tma_configs, key=['M', 'N', 'K'], pre_hook=tma_pre_hook)
@triton.jit
def _gemm_tma_kernel(
    a_desc, b_desc, c_desc,
    A, B, C, # Passed only for use by pre_hook
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, WARP_SPEC: tl.constexpr
):
    tile_id = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    pid_m, pid_n = _grouped_tile_coordinates(tile_id, num_pid_m, num_pid_n, GROUP_M)
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    num_k_tiles = K // BLOCK_K
    
    for k0 in tl.range(0, num_k_tiles, warp_specialize=WARP_SPEC):
        a = a_desc.load([offset_m, k0 * BLOCK_K])
        b = b_desc.load([offset_n, k0 * BLOCK_K])
        acc = tl.dot(a, b.T, acc)
        
    acc = acc.to(tl.bfloat16)
    c_desc.store([offset_m, offset_n], acc)


ptr_configs = [
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
]

@triton.autotune(configs=ptr_configs, key=['M', 'N', 'K'])
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
    
    for k0 in range(0, K // BLOCK_K):
        # We omit masks for N and K since N=7168 and K=5120 are perfectly divisible by all candidate blocks.
        a = tl.load(A_ptrs, mask=(offs_m[:, None] < M), other=0.0)
        b = tl.load(B_ptrs)
        
        acc = tl.dot(a, b.T, acc)
        
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk
        
    acc = acc.to(tl.bfloat16)
    
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(C_ptrs, acc, mask=(offs_m[:, None] < M))


def is_tma_supported(t: torch.Tensor) -> bool:
    """Verifies that the tensor physical memory layout meets strict TMA alignment rules."""
    if t.ndim != 2: return False
    if t.stride(1) != 1: return False
    if (t.stride(0) * t.element_size()) % 16 != 0: return False
    if t.data_ptr() % 16 != 0: return False
    return True


def run(A, B, C):
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N, _ = B.shape
    
    use_tma = is_tma_supported(A) and is_tma_supported(B) and is_tma_supported(C)
    
    if use_tma:
        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
        _gemm_tma_kernel[grid](
            a_desc=None, b_desc=None, c_desc=None,
            A=A, B=B, C=C,
            M=M, N=N, K=K
        )
    else:
        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
        _gemm_ptr_kernel[grid](
            A_ptr=A, B_ptr=B, C_ptr=C,
            M=M, N=N, K=K,
            stride_am=A.stride(0), stride_ak=A.stride(1),
            stride_bn=B.stride(0), stride_bk=B.stride(1),
            stride_cm=C.stride(0), stride_cn=C.stride(1)
        )