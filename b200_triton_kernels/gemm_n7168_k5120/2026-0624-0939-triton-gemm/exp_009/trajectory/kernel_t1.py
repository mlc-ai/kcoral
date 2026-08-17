import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _gemm_kernel_persistent(
    a_desc,
    b_desc,
    C,
    M,
    N,
    K,
    stride_cm,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    
    for tile_id in tl.range(start_pid, num_tiles, tl.num_programs(0)):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        for k_step in range(tl.cdiv(K, BLOCK_K)):
            a = a_desc.load([offset_m, k_step * BLOCK_K])
            b = b_desc.load([offset_n, k_step * BLOCK_K])
            acc = tl.dot(a, b.T, acc)
        
        offs_m = offset_m + tl.arange(0, BLOCK_M)
        cols = offset_n + tl.arange(0, BLOCK_N)
        mask = offs_m[:, None] < M
        out_ptr = C + offs_m[:, None] * stride_cm + cols[None, :] * 1
        tl.store(out_ptr, acc.to(tl.bfloat16), mask=mask)


def run(A, B, C):
    """Compute C = A @ B.T into the preallocated output tensor C."""
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 64
    
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_N, BLOCK_K])
    
    num_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
    num_tiles_m = triton.cdiv(M, BLOCK_M)
    num_tiles_n = triton.cdiv(N, BLOCK_N)
    num_tiles = num_tiles_m * num_tiles_n
    grid = (min(num_sms, num_tiles),)
    
    _gemm_kernel_persistent[grid](
        a_desc,
        b_desc,
        C,
        M,
        N,
        K,
        C.stride(0),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )