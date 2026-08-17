import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _gemm_kernel(
    a_desc,
    b_desc,
    c_desc,
    M,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    STAGES: tl.constexpr,
):
    start_pid = tl.program_id(0)
    tile_stride = tl.num_programs(0)
    
    N = 7168
    K = 5120
    
    assert M % BLOCK_M == 0, f"M={M} not evenly divisible by BLOCK_M={BLOCK_M}"
    assert N % BLOCK_N == 0, f"N={N} not evenly divisible by BLOCK_N={BLOCK_N}"
    assert K % BLOCK_K == 0, f"K={K} not evenly divisible by BLOCK_K={BLOCK_K}"
    
    num_tiles_m = M // BLOCK_M
    num_tiles_n = N // BLOCK_N
    
    for tile_id in range(start_pid, num_tiles_m * num_tiles_n, tile_stride):
        pid_m = tile_id // num_tiles_n
        pid_n = tile_id % num_tiles_n
        
        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        for k_tile in tl.range(
            0,
            K // BLOCK_K,
            num_stages=STAGES,
            range_unit=BLOCK_K,
            range_stride=BLOCK_K,
        ):
            offset_k = k_tile * BLOCK_K
            
            a = a_desc.load([offset_m, offset_k])
            b = b_desc.load([offset_n, offset_k])
            
            acc += tl.dot(a, b.T)
        
        acc_bf16 = acc.to(tl.bfloat16)
        c_desc.store([offset_m, offset_n], acc_bf16)


def run(A, B, C):
    """Perform destination-passing GEMM computation C = A @ B.T."""
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    BLOCK_M = 256
    BLOCK_N = 256
    BLOCK_K = 128
    
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_N, BLOCK_K])
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N])
    
    num_tiles_m = M // BLOCK_M
    num_tiles_n = N // BLOCK_N
    num_tiles = num_tiles_m * num_tiles_n
    
    num_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
    grid = (min(num_sms, num_tiles),)
    
    _gemm_kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        M,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        STAGES=5,
        num_warps=8,
        num_stages=5,
    )