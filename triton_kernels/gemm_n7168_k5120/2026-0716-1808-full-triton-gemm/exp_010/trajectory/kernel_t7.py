import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _gemm_kernel(
    a_desc,
    b_desc,
    C,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    __launch_bounds__(128, 1)
    
    # Because the layout is known to be perfectly contiguous, 
    # we can treat C's strides as constexpr constants.
    stride_C_m = N 
    stride_C_n = 1
    
    start_pid = tl.program_id(0)
    tile_stride = tl.num_programs(0)
    
    num_tiles_m = M // BLOCK_M
    num_tiles_n = N // BLOCK_N
    
    for tile_id in range(start_pid, num_tiles_m * num_tiles_n, tile_stride):
        pid_m = tile_id // num_tiles_n
        pid_n = tile_id % num_tiles_n
        
        row_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        col_idx = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        for k_tile in range(K // BLOCK_K):
            offset_k = k_tile * BLOCK_K
            
            # Asynchronously fetch contiguous blocks utilizing TMA hardware.
            a = a_desc.load([pid_m * BLOCK_M, offset_k])
            b = b_desc.load([pid_n * BLOCK_N, offset_k])
            
            acc += tl.dot(a, b.T, acc)
            
        c_store = acc.to(tl.bfloat16)
        ptr = C + row_idx[:, None] * stride_C_m + col_idx[None, :] * stride_C_n
        valid_row = row_idx[:, None] < M
        valid_col = col_idx[None, :] < N
        tl.store(ptr, c_store, mask=valid_row & valid_col)


def run(A, B, C):
    """Perform destination-passing GEMM computation C = A @ B.T."""
    torch.cuda.set_device(A.device)
    
    M, K = A.shape
    N = B.shape[0]
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 256
    
    # Utilize Hopper Tensor Descriptors for optimized TMA movement.
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_N, BLOCK_K])
    
    num_tiles_m = M // BLOCK_M
    num_tiles_n = N // BLOCK_N
    num_tiles = num_tiles_m * num_tiles_n
    
    num_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
    grid = (min(num_sms, num_tiles),)
    
    _gemm_kernel[grid](
        a_desc,
        b_desc,
        C,
        M,
        N,
        K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )