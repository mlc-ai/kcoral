import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def persistent_gemm(
    a_desc, b_desc, c_desc,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    start_pid = tl.program_id(0)
    tile_stride = tl.num_programs(0)
    
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    
    for tile_id in range(start_pid, num_tiles, tile_stride):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        
        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        for k_tile in range(num_k_tiles):
            offset_k = k_tile * BLOCK_K
            a = a_desc.load([offset_m, offset_k])
            b = b_desc.load([offset_k, offset_n])
            acc = tl.dot(a, b, acc)
        
        out = acc.to(tl.bfloat16)
        c_desc.store([offset_m, offset_n], out)


def run(A, B, C):
    """Compute C = A @ B.T into a preallocated CUDA tensor."""
    torch.cuda.set_device(A.device)
    
    M, K = A.shape
    N = B.shape[0]
    
    assert K == 5120
    assert N == 7168
    
    B = B.permute(1, 0)  # View B as [K, N] matching the logical B.T operand perfectly
    
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 128
    
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_K, BLOCK_N])
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N])
    
    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    
    NUM_SMS = 132
    grid = (min(NUM_SMS, num_tiles),)
    
    persistent_gemm[grid](
        a_desc, b_desc, c_desc, M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=4,
    )