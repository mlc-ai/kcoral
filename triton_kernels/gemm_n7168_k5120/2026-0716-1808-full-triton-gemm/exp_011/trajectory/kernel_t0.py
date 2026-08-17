import torch
import triton
import triton.language as tl


@triton.jit
def gemm_kernel(A, B, C, M, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    Fused GEMM pass for computing C = A @ B.T.
    - A has shape [M, 5120]
    - B has shape [7168, 5120]
    - C has shape [M, 7168]
    We traverse the full K=5120 hidden states exactly once per program instance.
    """
    # Define our coordinate space in the output C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterating across chunks of the 5120 hidden features (K dimension)
    for k in range(0, 5120, BLOCK_K):
        # A's stride on its second dimension is simply 1, making it fully contiguous
        A_ptr = A + row_offsets[:, None] * 5120 + k + tl.arange(0, BLOCK_K)[None, :]
        a_tile = tl.load(A_ptr, mask=(row_offsets[:, None] < M), other=0.0)
        
        # Identically, B's inner dimension is contiguous allowing seamless tiling
        B_ptr = B + col_offsets[:, None] * 5120 + k + tl.arange(0, BLOCK_K)[None, :]
        b_tile = tl.load(B_ptr)
        
        # Core Tensor Core calculation mapping effectively forming A @ B.T chunk-by-chunk
        acc = tl.dot(a_tile, b_tile.T, acc)

    out_ptr = C + row_offsets[:, None] * 7168 + col_offsets[None, :]
    tl.store(out_ptr, acc.to(tl.bfloat16), mask=(row_offsets[:, None] < M))


def run(A, B, C):
    """
    Compute C = A @ B.T over the entire tensors using a coarsely tiled 2-pass strategy 
    mapped optimally over SM resources for the static problem topology (Qwen3 14B qkv_proj).
    """
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    num_tiles_m = triton.cdiv(M, 128)
    num_tiles_n = 7 # Fixed aspect: 7168 / 1024 = 7 tiles spanning N
    
    grid = (num_tiles_m, num_tiles_n)
    
    gemm_kernel[grid](
        A, B, C, M,
        BLOCK_M=128, BLOCK_N=1024, BLOCK_K=1024
    )