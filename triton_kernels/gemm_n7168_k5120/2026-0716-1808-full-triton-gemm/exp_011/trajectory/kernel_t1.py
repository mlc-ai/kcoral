import torch
import triton
import triton.language as tl


@triton.jit
def gemm_kernel(A, B, C, M, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    Persistent GEMM pass for computing C = A @ B.T.
    A has shape [M, 5120], B has shape [7168, 5120], C has shape [M, 7168].
    Utilizes a single dimension cyclic schedule mapping multiple tiles to the same warp group.
    """
    start_pid = tl.program_id(0)
    tile_stride = tl.num_programs(0)
    
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(7168, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    
    for tile_id in range(start_pid, num_tiles, tile_stride):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        for k in range(0, 5120, BLOCK_K):
            row_offsets = tl.arange(0, BLOCK_M)
            col_offsets_k = tl.arange(0, BLOCK_K)
            A_ptr = A + (offset_m + row_offsets)[:, None] * 5120 + (k + col_offsets_k)[None, :]
            a_tile = tl.load(A_ptr, mask=(offset_m + row_offsets)[:, None] < M, other=0.0)
            
            row_offsets_n = tl.arange(0, BLOCK_N)
            B_ptr = B + (offset_n + row_offsets_n)[:, None] * 5120 + (k + col_offsets_k)[None, :]
            b_tile = tl.load(B_ptr, mask=(offset_n + row_offsets_n)[:, None] < 7168, other=0.0)
            
            acc = tl.dot(a_tile, b_tile.T, acc)
        
        row_offsets = tl.arange(0, BLOCK_M)
        col_offsets = tl.arange(0, BLOCK_N)
        out_ptr = C + (offset_m + row_offsets)[:, None] * 7168 + (offset_n + col_offsets)[None, :]
        tl.store(out_ptr, acc.to(tl.bfloat16), mask=((offset_m + row_offsets)[:, None] < M) & ((offset_n + col_offsets)[None, :] < 7168))


def run(A, B, C):
    """
    Compute C = A @ B.T over the entire tensors using a persistently scheduled strategy 
    mapped optimally over Hopper SM resources for the static problem topology.
    """
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    # Limit grid to hardware SM count (132 for Hopper) to enable persistent scheduling
    grid = (min(132, triton.cdiv(M, 128) * triton.cdiv(7168, 128)),)
    
    gemm_kernel[grid](
        A, B, C, M,
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
        num_warps=4, num_stages=3
    )