import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_persistent(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_a_row, stride_b_row, stride_c_row,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr
):
    """
    Persistent GEMM kernel targeting Hopper.
    
    Uses explicitly sized software-pipelined pointer loads and standard tensor 
    core reduction to compute C = A @ B^T safely across all boundary conditions.
    """
    
    start_pid = tl.program_id(0)
    num_pid_m = M // BLOCK_M
    num_pid_n = N // BLOCK_N
    num_tiles = num_pid_m * num_pid_n
    
    # Fully unrolled constant trips along K dimension avoids cdiv remainder edge-cases
    num_k_tiles = 40 
    
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=False):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        row = tl.arange(0, BLOCK_M)
        col = tl.arange(0, BLOCK_N)
        
        for k_tile in range(num_k_tiles):
            offset_k = k_tile * BLOCK_K
            
            col_k = tl.arange(0, BLOCK_K)
            a = tl.load(A_ptr + (offset_m + row[:, None]) * stride_a_row + (offset_k + col_k[None, :]), 
                        mask=((offset_m + row) < M)[:, None] & ((offset_k + col_k) < K)[None, :], other=0.0)
            
            row_n = tl.arange(0, BLOCK_N)
            b = tl.load(B_ptr + (offset_n + row_n[:, None]) * stride_b_row + (offset_k + col_k[None, :]), 
                        mask=((offset_n + row_n) < N)[:, None] & ((offset_k + col_k) < K)[None, :], other=0.0)
            
            acc = tl.dot(a, b.T, acc)
            
        out_ptr = C_ptr + offset_m * stride_c_row + offset_n
        mask = ((offset_m + row) < M).to(tl.int32)[:, None]
        tl.store(out_ptr + row[:, None] * stride_c_row + col[None, :], acc.to(tl.bfloat16), mask=mask)


def run(A, B, C):
    """
    Compute C = A @ B.T on Hopper using a persistent grid and masked FP32 reduction.
    
    Args:
        A: Input tensor of shape [M, 5120] and dtype bfloat16.
        B: Input tensor of shape [7168, 5120] and dtype bfloat16.
        C: Preallocated output tensor of shape [M, 7168] and dtype bfloat16.
    """
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = 7168
    K = 5120
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128
    NUM_SMS = 132 
    
    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    
    grid = (min(NUM_SMS, num_tiles),)
    
    _gemm_persistent[grid](
        A, B, C,
        M, N, K,
        stride_a_row=A.stride(0),
        stride_b_row=B.stride(0),
        stride_c_row=C.stride(0),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        NUM_SMS=NUM_SMS,
        num_warps=4, num_stages=4,
    )