import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _persistent_gemm_kernel(
    a_desc, b_desc, c_desc,
    M, N, K,
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_N: tl.constexpr,
    transpose_A: tl.constexpr,
    transpose_B: tl.constexpr,
):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=False):
        # Explicitly resolve our (BLOCK_M, BLOCK_N) destination tile
        # We iterate M fastest to allow cheaper inner-loop reuse of B blocks.
        num_pid_in_group = num_pid_m * GROUP_N
        group_id = tile_id // num_pid_in_group
        first_pid_n = group_id * GROUP_N
        group_size_n = min(num_pid_n - first_pid_n, GROUP_N)
        pid_in_group = tile_id % num_pid_in_group
        pid_m = pid_in_group % num_pid_m
        pid_n = first_pid_n + (pid_in_group // num_pid_m) % group_size_n
        
        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        # Iterate over the contracted K dimension resolving TMA descriptors directly.
        num_k_tiles = tl.cdiv(K, BLOCK_K)
        for k_tile in range(num_k_tiles):
            offset_k = k_tile * BLOCK_K
            
            if transpose_A:
                a = a_desc.load([offset_k, offset_m])
                a = a.T
            else:
                a = a_desc.load([offset_m, offset_k])
                
            if transpose_B:
                b = b_desc.load([offset_k, offset_n])
            else:
                b = b_desc.load([offset_n, offset_k])
                b = b.T
                
            acc = tl.dot(a, b, acc)
            
        c_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))


def run(A, B, C):
    """Compute ``C = A @ B.T`` into a preallocated CUDA tensor."""
    torch.cuda.set_device(A.device)
    
    M, K = A.shape
    N = B.shape[0]
    
    block_m = 128
    block_n = 128
    block_k = 128
    
    NUM_SMS = 132
    GROUP_N = 8
    
    # Occasionally frameworks hand us transposed inputs. Descriptors strictly require their leading strides to be 16-byte aligned, which contiguous `[M, K]` guarantees, but transposed `[M, K]` (with stride 1 along M) breaks. Flip them safely.
    if A.stride(0) == 1:
        A = A.transpose(0, 1)
        transpose_A = True
    else:
        transpose_A = False
        
    if B.stride(0) == 1:
        B = B.transpose(0, 1)
        transpose_B = True
    else:
        transpose_B = False
    
    if transpose_A:
        a_desc = TensorDescriptor.from_tensor(A, [block_k, block_m])
    else:
        a_desc = TensorDescriptor.from_tensor(A, [block_m, block_k])
        
    if transpose_B:
        b_desc = TensorDescriptor.from_tensor(B, [block_k, block_n])
    else:
        b_desc = TensorDescriptor.from_tensor(B, [block_n, block_k])
        
    c_desc = TensorDescriptor.from_tensor(C, [block_m, block_n])
    
    num_pid_m = triton.cdiv(M, block_m)
    num_pid_n = triton.cdiv(N, block_n)
    num_tiles = num_pid_m * num_pid_n
    
    grid = (min(NUM_SMS, num_tiles),)
    
    _persistent_gemm_kernel[grid](
        a_desc, b_desc, c_desc, M, N, K,
        NUM_SMS=NUM_SMS,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        GROUP_N=GROUP_N,
        transpose_A=transpose_A,
        transpose_B=transpose_B,
        num_warps=8,
        num_stages=4,
    )