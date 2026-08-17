import torch
import triton
import triton.language as tl


def alloc_fn(size: int, alignment: int, stream):
    """Allocator for device-created tensor descriptors."""
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(alloc_fn)


@triton.jit
def _gemm_persistent(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr
):
    """
    Persistent GEMM kernel targeting Hopper.
    
    Uses Hardware Transactional Memory Access (TMA) via dynamically created TensorDescriptors 
    and optimized tensor core reduction to compute C = A @ B^T.
    """
    
    a_desc = tl.make_tensor_descriptor(
        A_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K], padding_option="zero"
    )
    b_desc = tl.make_tensor_descriptor(
        B_ptr, shape=[N, K], strides=[K, 1], block_shape=[BLOCK_N, BLOCK_K], padding_option="zero"
    )
    c_desc = tl.make_tensor_descriptor(
        C_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N], padding_option="zero"
    )
    
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    
    for tile_id in range(start_pid, num_tiles, NUM_SMS):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        for k_tile in range(0, tl.cdiv(K, BLOCK_K)):
            offset_k = k_tile * BLOCK_K
            
            a = a_desc.load([offset_m, offset_k])
            b = b_desc.load([offset_n, offset_k])
            
            acc = tl.dot(a, b.T, acc)
            
        c = acc.to(tl.bfloat16)
        c_desc.store([offset_m, offset_n], c)


def run(A, B, C):
    """
    Compute C = A @ B.T on Hopper using a persistent grid and TMA.
    
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
    BLOCK_K = 256
    NUM_SMS = 132 
    
    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    
    grid = (min(NUM_SMS, num_tiles),)
    
    _gemm_persistent[grid](
        A, B, C,
        M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        NUM_SMS=NUM_SMS,
        num_warps=8, num_stages=2,
    )