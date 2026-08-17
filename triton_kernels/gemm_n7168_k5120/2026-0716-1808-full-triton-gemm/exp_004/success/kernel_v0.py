import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _gemm(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Computes C = A @ B.T using TMA descriptors.
    Utilizes grouped tile ordering to increase L2 residency probability.
    """
    
    # Decode flat tile ID into logically grouped coordinates
    tile_id = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    GROUP_M = 4
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    
    pid_in_group = tile_id % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    # Iterating straightforwardly over the entire feature width blocks.
    # Relying on underlying descriptor bounds guarantees safe zero-padding 
    # seamlessly resolves boundary conditions implicitly during load fetch.
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    for k_iter in range(num_k_tiles):
        offset_k = k_iter * BLOCK_K
        a = a_desc.load([offset_m, offset_k])
        b = b_desc.load([offset_n, offset_k])
        
        # B acts conceptually as [BLOCK_N, BLOCK_K] internally, 
        # transposing locally aligns it structurally matching standard expectations.
        acc = tl.dot(a, b.T, acc)
        
    # Safely cast accumulated precision down to output format boundaries.
    out_val = acc.to(tl.bfloat16)
    c_desc.store([offset_m, offset_n], out_val)


def run(A, B, C):
    """
    Efficient destination-passing wrapper launching our TMA GEMM. 
    
    Takes preallocated tensors explicitly matching definition sequence constraints.
    """
    torch.cuda.set_device(A.device)
    
    M, K = A.shape
    N, _ = B.shape
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128
    
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_N, BLOCK_K])
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N])
    
    num_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (num_tiles,)
    
    _gemm[grid](
        a_desc, b_desc, c_desc,
        M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=8,
    )