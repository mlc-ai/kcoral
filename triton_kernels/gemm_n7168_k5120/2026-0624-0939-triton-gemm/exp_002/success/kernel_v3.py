import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _gemm_kernel_device(
    ptr_a, ptr_b, ptr_c,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    dtype = ptr_c.dtype.element_ty
    
    a_desc = tl.make_tensor_descriptor(
        ptr_a, shape=[M, K], strides=[K, 1],
        block_shape=[BLOCK_M, BLOCK_K], padding_option="zero")
    
    b_desc = tl.make_tensor_descriptor(
        ptr_b, shape=[N, K], strides=[K, 1],
        block_shape=[BLOCK_N, BLOCK_K], padding_option="zero")
    
    c_desc = tl.make_tensor_descriptor(
        ptr_c, shape=[M, N], strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N])

    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n

    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        flatten=False,
        warp_specialize=WARP_SPECIALIZE,
    ):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        num_k_tiles = tl.cdiv(K, BLOCK_K)
        
        for k_tile in range(num_k_tiles):
            offset_k = k_tile * BLOCK_K
            a = a_desc.load([offset_m, offset_k])
            b = b_desc.load([offset_n, offset_k])
            acc = tl.dot(a, b.T, acc)
            
        c_desc.store([offset_m, offset_n], acc.to(dtype))


@triton.jit
def _gemm_kernel_host(
    a_desc, b_desc, c_desc,
    M, N, K,
    STRIDE_BK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_offset = pid_m * BLOCK_M
    n_offset = pid_n * BLOCK_N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    
    for k_tile in range(num_k_tiles):
        a = a_desc.load([m_offset, k_tile * BLOCK_K])
        b = b_desc.load([n_offset, k_tile * BLOCK_K])
        acc = tl.dot(a, b.T, acc)
        
    c_desc.store([m_offset, n_offset], acc.to(tl.bfloat16))


@triton.jit
def _gemm_kernel_ptr(
    ptr_a, ptr_b, ptr_c,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_offset = pid_m * BLOCK_M
    n_offset = pid_n * BLOCK_N
    
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    
    for k_tile in range(num_k_tiles):
        offs_k = tl.arange(0, BLOCK_K)
        a_ptr = ptr_a + (m_offset + offs_m)[:, None] * stride_am + (k_tile * BLOCK_K + offs_k)[None, :] * stride_ak
        b_ptr = ptr_b + (n_offset + offs_n)[:, None] * stride_bk + (k_tile * BLOCK_K + offs_k)[None, :] * stride_bn
        
        a_tile = tl.load(a_ptr, mask=(m_offset + offs_m)[:, None] < M & (offs_k[None, :] < K), other=0.0)
        b_tile = tl.load(b_ptr, mask=(offs_k[:, None] < K), other=0.0)
        
        acc = tl.dot(a_tile, b_tile.T, acc)
        
    c_ptr = ptr_c + (m_offset + offs_m)[:, None] * stride_cm + (n_offset + offs_n)[None, :] * stride_cn
    out = acc.to(tl.bfloat16)
    tl.store(c_ptr, out, mask=(m_offset + offs_m)[:, None] < M)


def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)


def run(A, B, C):
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 64
    
    b_contiguous = B.stride(0) == K and B.stride(1) == 1
    
    if b_contiguous:
        NUM_SMS = 132
        num_pid_m = triton.cdiv(M, BLOCK_M)
        num_pid_n = triton.cdiv(N, BLOCK_N)
        num_tiles = num_pid_m * num_pid_n
        grid = (min(NUM_SMS, num_tiles),)
        
        _gemm_kernel_device[grid](
            A, B, C, M, N, K,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            NUM_SMS=NUM_SMS, WARP_SPECIALIZE=True,
            num_warps=8, num_stages=3,
        )
    else:
        a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
        b_desc = TensorDescriptor.from_tensor(B, [BLOCK_N, BLOCK_K])
        c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N])
        
        num_pid_m = triton.cdiv(M, BLOCK_M)
        num_pid_n = triton.cdiv(N, BLOCK_N)
        grid = (num_pid_m, num_pid_n)
        
        _gemm_kernel_host[grid](
            a_desc, b_desc, c_desc, M, N, K,
            STRIDE_BK=1,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )