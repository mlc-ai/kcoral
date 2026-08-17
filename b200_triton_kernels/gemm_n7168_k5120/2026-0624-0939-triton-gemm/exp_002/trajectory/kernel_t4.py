import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "num_warps": 8, "num_stages": 2, "USE_DESC": True}),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "num_warps": 8, "num_stages": 4, "USE_DESC": True}),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _gemm_kernel(
    a_desc,
    b_desc,
    c_desc,
    ptr_a,
    ptr_b,
    ptr_c,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_DESC: tl.constexpr,
):
    @triton.jit
    def next_l(ptr, mask, a_desc, b_desc, m_offset, k_offset, BLOCK_M, BLOCK_K):
        if a_desc is not None:
            return a_desc.load([m_offset, k_offset])
        else:
            return tl.load(ptr, mask=mask, other=0.0)

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    k = (pid_m * BLOCK_M) % K
    m_offset = (k // BLOCK_M) * num_pid_m + (pid_m % BLOCK_M)
    n_offset = pid_n * BLOCK_N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    if USE_DESC:
        a_evens = next_l(None, None, a_desc, None, m_offset, 0, BLOCK_M, BLOCK_K)
        b_evens = next_l(None, None, None, b_desc, n_offset, 0, BLOCK_N, BLOCK_K)
        a_odds = tl.zeros((BLOCK_M, BLOCK_K), tl.bfloat16)
        b_odds = tl.zeros((BLOCK_N, BLOCK_K), tl.bfloat16)
        
        even_phase = True
        for step in range(K // BLOCK_K):
            if even_phase:
                next_a = ptr_a + ((m_start + offs_m)[:, None] * stride_am) + ((k + step + 1) * offs_k[None, :] * stride_ak)
                next_b = ptr_b + ((n_start + offs_n)[:, None] * stride_bn) + ((k + step + 1) * offs_k[None, :] * stride_bk)
            else:
                next_a = ptr_a + ((m_start + offs_m)[:, None] * stride_am) + (((k + step) % len_k) * offs_k[None, :] * stride_ak)
                next_b = ptr_b + ((n_start + offs_n)[:, None] * stride_bn) + (((k + step) % len_k) * offs_k[None, :] * stride_bk)
            
            a_in = a_evens if even_phase else a_odds
            b_in = b_evens if even_phase else b_odds
            
            if k + step < K:
                acc = tl.dot(a_in, b_in.T, acc)
                
                even_phase ^= (k + step + 1 >= K)
        
        c_desc.store([m_offset, n_offset], acc.to(tl.bfloat16))
    else:
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        mask_m = (m_start + offs_m) < M
        len_k = K // BLOCK_K
        
        a_evens = next_l(ptr_a + m_start*stride_am + 0*offs_k[None,:]*stride_ak, mask_m[:, None], None, None, m_start, 0, BLOCK_M, BLOCK_K)
        b_evens = next_l(ptr_b + n_start*stride_bn + 0*offs_k[None,:]*stride_bk, mask_m[:, None], None, None, n_start, 0, BLOCK_N, BLOCK_K)
        a_odds = tl.zeros((BLOCK_M, BLOCK_K), tl.bfloat16)
        b_odds = tl.zeros((BLOCK_N, BLOCK_K), tl.bfloat16)
        
        even_phase = True
        for step in range(len_k):
            if even_phase:
                next_a = ptr_a + ((m_start + offs_m)[:, None] * stride_am) + ((k + step + 1) * offs_k[None, :] * stride_ak)
                next_b = ptr_b + ((n_start + offs_n)[:, None] * stride_bn) + ((k + step + 1) * offs_k[None, :] * stride_bk)
            else:
                next_a = ptr_a + ((m_start + offs_m)[:, None] * stride_am) + (((k + step) % len_k) * offs_k[None, :] * stride_ak)
                next_b = ptr_b + ((n_start + offs_n)[:, None] * stride_bn) + (((k + step) % len_k) * offs_k[None, :] * stride_bk)
            
            a_in = a_evens if even_phase else a_odds
            b_in = b_evens if even_phase else b_odds
            
            if k + step < K:
                acc = tl.dot(a_in, b_in.T, acc)
                
                even_phase ^= (k + step + 1 >= K)
        
        out = acc.to(tl.bfloat16)
        c_ptr = ptr_c + (m_start + offs_m)[:, None] * stride_cm + (n_start + offs_n)[None, :] * stride_cn
        tl.store(c_ptr, out, mask=mask_m[:, None])


def run(A, B, C):
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    a_desc = TensorDescriptor.from_tensor(A, [64, 32])
    b_desc = TensorDescriptor.from_tensor(B, [64, 32])
    c_desc = TensorDescriptor.from_tensor(C, [64, 64])
    
    num_pid_m = triton.cdiv(M, 64)
    num_pid_n = triton.cdiv(N, 64)
    
    grid = lambda META: (num_pid_m, num_pid_n)
    
    if M <= 0 or N <= 0 or K <= 0:
        return

    USE_DESC = True
    
    if USE_DESC:
        _gemm_kernel[grid](
            a_desc, b_desc, c_desc,
            None, None, None,
            M, N, K,
            0, 0, 0, 0, 0, 0,
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            USE_DESC=USE_DESC
        )
    else:
        stride_am, stride_ak = A.stride(0), A.stride(1)
        stride_bk, stride_bn = B.stride(0), B.stride(1)
        stride_cm, stride_cn = C.stride(0), C.stride(1)
        
        _gemm_kernel[grid](
            None, None, None,
            A, B, C,
            M, N, K,
            stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            USE_DESC=USE_DESC
        )