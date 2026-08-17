import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import triton.tma as tma


@triton.jit
def wait(barrier, expected_phase_0):
    tma.commit_group()
    tma.wait_group()
    phase = tma.readBarrier(barrier)
    while phase != expected_phase_0[0]:
        expected_phase_0[0] = 1 - expected_phase_0[0]
        tl.wait(1)
        phase = tma.readBarrier(barrier)


@triton.jit
def _gemm_kernel(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    shm_a_ptr,
    shm_b_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    __launch_bounds__(max_threads_per_block=128)
    
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    a_desc_0 = tma.build_descriptor(
        shm_a_ptr + 0 * BLOCK_M * BLOCK_K,
        shape=[BLOCK_M, BLOCK_K],
        strides=[16, BLOCK_M * 16],
        col_major=True,
    )
    a_desc_1 = tma.build_descriptor(
        shm_a_ptr + 1 * BLOCK_M * BLOCK_K,
        shape=[BLOCK_M, BLOCK_K],
        strides=[16, BLOCK_M * 16],
        col_major=True,
    )
    
    b_desc_0 = tma.build_descriptor(
        shm_b_ptr + 0 * BLOCK_N * BLOCK_K,
        shape=[BLOCK_N, BLOCK_K],
        strides=[16, BLOCK_N * 16],
        col_major=True,
    )
    b_desc_1 = tma.build_descriptor(
        shm_b_ptr + 1 * BLOCK_N * BLOCK_K,
        shape=[BLOCK_N, BLOCK_K],
        strides=[16, BLOCK_N * 16],
        col_major=True,
    )
    
    barrier_0 = tma.create_barrier(initial_value=0, phase=0)
    barrier_1 = tma.create_barrier(initial_value=0, phase=0)
    
    expected_phase_0 = [0, 0]
    
    tma.issue_tma_async(a_desc_0, a_desc, [offset_m, 0])
    tma.signalBarrier(barrier_0)
    tma.issue_tma_async(b_desc_0, b_desc, [offset_n, 0])
    tma.signalBarrier(barrier_0)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    num_k_tiles = K // BLOCK_K
    
    for k_tile in range(num_k_tiles):
        next_k = k_tile + 1
        buf_idx = k_tile % 2
        next_buf_idx = 1 - buf_idx
        
        if next_k < num_k_tiles:
            next_offset_k = next_k * BLOCK_K
            next_a_desc = a_desc_0 if next_buf_idx == 0 else a_desc_1
            next_b_desc = b_desc_0 if next_buf_idx == 0 else b_desc_1
            next_barrier = barrier_0 if next_buf_idx == 0 else barrier_1
            
            tma.issue_tma_async(next_a_desc, a_desc, [offset_m, next_offset_k])
            tma.signalBarrier(next_barrier)
            tma.issue_tma_async(next_b_desc, b_desc, [offset_n, next_offset_k])
            tma.signalBarrier(next_barrier)
            
        curr_barrier = barrier_0 if buf_idx == 0 else barrier_1
        wait(curr_barrier, expected_phase_0)
        
        a_buf = tma.load(a_desc_0 if buf_idx == 0 else a_desc_1)
        b_buf = tma.load(b_desc_0 if buf_idx == 0 else b_desc_1)
        
        acc += tl.dot(a_buf, b_buf.T)
    
    acc_bf16 = acc.to(tl.bfloat16)
    c_desc.store([offset_m, offset_n], acc_bf16)


def run(A, B, C):
    """Perform destination-passing GEMM computation C = A @ B.T."""
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 160
    
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_N, BLOCK_K])
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N])
    
    shm_a = torch.empty((2, BLOCK_M, BLOCK_K), dtype=A.dtype, device=A.device)
    shm_b = torch.empty((2, BLOCK_N, BLOCK_K), dtype=B.dtype, device=B.device)
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    _gemm_kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        M,
        N,
        K,
        shm_a_ptr=shm_a.__cuda_array_interface__["data"][0],
        shm_b_ptr=shm_b.__cuda_array_interface__["data"][0],
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=8,
        num_stages=4,
    )