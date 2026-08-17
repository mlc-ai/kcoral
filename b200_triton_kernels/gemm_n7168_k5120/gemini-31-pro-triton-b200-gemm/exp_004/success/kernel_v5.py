import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

@triton.jit
def _grouped_tile_coordinates(
    tile_id, num_pid_m, num_pid_n, GROUP_M: tl.constexpr,
):
    """L2-cache aware grouped tile mapping to enhance reuse of operand B."""
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)

    pid_in_group = tile_id % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m
    return pid_m, pid_n


@triton.jit
def _gemm_tma_kernel_host(
    a_desc, b_desc, c_desc,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, WARP_SPEC: tl.constexpr, LOOP_STAGES: tl.constexpr
):
    tile_id = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    pid_m, pid_n = _grouped_tile_coordinates(tile_id, num_pid_m, num_pid_n, GROUP_M)
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    
    # Loop over K; apply warp specialization and descriptor-load software pipelining
    # Both num_stages=LOOP_STAGES and warp_specialize=WARP_SPEC are required loop intrinsics here for Blackwell
    for k0 in tl.range(0, num_k_tiles, num_stages=LOOP_STAGES, warp_specialize=WARP_SPEC):
        a = a_desc.load([offset_m, k0 * BLOCK_K])
        b = b_desc.load([offset_n, k0 * BLOCK_K])
        
        # Multiply loaded block [BLOCK_M, BLOCK_K] with explicitly transposed B block [BLOCK_K, BLOCK_N]
        # This matches Blackwell's physical fp8/bf16 ideal contiguous block packing layout expectations natively.
        acc = tl.dot(a, b.T, acc)
        
    acc = acc.to(tl.bfloat16)
    c_desc.store([offset_m, offset_n], acc)


ptr_configs = [
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
]

@triton.autotune(configs=ptr_configs, key=['M', 'N', 'K'])
@triton.jit
def _gemm_ptr_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr
):
    """Fallback kernel mapping for arbitrary unaligned boundary geometries."""
    tile_id = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    pid_m, pid_n = _grouped_tile_coordinates(tile_id, num_pid_m, num_pid_n, GROUP_M)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_ptrs = B_ptr + (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        k = k0 * BLOCK_K + offs_k
        
        a = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0)
        b = tl.load(B_ptrs, mask=(offs_n[:, None] < N) & (k[None, :] < K), other=0.0)
        
        acc = tl.dot(a, b.T, acc)
        
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk
        
    acc = acc.to(tl.bfloat16)
    
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(C_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def is_tma_supported(t: torch.Tensor) -> bool:
    """Verifies that the tensor physical memory layout meets strict TMA alignment invariants."""
    if t.ndim != 2: return False
    if t.stride(1) != 1: return False
    if (t.stride(0) * t.element_size()) % 16 != 0: return False
    if t.data_ptr() % 16 != 0: return False
    return True


_best_tma_config = None

def run(A, B, C):
    """Compute C = A @ B.T into a preallocated CUDA tensor."""
    global _best_tma_config
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N, _ = B.shape
    
    use_tma = is_tma_supported(A) and is_tma_supported(B) and is_tma_supported(C)
    
    if use_tma:
        if _best_tma_config is None:
            # Custom lightweight autotuner bypasses `@triton.autotune` descriptor cache type inference limits.
            # Grid permutations structured as:
            # (BLOCK_M, BLOCK_N, BLOCK_K, warps, launch_stages, loop_stages, warp_specialize, group_m, num_ctas)
            configs = [
                # Top math-to-memory ratios maximizing ILP via large tile volumes
                (256, 128, 128, 8, 3, 3, True, 8, 1),
                (128, 256, 128, 8, 3, 3, True, 8, 1),
                (256, 128, 128, 8, 3, 3, True, 16, 1),
                
                # Deeper SMEM software pipelines mapped for latency-hiding
                (256, 128, 64,  8, 4, 4, True, 8, 1),
                (128, 256, 64,  8, 4, 4, True, 8, 1),
                
                # Clustering permutations to augment L2 hit-rate behavior
                (256, 128, 128, 8, 3, 3, True, 8, 2),
                (128, 256, 128, 8, 3, 3, True, 8, 2),
                
                # Symmetric block boundaries
                (128, 128, 128, 8, 3, 3, True, 8, 1),
                (128, 128, 128, 8, 4, 4, True, 8, 1),
                
                # Fallback comparisons sans automated Warp Specialization logic
                (256, 128, 128, 8, 3, 3, False, 8, 1),
                (128, 256, 128, 8, 3, 3, False, 8, 1),
            ]
            
            best_time = float('inf')
            best_cfg = configs[0]
            
            for cfg in configs:
                BM, BN, BK, warps, launch_stages, loop_stages, warp_spec, group_m, ctas = cfg
                try:
                    a_desc = TensorDescriptor.from_tensor(A, [BM, BK])
                    b_desc = TensorDescriptor.from_tensor(B, [BN, BK])
                    c_desc = TensorDescriptor.from_tensor(C, [BM, BN])
                    
                    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
                    
                    # Warmup compilation & execute
                    _gemm_tma_kernel_host[grid](
                        a_desc, b_desc, c_desc,
                        M, N, K,
                        BM, BN, BK,
                        group_m, warp_spec, loop_stages,
                        num_warps=warps, num_stages=launch_stages, num_ctas=ctas
                    )
                    
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(3):
                        _gemm_tma_kernel_host[grid](
                            a_desc, b_desc, c_desc,
                            M, N, K,
                            BM, BN, BK,
                            group_m, warp_spec, loop_stages,
                            num_warps=warps, num_stages=launch_stages, num_ctas=ctas
                        )
                    end.record()
                    torch.cuda.synchronize()
                    t = start.elapsed_time(end)
                    
                    if t < best_time:
                        best_time = t
                        best_cfg = cfg
                except Exception:
                    # Ignore kernels that exceed SMEM budgets inherently limited to configs
                    pass
                    
            _best_tma_config = best_cfg
            
        BM, BN, BK, warps, launch_stages, loop_stages, warp_spec, group_m, ctas = _best_tma_config
        
        a_desc = TensorDescriptor.from_tensor(A, [BM, BK])
        b_desc = TensorDescriptor.from_tensor(B, [BN, BK])
        c_desc = TensorDescriptor.from_tensor(C, [BM, BN])
        
        grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
        _gemm_tma_kernel_host[grid](
            a_desc, b_desc, c_desc,
            M, N, K,
            BM, BN, BK,
            group_m, warp_spec, loop_stages,
            num_warps=warps, num_stages=launch_stages, num_ctas=ctas
        )
    else:
        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
        _gemm_ptr_kernel[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1)
        )