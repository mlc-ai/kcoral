#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace gemm_blackwell {

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;

extern "C" __global__ void gemm_kernel(
    const __grid_constant__ CUtensorMap tma_a_desc,
    const __grid_constant__ CUtensorMap tma_b_desc,
    __nv_bfloat16* A,
    __nv_bfloat16* B,
    __nv_bfloat16* C,
    int M, int N, int K)
{
    extern __shared__ char smem_raw[];
    
    // Aligned shared memory for mbarrier
    uint64_t* mbar = reinterpret_cast<uint64_t*>(
        reinterpret_cast<char*>(smem_raw) + ((sizeof(uint64_t) + 15) & ~15));
    __nv_bfloat16* A_smem = reinterpret_cast<__nv_bfloat16*>(
        reinterpret_cast<char*>(smem_raw) + ((sizeof(uint64_t) + 15) & ~15) + 64);
    __nv_bfloat16* B_smem = reinterpret_cast<__nv_bfloat16*>(
        A_smem + BM * BK);
    
    // Phase 0: Initialize mbarrier (only thread 0)
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], 0;" 
                     :: "r"(static_cast<unsigned>(__cvta_generic_to_shared(mbar))));
    }
    __syncthreads();
    
    int tid = threadIdx.x;
    int m_start = static_cast<int>(blockIdx.y) * BM;
    int n_start = static_cast<int>(blockIdx.x) * BN;
    
    // Number of rows/cols this thread owns
    int num_rows = 4;
    int row_stride = 32; // 256/8 = 32 threads share rows
    
    // Compute base row and column for this thread
    int group = tid / 32;          // 0..7 (which group of 32 threads)
    int idx_in_group = tid % 32;   // 0..31 within group
    
    // Each thread owns 8 rows: group*8 .. group*8+7
    int row_base = group * 8;
    // Column assignment within the BN block
    int col = idx_in_group;
    
    float accum[8] = {0.0f};
    
    for (int k_block = 0; k_block < K; k_block += BK) {
        // Check bounds for k_block
        int k_eff = k_block;
        int bk_eff = BK;
        
        // Initiate TMA load for A: rows [m_start, m_start+BM), cols [k_eff, k_eff+bk_eff)
        if (tid == 0 && m_start < M) {
            int coord0 = 0; // relative row offset in the box
            int coord1 = k_eff; // absolute column in global A
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%2, %3}], [%4];"
                :: "r"(static_cast<unsigned>(__cvta_generic_to_shared(A_smem))),
                   "l"(reinterpret_cast<uint64_t>(&tma_a_desc)),
                   "r"(coord0), "r"(coord1),
                   "r"(static_cast<unsigned>(__cvta_generic_to_shared(mbar)))
                : "memory");
        }
        
        // Initiate TMA load for B: rows [n_start, n_start+BN), cols [k_eff, k_eff+bk_eff)
        if (tid == 0 && n_start < N) {
            int coord0 = 0;
            int coord1 = k_eff;
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%2, %3}], [%4];"
                :: "r"(static_cast<unsigned>(__cvta_generic_to_shared(B_smem))),
                   "l"(reinterpret_cast<uint64_t>(&tma_b_desc)),
                   "r"(coord0), "r"(coord1),
                   "r"(static_cast<unsigned>(__cvta_generic_to_shared(mbar)))
                : "memory");
        }
        
        __syncthreads();
        
        // Wait for mbarrier (both TMA loads complete)
        if (m_start < M || n_start < N) {
            uint32_t bar_addr = static_cast<unsigned>(__cvta_generic_to_shared(mbar));
            asm volatile(
                "{\n"
                ".reg .pred p;\n"
                "1:\n"
                "mbarrier.try_wait.parity.shared.b64 p, [%0], 0;\n"
                "@!p bra 1;\n"
                "}\n"
                :: "r"(bar_addr));
        }
        __syncthreads();
        
        // Perform GEMM fragment: each thread computes its 8 row contributions
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            int row = row_base + r;
            if (row >= BM) continue;
            
            if (m_start + row < M && n_start + col < N) {
                #pragma unroll
                for (int k_ = 0; k_ < BK; ++k_) {
                    accum[r] += static_cast<float>(A_smem[(row) * BK + k_].x) 
                              * static_cast<float>(B_smem[(col) * BK + k_].x);
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store results
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        int row = row_base + r;
        if (row >= BM) continue;
        
        int m_global = m_start + row;
        int n_global = n_start + col;
        
        if (m_global < M && n_global < N) {
            C[static_cast<long long>(m_global) * N + n_global] = 
                __float2bfloat16(accum[r]);
        }
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id()));
    
    int M = static_cast<int>(A.size(0));
    int N = 7168;
    int K = 5120;
    
    __nv_bfloat16* d_A = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* d_B = static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* d_C = static_cast<__nv_bfloat16*>(C.data_ptr());
    
    // Grid dimensions
    int grid_x = (N + BN - 1) / BN;
    int grid_y = (M + BM - 1) / BM;
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(256, 1, 1);
    
    // Shared memory: mbarrier(8) + padding + A_smem(32KB) + B_smem(32KB)
    int smem_size = 64 + BM * BK * sizeof(__nv_bfloat16) + BN * BK * sizeof(__nv_bfloat16);
    smem_size = (smem_size + 255) & ~255; // Align to 256 bytes
    
    // Create TMA descriptors
    // For A: shape [M, K], load [BM, BK] tiles
    CUtensorMap tma_a;
    cuuint64_t globalDimA[2] = {static_cast<cuuint64_t>(K), static_cast<cuuint64_t>(M)};
    cuuint64_t globalStrideA[1] = {static_cast<cuuint64_t>(K) * sizeof(__nv_bfloat16)};
    cuuint32_t boxDimA[2] = {BK, BM};
    cuuint32_t elemStrideA[2] = {1, 1};
    CUresult ret_a = cuTensorMapEncodeTiled(
        &tma_a,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        d_A,
        globalDimA, globalStrideA,
        boxDimA, elemStrideA,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (ret_a != CUDA_SUCCESS) {
        const char* err_str;
        cuGetErrorName(ret_a, &err_str);
        fprintf(stderr, "TMA A encode failed: %s\n", err_str);
    }
    
    // For B: shape [N, K], load [BN, BK] tiles
    CUtensorMap tma_b;
    cuuint64_t globalDimB[2] = {static_cast<cuuint64_t>(K), static_cast<cuuint64_t>(N)};
    cuuint64_t globalStrideB[1] = {static_cast<cuuint64_t>(K) * sizeof(__nv_bfloat16)};
    cuuint32_t boxDimB[2] = {BK, BN};
    cuuint32_t elemStrideB[2] = {1, 1};
    CUresult ret_b = cuTensorMapEncodeTiled(
        &tma_b,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        d_B,
        globalDimB, globalStrideB,
        boxDimB, elemStrideB,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (ret_b != CUDA_SUCCESS) {
        const char* err_str;
        cuGetErrorName(ret_b, &err_str);
        fprintf(stderr, "TMA B encode failed: %s\n", err_str);
    }
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id()));
    
    // Launch with cluster configuration for Blackwell
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.sharedMemBytes = smem_size;
    config.stream = stream;
    
    // Set cluster dimension (2 CTAs per cluster for potential cta_group::2 ops)
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    auto launch_fn = [](void* fn, cudaLaunchConfig_t cfg, void** args) {
        cudaLaunchKernelEx(&cfg, reinterpret_cast<void(*)(void)>(fn), args[0], args[1], 
                          args[2], args[3], args[4], args[5], args[6], args[7]);
    };
    
    void* args[] = {
        (void*)&tma_a, (void*)&tma_b, 
        (void*)d_A, (void*)d_B, (void*)d_C,
        (void*)(uintptr_t)M, (void*)(uintptr_t)N, (void*)(uintptr_t)K
    };
    
    // Direct kernel launch (simpler approach)
    gemm_kernel<<<grid, block, smem_size, stream>>>(
        tma_a, tma_b, d_A, d_B, d_C, M, N, K);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace gemm_blackwell

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_blackwell::run);