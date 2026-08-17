#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <mma.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_gemm {

using namespace nvcuda;

__global__ void gemm_kernel(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, 
                           int M, int N, int K) {
    int m_start = blockIdx.x * 128;
    int n_start = blockIdx.y * 128;
    
    extern __shared__ __align__(16) char smem[];
    __nv_bfloat16 (*smem_A)[16] = (__nv_bfloat16(*)[16])smem;
    __nv_bfloat16 (*smem_B)[16] = (__nv_bfloat16(*)[16])(smem + 4096);
    __nv_bfloat16 (*smem_C)[128] = (__nv_bfloat16(*)[128])(smem + 8192);
    
    // 4 warps, each computing a 64x64 quadrant of the 128x128 output tile
    wmma::fragment<float, 64, 64, 16, __nv_bfloat16> d_A, d_B, d_C;
    wmma::fill_fragment(d_C, 0);
    
    int row_idx = (threadIdx.y / 2) * 64;
    int col_idx = (threadIdx.y % 2) * 64;
    
    for (int k = 0; k < K; k += 16) {
        int row = threadIdx.y * 32 + threadIdx.x;
        
        if (row < 128) {
            // Use float4 (16-byte) vectorized loads for efficient bandwidth utilization
            float4 a0 = *reinterpret_cast<const float4*>(&A[(m_start + row) * K + k]);
            float4 a1 = *reinterpret_cast<const float4*>(&A[(m_start + row) * K + k + 8]);
            *reinterpret_cast<float4*>(&smem_A[row][0]) = a0;
            *reinterpret_cast<float4*>(&smem_A[row][8]) = a1;
            
            float4 b0 = *reinterpret_cast<const float4*>(&B[(n_start + row) * K + k]);
            float4 b1 = *reinterpret_cast<const float4*>(&B[(n_start + row) * K + k + 8]);
            *reinterpret_cast<float4*>(&smem_B[row][0]) = b0;
            *reinterpret_cast<float4*>(&smem_B[row][8]) = b1;
        }
        
        __syncthreads();
        
        // A is K-Major (row-major), B needs to be transposed for MN-Major (col-major) consumption
        wmma::load_matrix_sync(d_A, smem_A[row_idx], 16, wmma::mem_row_major);
        wmma::load_matrix_sync(d_B, smem_B[col_idx], 16, wmma::mem_col_major);
        
        wmma::mma_sync(d_C, d_A, d_B);
        
        __syncthreads();
    }
    
    wmma::store_matrix_sync(smem_C[row_idx][col_idx], d_C, 128, wmma::mem_row_major);
    __syncthreads();
    
    // Coalesced epilogue: global store maps consecutive threads to contiguous columns within a row
    for (int i = threadIdx.y * 32 + threadIdx.x; i < 16384; i += 128) {
        int r = i / 128;
        int c = i % 128;
        if (m_start + r < M && n_start + c < N) {
            C[(m_start + r) * N + (n_start + c)] = smem_C[r][c];
        }
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    int64_t M_val = A.size(0);
    int64_t K_val = A.size(1);
    int64_t N_val = B.size(0);
    
    dim3 grid((M_val + 127) / 128, (N_val + 127) / 128);
    dim3 block(32, 4);
    
    CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 40960));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    gemm_kernel<<<grid, block, 40960, stream>>>(
        static_cast<__nv_bfloat16*>(A.data_ptr()),
        static_cast<__nv_bfloat16*>(B.data_ptr()),
        static_cast<__nv_bfloat16*>(C.data_ptr()),
        M_val, N_val, K_val
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_gemm::run);

}  // namespace tvm_ffi_gemm