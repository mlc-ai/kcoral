#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <stdio.h>
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

namespace tvm_ffi_gemm_cuda {

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 16;
constexpr int NUM_THREADS = 256;

__global__ void gemm_kernel(
    const __nv_bfloat16* __restrict__ A, 
    const __nv_bfloat16* __restrict__ B, 
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K) {
    
    __shared__ __nv_bfloat16 A_smem[2][BM][BK];
    __shared__ __nv_bfloat16 B_smem[2][BN][BK];
    __shared__ float C_smem[BM][BN];
    
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int warp_m = warp_id % 2;
    int warp_n = warp_id / 2;
    
    using namespace nvcuda::wmma;
    fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a_frag[4];
    fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> b_frag[2];
    fragment<accumulator, 16, 16, 16, float> c_frag[4][2];
    
    #pragma unroll
    for (int m = 0; m < 4; m++) {
        #pragma unroll
        for (int n = 0; n < 2; n++) {
            fill_fragment(c_frag[m][n], 0.0f);
        }
    }
    
    // Load first tile
    int buf = 0;
    for (int i = tid; i < BM * BK; i += NUM_THREADS) {
        int row = i / BK, col = i % BK;
        int gm_row = blockIdx.x * BM + row;
        int gm_col = col;
        A_smem[buf][row][col] = (gm_row < M) ? A[(uint64_t)gm_row * K + gm_col] : __float2bfloat16(0.0f);
    }
    for (int i = tid; i < BN * BK; i += NUM_THREADS) {
        int row = i / BK, col = i % BK;
        int gm_row = blockIdx.y * BN + row;
        int gm_col = col;
        B_smem[buf][row][col] = (gm_row < N) ? B[(uint64_t)gm_row * K + gm_col] : __float2bfloat16(0.0f);
    }
    __syncthreads();
    
    for (int k = 0; k < K; k += BK) {
        // Load next tile if not last
        int next_buf = 1 - buf;
        if (k + BK < K) {
            for (int i = tid; i < BM * BK; i += NUM_THREADS) {
                int row = i / BK, col = i % BK;
                int gm_row = blockIdx.x * BM + row;
                int gm_col = k + BK + col;
                A_smem[next_buf][row][col] = (gm_row < M) ? A[(uint64_t)gm_row * K + gm_col] : __float2bfloat16(0.0f);
            }
            for (int i = tid; i < BN * BK; i += NUM_THREADS) {
                int row = i / BK, col = i % BK;
                int gm_row = blockIdx.y * BN + row;
                int gm_col = k + BK + col;
                B_smem[next_buf][row][col] = (gm_row < N) ? B[(uint64_t)gm_row * K + gm_col] : __float2bfloat16(0.0f);
            }
        }
        
        // Compute on current buffer
        #pragma unroll
        for (int m = 0; m < 4; m++) {
            load_matrix_sync(a_frag[m], &A_smem[buf][warp_m * 64 + m * 16][0], BK);
        }
        #pragma unroll
        for (int n = 0; n < 2; n++) {
            load_matrix_sync(b_frag[n], &B_smem[buf][warp_n * 32 + n * 16][0], BK);
        }
        #pragma unroll
        for (int m = 0; m < 4; m++) {
            #pragma unroll
            for (int n = 0; n < 2; n++) {
                mma_sync(c_frag[m][n], a_frag[m], b_frag[n], c_frag[m][n]);
            }
        }
        
        __syncthreads();
        buf = next_buf;
        if (k + BK < K) __syncthreads();
    }
    
    // Store accumulators
    #pragma unroll
    for (int m = 0; m < 4; m++) {
        #pragma unroll
        for (int n = 0; n < 2; n++) {
            store_matrix_sync(&C_smem[warp_m * 64 + m * 16][warp_n * 32 + n * 16],
                c_frag[m][n], BN, mem_row_major);
        }
    }
    __syncthreads();
    
    // Convert and write to global memory
    for (int i = tid; i < BM * BN; i += NUM_THREADS) {
        int row = i / BN;
        int col = i % BN;
        int gm_row = blockIdx.x * BM + row;
        int gn_col = blockIdx.y * BN + col;
        if (gm_row < M && gn_col < N) {
            C[(uint64_t)gm_row * N + gn_col] = __float2bfloat16(C_smem[row][col]);
        }
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);
    
    const __nv_bfloat16* A_ptr = static_cast<const __nv_bfloat16*>(A.data_ptr());
    const __nv_bfloat16* B_ptr = static_cast<const __nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());
    
    dim3 block(NUM_THREADS);
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    gemm_kernel<<<grid, block, 0, stream>>>(A_ptr, B_ptr, C_ptr, (int)M, (int)N, (int)K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_gemm_cuda::run);

}  // namespace tvm_ffi_gemm_cuda