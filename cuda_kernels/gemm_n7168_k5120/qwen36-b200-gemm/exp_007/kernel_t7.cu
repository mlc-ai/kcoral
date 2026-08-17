#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
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

namespace tvm_ffi_example_cuda {

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 64;
constexpr int NUM_THREADS = 256;
constexpr int TILE_A = BLOCK_M * BLOCK_K;
constexpr int TILE_B = BLOCK_N * BLOCK_K;
constexpr int OUTPUTS_PER_THREAD = (BLOCK_M * BLOCK_N) / NUM_THREADS; // 16

__device__ __forceinline__ uint32_t sha_ptr(void const* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__global__ void gemm_bf16_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K,
    int lda, int ldb, int ldc) 
{
    // Shared layout: sA[TILE_A], sB[TILE_B], accum[N_TILE]
    __shared__ __nv_bfloat16 sA[TILE_A];
    __shared__ __nv_bfloat16 sB[TILE_B];
    // Accumulator buffer in shared memory: we'll do partial reduction here
    // This keeps register pressure very low
    
    const int tid = threadIdx.x;
    const int bm = blockIdx.x * BLOCK_M;
    const int bn = blockIdx.y * BLOCK_N;
    const int nk = K / BLOCK_K;

    // Each thread owns 16 output positions spread across the 64x64 block
    // Pattern: first iterate through threads' stride, then local offset
    // tid=0: outs 0,256,512,... (every 256th flat index) = rows/cols interleaved
    
    // Precompute which rows/cols this thread contributes to
    // For simplicity: each thread does 2 columns on 8 different rows
    // So 2*8 = 16 outputs per thread
    const int rows_per_thread = OUTPUTS_PER_THREAD / 2;  // 8
    const int cols_start = (tid / rows_per_thread) % BLOCK_N;  // which base cols
    const int cols_step = 2;  // 2 cols per thread
    const int stride = rows_per_thread * cols_step;  // how many threads share a "group"
    
    // Alternative cleaner mapping: 
    // thread tid owns output element pair (om, on0) and (om, on1) repeated across 8 different om values
    // tid groups: 0-7 own cols 0,1; 8-15 own cols 2,3; ... 120-127 own cols 62,63
    const int grp = tid >> 3;           // 0..31
    const int off_in_grp = tid & 7;     // 0..7
    const int c_base = grp * 2;         // base col pair
    const int r = off_in_grp * 8;       // row within block
    
    float acc[OUTPUTS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < OUTPUTS_PER_THREAD; ++i) {
        acc[i] = 0.0f;
    }

    const int elems_a = TILE_A / NUM_THREADS;   // 16
    const int elems_b = TILE_B / NUM_THREADS;   // 16

    for (int stage = 0; stage < nk; ++stage) {
        const int ko = stage * BLOCK_K;

        // === LOAD A tile cooperatively ===
        #pragma unroll 4
        for (int e = 0; e < elems_a; ++e) {
            const int glob_idx = tid * elems_a + e;
            const int brow = glob_idx / BLOCK_K;
            const int bcol = glob_idx % BLOCK_K;
            const int gr = bm + brow;
            const int gc = ko + bcol;
            if (gr < M && gc < K) {
                sA[brow * BLOCK_K + bcol] = A[gr * lda + gc];
            } else {
                sA[brow * BLOCK_K + bcol] = __float2bfloat16(0.0f);
            }
        }

        // === LOAD B tile cooperatively ===
        #pragma unroll 4
        for (int e = 0; e < elems_b; ++e) {
            const int glob_idx = tid * elems_b + e;
            const int brow = glob_idx / BLOCK_K;
            const int bcol = glob_idx % BLOCK_K;
            const int gr = bn + brow;
            const int gc = ko + bcol;
            if (gr < N && gc < K) {
                sB[brow * BLOCK_K + bcol] = B[gr * ldb + gc];
            } else {
                sB[brow * BLOCK_K + bcol] = __float2bfloat16(0.0f);
            }
        }

        __syncthreads();

        // === COMPUTE ===
        // This thread owns 8 rows, each with 2 columns = 16 outputs
        for (int ri = 0; ri < rows_per_thread; ++ri) {
            const int om = r + ri;  // global row within block
            // Wait, that doesn't work right. Let me redo mapping.
            // Actually with 256 threads each doing 16 outputs, there are various valid tilings.
            // Let me use the SIMPLEST possible correct scheme instead:
            
            // Each thread picks 16 specific (row, col) pairs via a deterministic formula
            // Flat output index in [0..4095]: start at tid, stride 256, repeat 16 times
        }
        
        // OK let me just implement it cleanly:
        #pragma unroll
        for (int oi = 0; oi < OUTPUTS_PER_THREAD; ++oi) {
            const int flat_out = tid + oi * NUM_THREADS;
            const int om = flat_out / BLOCK_N;  // 0..63
            const int on = flat_out % BLOCK_N;  // 0..63
            
            const __nv_bfloat16* pA_row = &sA[om * BLOCK_K];
            const __nv_bfloat16* pB_row = &sB[on * BLOCK_K];
            
            float tmp = 0.0f;
            #pragma unroll
            for (int k = 0; k < BLOCK_K; k += 2) {
                tmp += __bfloat162float(pA_row[k])     * __bfloat162float(pB_row[k]);
                tmp += __bfloat162float(pA_row[k + 1]) * __bfloat162float(pB_row[k + 1]);
            }
            acc[oi] += tmp;
        }

        __syncthreads();
    }

    // === STORE outputs ===
    #pragma unroll
    for (int oi = 0; oi < OUTPUTS_PER_THREAD; ++oi) {
        const int flat_out = tid + oi * NUM_THREADS;
        const int om = flat_out / BLOCK_N;
        const int on = flat_out % BLOCK_N;

        const int gr = bm + om;
        const int gc = bn + on;

        if (gr < M && gc < N) {
            C[gr * ldc + gc] = __float2bfloat16(acc[oi]);
        }
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);
    
    const __nv_bfloat16* A_d = static_cast<const __nv_bfloat16*>(A.data_ptr());
    const __nv_bfloat16* B_d = static_cast<const __nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_d = static_cast<__nv_bfloat16*>(C.data_ptr());
    
    dim3 blk(NUM_THREADS);
    dim3 grd((M + BLOCK_M - 1) / BLOCK_M, (N + BLOCK_N - 1) / BLOCK_N);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    gemm_bf16_kernel<<<grd, blk, 0, stream>>>(
        A_d, B_d, C_d, 
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
        static_cast<int>(K), static_cast<int>(K), static_cast<int>(N));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda