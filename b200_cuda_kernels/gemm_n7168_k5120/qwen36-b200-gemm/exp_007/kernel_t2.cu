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
constexpr int NUM_THREADS = 128;
constexpr int PIPE_DEPTH = 2;
constexpr int TILE_A = BLOCK_M * BLOCK_K; // elements per A tile
constexpr int TILE_B = BLOCK_N * BLOCK_K; // elements per B tile
constexpr int TILE_BYTES = (TILE_A + TILE_B) * sizeof(__nv_bfloat16);
constexpr int BARRIER_SPACE = PIPE_DEPTH * 2 * sizeof(uint64_t);

// ---- mbarrier helpers ----
__device__ __forceinline__ uint32_t sha_ptr(void const* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mb_init(uint64_t* bar, uint32_t cnt) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" : : "r"(sha_ptr(bar)), "r"(cnt));
}
__device__ __forceinline__ void mb_fence_init() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
__device__ __forceinline__ void mb_arrive_tx(uint64_t* bar, uint32_t tx) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        : : "r"(sha_ptr(bar)), "r"(tx) : "memory");
}
__device__ __forceinline__ void mb_arrive(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" : : "r"(sha_ptr(bar)) : "memory");
}
__device__ __forceinline__ void mb_wait_parity(uint64_t* bar, uint32_t parity) {
    asm volatile("{\n.reg .pred P;\nLOOP_%=:\nmbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n@!P bra LOOP_%=;\n}\n"
        : : "r"(sha_ptr(bar)), "r"(parity));
}

__global__ void gemm_bf16_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K,
    int lda, int ldb, int ldc) 
{
    extern __shared__ __nv_bfloat16 smem[];

    const int tid = threadIdx.x;
    const int bm = blockIdx.x * BLOCK_M;
    const int bn = blockIdx.y * BLOCK_N;

    // Barrier locations
    __nv_bfloat16* smem_barrier_base = smem + PIPE_DEPTH * (TILE_A + TILE_B);
    uint64_t* barriers = reinterpret_cast<uint64_t*>(smem_barrier_base);

    // Init barriers (tid==0 only)
    if (tid == 0) {
        for (int s = 0; s < PIPE_DEPTH; ++s) {
            mb_init(&barriers[s], 0);                    // ready: tx-count only
            mb_init(&barriers[PIPE_DEPTH + s], NUM_THREADS); // empty: thread arrivals
        }
        mb_fence_init();
    }
    __syncthreads();

    // Seed all stages as empty
    for (int s = 0; s < PIPE_DEPTH; ++s) {
        mb_arrive(&barriers[PIPE_DEPTH + s]);
    }

    const int nk = K / BLOCK_K;

    for (int stage = 0; stage < nk; ++stage) {
        const int si = stage & 1; // mod 2 for PIPE_DEPTH=2
        const int ko = stage * BLOCK_K;

        // Offsets into shared memory for this stage
        int base_offset = si * (TILE_A + TILE_B);
        __nv_bfloat16* tileA = smem + base_offset;
        __nv_bfloat16* tileB = smem + base_offset + TILE_A;

        // Wait for consumer to release this stage
        mb_wait_parity(&barriers[PIPE_DEPTH + si], si);

        // === LOAD A tile: tileA[brow*BLOCK_K + bcol] = A[bm+brow, ko+bcol] ===
        const int elems_per_thread_a = TILE_A / NUM_THREADS; // 32

        #pragma unroll
        for (int e = 0; e < elems_per_thread_a; ++e) {
            const int glob_idx = tid * elems_per_thread_a + e;
            const int brow = glob_idx / BLOCK_K;
            const int bcol = glob_idx % BLOCK_K;
            
            const int gr = bm + brow;
            const int gc = ko + bcol;
            
            if (gr < M && gc < K) {
                tileA[brow * BLOCK_K + bcol] = A[gr * lda + gc];
            } else {
                tileA[brow * BLOCK_K + bcol] = __float2bfloat16(0.0f);
            }
        }

        // === LOAD B tile: tileB[brow*BLOCK_K + bcol] = B[bn+brow, ko+bcol] ===
        const int elems_per_thread_b = TILE_B / NUM_THREADS; // 32

        #pragma unroll
        for (int e = 0; e < elems_per_thread_b; ++e) {
            const int glob_idx = tid * elems_per_thread_b + e;
            const int brow = glob_idx / BLOCK_K;
            const int bcol = glob_idx % BLOCK_K;
            
            const int gr = bn + brow;
            const int gc = ko + bcol;
            
            if (gr < N && gc < K) {
                tileB[brow * BLOCK_K + bcol] = B[gr * ldb + gc];
            } else {
                tileB[brow * BLOCK_K + bcol] = __float2bfloat16(0.0f);
            }
        }

        __syncthreads();

        // Signal data ready
        mb_arrive_tx(&barriers[si], 0);

        // === COMPUTE ===
        mb_wait_parity(&barriers[si], si);

        // Each thread computes 0.5 rows x 2 cols of output
        const int my_row = tid % BLOCK_M;
        const int my_col0 = (tid / BLOCK_M) * 2;
        const int my_col1 = my_col0 + 1;

        float acc0 = 0.0f;
        float acc1 = 0.0f;

        const __nv_bfloat16* pA = tileA + my_row * BLOCK_K;
        const __nv_bfloat16* pB0 = tileB + my_col0 * BLOCK_K;
        const __nv_bfloat16* pB1 = tileB + my_col1 * BLOCK_K;

        #pragma unroll
        for (int k = 0; k < BLOCK_K; ++k) {
            float a_val = __bfloat162float(pA[k]);
            float b0_val = __bfloat162float(pB0[k]);
            float b1_val = __bfloat162float(pB1[k]);
            acc0 += a_val * b0_val;
            acc1 += a_val * b1_val;
        }

        const int gr = bm + my_row;
        const int gc0 = bn + my_col0;
        const int gc1 = bn + my_col1;

        if (gr < M && gc0 < N) {
            C[gr * ldc + gc0] = __float2bfloat16(acc0);
        }
        if (gr < M && gc1 < N) {
            C[gr * ldc + gc1] = __float2bfloat16(acc1);
        }

        // Release this stage for next iteration
        mb_arrive(&barriers[PIPE_DEPTH + si]);
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
    
    size_t smem_elems = PIPE_DEPTH * (TILE_A + TILE_B) + BARRIER_SPACE / sizeof(__nv_bfloat16);
    size_t smem_bytes = smem_elems * sizeof(__nv_bfloat16);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    gemm_bf16_kernel<<<grd, blk, static_cast<size_t>(smem_bytes), stream>>>(
        A_d, B_d, C_d, 
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
        static_cast<int>(K), static_cast<int>(K), static_cast<int>(N));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda