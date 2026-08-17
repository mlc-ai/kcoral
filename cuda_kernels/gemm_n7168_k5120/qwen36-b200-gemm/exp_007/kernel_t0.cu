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
constexpr int WARP_SIZE = 32;
constexpr int NUM_THREADS = 128;
constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
constexpr int PIPE_DEPTH = 2;

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
__device__ __forceinline__ void mb_wait(uint64_t* bar, uint32_t parity) {
    asm volatile("{\n.reg .pred P;\nLOOP_%=:\nmbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n@!P bra LOOP_%=;\n}\n"
        : : "r"(sha_ptr(bar)), "r"(parity));
}

// Shared memory: A_tiles[PIPE][BM*BK], B_tiles[PIPE][BN*BK], then barriers
struct SharedMem {
    __nv_bfloat16 a[PIPE_DEPTH][BLOCK_M * BLOCK_K];
    __nv_bfloat16 b[PIPE_DEPTH][BLOCK_N * BLOCK_K];
    uint64_t ready[PIPE_DEPTH];
    uint64_t empty[PIPE_DEPTH];
};

__global__ void gemm_bf16_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K,
    int lda, int ldb, int ldc) 
{
    extern __shared__ char smem_char[];
    SharedMem& smem = *reinterpret_cast<SharedMem*>(smem_char);

    const int tid = threadIdx.x;
    const int bm = blockIdx.x * BLOCK_M;
    const int bn = blockIdx.y * BLOCK_N;

    // Init barriers (tid==0)
    if (tid == 0) {
        for (int s = 0; s < PIPE_DEPTH; ++s) {
            mb_init(&smem.ready[s], 0);     // Only tx-count
            mb_init(&smem.empty[s], NUM_THREADS);
        }
        mb_fence_init();
    }
    __syncthreads();

    // Seed all stages as empty
    for (int s = 0; s < PIPE_DEPTH; ++s) {
        mb_arrive(&smem.empty[s]);
    }

    const int nk = K / BLOCK_K;

    for (int stage = 0; stage < nk; ++stage) {
        const int si = stage % PIPE_DEPTH;
        const int ko = stage * BLOCK_K;

        // Wait for previous consumer to finish
        mb_wait(&smem.empty[si], si);

        // === LOAD A into smem.a[si]: shape [BLOCK_M, BLOCK_K] ===
        // A is row-major [M, K]. Tile at A[bm+brow, ko+bcol]
        // Each thread loads contiguous segments of the tile
        const int total_elems_a = BLOCK_M * BLOCK_K;
        const int elems_per_thread_a = total_elems_a / NUM_THREADS;  // 32

        for (int e = 0; e < elems_per_thread_a; ++e) {
            const int global_elem = tid * elems_per_thread_a + e;
            const int brow = global_elem / BLOCK_K;
            const int bcol = global_elem % BLOCK_K;
            
            const int g_row = bm + brow;
            const int g_col = ko + bcol;
            
            if (g_row < M && g_col < K) {
                smem.a[si][brow * BLOCK_K + bcol] = A[g_row * lda + g_col];
            } else {
                smem.a[si][brow * BLOCK_K + bcol] = __float2bfloat16(0.0f);
            }
        }

        // === LOAD B into smem.b[si]: shape [BLOCK_N, BLOCK_K] (transpose of B.T tile) ===
        // Original B is [N, K] row-major. B.T[k,n] = B[n,k].
        // We store smem.b[brow][bcol] = B[bn+brow, ko+bcol] so that 
        // in the gemm loop we can do A[brow,bkok] * B[bkok,bcoln] -> C[brow,bcoln]
        // The inner loop iterates kok, so B needs to be accessible as B[kok, bcoln]
        // Store as [BLOCK_N, BLOCK_K] = [brow, bcol]
        
        const int total_elems_b = BLOCK_N * BLOCK_K;
        const int elems_per_thread_b = total_elems_b / NUM_THREADS;  // 32

        for (int e = 0; e < elems_per_thread_b; ++e) {
            const int global_elem = tid * elems_per_thread_b + e;
            const int brow = global_elem / BLOCK_K;
            const int bcol = global_elem % BLOCK_K;
            
            const int g_row = bn + brow;
            const int g_col = ko + bcol;
            
            if (g_row < N && g_col < K) {
                smem.b[si][brow * BLOCK_K + bcol] = B[g_row * ldb + g_col];
            } else {
                smem.b[si][brow * BLOCK_K + bcol] = __float2bfloat16(0.0f);
            }
        }

        __syncthreads();  // Ensure all smem writes visible (simple sync, replace with mbarrier tx tracking for perf)

        // Signal ready (arriving with tx expectation that was implicitly satisfied by the sync above)
        // For correct mbarrier usage with manual loads: arrive with tx_bytes = 0 (already done)
        mb_arrive_tx(&smem.ready[si], 0);

        // === COMPUTE ===
        // Wait for ready
        mb_wait(&smem.ready[si], si);

        // Each thread computes subset of BM x BN output
        // 128 threads, 64 rows, 2 cols per thread
        const int my_row = tid % BLOCK_M;
        const int my_c0 = (tid / BLOCK_M) * 2;
        const int my_c1 = my_c0 + 1;

        float acc0 = 0.0f;
        float acc1 = 0.0f;

        const __nv_bfloat16* pA = smem.a[si] + my_row * BLOCK_K;

        #pragma unroll
        for (int k = 0; k < BLOCK_K; ++k) {
            float a_val = __bfloat162float(pA[k]);
            float b0_val = __bfloat162float(smem.b[si][my_c0 * BLOCK_K + k]);
            float b1_val = __bfloat162float(smem.b[si][my_c1 * BLOCK_K + k]);
            acc0 += a_val * b0_val;
            acc1 += a_val * b1_val;
        }

        const int gr = bm + my_row;
        const int gc0 = bn + my_c0;
        const int gc1 = bn + my_c1;

        if (gr < M && gc0 < N) {
            C[gr * ldc + gc0] = __float2bfloat16(acc0);
        }
        if (gr < M && gc1 < N) {
            C[gr * ldc + gc1] = __float2bfloat16(acc1);
        }

        // Release empty
        mb_arrive(&smem.empty[si]);
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
    
    size_t smem_size = PIPE_DEPTH * (BLOCK_M * BLOCK_K + BLOCK_N * BLOCK_K) * sizeof(__nv_bfloat16) 
                      + PIPE_DEPTH * 2 * sizeof(uint64_t);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    gemm_bf16_kernel<<<grd, blk, static_cast<size_t>(smem_size), stream>>>(
        A_d, B_d, C_d, 
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
        static_cast<int>(K), static_cast<int>(K), static_cast<int>(N));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda