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

__global__ void gemm_bf16_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K,
    int lda, int ldb, int ldc) 
{
    extern __shared__ char smem_char[];

    // Layout: for each stage s in [0, PIPE_DEPTH):
    //   A_tile[s]:     [s * TILE_BYTES, s * TILE_BYTES + BM*BK*2)
    //   B_tile[s]:     [s * TILE_BYTES + BM*BK*2, (s+1) * TILE_BYTES - 16)
    //   ready[s]:      [(PIPE_DEPTH * TILE_BYTES) + s*16]
    //   empty[s]:      [(PIPE_DEPTH * TILE_BYTES) + (PIPE_DEPTH+s)*8]
    
    const int tile_bytes = (BLOCK_M * BLOCK_K + BLOCK_N * BLOCK_K) * sizeof(__nv_bfloat16);
    const int barrier_base = PIPE_DEPTH * tile_bytes;
    
    __nv_bfloat16* sa[PIPE_DEPTH];
    __nv_bfloat16* sb[PIPE_DEPTH];
    uint64_t* bar_ready[PIPE_DEPTH];
    uint64_t* bar_empty[PIPE_DEPTH];

    for (int s = 0; s < PIPE_DEPTH; ++s) {
        sa[s] = reinterpret_cast<__nv_bfloat16*>(smem_char + s * tile_bytes);
        sb[s] = reinterpret_cast<__nv_bfloat16*>(smem_char + s * tile_bytes + BLOCK_M * BLOCK_K * sizeof(__nv_bfloat16));
        bar_ready[s] = reinterpret_cast<uint64_t*>(smem_char + barrier_base + s * 16);
        bar_empty[s] = reinterpret_cast<uint64_t*>(smem_char + barrier_base + PIPE_DEPTH * 8 + s * 16);
    }

    const int tid = threadIdx.x;
    const int bm = blockIdx.x * BLOCK_M;
    const int bn = blockIdx.y * BLOCK_N;

    // Init barriers (tid==0)
    if (tid == 0) {
        for (int s = 0; s < PIPE_DEPTH; ++s) {
            mb_init(bar_ready[s], 0);         // Only tx-count tracked
            mb_init(bar_empty[s], NUM_THREADS); // Thread arrivals only
        }
        mb_fence_init();
    }
    __syncthreads();

    // Seed all stages as empty (producer can start loading)
    for (int s = 0; s < PIPE_DEPTH; ++s) {
        mb_arrive(bar_empty[s]);
    }

    const int nk = K / BLOCK_K;

    for (int stage = 0; stage < nk; ++stage) {
        const int si = stage % PIPE_DEPTH;
        const int ko = stage * BLOCK_K;

        // Wait for previous consumer to finish this stage
        mb_wait(bar_empty[si], si & 1);

        // === LOAD A tile into sa[si]: shape [BLOCK_M, BLOCK_K] ===
        // Each thread loads contiguous elements from the global A matrix
        const int total_elems_a = BLOCK_M * BLOCK_K;
        const int elems_per_thread_a = total_elems_a / NUM_THREADS; // 32

        for (int e = 0; e < elems_per_thread_a; ++e) {
            const int glob_idx = tid * elems_per_thread_a + e;
            const int brow = glob_idx / BLOCK_K;
            const int bcol = glob_idx % BLOCK_K;
            
            const int gr = bm + brow;
            const int gc = ko + bcol;
            
            if (gr < M && gc < K) {
                sa[si][brow * BLOCK_K + bcol] = A[gr * lda + gc];
            } else {
                sa[si][brow * BLOCK_K + bcol] = __float2bfloat16(0.0f);
            }
        }

        // === LOAD B tile into sb[si]: shape [BLOCK_N, BLOCK_K] ===
        // B is [N, K] row-major. We store B[bn+brow, ko+bcol] at sb[brow*BLOCK_K+bcol].
        // In GEMM: C[r,c] += sum_k(A[r,k] * B.T[k,c]) = sum_k(A[r,k] * B[c,k])
        // So inner loop over k accesses A[r,k] and B[c,k]. With our layout:
        //   A stored as [BLOCK_M, BLOCK_K]: A_sm[row*k_dim]
        //   B stored as [BLOCK_N, BLOCK_K]: B_sm[col*k_dim]
        
        const int total_elems_b = BLOCK_N * BLOCK_K;
        const int elems_per_thread_b = total_elems_b / NUM_THREADS; // 32

        for (int e = 0; e < elems_per_thread_b; ++e) {
            const int glob_idx = tid * elems_per_thread_b + e;
            const int brow = glob_idx / BLOCK_K;
            const int bcol = glob_idx % BLOCK_K;
            
            const int gr = bn + brow;
            const int gc = ko + bcol;
            
            if (gr < N && gc < K) {
                sb[si][brow * BLOCK_K + bcol] = B[gr * ldb + gc];
            } else {
                sb[si][brow * BLOCK_K + bcol] = __float2bfloat16(0.0f);
            }
        }

        __syncthreads();

        // Signal data ready (no async ops, so tx=0, just thread arrival)
        mb_arrive_tx(bar_ready[si], 0);

        // === COMPUTE phase ===
        // Wait for producer to finish
        mb_wait(bar_ready[si], si & 1);

        // Each thread computes subset of BLOCK_M x BLOCK_N output
        // 128 threads: each gets (BLOCK_M/128)*BLOCK_N = 0.5 * 64 = 32 outputs... 
        // Actually: 64 rows / 128 threads = 0.5 rows per thread -> pair threads share a row
        // tid%64 gives the row, tid/64 gives which of 2 cols
        
        const int my_row = tid % BLOCK_M;          // 0..63
        const int my_col_offset = (tid / BLOCK_M) * 2; // 0 or 2
        
        float acc0 = 0.0f;
        float acc1 = 0.0f;

        const __nv_bfloat16* pA_row = sa[si] + my_row * BLOCK_K;
        const __nv_bfloat16* pB_c0 = sb[si] + my_col_offset * BLOCK_K;
        const __nv_bfloat16* pB_c1 = sb[si] + (my_col_offset + 1) * BLOCK_K;

        #pragma unroll
        for (int k = 0; k < BLOCK_K; ++k) {
            float a_val = __bfloat162float(pA_row[k]);
            float b0_val = __bfloat162float(pB_c0[k]);
            float b1_val = __bfloat162float(pB_c1[k]);
            acc0 += a_val * b0_val;
            acc1 += a_val * b1_val;
        }

        const int gr = bm + my_row;
        const int gc0 = bn + my_col_offset;
        const int gc1 = bn + my_col_offset + 1;

        if (gr < M && gc0 < N) {
            C[gr * ldc + gc0] = __float2bfloat16(acc0);
        }
        if (gr < M && gc1 < N) {
            C[gr * ldc + gc1] = __float2bfloat16(acc1);
        }

        // Release this stage for next producer iteration
        mb_arrive(bar_empty[si]);
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
                      + (PIPE_DEPTH * 2) * sizeof(uint64_t);
    
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