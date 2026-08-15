#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define CUDA_CHECK(call) do {                                         \
    cudaError_t _e = (call);                                          \
    if (_e != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                cudaGetErrorString(_e), __FILE__, __LINE__);          \
        exit(1);                                                      \
    }                                                                 \
} while(0)

// ── Shared Memory Barrier Primitives ────────────────────────────────

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}

__device__ __forceinline__ void mbarrier_arrive_fn(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)) : "memory");
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase_parity) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(phase_parity) : "memory");
}

// ── TMA Async Load Wrapper ─────────────────────────────────────────

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
           "l"((uint64_t)d),
           "r"((uint32_t)__cvta_generic_to_shared(bar)),
           "r"(c0), "r"(c1) : "memory");
}

// ── Host-Side TMA Descriptor Builder ───────────────────────────────

static inline CUresult create_tma_2d_descriptor_2B(
    CUtensorMap* d, void* globalAddress,
    uint64_t gmem_inner_dim, uint64_t gmem_outer_dim,
    uint32_t smem_inner_dim, uint32_t smem_outer_dim,
    CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill)
{
    cuuint64_t globalDim[2]       = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1]   = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2]          = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2]  = {1, 1};
    return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
                                  globalAddress, globalDim, globalStrides,
                                  boxDim, elementStrides,
                                  CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                                  l2Promotion, oobFill);
}

// ── Kernel Constants ────────────────────────────────────────────────

constexpr uint32_t BLOCK_M   = 64;
constexpr uint32_t BLOCK_N   = 64;
constexpr uint32_t K_STEP    = 64;
constexpr uint32_t STAGES    = 2;

// ── TMA-GEMM Kernel ────────────────────────────────────────────────

__global__ void gemm_bf16_kernel(
    CUtensorMap desc_A,
    CUtensorMap desc_B,
    __nv_bfloat16*                    C,
    int64_t                           M,
    int64_t                           N,
    int64_t                           K)
{
    extern __shared__ __align__(128) unsigned char smem[];

    size_t elems_A = BLOCK_M * K_STEP;
    size_t elems_B = BLOCK_N * K_STEP;
    
    __nv_bfloat16* sA[STAGES];
    __nv_bfloat16* sB[STAGES];
    uint64_t*      bar[STAGES];

    __nv_bfloat16* ptr = reinterpret_cast<__nv_bfloat16*>(smem);
    for(int i=0; i<STAGES; ++i) {
        sA[i] = ptr; ptr += elems_A;
        sB[i] = ptr; ptr += elems_B;
    }
    uint64_t* bar_ptr = reinterpret_cast<uint64_t*>(ptr);
    bar[0] = bar_ptr + 0;
    bar[1] = bar_ptr + 1;

    int64_t m_block = blockIdx.y * BLOCK_M;
    int64_t n_block = blockIdx.x * BLOCK_N;

    // Guard against out-of-bounds TMA dispatches for non-multiple matrix dimensions
    if (m_block >= M || n_block >= N) return;

    // Initialize barriers: count=1, thread 0 arrives immediately so pending_arrivals=0.
    // TX-count will be managed purely by TMA issues/completions.
    if(threadIdx.x == 0) {
        init_smem_barrier_fn(bar[0], 1);
        init_smem_barrier_fn(bar[1], 1);
        mbarrier_arrive_fn(bar[0]);
        mbarrier_arrive_fn(bar[1]);
        fence_proxy_async_fn();
    }
    __syncthreads();

    int row = threadIdx.x;
    int col = threadIdx.x;
    float acc = 0.0f;

    int stage_read  = 0;
    int stage_write = 1;

    int64_t num_K_steps = (K + K_STEP - 1) / K_STEP;
    for(int64_t ks = 0; ks < num_K_steps; ++ks) {
        // Issue async loads for the NEXT stage (overlaps with compute below)
        tma_load_2d_fn(&desc_A, bar[stage_write], sA[stage_write], 
                       static_cast<int32_t>(ks * K_STEP), static_cast<int32_t>(m_block));
        tma_load_2d_fn(&desc_B, bar[stage_write], sB[stage_write],
                       static_cast<int32_t>(ks * K_STEP), static_cast<int32_t>(n_block));
        
        // Wait for PREVIOUS stage data to be fully loaded
        mbarrier_wait_fn(bar[stage_read], static_cast<uint32_t>(ks & 1));

        // Dot-product accumulation for this K-slice
#pragma unroll
        for(int k = 0; k < K_STEP; ++k) {
            acc += __bfloat162float(sA[stage_read][row * K_STEP + k]) *
                   __bfloat162float(sB[stage_read][col * K_STEP + k]);
        }
        
        // Rotate double-buffer pointers
        stage_read  ^= 1;
        stage_write ^= 1;
    }
    
    // Ensure final TMA transfer completes before epilogue
    if (num_K_steps > 0) {
        mbarrier_wait_fn(bar[stage_write ^ 1], 1);
    }

    int64_t global_row = m_block + row;
    int64_t global_col = n_block + col;
    if(global_row < M && global_col < N) {
        C[global_row * N + global_col] = __float2bfloat16(acc);
    }
}

// ── Host Runner (TVM-FFI) ──────────────────────────────────────────

namespace gemm_sm100_optimized {

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));

    int64_t M = A.size(0);
    int64_t N = B.size(0);
    int64_t K = A.size(1);

    __nv_bfloat16* dA = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* dB = static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* dC = static_cast<__nv_bfloat16*>(C.data_ptr());

    // Build TMA descriptors
    // A: shape [M, K] -> inner=K, outer=M
    CUtensorMap tma_A;
    create_tma_2d_descriptor_2B(&tma_A, dA, K, M, K_STEP, BLOCK_M,
                                CU_TENSOR_MAP_SWIZZLE_128B,
                                CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    // B: shape [N, K] -> inner=K, outer=N
    CUtensorMap tma_B;
    create_tma_2d_descriptor_2B(&tma_B, dB, K, N, K_STEP, BLOCK_N,
                                CU_TENSOR_MAP_SWIZZLE_128B,
                                CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    dim3 block(BLOCK_M);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);

    size_t stage_size = (BLOCK_M + BLOCK_N) * K_STEP * sizeof(__nv_bfloat16);
    size_t smem_bytes = STAGES * stage_size + STAGES * sizeof(uint64_t);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    gemm_bf16_kernel<<<grid, block, smem_bytes, stream>>>(tma_A, tma_B, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_sm100_optimized::run);

}  // namespace gemm_sm100_optimized