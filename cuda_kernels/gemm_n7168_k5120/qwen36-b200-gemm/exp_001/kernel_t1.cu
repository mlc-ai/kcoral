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

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
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
    // c0 -> inner dim (K), c1 -> outer dim (M or N)
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

    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        l2Promotion, oobFill);
}

// ── Kernel Constants ────────────────────────────────────────────────

constexpr uint32_t BLOCK_M   = 128;
constexpr uint32_t BLOCK_N   = 128;
constexpr uint32_t K_STEP    =  64;
constexpr uint32_t STAGES    =   2;

// ── TMA-GEMM Kernel ────────────────────────────────────────────────

__global__ void gemm_bf16_kernel(
    const __grid_constant__ CUtensorMap desc_A,
    const __grid_constant__ CUtensorMap desc_B,
    __nv_bfloat16*                    C,
    int64_t                           M,
    int64_t                           N,
    int64_t                           K)
{
    extern __shared__ __align__(128) unsigned char smem[];

    // Split shared memory: Stage 0, Stage 1 for A and B, then barriers
    __nv_bfloat16 (*smem_A)[K_STEP] = reinterpret_cast<__nv_bfloat16(*)[K_STEP]>(smem);
    size_t stage_bytes = BLOCK_M * K_STEP * sizeof(__nv_bfloat16);
    __nv_bfloat16 (*smem_B)[K_STEP] = reinterpret_cast<__nv_bfloat16(*)[K_STEP]>(smem + STAGES * stage_bytes);
    uint64_t*      bars             = reinterpret_cast<uint64_t*>(
                                           smem + STAGES * (stage_bytes + BLOCK_N * K_STEP * sizeof(__nv_bfloat16)));

    // Grid geometry
    int64_t m_block = blockIdx.y * BLOCK_M;
    int64_t n_block = blockIdx.x * BLOCK_N;

    // Initialize mbarriers (only thread 0)
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bars + 0, 1);
        init_smem_barrier_fn(bars + 1, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    int64_t num_K_steps = (K + K_STEP - 1) / K_STEP;
    int     pstage_prev = 1;
    int     pstage_next = 0;

    // Each thread owns exactly one (row, col) pair in the BM x BN tile
    int row = threadIdx.x;      // 0 .. 127
    int col = threadIdx.x;      // 0 .. 127
    float acc = 0.0f;

    for (int64_t ks = 0; ks < num_K_steps; ++ks) {
        // Issue TMA loads for NEXT stage (overlaps with compute)
        tma_load_2d_fn(&desc_A, bars + pstage_next,
                       smem_A[pstage_next],
                       (int32_t)(ks * K_STEP),
                       (int32_t)m_block);
        tma_load_2d_fn(&desc_B, bars + pstage_next,
                       smem_B[pstage_next],
                       (int32_t)(ks * K_STEP),
                       (int32_t)n_block);

        // Synchronize: wait for PREVIOUS stage data to be ready
        mbarrier_wait_fn(bars + pstage_prev, /* parity */ 0);

        // ── Matmul fragment ───────────────────────────────────────
        const __nv_bfloat16* sa = &smem_A[pstage_prev][0][0];
        const __nv_bfloat16* sb = &smem_B[pstage_prev][0][0];

        // Coalesced reads: within a warp, threads access consecutive K indices
        for (int k = 0; k < K_STEP; ++k) {
            float val_a = __bfloat162float(sa[row * K_STEP + k]);
            float val_b = __bfloat162float(sb[col * K_STEP + k]);
            acc += val_a * val_b;
        }

        // Swap double-buffer pointers
        pstage_prev ^= 1;
        pstage_next ^= 1;
    }

    // Final wait to ensure last stage completed before writing output
    mbarrier_wait_fn(bars + ((num_K_steps % 2 == 0) ? 0 : 1), /* parity */ 0);

    // Epilogue: write accumulated result to global memory
    int64_t global_row = m_block + row;
    int64_t global_col = n_block + col;
    if (global_row < M && global_col < N) {
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
    // A: [M, K] -> tensor viewed as [K, M] for contiguous K-stride loads
    CUtensorMap tma_A;
    create_tma_2d_descriptor_2B(&tma_A, dA, K, M,
                                K_STEP, BLOCK_M,
                                CU_TENSOR_MAP_SWIZZLE_128B,
                                CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    // B: [N, K] -> tensor viewed as [K, N]
    CUtensorMap tma_B;
    create_tma_2d_descriptor_2B(&tma_B, dB, K, N,
                                K_STEP, BLOCK_N,
                                CU_TENSOR_MAP_SWIZZLE_128B,
                                CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    // Launch config
    dim3 block(BLOCK_M);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);

    size_t stage_size = (BLOCK_M + BLOCK_N) * K_STEP * sizeof(__nv_bfloat16);
    size_t smem_bytes = STAGES * stage_size + STAGES * sizeof(uint64_t);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    cudaLaunchConfig_t cfg{};
    cfg.gridDim   = grid;
    cfg.blockDim  = block;
    cfg.dynamicSmemBytes = smem_bytes;
    cfg.stream    = stream;

    cudaLaunchAttribute attr;
    attr.id        = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = 1;
    cfg.attrs      = &attr;
    cfg.numAttrs   = 1;

    CUDA_CHECK(cudaLaunchKernelEx(&cfg, gemm_bf16_kernel,
                                  tma_A, tma_B, dC, M, N, K));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_sm100_optimized::run);

}  // namespace gemm_sm100_optimized