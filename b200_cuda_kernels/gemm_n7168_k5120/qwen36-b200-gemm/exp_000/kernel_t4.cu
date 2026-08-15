#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_RT_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                          \
    if (_e != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                  \
                cudaGetErrorString(_e), __FILE__, __LINE__);          \
        exit(1);                                                      \
    }                                                                 \
} while(0)

#define CUDA_DRV_CHECK(call) do {                                     \
    CUresult _r = (call);                                             \
    if (_r != CUDA_SUCCESS) {                                         \
        const char *_msg;                                             \
        cuGetErrorName(_r, &_msg);                                    \
        fprintf(stderr, "CUDA Driver error %s at %s:%d\n",           \
                _msg, __FILE__, __LINE__);                            \
        exit(1);                                                      \
    }                                                                 \
} while(0)

namespace tvm_ffi_gemm_blackwell {

CUresult create_tma_2d_descriptor(
    CUtensorMap* d, void* globalAddress,
    uint64_t gmem_inner_dim, uint64_t gmem_outer_dim,
    uint32_t smem_inner_dim, uint32_t smem_outer_dim,
    CUtensorMapDataType dataType,
    CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, dataType, 2, globalAddress,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill);
}

constexpr uint32_t BLOCK_M = 64;
constexpr uint32_t BLOCK_N = 128;
constexpr uint32_t BLOCK_K = 16;
constexpr uint32_t NUM_THREADS = 128;

constexpr uint32_t SMEM_A_BYTES = BLOCK_M * BLOCK_K * 2;
constexpr uint32_t SMEM_B_BYTES = BLOCK_N * BLOCK_K * 2;
// Barrier takes 8 bytes, plus padding to 128-byte alignment
constexpr uint32_t TOTAL_SMEM_BYTES = 128 + SMEM_A_BYTES + SMEM_B_BYTES;

__device__ __forceinline__ void init_barrier(uint32_t addr, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(addr), "r"(count));
}

__device__ __forceinline__ void fence_barrier_init() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void barrier_wait(uint32_t addr, uint32_t parity) {
    asm volatile(
        ".reg .pred p_done;\n\t"
        "BARRIER_WAIT_%=: \n\t"
        "mbarrier.try_wait.parity.shared.b64 p_done, [%0], %1;\n\t"
        "@!p_done bra BARRIER_WAIT_%=;\n\t"
        : : "r"(addr), "r"(parity) : "memory");
}

__device__ __forceinline__ void tma_load(const CUtensorMap* desc, uint32_t bar_addr,
    uint32_t dst_addr, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%3, %4}], [%2];"
        : : "r"(dst_addr), "l"(desc), "r"(bar_addr), "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ uint64_t build_smem_desc(uint32_t addr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    d |= ((addr & 0x3FFFF) >> 4);
    d |= ((uint64_t)((lbo & 0x3FFFF) >> 4)) << 16;
    d |= ((uint64_t)((sbo & 0x3FFFF) >> 4)) << 32;
    d |= 1ULL << 46;  // version
    d |= 2ULL << 61;  // 128B swizzle
    return d;
}

__device__ __forceinline__ uint32_t build_idesc(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= 1u << 4;     // dtype FP32
    d |= 1u << 7;     // atype BF16
    d |= 1u << 10;    // btype BF16
    d |= ((N >> 3) & 0x3Fu) << 17;
    d |= ((M >> 4) & 0x1Fu) << 24;
    return d;
}

__global__ void gemm_kernel(const __nv_bfloat16* __restrict__ A,
                             const __nv_bfloat16* __restrict__ B,
                             __nv_bfloat16* __restrict__ C,
                             const __grid_constant__ CUtensorMap tma_a,
                             const __grid_constant__ CUtensorMap tma_b,
                             uint32_t M, uint32_t N, uint32_t K) {

    extern __shared__ unsigned char smem[];

    uint32_t base = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t bar_addr = base;
    uint32_t smem_a_addr = base + 128;
    uint32_t smem_b_addr = base + 128 + SMEM_A_BYTES;

    if (threadIdx.x == 0) {
        init_barrier(bar_addr, 2);
        fence_barrier_init();
    }
    __syncthreads();

    uint32_t bx = blockIdx.x;
    uint32_t n_blks = (N + BLOCK_N - 1) / BLOCK_N;
    uint32_t mb = bx / n_blks;
    uint32_t nb = bx % n_blks;

    uint32_t a_off = mb * BLOCK_M;
    uint32_t b_off = nb * BLOCK_N;
    uint32_t k_iters = K / BLOCK_K;

    // Descriptor setup
    uint32_t sbo = 1024;
    uint64_t desc_a = build_smem_desc(smem_a_addr, (BLOCK_M / 8) * sbo, sbo);
    uint64_t desc_b = build_smem_desc(smem_b_addr, (BLOCK_N / 8) * sbo, sbo);
    uint32_t idesc = build_idesc(BLOCK_M, BLOCK_N);

    // Allocate TMEM
    uint32_t tmem_c;
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 : "=r"(tmem_c) : "r"(BLOCK_N));

    // Clear accumulator
    asm volatile("{\n\t"
        ".reg .pred p_clear;\n\t"
        "setp.eq.b32 p_clear, 0, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p_clear;\n\t"
        "}" : : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc));
    
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                 : : "r"(bar_addr));
    if (threadIdx.x == 0) { barrier_wait(bar_addr, 0); }
    __syncthreads();

    // Prefetch TMA
    if (threadIdx.x == 0) {
        asm volatile("prefetch.tensormap [%0];" : : "l"(&tma_a));
        asm volatile("prefetch.tensormap [%0];" : : "l"(&tma_b));
    }

    // K-loop
    for (uint32_t ki = 0; ki < k_iters; ki++) {
        uint32_t ko = ki * BLOCK_K;

        tma_load(&tma_a, bar_addr, smem_a_addr, static_cast<int32_t>(ko), static_cast<int32_t>(a_off));
        tma_load(&tma_b, bar_addr, smem_b_addr, static_cast<int32_t>(ko), static_cast<int32_t>(b_off));

        if (threadIdx.x == 0) { barrier_wait(bar_addr, 0); }
        __syncthreads();

        asm volatile("{\n\t"
            ".reg .pred p_acc;\n\t"
            "setp.ne.b32 p_acc, 1, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p_acc;\n\t"
            "}" : : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc));

        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                     : : "r"(bar_addr));

        if (threadIdx.x == 0) { barrier_wait(bar_addr, 0); }
        __syncthreads();
    }

    // Epilogue
    uint32_t tid = threadIdx.x;
    uint32_t row = tid;
    if (row < BLOCK_M) {
        uint32_t gr = a_off + row;
        if (gr < M) {
            for (uint32_t col = 0; col < BLOCK_N; col += 4) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                    : "r"(static_cast<uint32_t>(col)));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

                float f0 = __uint_as_float(r0);
                float f1 = __uint_as_float(r1);
                float f2 = __uint_as_float(r2);
                float f3 = __uint_as_float(r3);

                uint32_t nc = b_off + col;
                __nv_bfloat16* out = C + (uint64_t)gr * N + nc;
                if (nc + 3 < N) {
                    out[0] = __float2bfloat16(f0);
                    out[1] = __float2bfloat16(f1);
                    out[2] = __float2bfloat16(f2);
                    out[3] = __float2bfloat16(f3);
                } else {
                    if (nc < N) out[0] = __float2bfloat16(f0);
                    if (nc + 1 < N) out[1] = __float2bfloat16(f1);
                    if (nc + 2 < N) out[2] = __float2bfloat16(f2);
                }
            }
        }
    }

    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 : : "r"(tmem_c), "r"(BLOCK_N));
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_RT_CHECK(cudaSetDevice(A.device().device_id));

    uint32_t M = static_cast<uint32_t>(A.size(0));
    uint32_t N = 7168u;
    uint32_t K = 5120u;

    const __nv_bfloat16* A_data = static_cast<const __nv_bfloat16*>(A.data_ptr());
    const __nv_bfloat16* B_data = static_cast<const __nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_data = static_cast<__nv_bfloat16*>(C.data_ptr());

    CUtensorMap h_tma_a, h_tma_b;
    CUDA_DRV_CHECK(create_tma_2d_descriptor(
        &h_tma_a, const_cast<void*>(static_cast<const void*>(A_data)),
        K, M, BLOCK_K, BLOCK_M,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    CUDA_DRV_CHECK(create_tma_2d_descriptor(
        &h_tma_b, const_cast<void*>(static_cast<const void*>(B_data)),
        K, N, BLOCK_K, BLOCK_N,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    CUtensorMap *d_ta = nullptr, *d_tb = nullptr;
    CUDA_DRV_CHECK(cuMemAlloc(reinterpret_cast<CUdeviceptr*>(&d_ta), sizeof(CUtensorMap)));
    CUDA_DRV_CHECK(cuMemAlloc(reinterpret_cast<CUdeviceptr*>(&d_tb), sizeof(CUtensorMap)));

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    CUDA_DRV_CHECK(cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(d_ta), &h_tma_a, sizeof(CUtensorMap), stream));
    CUDA_DRV_CHECK(cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(d_tb), &h_tma_b, sizeof(CUtensorMap), stream));

    dim3 blk(NUM_THREADS);
    uint32_t gm = (M + BLOCK_M - 1) / BLOCK_M;
    uint32_t gn = (N + BLOCK_N - 1) / BLOCK_N;
    dim3 grd(gm * gn);

    gemm_kernel<<<grd, blk, TOTAL_SMEM_BYTES, stream>>>(
        A_data, B_data, C_data, *d_ta, *d_tb, M, N, K);

    CUDA_RT_CHECK(cudaGetLastError());
    CUDA_RT_CHECK(cudaStreamSynchronize(stream));

    CUDA_DRV_CHECK(cuMemFree(reinterpret_cast<CUdeviceptr>(d_ta)));
    CUDA_DRV_CHECK(cuMemFree(reinterpret_cast<CUdeviceptr>(d_tb)));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_gemm_blackwell::run);

}  // namespace tvm_ffi_gemm_blackwell