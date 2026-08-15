#include <cuda_bf16.h>
#include <cuda_runtime.h>
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

namespace gemm_blackwell {

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 16;

// ============ Device helper functions ============

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase));
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void prefetch_tma_descriptor_fn(const CUtensorMap* d) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(d) : "memory");
}

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

// No-swizzle SMEM descriptor for K-major layout
// K-Major no swizzle:
//   ATOM_MMODE_DIM = BM = 128
//   ATOM_KMODE_DIM = 16/2 = 8 (span = 16B)
//   SBO = 8 * 16 = 128 (8 spans along M)
//   LBO = (128/8) * 128 = 2048
__device__ __forceinline__ uint64_t make_smem_desc_no_swizzle(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)((addr & 0x3FFFF) >> 4);
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   // version = 1
    d |= (uint64_t)0 << 61;   // no swizzle
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_bf16_cg1(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    // D type = FP32
    d |= (1u << 7);    // A type = BF16
    d |= (1u << 10);   // B type = BF16
    d |= (0u << 15);   // A K-Major
    d |= (0u << 16);   // B K-Major
    d |= ((N / 8) << 17);   // N dim
    d |= ((M / 16) << 24);  // M dim
    return d;
}

__device__ __forceinline__ void umma_f16_cg1_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "r"(a));
}

__device__ __forceinline__ uint32_t pack_bf16_fn(uint32_t fp32_a, uint32_t fp32_b) {
    __nv_bfloat16 a = __float2bfloat16(__uint_as_float(fp32_a));
    __nv_bfloat16 b = __float2bfloat16(__uint_as_float(fp32_b));
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&a)),
          "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}

// ============ Kernel ============

__global__ __launch_bounds__(128, 1)
void gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* __restrict__ C, int M, int N, int K) {

    __shared__ __align__(16) __nv_bfloat16 smemA[2][BM * BK];
    __shared__ __align__(16) __nv_bfloat16 smemB[2][BN * BK];
    __shared__ __align__(8) uint64_t mbar[3];
    __shared__ uint32_t tmem_addr;

    const uint32_t m_block = blockIdx.x * BM;
    const uint32_t n_block = blockIdx.y * BN;
    const uint32_t k_tiles = K / BK;

    // 1. Allocate TMEM (128 columns for 128x128 FP32 accumulator)
    if (threadIdx.x < 32) {
        tmem_alloc_cg1_fn(&tmem_addr, 128);
    }
    __syncthreads();
    const uint32_t tmem_c = tmem_addr;

    // 2. Initialize mbarriers
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&mbar[0], 1);
        init_smem_barrier_fn(&mbar[1], 1);
        init_smem_barrier_fn(&mbar[2], 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    // 3. Prefetch TMA descriptors
    prefetch_tma_descriptor_fn(&tma_A);
    prefetch_tma_descriptor_fn(&tma_B);

    // 4. Issue first TMA load (buffer 0)
    if (threadIdx.x == 0) {
        uint32_t tx_bytes = BM * BK * 2 + BN * BK * 2;
        mbarrier_arrive_and_expect_tx_fn(&mbar[0], tx_bytes);
        tma_load_2d_fn(&tma_A, &mbar[0], &smemA[0][0], 0, (int32_t)m_block);
        tma_load_2d_fn(&tma_B, &mbar[0], &smemB[0][0], 0, (int32_t)n_block);
    }

    // K-Major no-swizzle descriptor constants
    // SBO = 8 * 16 = 128, LBO = (128/8) * 128 = 2048
    const uint32_t SBO = 128;
    const uint32_t LBO = 2048;

    // 5. Main K-loop with double buffering
    uint32_t tma_phase[2] = {0, 0};
    uint32_t umma_phase = 0;

    #pragma unroll 1
    for (uint32_t kt = 0; kt < k_tiles; ++kt) {
        const uint32_t buf = kt % 2;

        // Wait for TMA load of current buffer
        mbarrier_wait_fn(&mbar[buf], tma_phase[buf]);
        tma_phase[buf] ^= 1;

        // Issue next TMA load (double buffering)
        if (kt + 1 < k_tiles) {
            if (threadIdx.x == 0) {
                const uint32_t next_buf = (kt + 1) % 2;
                const uint32_t tx_bytes = BM * BK * 2 + BN * BK * 2;
                mbarrier_arrive_and_expect_tx_fn(&mbar[next_buf], tx_bytes);
                tma_load_2d_fn(&tma_A, &mbar[next_buf], &smemA[next_buf][0],
                               (int32_t)((kt + 1) * BK), (int32_t)m_block);
                tma_load_2d_fn(&tma_B, &mbar[next_buf], &smemB[next_buf][0],
                               (int32_t)((kt + 1) * BK), (int32_t)n_block);
            }
        }

        // Fence: ensure async proxy SMEM writes are visible for UMMA
        fence_async_shared_fn();

        // Issue UMMA (single thread issues)
        if (threadIdx.x == 0) {
            const uint64_t desc_a = make_smem_desc_no_swizzle(&smemA[buf][0], LBO, SBO);
            const uint64_t desc_b = make_smem_desc_no_swizzle(&smemB[buf][0], LBO, SBO);
            const uint32_t idesc = make_instr_desc_bf16_cg1(BM, BN);
            const uint32_t accum = (kt == 0) ? 0 : 1;
            umma_f16_cg1_fn(tmem_c, desc_a, desc_b, idesc, accum);
            umma_commit_cg1_fn(&mbar[2]);
        }

        // Wait for UMMA completion
        mbarrier_wait_fn(&mbar[2], umma_phase);
        umma_phase ^= 1;
    }

    // 6. Epilogue: TMEM (FP32) -> convert to BF16 -> store to global
    const uint32_t global_row = m_block + threadIdx.x;

    for (uint32_t col = 0; col < (uint32_t)BN; col += 8) {
        uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
            : "r"(tmem_c + col));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r4), "=r"(r5), "=r"(r6), "=r"(r7)
            : "r"(tmem_c + col + 4));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

        if (global_row < (uint32_t)M) {
            const uint32_t global_col = n_block + col;
            if (global_col + 7 < (uint32_t)N) {
                uint4 data;
                data.x = pack_bf16_fn(r0, r1);
                data.y = pack_bf16_fn(r2, r3);
                data.z = pack_bf16_fn(r4, r5);
                data.w = pack_bf16_fn(r6, r7);
                *reinterpret_cast<uint4*>(C + (uint64_t)global_row * N + global_col) = data;
            } else {
                __nv_bfloat16 vals[8] = {
                    __float2bfloat16(__uint_as_float(r0)),
                    __float2bfloat16(__uint_as_float(r1)),
                    __float2bfloat16(__uint_as_float(r2)),
                    __float2bfloat16(__uint_as_float(r3)),
                    __float2bfloat16(__uint_as_float(r4)),
                    __float2bfloat16(__uint_as_float(r5)),
                    __float2bfloat16(__uint_as_float(r6)),
                    __float2bfloat16(__uint_as_float(r7))
                };
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    if (global_col + j < (uint32_t)N) {
                        C[(uint64_t)global_row * N + global_col + j] = vals[j];
                    }
                }
            }
        }
    }

    __syncthreads();

    // 7. Deallocate TMEM
    if (threadIdx.x < 32) {
        tmem_dealloc_cg1_fn(tmem_c, 128);
    }
}

// ============ Host function ============

void create_tma_desc(CUtensorMap* d, void* ptr, uint64_t inner_dim, uint64_t outer_dim,
                     uint32_t box_inner, uint32_t box_outer, CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[2] = {inner_dim, outer_dim};
    cuuint64_t globalStrides[1] = {inner_dim * 2};
    cuuint32_t boxDim[2] = {box_inner, box_outer};
    cuuint32_t elementStrides[2] = {1, 1};
    CUresult r = cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        ptr, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
    if (r != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(r, &errStr);
        fprintf(stderr, "TMA desc error: %s\n", errStr ? errStr : "unknown");
        exit(1);
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

    CUtensorMap tma_A, tma_B;
    create_tma_desc(&tma_A, (void*)A_ptr, K, M, BK, BM, CU_TENSOR_MAP_SWIZZLE_NONE);
    create_tma_desc(&tma_B, (void*)B_ptr, K, N, BK, BN, CU_TENSOR_MAP_SWIZZLE_NONE);

    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN, 1);
    dim3 block(128, 1, 1);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    gemm_kernel<<<grid, block, 0, stream>>>(tma_A, tma_B, C_ptr, (int)M, (int)N, (int)K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_blackwell::run);

}  // namespace gemm_blackwell