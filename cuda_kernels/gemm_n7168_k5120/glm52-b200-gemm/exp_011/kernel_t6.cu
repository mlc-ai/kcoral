#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define CU_CHECK(call) do { \
    CUresult _r = (call); \
    if (_r != CUDA_SUCCESS) { \
        const char* err_str; \
        cuGetErrorString(_r, &err_str); \
        fprintf(stderr, "CU error %s at %s:%d\n", \
                err_str ? err_str : "unknown", __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// ---- Helper functions ----

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
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

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
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

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_store_2d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_store_fence_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= (0u << 15);   // a_major = 0 (K-Major)
    d |= (0u << 16);   // b_major = 0 (K-Major)
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_128b_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   // version = 1 (SM100)
    uint32_t base_offset = (addr >> 7) & 0x7;
    d |= (uint64_t)base_offset << 49;
    d |= (uint64_t)2 << 61;   // SWIZZLE_128B
    return d;
}

namespace gemm_cuda {

constexpr uint32_t BM = 128;
constexpr uint32_t BN = 256;
constexpr uint32_t BK = 64;
constexpr uint32_t K_MMA = 16;
constexpr uint32_t NUM_STAGES = 3;
constexpr uint32_t BN_PAD = BN + 8;
constexpr uint32_t A_TILE_BYTES = BM * BK * 2;   // 16384
constexpr uint32_t B_TILE_BYTES = BN * BK * 2;   // 32768
constexpr uint32_t TMA_BYTES = A_TILE_BYTES + B_TILE_BYTES; // 49152

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
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.shared::cta.b64 [%0];"
        :: "r"(a));
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

__global__ void gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    const __grid_constant__ CUtensorMap tma_C,
    __nv_bfloat16* C, uint32_t M, uint32_t N, uint32_t K) {

    extern __shared__ __align__(128) uint8_t smem[];

    // Shared memory layout: 3 stages of A+B tiles, then epilogue buffer, then barriers
    __nv_bfloat16* smem_A[NUM_STAGES];
    __nv_bfloat16* smem_B[NUM_STAGES];
    #pragma unroll
    for (int i = 0; i < NUM_STAGES; i++) {
        smem_A[i] = ((__nv_bfloat16*)smem) + i * BM * BK;
        smem_B[i] = ((__nv_bfloat16*)smem) + NUM_STAGES * BM * BK + i * BN * BK;
    }

    __nv_bfloat16* smem_out = ((__nv_bfloat16*)smem) + NUM_STAGES * (BM * BK + BN * BK);

    // Barriers: 3 TMA + 1 MMA + tmem addr
    uint64_t* bar = (uint64_t*)(smem_out + BM * BN_PAD);
    uint64_t* bar_tma[NUM_STAGES];
    uint64_t* bar_mma[2];
    #pragma unroll
    for (int i = 0; i < NUM_STAGES; i++) bar_tma[i] = &bar[i];
    bar_mma[0] = &bar[NUM_STAGES];
    bar_mma[1] = &bar[NUM_STAGES + 1];
    uint32_t* tmem_addr_smem = (uint32_t*)(bar + NUM_STAGES + 2);

    uint32_t m_block = blockIdx.y;
    uint32_t n_block = blockIdx.x;
    uint32_t m_coord = m_block * BM;
    uint32_t n_coord = n_block * BN;

    // Initialize barriers
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < NUM_STAGES; i++) {
            init_smem_barrier_fn(bar_tma[i], 1);
        }
        init_smem_barrier_fn(bar_mma[0], 1);
        init_smem_barrier_fn(bar_mma[1], 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    // Allocate TMEM (256 columns for 128x256 FP32 accumulator)
    if (threadIdx.x < 32) {
        tmem_alloc_cg1_fn(tmem_addr_smem, 256);
    }
    __syncthreads();
    uint32_t tmem_addr = *tmem_addr_smem;

    uint32_t idesc = make_instr_desc_fn(BM, BN);
    uint32_t num_k_tiles = K / BK;

    // Phase trackers
    uint32_t phase_tma[NUM_STAGES] = {0, 0, 0};
    uint32_t phase_mma[2] = {0, 0};

    // Prologue: issue first 2 TMA loads
    if (threadIdx.x == 0) {
        for (int s = 0; s < NUM_STAGES - 1 && s < num_k_tiles; s++) {
            mbarrier_arrive_and_expect_tx_fn(bar_tma[s], TMA_BYTES);
            tma_load_2d_fn(&tma_A, bar_tma[s], smem_A[s], (int32_t)(s * BK), (int32_t)m_coord);
            tma_load_2d_fn(&tma_B, bar_tma[s], smem_B[s], (int32_t)(s * BK), (int32_t)n_coord);
        }
    }

    uint32_t mma_idx = 0;
    for (uint32_t k = 0; k < num_k_tiles; k++) {
        uint32_t slot = k % NUM_STAGES;

        // Wait for TMA[k]
        mbarrier_wait_fn(bar_tma[slot], phase_tma[slot]);
        phase_tma[slot] ^= 1;

        // Issue next TMA load (k + NUM_STAGES - 1)
        uint32_t next_k = k + NUM_STAGES - 1;
        if (next_k < num_k_tiles && threadIdx.x == 0) {
            uint32_t next_slot = next_k % NUM_STAGES;
            mbarrier_arrive_and_expect_tx_fn(bar_tma[next_slot], TMA_BYTES);
            tma_load_2d_fn(&tma_A, bar_tma[next_slot], smem_A[next_slot], (int32_t)(next_k * BK), (int32_t)m_coord);
            tma_load_2d_fn(&tma_B, bar_tma[next_slot], smem_B[next_slot], (int32_t)(next_k * BK), (int32_t)n_coord);
        }

        // Wait for MMA[k - NUM_STAGES + 1] before reusing this buffer
        // With 3 stages: buffer[k%3] was last used by MMA[k-3]. We need MMA[k-3] done.
        // But we issue MMA[k-2] 2 iterations ago. So we wait for MMA from 2 iterations ago.
        if (k >= 2) {
            uint32_t wait_mma_idx = (mma_idx - 2) % 2;
            mbarrier_wait_fn(bar_mma[wait_mma_idx], phase_mma[wait_mma_idx]);
            phase_mma[wait_mma_idx] ^= 1;
        }

        // Fence: make TMA writes visible to UMMA
        fence_async_shared_fn();

        // Issue 4 MMAs (each K=16, total BK=64)
        if (threadIdx.x == 0) {
            for (uint32_t k_sub = 0; k_sub < BK / K_MMA; k_sub++) {
                uint64_t desc_a = make_smem_desc_128b_fn(
                    smem_A[slot] + k_sub * K_MMA, 1, 1024);
                uint64_t desc_b = make_smem_desc_128b_fn(
                    smem_B[slot] + k_sub * K_MMA, 1, 1024);
                uint32_t accum = (k > 0 || k_sub > 0) ? 1 : 0;
                umma_f16_cg1_fn(tmem_addr, desc_a, desc_b, idesc, accum);
            }
            uint32_t cur_mma_bar = mma_idx % 2;
            umma_commit_cg1_fn(bar_mma[cur_mma_bar]);
        }
        mma_idx++;
    }

    // Wait for remaining MMAs
    if (num_k_tiles >= 2) {
        mbarrier_wait_fn(bar_mma[(mma_idx - 2) % 2], phase_mma[(mma_idx - 2) % 2]);
        phase_mma[(mma_idx - 2) % 2] ^= 1;
    }
    {
        uint32_t last_bar = (mma_idx - 1) % 2;
        mbarrier_wait_fn(bar_mma[last_bar], phase_mma[last_bar]);
        phase_mma[last_bar] ^= 1;
    }

    // ---- Epilogue: TMEM -> SMEM -> TMA Store ----

    // Phase 1: Load from TMEM, convert FP32->BF16, write to SMEM with padding
    for (uint32_t col = 0; col < BN; col += 8) {
        uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),
              "=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7)
            : "r"(tmem_addr + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

        uint32_t row = threadIdx.x;
        uint32_t* out = reinterpret_cast<uint32_t*>(
            smem_out + row * BN_PAD + col);
        out[0] = pack_bf16_fn(r0, r1);
        out[1] = pack_bf16_fn(r2, r3);
        out[2] = pack_bf16_fn(r4, r5);
        out[3] = pack_bf16_fn(r6, r7);
    }
    __syncthreads();

    // Phase 2: TMA store from SMEM to global
    // Need to unswizzle: copy from padded SMEM to a contiguous TMA-compatible buffer
    // Use the smem_A[0] buffer as scratch (it's no longer needed)
    __nv_bfloat16* smem_store = smem_A[0];
    for (uint32_t v = threadIdx.x; v < BM * BN / 8; v += 128) {
        uint32_t row = v / (BN / 8);
        uint32_t col8 = v % (BN / 8);
        uint32_t col = col8 * 8;
        int4 data = *reinterpret_cast<int4*>(smem_out + row * BN_PAD + col);
        *reinterpret_cast<int4*>(smem_store + row * BN + col) = data;
    }
    __syncthreads();

    // TMA store
    if (threadIdx.x == 0) {
        tma_store_fence_fn();
        // Store first 64 rows
        tma_store_2d_fn(&tma_C, smem_store, (int32_t)n_coord, (int32_t)m_coord);
        // Store next 64 rows
        tma_store_2d_fn(&tma_C, smem_store + 64 * BN, (int32_t)n_coord, (int32_t)(m_coord + 64));
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }
    __syncthreads();

    // Deallocate TMEM
    if (threadIdx.x < 32) {
        tmem_dealloc_cg1_fn(tmem_addr, 256);
    }
}

// TMA descriptor creation for BF16 2D tensors
static CUresult create_tma_desc(CUtensorMap* d, void* ptr,
                                uint64_t inner_dim, uint64_t outer_dim,
                                uint32_t box_inner, uint32_t box_outer,
                                CUtensorMapSwizzle swizzle,
                                CUtensorMapL2promotion l2prom,
                                CUtensorMapFloatOOBfill oob) {
    cuuint64_t globalDim[2] = {inner_dim, outer_dim};
    cuuint64_t globalStrides[1] = {inner_dim * 2};
    cuuint32_t boxDim[2] = {box_inner, box_outer};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        ptr,
        globalDim,
        globalStrides,
        boxDim,
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        l2prom,
        oob);
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));

    int64_t M = A.size(0);
    int64_t N = B.size(0);
    int64_t K = A.size(1);

    __nv_bfloat16* A_ptr = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* B_ptr = static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());

    CUtensorMap dA, dB, dC;
    CU_CHECK(create_tma_desc(&dA, A_ptr, (uint64_t)K, (uint64_t)M, BK, BM,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA));

    CU_CHECK(create_tma_desc(&dB, B_ptr, (uint64_t)K, (uint64_t)N, BK, BN,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA));

    // C descriptor: [M, N] row-major, boxDim={BN, BM}={256, 128}
    CU_CHECK(create_tma_desc(&dC, C_ptr, (uint64_t)N, (uint64_t)M, BN, BM,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    dim3 grid((uint32_t)((N + BN - 1) / BN), (uint32_t)((M + BM - 1) / BM), 1);
    dim3 block(128, 1, 1);

    size_t smem_size = NUM_STAGES * (A_TILE_BYTES + B_TILE_BYTES)
                      + BM * BN_PAD * 2              // padded epilogue
                      + BM * BN * 2                   // contiguous store buffer
                      + (NUM_STAGES + 2) * 8 + 4;    // barriers + tmem addr
    smem_size = (smem_size + 127) & ~127;

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));
    CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));

    gemm_kernel<<<grid, block, smem_size, stream>>>(
        dA, dB, dC, C_ptr, (uint32_t)M, (uint32_t)N, (uint32_t)K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_cuda::run);

}  // namespace gemm_cuda