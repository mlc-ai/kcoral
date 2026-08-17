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
        const char* errStr; \
        cuGetErrorString(_r, &errStr); \
        fprintf(stderr, "CU error %s at %s:%d\n", \
                errStr ? errStr : "unknown", __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace gemm_cuda {

constexpr int BM = 128;
constexpr int BN = 256;
constexpr int BK_CHUNK = 64;
constexpr int NUM_THREADS = 128;
constexpr int SMEM_SIZE = 163968;

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress,
    uint64_t gmem_inner_dim, uint64_t gmem_outer_dim,
    uint32_t smem_inner_dim, uint32_t smem_outer_dim,
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        globalAddress, globalDim, globalStrides,
        boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle, l2Promotion, oobFill);
}

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

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
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

__device__ __forceinline__ void prefetch_tma_descriptor_fn(const CUtensorMap* d) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(d) : "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_128b_swizzle_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= (0u << 15);
    d |= (0u << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
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
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
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

__global__ __launch_bounds__(128, 1) void gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    const __grid_constant__ CUtensorMap tma_C,
    int M, int N, int K) {

    extern __shared__ __align__(128) uint8_t smem_raw[];

    __nv_bfloat16* smem_A_base = (__nv_bfloat16*)smem_raw;
    __nv_bfloat16* smem_B_base = (__nv_bfloat16*)(smem_raw + 32768);
    uint64_t* mbar_tma = (uint64_t*)(smem_raw + 98304);
    uint64_t* mbar_mma = (uint64_t*)(smem_raw + 98320);
    uint32_t* tmem_addr_ptr = (uint32_t*)(smem_raw + 98336);
    __nv_bfloat16* smem_out = (__nv_bfloat16*)(smem_raw + 98432);

    int tid = threadIdx.x;
    int m_offset = blockIdx.y * BM;
    int n_offset = blockIdx.x * BN;

    if (tid == 0) {
        prefetch_tma_descriptor_fn(&tma_A);
        prefetch_tma_descriptor_fn(&tma_B);
        prefetch_tma_descriptor_fn(&tma_C);
    }

    if (tid < 32) {
        tmem_alloc_cg1_fn(tmem_addr_ptr, 256);
    }
    __syncthreads();
    uint32_t tmem_addr = *tmem_addr_ptr;

    if (tid == 0) {
        init_smem_barrier_fn(&mbar_tma[0], 1);
        init_smem_barrier_fn(&mbar_tma[1], 1);
        init_smem_barrier_fn(&mbar_mma[0], 1);
        init_smem_barrier_fn(&mbar_mma[1], 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    int num_k_chunks = K / BK_CHUNK;
    constexpr uint32_t tile_bytes_A = BM * BK_CHUNK * 2;
    constexpr uint32_t tile_bytes_B = BN * BK_CHUNK * 2;
    constexpr uint32_t tile_bytes = tile_bytes_A + tile_bytes_B;

    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar_tma[0], tile_bytes);
        tma_load_2d_fn(&tma_A, &mbar_tma[0], smem_A_base, 0, m_offset);
        tma_load_2d_fn(&tma_B, &mbar_tma[0], smem_B_base, 0, n_offset);
    }

    uint32_t idesc = make_instr_desc_fn(BM, BN);
    constexpr uint32_t SBO = 1024;
    constexpr uint32_t LBO = 1;

    uint32_t tma_phase[2] = {0, 0};
    uint32_t mma_phase[2] = {0, 0};

    for (int kc = 0; kc < num_k_chunks; kc++) {
        int stage = kc % 2;
        int next_stage = (kc + 1) % 2;

        mbarrier_wait_fn(&mbar_tma[stage], tma_phase[stage]);
        tma_phase[stage] ^= 1;

        if (kc >= 1) {
            mbarrier_wait_fn(&mbar_mma[next_stage], mma_phase[next_stage]);
            mma_phase[next_stage] ^= 1;
        }

        if (kc + 1 < num_k_chunks) {
            __nv_bfloat16* next_A = smem_A_base + next_stage * (BM * BK_CHUNK);
            __nv_bfloat16* next_B = smem_B_base + next_stage * (BN * BK_CHUNK);
            if (tid == 0) {
                mbarrier_arrive_and_expect_tx_fn(&mbar_tma[next_stage], tile_bytes);
                tma_load_2d_fn(&tma_A, &mbar_tma[next_stage], next_A, (kc + 1) * BK_CHUNK, m_offset);
                tma_load_2d_fn(&tma_B, &mbar_tma[next_stage], next_B, (kc + 1) * BK_CHUNK, n_offset);
            }
        }

        __nv_bfloat16* smem_A = smem_A_base + stage * (BM * BK_CHUNK);
        __nv_bfloat16* smem_B = smem_B_base + stage * (BN * BK_CHUNK);
        if (tid == 0) {
            for (int ks = 0; ks < 4; ks++) {
                uint64_t desc_a = make_smem_desc_128b_swizzle_fn(smem_A + ks * 16, LBO, SBO);
                uint64_t desc_b = make_smem_desc_128b_swizzle_fn(smem_B + ks * 16, LBO, SBO);
                int k_global = kc * 4 + ks;
                umma_f16_cg1_fn(tmem_addr, desc_a, desc_b, idesc, k_global > 0 ? 1 : 0);
            }
            umma_commit_cg1_fn(&mbar_mma[stage]);
        }
    }

    mbarrier_wait_fn(&mbar_mma[(num_k_chunks - 1) % 2], mma_phase[(num_k_chunks - 1) % 2]);

    // Epilogue: TMEM -> SMEM -> TMA store
    for (uint32_t col = 0; col < (uint32_t)BN; col += 32) {
        uint32_t r0[16], r1[16];
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
            : "=r"(r0[0]), "=r"(r0[1]), "=r"(r0[2]), "=r"(r0[3]),
              "=r"(r0[4]), "=r"(r0[5]), "=r"(r0[6]), "=r"(r0[7]),
              "=r"(r0[8]), "=r"(r0[9]), "=r"(r0[10]), "=r"(r0[11]),
              "=r"(r0[12]), "=r"(r0[13]), "=r"(r0[14]), "=r"(r0[15])
            : "r"(tmem_addr + col));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
            : "=r"(r1[0]), "=r"(r1[1]), "=r"(r1[2]), "=r"(r1[3]),
              "=r"(r1[4]), "=r"(r1[5]), "=r"(r1[6]), "=r"(r1[7]),
              "=r"(r1[8]), "=r"(r1[9]), "=r"(r1[10]), "=r"(r1[11]),
              "=r"(r1[12]), "=r"(r1[13]), "=r"(r1[14]), "=r"(r1[15])
            : "r"(tmem_addr + col + 16));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

        uint32_t base0 = tid * BN + col;
        uint32_t base1 = tid * BN + col + 16;
        for (int i = 0; i < 16; i += 2) {
            *(uint32_t*)&smem_out[base0 + i] = pack_bf16_fn(r0[i], r0[i+1]);
            *(uint32_t*)&smem_out[base1 + i] = pack_bf16_fn(r1[i], r1[i+1]);
        }
    }
    __syncthreads();

    if (tid == 0) {
        tma_store_fence_fn();
        tma_store_2d_fn(&tma_C, smem_out, n_offset, m_offset);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }
    __syncthreads();

    if (tid < 32) {
        tmem_dealloc_cg1_fn(tmem_addr, 256);
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);

    __nv_bfloat16* A_ptr = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* B_ptr = static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    CUtensorMap tma_A, tma_B, tma_C;

    CU_CHECK(create_tma_2d_descriptor_2B(&tma_A, A_ptr, K, M, BK_CHUNK, BM,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    CU_CHECK(create_tma_2d_descriptor_2B(&tma_B, B_ptr, K, N, BK_CHUNK, BN,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    CU_CHECK(create_tma_2d_descriptor_2B(&tma_C, C_ptr, N, M, BN, BM,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    int grid_x = (N + BN - 1) / BN;
    int grid_y = (M + BM - 1) / BM;
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(NUM_THREADS, 1, 1);

    CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));

    gemm_kernel<<<grid, block, SMEM_SIZE, stream>>>(tma_A, tma_B, tma_C, (int)M, (int)N, (int)K);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_cuda::run);

}  // namespace gemm_cuda