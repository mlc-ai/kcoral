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

#define CU_CHECK(call) do {                                        \
    CUresult _r = (call);                                          \
    if (_r != CUDA_SUCCESS) {                                      \
        const char* errStr;                                        \
        cuGetErrorString(_r, &errStr);                             \
        fprintf(stderr, "CU error %s at %s:%d\n",                  \
                errStr ? errStr : "unknown", __FILE__, __LINE__);  \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace gemm_cuda {

constexpr int BM = 128;          // rows per CTA
constexpr int BN = 256;          // cols per CTA (full N tile)
constexpr int BK = 64;
constexpr int BN_HALF = 128;     // half of BN for multicast loading
constexpr int BM_COMBINED = 256; // 2 * BM for cta_group::2
constexpr int NUM_THREADS = 128;

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ void cluster_sync_fn() {
    asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n" ::: "memory");
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_fn(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])) : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_cluster_fn(uint64_t* bar, uint32_t target_cta) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    uint32_t remote_a;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(remote_a) : "r"(a), "r"(target_cta));
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"
                 :: "r"(remote_a) : "memory");
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

__device__ __forceinline__ void tma_load_multicast_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, uint16_t mask) {
    uint64_t cache_hint = 0;
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%0], [%1, {%4, %5}], [%2], %3, %6;"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "h"(mask), "r"(c0), "r"(c1), "l"(cache_hint) : "memory");
}

__device__ __forceinline__ void prefetch_tma_descriptor_fn(const CUtensorMap* d) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(d) : "memory");
}

__device__ __forceinline__ void tmem_alloc_cg2_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_cg2_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
        :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void umma_f16_cg2_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_2sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(a), "h"((uint16_t)0x3));
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)((addr >> 7) & 7) << 49;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    // dtype = FP32
    d |= (1u << 7);    // atype = BF16
    d |= (1u << 10);   // btype = BF16
    d |= (0u << 15);   // a_major = 0 (K-Major)
    d |= (0u << 16);   // b_major = 0 (K-Major)
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

CUresult create_tma_2d_descriptor_bf16(CUtensorMap* d, void* globalAddress,
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
        globalAddress, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill);
}

__global__ __launch_bounds__(NUM_THREADS, 1)
void gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* __restrict__ C, int M, int N, int K) {

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const uint32_t cta = cluster_rank_fn();

    // Each CTA in the cluster handles a different M tile
    // CTA0: rows [blockIdx.x*256, blockIdx.x*256+128)
    // CTA1: rows [blockIdx.x*256+128, blockIdx.x*256+256)
    const int m_start = blockIdx.x * BM_COMBINED + cta * BM;
    const int n_start = blockIdx.y * BN;

    extern __shared__ __align__(1024) char smem_raw[];
    __nv_bfloat16* A_smem[2];
    __nv_bfloat16* B_smem[2];
    A_smem[0] = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    A_smem[1] = A_smem[0] + BM * BK;
    B_smem[0] = A_smem[1] + BM * BK;
    B_smem[1] = B_smem[0] + BN * BK;

    uint64_t* barriers = reinterpret_cast<uint64_t*>(
        (reinterpret_cast<uintptr_t>(B_smem[1] + BN * BK) + 7) & ~7ULL);
    uint64_t* full_bar = barriers;
    uint64_t* umma_bar = barriers + 2;
    uint64_t* ready_bar = barriers + 4;

    __shared__ uint32_t tmem_alloc_result;

    if (tid == 0) {
        init_smem_barrier_fn(&full_bar[0], 1);
        init_smem_barrier_fn(&full_bar[1], 1);
        init_smem_barrier_fn(&umma_bar[0], 1);
        init_smem_barrier_fn(&umma_bar[1], 1);
        if (cta == 0) {
            init_smem_barrier_fn(&ready_bar[0], 2);
            init_smem_barrier_fn(&ready_bar[1], 2);
        }
        fence_smem_barrier_init_fn();
    }
    cluster_sync_fn();

    if (warp_id == 0) {
        tmem_alloc_cg2_fn(&tmem_alloc_result, 256);
    }
    __syncthreads();
    const uint32_t tmem_addr = tmem_alloc_result;

    if (tid == 0) {
        prefetch_tma_descriptor_fn(&tma_A);
        prefetch_tma_descriptor_fn(&tma_B);
    }

    const int num_k_tiles = K / BK;
    const uint32_t idesc = make_instr_desc_fn(BM_COMBINED, BN);
    constexpr uint32_t tx_bytes = BM * BK * 2 + BN_HALF * BK * 2;  // A + B_half

    uint32_t phase_full[2] = {0, 0};
    uint32_t phase_umma[2] = {0, 0};
    uint32_t phase_ready[2] = {0, 0};

    // B half offset: CTA0 loads B[n:n+128], CTA1 loads B[n+128:n+256]
    const int b_half_off = cta * BN_HALF;
    const int b_smem_off = b_half_off * BK;  // element offset in B_smem

    // Prologue: issue TMA for stage 0
    if (tid == 0) {
        tma_load_2d_fn(&tma_A, &full_bar[0], A_smem[0], 0, m_start);
        tma_load_multicast_2d_fn(&tma_B, &full_bar[0],
            B_smem[0] + b_smem_off, 0, n_start + b_half_off, 0x3);
        mbarrier_arrive_and_expect_tx_fn(&full_bar[0], tx_bytes);
    }

    int stage = 0;
    for (int k = 0; k < num_k_tiles; k++) {
        int next_stage = 1 - stage;

        // Wait for local TMA
        mbarrier_wait_fn(&full_bar[stage], phase_full[stage]);
        phase_full[stage] ^= 1;

        // Signal ready_bar: CTA1 arrives remotely at CTA0's ready_bar
        if (tid == 0) {
            if (cta == 1) {
                mbarrier_arrive_cluster_fn(&ready_bar[stage], 0);
            } else {
                mbarrier_arrive_fn(&ready_bar[stage]);
            }
        }

        // CTA0: wait for both CTAs ready, then issue UMMA
        if (cta == 0) {
            if (tid == 0) {
                mbarrier_wait_fn(&ready_bar[stage], phase_ready[stage]);
                phase_ready[stage] ^= 1;
            }

            #pragma unroll
            for (int kk = 0; kk < 4; kk++) {
                uint64_t desc_a = make_smem_desc_sm100_fn(
                    reinterpret_cast<char*>(A_smem[stage]) + kk * 32, 1, 1024);
                uint64_t desc_b = make_smem_desc_sm100_fn(
                    reinterpret_cast<char*>(B_smem[stage]) + kk * 32, 1, 1024);
                uint32_t accum = (k == 0 && kk == 0) ? 0 : 1;
                if (tid == 0) {
                    umma_f16_cg2_fn(tmem_addr, desc_a, desc_b, idesc, accum);
                }
            }
            if (tid == 0) {
                umma_commit_2sm_fn(&umma_bar[stage]);
            }
        }

        // Both CTAs: wait for UMMA completion
        mbarrier_wait_fn(&umma_bar[stage], phase_umma[stage]);
        phase_umma[stage] ^= 1;

        // Issue TMA for next stage (k+1)
        if (tid == 0 && k + 1 < num_k_tiles) {
            int next_k = (k + 1) * BK;
            tma_load_2d_fn(&tma_A, &full_bar[next_stage], A_smem[next_stage],
                           next_k, m_start);
            tma_load_multicast_2d_fn(&tma_B, &full_bar[next_stage],
                B_smem[next_stage] + b_smem_off,
                next_k, n_start + b_half_off, 0x3);
            mbarrier_arrive_and_expect_tx_fn(&full_bar[next_stage], tx_bytes);
        }

        stage = next_stage;
    }

    __syncthreads();

    // ======================== Epilogue ========================
    __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(smem_raw);

    // Phase 1: TMEM (FP32) -> SMEM (BF16)
    // Each CTA reads its own 128 rows from its own TMEM
    #pragma unroll 4
    for (uint32_t col = 0; col < (uint32_t)BN; col += 16) {
        uint32_t r[4][4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint32_t addr = tmem_addr + (warp_id * 32 << 16) + col + i * 4;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r[i][0]), "=r"(r[i][1]), "=r"(r[i][2]), "=r"(r[i][3])
                : "r"(addr));
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint32_t base = (uint32_t)tid * BN + col + i * 4;
            smem_out[base + 0] = __float2bfloat16(__uint_as_float(r[i][0]));
            smem_out[base + 1] = __float2bfloat16(__uint_as_float(r[i][1]));
            smem_out[base + 2] = __float2bfloat16(__uint_as_float(r[i][2]));
            smem_out[base + 3] = __float2bfloat16(__uint_as_float(r[i][3]));
        }
    }
    __syncthreads();

    // Phase 2: SMEM -> Global (coalesced 128-bit stores)
    #pragma unroll
    for (uint32_t row = warp_id * 32; row < (uint32_t)(warp_id + 1) * 32; row++) {
        uint32_t global_row = m_start + row;
        if (global_row >= (uint32_t)M) continue;
        uint32_t col_start = lane_id * 8;
        uint32_t global_col = n_start + col_start;
        uint4 data = *reinterpret_cast<uint4*>(&smem_out[row * BN + col_start]);
        *reinterpret_cast<uint4*>(C + (uint64_t)global_row * N + global_col) = data;
    }

    __syncthreads();
    if (warp_id == 0) {
        tmem_dealloc_cg2_fn(tmem_addr, 256);
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));

    const int M = (int)A.size(0);
    const int N = 7168;
    const int K = 5120;

    __nv_bfloat16* A_ptr = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* B_ptr = static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());

    CUtensorMap tma_A_desc, tma_B_desc;

    CU_CHECK(create_tma_2d_descriptor_bf16(&tma_A_desc, A_ptr, K, M, BK, BM,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA));

    CU_CHECK(create_tma_2d_descriptor_bf16(&tma_B_desc, B_ptr, K, N, BK, BN_HALF,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA));

    const int m_tiles = M / BM_COMBINED;  // each cluster handles 256 rows
    const int n_tiles = N / BN;

    const int smem_bytes = 2 * (BM * BK + BN * BK) * 2 + 256;

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(C.device().device_type, C.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    cudaLaunchConfig_t config = {};
    config.gridDim = {(uint32_t)m_tiles, (uint32_t)n_tiles, 1};
    config.blockDim = {NUM_THREADS, 1, 1};
    config.dynamicSmemBytes = smem_bytes;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim = {2, 1, 1};
    config.attrs = attrs;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel,
        tma_A_desc, tma_B_desc, C_ptr, M, N, K));

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_cuda::run);

}  // namespace gemm_cuda