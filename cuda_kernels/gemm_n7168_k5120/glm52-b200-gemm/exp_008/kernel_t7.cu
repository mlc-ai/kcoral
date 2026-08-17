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

namespace tvm_ffi_gemm_cuda {

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 64;
constexpr int M_COMBINED = 256;
constexpr int N_COMBINED = 256;
constexpr int NUM_THREADS = 128;

CUresult create_tma_descriptor(CUtensorMap* d, void* ptr,
    uint64_t gdim0, uint64_t gdim1, uint32_t bdim0, uint32_t bdim1,
    CUtensorMapSwizzle swizzle) {
    CUtensorMapDataType dataType = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    cuuint64_t globalDim[2] = {gdim0, gdim1};
    cuuint64_t globalStrides[1] = {gdim0 * 2};
    cuuint32_t boxDim[2] = {bdim0, bdim1};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(d, dataType, 2, ptr, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

__device__ __forceinline__ uint32_t cluster_rank() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ void cluster_sync() {
    asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n" ::: "memory");
}

__device__ __forceinline__ void init_barrier(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}

__device__ __forceinline__ void fence_barrier_init() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void arrive_expect_tx_local(uint64_t* bar, uint32_t tx) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(tx) : "memory");
}

__device__ __forceinline__ uint32_t mapa_addr(void* ptr, uint32_t cta) {
    uint32_t local = (uint32_t)__cvta_generic_to_shared(ptr);
    uint32_t remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(remote) : "r"(local), "r"(cta));
    return remote;
}

__device__ __forceinline__ void arrive_expect_tx_remote(uint32_t bar_addr, uint32_t tx) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%0], %1;"
        :: "r"(bar_addr), "r"(tx) : "memory");
}

__device__ __forceinline__ void arrive_remote(uint32_t bar_addr) {
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"
        :: "r"(bar_addr) : "memory");
}

__device__ __forceinline__ void wait_barrier(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n .reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(phase));
}

__device__ __forceinline__ void tma_load_cg2(const CUtensorMap* d, uint64_t* bar,
    void* smem, int32_t c0, int32_t c1) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t ba = (uint32_t)__cvta_generic_to_shared(bar) & 0xFEFFFFFF;
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(sa), "l"((uint64_t)d), "r"(c0), "r"(c1), "r"(ba) : "memory");
}

__device__ __forceinline__ void tmem_alloc(uint32_t* dst, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
        :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void umma_cg2(
    uint32_t tmem_d, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_d), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_2sm(uint64_t* bar) {
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "h"((uint16_t)0x3));
}

__device__ __forceinline__ uint64_t make_desc(void* ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint32_t make_idesc(uint32_t M, uint32_t N) {
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

__device__ __forceinline__ void tmem_load_4(uint32_t col,
    uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void fence_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__global__ void __cluster_dims__(2, 1, 1) gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* __restrict__ C, int M, int N, int K) {

    __align__(1024) __shared__ __nv_bfloat16 A_smem[2][BM][BK];
    __align__(1024) __shared__ __nv_bfloat16 B_smem[2][BN][BK];
    __shared__ uint64_t load_bar[2];
    __shared__ uint64_t signal_bar[2];
    __shared__ uint64_t mma_bar;
    __shared__ uint32_t tmem_addr_val;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    uint32_t cta_rank = cluster_rank();

    if (tid == 0) {
        if (cta_rank == 0) {
            init_barrier(&load_bar[0], 2);
            init_barrier(&load_bar[1], 2);
        }
        init_barrier(&signal_bar[0], 1);
        init_barrier(&signal_bar[1], 1);
        init_barrier(&mma_bar, 1);
        fence_barrier_init();
    }
    __syncthreads();
    cluster_sync();

    if (tid < 32) {
        tmem_alloc(&tmem_addr_val, 256);
    }
    __syncthreads();
    uint32_t tmem_c = tmem_addr_val;

    int m_block = (blockIdx.x / 2) * M_COMBINED;
    int n_block = blockIdx.y * N_COMBINED;
    int bm = m_block + cta_rank * BM;
    int bn = n_block + cta_rank * BN;
    int k_tiles = K / BK;
    uint32_t idesc = make_idesc(M_COMBINED, N_COMBINED);
    uint32_t load_bytes = (uint32_t)(BM * BK * 2 + BN * BK * 2);

    auto issue_load = [&](int buf, int k_offset) {
        if (tid == 0) {
            if (cta_rank == 0) {
                arrive_expect_tx_local(&load_bar[buf], load_bytes);
            } else {
                uint32_t remote = mapa_addr(&load_bar[buf], 0);
                arrive_expect_tx_remote(remote, load_bytes);
            }
            tma_load_cg2(&tma_A, &load_bar[buf], &A_smem[buf][0][0], k_offset, bm);
            tma_load_cg2(&tma_B, &load_bar[buf], &B_smem[buf][0][0], k_offset, bn);
        }
    };

    // Prologue
    issue_load(0, 0);

    uint32_t accum_flag = 0;
    uint32_t phase_load[2] = {0, 0};
    uint32_t phase_signal[2] = {0, 0};
    uint32_t phase_mma = 0;

    for (int kt = 0; kt < k_tiles; kt++) {
        int buf = kt % 2;
        int next_buf = 1 - buf;

        // CTA0: wait for TMA load, then signal CTA1
        // CTA1: wait for CTA0's signal
        if (cta_rank == 0) {
            if (tid == 0) {
                wait_barrier(&load_bar[buf], phase_load[buf]);
                phase_load[buf] ^= 1;
                uint32_t remote = mapa_addr(&signal_bar[buf], 1);
                arrive_remote(remote);
            }
        } else {
            if (tid == 0) {
                wait_barrier(&signal_bar[buf], phase_signal[buf]);
                phase_signal[buf] ^= 1;
            }
        }
        __syncthreads();

        // CTA0: issue MMA
        if (cta_rank == 0 && tid == 0) {
            fence_async_shared();
            #pragma unroll
            for (int ks = 0; ks < BK / 16; ks++) {
                uint64_t desc_a = make_desc(&A_smem[buf][0][ks * 16], 1, 1024);
                uint64_t desc_b = make_desc(&B_smem[buf][0][ks * 16], 1, 1024);
                umma_cg2(tmem_c, desc_a, desc_b, idesc, accum_flag);
                accum_flag = 1;
            }
            umma_commit_2sm(&mma_bar);
        }

        // Issue next TMA load BEFORE wait_mma (overlaps TMA with MMA)
        if (kt + 1 < k_tiles) {
            issue_load(next_buf, (kt + 1) * BK);
        }

        // Wait for MMA (TMA load runs in parallel)
        wait_barrier(&mma_bar, phase_mma);
        phase_mma ^= 1;
    }

    // ===== Epilogue: TMEM -> SMEM -> Global =====
    __nv_bfloat16* C_smem = reinterpret_cast<__nv_bfloat16*>(A_smem);
    constexpr int OUT_N = N_COMBINED;

    for (int col = 0; col < OUT_N; col += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4(col, &r0, &r1, &r2, &r3);
        tmem_wait_ld();
        int row = warp_id * 32 + lane_id;
        C_smem[row * OUT_N + col + 0] = __float2bfloat16(__uint_as_float(r0));
        C_smem[row * OUT_N + col + 1] = __float2bfloat16(__uint_as_float(r1));
        C_smem[row * OUT_N + col + 2] = __float2bfloat16(__uint_as_float(r2));
        C_smem[row * OUT_N + col + 3] = __float2bfloat16(__uint_as_float(r3));
    }
    __syncthreads();

    // Coalesced global writes
    int out_m = m_block + cta_rank * BM;
    int out_n = n_block;

    for (int step = 0; step < BM / 4; ++step) {
        int row = step * 4 + warp_id;
        int gm_row = out_m + row;
        if (gm_row >= M) continue;
        int col_start = lane_id * 4;
        int gn_col = out_n + col_start;
        if (gn_col + 3 < N) {
            uint2 data = *reinterpret_cast<uint2*>(&C_smem[row * OUT_N + col_start]);
            *reinterpret_cast<uint2*>(C + (uint64_t)gm_row * N + gn_col) = data;
        } else {
            for (int j = 0; j < 4; j++) {
                if (gn_col + j < N) {
                    C[(uint64_t)gm_row * N + gn_col + j] = C_smem[row * OUT_N + col_start + j];
                }
            }
        }
    }

    __syncthreads();
    cluster_sync();
    if (tid < 32) {
        tmem_dealloc(tmem_c, 256);
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
    CU_CHECK(create_tma_descriptor(&tma_A, (void*)A_ptr, K, M, BK, BM, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_descriptor(&tma_B, (void*)B_ptr, K, N, BK, BN, CU_TENSOR_MAP_SWIZZLE_128B));

    dim3 grid((M + BM - 1) / BM, (N + N_COMBINED - 1) / N_COMBINED);
    dim3 block(NUM_THREADS);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, tma_A, tma_B, C_ptr, (int)M, (int)N, (int)K));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_gemm_cuda::run);

}  // namespace tvm_ffi_gemm_cuda