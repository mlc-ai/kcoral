#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdint.h>
#include <algorithm>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__);} } while(0)
#define CU_CHECK(call) do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){ const char* s; cuGetErrorString(_e,&s); fprintf(stderr,"CU error %s at %s:%d\n", s, __FILE__,__LINE__);} } while(0)

namespace gemm_sm100 {

constexpr int BM = 128, BN = 128, BK = 64;
constexpr int STAGES = 5;
constexpr int CM = 256, CN = 256;
constexpr int NBUF = 2;
constexpr int TMEM_TOTAL = 512;
constexpr int A_ELEMS = BM * BK;       // 8192
constexpr int B_ELEMS = BN * BK;       // 8192
constexpr int A_BYTES = A_ELEMS * 2;   // 16384
constexpr int B_BYTES = B_ELEMS * 2;   // 16384
constexpr uint32_t TX_BYTES = (uint32_t)(A_BYTES + B_BYTES) * 2; // both CTAs
constexpr int STAGING_BYTES = BM * CN * 2; // 65536
constexpr int SMEM_BYTES = STAGES*(A_BYTES+B_BYTES) + STAGING_BYTES + 256;
constexpr int GROUP_M = 8;
constexpr int THREADS = 192;

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r;
}
__device__ __forceinline__ void cluster_sync_fn() {
    asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n" ::: "memory");
}
__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}
__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(tx) : "memory");
}
__device__ __forceinline__ void mbarrier_arrive_fn(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)) : "memory");
}
__device__ __forceinline__ void mbarrier_arrive_cluster_fn(uint64_t* bar, uint32_t target_cta) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    uint32_t ra;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(a), "r"(target_cta));
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(ra));
}
__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n.reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(phase));
}
__device__ __forceinline__ void tma_load_2d_cg2_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t ba = (uint32_t)__cvta_generic_to_shared(&bar[0]) & 0xFEFFFFFF;
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(sa), "l"((uint64_t)d), "r"(c0), "r"(c1), "r"(ba) : "memory");
}
__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(ncols));
}
__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(addr), "r"(ncols));
}
__device__ __forceinline__ void tmem_relinquish_fn() {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
}
__device__ __forceinline__ void umma_f16_cg2_fn(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}
__device__ __forceinline__ void umma_commit_2sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
        :: "r"(a), "h"((uint16_t)0x3));
}
__device__ __forceinline__ uint64_t make_smem_desc_128b_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)((addr & 0x3FFFF) >> 4);
    d |= ((uint64_t)((lbo & 0x3FFFF) >> 4)) << 16;
    d |= ((uint64_t)((sbo & 0x3FFFF) >> 4)) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}
__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4); d |= (1u << 7); d |= (1u << 10);
    d |= ((N / 8) << 17); d |= ((M / 16) << 24);
    return d;
}
__device__ __forceinline__ void tmem_load_4x_fn(uint32_t addr, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(addr));
}
__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void named_barrier_sync_fn(int bar_id, int count) {
    asm volatile("barrier.sync.aligned %0, %1;" :: "r"(bar_id), "r"(count));
}

__device__ __forceinline__ void get_tile(int tile_id, int total_m, int total_n, int& m_tile, int& n_tile) {
    int num_in_group = GROUP_M * total_n;
    int group_id = tile_id / num_in_group;
    int first_m = group_id * GROUP_M;
    int gsize = total_m - first_m; if (gsize > GROUP_M) gsize = GROUP_M;
    int idx = tile_id % num_in_group;
    m_tile = first_m + (idx % gsize);
    n_tile = idx / gsize;
}

__global__ __launch_bounds__(THREADS) void gemm_kernel(
    const __grid_constant__ CUtensorMap tmaA,
    const __grid_constant__ CUtensorMap tmaB,
    __nv_bfloat16* __restrict__ C, int M, int N, int K, int num_clusters) {

    extern __shared__ __align__(1024) uint8_t smem[];
    __nv_bfloat16* A_smem = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* B_smem = reinterpret_cast<__nv_bfloat16*>(smem + STAGES * A_BYTES);
    __nv_bfloat16* staging = reinterpret_cast<__nv_bfloat16*>(smem + STAGES * (A_BYTES + B_BYTES));
    uint8_t* bar_base = smem + STAGES * (A_BYTES + B_BYTES) + STAGING_BYTES;
    uint64_t* full = reinterpret_cast<uint64_t*>(bar_base);
    uint64_t* empty = full + STAGES;
    uint64_t* acc_ready = empty + STAGES;
    uint64_t* epi_done = acc_ready + NBUF;
    uint32_t* tmem_ptr = reinterpret_cast<uint32_t*>(epi_done + NBUF);

    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;
    uint32_t rank = cluster_rank_fn();
    int cluster_c = blockIdx.x >> 1;

    int total_m = (M + CM - 1) / CM;
    int total_n = N / CN;
    int total_tiles = total_m * total_n;
    int numK = K / BK;

    if (tid == 0) {
        for (int s = 0; s < STAGES; s++) { init_smem_barrier_fn(&full[s], 1); init_smem_barrier_fn(&empty[s], 1); }
        for (int b = 0; b < NBUF; b++) { init_smem_barrier_fn(&acc_ready[b], 1); init_smem_barrier_fn(&epi_done[b], 2); }
    }
    fence_smem_barrier_init_fn();
    cluster_sync_fn();

    if (wid == 0) { tmem_alloc_fn(tmem_ptr, TMEM_TOTAL); tmem_relinquish_fn(); }
    __syncthreads();
    uint32_t tmem_base = *tmem_ptr;
    uint32_t idesc = make_instr_desc_fn(CM, CN);
    const uint32_t LBO = 1, SBO = 1024;

    if (wid == 5 && lane == 0) {
        // ---- Producer ----
        int iter = 0;
        for (int tile_id = cluster_c; tile_id < total_tiles; tile_id += num_clusters) {
            int m_tile, n_tile; get_tile(tile_id, total_m, total_n, m_tile, n_tile);
            int m_a = m_tile * CM + (int)rank * BM;
            int n_b = n_tile * CN + (int)rank * BN;
            for (int k = 0; k < numK; k++) {
                int s = iter % STAGES; int r = iter / STAGES;
                if (iter >= STAGES) mbarrier_wait_fn(&empty[s], (uint32_t)((r - 1) & 1));
                if (rank == 0) mbarrier_arrive_and_expect_tx_fn(&full[s], TX_BYTES);
                tma_load_2d_cg2_fn(&tmaA, &full[s], A_smem + s * A_ELEMS, k * BK, m_a);
                tma_load_2d_cg2_fn(&tmaB, &full[s], B_smem + s * B_ELEMS, k * BK, n_b);
                iter++;
            }
        }
    } else if (wid == 4 && lane == 0 && rank == 0) {
        // ---- MMA (leader CTA) ----
        int iter = 0, j = 0;
        for (int tile_id = cluster_c; tile_id < total_tiles; tile_id += num_clusters, j++) {
            int buf = j & 1; int t = j >> 1;
            if (j >= 2) mbarrier_wait_fn(&epi_done[buf], (uint32_t)((t - 1) & 1));
            uint32_t toff = tmem_base + (uint32_t)(buf * CN);
            for (int k = 0; k < numK; k++) {
                int s = iter % STAGES; int r = iter / STAGES;
                mbarrier_wait_fn(&full[s], (uint32_t)(r & 1));
                if (k < numK - 1) {
                    #pragma unroll
                    for (int ksub = 0; ksub < 4; ksub++) {
                        uint64_t da = make_smem_desc_128b_fn(A_smem + s * A_ELEMS + ksub * 16, LBO, SBO);
                        uint64_t db = make_smem_desc_128b_fn(B_smem + s * B_ELEMS + ksub * 16, LBO, SBO);
                        uint32_t accum = (k == 0 && ksub == 0) ? 0u : 1u;
                        umma_f16_cg2_fn(toff, da, db, idesc, accum);
                    }
                    umma_commit_2sm_fn(&empty[s]);
                } else {
                    #pragma unroll
                    for (int ksub = 0; ksub < 3; ksub++) {
                        uint64_t da = make_smem_desc_128b_fn(A_smem + s * A_ELEMS + ksub * 16, LBO, SBO);
                        uint64_t db = make_smem_desc_128b_fn(B_smem + s * B_ELEMS + ksub * 16, LBO, SBO);
                        umma_f16_cg2_fn(toff, da, db, idesc, 1u);
                    }
                    umma_commit_2sm_fn(&empty[s]);
                    {
                        uint64_t da = make_smem_desc_128b_fn(A_smem + s * A_ELEMS + 3 * 16, LBO, SBO);
                        uint64_t db = make_smem_desc_128b_fn(B_smem + s * B_ELEMS + 3 * 16, LBO, SBO);
                        umma_f16_cg2_fn(toff, da, db, idesc, 1u);
                    }
                    umma_commit_2sm_fn(&acc_ready[buf]);
                }
                iter++;
            }
        }
    } else if (wid < 4) {
        // ---- Epilogue warpgroup (warps 0-3) ----
        int j = 0;
        for (int tile_id = cluster_c; tile_id < total_tiles; tile_id += num_clusters, j++) {
            int buf = j & 1; int t = j >> 1;
            mbarrier_wait_fn(&acc_ready[buf], (uint32_t)(t & 1));
            tcgen05_fence_after_fn();
            uint32_t toff = tmem_base + (uint32_t)(buf * CN);
            #pragma unroll
            for (int c0 = 0; c0 < CN; c0 += 32) {
                uint32_t rg[32];
                #pragma unroll
                for (int q = 0; q < 32; q += 4)
                    tmem_load_4x_fn(toff + (uint32_t)(c0 + q), &rg[q], &rg[q+1], &rg[q+2], &rg[q+3]);
                tmem_load_fence_fn();
                int base = tid * CN + c0;
                #pragma unroll
                for (int q = 0; q < 32; q++) staging[base + q] = __float2bfloat16(__uint_as_float(rg[q]));
            }
            named_barrier_sync_fn(1, 128);
            if (tid == 0) {
                if (rank == 0) mbarrier_arrive_fn(&epi_done[buf]);
                else mbarrier_arrive_cluster_fn(&epi_done[buf], 0);
            }
            int m_tile, n_tile; get_tile(tile_id, total_m, total_n, m_tile, n_tile);
            int gm = m_tile * CM + (int)rank * BM;
            int gn = n_tile * CN;
            constexpr int NUINT2 = (BM * CN) / 4;
            constexpr int PER = CN / 4;
            for (int i = tid; i < NUINT2; i += 128) {
                int row = i / PER; int col = (i % PER) * 4;
                int gr = gm + row;
                if (gr < M) {
                    uint2 data = *reinterpret_cast<uint2*>(&staging[row * CN + col]);
                    *reinterpret_cast<uint2*>(&C[(int64_t)gr * N + gn + col]) = data;
                }
            }
            named_barrier_sync_fn(1, 128);
        }
    }

    __syncthreads();
    cluster_sync_fn();
    if (wid == 0) tmem_dealloc_fn(tmem_base, TMEM_TOTAL);
}

static CUresult make_tma_desc(CUtensorMap* d, void* gptr, uint64_t inner, uint64_t outer,
                              uint32_t binner, uint32_t bouter) {
    cuuint64_t gdim[2] = {(cuuint64_t)inner, (cuuint64_t)outer};
    cuuint64_t gstride[1] = {(cuuint64_t)(inner * 2)};
    cuuint32_t bdim[2] = {binner, bouter};
    cuuint32_t estride[2] = {1, 1};
    return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, gptr, gdim, gstride,
        bdim, estride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    int M = (int)A.size(0);
    int K = (int)A.size(1);
    int N = (int)B.size(0);
    __nv_bfloat16* aptr = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* bptr = static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* cptr = static_cast<__nv_bfloat16*>(C.data_ptr());

    CUtensorMap dA, dB;
    CU_CHECK(make_tma_desc(&dA, aptr, (uint64_t)K, (uint64_t)M, BK, BM));
    CU_CHECK(make_tma_desc(&dB, bptr, (uint64_t)K, (uint64_t)N, BK, BN));

    static bool attr_set = false;
    if (!attr_set) {
        CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
        attr_set = true;
    }

    int sm_count = 148;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, A.device().device_id);
    int num_clusters = sm_count / 2;
    int total_tiles = ((M + CM - 1) / CM) * (N / CN);
    if (num_clusters > total_tiles) num_clusters = total_tiles;
    if (num_clusters < 1) num_clusters = 1;
    int grid = num_clusters * 2;

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(grid, 1, 1);
    config.blockDim = dim3(THREADS, 1, 1);
    config.dynamicSmemBytes = SMEM_BYTES;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2; attrs[0].val.clusterDim.y = 1; attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs; config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, dA, dB, cptr, M, N, K, num_clusters));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_sm100::run);

}  // namespace gemm_sm100