#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdint.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__);} } while(0)
#define CU_CHECK(call) do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){ const char* s; cuGetErrorString(_e,&s); fprintf(stderr,"CU error %s at %s:%d\n", s, __FILE__,__LINE__);} } while(0)

namespace gemm_sm100 {

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
__device__ __forceinline__ uint64_t make_smem_desc_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)((addr & 0x3FFFF) >> 4);
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}
__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);   // c FP32
    d |= (1u << 7);   // a BF16
    d |= (1u << 10);  // b BF16
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
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

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 64;
constexpr int STAGES = 4;
constexpr int CM = 256;
constexpr int CN = 256;
constexpr int TMEM_N = 256;
constexpr int A_ELEMS = BM * BK;
constexpr int B_ELEMS = BN * BK;
constexpr int A_BYTES = A_ELEMS * 2;
constexpr int B_BYTES = B_ELEMS * 2;
constexpr uint32_t TX_BYTES = (uint32_t)(A_BYTES + B_BYTES) * 2;

__global__ __launch_bounds__(128) void gemm_kernel(
    const __grid_constant__ CUtensorMap tmaA,
    const __grid_constant__ CUtensorMap tmaB,
    __nv_bfloat16* __restrict__ C, int M, int N, int K) {

    extern __shared__ __align__(1024) uint8_t smem[];
    __nv_bfloat16* A_smem = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* B_smem = reinterpret_cast<__nv_bfloat16*>(smem + STAGES * A_BYTES);
    uint8_t* bar_base = smem + STAGES * (A_BYTES + B_BYTES);
    uint64_t* full = reinterpret_cast<uint64_t*>(bar_base);
    uint64_t* empty = full + STAGES;
    uint64_t* mma_done = empty + STAGES;
    uint32_t* tmem_ptr = reinterpret_cast<uint32_t*>(mma_done + 1);
    __nv_bfloat16* smem_out = A_smem;

    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;
    uint32_t rank = cluster_rank_fn();
    int cluster_id = blockIdx.x >> 1;
    int num_n_tiles = N / CN;
    int m_tile = cluster_id / num_n_tiles;
    int n_tile = cluster_id % num_n_tiles;
    int m_base = m_tile * CM;
    int n_base = n_tile * CN;
    int m_a_base = m_base + (int)rank * BM;
    int n_b_base = n_base + (int)rank * BN;
    int numK = K / BK;

    if (tid == 0) {
        for (int s = 0; s < STAGES; s++) {
            init_smem_barrier_fn(&full[s], 1);
            init_smem_barrier_fn(&empty[s], 1);
        }
        init_smem_barrier_fn(mma_done, 1);
    }
    fence_smem_barrier_init_fn();
    cluster_sync_fn();

    if (wid == 0) {
        tmem_alloc_fn(tmem_ptr, TMEM_N);
        tmem_relinquish_fn();
    }
    __syncthreads();
    uint32_t tmem_base = *tmem_ptr;
    uint32_t idesc = make_instr_desc_fn(CM, CN);

    if (wid == 1 && lane == 0) {
        // producer
        for (int k = 0; k < numK; k++) {
            int s = k % STAGES;
            int r = k / STAGES;
            if (k >= STAGES) {
                mbarrier_wait_fn(&empty[s], (uint32_t)((r - 1) & 1));
            }
            if (rank == 0) {
                mbarrier_arrive_and_expect_tx_fn(&full[s], TX_BYTES);
            }
            tma_load_2d_cg2_fn(&tmaA, &full[s], A_smem + s * A_ELEMS, k * BK, m_a_base);
            tma_load_2d_cg2_fn(&tmaB, &full[s], B_smem + s * B_ELEMS, k * BK, n_b_base);
        }
    } else if (wid == 0 && rank == 0 && lane == 0) {
        // consumer (MMA), leader CTA only
        for (int k = 0; k < numK; k++) {
            int s = k % STAGES;
            int r = k / STAGES;
            mbarrier_wait_fn(&full[s], (uint32_t)(r & 1));
            for (int ksub = 0; ksub < 4; ksub++) {
                uint64_t da = make_smem_desc_fn(A_smem + s * A_ELEMS + ksub * 16, 1, 1024);
                uint64_t db = make_smem_desc_fn(B_smem + s * B_ELEMS + ksub * 16, 1, 1024);
                uint32_t accum = (k == 0 && ksub == 0) ? 0u : 1u;
                umma_f16_cg2_fn(tmem_base, da, db, idesc, accum);
            }
            if (k + STAGES < numK) {
                umma_commit_2sm_fn(&empty[s]);
            }
        }
        umma_commit_2sm_fn(mma_done);
    }

    __syncthreads();
    mbarrier_wait_fn(mma_done, 0);
    __syncthreads();
    tcgen05_fence_after_fn();

    // Epilogue Phase 1: TMEM -> smem_out
    for (int col = 0; col < TMEM_N; col += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(tmem_base + (uint32_t)col, &r0, &r1, &r2, &r3);
        tmem_load_fence_fn();
        int base = tid * TMEM_N + col;
        smem_out[base + 0] = __float2bfloat16(__uint_as_float(r0));
        smem_out[base + 1] = __float2bfloat16(__uint_as_float(r1));
        smem_out[base + 2] = __float2bfloat16(__uint_as_float(r2));
        smem_out[base + 3] = __float2bfloat16(__uint_as_float(r3));
    }
    __syncthreads();

    // Epilogue Phase 2: smem_out -> C (coalesced)
    int global_m_start = m_base + (int)rank * BM;
    int global_n_start = n_base;
    constexpr int NUINT2 = (BM * TMEM_N) / 4;
    constexpr int PER = TMEM_N / 4;
    for (int i = tid; i < NUINT2; i += 128) {
        int row = i / PER;
        int col = (i % PER) * 4;
        int gr = global_m_start + row;
        int gc = global_n_start + col;
        if (gr < M) {
            uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * TMEM_N + col]);
            *reinterpret_cast<uint2*>(&C[(int64_t)gr * N + gc]) = data;
        }
    }

    __syncthreads();
    cluster_sync_fn();
    if (wid == 0) {
        tmem_dealloc_fn(tmem_base, TMEM_N);
    }
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

    CU