#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} }while(0)
#define CU_CHECK(call) do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){ const char* s=nullptr; \
  cuGetErrorString(_e,&s); fprintf(stderr,"CU error %s at %s:%d\n",s,__FILE__,__LINE__); exit(1);} }while(0)

namespace gemm_kernel_ns {

constexpr int BM_CTA = 128;   // M rows per CTA
constexpr int BN = 256;       // combined N tile
constexpr int BK = 64;        // K per stage
constexpr int NUM_STAGES = 4;

// ---------- device helpers ----------
__device__ __forceinline__ bool elect_one_sync_fn() {
    uint32_t pred;
    asm volatile("{\n.reg .pred p;\n elect.sync _|p, 0xFFFFFFFF;\n selp.b32 %0, 1, 0, p;\n}\n" : "=r"(pred));
    return pred != 0;
}
__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r;
}
__device__ __forceinline__ void cluster_sync_fn() {
    asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n" ::: "memory");
}
__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
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
        "{\n.reg .pred P;\n WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase));
}
__device__ __forceinline__ uint32_t map_to_cta0_fn(uint32_t local_addr) {
    uint32_t r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(local_addr), "r"(0));
    return r;
}
__device__ __forceinline__ void tma_load_cg2_raw(const CUtensorMap* d, uint32_t bar_addr, void* smem, int32_t c0, int32_t c1) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(sa), "l"((uint64_t)d), "r"(c0), "r"(c1), "r"(bar_addr) : "memory");
}
__device__ __forceinline__ void tmem_alloc2(uint32_t* dst, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(ncols));
}
__device__ __forceinline__ void tmem_dealloc2(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(addr), "r"(ncols));
}
__device__ __forceinline__ void umma_f16_cg2(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}
__device__ __forceinline__ void umma_commit_2sm(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
        :: "r"(a), "h"((uint16_t)0x3));
}
__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}
__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ uint64_t make_smem_desc(void* ptr, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    d |= (uint64_t)((addr & 0x3FFFF) >> 4);
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   // version 1 (SM100)
    d |= (uint64_t)2 << 61;   // SWIZZLE_128B
    return d;
}
__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    // c_format FP32
    d |= (1u << 7);    // a_format BF16
    d |= (1u << 10);   // b_format BF16
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

// ---------- kernel ----------
__global__ void gemm_kernel(const __grid_constant__ CUtensorMap dA,
                            const __grid_constant__ CUtensorMap dB,
                            __nv_bfloat16* __restrict__ C,
                            int M, int N, int K, int num_n_tiles) {
    extern __shared__ char smem_raw[];
    uintptr_t base = (uintptr_t)smem_raw;
    uintptr_t ab = (base + 1023u) & ~(uintptr_t)1023u;
    __nv_bfloat16* A_smem = reinterpret_cast<__nv_bfloat16*>(ab);              // NS*128*64
    __nv_bfloat16* B_smem = A_smem + NUM_STAGES * (BM_CTA * BK);               // NS*128*64
    uint64_t* full  = reinterpret_cast<uint64_t*>(B_smem + NUM_STAGES * (128 * BK)); // NS
    uint64_t* empty = full + NUM_STAGES;                                       // NS
    uint64_t* mma_done = empty + NUM_STAGES;                                   // 1
    uint32_t* tmem_ptr = reinterpret_cast<uint32_t*>(mma_done + 1);            // 1

    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    uint32_t rank = cluster_rank_fn();

    int cluster_id = blockIdx.x >> 1;
    int m_comb = cluster_id / num_n_tiles;
    int n_idx  = cluster_id % num_n_tiles;
    int m_base = m_comb * BN;      // combined M base (256)
    int n_base = n_idx  * BN;      // N base (256)
    int m_cta  = m_base + (int)rank * BM_CTA;   // this CTA's M rows
    int n_cta  = n_base + (int)rank * BM_CTA;   // this CTA's B N-cols (split 128/128)

    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < NUM_STAGES; s++) { init_smem_barrier_fn(full + s, 1); init_smem_barrier_fn(empty + s, 1); }
        init_smem_barrier_fn(mma_done, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    if (warp == 0) tmem_alloc2(tmem_ptr, BN);
    __syncthreads();
    cluster_sync_fn();

    uint32_t tmem_base = tmem_ptr[0];
    uint32_t idesc = make_instr_desc_fn(BN, BN); // combined M=256, N=256
    int num_k = K / BK; // 80
    const uint32_t TX = 4u * (uint32_t)(BM_CTA * BK * 2); // 4 loads * 16KB

    // ---- Producer: warp 1 of BOTH CTAs ----
    if (warp == 1) {
        if (elect_one_sync_fn()) {
            uint32_t full_local[NUM_STAGES];
            #pragma unroll
            for (int s = 0; s < NUM_STAGES; s++)
                full_local[s] = (uint32_t)__cvta_generic_to_shared(&full[s]);
            for (int it = 0; it < num_k; it++) {
                int s = it % NUM_STAGES;
                int u = it / NUM_STAGES;
                if (u >= 1) mbarrier_wait_fn(empty + s, (uint32_t)((u - 1) & 1));
                if (rank == 0) mbarrier_arrive_and_expect_tx_fn(full + s, TX);
                uint32_t bar_addr = (rank == 0) ? full_local[s] : map_to_cta0_fn(full_local[s]);
                __nv_bfloat16* As = A_smem + s * (BM_CTA * BK);
                __nv_bfloat16* Bs = B_smem + s * (128 * BK);
                int k0 = it * BK;
                tma_load_cg2_raw(&dA, bar_addr, As, k0, m_cta);
                tma_load_cg2_raw(&dB, bar_addr, Bs, k0, n_cta);
            }
        }
    }

    // ---- Consumer: warp 0 of leader (rank 0) ----
    if (rank == 0 && warp == 0) {
        if (elect_one_sync_fn()) {
            for (int it = 0; it < num_k; it++) {
                int s = it % NUM_STAGES;
                int u = it / NUM_STAGES;
                mbarrier_wait_fn(full + s, (uint32_t)(u & 1));
                __nv_bfloat16* As = A_smem + s * (BM_CTA * BK);
                __nv_bfloat16* Bs = B_smem + s * (128 * BK);
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    uint64_t da = make_smem_desc(As + j * 16, 1024);
                    uint64_t db = make_smem_desc(Bs + j * 16, 1024);
                    uint32_t accum = (it == 0 && j == 0) ? 0u : 1u;
                    umma_f16_cg2(tmem_base, da, db, idesc, accum);
                }
                umma_commit_2sm(empty + s);
            }
            umma_commit_2sm(mma_done);
        }
    }

    // ---- wait all MMAs done ----
    mbarrier_wait_fn(mma_done, 0);
    __syncthreads();

    // ---- Epilogue: TMEM -> SMEM -> coalesced global ----
    __nv_bfloat16* smem_out = A_smem; // reuse (128*256 bf16 = 65536 B)
    #pragma unroll
    for (int col = 0; col < BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(tmem_base + (uint32_t)col, &r0, &r1, &r2, &r3);
        tmem_load_fence_fn();
        uint32_t bp = tid * BN + col;
        smem_out[bp + 0] = __float2bfloat16(__uint_as_float(r0));
        smem_out[bp + 1] = __float2bfloat16(__uint_as_float(r1));
        smem_out[bp + 2] = __float2bfloat16(__uint_as_float(r2));
        smem_out[bp + 3] = __float2bfloat16(__uint_as_float(r3));
    }
    __syncthreads();

    #pragma unroll
    for (int step = 0; step < BM_CTA / 4; step++) {
        int row = step * 4 + warp;
        int grow = m_cta + row;
        int gcol = n_base + lane * 8;
        if (grow < M) {
            uint4 dv = *reinterpret_cast<uint4*>(&smem_out[row * BN + lane * 8]);
            *reinterpret_cast<uint4*>(C + (size_t)grow * N + gcol) = dv;
        }
    }

    __syncthreads();
    cluster_sync_fn();
    if (warp == 0) tmem_dealloc2(tmem_base, BN);
}

// ---------- host ----------
static CUresult make_tma_desc(CUtensorMap* d, void* ptr, uint64_t inner, uint64_t outer,
                              uint32_t box_inner, uint32_t box_outer) {
    cuuint64_t globalDim[2] = {inner, outer};
    cuuint64_t globalStrides[1] = {inner * 2};
    cuuint32_t boxDim[2] = {box_inner, box_outer};
    cuuint32_t elemStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ptr, globalDim, globalStrides, boxDim, elemStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);

    __nv_bfloat16* Aptr = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* Bptr = static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* Cptr = static_cast<__nv_bfloat16*>(C.data_ptr());

    CUtensorMap dA, dB;
    CU_CHECK(make_tma_desc(&dA, Aptr, (uint64_t)K, (uint64_t)M, BK, BM_CTA));
    CU_CHECK(make_tma_desc(&dB, Bptr, (uint64_t)K, (uint64_t)N, BK, 128));

    int num_n_tiles = (int)(N / BN);
    int num_m_comb  = (int)((M + BN - 1) / BN);
    int num_clusters = num_m_comb * num_n_tiles;
    int grid = num_clusters * 2;

    size_t smem_bytes = 1024 + (size_t)NUM_STAGES * (BM_CTA * BK + 128 * BK) * 2 + (2 * NUM_STAGES + 1) * 8 + 16;

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));

    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(grid, 1, 1);
    config.blockDim = dim3(128, 1, 1);
    config.dynamicSmemBytes = smem_bytes;
    config.stream = stream;
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = 2;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;
    config.attrs = attribute;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, dA, dB, Cptr, (int)M, (int)N, (int)K, num_n_tiles));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_kernel_ns::run);

}  // namespace gemm_kernel_ns