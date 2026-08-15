#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <cstdio>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define CU_CHECK(call) do { \
    CUresult _e = (call); \
    if (_e != CUDA_SUCCESS) { \
        const char* errStr; \
        cuGetErrorString(_e, &errStr); \
        fprintf(stderr, "CU error %s at %s:%d\n", errStr, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace tvm_mha {

constexpr int D = 128;
constexpr int BQ = 128; // 64 per CTA
constexpr int BK = 128;
constexpr int NUM_THREADS = 128;

// Helper functions from the provided structural docs
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

__device__ __forceinline__ void tma_load_multicast_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, uint16_t mask) {
    uint64_t cache_hint = 0;
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%0], [%1, {%4, %5}], [%2], %3, %6;"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "h"(mask), "r"(c0), "r"(c1), "l"(cache_hint) : "memory");
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

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmem_load_8x_fn(uint32_t col,
    uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
    uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),
     "=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(col));
}

__device__ __forceinline__ void tmem_store_8x_fn(uint32_t col,
    uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3,
    uint32_t r4, uint32_t r5, uint32_t r6, uint32_t r7) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
   :: "r"(col), "r"(r0),"r"(r1),"r"(r2),"r"(r3),
      "r"(r4),"r"(r5),"r"(r6),"r"(r7));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void st_shared_128_fn(uint32_t addr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
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

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, uint32_t a_major, uint32_t b_major) {
    uint32_t d = 0;
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= (a_major << 15);
    d |= (b_major << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
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

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill);
}

__global__ void mha_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    __nv_bfloat16* O_ptr, float* LSE_ptr, int B, int H, int S) {

    int cta_rank = cluster_rank_fn();
    int bh = blockIdx.x / 2;
    int q_block = (blockIdx.x % 2) + cta_rank * 2; // wait, this is wrong.
    // Actually, grid.x = B*H*S/64. Each CTA handles 64 rows.
    // Cluster has 2 CTAs. CTA0 handles rows q_block*128 .. q_block*128+63.
    // CTA1 handles rows q_block*128+64 .. q_block*128+127.
    int total_q_blocks = S / BQ;
    int bh_q = blockIdx.x;
    bh = bh_q / total_q_blocks;
    int q_block_idx = bh_q % total_q_blocks;
    int q_start = q_block_idx * BQ + cta_rank * (BQ / 2);

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    extern __shared__ char smem[];
    char* base = smem;
    auto align = [&](size_t a) {
        base = (char*)(((uintptr_t)base + a - 1) & ~(a - 1));
    };
    align(128);
    __nv_bfloat16* Q_smem = reinterpret_cast<__nv_bfloat16*>(base); base += 64 * 128 * 2;
    __nv_bfloat16* K_smem = reinterpret_cast<__nv_bfloat16*>(base); base += 128 * 128 * 2;
    __nv_bfloat16* V_smem = reinterpret_cast<__nv_bfloat16*>(base); base += 128 * 128 * 2;
    __nv_bfloat16* P_smem = reinterpret_cast<__nv_bfloat16*>(base); base += 64 * 128 * 2;
    align(8);
    uint64_t* mbar_tma = reinterpret_cast<uint64_t*>(base); base += 2 * 8;
    uint64_t* mbar_q = reinterpret_cast<uint64_t*>(base); base += 8;
    uint64_t* mbar_umma = reinterpret_cast<uint64_t*>(base); base += 2 * 8;
    uint32_t* tmem_addr_smem = reinterpret_cast<uint32_t*>(base); base += 4;
    float* m_smem = reinterpret_cast<float*>(base); base += 64 * 4;
    float* l_smem = reinterpret_cast<float*>(base); base += 64 * 4;

    if (tid == 0) {
        init_smem_barrier_fn(&mbar_tma[0], 1);
        init_smem_barrier_fn(&mbar_tma[1], 1);
        init_smem_barrier_fn(&mbar_q, 1);
        init_smem_barrier_fn(&mbar_umma[0], 1);
        init_smem_barrier_fn(&mbar_umma[1], 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    cluster_sync_fn();

    if (warp_id == 0) {
        tmem_alloc_fn(tmem_addr_smem, 256);
    }
    __syncthreads();
    uint32_t taddr_S = *tmem_addr_smem;
    uint32_t taddr_O = taddr_S + 128;

    const float scale = 0.08838834764831845f;

    // Init m, l
    for (int i = tid; i < 64; i += NUM_THREADS) {
        m_smem[i] = -INFINITY;
        l_smem[i] = 0.0f;
    }

    // Load Q
    mbarrier_arrive_and_expect_tx_fn(&mbar_q, 64 * 128 * 2);
    tma_load_2d_fn(&tma_Q, &mbar_q, Q_smem, 0, bh * S + q_start);
    mbarrier_wait_fn(&mbar_q, 0);

    for (int kv = 0; kv < S; kv += BK) {
        int phase = (kv / BK) % 2;

        // Load K and V
        if (cta_rank == 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_tma[phase], 128 * 128 * 2 * 2);
        }
        tma_load_multicast_2d_fn(&tma_K, &mbar_tma[phase], K_smem, 0, bh * S + kv, 0x3);
        tma_load_multicast_2d_fn(&tma_V, &mbar_tma[phase], V_smem, 0, bh * S + kv, 0x3);
        mbarrier_wait_fn(&mbar_tma[phase], phase);

        // Q @ K^T
        if (cta_rank == 0) {
            uint32_t idesc = make_instr_desc_fn(128, 128, 0, 0);
            for (int k = 0; k < 8; k++) {
                uint64_t a_desc = make_smem_desc_sm100_fn(Q_smem + k * 16, 1, 1024);
                uint64_t b_desc = make_smem_desc_sm100_fn(K_smem + k * 16, 1, 1024);
                umma_f16_cg2_fn(taddr_S, a_desc, b_desc, idesc, (k > 0) ? 1 : 0);
            }
            umma_commit_2sm_fn(&mbar_umma[0]);
        }
        mbarrier_wait_fn(&mbar_umma[0], phase);

        // Softmax
        if (warp_id < 2) {
            int row = lane_id + warp_id * 32;
            float m_old = m_smem[row];
            float l_old = l_smem[row];

            float m_block = -INFINITY;
            for (int c = 0; c < 128; c += 8) {
                uint32_t r[8];
                tmem_load_8x_fn(taddr_S + c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
                tmem_load_fence_fn();
                for (int i = 0; i < 8; i++) {
                    m_block = fmaxf(m_block, __uint_as_float(r[i]));
                }
            }

            float m_new = fmaxf(m_old, m_block);
            float rescale = (m_old == -INFINITY) ? 0.0f : __expf(m_old - m_new);

            // Rescale O
            for (int c = 0; c < 128; c += 8) {
                uint32_t o[8];
                tmem_load_8x_fn(taddr_O + c, &o[0], &o[1], &o[2], &o[3], &o[4], &o[5], &o[6], &o[7]);
                tmem_load_fence_fn();
                for (int i = 0; i < 8; i++) {
                    o[i] = __float_as_uint(__uint_as_float(o[i]) * rescale);
                }
                tmem_store_8x_fn(taddr_O + c, o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7]);
            }

            // Compute P and store to SMEM
            float l_block = 0.0f;
            for (int c = 0; c < 128; c += 8) {
                uint32_t r[8];
                tmem_load_8x_fn(taddr_S + c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
                tmem_load_fence_fn();
                uint32_t p[4];
                for (int i = 0; i < 4; i++) {
                    float f0 = __expf(__uint_as_float(r[i*2]) - m_new);
                    float f1 = __expf(__uint_as_float(r[i*2+1]) - m_new);
                    l_block += f0 + f1;
                    p[i] = pack_bf16_fn(__float_as_uint(f0 * scale), __float_as_uint(f1 * scale));
                }
                int chunk_idx = c / 8;
                uint32_t addr = (uint32_t)__cvta_generic_to_shared(P_smem + row * 128 + (((row % 8) ^ chunk_idx) * 8));
                st_shared_128_fn(addr, p[0], p[1], p[2], p[3]);
            }

            m_smem[row] = m_new;
            l_smem[row] = l_old * rescale + l_block;
        }
        __syncthreads();

        // P @ V
        if (cta_rank == 0) {
            uint32_t idesc_pv = make_instr_desc_fn(128, 128, 0, 1);
            for (int k = 0; k < 8; k++) {
                uint64_t a_desc = make_smem_desc_sm100_fn(P_smem + k * 16, 1, 1024);
                uint64_t b_desc = make_smem_desc_sm100_fn(V_smem + k * 16, 1, 1024);
                umma_f16_cg2_fn(taddr_O, a_desc, b_desc, idesc_pv, (k > 0 || kv > 0) ? 1 : 0);
            }
            umma_commit_2sm_fn(&mbar_umma[1]);
        }
        mbarrier_wait_fn(&mbar_umma[1], phase);
    }

    // Epilogue
    if (warp_id < 2) {
        int row = lane_id + warp_id * 32;
        if (q_start + row < S) {
            float inv_l = 1.0f / l_smem[row];
            float m = m_smem[row];
            LSE_ptr[bh * S + q_start + row] = m + logf(l_smem[row]);

            for (int c = 0; c < 128; c += 8) {
                uint32_t o[8];
                tmem_load_8x_fn(taddr_O + c, &o[0], &o[1], &o[2], &o[3], &o[4], &o[5], &o[6], &o[7]);
                tmem_load_fence_fn();
                uint32_t out[4];
                for (int i = 0; i < 4; i++) {
                    float f0 = __uint_as_float(o[i*2]) * inv_l;
                    float f1 = __uint_as_float(o[i*2+1]) * inv_l;
                    out[i] = pack_bf16_fn(__float_as_uint(f0), __float_as_uint(f1));
                }
                int chunk_idx = c / 8;
                uint32_t addr = (uint32_t)__cvta_generic_to_shared(Q_smem + row * 128 + (((row % 8) ^ chunk_idx) * 8));
                st_shared_128_fn(addr, out[0], out[1], out[2], out[3]);
            }
        }
    }
    __syncthreads();

    // TMA store O
    tma_store_2d_fn(&tma_O, Q_smem, 0, bh * S + q_start);
    tma_store_fence_fn();
    tma_store_commit_fn();
    tma_store_wait_fn<0>();

    __syncthreads();
    if (warp_id == 0) {
        tmem_dealloc_fn(taddr_S, 256);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);

    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUtensorMap tma_Q, tma_K, tma_V, tma_O;
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, (void*)Q_data, 128, B*H*S, 128, 64,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, (void*)K_data, 128, B*H*S, 128, 128,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, (void*)V_data, 128, B*H*S, 128, 128,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_O, (void*)O_data, 128, B*H*S, 128, 64,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    int smem_size = 64*128*2 + 128*128*2 + 128*128*2 + 64*128*2 + 8*5 + 4 + 64*4*2 + 256;

    CUDA_CHECK(cudaFuncSetAttribute(mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    int total_q_blocks = S / BQ;
    int grid_x = B * H * total_q_blocks;
    dim3 grid(grid_x, 1, 1);
    dim3 block(NUM_THREADS, 1, 1);

    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(&config, mha_kernel, tma_Q, tma_K, tma_V, tma_O, O_data, LSE_data, B, H, S));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_mha::run);

}  // namespace tvm_mha