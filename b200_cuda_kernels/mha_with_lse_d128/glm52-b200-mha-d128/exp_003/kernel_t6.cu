#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
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
    CUresult _e = (call); \
    if (_e != CUDA_SUCCESS) { \
        const char* errStr; \
        cuGetErrorString(_e, &errStr); \
        fprintf(stderr, "CU error %s at %s:%d\n", \
                errStr ? errStr : "unknown", __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace mha_attention {

constexpr int D = 128;
constexpr int BR = 64;
constexpr int BC = 64;
constexpr int DH = 64;
constexpr int WARPS = 4;
constexpr int THREADS = WARPS * 32;
constexpr int TILE_BYTES = 8192; // 64 rows * 128 bytes/row
constexpr float LN2_RCP = 1.4426950408889634f;

// ==================== TMA ====================

__device__ __forceinline__ void tma_load_2d(const CUtensorMap* d, uint64_t* bar,
    void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

CUresult create_tma_desc(CUtensorMap* d, void* globalAddress,
    uint64_t gmem_inner, uint64_t gmem_outer,
    uint32_t smem_inner, uint32_t smem_outer,
    CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[2] = {gmem_inner, gmem_outer};
    cuuint64_t globalStrides[1] = {gmem_inner * 2};
    cuuint32_t boxDim[2] = {smem_inner, smem_outer};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        globalAddress, globalDim, globalStrides,
        boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

// ==================== MBarrier ====================

__device__ __forceinline__ void init_mbar(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_mbar_init() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n.reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase));
}

__device__ __forceinline__ void fence_proxy_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

// ==================== TMEM ====================

__device__ __forceinline__ void tmem_alloc_cg1(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_cg1(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
        :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmem_load_4x(uint32_t col,
    uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_store_4x(uint32_t col,
    uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
        :: "r"(col), "r"(r0), "r"(r1), "r"(r2), "r"(r3));
}

__device__ __forceinline__ void tmem_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_wait_st() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

// ==================== UMMA ====================

__device__ __forceinline__ void umma_f16_cg1(
    uint32_t tmem_d, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_d), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_cg1(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "r"(a));
}

// ==================== Descriptors ====================

__device__ __forceinline__ uint64_t make_smem_desc_128B(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)((addr & 0x3FFFF) >> 4);
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   // version = 1
    uint32_t base_offset = (addr >> 7) & 0x7;
    d |= (uint64_t)base_offset << 49;
    d |= (uint64_t)2 << 61;   // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc(uint32_t M, uint32_t N, uint32_t a_major, uint32_t b_major) {
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

// ==================== Kernel ====================

__global__ __launch_bounds__(THREADS)
void mha_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S) {

    int bh = blockIdx.x;
    int q_blk = blockIdx.y;
    int b = bh / H;
    int h = bh % H;
    int q_start = q_blk * BR;
    if (q_start >= S) return;

    int bh_s_start = (b * H + h) * S;
    __nv_bfloat16* O_bh = O + ((size_t)(b * H + h) * S) * D;
    float* LSE_bh = LSE + (size_t)(b * H + h) * S;

    extern __shared__ char smem_buf[];
    // Align to 1024 bytes for 128B swizzling
    uintptr_t smem_int = (uintptr_t)smem_buf;
    smem_int = (smem_int + 1023) & ~1023;
    char* smem_base = (char*)smem_int;

    char* smem_Q0 = smem_base;
    char* smem_Q1 = smem_base + TILE_BYTES;
    char* smem_K0 = smem_base + 2 * TILE_BYTES;
    char* smem_K1 = smem_base + 3 * TILE_BYTES;
    char* smem_V0 = smem_base + 4 * TILE_BYTES;
    char* smem_V1 = smem_base + 5 * TILE_BYTES;
    char* smem_P  = smem_base + 6 * TILE_BYTES;

    __shared__ uint64_t mbar_tma;
    __shared__ uint64_t mbar_mma;
    __shared__ uint32_t tmem_addr_smem;
    __shared__ float smem_m[BR];
    __shared__ float smem_l[BR];
    __shared__ float smem_rescale[BR];

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;

    // Initialize mbarriers
    if (tid == 0) {
        init_mbar(&mbar_tma, 1);
        init_mbar(&mbar_mma, 1);
        fence_mbar_init();
    }
    __syncthreads();

    // Allocate TMEM: 256 columns (64 S + 64 O0 + 64 O1)
    if (warp_id == 0 && lane == 0) {
        tmem_alloc_cg1(&tmem_addr_smem, 256);
    }
    __syncthreads();
    uint32_t tmem_base = tmem_addr_smem;
    uint32_t tmem_S  = tmem_base;
    uint32_t tmem_O0 = tmem_base + 64;
    uint32_t tmem_O1 = tmem_base + 128;

    // Init m, l
    if (tid < BR) {
        smem_m[tid] = -INFINITY;
        smem_l[tid] = 0.0f;
    }
    __syncthreads();

    uint32_t phase_tma = 0;
    uint32_t phase_mma = 0;

    // Load Q (2 TMA loads) - ONLY thread 0 issues TMA
    if (tid == 0) {
        mbar_arrive_expect_tx(&mbar_tma, 2 * TILE_BYTES);
        tma_load_2d(&tma_Q, &mbar_tma, smem_Q0, 0, bh_s_start + q_start);
        tma_load_2d(&tma_Q, &mbar_tma, smem_Q1, DH, bh_s_start + q_start);
    }
    mbar_wait(&mbar_tma, phase_tma);
    phase_tma ^= 1;
    __syncthreads();

    const float scale = 1.0f / sqrtf((float)D);
    int n_kv = (S + BC - 1) / BC;

    // Instruction descriptors
    // QK^T: A=Q is K-major, B=K^T is K-major (D is contiguous in both)
    uint32_t idesc_qkt = make_instr_desc(64, 64, 0, 0);
    // PV: A=P is K-major, B=V is MN-major (N/D is contiguous in shared memory)
    uint32_t idesc_pv = make_instr_desc(64, 64, 0, 1);

    // Descriptor constants
    constexpr uint32_t KMAJOR_LBO = 1;
    constexpr uint32_t KMAJOR_SBO = 1024;
    constexpr uint32_t NMAJOR_LBO = 8192;
    constexpr uint32_t NMAJOR_SBO = 1024;

    for (int kv = 0; kv < n_kv; kv++) {
        int kv_start = kv * BC;
        int valid_kv = min(BC, S - kv_start);

        // Load K, V (4 TMA loads) - ONLY thread 0
        if (tid == 0) {
            mbar_arrive_expect_tx(&mbar_tma, 4 * TILE_BYTES);
            tma_load_2d(&tma_K, &mbar_tma, smem_K0, 0, bh_s_start + kv_start);
            tma_load_2d(&tma_K, &mbar_tma, smem_K1, DH, bh_s_start + kv_start);
            tma_load_2d(&tma_V, &mbar_tma, smem_V0, 0, bh_s_start + kv_start);
            tma_load_2d(&tma_V, &mbar_tma, smem_V1, DH, bh_s_start + kv_start);
        }
        mbar_wait(&mbar_tma, phase_tma);
        phase_tma ^= 1;
        __syncthreads();

        // ===== QK^T: S = Q @ K^T =====
        // Both Q and K are K-major in shared memory (D is contiguous)
        // K offset for K-major: base + k * 32 bytes
        if (tid == 0) {
            for (int k = 0; k < 4; k++) {
                uint64_t desc_a = make_smem_desc_128B(smem_Q0 + k * 32, KMAJOR_LBO, KMAJOR_SBO);
                uint64_t desc_b = make_smem_desc_128B(smem_K0 + k * 32, KMAJOR_LBO, KMAJOR_SBO);
                umma_f16_cg1(tmem_S, desc_a, desc_b, idesc_qkt, (k > 0) ? 1 : 0);
            }
            for (int k = 0; k < 4; k++) {
                uint64_t desc_a = make_smem_desc_128B(smem_Q1 + k * 32, KMAJOR_LBO, KMAJOR_SBO);
                uint64_t desc_b = make_smem_desc_128B(smem_K1 + k * 32, KMAJOR_LBO, KMAJOR_SBO);
                umma_f16_cg1(tmem_S, desc_a, desc_b, idesc_qkt, 1);
            }
            umma_commit_cg1(&mbar_mma);
        }
        mbar_wait(&mbar_mma, phase_mma);
        phase_mma ^= 1;
        __syncthreads();

        // ===== Softmax Pass 1: Row Max =====
        if (warp_id < 2) {
            int row = warp_id * 32 + lane;
            float row_max = -INFINITY;
            for (int c = 0; c < 64; c += 4) {
                uint32_t r0, r1, r2, r3;
                tmem_load_4x(tmem_S + c, &r0, &r1, &r2, &r3);
                tmem_wait_ld();
                float s0 = __uint_as_float(r0) * scale;
                float s1 = __uint_as_float(r1) * scale;
                float s2 = __uint_as_float(r2) * scale;
                float s3 = __uint_as_float(r3) * scale;
                if (c >= valid_kv)     s0 = -INFINITY;
                if (c+1 >= valid_kv)   s1 = -INFINITY;
                if (c+2 >= valid_kv)   s2 = -INFINITY;
                if (c+3 >= valid_kv)   s3 = -INFINITY;
                row_max = fmaxf(row_max, fmaxf(fmaxf(s0, s1), fmaxf(s2, s3)));
            }
            float m_old = smem_m[row];
            float m_new = fmaxf(m_old, row_max);
            float rf = (m_old == -INFINITY) ? 1.0f : exp2f((m_old - m_new) * LN2_RCP);
            smem_rescale[row] = rf;
            smem_m[row] = m_new;
        }
        __syncthreads();

        // ===== Rescale O in TMEM =====
        if (kv > 0) {
            if (warp_id < 2) {
                int row = warp_id * 32 + lane;
                float rf = smem_rescale[row];
                for (int c = 0; c < 64; c += 4) {
                    uint32_t r0, r1, r2, r3;
                    tmem_load_4x(tmem_O0 + c, &r0, &r1, &r2, &r3);
                    tmem_wait_ld();
                    r0 = __float_as_uint(__uint_as_float(r0) * rf);
                    r1 = __float_as_uint(__uint_as_float(r1) * rf);
                    r2 = __float_as_uint(__uint_as_float(r2) * rf);
                    r3 = __float_as_uint(__uint_as_float(r3) * rf);
                    tmem_store_4x(tmem_O0 + c, r0, r1, r2, r3);
                }
                for (int c = 0; c < 64; c += 4) {
                    uint32_t r0, r1, r2, r3;
                    tmem_load_4x(tmem_O1 + c, &r0, &r1, &r2, &r3);
                    tmem_wait_ld();
                    r0 = __float_as_uint(__uint_as_float(r0) * rf);
                    r1 = __float_as_uint(__uint_as_float(r1) * rf);
                    r2 = __float_as_uint(__uint_as_float(r2) * rf);
                    r3 = __float_as_uint(__uint_as_float(r3) * rf);
                    tmem_store_4x(tmem_O1 + c, r0, r1, r2, r3);
                }
                tmem_wait_st();
            }
            __syncthreads();
        }

        // ===== Softmax Pass 2: P and Row Sum =====
        if (warp_id < 2) {
            int row = warp_id * 32 + lane;
            float m_new = smem_m[row];
            float rs = 0.0f;
            for (int c = 0; c < 64; c += 4) {
                uint32_t r0, r1, r2, r3;
                tmem_load_4x(tmem_S + c, &r0, &r1, &r2, &r3);
                tmem_wait_ld();
                float s0 = __uint_as_float(r0) * scale;
                float s1 = __uint_as_float(r1) * scale;
                float s2 = __uint_as_float(r2) * scale;
                float s3 = __uint_as_float(r3) * scale;
                if (c >= valid_kv)   s0 = -INFINITY;
                if (c+1 >= valid_kv) s1 = -INFINITY;
                if (c+2 >= valid_kv) s2 = -INFINITY;
                if (c+3 >= valid_kv) s3 = -INFINITY;
                float p0 = (s0 == -INFINITY) ? 0.0f : exp2f((s0 - m_new) * LN2_RCP);
                float p1 = (s1 == -INFINITY) ? 0.0f : exp2f((s1 - m_new) * LN2_RCP);
                float p2 = (s2 == -INFINITY) ? 0.0f : exp2f((s2 - m_new) * LN2_RCP);
                float p3 = (s3 == -INFINITY) ? 0.0f : exp2f((s3 - m_new) * LN2_RCP);
                rs += p0 + p1 + p2 + p3;
                // Store P to 128B swizzled shared memory (K-major)
                __nv_bfloat16 pvals[4] = {
                    __float2bfloat16(p0), __float2bfloat16(p1),
                    __float2bfloat16(p2), __float2bfloat16(p3)
                };
                int chunk_j = c / 8;
                int chunk_off = (c % 8) / 4;
                int swizzled_chunk = chunk_j ^ (row % 8);
                int byte_off = row * 128 + swizzled_chunk * 16 + chunk_off * 8;
                *reinterpret_cast<uint2*>(smem_P + byte_off) = *reinterpret_cast<uint2*>(pvals);
            }
            float rf = smem_rescale[row];
            smem_l[row] = smem_l[row] * rf + rs;
        }
        __syncthreads();

        // Fence: generic proxy writes (P) must be visible to async proxy (MMA)
        if (tid == 0) fence_proxy_async_shared();
        __syncthreads();

        // ===== PV: O += P @ V =====
        // P is K-major, V is MN-major (N-major) in shared memory
        // K offset for K-major P: base + k * 32 bytes
        // K offset for MN-major V: base + k * 2048 bytes (k*16 rows * 128 bytes/row, 8-row cores)
        if (tid == 0) {
            uint32_t accum_first = (kv > 0) ? 1 : 0;
            // O0 += P @ V0
            for (int k = 0; k < 4; k++) {
                uint64_t desc_a = make_smem_desc_128B(smem_P + k * 32, KMAJOR_LBO, KMAJOR_SBO);
                uint64_t desc_b = make_smem_desc_128B(smem_V0 + k * 2048, NMAJOR_LBO, NMAJOR_SBO);
                umma_f16_cg1(tmem_O0, desc_a, desc_b, idesc_pv, (k > 0) ? 1 : accum_first);
            }
            // O1 += P @ V1
            for (int k = 0; k < 4; k++) {
                uint64_t desc_a = make_smem_desc_128B(smem_P + k * 32, KMAJOR_LBO, KMAJOR_SBO);
                uint64_t desc_b = make_smem_desc_128B(smem_V1 + k * 2048, NMAJOR_LBO, NMAJOR_SBO);
                umma_f16_cg1(tmem_O1, desc_a, desc_b, idesc_pv, (k > 0) ? 1 : accum_first);
            }
            umma_commit_cg1(&mbar_mma);
        }
        mbar_wait(&mbar_mma, phase_mma);
        phase_mma ^= 1;
        __syncthreads();
    }

    // ===== Epilogue =====
    float* smem_O_out = reinterpret_cast<float*>(smem_K0); // Reuse K buffer

    if (warp_id < 2) {
        int row = warp_id * 32 + lane;
        float l = smem_l[row];
        float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
        for (int c = 0; c < 64; c += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x(tmem_O0 + c, &r0, &r1, &r2, &r3);
            tmem_wait_ld();
            smem_O_out[row * 128 + c + 0] = __uint_as_float(r0) * inv_l;
            smem_O_out[row * 128 + c + 1] = __uint_as_float(r1) * inv_l;
            smem_O_out[row * 128 + c + 2] = __uint_as_float(r2) * inv_l;
            smem_O_out[row * 128 + c + 3] = __uint_as_float(r3) * inv_l;
        }
        for (int c = 0; c < 64; c += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x(tmem_O1 + c, &r0, &r1, &r2, &r3);
            tmem_wait_ld();
            smem_O_out[row * 128 + 64 + c + 0] = __uint_as_float(r0) * inv_l;
            smem_O_out[row * 128 + 64 + c + 1] = __uint_as_float(r1) * inv_l;
            smem_O_out[row * 128 + 64 + c + 2] = __uint_as_float(r2) * inv_l;
            smem_O_out[row * 128 + 64 + c + 3] = __uint_as_float(r3) * inv_l;
        }
    }
    __syncthreads();

    // Coalesced vectorized store to global
    constexpr int I4_PER_ROW = D / 8;
    for (int i = tid; i < BR * I4_PER_ROW; i += THREADS) {
        int rrow = i / I4_PER_ROW;
        int col8 = i % I4_PER_ROW;
        int q_idx = q_start + rrow;
        if (q_idx < S) {
            __nv_bfloat16 vals[8];
            #pragma unroll
            for (int j = 0; j < 8; j++)
                vals[j] = __float2bfloat16(smem_O_out[rrow * 128 + col8 * 8 + j]);
            *reinterpret_cast<int4*>(&O_bh[q_idx * D + col8 * 8]) = *reinterpret_cast<int4*>(vals);
        }
    }

    // Store LSE
    for (int i = tid; i < BR; i += THREADS) {
        int q_idx = q_start + i;
        if (q_idx < S) {
            float lv = smem_l[i];
            float mv = smem_m[i];
            LSE_bh[q_idx] = mv + logf(lv);
        }
    }
    __syncthreads();

    // Deallocate TMEM
    if (warp_id == 0 && lane == 0) {
        tmem_dealloc_cg1(tmem_base, 256);
    }
}

// ==================== Host ====================

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    const int B = 4;
    const int H = 48;
    int64_t S = Q.size(2);

    const __nv_bfloat16* Q_d = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_d = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_d = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_d = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_d = static_cast<float*>(LSE.data_ptr());

    uint64_t total_rows = (uint64_t)B * H * S;
    CUtensorMap tma_Q, tma_K, tma_V;
    CU_CHECK(create_tma_desc(&tma_Q, (void*)Q_d, D, total_rows, DH, BR, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_desc(&tma_K, (void*)K_d, D, total_rows, DH, BC, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_desc(&tma_V, (void*)V_d, D, total_rows, DH, BC, CU_TENSOR_MAP_SWIZZLE_128B));

    int q_blocks = (S + BR - 1) / BR;
    dim3 grid(B * H, q_blocks);
    dim3 block(THREADS);

    size_t smem_size = 7 * TILE_BYTES + 1024; // 7 tiles + alignment padding

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(mha_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size));

    mha_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, O_d, LSE_d, B, H, (int)S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_attention::run);

}  // namespace mha_attention