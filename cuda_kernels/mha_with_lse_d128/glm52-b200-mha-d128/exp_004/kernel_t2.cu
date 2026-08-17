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
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        const char* errStr;                                        \
        cuGetErrorString(_e, &errStr);                             \
        fprintf(stderr, "CU error %s at %s:%d\n",                  \
                errStr ? errStr : "unknown", __FILE__, __LINE__);  \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_mha_cuda {

// ---- Constants ----
constexpr int D_HEAD = 128;
constexpr int BM = 128;
constexpr int BK = 64;
constexpr int NUM_THREADS = 128;
constexpr float SCALE = 0.08838834764f; // 1/sqrt(128)

constexpr int Q_SMEM_SIZE = 32768;   // 128 rows × 128 BF16 × 2 (2 spans)
constexpr int KV_SMEM_SIZE = 16384;  // 64 rows × 128 BF16 × 2 (2 spans)
constexpr int P_SMEM_SIZE = 16384;   // 128 rows × 64 BF16 × 2 (1 span)
constexpr int ML_SMEM_SIZE = 1024;   // 128 × 4B × 2

constexpr int SMEM_SIZE = Q_SMEM_SIZE + KV_SMEM_SIZE + P_SMEM_SIZE + ML_SMEM_SIZE + 256 + 1024;

// ---- MBarrier operations ----
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

// ---- TMA operations ----
__device__ __forceinline__ void prefetch_tma_descriptor_fn(const CUtensorMap* d) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(d) : "memory");
}

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem,
    int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

// ---- TMEM operations ----
__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_load_8x_fn(uint32_t col,
    uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
    uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),
     "=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(col));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_store_4x_fn(uint32_t col, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1, %2, %3, %4};"
   :: "r"(col), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
}

__device__ __forceinline__ void tmem_store_wait_fn() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

// ---- tcgen05 fence ----
__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

// ---- UMMA operations (cta_group::1) ----
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

// ---- Descriptor helpers ----
__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   // version = 1 (SM100)
    d |= (uint64_t)2 << 61;   // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    // dtype = FP32
    d |= (1u << 7);    // atype = BF16
    d |= (1u << 10);   // btype = BF16
    d |= (0u << 15);   // Transpose A = 0 (K-Major)
    d |= (0u << 16);   // Transpose B = 0 (K-Major)
    d |= ((N / 8) << 17);   // N >> 3
    d |= ((M / 16) << 24);  // M >> 4
    return d;
}

__device__ __forceinline__ uint64_t advance_desc(uint64_t desc, uint32_t byte_offset) {
    uint64_t base = desc & 0x3FFF;
    uint64_t new_base = base + (byte_offset >> 4);
    return (desc & 0xFFFFFFFFFFFFC000ULL) | (new_base & 0x3FFF);
}

// ---- TMA descriptor creation (host) ----
static CUresult create_tma_desc_bf16_4d(
    CUtensorMap* d, void* ptr,
    uint64_t dim_d, uint64_t dim_s, uint64_t dim_h, uint64_t dim_b,
    uint32_t box_d, uint32_t box_s,
    CUtensorMapFloatOOBfill oob_fill)
{
    cuuint64_t globalDim[4] = {dim_d, dim_s, dim_h, dim_b};
    cuuint64_t globalStrides[3] = {dim_d * 2, dim_s * dim_d * 2, dim_h * dim_s * dim_d * 2};
    cuuint32_t boxDim[4] = {box_d, box_s, 1, 1};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
        ptr, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, oob_fill);
}

// ---- Kernel ----
__global__ void attention_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S, int B, int H)
{
    const int q_block = blockIdx.x;
    const int bh = blockIdx.y;
    const int b = bh / H;
    const int h = bh % H;
    const int q_start = q_block * BM;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const bool row_valid = (q_start + tid < S);

    extern __shared__ char smem_raw[];
    // Align to 1024B for 128B swizzle
    char* smem = (char*)(((uintptr_t)smem_raw + 1023) & ~1023);

    __nv_bfloat16* Q_smem = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* KV_smem = Q_smem + Q_SMEM_SIZE / 2;   // 16384 BF16
    __nv_bfloat16* P_smem = KV_smem + KV_SMEM_SIZE / 2;  // 8192 BF16
    float* m_smem = reinterpret_cast<float*>(smem + Q_SMEM_SIZE + KV_SMEM_SIZE + P_SMEM_SIZE);
    float* l_smem = m_smem + BM;
    uint64_t* mbar = reinterpret_cast<uint64_t*>(l_smem + BM);
    uint32_t* tmem_addr_smem = reinterpret_cast<uint32_t*>(mbar + 3);

    // ---- TMEM allocation (warp 0) ----
    if (warp_id == 0) {
        tmem_alloc_cg1_fn(tmem_addr_smem, 256); // 256 columns (8 units of 32)
    }
    __syncthreads();
    uint32_t tmem_base = tmem_addr_smem[0];
    uint32_t tmem_S = tmem_base;           // S at column 0 (64 cols)
    uint32_t tmem_O = tmem_base + BK;      // O at column 64 (128 cols)

    // ---- Init mbarriers ----
    if (tid == 0) {
        init_smem_barrier_fn(&mbar[0], 1);
        init_smem_barrier_fn(&mbar[1], 1);
        init_smem_barrier_fn(&mbar[2], 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    // ---- Init m, l ----
    if (tid < BM) {
        m_smem[tid] = -INFINITY;
        l_smem[tid] = 0.0f;
    }

    // ---- Prefetch TMA descriptors ----
    prefetch_tma_descriptor_fn(&tma_Q);
    prefetch_tma_descriptor_fn(&tma_K);
    prefetch_tma_descriptor_fn(&tma_V);

    // ---- Load Q via 4D TMA (2 loads for D=128) ----
    int phase0 = 0, phase1 = 0, phase2 = 0;
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar[0], 16384);
        tma_load_4d_fn(&tma_Q, &mbar[0], Q_smem, 0, q_start, h, b);
        mbarrier_arrive_and_expect_tx_fn(&mbar[1], 16384);
        tma_load_4d_fn(&tma_Q, &mbar[1], Q_smem + 8192, 64, q_start, h, b);
    }
    mbarrier_wait_fn(&mbar[0], phase0); phase0 ^= 1;
    mbarrier_wait_fn(&mbar[1], phase1); phase1 ^= 1;

    // ---- Build UMMA descriptors ----
    // Q: K-major [M=128, K=128], LBO=1, SBO=1024
    uint64_t q_desc = make_smem_desc_sm100_fn(Q_smem, 1, 1024);
    // K: MN-major [K=128, N=64], LBO=16384, SBO=1024
    uint64_t k_desc = make_smem_desc_sm100_fn(KV_smem, 16384, 1024);
    // P: K-major [M=128, K=64], LBO=1, SBO=1024
    uint64_t p_desc = make_smem_desc_sm100_fn(P_smem, 1, 1024);
    // V: K-major [K=64, N=128], LBO=1, SBO=1024
    uint64_t v_desc = make_smem_desc_sm100_fn(KV_smem, 1, 1024);

    // Instruction descriptors
    // QK^T: A=Q K-major (a_trans=0), B=K^T MN-major (b_trans=1)
    uint32_t idesc_qk = make_instr_desc_fn(BM, BK) | (1u << 16);
    // PV: A=P K-major (a_trans=0), B=V K-major (b_trans=0)
    uint32_t idesc_pv = make_instr_desc_fn(BM, D_HEAD);

    // ---- Main loop over KV blocks ----
    for (int kv_start = 0; kv_start < S; kv_start += BK) {
        int kv_len = min(kv_start + BK, S) - kv_start;
        bool first_block = (kv_start == 0);

        // ---- Load K via 4D TMA (2 loads) ----
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar[0], 8192);
            tma_load_4d_fn(&tma_K, &mbar[0], KV_smem, 0, kv_start, h, b);
            mbarrier_arrive_and_expect_tx_fn(&mbar[1], 8192);
            tma_load_4d_fn(&tma_K, &mbar[1], KV_smem + 4096, 64, kv_start, h, b);
        }
        mbarrier_wait_fn(&mbar[0], phase0); phase0 ^= 1;
        mbarrier_wait_fn(&mbar[1], phase1); phase1 ^= 1;

        // ---- UMMA QK^T (K=128, 8 steps of K=16) ----
        if (tid == 0) {
            #pragma unroll
            for (int k = 0; k < D_HEAD; k += 16) {
                // Q: advance along K (inner dim). K=128 = 2 spans of 64.
                // Span 0 at Q_smem, span 1 at Q_smem + 16384 bytes
                uint32_t q_off = (k / 64) * 16384 + (k % 64) * 2;
                // K^T (MN-major): advance along K (stride dim). SBO=1024 per 8 K-elements.
                uint32_t k_off = (k / 8) * 1024;
                uint64_t a_d = advance_desc(q_desc, q_off);
                uint64_t b_d = advance_desc(k_desc, k_off);
                uint32_t accum = (k == 0) ? 0u : 1u;
                umma_f16_cg1_fn(tmem_S, a_d, b_d, idesc_qk, accum);
            }
            umma_commit_cg1_fn(&mbar[2]);
        }
        mbarrier_wait_fn(&mbar[2], phase2); phase2 ^= 1;
        tcgen05_fence_after_fn();

        // ---- Issue V TMA load (overlap with softmax) ----
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar[0], 8192);
            tma_load_4d_fn(&tma_V, &mbar[0], KV_smem, 0, kv_start, h, b);
            mbarrier_arrive_and_expect_tx_fn(&mbar[1], 8192);
            tma_load_4d_fn(&tma_V, &mbar[1], KV_smem + 4096, 64, kv_start, h, b);
        }

        // ---- Softmax Phase 1: Read S from TMEM, find row max ----
        float m_old = row_valid ? m_smem[tid] : -INFINITY;
        float l_old = row_valid ? l_smem[tid] : 0.0f;
        float row_max = -INFINITY;

        for (int c = 0; c < BK; c += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_S + c, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            if (row_valid) {
                float s0 = __uint_as_float(r0) * SCALE;
                float s1 = __uint_as_float(r1) * SCALE;
                float s2 = __uint_as_float(r2) * SCALE;
                float s3 = __uint_as_float(r3) * SCALE;
                if (c + 0 >= kv_len) s0 = -INFINITY;
                if (c + 1 >= kv_len) s1 = -INFINITY;
                if (c + 2 >= kv_len) s2 = -INFINITY;
                if (c + 3 >= kv_len) s3 = -INFINITY;
                row_max = fmaxf(row_max, fmaxf(fmaxf(s0, s1), fmaxf(s2, s3)));
            }
        }

        float m_new = row_valid ? fmaxf(m_old, row_max) : -INFINITY;

        // ---- Softmax Phase 2: Compute P = exp(S - m_new), write to swizzled SMEM ----
        float row_sum = 0.0f;

        for (int c = 0; c < BK; c += 8) {
            uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
            tmem_load_8x_fn(tmem_S + c, &r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
            tmem_load_fence_fn();

            uint32_t vals[8] = {r0, r1, r2, r3, r4, r5, r6, r7};
            float p[8];

            if (row_valid) {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    float sv = __uint_as_float(vals[j]) * SCALE;
                    if (c + j >= kv_len) sv = -INFINITY;
                    p[j] = __expf(sv - m_new);
                    row_sum += p[j];
                }
            } else {
                #pragma unroll
                for (int j = 0; j < 8; j++) p[j] = 0.0f;
            }

            // Write to 128B-swizzled P_smem
            int sub_chunk = c / 8;  // 0-7
            int phys_sub = (tid % 8) ^ sub_chunk;
            int byte_addr = tid * 128 + phys_sub * 16;
            __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(
                reinterpret_cast<char*>(P_smem) + byte_addr);
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                dst[j] = __float2bfloat16(p[j]);
            }
        }

        // ---- Update m, l in SMEM ----
        if (row_valid) {
            float rescale_factor = (m_old != -INFINITY) ? __expf(m_old - m_new) : 1.0f;
            l_smem[tid] = l_old * rescale_factor + row_sum;
            m_smem[tid] = m_new;
        }

        // ---- Conditional O rescale ----
        bool need_rescale = row_valid && !first_block && (m_old != m_new);
        uint32_t mask = __ballot_sync(0xFFFFFFFF, need_rescale);
        if (mask != 0) {
            float rescale = need_rescale ? __expf(m_old - m_new) : 1.0f;
            for (int c = 0; c < D_HEAD; c += 4) {
                uint32_t r0, r1, r2, r3;
                tmem_load_4x_fn(tmem_O + c, &r0, &r1, &r2, &r3);
                tmem_load_fence_fn();
                r0 = __float_as_uint(__uint_as_float(r0) * rescale);
                r1 = __float_as_uint(__uint_as_float(r1) * rescale);
                r2 = __float_as_uint(__uint_as_float(r2) * rescale);
                r3 = __float_as_uint(__uint_as_float(r3) * rescale);
                tmem_store_4x_fn(tmem_O + c, r0, r1, r2, r3);
            }
            tmem_store_wait_fn();
        }

        // ---- Fence P writes, wait for V ----
        __syncthreads();
        fence_async_shared_fn();
        mbarrier_wait_fn(&mbar[0], phase0); phase0 ^= 1;
        mbarrier_wait_fn(&mbar[1], phase1); phase1 ^= 1;

        // ---- UMMA PV (K=64, 4 steps of K=16) ----
        if (tid == 0) {
            #pragma unroll
            for (int k = 0; k < BK; k += 16) {
                uint32_t p_off = k * 2;
                uint32_t v_off = k * 2;
                uint64_t a_d = advance_desc(p_desc, p_off);
                uint64_t b_d = advance_desc(v_desc, v_off);
                uint32_t accum = (k == 0 && first_block) ? 0u : 1u;
                umma_f16_cg1_fn(tmem_O, a_d, b_d, idesc_pv, accum);
            }
            umma_commit_cg1_fn(&mbar[2]);
        }
        mbarrier_wait_fn(&mbar[2], phase2); phase2 ^= 1;
        tcgen05_fence_after_fn();

        __syncthreads();
    }

    // ---- Final normalization and output ----
    __syncthreads();

    float final_l = row_valid ? l_smem[tid] : 0.0f;
    float final_m = row_valid ? m_smem[tid] : -INFINITY;
    float inv_l = (final_l > 0.0f) ? (1.0f / final_l) : 0.0f;

    // Write LSE
    if (row_valid && final_l > 0.0f) {
        size_t lse_offset = (size_t)(b * H + h) * S + q_start + tid;
        LSE[lse_offset] = final_m + logf(final_l);
    }

    // Phase 1: TMEM -> SMEM (normalized BF16, standard row-major, reuse Q_smem)
    for (int c = 0; c < D_HEAD; c += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(tmem_O + c, &r0, &r1, &r2, &r3);
        tmem_load_fence_fn();
        float f0 = __uint_as_float(r0) * inv_l;
        float f1 = __uint_as_float(r1) * inv_l;
        float f2 = __uint_as_float(r2) * inv_l;
        float f3 = __uint_as_float(r3) * inv_l;
        int base = tid * D_HEAD + c;
        Q_smem[base + 0] = __float2bfloat16(f0);
        Q_smem[base + 1] = __float2bfloat16(f1);
        Q_smem[base + 2] = __float2bfloat16(f2);
        Q_smem[base + 3] = __float2bfloat16(f3);
    }
    __syncthreads();

    // Phase 2: SMEM -> Global (coalesced uint2 writes, 4 warps)
    size_t out_offset = (size_t)(b * H + h) * S * D_HEAD;
    __nv_bfloat16* O_bh = O + out_offset;
    int wid = tid / 32;
    int lid = tid % 32;
    int num_steps = BM / 4;
    for (int step = 0; step < num_steps; ++step) {
        int srow = step * 4 + wid;
        int global_row = q_start + srow;
        if (global_row >= S) continue;
        int col_start = lid * 4;
        uint2 data = *reinterpret_cast<uint2*>(&Q_smem[srow * D_HEAD + col_start]);
        *reinterpret_cast<uint2*>(O_bh + (size_t)global_row * D_HEAD + col_start) = data;
    }

    // ---- TMEM deallocation ----
    __syncthreads();
    if (warp_id == 0) {
        tmem_dealloc_cg1_fn(tmem_base, 256);
    }
}

// ---- Host function ----
void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    const int B = static_cast<int>(Q.size(0));
    const int H = static_cast<int>(Q.size(1));
    const int S = static_cast<int>(Q.size(2));
    const int D = static_cast<int>(Q.size(3));
    (void)D;

    __nv_bfloat16* Q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* K_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* V_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE.data_ptr());

    // Create 4D TMA descriptors: [D, S, H, B]
    CUtensorMap tma_Q, tma_K, tma_V;
    CU_CHECK(create_tma_desc_bf16_4d(&tma_Q, Q_ptr, D_HEAD, S, H, B, 64, BM,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA));
    CU_CHECK(create_tma_desc_bf16_4d(&tma_K, K_ptr, D_HEAD, S, H, B, 64, BK,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA));
    CU_CHECK(create_tma_desc_bf16_4d(&tma_V, V_ptr, D_HEAD, S, H, B, 64, BK,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA));

    const int num_q_blocks = (S + BM - 1) / BM;
    dim3 grid(num_q_blocks, B * H);
    dim3 block(NUM_THREADS);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(attention_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));

    attention_kernel<<<grid, block, SMEM_SIZE, stream>>>(
        tma_Q, tma_K, tma_V, O_ptr, LSE_ptr, S, B, H);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_cuda::run);

}  // namespace tvm_ffi_mha_cuda