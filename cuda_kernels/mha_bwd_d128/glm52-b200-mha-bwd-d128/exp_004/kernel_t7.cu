#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

namespace flash_attn_bwd {

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
        const char* _err; \
        cuGetErrorString(_r, &_err); \
        fprintf(stderr, "CU error %s at %s:%d\n", _err, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

constexpr int D = 128;
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int THREADS = 128;

// ---- Descriptor helpers ----

__device__ __forceinline__ uint64_t make_sdesc_kmajor(void* ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)0 << 16;
    d |= (uint64_t)64 << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint64_t make_sdesc_mmajor(void* ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)1024 << 16;
    d |= (uint64_t)64 << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint64_t desc_off(uint64_t base, uint32_t bytes) {
    uint64_t d = base;
    uint32_t old_addr = (uint32_t)((d & 0x3FFF) << 4);
    uint32_t new_addr = old_addr + bytes;
    d &= ~((uint64_t)0x3FFF);
    d |= (uint64_t)(new_addr & 0x3FFFF) >> 4;
    return d;
}

__device__ __forceinline__ uint32_t k_kmajor_off(int k) {
    return (k / 4) * 16384 + (k % 4) * 32;
}

__device__ __forceinline__ uint32_t k_mmajor_off(int k) {
    return k * 2048;
}

__device__ __forceinline__ uint32_t make_idesc(uint32_t ta, uint32_t tb) {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= (ta << 15);
    d |= (tb << 16);
    d |= (16u << 17);
    d |= (8u << 24);
    return d;
}

// ---- UMMA ----

__device__ __forceinline__ void umma(uint32_t td, uint64_t da, uint64_t db, uint32_t id, uint32_t acc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(td), "l"(da), "l"(db), "r"(id), "r"(acc));
}

__device__ __forceinline__ void umma_commit(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(a));
}

__device__ __forceinline__ void tcgen05_fence_after() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

// ---- TMEM ----

__device__ __forceinline__ void tmem_alloc(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmem_ld8(uint32_t col,
    uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
    uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),
     "=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(col));
}

__device__ __forceinline__ void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }

// ---- TMA ----

__device__ __forceinline__ void tma_load(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d), "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

// ---- Mbarrier ----

__device__ __forceinline__ void mbarrier_init(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void mbarrier_arrive_etx(uint64_t* bar, uint32_t tx) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx) : "memory");
}

__device__ __forceinline__ void mbarrier_wait(uint64_t* bar, uint32_t phase) {
    asm volatile("{\n.reg .pred P;\nWAIT_%=:\nmbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n@!P bra WAIT_%=;\n}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase));
}

__device__ __forceinline__ void fence_mbarrier_init() { asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory"); }
__device__ __forceinline__ void fence_async_shared() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }

// ---- Swizzled SMEM addressing ----

__device__ __forceinline__ uint32_t swiz_addr(uint32_t base, int k, int m) {
    int atom = k / 64;
    int k_within = k % 64;
    int k_chunk = k_within / 8;
    int k_offset = k_within % 8;
    int m_group = m / 8;
    int m_within = m % 8;
    return base + atom * 16384 + m_group * 1024 + ((m_within ^ k_chunk) * 128) + k_offset * 2;
}

__device__ __forceinline__ void store_bf16_smem(uint32_t addr, __nv_bfloat16 val) {
    uint16_t v = *reinterpret_cast<uint16_t*>(&val);
    asm volatile("st.shared.b16 [%0], %1;" :: "r"(addr), "h"(v));
}

__device__ __forceinline__ __nv_bfloat16 load_bf16_smem(uint32_t addr) {
    uint16_t v;
    asm volatile("ld.shared.b16 %0, [%1];" : "=h"(v) : "r"(addr));
    return *reinterpret_cast<__nv_bfloat16*>(&v);
}

__device__ __forceinline__ uint32_t pack_bf16(float f0, float f1) {
    __nv_bfloat162 v = __floats2bfloat162_rn(f0, f1);
    return *reinterpret_cast<uint32_t*>(&v);
}

// ---- D kernel ----

__global__ void compute_D_kernel(const __nv_bfloat16* dO, const __nv_bfloat16* O, float* D_out, int total_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= total_rows) return;
    const __nv_bfloat162* dO_r = reinterpret_cast<const __nv_bfloat162*>(dO + (size_t)row * D);
    const __nv_bfloat162* O_r = reinterpret_cast<const __nv_bfloat162*>(O + (size_t)row * D);
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < D / 2; i++) {
        float2 d = __bfloat1622float2(dO_r[i]);
        float2 o = __bfloat1622float2(O_r[i]);
        sum += d.x * o.x + d.y * o.y;
    }
    D_out[row] = sum;
}

// ---- dK_dV kernel ----

__global__ void dK_dV_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* __restrict__ L,
    const float* __restrict__ D_vals,
    __nv_bfloat16* __restrict__ dK_out,
    __nv_bfloat16* __restrict__ dV_out,
    int B, int H, int S)
{
    int bh = blockIdx.x;
    int kv_tile = blockIdx.y;
    int bn_start = kv_tile * BN;
    if (bn_start >= S) return;
    int b = bh / H, h = bh % H;
    int64_t bh_off = (int64_t)(b * H + h) * S;

    const float* L_bh = L + bh_off;
    const float* D_bh = D_vals + bh_off;
    __nv_bfloat16* dK_bh = dK_out + bh_off * D;
    __nv_bfloat16* dV_bh = dV_out + bh_off * D;

    extern __shared__ char smem_raw[];
    // Align to 1024 for 128B swizzle
    uintptr_t aligned = ((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023;
    char* base = reinterpret_cast<char*>(aligned);
    __nv_bfloat16* K_smem = reinterpret_cast<__nv_bfloat16*>(base);
    __nv_bfloat16* V_smem = K_smem + 128 * 128;
    __nv_bfloat16* Q_smem = V_smem + 128 * 128;
    __nv_bfloat16* dO_smem = Q_smem + 128 * 128;
    __nv_bfloat16* P_smem = dO_smem + 128 * 128;
    __nv_bfloat16* dS_smem = P_smem + 128 * 128;
    float* L_smem = reinterpret_cast<float*>(dS_smem + 128 * 128);
    float* D_smem = L_smem + 128;

    __shared__ __align__(8) uint64_t bar_tma;
    __shared__ __align__(8) uint64_t bar_mma;
    __shared__ uint32_t tmem_addr_smem;

    int tid = threadIdx.x;
    int warp_id = tid / 32;

    if (tid == 0) {
        mbarrier_init(&bar_tma, 1);
        mbarrier_init(&bar_mma, 1);
        fence_mbarrier_init();
    }
    __syncthreads();

    if (warp_id == 0) {
        tmem_alloc(&tmem_addr_smem, 512);
    }
    __syncthreads();
    uint32_t tmem_base = tmem_addr_smem;

    constexpr uint32_t TMEM_S_OFF = 0;
    constexpr uint32_t TMEM_dP_OFF = 128;
    constexpr uint32_t TMEM_dV_OFF = 256;
    constexpr uint32_t TMEM_dK_OFF = 384;

    uint64_t k_km = make_sdesc_kmajor(K_smem);
    uint64_t v_km = make_sdesc_kmajor(V_smem);
    uint64_t q_km = make_sdesc_kmajor(Q_smem);
    uint64_t q_mm = make_sdesc_mmajor(Q_smem);
    uint64_t do_km = make_sdesc_kmajor(dO_smem);
    uint64_t do_mm = make_sdesc_mmajor(dO_smem);
    uint64_t p_km = make_sdesc_kmajor(P_smem);
    uint64_t ds_km = make_sdesc_kmajor(dS_smem);

    uint32_t idesc_00 = make_idesc(0, 0);
    uint32_t idesc_01 = make_idesc(0, 1);

    const float scale = 0.08838834764831845f;
    uint32_t mma_phase = 0;
    uint32_t tma_phase = 0;

    int kv_coord = (int)(bh_off + bn_start);
    uint32_t k_smem_base = (uint32_t)__cvta_generic_to_shared(K_smem);
    uint32_t v_smem_base = (uint32_t)__cvta_generic_to_shared(V_smem);
    uint32_t q_smem_base = (uint32_t)__cvta_generic_to_shared(Q_smem);
    uint32_t do_smem_base = (uint32_t)__cvta_generic_to_shared(dO_smem);
    uint32_t p_smem_base = (uint32_t)__cvta_generic_to_shared(P_smem);
    uint32_t ds_smem_base = (uint32_t)__cvta_generic_to_shared(dS_smem);

    // Load K
    if (tid == 0) {
        mbarrier_arrive_etx(&bar_tma, 32768);
        tma_load(&tma_K, &bar_tma, K_smem, 0, kv_coord);
        tma_load(&tma_K, &bar_tma, (char*)K_smem + 16384, 64, kv_coord);
    }
    mbarrier_wait(&bar_tma, tma_phase); tma_phase ^= 1;

    // Load V
    if (tid == 0) {
        mbarrier_arrive_etx(&bar_tma, 32768);
        tma_load(&tma_V, &bar_tma, V_smem, 0, kv_coord);
        tma_load(&tma_V, &bar_tma, (char*)V_smem + 16384, 64, kv_coord);
    }
    mbarrier_wait(&bar_tma, tma_phase); tma_phase ^= 1;

    int num_q_tiles = (S + BM - 1) / BM;
    int atom = tid / 64;
    int k_chunk = (tid % 64) / 8;
    int k_offset = tid % 8;

    for (int qt = 0; qt < num_q_tiles; qt++) {
        int bm_start = qt * BM;
        int q_coord = (int)(bh_off + bm_start);

        // Load Q, dO
        if (tid == 0) {
            mbarrier_arrive_etx(&bar_tma, 65536);
            tma_load(&tma_Q, &bar_tma, Q_smem, 0, q_coord);
            tma_load(&tma_Q, &bar_tma, (char*)Q_smem + 16384, 64, q_coord);
            tma_load(&tma_dO, &bar_tma, dO_smem, 0, q_coord);
            tma_load(&tma_dO, &bar_tma, (char*)dO_smem + 16384, 64, q_coord);
        }
        mbarrier_wait(&bar_tma, tma_phase); tma_phase ^= 1;

        // Load L, D
        if (tid < 128) {
            int gr = bm_start + tid;
            L_smem[tid] = (gr < S) ? L_bh[gr] : 0.0f;
            D_smem[tid] = (gr < S) ? D_bh[gr] : 0.0f;
        }
        __syncthreads();

        float l_reg = L_smem[tid];
        float d_reg = D_smem[tid];
        int bm = tid;
        int gr = bm_start + bm;
        bool row_valid = (gr < S);

        // MMA 1: S = Q @ K^T
        if (tid == 0) {
            for (int k = 0; k < 8; k++) {
                umma(tmem_base + TMEM_S_OFF,
                    desc_off(q_km, k_kmajor_off(k)),
                    desc_off(k_km, k_kmajor_off(k)),
                    idesc_00, (k == 0) ? 0 : 1);
            }
            umma_commit(&bar_mma);
        }
        __syncthreads();
        mbarrier_wait(&bar_mma, mma_phase); mma_phase ^= 1;
        __syncthreads();

        // P = exp(S * scale - L), write to P_smem (swizzled K-major)
        for (int chunk = 0; chunk < 16; chunk++) {
            uint32_t s0,s1,s2,s3,s4,s5,s6,s7;
            tmem_ld8(tmem_base + TMEM_S_OFF + chunk * 8,
                &s0,&s1,&s2,&s3,&s4,&s5,&s6,&s7);
            tmem_wait_ld();

            float sv[8] = {
                __uint_as_float(s0), __uint_as_float(s1),
                __uint_as_float(s2), __uint_as_float(s3),
                __uint_as_float(s4), __uint_as_float(s5),
                __uint_as_float(s6), __uint_as_float(s7)
            };

            for (int j = 0; j < 8; j++) {
                int bn = chunk * 8 + j;
                int gc = bn_start + bn;
                float p = (row_valid && gc < S) ? __expf(sv[j] * scale - l_reg) : 0.0f;
                uint32_t addr = swiz_addr(p_smem_base, bm, bn);
                store_bf16_smem(addr, __float2bfloat16(p));
            }
        }
        __syncthreads();
        fence_async_shared();

        // MMA 2: dP = dO @ V^T
        if (tid == 0) {
            tcgen05_fence_after();
            for (int k = 0; k < 8; k++) {
                umma(tmem_base + TMEM_dP_OFF,
                    desc_off(do_km, k_kmajor_off(k)),
                    desc_off(v_km, k_kmajor_off(k)),
                    idesc_00, (k == 0) ? 0 : 1);
            }
            umma_commit(&bar_mma);
        }
        __syncthreads();
        mbarrier_wait(&bar_mma, mma_phase); mma_phase ^= 1;
        __syncthreads();

        // MMA 3: dV += P^T @ dO (ta=0, tb=1)
        if (tid == 0) {
            tcgen05_fence_after();
            for (int k = 0; k < 8; k++) {
                umma(tmem_base + TMEM_dV_OFF,
                    desc_off(p_km, k_kmajor_off(k)),
                    desc_off(do_mm, k_mmajor_off(k)),
                    idesc_01, (qt == 0 && k == 0) ? 0 : 1);
            }
            umma_commit(&bar_mma);
        }
        __syncthreads();
        mbarrier_wait(&bar_mma, mma_phase); mma_phase ^= 1;
        __syncthreads();

        // dS = P * (dP - D) * scale, write to dS_smem
        for (int chunk = 0; chunk < 16; chunk++) {
            uint32_t dp0,dp1,dp2,dp3,dp4,dp5,dp6,dp7;
            tmem_ld8(tmem_base + TMEM_dP_OFF + chunk * 8,
                &dp0,&dp1,&dp2,&dp3,&dp4,&dp5,&dp6,&dp7);

            __nv_bfloat16 pv[8];
            for (int j = 0; j < 8; j++) {
                int bn = chunk * 8 + j;
                uint32_t addr = swiz_addr(p_smem_base, bm, bn);
                pv[j] = load_bf16_smem(addr);
            }
            tmem_wait_ld();

            float dpv[8] = {
                __uint_as_float(dp0), __uint_as_float(dp1),
                __uint_as_float(dp2), __uint_as_float(dp3),
                __uint_as_float(dp4), __uint_as_float(dp5),
                __uint_as_float(dp6), __uint_as_float(dp7)
            };

            for (int j = 0; j < 8; j++) {
                int bn = chunk * 8 + j;
                int gc = bn_start + bn;
                float p = __bfloat162float(pv[j]);
                float ds = (row_valid && gc < S) ? p * (dpv[j] - d_reg) * scale : 0.0f;
                uint32_t addr = swiz_addr(ds_smem_base, bm, bn);
                store_bf16_smem(addr, __float2bfloat16(ds));
            }
        }
        __syncthreads();
        fence_async_shared();

        // MMA 4: dK += dS^T @ Q (ta=0, tb=1)
        if (tid == 0) {
            tcgen05_fence_after();
            for (int k = 0; k < 8; k++) {
                umma(tmem_base + TMEM_dK_OFF,
                    desc_off(ds_km, k_kmajor_off(k)),
                    desc_off(q_mm, k_mmajor_off(k)),
                    idesc_01, (qt == 0 && k == 0) ? 0 : 1);
            }
            umma_commit(&bar_mma);
        }
        __syncthreads();
        mbarrier_wait(&bar_mma, mma_phase); mma_phase ^= 1;
        __syncthreads();
    }

    // Epilogue: store dK, dV
    int row = bn_start + tid;
    bool row_valid = (row < S);

    for (int is_dK = 0; is_dK <= 1; is_dK++) {
        uint32_t tmem_off = is_dK ? (tmem_base + TMEM_dK_OFF) : (tmem_base + TMEM_dV_OFF);
        __nv_bfloat16* out_ptr = is_dK ? dK_bh : dV_bh;

        for (int chunk = 0; chunk < 16; chunk++) {
            uint32_t r0,r1,r2,r3,r4,r5,r6,r7;
            tmem_ld8(tmem_off + chunk * 8, &r0,&r1,&r2,&r3,&r4,&r5,&r6,&r7);
            tmem_wait_ld();

            if (row_valid) {
                uint32_t packed[4];
                packed[0] = pack_bf16(__uint_as_float(r0), __uint_as_float(r1));
                packed[1] = pack_bf16(__uint_as_float(r2), __uint_as_float(r3));
                packed[2] = pack_bf16(__uint_as_float(r4), __uint_as_float(r5));
                packed[3] = pack_bf16(__uint_as_float(r6), __uint_as_float(r7));
                uint4 val = make_uint4(packed[0], packed[1], packed[2], packed[3]);
                *reinterpret_cast<uint4*>(out_ptr + (size_t)row * D + chunk * 8) = val;
            }
        }
        __syncthreads();
    }

    __syncthreads();
    if (warp_id == 0) {
        tmem_dealloc(tmem_base, 512);
    }
}

// ---- dQ kernel ----

__global__ void dQ_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* __restrict__ L,
    const float* __restrict__ D_vals,
    __nv_bfloat16* __restrict__ dQ_out,
    int B, int H, int S)
{
    int bh = blockIdx.x;
    int q_tile = blockIdx.y;
    int bm_start = q_tile * BM;
    if (bm_start >= S) return;
    int b = bh / H, h = bh % H;
    int64_t bh_off = (int64_t)(b * H + h) * S;

    const float* L_bh = L + bh_off;
    const float* D_bh = D_vals + bh_off;
    __nv_bfloat16* dQ_bh = dQ_out + bh_off * D;

    extern __shared__ char smem_raw[];
    uintptr_t aligned = ((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023;
    char* base = reinterpret_cast<char*>(aligned);
    __nv_bfloat16* Q_smem = reinterpret_cast<__nv_bfloat16*>(base);
    __nv_bfloat16* dO_smem = Q_smem + 128 * 128;
    __nv_bfloat16* K_smem = dO_smem + 128 * 128;
    __nv_bfloat16* V_smem = K_smem + 128 * 128;
    __nv_bfloat16* P_smem = V_smem + 128 * 128;
    __nv_bfloat16* dS_smem = P_smem + 128 * 128;
    float* L_smem = reinterpret_cast<float*>(dS_smem + 128 * 128);
    float* D_smem = L_smem + 128;

    __shared__ __align__(8) uint64_t bar_tma;
    __shared__ __align__(8) uint64_t bar_mma;
    __shared__ uint32_t tmem_addr_smem;

    int tid = threadIdx.x;
    int warp_id = tid / 32;

    if (tid == 0) {
        mbarrier_init(&bar_tma, 1);
        mbarrier_init(&bar_mma, 1);
        fence_mbarrier_init();
    }
    __syncthreads();

    if (warp_id == 0) {
        tmem_alloc(&tmem_addr_smem, 256);
    }
    __syncthreads();
    uint32_t tmem_base = tmem_addr_smem;

    constexpr uint32_t TMEM_S_OFF = 0;
    constexpr uint32_t TMEM_dP_OFF = 128;

    uint64_t k_km = make_sdesc_kmajor(K_smem);
    uint64_t k_mm = make_sdesc_mmajor(K_smem);
    uint64_t v_km = make_sdesc_kmajor(V_smem);
    uint64_t q_km = make_sdesc_kmajor(Q_smem);
    uint64_t do_km = make_sdesc_kmajor(dO_smem);
    uint64_t p_km = make_sdesc_kmajor(P_smem);
    uint64_t ds_km = make_sdesc_kmajor(dS_smem);
    uint64_t ds_mm = make_sdesc_mmajor(dS_smem);

    uint32_t idesc_00 = make_idesc(0, 0);
    uint32_t idesc_11 = make_idesc(1, 1);

    const float scale = 0.08838834764831845f;
    uint32_t mma_phase = 0;
    uint32_t tma_phase = 0;

    int q_coord = (int)(bh_off + bm_start);

    uint32_t p_smem_base = (uint32_t)__cvta_generic_to_shared(P_smem);
    uint32_t ds_smem_base = (uint32_t)__cvta_generic_to_shared(dS_smem);

    // Load Q, dO once
    if (tid == 0) {
        mbarrier_arrive_etx(&bar_tma, 65536);
        tma_load(&tma_Q, &bar_tma, Q_smem, 0, q_coord);
        tma_load(&tma_Q, &bar_tma, (char*)Q_smem + 16384, 64, q_coord);
        tma_load(&tma_dO, &bar_tma, dO_smem, 0, q_coord);
        tma_load(&tma_dO, &bar_tma, (char*)dO_smem + 16384, 64, q_coord);
    }
    mbarrier_wait(&bar_tma, tma_phase); tma_phase ^= 1;

    // Load L, D
    if (tid < 128) {
        int gr = bm_start + tid;
        L_smem[tid] = (gr < S) ? L_bh[gr] : 0.0f;
        D_smem[tid] = (gr < S) ? D_bh[gr] : 0.0f;
    }
    __syncthreads();

    float l_reg = L_smem[tid];
    float d_reg = D_smem[tid];
    int bm = tid;
    int gr = bm_start + bm;
    bool row_valid = (gr < S);
    int atom = tid / 64;
    int k_chunk = (tid % 64) / 8;
    int k_offset = tid % 8;

    int num_kv_tiles = (S + BN - 1) / BN;

    for (int kt = 0; kt < num_kv_tiles; kt++) {
        int bn_start = kt * BN;
        int kv_coord = (int)(bh_off + bn_start);

        // Load K, V
        if (tid == 0) {
            mbarrier_arrive_etx(&bar_tma, 65536);
            tma_load(&tma_K, &bar_tma, K_smem, 0, kv_coord);
            tma_load(&tma_K, &bar_tma, (char*)K_smem + 16384, 64, kv_coord);
            tma_load(&tma_V, &bar_tma, V_smem, 0, kv_coord);
            tma_load(&tma_V, &bar_tma, (char*)V_smem + 16384, 64, kv_coord);
        }
        mbarrier_wait(&bar_tma, tma_phase); tma_phase ^= 1;

        // MMA 1: S = Q @ K^T
        if (tid == 0) {
            for (int k = 0; k < 8; k++) {
                umma(tmem_base + TMEM_S_OFF,
                    desc_off(q_km, k_kmajor_off(k)),
                    desc_off(k_km, k_kmajor_off(k)),
                    idesc_00, (k == 0) ? 0 : 1);
            }
            umma_commit(&bar_mma);
        }
        __syncthreads();
        mbarrier_wait(&bar_mma, mma_phase); mma_phase ^= 1;
        __syncthreads();

        // P = exp(S * scale - L)
        for (int chunk = 0; chunk < 16; chunk++) {
            uint32_t s0,s1,s2,s3,s4,s5,s6,s7;
            tmem_ld8(tmem_base + TMEM_S_OFF + chunk * 8,
                &s0,&s1,&s2,&s3,&s4,&s5,&s6,&s7);
            tmem_wait_ld();

            float sv[8] = {
                __uint_as_float(s0), __uint_as_float(s1),
                __uint_as_float(s2), __uint_as_float(s3),
                __uint_as_float(s4), __uint_as_float(s5),
                __uint_as_float(s6), __uint_as_float(s7)
            };

            for (int j = 0; j < 8; j++) {
                int bn = chunk * 8 + j;
                int gc = bn_start + bn;
                float p = (row_valid && gc < S) ? __expf(sv[j] * scale - l_reg) : 0.0f;
                uint32_t addr = swiz_addr(p_smem_base, bm, bn);
                store_bf16_smem(addr, __float2bfloat16(p));
            }
        }
        __syncthreads();
        fence_async_shared();

        // MMA 2: dP = dO @ V^T
        if (tid == 0) {
            tcgen05_fence_after();
            for (int k = 0; k < 8; k++) {
                umma(tmem_base + TMEM_dP_OFF,
                    desc_off(do_km, k_kmajor_off(k)),
                    desc_off(v_km, k_kmajor_off(k)),
                    idesc_00, (k == 0) ? 0 : 1);
            }
            umma_commit(&bar_mma);
        }
        __syncthreads();
        mbarrier_wait(&bar_mma, mma_phase); mma_phase ^= 1;
        __syncthreads();

        // dS = P * (dP - D) * scale
        for (int chunk = 0; chunk < 16; chunk++) {
            uint32_t dp0,dp1,dp2,dp3,dp4,dp5,dp6,dp7;
            tmem_ld8(tmem_base + TMEM_dP_OFF + chunk * 8,
                &dp0,&dp1,&dp2,&dp3,&dp4,&dp5,&dp6,&dp7);

            __nv_bfloat16 pv[8];
            for (int j = 0; j < 8; j++) {
                int bn = chunk * 8 + j;
                uint32_t addr = swiz_addr(p_smem_base, bm, bn);
                pv[j] = load_bf16_smem(addr);
            }
            tmem_wait_ld();

            float dpv[8] = {
                __uint_as_float(dp0), __uint_as_float(dp1),
                __uint_as_float(dp2), __uint_as_float(dp3),
                __uint_as_float(dp4), __uint_as_float(dp5),
                __uint_as_float(dp6), __uint_as_float(dp7)
            };

            for (int j = 0; j < 8; j++) {
                int bn = chunk * 8 + j;
                int gc = bn_start + bn;
                float p = __bfloat162float(pv[j]);
                float ds = (row_valid && gc < S) ? p * (dpv[j] - d_reg) * scale : 0.0f;
                uint32_t addr = swiz_addr(ds_smem_base, bm, bn);
                store_bf16_smem(addr, __float2bfloat16(ds));
            }
        }
        __syncthreads();
        fence_async_shared();

        // MMA 3: dQ += dS @ K (ta=1, tb=1)
        if (tid == 0) {
            tcgen05_fence_after();
            for (int k = 0; k < 8; k++) {
                umma(tmem_base + TMEM_S_OFF,
                    desc_off(ds_mm, k_mmajor_off(k)),
                    desc_off(k_mm, k_mmajor_off(k)),
                    idesc_11, (kt == 0 && k == 0) ? 0 : 1);
            }
            umma_commit(&bar_mma);
        }
        __syncthreads();
        mbarrier_wait(&bar_mma, mma_phase); mma_phase ^= 1;
        __syncthreads();
    }

    // Epilogue: store dQ
    for (int chunk = 0; chunk < 16; chunk++) {
        uint32_t r0,r1,r2,r3,r4,r5,r6,r7;
        tmem_ld8(tmem_base + TMEM_S_OFF + chunk * 8,
            &r0,&r1,&r2,&r3,&r4,&r5,&r6,&r7);
        tmem_wait_ld();

        if (row_valid) {
            uint32_t packed[4];
            packed[0] = pack_bf16(__uint_as_float(r0), __uint_as_float(r1));
            packed[1] = pack_bf16(__uint_as_float(r2), __uint_as_float(r3));
            packed[2] = pack_bf16(__uint_as_float(r4), __uint_as_float(r5));
            packed[3] = pack_bf16(__uint_as_float(r6), __uint_as_float(r7));
            uint4 val = make_uint4(packed[0], packed[1], packed[2], packed[3]);
            *reinterpret_cast<uint4*>(dQ_bh + (size_t)gr * D + chunk * 8) = val;
        }
    }

    __syncthreads();
    if (warp_id == 0) {
        tmem_dealloc(tmem_base, 256);
    }
}

// ---- TMA descriptor creation ----

CUresult create_tma_desc(CUtensorMap* d, void* ptr, uint64_t inner, uint64_t outer) {
    cuuint64_t globalDim[2] = {inner, outer};
    cuuint64_t globalStrides[1] = {inner * 2};
    cuuint32_t boxDim[2] = {64, 128};
    cuuint32_t elemStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        ptr, globalDim, globalStrides, boxDim, elemStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
}

// ---- Host run function ----

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {

    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    int B = (int)Q.size(0), H = (int)Q.size(1), S = (int)Q.size(2);
    int total_rows = B * H * S;

    float* D_temp;
    CUDA_CHECK(cudaMalloc(&D_temp, (size_t)total_rows * sizeof(float)));

    {
        int threads = 256, blocks = (total_rows + threads - 1) / threads;
        compute_D_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(dO.data_ptr()),
            static_cast<const __nv_bfloat16*>(O.data_ptr()),
            D_temp, total_rows);
    }

    CUtensorMap tma_Q, tma_K, tma_V, tma_dO;
    int64_t total_outer = (int64_t)B * H * S;
    CU_CHECK(create_tma_desc(&tma_Q, Q.data_ptr(), D, total_outer));
    CU_CHECK(create_tma_desc(&tma_K, K.data_ptr(), D, total_outer));
    CU_CHECK(create_tma_desc(&tma_V, V.data_ptr(), D, total_outer));
    CU_CHECK(create_tma_desc(&tma_dO, dO.data_ptr(), D, total_outer));

    int smem_size = 6 * 128 * 128 * 2 + 1024 + 2048;

    {
        dim3 grid(B * H, (S + BN - 1) / BN, 1);
        dim3 block(THREADS, 1, 1);
        CUDA_CHECK(cudaFuncSetAttribute(dK_dV_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        dK_dV_kernel<<<grid, block, smem_size, stream>>>(
            tma_Q, tma_K, tma_V, tma_dO,
            static_cast<const float*>(L.data_ptr()), D_temp,
            static_cast<__nv_bfloat16*>(dK.data_ptr()),
            static_cast<__nv_bfloat16*>(dV.data_ptr()), B, H, S);
    }

    {
        dim3 grid(B * H, (S + BM - 1) / BM, 1);
        dim3 block(THREADS, 1, 1);
        CUDA_CHECK(cudaFuncSetAttribute(dQ_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        dQ_kernel<<<grid, block, smem_size, stream>>>(
            tma_Q, tma_K, tma_V, tma_dO,
            static_cast<const float*>(L.data_ptr()), D_temp,
            static_cast<__nv_bfloat16*>(dQ.data_ptr()), B, H, S);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(D_temp));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flash_attn_bwd::run);

}  // namespace flash_attn_bwd