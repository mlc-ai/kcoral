#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
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

#define UMMA_F16(TMEM, DA, DB, IDESC, ACCUM) \
    asm volatile( \
        "{\n.reg .pred p;\n" \
        "setp.ne.b32 p, %4, 0;\n" \
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n" \
        :: "r"(TMEM), "l"(DA), "l"(DB), "r"(IDESC), "r"(ACCUM))

#define UMMA_COMMIT(MBAR) \
    do { \
        uint64_t _mbar_addr = (uint64_t)(MBAR); \
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "l"(_mbar_addr)); \
    } while(0)

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmem_load_8x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3, uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),"=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(col));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n.reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n" :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(phase));
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y; asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y;
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)), "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_store_2d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];"
        :: "l"((uint64_t)d), "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
}

// SMEM descriptors always K-major because our tiles load natively as row-major block arrays
__device__ __forceinline__ uint64_t make_smem_desc(void* smem_ptr) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint64_t d = 0;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)(1) << 16;
    d |= (uint64_t)(1024 >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ void advance_desc_k(uint64_t& desc, uint32_t K_step_bytes) {
    uint32_t addr = (desc & 0x3FFF);
    addr += (K_step_bytes >> 4);
    desc = (desc & ~0x3FFF) | (addr & 0x3FFF);
}

__device__ __forceinline__ uint32_t make_instr_desc(uint32_t M, uint32_t N, bool a_major, bool b_major) {
    uint32_t d = 0;
    d |= (1u << 4) | (1u << 7) | (1u << 10);
    d |= (a_major ? 1u : 0u) << 15;
    d |= (b_major ? 1u : 0u) << 16;
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ void store_tmem_to_smem_swizzled(uint32_t tmem_base, __nv_bfloat16* smem) {
    for (int col = 0; col < 64; col += 8) {
        uint32_t r[8];
        tmem_load_8x_fn(tmem_base + col, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
        tmem_load_fence_fn();
        if (threadIdx.x < 64) {
            int row = threadIdx.x;
            int swiz_col_blk = (col / 8) ^ (row % 8);
            for (int c = 0; c < 8; c++) {
                smem[row * 64 + swiz_col_blk * 8 + c] = __float2bfloat16(__uint_as_float(r[c]));
            }
        }
    }
}

__global__ void compute_delta_kernel(const __nv_bfloat16* O, const __nv_bfloat16* dO, float* Delta, int S) {
    int b = blockIdx.z, h = blockIdx.y, s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < S) {
        float delta = 0.0f;
        int offset = (b * gridDim.y * S + h * S + s) * 128;
        for (int d = 0; d < 128; d += 8) {
            float4 o_vec = *(const float4*)&O[offset + d];
            float4 do_vec = *(const float4*)&dO[offset + d];
            __nv_bfloat162* o2 = (__nv_bfloat162*)&o_vec;
            __nv_bfloat162* do2 = (__nv_bfloat162*)&do_vec;
            for (int i = 0; i < 4; ++i) {
                delta += __bfloat162float(o2[i].x) * __bfloat162float(do2[i].x);
                delta += __bfloat162float(o2[i].y) * __bfloat162float(do2[i].y);
            }
        }
        Delta[b * gridDim.y * S + h * S + s] = delta;
    }
}

__global__ void kernel_dq(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_dQ,
    const float* L, const float* Delta, int S_total) {
    
    int b = blockIdx.z, h = blockIdx.y, i = blockIdx.x, S_idx = i * 64;
    if (S_idx >= S_total) return;
    int batch_row_offset = (b * gridDim.y + h) * S_total + S_idx;

    extern __shared__ char smem_raw[];
    uintptr_t smem_addr = (uintptr_t)smem_raw;
    smem_addr = (smem_addr + 127) & ~127ULL;
    __nv_bfloat16* smem = (__nv_bfloat16*)smem_addr;

    __nv_bfloat16 *smem_Q0 = smem, *smem_Q1 = smem_Q0 + 4096;
    __nv_bfloat16 *smem_dO0 = smem_Q1 + 4096, *smem_dO1 = smem_dO0 + 4096;
    __nv_bfloat16 *smem_K0 = smem_dO1 + 4096, *smem_K1 = smem_K0 + 4096;
    __nv_bfloat16 *smem_V0 = smem_K1 + 4096, *smem_V1 = smem_V0 + 4096;
    __nv_bfloat16 *smem_dS = smem_V1 + 4096;
    float *smem_L = (float*)(smem_dS + 4096), *smem_Delta = smem_L + 64;
    uint64_t* mbar_tma = (uint64_t*)(smem_Delta + 64);
    uint64_t* mbar_umma = mbar_tma + 1;
    uint32_t* smem_tmem_base = (uint32_t*)(mbar_umma + 1);
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_tma, 1);
        init_smem_barrier_fn(mbar_umma, 1);
        mbarrier_arrive_and_expect_tx_fn(mbar_tma, 8192 * 4);
        tma_load_2d_fn(&tma_Q, mbar_tma, smem_Q0, 0, batch_row_offset);
        tma_load_2d_fn(&tma_Q, mbar_tma, smem_Q1, 64, batch_row_offset);
        tma_load_2d_fn(&tma_dO, mbar_tma, smem_dO0, 0, batch_row_offset);
        tma_load_2d_fn(&tma_dO, mbar_tma, smem_dO1, 64, batch_row_offset);
    }
    
    if (threadIdx.x < 32) tmem_alloc_fn(smem_tmem_base, 256);
    if (threadIdx.x < 64) {
        int S_off = (b * gridDim.y + h) * S_total;
        smem_L[threadIdx.x] = (S_idx + threadIdx.x < S_total) ? L[S_off + S_idx + threadIdx.x] : 0.0f;
        smem_Delta[threadIdx.x] = (S_idx + threadIdx.x < S_total) ? Delta[S_off + S_idx + threadIdx.x] : 0.0f;
    }
    __syncthreads();
    
    uint32_t tmem_base = *smem_tmem_base;
    uint32_t tmem_S = tmem_base, tmem_dP = tmem_base + 64, tmem_dQ0 = tmem_base + 128, tmem_dQ1 = tmem_base + 192;

    if (threadIdx.x == 0) mbarrier_wait_fn(mbar_tma, 0);
    __syncthreads();
    
    float scale = 1.0f / sqrtf(128.0f);
    uint32_t phase_tma = 1, phase_umma = 0;

    for (int j = 0; j <= i; j++) {
        int K_idx = j * 64;
        int k_batch_row = (b * gridDim.y + h) * S_total + K_idx;
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_tma, 8192 * 4);
            tma_load_2d_fn(&tma_K, mbar_tma, smem_K0, 0, k_batch_row);
            tma_load_2d_fn(&tma_K, mbar_tma, smem_K1, 64, k_batch_row);
            tma_load_2d_fn(&tma_V, mbar_tma, smem_V0, 0, k_batch_row);
            tma_load_2d_fn(&tma_V, mbar_tma, smem_V1, 64, k_batch_row);
        }
        __syncthreads();
        if (threadIdx.x == 0) mbarrier_wait_fn(mbar_tma, phase_tma);
        __syncthreads();
        phase_tma ^= 1;

        uint32_t idesc_S = make_instr_desc(64, 64, false, true); 
        __nv_bfloat16 *Q_ptrs[2] = {smem_Q0, smem_Q1}, *K_ptrs[2] = {smem_K0, smem_K1};
        __nv_bfloat16 *dO_ptrs[2] = {smem_dO0, smem_dO1}, *V_ptrs[2] = {smem_V0, smem_V1};
        
        for (int p = 0; p < 2; p++) {
            uint64_t dAQ = make_smem_desc(Q_ptrs[p]), dBK = make_smem_desc(K_ptrs[p]);
            uint64_t dAO = make_smem_desc(dO_ptrs[p]), dBV = make_smem_desc(V_ptrs[p]);
            for (int k = 0; k < 4; k++) {
                int accum = (p > 0 || k > 0) ? 1 : 0;
                if (threadIdx.x == 0) {
                    UMMA_F16(tmem_S, dAQ, dBK, idesc_S, accum);
                    UMMA_F16(tmem_dP, dAO, dBV, idesc_S, accum);
                }
                advance_desc_k(dAQ, 32); advance_desc_k(dBK, 32); advance_desc_k(dAO, 32); advance_desc_k(dBV, 32);
            }
        }
        
        if (threadIdx.x == 0) UMMA_COMMIT(mbar_umma);
        if (threadIdx.x == 0) mbarrier_wait_fn(mbar_umma, phase_umma);
        __syncthreads();
        phase_umma ^= 1;

        for (int col = 0; col < 64; col += 8) {
            uint32_t rS[8], rDP[8];
            tmem_load_8x_fn(tmem_S + col, &rS[0], &rS[1], &rS[2], &rS[3], &rS[4], &rS[5], &rS[6], &rS[7]);
            tmem_load_8x_fn(tmem_dP + col, &rDP[0], &rDP[1], &rDP[2], &rDP[3], &rDP[4], &rDP[5], &rDP[6], &rDP[7]);
            tmem_load_fence_fn();
            if (threadIdx.x < 64) {
                int row = threadIdx.x;
                int swiz_col_blk = (col / 8) ^ (row % 8);
                float lval = smem_L[row], delt = smem_Delta[row];
                for (int c = 0; c < 8; c++) {
                    float s = __uint_as_float(rS[c]), dp = __uint_as_float(rDP[c]), p = 0.0f, ds = 0.0f;
                    int k_pos = K_idx + col + c;
                    int q_pos = S_idx + row;
                    if (k_pos <= q_pos && q_pos < S_total && k_pos < S_total) {
                        p = fast_exp2f_fn((s * scale - lval) * 1.44269504f);
                        ds = p * (dp - delt) * scale;
                    }
                    smem_dS[row * 64 + swiz_col_blk * 8 + c] = __float2bfloat16(ds);
                }
            }
        }
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

        uint32_t idesc_dQ = make_instr_desc(64, 64, false, false); 
        uint32_t tmem_dQs[2] = {tmem_dQ0, tmem_dQ1};
        for (int p = 0; p < 2; p++) {
            uint64_t dA = make_smem_desc(smem_dS), dB = make_smem_desc(K_ptrs[p]);
            for (int k = 0; k < 4; k++) {
                int accum = (j > 0 || k > 0) ? 1 : 0;
                if (threadIdx.x == 0) UMMA_F16(tmem_dQs[p], dA, dB, idesc_dQ, accum);
                advance_desc_k(dA, 32); advance_desc_k(dB, 32);
            }
        }
        if (threadIdx.x == 0) UMMA_COMMIT(mbar_umma);
        if (threadIdx.x == 0) mbarrier_wait_fn(mbar_umma, phase_umma);
        __syncthreads();
        phase_umma ^= 1;
    }
    
    store_tmem_to_smem_swizzled(tmem_dQ0, smem_Q0);
    store_tmem_to_smem_swizzled(tmem_dQ1, smem_Q1);
    __syncthreads();
    
    if (threadIdx.x == 0) {
        tma_store_2d_fn(&tma_dQ, smem_Q0, 0, batch_row_offset);
        tma_store_2d_fn(&tma_dQ, smem_Q1, 64, batch_row_offset);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }
    if (threadIdx.x < 32) tmem_dealloc_fn(tmem_base, 256);
}

__global__ void kernel_dk_dv(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_dK,
    const __grid_constant__ CUtensorMap tma_dV,
    const float* L, const float* Delta, int S_total) {
    
    int b = blockIdx.z, h = blockIdx.y, j = blockIdx.x, K_idx = j * 64;
    if (K_idx >= S_total) return;
    int k_batch_row = (b * gridDim.y + h) * S_total + K_idx;

    extern __shared__ char smem_raw[];
    uintptr_t smem_addr = (uintptr_t)smem_raw;
    smem_addr = (smem_addr + 127) & ~127ULL;
    __nv_bfloat16* smem = (__nv_bfloat16*)smem_addr;

    __nv_bfloat16 *smem_Q0 = smem, *smem_Q1 = smem_Q0 + 4096;
    __nv_bfloat16 *smem_dO0 = smem_Q1 + 4096, *smem_dO1 = smem_dO0 + 4096;
    __nv_bfloat16 *smem_K0 = smem_dO1 + 4096, *smem_K1 = smem_K0 + 4096;
    __nv_bfloat16 *smem_V0 = smem_K1 + 4096, *smem_V1 = smem_V0 + 4096;
    __nv_bfloat16 *smem_dS = smem_V1 + 4096, *smem_PT = smem_dS + 4096;
    float *smem_L = (float*)(smem_PT + 4096), *smem_Delta = smem_L + 64;
    uint64_t* mbar_tma = (uint64_t*)(smem_Delta + 64);
    uint64_t* mbar_umma = mbar_tma + 1;
    uint32_t* smem_tmem_base = (uint32_t*)(mbar_umma + 1);
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_tma, 1);
        init_smem_barrier_fn(mbar_umma, 1);
        mbarrier_arrive_and_expect_tx_fn(mbar_tma, 8192 * 4);
        tma_load_2d_fn(&tma_K, mbar_tma, smem_K0, 0, k_batch_row);
        tma_load_2d_fn(&tma_K, mbar_tma, smem_K1, 64, k_batch_row);
        tma_load_2d_fn(&tma_V, mbar_tma, smem_V0, 0, k_batch_row);
        tma_load_2d_fn(&tma_V, mbar_tma, smem_V1, 64, k_batch_row);
    }
    if (threadIdx.x < 32) tmem_alloc_fn(smem_tmem_base, 512);
    __syncthreads();
    
    uint32_t tmem_base = *smem_tmem_base;
    uint32_t tmem_ST = tmem_base, tmem_dPT = tmem_base + 64, tmem_dK0 = tmem_base + 128, tmem_dK1 = tmem_base + 192, tmem_dV0 = tmem_base + 256, tmem_dV1 = tmem_base + 320;

    if (threadIdx.x == 0) mbarrier_wait_fn(mbar_tma, 0);
    __syncthreads();
    
    float scale = 1.0f / sqrtf(128.0f);
    uint32_t phase_tma = 1, phase_umma = 0;

    for (int i = j; i < (S_total + 63) / 64; i++) {
        int S_idx = i * 64;
        int batch_row_offset = (b * gridDim.y + h) * S_total + S_idx;
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_tma, 8192 * 4);
            tma_load_2d_fn(&tma_Q, mbar_tma, smem_Q0, 0, batch_row_offset);
            tma_load_2d_fn(&tma_Q, mbar_tma, smem_Q1, 64, batch_row_offset);
            tma_load_2d_fn(&tma_dO, mbar_tma, smem_dO0, 0, batch_row_offset);
            tma_load_2d_fn(&tma_dO, mbar_tma, smem_dO1, 64, batch_row_offset);
        }
        
        if (threadIdx.x < 64) {
            int S_off = (b * gridDim.y + h) * S_total;
            smem_L[threadIdx.x] = (S_idx + threadIdx.x < S_total) ? L[S_off + S_idx + threadIdx.x] : 0.0f;
            smem_Delta[threadIdx.x] = (S_idx + threadIdx.x < S_total) ? Delta[S_off + S_idx + threadIdx.x] : 0.0f;
        }
        __syncthreads();
        if (threadIdx.x == 0) mbarrier_wait_fn(mbar_tma, phase_tma);
        __syncthreads();
        phase_tma ^= 1;

        uint32_t idesc_ST = make_instr_desc(64, 64, false, true); 
        __nv_bfloat16 *Q_ptrs[2] = {smem_Q0, smem_Q1}, *K_ptrs[2] = {smem_K0, smem_K1};
        __nv_bfloat16 *dO_ptrs[2] = {smem_dO0, smem_dO1}, *V_ptrs[2] = {smem_V0, smem_V1};
        
        for (int p = 0; p < 2; p++) {
            uint64_t dAK = make_smem_desc(K_ptrs[p]), dBQ = make_smem_desc(Q_ptrs[p]);
            uint64_t dAV = make_smem_desc(V_ptrs[p]), dBO = make_smem_desc(dO_ptrs[p]);
            for (int k = 0; k < 4; k++) {
                int accum = (p > 0 || k > 0) ? 1 : 0;
                if (threadIdx.x == 0) {
                    UMMA_F16(tmem_ST, dAK, dBQ, idesc_ST, accum);
                    UMMA_F16(tmem_dPT, dAV, dBO, idesc_ST, accum);
                }
                advance_desc_k(dAK, 32); advance_desc_k(dBQ, 32); advance_desc_k(dAV, 32); advance_desc_k(dBO, 32);
            }
        }
        
        if (threadIdx.x == 0) UMMA_COMMIT(mbar_umma);
        if (threadIdx.x == 0) mbarrier_wait_fn(mbar_umma, phase_umma);
        __syncthreads();
        phase_umma ^= 1;

        for (int col = 0; col < 64; col += 8) {
            uint32_t rS[8], rDP[8];
            tmem_load_8x_fn(tmem_ST + col, &rS[0], &rS[1], &rS[2], &rS[3], &rS[4], &rS[5], &rS[6], &rS[7]);
            tmem_load_8x_fn(tmem_dPT + col, &rDP[0], &rDP[1], &rDP[2], &rDP[3], &rDP[4], &rDP[5], &rDP[6], &rDP[7]);
            tmem_load_fence_fn();
            if (threadIdx.x < 64) {
                int row = threadIdx.x; 
                int swiz_col_blk = (col / 8) ^ (row % 8);
                for (int c = 0; c < 8; c++) {
                    float s = __uint_as_float(rS[c]), dp = __uint_as_float(rDP[c]), p = 0.0f, ds = 0.0f;
                    int q_pos = S_idx + col + c;
                    int k_pos = K_idx + row;
                    if (k_pos <= q_pos && q_pos < S_total && k_pos < S_total) {
                        p = fast_exp2f_fn((s * scale - smem_L[col + c]) * 1.44269504f);
                        ds = p * (dp - smem_Delta[col + c]) * scale;
                    }
                    int dest = row * 64 + swiz_col_blk * 8 + c;
                    smem_dS[dest] = __float2bfloat16(ds);
                    smem_PT[dest] = __float2bfloat16(p);
                }
            }
        }
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

        uint32_t idesc_dK = make_instr_desc(64, 64, false, false); 
        for (int p = 0; p < 2; p++) {
            uint64_t dAV = make_smem_desc(smem_PT), dBO = make_smem_desc(dO_ptrs[p]);
            uint64_t dAK = make_smem_desc(smem_dS), dBQ = make_smem_desc(Q_ptrs[p]);
            for (int k = 0; k < 4; k++) {
                int accum = (i > j || k > 0) ? 1 : 0;
                if (threadIdx.x == 0) {
                    UMMA_F16(tmem_dV0 + p * 64, dAV, dBO, idesc_dK, accum);
                    UMMA_F16(tmem_dK0 + p * 64, dAK, dBQ, idesc_dK, accum);
                }
                advance_desc_k(dAV, 32); advance_desc_k(dBO, 32); advance_desc_k(dAK, 32); advance_desc_k(dBQ, 32);
            }
        }
        if (threadIdx.x == 0) UMMA_COMMIT(mbar_umma);
        if (threadIdx.x == 0) mbarrier_wait_fn(mbar_umma, phase_umma);
        __syncthreads();
        phase_umma ^= 1;
    }
    
    store_tmem_to_smem_swizzled(tmem_dK0, smem_K0);
    store_tmem_to_smem_swizzled(tmem_dK1, smem_K1);
    store_tmem_to_smem_swizzled(tmem_dV0, smem_V0);
    store_tmem_to_smem_swizzled(tmem_dV1, smem_V1);
    __syncthreads();
    
    if (threadIdx.x == 0) {
        tma_store_2d_fn(&tma_dK, smem_K0, 0, k_batch_row);
        tma_store_2d_fn(&tma_dK, smem_K1, 64, k_batch_row);
        tma_store_2d_fn(&tma_dV, smem_V0, 0, k_batch_row);
        tma_store_2d_fn(&tma_dV, smem_V1, 64, k_batch_row);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }
    if (threadIdx.x < 32) tmem_dealloc_fn(tmem_base, 512);
}

namespace tvm_ffi_example_cuda {

    void create_tma_2d(CUtensorMap* tma, void* ptr, uint64_t rows, uint32_t cols) {
        cuuint64_t globalDim[2] = {cols, rows};
        cuuint64_t globalStrides[1] = {cols * 2};
        cuuint32_t boxDim[2] = {64, 64};
        cuuint32_t elementStrides[2] = {1, 1};
        cuTensorMapEncodeTiled(tma, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ptr,
            globalDim, globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    }

    void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
             tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
             tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
                 
        CUDA_CHECK(cudaSetDevice(Q.device().device_id));
        cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
        
        int64_t B = Q.size(0), H = Q.size(1), S = Q.size(2);
        
        float* d_Delta;
        CUDA_CHECK(cudaMallocAsync(&d_Delta, B * H * S * sizeof(float), stream));
        
        dim3 grid_delta((S + 127) / 128, H, B);
        compute_delta_kernel<<<grid_delta, 128, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(O.data_ptr()),
            static_cast<const __nv_bfloat16*>(dO.data_ptr()),
            d_Delta, S);

        CUtensorMap tma_Q, tma_K, tma_V, tma_dO, tma_dQ, tma_dK, tma_dV;
        create_tma_2d(&tma_Q, Q.data_ptr(), B * H * S, 128);
        create_tma_2d(&tma_K, K.data_ptr(), B * H * S, 128);
        create_tma_2d(&tma_V, V.data_ptr(), B * H * S, 128);
        create_tma_2d(&tma_dO, dO.data_ptr(), B * H * S, 128);
        create_tma_2d(&tma_dQ, dQ.data_ptr(), B * H * S, 128);
        create_tma_2d(&tma_dK, dK.data_ptr(), B * H * S, 128);
        create_tma_2d(&tma_dV, dV.data_ptr(), B * H * S, 128);
        
        int smem_bytes = 94208;
        CUDA_CHECK(cudaFuncSetAttribute(kernel_dq, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
        CUDA_CHECK(cudaFuncSetAttribute(kernel_dk_dv, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
        
        dim3 grid_blocks((S + 63) / 64, H, B);
        
        kernel_dq<<<grid_blocks, 128, smem_bytes, stream>>>(
            tma_Q, tma_K, tma_V, tma_dO, tma_dQ,
            static_cast<const float*>(L.data_ptr()), d_Delta, S);
            
        kernel_dk_dv<<<grid_blocks, 128, smem_bytes, stream>>>(
            tma_Q, tma_K, tma_V, tma_dO, tma_dK, tma_dV,
            static_cast<const float*>(L.data_ptr()), d_Delta, S);
            
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaFreeAsync(d_Delta, stream));
    }

    TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);
}