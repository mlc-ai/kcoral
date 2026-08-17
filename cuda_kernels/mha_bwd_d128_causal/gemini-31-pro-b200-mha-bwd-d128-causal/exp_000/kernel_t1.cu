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

__device__ __forceinline__ uint64_t make_smem_desc_128b(void* smem_ptr, uint32_t K_full, bool is_k_major) {
    uint32_t sbo = 1024;
    uint32_t lbo = is_k_major ? 1 : (K_full / 8) * 1024;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint64_t d = 0;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
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

__device__ void cp_async_swizzled(const __nv_bfloat16* gmem, __nv_bfloat16* smem, int S_idx, int d_idx, int S_total) {
    for (int i = threadIdx.x; i < 64 * 64 / 8; i += blockDim.x) {
        int row = i / 8;
        int col_blk = i % 8;
        int g_row = S_idx + row;
        int g_col = d_idx + col_blk * 8;
        int swiz_col_blk = col_blk ^ (row % 8);
        
        uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(&smem[row * 64 + swiz_col_blk * 8]);
        if (g_row < S_total) {
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(&gmem[g_row * 128 + g_col]));
        } else {
            *(float4*)(&smem[row * 64 + swiz_col_blk * 8]) = {0,0,0,0};
        }
    }
}

__device__ void store_tmem_to_global(uint32_t tmem_base, __nv_bfloat16* smem, __nv_bfloat16* HBM, int S_idx, int d_idx, int S_total) {
    for (int col = 0; col < 64; col += 8) {
        uint32_t r[8];
        tmem_load_8x_fn(tmem_base + col, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
        tmem_load_fence_fn();
        if (threadIdx.x < 64) {
            for (int c = 0; c < 8; c++) smem[threadIdx.x * 64 + col + c] = __float2bfloat16(__uint_as_float(r[c]));
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 64 * 64 / 8; i += blockDim.x) {
        int row = i / 8;
        int col_blk = i % 8;
        int g_row = S_idx + row;
        if (g_row < S_total) {
            *(float4*)(&HBM[g_row * 128 + d_idx + col_blk * 8]) = *(float4*)(&smem[row * 64 + col_blk * 8]);
        }
    }
    __syncthreads();
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

__global__ void kernel_dq(const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V, const __nv_bfloat16* dO,
                          const float* L, const float* Delta, __nv_bfloat16* dQ, int S_total) {
    int b = blockIdx.z, h = blockIdx.y, i = blockIdx.x, S_idx = i * 64;
    if (S_idx >= S_total) return;

    int head_off = (b * gridDim.y * S_total + h * S_total) * 128;
    int S_off = (b * gridDim.y * S_total + h * S_total);
    const __nv_bfloat16 *Q_h = Q + head_off, *K_h = K + head_off, *V_h = V + head_off, *dO_h = dO + head_off;
    const float *L_h = L + S_off, *Delta_h = Delta + S_off;
    __nv_bfloat16* dQ_h = dQ + head_off;

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
    uint64_t* mbar = (uint64_t*)(smem_Delta + 64);
    
    if (threadIdx.x == 0) init_smem_barrier_fn(mbar, 1);

    uint32_t tmem_base; tmem_alloc_fn(&tmem_base, 256);
    uint32_t tmem_S = tmem_base, tmem_dP = tmem_base + 64, tmem_dQ0 = tmem_base + 128, tmem_dQ1 = tmem_base + 192;

    cp_async_swizzled(Q_h, smem_Q0, S_idx, 0, S_total); cp_async_swizzled(Q_h, smem_Q1, S_idx, 64, S_total);
    cp_async_swizzled(dO_h, smem_dO0, S_idx, 0, S_total); cp_async_swizzled(dO_h, smem_dO1, S_idx, 64, S_total);
    if (threadIdx.x < 64) {
        smem_L[threadIdx.x] = (S_idx + threadIdx.x < S_total) ? L_h[S_idx + threadIdx.x] : 0.0f;
        smem_Delta[threadIdx.x] = (S_idx + threadIdx.x < S_total) ? Delta_h[S_idx + threadIdx.x] : 0.0f;
    }
    asm volatile("cp.async.commit_group;\n cp.async.wait_group 0;\n" ::);
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

    float scale = 1.0f / sqrtf(128.0f);
    uint32_t phase = 0;

    for (int j = 0; j <= i; j++) {
        int K_idx = j * 64;
        cp_async_swizzled(K_h, smem_K0, K_idx, 0, S_total); cp_async_swizzled(K_h, smem_K1, K_idx, 64, S_total);
        cp_async_swizzled(V_h, smem_V0, K_idx, 0, S_total); cp_async_swizzled(V_h, smem_V1, K_idx, 64, S_total);
        asm volatile("cp.async.commit_group;\n cp.async.wait_group 0;\n" ::);
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

        uint32_t idesc_S = make_instr_desc(64, 64, true, false);
        __nv_bfloat16 *Q_ptrs[2] = {smem_Q0, smem_Q1}, *K_ptrs[2] = {smem_K0, smem_K1}, *dO_ptrs[2] = {smem_dO0, smem_dO1}, *V_ptrs[2] = {smem_V0, smem_V1};
        
        for (int p = 0; p < 2; p++) {
            uint64_t dAQ = make_smem_desc_128b(Q_ptrs[p], 64, false), dBK = make_smem_desc_128b(K_ptrs[p], 64, true);
            uint64_t dAO = make_smem_desc_128b(dO_ptrs[p], 64, false), dBV = make_smem_desc_128b(V_ptrs[p], 64, true);
            for (int k = 0; k < 4; k++) {
                int accum = (p > 0 || k > 0) ? 1 : 0;
                if (threadIdx.x == 0) {
                    asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, %4;\n" :: "r"(tmem_S), "l"(dAQ), "l"(dBK), "r"(idesc_S), "r"(accum));
                    asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, %4;\n" :: "r"(tmem_dP), "l"(dAO), "l"(dBV), "r"(idesc_S), "r"(accum));
                }
                advance_desc_k(dAQ, 32); advance_desc_k(dBK, 32); advance_desc_k(dAO, 32); advance_desc_k(dBV, 32);
            }
        }
        
        if (threadIdx.x == 0) asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(mbar)));
        mbarrier_wait_fn(mbar, phase); phase ^= 1;

        for (int col = 0; col < 64; col += 8) {
            uint32_t rS[8], rDP[8];
            tmem_load_8x_fn(tmem_S + col, &rS[0], &rS[1], &rS[2], &rS[3], &rS[4], &rS[5], &rS[6], &rS[7]);
            tmem_load_8x_fn(tmem_dP + col, &rDP[0], &rDP[1], &rDP[2], &rDP[3], &rDP[4], &rDP[5], &rDP[6], &rDP[7]);
            tmem_load_fence_fn();
            if (threadIdx.x < 64) {
                float lval = smem_L[threadIdx.x], delt = smem_Delta[threadIdx.x];
                for (int c = 0; c < 8; c++) {
                    float s = __uint_as_float(rS[c]), dp = __uint_as_float(rDP[c]), p = 0.0f, ds = 0.0f;
                    if (K_idx + col + c <= S_idx + threadIdx.x) {
                        p = fast_exp2f_fn((s * scale - lval) * 1.44269504f);
                        ds = p * (dp - delt) * scale;
                    }
                    smem_dS[threadIdx.x * 64 + ((col + c)/8 ^ (threadIdx.x % 8)) * 8 + (col + c)%8] = __float2bfloat16(ds);
                }
            }
        }
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

        uint32_t idesc_dQ = make_instr_desc(64, 64, true, true);
        uint32_t tmem_dQs[2] = {tmem_dQ0, tmem_dQ1};
        for (int p = 0; p < 2; p++) {
            uint64_t dA = make_smem_desc_128b(smem_dS, 64, false), dB = make_smem_desc_128b(K_ptrs[p], 64, false);
            for (int k = 0; k < 4; k++) {
                int accum = (j > 0 || k > 0) ? 1 : 0;
                if (threadIdx.x == 0) {
                    asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, %4;\n" :: "r"(tmem_dQs[p]), "l"(dA), "l"(dB), "r"(idesc_dQ), "r"(accum));
                }
                advance_desc_k(dA, 32); advance_desc_k(dB, 2048);
            }
        }
        if (threadIdx.x == 0) asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(mbar)));
        mbarrier_wait_fn(mbar, phase); phase ^= 1;
    }
    
    store_tmem_to_global(tmem_dQ0, smem_Q0, dQ_h, S_idx, 0, S_total);
    store_tmem_to_global(tmem_dQ1, smem_Q1, dQ_h, S_idx, 64, S_total);
    tmem_dealloc_fn(tmem_base, 256);
}

__global__ void kernel_dk_dv(const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V, const __nv_bfloat16* dO,
                             const float* L, const float* Delta, __nv_bfloat16* dK, __nv_bfloat16* dV, int S_total) {
    int b = blockIdx.z, h = blockIdx.y, j = blockIdx.x, K_idx = j * 64;
    if (K_idx >= S_total) return;

    int head_off = (b * gridDim.y * S_total + h * S_total) * 128;
    int S_off = (b * gridDim.y * S_total + h * S_total);
    const __nv_bfloat16 *Q_h = Q + head_off, *K_h = K + head_off, *V_h = V + head_off, *dO_h = dO + head_off;
    const float *L_h = L + S_off, *Delta_h = Delta + S_off;
    __nv_bfloat16 *dK_h = dK + head_off, *dV_h = dV + head_off;

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
    uint64_t* mbar = (uint64_t*)(smem_Delta + 64);
    
    if (threadIdx.x == 0) init_smem_barrier_fn(mbar, 1);

    uint32_t tmem_base; tmem_alloc_fn(&tmem_base, 512);
    uint32_t tmem_ST = tmem_base, tmem_dPT = tmem_base + 64, tmem_dK0 = tmem_base + 128, tmem_dK1 = tmem_base + 192, tmem_dV0 = tmem_base + 256, tmem_dV1 = tmem_base + 320;

    cp_async_swizzled(K_h, smem_K0, K_idx, 0, S_total); cp_async_swizzled(K_h, smem_K1, K_idx, 64, S_total);
    cp_async_swizzled(V_h, smem_V0, K_idx, 0, S_total); cp_async_swizzled(V_h, smem_V1, K_idx, 64, S_total);
    asm volatile("cp.async.commit_group;\n cp.async.wait_group 0;\n" ::);
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

    float scale = 1.0f / sqrtf(128.0f);
    uint32_t phase = 0;

    for (int i = j; i < (S_total + 63) / 64; i++) {
        int S_idx = i * 64;
        cp_async_swizzled(Q_h, smem_Q0, S_idx, 0, S_total); cp_async_swizzled(Q_h, smem_Q1, S_idx, 64, S_total);
        cp_async_swizzled(dO_h, smem_dO0, S_idx, 0, S_total); cp_async_swizzled(dO_h, smem_dO1, S_idx, 64, S_total);
        if (threadIdx.x < 64) {
            smem_L[threadIdx.x] = (S_idx + threadIdx.x < S_total) ? L_h[S_idx + threadIdx.x] : 0.0f;
            smem_Delta[threadIdx.x] = (S_idx + threadIdx.x < S_total) ? Delta_h[S_idx + threadIdx.x] : 0.0f;
        }
        asm volatile("cp.async.commit_group;\n cp.async.wait_group 0;\n" ::);
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

        uint32_t idesc_ST = make_instr_desc(64, 64, true, false);
        __nv_bfloat16 *Q_ptrs[2] = {smem_Q0, smem_Q1}, *K_ptrs[2] = {smem_K0, smem_K1}, *dO_ptrs[2] = {smem_dO0, smem_dO1}, *V_ptrs[2] = {smem_V0, smem_V1};
        
        for (int p = 0; p < 2; p++) {
            uint64_t dAK = make_smem_desc_128b(K_ptrs[p], 64, false), dBQ = make_smem_desc_128b(Q_ptrs[p], 64, true);
            uint64_t dAV = make_smem_desc_128b(V_ptrs[p], 64, false), dBO = make_smem_desc_128b(dO_ptrs[p], 64, true);
            for (int k = 0; k < 4; k++) {
                int accum = (p > 0 || k > 0) ? 1 : 0;
                if (threadIdx.x == 0) {
                    asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, %4;\n" :: "r"(tmem_ST), "l"(dAK), "l"(dBQ), "r"(idesc_ST), "r"(accum));
                    asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, %4;\n" :: "r"(tmem_dPT), "l"(dAV), "l"(dBO), "r"(idesc_ST), "r"(accum));
                }
                advance_desc_k(dAK, 32); advance_desc_k(dBQ, 32); advance_desc_k(dAV, 32); advance_desc_k(dBO, 32);
            }
        }
        
        if (threadIdx.x == 0) asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(mbar)));
        mbarrier_wait_fn(mbar, phase); phase ^= 1;

        for (int col = 0; col < 64; col += 8) {
            uint32_t rS[8], rDP[8];
            tmem_load_8x_fn(tmem_ST + col, &rS[0], &rS[1], &rS[2], &rS[3], &rS[4], &rS[5], &rS[6], &rS[7]);
            tmem_load_8x_fn(tmem_dPT + col, &rDP[0], &rDP[1], &rDP[2], &rDP[3], &rDP[4], &rDP[5], &rDP[6], &rDP[7]);
            tmem_load_fence_fn();
            if (threadIdx.x < 64) {
                for (int c = 0; c < 8; c++) {
                    float s = __uint_as_float(rS[c]), dp = __uint_as_float(rDP[c]), p = 0.0f, ds = 0.0f;
                    if (K_idx + threadIdx.x <= S_idx + col + c) {
                        p = fast_exp2f_fn((s * scale - smem_L[col + c]) * 1.44269504f);
                        ds = p * (dp - smem_Delta[col + c]) * scale;
                    }
                    int dest = threadIdx.x * 64 + ((col + c)/8 ^ (threadIdx.x % 8)) * 8 + (col + c)%8;
                    smem_dS[dest] = __float2bfloat16(ds);
                    smem_PT[dest] = __float2bfloat16(p);
                }
            }
        }
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

        uint32_t idesc_dK = make_instr_desc(64, 64, true, true);
        for (int p = 0; p < 2; p++) {
            uint64_t dAV = make_smem_desc_128b(smem_PT, 64, false), dBO = make_smem_desc_128b(dO_ptrs[p], 64, false);
            uint64_t dAK = make_smem_desc_128b(smem_dS, 64, false), dBQ = make_smem_desc_128b(Q_ptrs[p], 64, false);
            for (int k = 0; k < 4; k++) {
                int accum = (i > j || k > 0) ? 1 : 0;
                if (threadIdx.x == 0) {
                    asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, %4;\n" :: "r"(tmem_dV0 + p * 64), "l"(dAV), "l"(dBO), "r"(idesc_dK), "r"(accum));
                    asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, %4;\n" :: "r"(tmem_dK0 + p * 64), "l"(dAK), "l"(dBQ), "r"(idesc_dK), "r"(accum));
                }
                advance_desc_k(dAV, 32); advance_desc_k(dBO, 2048); advance_desc_k(dAK, 32); advance_desc_k(dBQ, 2048);
            }
        }
        if (threadIdx.x == 0) asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(mbar)));
        mbarrier_wait_fn(mbar, phase); phase ^= 1;
    }
    
    store_tmem_to_global(tmem_dK0, smem_Q0, dK_h, K_idx, 0, S_total);
    store_tmem_to_global(tmem_dK1, smem_Q1, dK_h, K_idx, 64, S_total);
    store_tmem_to_global(tmem_dV0, smem_V0, dV_h, K_idx, 0, S_total);
    store_tmem_to_global(tmem_dV1, smem_V1, dV_h, K_idx, 64, S_total);
    tmem_dealloc_fn(tmem_base, 512);
}

namespace tvm_ffi_example_cuda {
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
        
        int smem_bytes = 94208;
        CUDA_CHECK(cudaFuncSetAttribute(kernel_dq, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
        CUDA_CHECK(cudaFuncSetAttribute(kernel_dk_dv, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
        
        dim3 grid_blocks((S + 63) / 64, H, B);
        
        kernel_dq<<<grid_blocks, 128, smem_bytes, stream>>>(
            static_cast<const __nv_bfloat16*>(Q.data_ptr()), static_cast<const __nv_bfloat16*>(K.data_ptr()),
            static_cast<const __nv_bfloat16*>(V.data_ptr()), static_cast<const __nv_bfloat16*>(dO.data_ptr()),
            static_cast<const float*>(L.data_ptr()), d_Delta, static_cast<__nv_bfloat16*>(dQ.data_ptr()), S);
            
        kernel_dk_dv<<<grid_blocks, 128, smem_bytes, stream>>>(
            static_cast<const __nv_bfloat16*>(Q.data_ptr()), static_cast<const __nv_bfloat16*>(K.data_ptr()),
            static_cast<const __nv_bfloat16*>(V.data_ptr()), static_cast<const __nv_bfloat16*>(dO.data_ptr()),
            static_cast<const float*>(L.data_ptr()), d_Delta, static_cast<__nv_bfloat16*>(dK.data_ptr()), static_cast<__nv_bfloat16*>(dV.data_ptr()), S);
            
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaFreeAsync(d_Delta, stream));
    }

    TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);
}