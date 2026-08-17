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

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

__device__ __forceinline__ void tma_store_4d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group [%0, {%2, %3, %4, %5}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
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
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
        :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmem_load_4x(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tcgen05_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_commit_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(a));
}

__device__ __forceinline__ void umma_f16_cg1_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, bool accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"((uint32_t)accum));
}

// Since all SMEM arrays are loaded via SWIZZLE_NONE with box0=64, their inner chunk matches exactly generic 64 layout.
__device__ __forceinline__ uint64_t get_desc(void* smem_ptr, bool trans) {
    uint32_t cols = 64;
    uint32_t sbo, lbo;
    if (!trans) {
        sbo = 128;
        lbo = cols * 16;
    } else {
        lbo = 128;
        sbo = cols * 16;
    }
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)0 << 61; // None
    return d;
}

__device__ __forceinline__ uint64_t advance_desc(uint64_t desc, bool trans) {
    uint32_t cols = 64;
    uint32_t bytes = trans ? cols * 32 : 32;
    uint32_t addr = (desc & 0x3FFF) << 4;
    addr += bytes;
    desc &= ~0x3FFFull;
    desc |= (addr >> 4);
    return desc;
}

__device__ __forceinline__ void write_smem_unswizzled(void* smem_ptr, int row, int col, int cols_total, uint32_t val0, uint32_t val1) {
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem_ptr) + row * (cols_total * 2) + col * 2;
    asm volatile("st.shared.v2.b32 [%0], {%1, %2};" :: "r"(smem_addr), "r"(val0), "r"(val1) : "memory");
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, bool trans_a, bool trans_b) {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= (0u << 15);
    d |= (0u << 16);
    if (trans_a) d |= (1u << 15);
    if (trans_b) d |= (1u << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__global__ void precompute_D_kernel(const __nv_bfloat16* O, const __nv_bfloat16* dO, float* D, int S, int d) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < S) {
        float sum = 0.0f;
        int offset = (b * gridDim.y * S + h * S + s) * d;
        for (int i = 0; i < d; ++i) {
            float o_val = __bfloat162float(O[offset + i]);
            float do_val = __bfloat162float(dO[offset + i]);
            sum += o_val * do_val;
        }
        D[b * gridDim.y * S + h * S + s] = sum;
    }
}

__global__ __launch_bounds__(128) void mha_bwd_kernel1(
    const __grid_constant__ CUtensorMap tma_Q0, const __grid_constant__ CUtensorMap tma_Q1,
    const __grid_constant__ CUtensorMap tma_dO0, const __grid_constant__ CUtensorMap tma_dO1,
    const __grid_constant__ CUtensorMap tma_K0, const __grid_constant__ CUtensorMap tma_K1,
    const __grid_constant__ CUtensorMap tma_V0, const __grid_constant__ CUtensorMap tma_V1,
    const __grid_constant__ CUtensorMap tma_dQ0, const __grid_constant__ CUtensorMap tma_dQ1,
    const float* L, const float* D, int S_len) 
{
    setmaxnreg_inc_sync_fn<256>();
    int b = blockIdx.z;
    int h = blockIdx.y;
    int m = blockIdx.x * 128;
    if (m >= S_len) return;

    __shared__ __align__(128) __nv_bfloat16 smem_Q0[128 * 64];
    __shared__ __align__(128) __nv_bfloat16 smem_Q1[128 * 64];
    __shared__ __align__(128) __nv_bfloat16 smem_dO0[128 * 64];
    __shared__ __align__(128) __nv_bfloat16 smem_dO1[128 * 64];
    __shared__ __align__(128) __nv_bfloat16 smem_K0[2][64 * 64];
    __shared__ __align__(128) __nv_bfloat16 smem_K1[2][64 * 64];
    __shared__ __align__(128) __nv_bfloat16 smem_V0[2][64 * 64];
    __shared__ __align__(128) __nv_bfloat16 smem_V1[2][64 * 64];
    __shared__ __align__(128) __nv_bfloat16 smem_P[128 * 64];
    __shared__ float smem_L[128];
    __shared__ float smem_D[128];
    
    __shared__ uint64_t mbar_Q;
    __shared__ uint64_t mbar_KV[2];
    __shared__ uint64_t mbar_umma;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&mbar_Q, 1);
        init_smem_barrier_fn(&mbar_KV[0], 1);
        init_smem_barrier_fn(&mbar_KV[1], 1);
        init_smem_barrier_fn(&mbar_umma, 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar_Q, 4 * 128 * 64 * 2);
        tma_load_4d_fn(&tma_Q0, &mbar_Q, smem_Q0, 0, m, h, b);
        tma_load_4d_fn(&tma_Q1, &mbar_Q, smem_Q1, 64, m, h, b);
        tma_load_4d_fn(&tma_dO0, &mbar_Q, smem_dO0, 0, m, h, b);
        tma_load_4d_fn(&tma_dO1, &mbar_Q, smem_dO1, 64, m, h, b);
    }

    if (threadIdx.x < 128) {
        int idx = m + threadIdx.x;
        if (idx < S_len) {
            int offset = b * gridDim.y * S_len + h * S_len + idx;
            smem_L[threadIdx.x] = L[offset];
            smem_D[threadIdx.x] = D[offset];
        }
    }

    __shared__ uint32_t tmem_base;
    if (threadIdx.x < 32) tmem_alloc_fn(&tmem_base, 192);
    __syncthreads();
    
    uint32_t tmem_S = tmem_base;
    uint32_t tmem_dQ0 = tmem_base + 64;
    uint32_t tmem_dQ1 = tmem_base + 128;

    mbarrier_wait_fn(&mbar_Q, 0);

    int db = 0;
    uint32_t phase_KV[2] = {0, 0};
    uint32_t phase_umma = 0;
    
    if (threadIdx.x == 0 && S_len > 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar_KV[0], 4 * 64 * 64 * 2);
        tma_load_4d_fn(&tma_K0, &mbar_KV[0], smem_K0[0], 0, 0, h, b);
        tma_load_4d_fn(&tma_K1, &mbar_KV[0], smem_K1[0], 64, 0, h, b);
        tma_load_4d_fn(&tma_V0, &mbar_KV[0], smem_V0[0], 0, 0, h, b);
        tma_load_4d_fn(&tma_V1, &mbar_KV[0], smem_V1[0], 64, 0, h, b);
    }

    float scale = 1.0f / sqrtf(128.0f);
    float l_val = smem_L[threadIdx.x];
    float d_val = smem_D[threadIdx.x];
    bool valid_row = (m + threadIdx.x < S_len);

    for (int n = 0; n < S_len; n += 64) {
        mbarrier_wait_fn(&mbar_KV[db], phase_KV[db]);
        phase_KV[db] ^= 1;

        int next_n = n + 64;
        if (threadIdx.x == 0 && next_n < S_len) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_KV[db^1], 4 * 64 * 64 * 2);
            tma_load_4d_fn(&tma_K0, &mbar_KV[db^1], smem_K0[db^1], 0, next_n, h, b);
            tma_load_4d_fn(&tma_K1, &mbar_KV[db^1], smem_K1[db^1], 64, next_n, h, b);
            tma_load_4d_fn(&tma_V0, &mbar_KV[db^1], smem_V0[db^1], 0, next_n, h, b);
            tma_load_4d_fn(&tma_V1, &mbar_KV[db^1], smem_V1[db^1], 64, next_n, h, b);
        }

        if (threadIdx.x == 0) {
            uint64_t desc_Q0 = get_desc(smem_Q0, false);
            uint64_t desc_K0 = get_desc(smem_K0[db], true);
            uint32_t idesc_S = make_instr_desc_fn(128, 64, false, true);
            for (int k = 0; k < 4; ++k) {
                umma_f16_cg1_fn(tmem_S, desc_Q0, desc_K0, idesc_S, k > 0);
                desc_Q0 = advance_desc(desc_Q0, false);
                desc_K0 = advance_desc(desc_K0, true);
            }
            uint64_t desc_Q1 = get_desc(smem_Q1, false);
            uint64_t desc_K1 = get_desc(smem_K1[db], true);
            for (int k = 0; k < 4; ++k) {
                umma_f16_cg1_fn(tmem_S, desc_Q1, desc_K1, idesc_S, true);
                desc_Q1 = advance_desc(desc_Q1, false);
                desc_K1 = advance_desc(desc_K1, true);
            }
            tcgen05_commit_fn(&mbar_umma);
        }
        mbarrier_wait_fn(&mbar_umma, phase_umma);
        phase_umma ^= 1;

        uint32_t p_regs[32];
        for (int c = 0; c < 64; c += 4) {
            uint32_t r[4];
            tmem_load_4x(tmem_S + c, &r[0], &r[1], &r[2], &r[3]);
            tcgen05_wait_ld();
            float f[4];
            f[0] = __uint_as_float(r[0]); f[1] = __uint_as_float(r[1]); f[2] = __uint_as_float(r[2]); f[3] = __uint_as_float(r[3]);
            
            bool valid_0 = valid_row && (n + c + 0 < S_len);
            bool valid_1 = valid_row && (n + c + 1 < S_len);
            bool valid_2 = valid_row && (n + c + 2 < S_len);
            bool valid_3 = valid_row && (n + c + 3 < S_len);
            
            f[0] = valid_0 ? fast_exp2f_fn((f[0] * scale - l_val) * 1.44269504f) : 0.0f;
            f[1] = valid_1 ? fast_exp2f_fn((f[1] * scale - l_val) * 1.44269504f) : 0.0f;
            f[2] = valid_2 ? fast_exp2f_fn((f[2] * scale - l_val) * 1.44269504f) : 0.0f;
            f[3] = valid_3 ? fast_exp2f_fn((f[3] * scale - l_val) * 1.44269504f) : 0.0f;
            
            p_regs[(c/4)*2 + 0] = pack_bf16_fn(__float_as_uint(f[0]), __float_as_uint(f[1]));
            p_regs[(c/4)*2 + 1] = pack_bf16_fn(__float_as_uint(f[2]), __float_as_uint(f[3]));
            
            uint32_t* smem_ptr = (uint32_t*)&smem_P[threadIdx.x * 64 + c];
            smem_ptr[0] = p_regs[(c/4)*2 + 0];
            smem_ptr[1] = p_regs[(c/4)*2 + 1];
        }

        if (threadIdx.x == 0) {
            uint64_t desc_dO0 = get_desc(smem_dO0, false);
            uint64_t desc_V0 = get_desc(smem_V0[db], false);
            uint32_t idesc_dP = make_instr_desc_fn(128, 64, false, false);
            for (int k = 0; k < 4; ++k) {
                umma_f16_cg1_fn(tmem_S, desc_dO0, desc_V0, idesc_dP, k > 0);
                desc_dO0 = advance_desc(desc_dO0, false);
                desc_V0 = advance_desc(desc_V0, false);
            }
            uint64_t desc_dO1 = get_desc(smem_dO1, false);
            uint64_t desc_V1 = get_desc(smem_V1[db], false);
            for (int k = 0; k < 4; ++k) {
                umma_f16_cg1_fn(tmem_S, desc_dO1, desc_V1, idesc_dP, true);
                desc_dO1 = advance_desc(desc_dO1, false);
                desc_V1 = advance_desc(desc_V1, false);
            }
            tcgen05_commit_fn(&mbar_umma);
        }
        mbarrier_wait_fn(&mbar_umma, phase_umma);
        phase_umma ^= 1;

        for (int c = 0; c < 64; c += 4) {
            uint32_t r[4];
            tmem_load_4x(tmem_S + c, &r[0], &r[1], &r[2], &r[3]);
            tcgen05_wait_ld();
            uint32_t p01 = p_regs[(c/4)*2 + 0];
            uint32_t p23 = p_regs[(c/4)*2 + 1];
            __nv_bfloat16* p_ptr = (__nv_bfloat16*)&p01;
            float p0 = __bfloat162float(p_ptr[0]); float p1 = __bfloat162float(p_ptr[1]);
            p_ptr = (__nv_bfloat16*)&p23;
            float p2 = __bfloat162float(p_ptr[0]); float p3 = __bfloat162float(p_ptr[1]);
            
            float d0 = p0 * (__uint_as_float(r[0]) - d_val) * scale;
            float d1 = p1 * (__uint_as_float(r[1]) - d_val) * scale;
            float d2 = p2 * (__uint_as_float(r[2]) - d_val) * scale;
            float d3 = p3 * (__uint_as_float(r[3]) - d_val) * scale;
            
            uint32_t ds01 = pack_bf16_fn(__float_as_uint(d0), __float_as_uint(d1));
            uint32_t ds23 = pack_bf16_fn(__float_as_uint(d2), __float_as_uint(d3));
            write_smem_unswizzled(smem_P, threadIdx.x, c, 64, ds01, ds23);
        }

        __syncthreads();
        fence_proxy_async_fn();

        if (threadIdx.x == 0) {
            uint64_t desc_dS = get_desc(smem_P, false);
            uint64_t desc_K0_dQ = get_desc(smem_K0[db], true);
            uint32_t idesc_dQ = make_instr_desc_fn(128, 64, false, true);
            for (int k = 0; k < 4; ++k) {
                umma_f16_cg1_fn(tmem_dQ0, desc_dS, desc_K0_dQ, idesc_dQ, (n > 0) || (k > 0));
                desc_dS = advance_desc(desc_dS, false);
                desc_K0_dQ = advance_desc(desc_K0_dQ, true);
            }
            desc_dS = get_desc(smem_P, false);
            uint64_t desc_K1_dQ = get_desc(smem_K1[db], true);
            for (int k = 0; k < 4; ++k) {
                umma_f16_cg1_fn(tmem_dQ1, desc_dS, desc_K1_dQ, idesc_dQ, (n > 0) || (k > 0));
                desc_dS = advance_desc(desc_dS, false);
                desc_K1_dQ = advance_desc(desc_K1_dQ, true);
            }
            tcgen05_commit_fn(&mbar_umma);
        }
        mbarrier_wait_fn(&mbar_umma, phase_umma);
        phase_umma ^= 1;
        
        db ^= 1;
    }

    for (int c = 0; c < 64; c += 4) {
        uint32_t r[4];
        tmem_load_4x(tmem_dQ0 + c, &r[0], &r[1], &r[2], &r[3]);
        tcgen05_wait_ld();
        uint32_t q01 = pack_bf16_fn(r[0], r[1]);
        uint32_t q23 = pack_bf16_fn(r[2], r[3]);
        uint32_t* ptr = (uint32_t*)&smem_Q0[threadIdx.x * 64 + c];
        ptr[0] = q01; ptr[1] = q23;
    }
    for (int c = 0; c < 64; c += 4) {
        uint32_t r[4];
        tmem_load_4x(tmem_dQ1 + c, &r[0], &r[1], &r[2], &r[3]);
        tcgen05_wait_ld();
        uint32_t q01 = pack_bf16_fn(r[0], r[1]);
        uint32_t q23 = pack_bf16_fn(r[2], r[3]);
        uint32_t* ptr = (uint32_t*)&smem_Q1[threadIdx.x * 64 + c];
        ptr[0] = q01; ptr[1] = q23;
    }

    __syncthreads();
    fence_proxy_async_fn();
    
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_base, 192);
    }
    if (threadIdx.x == 0) {
        tma_store_4d_fn(&tma_dQ0, smem_Q0, 0, m, h, b);
        tma_store_4d_fn(&tma_dQ1, smem_Q1, 64, m, h, b);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }
}

__global__ __launch_bounds__(128) void mha_bwd_kernel2(
    const __grid_constant__ CUtensorMap tma_K0, const __grid_constant__ CUtensorMap tma_K1,
    const __grid_constant__ CUtensorMap tma_V0, const __grid_constant__ CUtensorMap tma_V1,
    const __grid_constant__ CUtensorMap tma_Q0, const __grid_constant__ CUtensorMap tma_Q1,
    const __grid_constant__ CUtensorMap tma_dO0, const __grid_constant__ CUtensorMap tma_dO1,
    const __grid_constant__ CUtensorMap tma_dK0, const __grid_constant__ CUtensorMap tma_dK1,
    const __grid_constant__ CUtensorMap tma_dV0, const __grid_constant__ CUtensorMap tma_dV1,
    const float* L, const float* D, int S_len) 
{
    setmaxnreg_inc_sync_fn<256>();
    int b = blockIdx.z;
    int h = blockIdx.y;
    int n = blockIdx.x * 128;
    if (n >= S_len) return;

    __shared__ __align__(128) __nv_bfloat16 smem_K0[128 * 64];
    __shared__ __align__(128) __nv_bfloat16 smem_K1[128 * 64];
    __shared__ __align__(128) __nv_bfloat16 smem_V0[128 * 64];
    __shared__ __align__(128) __nv_bfloat16 smem_V1[128 * 64];
    __shared__ __align__(128) __nv_bfloat16 smem_Q0[2][64 * 64];
    __shared__ __align__(128) __nv_bfloat16 smem_Q1[2][64 * 64];
    __shared__ __align__(128) __nv_bfloat16 smem_dO0[2][64 * 64];
    __shared__ __align__(128) __nv_bfloat16 smem_dO1[2][64 * 64];
    __shared__ __align__(128) __nv_bfloat16 smem_P[128 * 64];
    __shared__ float smem_L[64];
    __shared__ float smem_D[64];
    
    __shared__ uint64_t mbar_KV;
    __shared__ uint64_t mbar_Q[2];
    __shared__ uint64_t mbar_umma;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&mbar_KV, 1);
        init_smem_barrier_fn(&mbar_Q[0], 1);
        init_smem_barrier_fn(&mbar_Q[1], 1);
        init_smem_barrier_fn(&mbar_umma, 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar_KV, 4 * 128 * 64 * 2);
        tma_load_4d_fn(&tma_K0, &mbar_KV, smem_K0, 0, n, h, b);
        tma_load_4d_fn(&tma_K1, &mbar_KV, smem_K1, 64, n, h, b);
        tma_load_4d_fn(&tma_V0, &mbar_KV, smem_V0, 0, n, h, b);
        tma_load_4d_fn(&tma_V1, &mbar_KV, smem_V1, 64, n, h, b);
    }

    __shared__ uint32_t tmem_base;
    if (threadIdx.x < 32) tmem_alloc_fn(&tmem_base, 320);
    __syncthreads();
    
    uint32_t tmem_S = tmem_base;
    uint32_t tmem_dK0 = tmem_base + 64;
    uint32_t tmem_dK1 = tmem_base + 128;
    uint32_t tmem_dV0 = tmem_base + 192;
    uint32_t tmem_dV1 = tmem_base + 256;

    mbarrier_wait_fn(&mbar_KV, 0);

    int db = 0;
    uint32_t phase_Q[2] = {0, 0};
    uint32_t phase_umma = 0;
    
    if (threadIdx.x == 0 && S_len > 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar_Q[0], 4 * 64 * 64 * 2);
        tma_load_4d_fn(&tma_Q0, &mbar_Q[0], smem_Q0[0], 0, 0, h, b);
        tma_load_4d_fn(&tma_Q1, &mbar_Q[0], smem_Q1[0], 64, 0, h, b);
        tma_load_4d_fn(&tma_dO0, &mbar_Q[0], smem_dO0[0], 0, 0, h, b);
        tma_load_4d_fn(&tma_dO1, &mbar_Q[0], smem_dO1[0], 64, 0, h, b);
    }

    float scale = 1.0f / sqrtf(128.0f);

    for (int m = 0; m < S_len; m += 64) {
        mbarrier_wait_fn(&mbar_Q[db], phase_Q[db]);
        phase_Q[db] ^= 1;
        
        int next_m = m + 64;
        if (threadIdx.x == 0 && next_m < S_len) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_Q[db^1], 4 * 64 * 64 * 2);
            tma_load_4d_fn(&tma_Q0, &mbar_Q[db^1], smem_Q0[db^1], 0, next_m, h, b);
            tma_load_4d_fn(&tma_Q1, &mbar_Q[db^1], smem_Q1[db^1], 64, next_m, h, b);
            tma_load_4d_fn(&tma_dO0, &mbar_Q[db^1], smem_dO0[db^1], 0, next_m, h, b);
            tma_load_4d_fn(&tma_dO1, &mbar_Q[db^1], smem_dO1[db^1], 64, next_m, h, b);
        }

        if (threadIdx.x < 64) {
            int idx = m + threadIdx.x;
            if (idx < S_len) {
                int offset = b * gridDim.y * S_len + h * S_len + idx;
                smem_L[threadIdx.x] = L[offset];
                smem_D[threadIdx.x] = D[offset];
            } else {
                smem_L[threadIdx.x] = 0.0f;
                smem_D[threadIdx.x] = 0.0f;
            }
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            uint64_t desc_K0 = get_desc(smem_K0, false);
            uint64_t desc_Q0 = get_desc(smem_Q0[db], false);
            uint32_t idesc_S = make_instr_desc_fn(128, 64, false, false);
            for (int k = 0; k < 4; ++k) {
                umma_f16_cg1_fn(tmem_S, desc_K0, desc_Q0, idesc_S, k > 0);
                desc_K0 = advance_desc(desc_K0, false);
                desc_Q0 = advance_desc(desc_Q0, false);
            }
            uint64_t desc_K1 = get_desc(smem_K1, false);
            uint64_t desc_Q1 = get_desc(smem_Q1[db], false);
            for (int k = 0; k < 4; ++k) {
                umma_f16_cg1_fn(tmem_S, desc_K1, desc_Q1, idesc_S, true);
                desc_K1 = advance_desc(desc_K1, false);
                desc_Q1 = advance_desc(desc_Q1, false);
            }
            tcgen05_commit_fn(&mbar_umma);
        }
        mbarrier_wait_fn(&mbar_umma, phase_umma);
        phase_umma ^= 1;

        uint32_t p_regs[32];
        for (int c = 0; c < 64; c += 4) {
            uint32_t r[4];
            tmem_load_4x(tmem_S + c, &r[0], &r[1], &r[2], &r[3]);
            tcgen05_wait_ld();
            float f[4];
            f[0] = __uint_as_float(r[0]); f[1] = __uint_as_float(r[1]); f[2] = __uint_as_float(r[2]); f[3] = __uint_as_float(r[3]);
            bool valid = (n + threadIdx.x < S_len);
            bool valid_0 = valid && (m + c + 0 < S_len);
            bool valid_1 = valid && (m + c + 1 < S_len);
            bool valid_2 = valid && (m + c + 2 < S_len);
            bool valid_3 = valid && (m + c + 3 < S_len);
            float l_val = smem_L[c+0];
            f[0] = valid_0 ? fast_exp2f_fn((f[0] * scale - l_val) * 1.44269504f) : 0.0f;
            l_val = smem_L[c+1];
            f[1] = valid_1 ? fast_exp2f_fn((f[1] * scale - l_val) * 1.44269504f) : 0.0f;
            l_val = smem_L[c+2];
            f[2] = valid_2 ? fast_exp2f_fn((f[2] * scale - l_val) * 1.44269504f) : 0.0f;
            l_val = smem_L[c+3];
            f[3] = valid_3 ? fast_exp2f_fn((f[3] * scale - l_val) * 1.44269504f) : 0.0f;
            p_regs[(c/4)*2 + 0] = pack_bf16_fn(__float_as_uint(f[0]), __float_as_uint(f[1]));
            p_regs[(c/4)*2 + 1] = pack_bf16_fn(__float_as_uint(f[2]), __float_as_uint(f[3]));
            write_smem_unswizzled(smem_P, threadIdx.x, c, 64, p_regs[(c/4)*2 + 0], p_regs[(c/4)*2 + 1]);
        }
        
        __syncthreads();
        fence_proxy_async_fn();

        if (threadIdx.x == 0) {
            uint64_t desc_P = get_desc(smem_P, true);
            uint64_t desc_dO0_dV = get_desc(smem_dO0[db], true);
            uint32_t idesc_dV = make_instr_desc_fn(128, 64, true, true);
            for (int k = 0; k < 4; ++k) {
                umma_f16_cg1_fn(tmem_dV0, desc_P, desc_dO0_dV, idesc_dV, (m > 0) || (k > 0));
                desc_P = advance_desc(desc_P, true);
                desc_dO0_dV = advance_desc(desc_dO0_dV, true);
            }
            desc_P = get_desc(smem_P, true);
            uint64_t desc_dO1_dV = get_desc(smem_dO1[db], true);
            for (int k = 0; k < 4; ++k) {
                umma_f16_cg1_fn(tmem_dV1, desc_P, desc_dO1_dV, idesc_dV, (m > 0) || (k > 0));
                desc_P = advance_desc(desc_P, true);
                desc_dO1_dV = advance_desc(desc_dO1_dV, true);
            }
            tcgen05_commit_fn(&mbar_umma);
        }
        mbarrier_wait_fn(&mbar_umma, phase_umma);
        phase_umma ^= 1;

        if (threadIdx.x == 0) {
            uint64_t desc_V0 = get_desc(smem_V0, false);
            uint64_t desc_dO0_dP = get_desc(smem_dO0[db], false);
            uint32_t idesc_dP = make_instr_desc_fn(128, 64, false, false);
            for (int k = 0; k < 4; ++k) {
                umma_f16_cg1_fn(tmem_S, desc_V0, desc_dO0_dP, idesc_dP, k > 0);
                desc_V0 = advance_desc(desc_V0, false);
                desc_dO0_dP = advance_desc(desc_dO0_dP, false);
            }
            uint64_t desc_V1 = get_desc(smem_V1, false);
            uint64_t desc_dO1_dP = get_desc(smem_dO1[db], false);
            for (int k = 0; k < 4; ++k) {
                umma_f16_cg1_fn(tmem_S, desc_V1, desc_dO1_dP, idesc_dP, true);
                desc_V1 = advance_desc(desc_V1, false);
                desc_dO1_dP = advance_desc(desc_dO1_dP, false);
            }
            tcgen05_commit_fn(&mbar_umma);
        }
        mbarrier_wait_fn(&mbar_umma, phase_umma);
        phase_umma ^= 1;
        
        for (int c = 0; c < 64; c += 4) {
            uint32_t r[4];
            tmem_load_4x(tmem_S + c, &r[0], &r[1], &r[2], &r[3]);
            tcgen05_wait_ld();
            uint32_t p01 = p_regs[(c/4)*2 + 0];
            uint32_t p23 = p_regs[(c/4)*2 + 1];
            __nv_bfloat16* p_ptr = (__nv_bfloat16*)&p01;
            float p0 = __bfloat162float(p_ptr[0]); float p1 = __bfloat162float(p_ptr[1]);
            p_ptr = (__nv_bfloat16*)&p23;
            float p2 = __bfloat162float(p_ptr[0]); float p3 = __bfloat162float(p_ptr[1]);
            
            float d_val = smem_D[c+0];
            float d0 = p0 * (__uint_as_float(r[0]) - d_val) * scale;
            d_val = smem_D[c+1];
            float d1 = p1 * (__uint_as_float(r[1]) - d_val) * scale;
            d_val = smem_D[c+2];
            float d2 = p2 * (__uint_as_float(r[2]) - d_val) * scale;
            d_val = smem_D[c+3];
            float d3 = p3 * (__uint_as_float(r[3]) - d_val) * scale;
            
            uint32_t ds01 = pack_bf16_fn(__float_as_uint(d0), __float_as_uint(d1));
            uint32_t ds23 = pack_bf16_fn(__float_as_uint(d2), __float_as_uint(d3));
            write_smem_unswizzled(smem_P, threadIdx.x, c, 64, ds01, ds23);
        }

        __syncthreads();
        fence_proxy_async_fn();

        if (threadIdx.x == 0) {
            uint64_t desc_dS = get_desc(smem_P, true);
            uint64_t desc_Q0_dK = get_desc(smem_Q0[db], true);
            uint32_t idesc_dK = make_instr_desc_fn(128, 64, true, true);
            for (int k = 0; k < 4; ++k) {
                umma_f16_cg1_fn(tmem_dK0, desc_dS, desc_Q0_dK, idesc_dK, (m > 0) || (k > 0));
                desc_dS = advance_desc(desc_dS, true);
                desc_Q0_dK = advance_desc(desc_Q0_dK, true);
            }
            desc_dS = get_desc(smem_P, true);
            uint64_t desc_Q1_dK = get_desc(smem_Q1[db], true);
            for (int k = 0; k < 4; ++k) {
                umma_f16_cg1_fn(tmem_dK1, desc_dS, desc_Q1_dK, idesc_dK, (m > 0) || (k > 0));
                desc_dS = advance_desc(desc_dS, true);
                desc_Q1_dK = advance_desc(desc_Q1_dK, true);
            }
            tcgen05_commit_fn(&mbar_umma);
        }
        mbarrier_wait_fn(&mbar_umma, phase_umma);
        phase_umma ^= 1;
        db ^= 1;
    }

    for (int c = 0; c < 64; c += 4) {
        uint32_t r[4];
        tmem_load_4x(tmem_dK0 + c, &r[0], &r[1], &r[2], &r[3]); tcgen05_wait_ld();
        ((uint32_t*)&smem_K0[threadIdx.x * 64 + c])[0] = pack_bf16_fn(r[0], r[1]);
        ((uint32_t*)&smem_K0[threadIdx.x * 64 + c])[1] = pack_bf16_fn(r[2], r[3]);
        tmem_load_4x(tmem_dK1 + c, &r[0], &r[1], &r[2], &r[3]); tcgen05_wait_ld();
        ((uint32_t*)&smem_K1[threadIdx.x * 64 + c])[0] = pack_bf16_fn(r[0], r[1]);
        ((uint32_t*)&smem_K1[threadIdx.x * 64 + c])[1] = pack_bf16_fn(r[2], r[3]);
        
        tmem_load_4x(tmem_dV0 + c, &r[0], &r[1], &r[2], &r[3]); tcgen05_wait_ld();
        ((uint32_t*)&smem_V0[threadIdx.x * 64 + c])[0] = pack_bf16_fn(r[0], r[1]);
        ((uint32_t*)&smem_V0[threadIdx.x * 64 + c])[1] = pack_bf16_fn(r[2], r[3]);
        tmem_load_4x(tmem_dV1 + c, &r[0], &r[1], &r[2], &r[3]); tcgen05_wait_ld();
        ((uint32_t*)&smem_V1[threadIdx.x * 64 + c])[0] = pack_bf16_fn(r[0], r[1]);
        ((uint32_t*)&smem_V1[threadIdx.x * 64 + c])[1] = pack_bf16_fn(r[2], r[3]);
    }

    __syncthreads();
    fence_proxy_async_fn();
    
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_base, 320);
    }
    if (threadIdx.x == 0) {
        tma_store_4d_fn(&tma_dK0, smem_K0, 0, n, h, b);
        tma_store_4d_fn(&tma_dK1, smem_K1, 64, n, h, b);
        tma_store_4d_fn(&tma_dV0, smem_V0, 0, n, h, b);
        tma_store_4d_fn(&tma_dV1, smem_V1, 64, n, h, b);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }
}

namespace tvm_ffi_mha_bwd {

CUresult create_tma_4d_descriptor_2B(CUtensorMap* d, void* globalAddress, 
    uint64_t dim0, uint64_t dim1, uint64_t dim2, uint64_t dim3,
    uint32_t box0, uint32_t box1) {
    cuuint64_t globalDim[4] = {dim0, dim1, dim2, dim3};
    cuuint64_t globalStrides[3] = {dim0*2, dim0*dim1*2, dim0*dim1*dim2*2};
    cuuint32_t boxDim[4] = {box0, box1, 1, 1};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, globalAddress, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    float* D_global;
    CUDA_CHECK(cudaMalloc(&D_global, B * H * S * sizeof(float)));
    
    dim3 grid_D((S + 127) / 128, H, B);
    precompute_D_kernel<<<grid_D, 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(O.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        D_global, S, d);
    CUDA_CHECK(cudaGetLastError());
    
    CUtensorMap tma_Q0_k1, tma_Q1_k1, tma_dO0_k1, tma_dO1_k1, tma_K0_k1, tma_K1_k1, tma_V0_k1, tma_V1_k1, tma_dQ0_k1, tma_dQ1_k1;
    create_tma_4d_descriptor_2B(&tma_Q0_k1, Q.data_ptr(), 128, S, H, B, 64, 128);
    create_tma_4d_descriptor_2B(&tma_Q1_k1, Q.data_ptr(), 128, S, H, B, 64, 128);
    create_tma_4d_descriptor_2B(&tma_dO0_k1, dO.data_ptr(), 128, S, H, B, 64, 128);
    create_tma_4d_descriptor_2B(&tma_dO1_k1, dO.data_ptr(), 128, S, H, B, 64, 128);
    create_tma_4d_descriptor_2B(&tma_K0_k1, K.data_ptr(), 128, S, H, B, 64, 64);
    create_tma_4d_descriptor_2B(&tma_K1_k1, K.data_ptr(), 128, S, H, B, 64, 64);
    create_tma_4d_descriptor_2B(&tma_V0_k1, V.data_ptr(), 128, S, H, B, 64, 64);
    create_tma_4d_descriptor_2B(&tma_V1_k1, V.data_ptr(), 128, S, H, B, 64, 64);
    create_tma_4d_descriptor_2B(&tma_dQ0_k1, dQ.data_ptr(), 128, S, H, B, 64, 128);
    create_tma_4d_descriptor_2B(&tma_dQ1_k1, dQ.data_ptr(), 128, S, H, B, 64, 128);
    
    dim3 grid1((S + 127) / 128, H, B);
    mha_bwd_kernel1<<<grid1, 128, 0, stream>>>(
        tma_Q0_k1, tma_Q1_k1, tma_dO0_k1, tma_dO1_k1, 
        tma_K0_k1, tma_K1_k1, tma_V0_k1, tma_V1_k1, 
        tma_dQ0_k1, tma_dQ1_k1,
        static_cast<const float*>(L.data_ptr()), D_global, S);
    CUDA_CHECK(cudaGetLastError());

    CUtensorMap tma_K0_k2, tma_K1_k2, tma_V0_k2, tma_V1_k2, tma_Q0_k2, tma_Q1_k2, tma_dO0_k2, tma_dO1_k2, tma_dK0_k2, tma_dK1_k2, tma_dV0_k2, tma_dV1_k2;
    create_tma_4d_descriptor_2B(&tma_K0_k2, K.data_ptr(), 128, S, H, B, 64, 128);
    create_tma_4d_descriptor_2B(&tma_K1_k2, K.data_ptr(), 128, S, H, B, 64, 128);
    create_tma_4d_descriptor_2B(&tma_V0_k2, V.data_ptr(), 128, S, H, B, 64, 128);
    create_tma_4d_descriptor_2B(&tma_V1_k2, V.data_ptr(), 128, S, H, B, 64, 128);
    create_tma_4d_descriptor_2B(&tma_Q0_k2, Q.data_ptr(), 128, S, H, B, 64, 64);
    create_tma_4d_descriptor_2B(&tma_Q1_k2, Q.data_ptr(), 128, S, H, B, 64, 64);
    create_tma_4d_descriptor_2B(&tma_dO0_k2, dO.data_ptr(), 128, S, H, B, 64, 64);
    create_tma_4d_descriptor_2B(&tma_dO1_k2, dO.data_ptr(), 128, S, H, B, 64, 64);
    create_tma_4d_descriptor_2B(&tma_dK0_k2, dK.data_ptr(), 128, S, H, B, 64, 128);
    create_tma_4d_descriptor_2B(&tma_dK1_k2, dK.data_ptr(), 128, S, H, B, 64, 128);
    create_tma_4d_descriptor_2B(&tma_dV0_k2, dV.data_ptr(), 128, S, H, B, 64, 128);
    create_tma_4d_descriptor_2B(&tma_dV1_k2, dV.data_ptr(), 128, S, H, B, 64, 128);
    
    mha_bwd_kernel2<<<grid1, 128, 0, stream>>>(
        tma_K0_k2, tma_K1_k2, tma_V0_k2, tma_V1_k2, 
        tma_Q0_k2, tma_Q1_k2, tma_dO0_k2, tma_dO1_k2, 
        tma_dK0_k2, tma_dK1_k2, 
        tma_dV0_k2, tma_dV1_k2,
        static_cast<const float*>(L.data_ptr()), D_global, S);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaFree(D_global));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

}  // namespace tvm_ffi_mha_bwd