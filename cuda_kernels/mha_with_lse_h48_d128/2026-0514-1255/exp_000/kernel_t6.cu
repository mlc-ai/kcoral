#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <stdio.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

// ----------------------------------------------------------------------------
// TMEM and MBarrier Helper Functions
// ----------------------------------------------------------------------------

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count) : "memory");
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
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase) : "memory");
}

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_before_fn() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(a), "r"(ncols) : "memory");
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(addr), "r"(ncols) : "memory");
}

__device__ __forceinline__ void tmem_load_8x_fn(uint32_t col,
    uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
    uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),
                   "=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(col) : "memory");
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void umma_f16_cg1_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum) : "memory");
}

// ----------------------------------------------------------------------------
// Descriptor construction and TMA Ops
// ----------------------------------------------------------------------------

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, bool trans_a, bool trans_b) {
    uint32_t d = 0;
    d |= (1u << 4);           // c_format = FP32
    d |= (1u << 7);           // a_format = BF16
    d |= (1u << 10);          // b_format = BF16
    if (trans_a) d |= (1u << 15);
    if (trans_b) d |= (1u << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)((addr >> 4) & 0x3FFF);
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;   
    uint64_t base_offset = (addr >> 7) & 0x7;
    d |= (base_offset << 49);
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_mn_major(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)((addr >> 4) & 0x3FFF);
    d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;   
    uint64_t base_offset = (addr >> 7) & 0x7;
    d |= (base_offset << 49);
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint64_t advance_smem_desc_k(uint64_t desc, uint32_t bytes) {
    uint32_t addr = (desc & 0x3FFF) << 4;
    addr = addr + bytes;
    desc &= ~0x3FFFull;
    desc &= ~(0x7ull << 49);
    desc |= (addr >> 4) & 0x3FFF;
    uint64_t base_offset = (addr >> 7) & 0x7;
    desc |= (base_offset << 49);
    return desc;
}

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ void tma_store_3d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [%0, {%2, %3, %4}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
}

// ----------------------------------------------------------------------------
// Math & Swizzling
// ----------------------------------------------------------------------------

__device__ __forceinline__ uint32_t pack_bf16_fn(uint32_t fp32_a, uint32_t fp32_b) {
    __nv_bfloat16 a = __float2bfloat16(__uint_as_float(fp32_a));
    __nv_bfloat16 b = __float2bfloat16(__uint_as_float(fp32_b));
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};" : "=r"(result) : "h"(*reinterpret_cast<uint16_t*>(&a)), "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}

__device__ __forceinline__ uint32_t swizzle_128B(uint32_t row, uint32_t col_bytes) {
    uint32_t x_chunk = (col_bytes / 16) % 8;
    uint32_t y_chunk = row % 8;
    uint32_t swizzled_x_chunk = x_chunk ^ y_chunk;
    return (col_bytes & ~0x7F) | (swizzled_x_chunk * 16) | (col_bytes & 0xF);
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// Emulates 2^x via Cody-Waite polynomial approximation ensuring optimal pipelining overlay
__device__ __forceinline__ float2 ex2_emulation_packed_asm_fn(float x, float y) {
    float ox, oy;
    asm("{\n\t"
        ".reg .f32 f1, f2, f3, f4, f5, f6, f7;\n\t"
        ".reg .b64 l1, l2, l3, l4, l5, l6, l7, l8, l9, l10;\n\t"
        ".reg .s32 r1, r2, r3, r4, r5, r6, r7, r8;\n\t"
        "max.ftz.f32 f1, %2, 0fC2FE0000;\n\t"
        "max.ftz.f32 f2, %3, 0fC2FE0000;\n\t"
        "mov.b64 l1, {f1, f2};\n\t"
        "mov.f32 f3, 0f4B400000;\n\t"
        "mov.b64 l2, {f3, f3};\n\t"
        "add.rm.ftz.f32x2 l7, l1, l2;\n\t"
        "sub.rn.ftz.f32x2 l8, l7, l2;\n\t"
        "sub.rn.ftz.f32x2 l9, l1, l8;\n\t"
        "mov.f32 f7, 0f3D9DF09D;\n\t"
        "mov.b64 l6, {f7, f7};\n\t"
        "mov.f32 f6, 0f3E6906A4;\n\t"
        "mov.b64 l5, {f6, f6};\n\t"
        "mov.f32 f5, 0f3F31F519;\n\t"
        "mov.b64 l4, {f5, f5};\n\t"
        "mov.f32 f4, 0f3F800000;\n\t"
        "mov.b64 l3, {f4, f4};\n\t"
        "fma.rn.ftz.f32x2 l10, l9, l6, l5;\n\t"
        "fma.rn.ftz.f32x2 l10, l10, l9, l4;\n\t"
        "fma.rn.ftz.f32x2 l10, l10, l9, l3;\n\t"
        "mov.b64 {r1, r2}, l7;\n\t"
        "mov.b64 {r3, r4}, l10;\n\t"
        "shl.b32 r5, r1, 23;\n\t"
        "add.s32 r7, r5, r3;\n\t"
        "and.b32 r7, r7, 0x7FFFFFFF;\n\t"  // <---- Force sign bit to 0 to safeguard addition carry
        "shl.b32 r6, r2, 23;\n\t"
        "add.s32 r8, r6, r4;\n\t"
        "and.b32 r8, r8, 0x7FFFFFFF;\n\t"  // <---- Force sign bit to 0 to safeguard addition carry
        "mov.b32 %0, r7;\n\t"
        "mov.b32 %1, r8;\n\t"
        "}\n" : "=f"(ox), "=f"(oy) : "f"(x), "f"(y));
    return make_float2(ox, oy);
}

// ----------------------------------------------------------------------------
// Core Blackwell MHA Kernel
// ----------------------------------------------------------------------------
__global__ __launch_bounds__(128) void mha_fwd_sm100_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    float* lse_ptr,
    int seq_len
) {
    setmaxnreg_inc_sync_fn<248>();
    
    int n_q = blockIdx.x;
    int bh = blockIdx.z * gridDim.y + blockIdx.y;
    int tid = threadIdx.x;
    
    extern __shared__ __align__(128) char smem[];
    char* smem_Q0 = smem;
    char* smem_Q1 = smem + 16384;
    char* smem_K0 = smem + 32768;
    char* smem_K1 = smem + 49152;
    char* smem_V0 = smem + 65536;
    char* smem_V1 = smem + 81920;
    char* smem_S  = smem + 98304;
    
    uint64_t* mbar = (uint64_t*)(smem_S + 32768);
    uint64_t* mbar_umma = mbar + 1;
    
    if (tid == 0) {
        init_smem_barrier_fn(mbar, 1);
        init_smem_barrier_fn(mbar_umma, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    uint32_t phase = 0;
    uint32_t phase_umma = 0;
    uint64_t mbar_umma_ptr = (uint64_t)mbar_umma;
    
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar, 6 * 16384);
        tma_load_3d_fn(&tma_Q, mbar, smem_Q0, 0, n_q * 128, bh);
        tma_load_3d_fn(&tma_Q, mbar, smem_Q1, 64, n_q * 128, bh);
        tma_load_3d_fn(&tma_K, mbar, smem_K0, 0, 0, bh);
        tma_load_3d_fn(&tma_K, mbar, smem_K1, 64, 0, bh);
        tma_load_3d_fn(&tma_V, mbar, smem_V0, 0, 0, bh);
        tma_load_3d_fn(&tma_V, mbar, smem_V1, 64, 0, bh);
    }
    
    __shared__ uint32_t tmem_S, tmem_SV;
    if (tid < 32) {
        tmem_alloc_fn(&tmem_S, 128);
        tmem_alloc_fn(&tmem_SV, 64);
    }
    __syncthreads();
    
    float O0_reg[64] = {0};
    float O1_reg[64] = {0};
    float m_prev = -50000.0f;
    float l_prev = 0.0f;
    
    const float scale_S = 1.44269504089f / 11.313708499f; // log2(e) / sqrt(128)
    
    uint64_t desc_Q0 = make_smem_desc_sm100_fn(smem_Q0, 1024);
    uint64_t desc_Q1 = make_smem_desc_sm100_fn(smem_Q1, 1024);
    uint32_t idesc_QK = make_instr_desc_fn(128, 128, false, false);
    
    uint64_t desc_S_base = make_smem_desc_sm100_fn(smem_S, 2048);
    uint32_t idesc_SV = make_instr_desc_fn(128, 64, false, true);
    
    mbarrier_wait_fn(mbar, phase);
    phase ^= 1;
    
    int num_blocks = (seq_len + 127) / 128;
    for (int n = 0; n < num_blocks; ++n) {
        if (n > 0) {
            if (tid == 0) {
                mbarrier_arrive_and_expect_tx_fn(mbar, 4 * 16384);
                tma_load_3d_fn(&tma_K, mbar, smem_K0, 0, n * 128, bh);
                tma_load_3d_fn(&tma_K, mbar, smem_K1, 64, n * 128, bh);
                tma_load_3d_fn(&tma_V, mbar, smem_V0, 0, n * 128, bh);
                tma_load_3d_fn(&tma_V, mbar, smem_V1, 64, n * 128, bh);
            }
            mbarrier_wait_fn(mbar, phase);
            phase ^= 1;
        }
        
        uint64_t desc_K0 = make_smem_desc_sm100_fn(smem_K0, 1024);
        uint64_t desc_K1 = make_smem_desc_sm100_fn(smem_K1, 1024);
        
        if (tid == 0) {
            uint64_t da = desc_Q0;
            uint64_t db = desc_K0;
            for (int k = 0; k < 4; ++k) {
                umma_f16_cg1_fn(tmem_S, da, db, idesc_QK, k == 0 ? 0 : 1);
                da = advance_smem_desc_k(da, 32);
                db = advance_smem_desc_k(db, 32);
            }
            
            da = desc_Q1;
            db = desc_K1;
            for (int k = 0; k < 4; ++k) {
                umma_f16_cg1_fn(tmem_S, da, db, idesc_QK, 1);
                da = advance_smem_desc_k(da, 32);
                db = advance_smem_desc_k(db, 32);
            }
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "l"(mbar_umma_ptr) : "memory");
        }
        mbarrier_wait_fn(mbar_umma, phase_umma);
        phase_umma ^= 1;
        
        float m_curr = -50000.0f;
        for (int c = 0; c < 128; c += 64) {
            uint32_t r[64];
            for (int i = 0; i < 64; i += 8) {
                tmem_load_8x_fn(tmem_S + c + i, &r[i], &r[i+1], &r[i+2], &r[i+3], &r[i+4], &r[i+5], &r[i+6], &r[i+7]);
            }
            tmem_load_fence_fn();
            for (int i = 0; i < 64; ++i) {
                float val = __uint_as_float(r[i]) * scale_S;
                if (n * 128 + c + i >= seq_len) val = -50000.0f;
                m_curr = fmaxf(m_curr, val);
            }
        }
        
        float m_j = fmaxf(m_prev, m_curr);
        float scale = fast_exp2f_fn(m_prev - m_j);
        for (int i = 0; i < 64; ++i) {
            O0_reg[i] *= scale;
            O1_reg[i] *= scale;
        }
        float l_j = l_prev * scale;
        
        for (int c = 0; c < 128; c += 64) {
            uint32_t r[64];
            for (int i = 0; i < 64; i += 8) {
                tmem_load_8x_fn(tmem_S + c + i, &r[i], &r[i+1], &r[i+2], &r[i+3], &r[i+4], &r[i+5], &r[i+6], &r[i+7]);
            }
            tmem_load_fence_fn();
            for (int i = 0; i < 64; i += 2) {
                float v0 = __uint_as_float(r[i]) * scale_S;
                float v1 = __uint_as_float(r[i+1]) * scale_S;
                if (n * 128 + c + i >= seq_len) v0 = -50000.0f;
                if (n * 128 + c + i + 1 >= seq_len) v1 = -50000.0f;
                float2 val = make_float2(v0 - m_j, v1 - m_j);
                
                val = ex2_emulation_packed_asm_fn(val.x, val.y);
                l_j += val.x + val.y;
                
                uint32_t pack = pack_bf16_fn(__float_as_uint(val.x), __float_as_uint(val.y));
                uint32_t col_bytes = (c + i) * 2;
                uint32_t offset = tid * 256 + swizzle_128B(tid, col_bytes);
                *(uint32_t*)((char*)smem_S + offset) = pack;
            }
        }
        
        tcgen05_fence_before_fn();
        __syncthreads();
        tcgen05_fence_after_fn();
        fence_async_shared_fn();
        
        uint64_t desc_V0 = make_smem_desc_sm100_mn_major(smem_V0, 16384, 1024);
        if (tid == 0) {
            uint64_t da = desc_S_base;
            uint64_t db = desc_V0;
            for (int k = 0; k < 8; ++k) {
                umma_f16_cg1_fn(tmem_SV, da, db, idesc_SV, k == 0 ? 0 : 1);
                da = advance_smem_desc_k(da, 32);
                db = advance_smem_desc_k(db, 2048);
            }
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "l"(mbar_umma_ptr) : "memory");
        }
        mbarrier_wait_fn(mbar_umma, phase_umma);
        phase_umma ^= 1;
        
        for (int c = 0; c < 64; c += 8) {
            uint32_t r[8];
            tmem_load_8x_fn(tmem_SV + c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
            tmem_load_fence_fn();
            for (int i = 0; i < 8; ++i) {
                O0_reg[c + i] += __uint_as_float(r[i]);
            }
        }
        
        tcgen05_fence_before_fn();
        __syncthreads();
        tcgen05_fence_after_fn();
        
        uint64_t desc_V1 = make_smem_desc_sm100_mn_major(smem_V1, 16384, 1024);
        if (tid == 0) {
            uint64_t da = desc_S_base;
            uint64_t db = desc_V1;
            for (int k = 0; k < 8; ++k) {
                umma_f16_cg1_fn(tmem_SV, da, db, idesc_SV, k == 0 ? 0 : 1);
                da = advance_smem_desc_k(da, 32);
                db = advance_smem_desc_k(db, 2048);
            }
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "l"(mbar_umma_ptr) : "memory");
        }
        mbarrier_wait_fn(mbar_umma, phase_umma);
        phase_umma ^= 1;
        
        for (int c = 0; c < 64; c += 8) {
            uint32_t r[8];
            tmem_load_8x_fn(tmem_SV + c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
            tmem_load_fence_fn();
            for (int i = 0; i < 8; ++i) {
                O1_reg[c + i] += __uint_as_float(r[i]);
            }
        }
        
        tcgen05_fence_before_fn();
        __syncthreads();
        tcgen05_fence_after_fn();
        
        m_prev = m_j;
        l_prev = l_j;
    }
    
    char* smem_O0 = smem_K0;
    char* smem_O1 = smem_K1;
    for (int c = 0; c < 64; c += 2) {
        float2 val0 = make_float2(O0_reg[c] / l_prev, O0_reg[c+1] / l_prev);
        uint32_t pack0 = pack_bf16_fn(__float_as_uint(val0.x), __float_as_uint(val0.y));
        uint32_t offset0 = tid * 128 + swizzle_128B(tid, c * 2);
        *(uint32_t*)(smem_O0 + offset0) = pack0;
        
        float2 val1 = make_float2(O1_reg[c] / l_prev, O1_reg[c+1] / l_prev);
        uint32_t pack1 = pack_bf16_fn(__float_as_uint(val1.x), __float_as_uint(val1.y));
        uint32_t offset1 = tid * 128 + swizzle_128B(tid, c * 2);
        *(uint32_t*)(smem_O1 + offset1) = pack1;
    }
    
    __syncthreads();
    fence_async_shared_fn();
    
    if (tid == 0) {
        tma_store_3d_fn(&tma_O, smem_O0, 0, n_q * 128, bh);
        tma_store_3d_fn(&tma_O, smem_O1, 64, n_q * 128, bh);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }
    
    if (tid < 128) {
        int s_idx = n_q * 128 + tid;
        if (s_idx < seq_len) {
            float lse_val = m_prev * 0.6931471805599453f + logf(l_prev);
            lse_ptr[bh * seq_len + s_idx] = lse_val;
        }
    }
    
    if (tid < 32) {
        tmem_dealloc_fn(tmem_S, 128);
        tmem_dealloc_fn(tmem_SV, 64);
    }
}

namespace mha_h48_d128 {

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t dim0, uint64_t dim1, uint64_t dim2, uint32_t box0, uint32_t box1, uint32_t box2, CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[3] = {dim0, dim1, dim2};
    cuuint64_t globalStrides[2] = {dim0 * 2, dim0 * dim1 * 2};
    cuuint32_t boxDim[3] = {box0, box1, box2};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, globalAddress, globalDim, globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t seq_len = Q.size(2);
    int64_t D = Q.size(3);
    
    void* q_ptr = Q.data_ptr();
    void* k_ptr = K.data_ptr();
    void* v_ptr = V.data_ptr();
    void* o_ptr = O.data_ptr();
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O;
    CUresult res;
    res = create_tma_3d_descriptor_2B(&tma_Q, q_ptr, D, seq_len, B * H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "Failed TMA Q\n"); exit(1); }
    res = create_tma_3d_descriptor_2B(&tma_K, k_ptr, D, seq_len, B * H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "Failed TMA K\n"); exit(1); }
    res = create_tma_3d_descriptor_2B(&tma_V, v_ptr, D, seq_len, B * H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "Failed TMA V\n"); exit(1); }
    res = create_tma_3d_descriptor_2B(&tma_O, o_ptr, D, seq_len, B * H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "Failed TMA O\n"); exit(1); }
    
    dim3 grid((seq_len + 127) / 128, H, B);
    dim3 block(128);
    int smem_bytes = 131088;
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUDA_CHECK(cudaFuncSetAttribute(mha_fwd_sm100_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    
    mha_fwd_sm100_kernel<<<grid, block, smem_bytes, stream>>>(tma_Q, tma_K, tma_V, tma_O, lse_ptr, seq_len);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace mha_h48_d128

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_h48_d128::run);