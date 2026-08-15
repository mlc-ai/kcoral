#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
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

namespace tvm_ffi_mha {

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void umma_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "r"(a)); 
}

__device__ __forceinline__ void umma_f16_cg1_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   // version = 1 (SM100)
    d |= (uint64_t)2 << 61;   // layout_type = 2 (SWIZZLE_128B)
    return d;
}

__device__ __forceinline__ uint32_t get_swizzled_smem_addr(uint32_t base_addr, uint32_t row, uint32_t col_elements) {
    uint32_t chunk_idx = col_elements / 8;
    uint32_t chunk_swizzled = (row % 8) ^ chunk_idx;
    uint32_t offset = row * 128 + chunk_swizzled * 16;
    return base_addr + offset;
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

__global__ void mha_fwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O_out,
    float* LSE_out,
    int S_seq,
    int H) 
{
    extern __shared__ __align__(1024) char smem[];
    __nv_bfloat16* Q_smem_0 = (__nv_bfloat16*)smem;
    __nv_bfloat16* Q_smem_1 = Q_smem_0 + 128 * 64;
    
    __nv_bfloat16* K_smem_0_0 = Q_smem_1 + 128 * 64;
    __nv_bfloat16* K_smem_1_0 = K_smem_0_0 + 128 * 64;
    __nv_bfloat16* K_smem_0_1 = K_smem_1_0 + 128 * 64;
    __nv_bfloat16* K_smem_1_1 = K_smem_0_1 + 128 * 64;
    
    __nv_bfloat16* V_smem_0_0 = K_smem_1_1 + 128 * 64;
    __nv_bfloat16* V_smem_1_0 = V_smem_0_0 + 128 * 64;
    __nv_bfloat16* V_smem_0_1 = V_smem_1_0 + 128 * 64;
    __nv_bfloat16* V_smem_1_1 = V_smem_0_1 + 128 * 64;
    
    __nv_bfloat16* P_smem_0 = V_smem_1_1 + 128 * 64;
    __nv_bfloat16* P_smem_1 = P_smem_0 + 128 * 64;
    
    uint64_t* mbar_Q = (uint64_t*)(P_smem_1 + 128 * 64);
    uint64_t* mbar_K_0 = mbar_Q + 1;
    uint64_t* mbar_K_1 = mbar_K_0 + 1;
    uint64_t* mbar_V_0 = mbar_K_1 + 1;
    uint64_t* mbar_V_1 = mbar_V_0 + 1;
    uint64_t* mbar_UMMA = mbar_V_1 + 1;
    uint32_t* tmem_base_ptr = (uint32_t*)(mbar_UMMA + 1);

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_K_0, 1);
        init_smem_barrier_fn(mbar_K_1, 1);
        init_smem_barrier_fn(mbar_V_0, 1);
        init_smem_barrier_fn(mbar_V_1, 1);
        init_smem_barrier_fn(mbar_UMMA, 1);
    }
    if (threadIdx.x < 32) {
        tmem_alloc_cg1_fn(tmem_base_ptr, 256);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t tmem_base = *tmem_base_ptr;
    uint32_t S_tmem = tmem_base;
    uint32_t O_tmem = tmem_base + 128;

    for (int i = 0; i < 128; i += 8) {
        uint32_t r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0;
        asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%8], {%0,%1,%2,%3,%4,%5,%6,%7};"
                     :: "r"(r0),"r"(r1),"r"(r2),"r"(r3),"r"(r4),"r"(r5),"r"(r6),"r"(r7), "r"(O_tmem + i) : "memory");
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");

    int batch = blockIdx.z;
    int head = blockIdx.y;
    int m_start = blockIdx.x * 128;
    int coord1_base = batch * H * S_seq + head * S_seq;

    uint32_t phase_Q = 0;
    uint32_t phase_K[2] = {0, 0};
    uint32_t phase_V[2] = {0, 0};
    uint32_t phase_UMMA = 0;
    int k_idx = 0;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 128*64*2 * 2);
        tma_load_2d_fn(&tma_Q, mbar_Q, Q_smem_0, 0, coord1_base + m_start);
        tma_load_2d_fn(&tma_Q, mbar_Q, Q_smem_1, 64, coord1_base + m_start);
        
        mbarrier_arrive_and_expect_tx_fn(mbar_K_0, 128*64*2 * 2);
        tma_load_2d_fn(&tma_K, mbar_K_0, K_smem_0_0, 0, coord1_base + 0);
        tma_load_2d_fn(&tma_K, mbar_K_0, K_smem_0_1, 64, coord1_base + 0);
        
        mbarrier_arrive_and_expect_tx_fn(mbar_V_0, 128*64*2 * 2);
        tma_load_2d_fn(&tma_V, mbar_V_0, V_smem_0_0, 0, coord1_base + 0);
        tma_load_2d_fn(&tma_V, mbar_V_0, V_smem_0_1, 64, coord1_base + 0);
    }
    mbarrier_wait_fn(mbar_Q, phase_Q);
    phase_Q ^= 1;

    float m_val = -INFINITY;
    float l_val = 0.0f;
    bool is_valid_q = (m_start + threadIdx.x < S_seq);

    uint32_t idesc_QK = 0;
    idesc_QK |= (1u << 4);
    idesc_QK |= (1u << 7);
    idesc_QK |= (1u << 10);
    idesc_QK |= (0u << 15);
    idesc_QK |= (0u << 16);
    idesc_QK |= ((128 / 8) << 17);
    idesc_QK |= ((128 / 16) << 24);

    uint32_t idesc_PV = 0;
    idesc_PV |= (1u << 4);
    idesc_PV |= (1u << 7);
    idesc_PV |= (1u << 10);
    idesc_PV |= (0u << 15);
    idesc_PV |= (1u << 16);
    idesc_PV |= ((64 / 8) << 17);
    idesc_PV |= ((128 / 16) << 24);

    for (int n_start = 0; n_start <= m_start; n_start += 128, k_idx ^= 1) {
        uint64_t* cur_mbar_K = (k_idx == 0) ? mbar_K_0 : mbar_K_1;
        uint64_t* cur_mbar_V = (k_idx == 0) ? mbar_V_0 : mbar_V_1;
        
        void* cur_K_0 = (k_idx == 0) ? K_smem_0_0 : K_smem_1_0;
        void* cur_K_1 = (k_idx == 0) ? K_smem_0_1 : K_smem_1_1;
        void* cur_V_0 = (k_idx == 0) ? V_smem_0_0 : V_smem_1_0;
        void* cur_V_1 = (k_idx == 0) ? V_smem_0_1 : V_smem_1_1;
        
        int next_n = n_start + 128;
        if (threadIdx.x == 0 && next_n <= m_start) {
            uint64_t* next_mbar_K = (k_idx == 0) ? mbar_K_1 : mbar_K_0;
            uint64_t* next_mbar_V = (k_idx == 0) ? mbar_V_1 : mbar_V_0;
            void* next_K_0 = (k_idx == 0) ? K_smem_1_0 : K_smem_0_0;
            void* next_K_1 = (k_idx == 0) ? K_smem_1_1 : K_smem_0_1;
            void* next_V_0 = (k_idx == 0) ? V_smem_1_0 : V_smem_0_0;
            void* next_V_1 = (k_idx == 0) ? V_smem_1_1 : V_smem_0_1;
            
            mbarrier_arrive_and_expect_tx_fn(next_mbar_K, 128*64*2 * 2);
            tma_load_2d_fn(&tma_K, next_mbar_K, next_K_0, 0, coord1_base + next_n);
            tma_load_2d_fn(&tma_K, next_mbar_K, next_K_1, 64, coord1_base + next_n);
            
            mbarrier_arrive_and_expect_tx_fn(next_mbar_V, 128*64*2 * 2);
            tma_load_2d_fn(&tma_V, next_mbar_V, next_V_0, 0, coord1_base + next_n);
            tma_load_2d_fn(&tma_V, next_mbar_V, next_V_1, 64, coord1_base + next_n);
        }

        mbarrier_wait_fn(cur_mbar_K, phase_K[k_idx]);
        phase_K[k_idx] ^= 1;
        
        if (threadIdx.x == 0) {
            uint32_t addr_Q0 = (uint32_t)__cvta_generic_to_shared(Q_smem_0);
            uint32_t addr_Q1 = (uint32_t)__cvta_generic_to_shared(Q_smem_1);
            uint32_t addr_K0 = (uint32_t)__cvta_generic_to_shared(cur_K_0);
            uint32_t addr_K1 = (uint32_t)__cvta_generic_to_shared(cur_K_1);
            
            for (int k_step = 0; k_step < 8; ++k_step) {
                void* q_ptr = (k_step < 4) ? (void*)(addr_Q0 + k_step * 32) : (void*)(addr_Q1 + (k_step - 4) * 32);
                void* k_ptr = (k_step < 4) ? (void*)(addr_K0 + k_step * 32) : (void*)(addr_K1 + (k_step - 4) * 32);
                uint64_t desc_Q = make_smem_desc_sm100_fn(q_ptr, 1, 1024); 
                uint64_t desc_K = make_smem_desc_sm100_fn(k_ptr, 1, 1024);
                uint32_t accum = (k_step == 0) ? 0 : 1;
                umma_f16_cg1_fn(S_tmem, desc_Q, desc_K, idesc_QK, accum);
            }
            umma_commit_cg1_fn(mbar_UMMA);
        }
        mbarrier_wait_fn(mbar_UMMA, phase_UMMA);
        phase_UMMA ^= 1;
        
        bool is_causal_block = (n_start == m_start);
        int row_valid_len = 128;
        if (!is_valid_q) {
            row_valid_len = 0;
        } else {
            if (is_causal_block) {
                row_valid_len = (m_start + threadIdx.x) - n_start + 1;
            }
            if (n_start + row_valid_len > S_seq) {
                row_valid_len = S_seq - n_start;
            }
            if (row_valid_len < 0) row_valid_len = 0;
        }

        float row_max = -INFINITY;
        for (int i = 0; i < 128; i += 8) {
            uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                         : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(S_tmem + i));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float f0 = __uint_as_float(r0) * 0.08838834764f;
            float f1 = __uint_as_float(r1) * 0.08838834764f;
            float f2 = __uint_as_float(r2) * 0.08838834764f;
            float f3 = __uint_as_float(r3) * 0.08838834764f;
            float f4 = __uint_as_float(r4) * 0.08838834764f;
            float f5 = __uint_as_float(r5) * 0.08838834764f;
            float f6 = __uint_as_float(r6) * 0.08838834764f;
            float f7 = __uint_as_float(r7) * 0.08838834764f;
            
            if (i + 0 >= row_valid_len) f0 = -INFINITY;
            if (i + 1 >= row_valid_len) f1 = -INFINITY;
            if (i + 2 >= row_valid_len) f2 = -INFINITY;
            if (i + 3 >= row_valid_len) f3 = -INFINITY;
            if (i + 4 >= row_valid_len) f4 = -INFINITY;
            if (i + 5 >= row_valid_len) f5 = -INFINITY;
            if (i + 6 >= row_valid_len) f6 = -INFINITY;
            if (i + 7 >= row_valid_len) f7 = -INFINITY;
            
            row_max = fmaxf(row_max, f0);
            row_max = fmaxf(row_max, f1);
            row_max = fmaxf(row_max, f2);
            row_max = fmaxf(row_max, f3);
            row_max = fmaxf(row_max, f4);
            row_max = fmaxf(row_max, f5);
            row_max = fmaxf(row_max, f6);
            row_max = fmaxf(row_max, f7);
        }
        
        float m_new = fmaxf(m_val, row_max);
        float scale = 1.0f;
        if (m_val != -INFINITY) {
            scale = fast_exp2f_fn((m_val - m_new) * 1.44269504089f);
        }
        
        if (__any_sync(0xffffffff, scale < 1.0f)) {
            for (int i = 0; i < 128; i += 8) {
                uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                             : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(O_tmem + i));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                
                float o0 = __uint_as_float(r0) * scale;
                float o1 = __uint_as_float(r1) * scale;
                float o2 = __uint_as_float(r2) * scale;
                float o3 = __uint_as_float(r3) * scale;
                float o4 = __uint_as_float(r4) * scale;
                float o5 = __uint_as_float(r5) * scale;
                float o6 = __uint_as_float(r6) * scale;
                float o7 = __uint_as_float(r7) * scale;
                
                r0 = __float_as_uint(o0);
                r1 = __float_as_uint(o1);
                r2 = __float_as_uint(o2);
                r3 = __float_as_uint(o3);
                r4 = __float_as_uint(o4);
                r5 = __float_as_uint(o5);
                r6 = __float_as_uint(o6);
                r7 = __float_as_uint(o7);
                
                asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%8], {%0,%1,%2,%3,%4,%5,%6,%7};"
                             :: "r"(r0),"r"(r1),"r"(r2),"r"(r3),"r"(r4),"r"(r5),"r"(r6),"r"(r7), "r"(O_tmem + i) : "memory");
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        }
        
        float row_sum = 0.0f;
        uint32_t addr_P0 = (uint32_t)__cvta_generic_to_shared(P_smem_0);
        uint32_t addr_P1 = (uint32_t)__cvta_generic_to_shared(P_smem_1);

        for (int i = 0; i < 128; i += 8) {
            uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                         : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(S_tmem + i));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float f0 = __uint_as_float(r0) * 0.08838834764f;
            float f1 = __uint_as_float(r1) * 0.08838834764f;
            float f2 = __uint_as_float(r2) * 0.08838834764f;
            float f3 = __uint_as_float(r3) * 0.08838834764f;
            float f4 = __uint_as_float(r4) * 0.08838834764f;
            float f5 = __uint_as_float(r5) * 0.08838834764f;
            float f6 = __uint_as_float(r6) * 0.08838834764f;
            float f7 = __uint_as_float(r7) * 0.08838834764f;
            
            f0 = fast_exp2f_fn((f0 - m_new) * 1.44269504089f);
            f1 = fast_exp2f_fn((f1 - m_new) * 1.44269504089f);
            f2 = fast_exp2f_fn((f2 - m_new) * 1.44269504089f);
            f3 = fast_exp2f_fn((f3 - m_new) * 1.44269504089f);
            f4 = fast_exp2f_fn((f4 - m_new) * 1.44269504089f);
            f5 = fast_exp2f_fn((f5 - m_new) * 1.44269504089f);
            f6 = fast_exp2f_fn((f6 - m_new) * 1.44269504089f);
            f7 = fast_exp2f_fn((f7 - m_new) * 1.44269504089f);
            
            if (i + 0 >= row_valid_len) f0 = 0.0f;
            if (i + 1 >= row_valid_len) f1 = 0.0f;
            if (i + 2 >= row_valid_len) f2 = 0.0f;
            if (i + 3 >= row_valid_len) f3 = 0.0f;
            if (i + 4 >= row_valid_len) f4 = 0.0f;
            if (i + 5 >= row_valid_len) f5 = 0.0f;
            if (i + 6 >= row_valid_len) f6 = 0.0f;
            if (i + 7 >= row_valid_len) f7 = 0.0f;
            
            row_sum += f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7;
            
            uint32_t p01 = pack_bf16_fn(__float_as_uint(f0), __float_as_uint(f1));
            uint32_t p23 = pack_bf16_fn(__float_as_uint(f2), __float_as_uint(f3));
            uint32_t p45 = pack_bf16_fn(__float_as_uint(f4), __float_as_uint(f5));
            uint32_t p67 = pack_bf16_fn(__float_as_uint(f6), __float_as_uint(f7));
            
            uint32_t smem_addr;
            if (i < 64) {
                smem_addr = get_swizzled_smem_addr(addr_P0, threadIdx.x, i);
            } else {
                smem_addr = get_swizzled_smem_addr(addr_P1, threadIdx.x, i - 64);
            }
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" 
                         :: "r"(smem_addr), "r"(p01), "r"(p23), "r"(p45), "r"(p67));
        }
        
        if (m_val != -INFINITY) {
            l_val = l_val * scale + row_sum;
        } else {
            l_val = row_sum;
        }
        m_val = m_new;
        
        fence_async_shared_fn();
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        __syncthreads();
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        
        mbarrier_wait_fn(cur_mbar_V, phase_V[k_idx]);
        phase_V[k_idx] ^= 1;
        
        if (threadIdx.x == 0) {
            uint32_t addr_V0 = (uint32_t)__cvta_generic_to_shared(cur_V_0);
            uint32_t addr_V1 = (uint32_t)__cvta_generic_to_shared(cur_V_1);
            
            for (int k_step = 0; k_step < 8; ++k_step) {
                void* p_ptr = (k_step < 4) ? (void*)(addr_P0 + k_step * 32) : (void*)(addr_P1 + (k_step - 4) * 32);
                void* v0_ptr = (void*)(addr_V0 + k_step * 2048);
                void* v1_ptr = (void*)(addr_V1 + k_step * 2048);
                
                uint64_t desc_P = make_smem_desc_sm100_fn(p_ptr, 1, 1024); 
                uint64_t desc_V0 = make_smem_desc_sm100_fn(v0_ptr, 8192, 1024); 
                uint64_t desc_V1 = make_smem_desc_sm100_fn(v1_ptr, 8192, 1024); 
                
                umma_f16_cg1_fn(O_tmem, desc_P, desc_V0, idesc_PV, 1);
                umma_f16_cg1_fn(O_tmem + 64, desc_P, desc_V1, idesc_PV, 1);
            }
            umma_commit_cg1_fn(mbar_UMMA);
        }
        mbarrier_wait_fn(mbar_UMMA, phase_UMMA);
        phase_UMMA ^= 1;
    }
    
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    
    float inv_l = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;
    __nv_bfloat16* O_smem = P_smem_0;
    
    for (uint32_t col = 0; col < 128; col += 8) {
        uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(O_tmem + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float f0 = __uint_as_float(r0) * inv_l;
        float f1 = __uint_as_float(r1) * inv_l;
        float f2 = __uint_as_float(r2) * inv_l;
        float f3 = __uint_as_float(r3) * inv_l;
        float f4 = __uint_as_float(r4) * inv_l;
        float f5 = __uint_as_float(r5) * inv_l;
        float f6 = __uint_as_float(r6) * inv_l;
        float f7 = __uint_as_float(r7) * inv_l;
        
        uint32_t base = threadIdx.x * 128 + col;
        O_smem[base + 0] = __float2bfloat16(f0);
        O_smem[base + 1] = __float2bfloat16(f1);
        O_smem[base + 2] = __float2bfloat16(f2);
        O_smem[base + 3] = __float2bfloat16(f3);
        O_smem[base + 4] = __float2bfloat16(f4);
        O_smem[base + 5] = __float2bfloat16(f5);
        O_smem[base + 6] = __float2bfloat16(f6);
        O_smem[base + 7] = __float2bfloat16(f7);
    }
    
    __syncthreads();
    
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    for (uint32_t step = 0; step < 32; ++step) {
        uint32_t row = step * 4 + warp_id;
        uint32_t global_row = m_start + row;
        uint32_t col_start = lane_id * 4;
        
        if (global_row < S_seq) {
            uint2 data = *reinterpret_cast<uint2*>(&O_smem[row * 128 + col_start]);
            *reinterpret_cast<uint2*>(O_out + coord1_base * 128 + global_row * 128 + col_start) = data;
        }
    }
    
    if (is_valid_q) {
        float lse = (m_val == -INFINITY) ? -INFINITY : (m_val + logf(l_val));
        LSE_out[batch * H * S_seq + head * S_seq + m_start + threadIdx.x] = lse;
    }
    
    if (threadIdx.x < 32) {
        tmem_dealloc_cg1_fn(tmem_base, 256);
    }
}

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        globalAddress,
        globalDim,
        globalStrides,
        boxDim,
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        l2Promotion,
        oobFill
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int B = Q.size(0);
    int H = Q.size(1);
    int S_seq = Q.size(2);
    int D = Q.size(3);
    
    uint64_t total_S = B * H * S_seq;
    
    CUtensorMap tma_Q, tma_K, tma_V;
    create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), D, total_S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), D, total_S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), D, total_S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    int blocks_x = (S_seq + 127) / 128;
    dim3 grid(blocks_x, H, B);
    dim3 block(128);
    
    int smem_size = 196608 + 128; // ~192 KB
    CUDA_CHECK(cudaFuncSetAttribute(mha_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    mha_fwd_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, 
        static_cast<__nv_bfloat16*>(O.data_ptr()), 
        static_cast<float*>(LSE.data_ptr()), 
        S_seq, H
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}  // namespace tvm_ffi_mha