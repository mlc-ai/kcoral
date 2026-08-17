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

namespace tvm_ffi_mha {

// -------------------------------------------------------------------------
// Device Helper Functions
// -------------------------------------------------------------------------

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
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

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_store_2d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_load_8x_fn(uint32_t tmem_addr,
    uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
    uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
        : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),
          "=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(tmem_addr));
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

__device__ __forceinline__ void umma_f16_cg1_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void tcgen05_commit_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared.b64 [%0];"
        :: "r"(a));
}

__device__ __forceinline__ uint64_t make_smem_desc_k_major(void* smem_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    constexpr uint32_t SBO = 1024;
    constexpr uint32_t LBO = 1;
    d |= (uint64_t)((LBO & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((SBO & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_n_major(void* smem_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    constexpr uint32_t AT_N = 64;
    constexpr uint32_t AT_K = 128;
    constexpr uint32_t SBO = AT_N * 8; // 512
    constexpr uint32_t LBO = (AT_K / 8) * SBO; // 8192
    d |= (uint64_t)((LBO & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((SBO & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, bool transpose_b = false) {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= (0u << 15);
    d |= ((transpose_b) ? (1u << 16) : (0u << 16));
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ uint32_t pack_fp32_to_bf16(float fa, float fb) {
    __nv_bfloat16 a = __float2bfloat16(fa);
    __nv_bfloat16 b = __float2bfloat16(fb);
    uint16_t a_bits = *(reinterpret_cast<uint16_t*>(&a));
    uint16_t b_bits = *(reinterpret_cast<uint16_t*>(&b));
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(a_bits), "h"(b_bits));
    return result;
}

__device__ __forceinline__ void st_shared_128_fn(uint32_t addr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
}

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, const void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        const_cast<void*>(globalAddress),
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

// -------------------------------------------------------------------------
// Main Kernel
// -------------------------------------------------------------------------

__global__ __launch_bounds__(128, 1) void mha_kernel(
    const __grid_constant__ CUtensorMap tma_Q0,
    const __grid_constant__ CUtensorMap tma_Q1,
    const __grid_constant__ CUtensorMap tma_K0,
    const __grid_constant__ CUtensorMap tma_K1,
    const __grid_constant__ CUtensorMap tma_V0,
    const __grid_constant__ CUtensorMap tma_V1,
    const __grid_constant__ CUtensorMap tma_P,
    const __grid_constant__ CUtensorMap tma_O,
    float* LSE,
    int S, int B_H)
{
    int q_tile = blockIdx.x;
    int bh = blockIdx.z;
    int tid = threadIdx.x;
    int q_base = q_tile * 128;
    
    extern __shared__ __align__(1024) char smem[];
    
    __nv_bfloat16* smem_Q0 = (__nv_bfloat16*)smem;                
    __nv_bfloat16* smem_Q1 = (__nv_bfloat16*)(smem + 16384);      
    __nv_bfloat16* smem_K0 = (__nv_bfloat16*)(smem + 32768);      
    __nv_bfloat16* smem_K1 = (__nv_bfloat16*)(smem + 49152);      
    __nv_bfloat16* smem_V0 = (__nv_bfloat16*)(smem + 65536);      
    __nv_bfloat16* smem_V1 = (__nv_bfloat16*)(smem + 81920);      
    __nv_bfloat16* smem_P  = (__nv_bfloat16*)(smem + 98304);      
    __nv_bfloat16* smem_O  = (__nv_bfloat16*)(smem + 114688);     
    
    uint64_t* bar_Q   = (uint64_t*)(smem + 115712);               
    uint64_t* bar_KV0 = (uint64_t*)(smem + 115720);               
    uint64_t* bar_KV1 = (uint64_t*)(smem + 115728);               
    uint64_t* bar_QK  = (uint64_t*)(smem + 115736);               
    uint64_t* bar_PV  = (uint64_t*)(smem + 115744);               
    
    float* smem_m     = (float*)(smem + 115752);                   
    float* smem_l     = (float*)(smem + 115768);                   
    float* smem_m_prev = (float*)(smem + 115784);                 
    float* smem_l_prev = (float*)(smem + 115800);                 
    
    if (tid == 0) {
        init_smem_barrier_fn(bar_Q, 1);
        init_smem_barrier_fn(bar_KV0, 1);
        init_smem_barrier_fn(bar_KV1, 1);
        init_smem_barrier_fn(bar_QK, 1);
        init_smem_barrier_fn(bar_PV, 1);
        
        mbarrier_arrive_and_expect_tx_fn(bar_Q, 32768);
        tma_load_2d_fn(&tma_Q0, bar_Q, smem_Q0, 0, bh * S + q_base);
        tma_load_2d_fn(&tma_Q1, bar_Q, smem_Q1, 64, bh * S + q_base);
    }
    
    uint32_t tmem_S;
    if (tid == 0) {
        tmem_alloc_fn(&tmem_S, 192);
    }
    __syncthreads();
    
    uint32_t tmem_P = tmem_S + 128;
    uint32_t tmem_O = tmem_S;
    uint32_t tmem_P1 = tmem_S + 192;
    
    if (tid < 4) {
        smem_m[tid] = -INFINITY;
        smem_l[tid] = 0.0f;
        smem_m_prev[tid] = -INFINITY;
        smem_l_prev[tid] = 0.0f;
    }
    __sync_warp();
    
    mbarrier_wait_fn(bar_Q, 0);
    fence_proxy_async_fn();
    
    uint64_t desc_Q0 = make_smem_desc_k_major(smem_Q0);
    uint64_t desc_Q1 = make_smem_desc_k_major(smem_Q1);
    uint64_t desc_K0 = make_smem_desc_k_major(smem_K0);
    uint64_t desc_K1 = make_smem_desc_k_major(smem_K1);
    uint64_t desc_V0 = make_smem_desc_n_major(smem_V0);
    uint64_t desc_V1 = make_smem_desc_n_major(smem_V1);
    uint64_t desc_P  = make_smem_desc_k_major(smem_P);
    
    uint32_t idesc_QK = make_instr_desc_fn(128, 128, false);
    uint32_t idesc_PV = make_instr_desc_fn(128, 128, true);
    
    float global_max = -INFINITY;
    float global_sum = 0.0f;
    float scale = 1.0f / __sqrtf(128.0f);
    
    int num_steps = (S + 127) / 128;
    int phase_kv[2] = {0, 0};
    int phase_qk = 0;
    int phase_pv = 0;
    int phase_p = 0;
    
    if (num_steps > 0) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_KV0, 65536);
            tma_load_2d_fn(&tma_K0, bar_KV0, smem_K0, 0, bh * S + 0);
            tma_load_2d_fn(&tma_K1, bar_KV0, smem_K1, 64, bh * S + 0);
            tma_load_2d_fn(&tma_V0, bar_KV0, smem_V0, 0, bh * S + 0);
            tma_load_2d_fn(&tma_V1, bar_KV0, smem_V1, 64, bh * S + 0);
        }
    }
    
    for (int step = 0; step < num_steps; ++step) {
        int buf_idx = step % 2;
        int next_buf_idx = (step + 1) % 2;
        __nv_bfloat16* smem_K_buf = (buf_idx == 0) ? smem_K0 : smem_K1;
        __nv_bfloat16* smem_V_buf = (buf_idx == 0) ? smem_V0 : smem_V1;
        uint64_t* cur_bar = (buf_idx == 0) ? bar_KV0 : bar_KV1;
        
        if (step + 1 < num_steps) {
            if (tid == 0) {
                __nv_bfloat16* next_K = (next_buf_idx == 0) ? smem_K0 : smem_K1;
                __nv_bfloat16* next_V = (next_buf_idx == 0) ? smem_V0 : smem_V1;
                uint64_t* next_bar = (next_buf_idx == 0) ? bar_KV0 : bar_KV1;
                
                mbarrier_arrive_and_expect_tx_fn(next_bar, 65536);
                tma_load_2d_fn(&tma_K0, next_bar, next_K, 0, bh * S + (step + 1) * 128);
                tma_load_2d_fn(&tma_K1, next_bar, next_K + 8192, 64, bh * S + (step + 1) * 128);
                tma_load_2d_fn(&tma_V0, next_bar, next_V, 0, bh * S + (step + 1) * 128);
                tma_load_2d_fn(&tma_V1, next_bar, next_V + 8192, 64, bh * S + (step + 1) * 128);
            }
        }
        
        mbarrier_wait_fn(cur_bar, phase_kv[buf_idx]);
        phase_kv[buf_idx] ^= 1;
        fence_proxy_async_fn();
        
        if (tid == 0) {
            for (int k_step = 0; k_step < 4; ++k_step) {
                uint64_t desc_Q0_k = desc_Q0 + (k_step * 2);
                uint64_t desc_K0_k = desc_K_buf + (k_step * 2);
                umma_f16_cg1_fn(tmem_S, desc_Q0_k, desc_K0_k, idesc_QK, k_step == 0 ? 0 : 1);
            }
            for (int k_step = 0; k_step < 4; ++k_step) {
                uint64_t desc_Q1_k = desc_Q1 + (k_step * 2);
                uint64_t desc_K1_k = desc_K_buf + 8192 + (k_step * 2);
                umma_f16_cg1_fn(tmem_S + 64, desc_Q1_k, desc_K1_k, idesc_QK, 1);
            }
            tcgen05_commit_fn(bar_QK);
        }
        mbarrier_wait_fn(bar_QK, phase_qk);
        phase_qk ^= 1;
        
        tmem_load_fence_fn();
        float row_max = -INFINITY;
        float p[128]; 
        
        for (int col = 0; col < 128; col += 8) {
            uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
            tmem_load_8x_fn(tmem_S + col, &r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
            
            float f0 = __uint_as_float(r0) * scale;
            float f1 = __uint_as_float(r1) * scale;
            float f2 = __uint_as_float(r2) * scale;
            float f3 = __uint_as_float(r3) * scale;
            float f4 = __uint_as_float(r4) * scale;
            float f5 = __uint_as_float(r5) * scale;
            float f6 = __uint_as_float(r6) * scale;
            float f7 = __uint_as_float(r7) * scale;
            
            if (step * 128 + col + 0 >= S) f0 = -INFINITY;
            if (step * 128 + col + 1 >= S) f1 = -INFINITY;
            if (step * 128 + col + 2 >= S) f2 = -INFINITY;
            if (step * 128 + col + 3 >= S) f3 = -INFINITY;
            if (step * 128 + col + 4 >= S) f4 = -INFINITY;
            if (step * 128 + col + 5 >= S) f5 = -INFINITY;
            if (step * 128 + col + 6 >= S) f6 = -INFINITY;
            if (step * 128 + col + 7 >= S) f7 = -INFINITY;
            
            row_max = max(row_max, max(f0, max(f1, max(f2, max(f3, max(f4, max(f5, max(f6, f7))))))));
            
            p[col+0] = f0; p[col+1] = f1; p[col+2] = f2; p[col+3] = f3;
            p[col+4] = f4; p[col+5] = f5; p[col+6] = f6; p[col+7] = f7;
        }
        
        float m = row_max;
        m = max(m, __shfl_xor_sync(0xFFFFFFFF, m, 1));
        m = max(m, __shfl_xor_sync(0xFFFFFFFF, m, 2));
        m = max(m, __shfl_xor_sync(0xFFFFFFFF, m, 4));
        m = max(m, __shfl_xor_sync(0xFFFFFFFF, m, 8));
        
        if (tid % 32 == 0) smem_m[tid / 32] = m;
        __sync_warp();
        
        float prev_max = smem_m_prev[tid / 32];
        float new_max = max(prev_max, smem_m[tid / 32]);
        float factor = (new_max > prev_max) ? __expf(prev_max - new_max) : 1.0f;
        
        if (new_max > prev_max) {
            if (tid % 32 == 0) {
                smem_l_prev[tid / 32] *= factor;
            }
        }
        
        float row_sum = 0.0f;
        for(int i=0; i<128; ++i) {
            float val = p[i];
            float p_val = (val == -INFINITY) ? 0.0f : __expf(val - smem_m[tid / 32]);
            p[i] = p_val;
            row_sum += p_val;
        }
        
        float sum = row_sum;
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, 1);
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, 2);
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, 4);
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, 8);
        
        if (tid % 32 == 0) {
            smem_l[tid / 32] = smem_l_prev[tid / 32] + sum * __expf(smem_m[tid / 32] - new_max);
            smem_m_prev[tid / 32] = new_max;
        }
        __sync_warp();
        
        float f_new_max = new_max; // Capture before modification
        
        for(int i=0; i<128; ++i) {
            p[i] *= __expf(smem_m[tid / 32] - f_new_max);
        }
        
        for (int col = 0; col < 128; col += 8) {
            uint32_t p0 = pack_fp32_to_bf16(p[col], p[col+1]);
            uint32_t p1 = pack_fp32_to_bf16(p[col+2], p[col+3]);
            uint32_t p2 = pack_fp32_to_bf16(p[col+4], p[col+5]);
            uint32_t p3 = pack_fp32_to_bf16(p[col+6], p[col+7]);
            
            int phys_col = ((tid % 8) ^ (col / 8)) * 8 + (col % 8);
            uint32_t dst = (uint32_t)__cvta_generic_to_shared(&smem_P[tid * 128 + phys_col]);
            st_shared_128_fn(dst, p0, p1, p2, p3);
        }
        
        __syncthreads();
        fence_proxy_async_fn(); 
        
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_P, 32768); 
            tma_load_2d_fn(&tma_P, bar_P, smem_P, 0, bh * S + q_base);
            tma_load_2d_fn(&tma_P, bar_P, smem_P + 8192, 64, bh * S + q_base);
        }
        mbarrier_wait_fn(bar_P, phase_p);
        phase_p ^= 1;
        fence_proxy_async_fn();
        
        if (tid == 0) {
            for (int k_step = 0; k_step < 4; ++k_step) {
                uint64_t desc_P_k = desc_P + (k_step * 2);
                uint64_t desc_V0_k = desc_V_buf + (k_step * 32);
                umma_f16_cg1_fn(tmem_O, desc_P_k, desc_V0_k, idesc_PV, k_step == 0 ? 0 : 1);
            }
            for (int k_step = 0; k_step < 4; ++k_step) {
                uint64_t desc_P_k = desc_P + 8192 + (k_step * 2);
                uint64_t desc_V1_k = desc_V_buf + 8192 + (k_step * 32);
                umma_f16_cg1_fn(tmem_O + 64, desc_P_k, desc_V1_k, idesc_PV, 1);
            }
            tcgen05_commit_fn(bar_PV);
        }
        mbarrier_wait_fn(bar_PV, phase_pv);
        phase_pv ^= 1;
        
        global_max = new_max;
        global_sum = smem_l[tid / 32];
    }
    
    tmem_load_fence_fn();
    float out_val0_x = -INFINITY, out_val0_y = -INFINITY;
    float out_val1_x = -INFINITY, out_val1_y = -INFINITY;
    float out_val2_x = -INFINITY, out_val2_y = -INFINITY;
    float out_val3_x = -INFINITY, out_val3_y = -INFINITY;
    
    if (global_sum > 0.0f) {
        for (int col = 0; col < 128; col += 8) {
            uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
            tmem_load_8x_fn(tmem_O + col, &r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
            out_val0_x = __uint_as_float(r0) / global_sum;
            out_val0_y = __uint_as_float(r1) / global_sum;
            out_val1_x = __uint_as_float(r2) / global_sum;
            out_val1_y = __uint_as_float(r3) / global_sum;
            out_val2_x = __uint_as_float(r4) / global_sum;
            out_val2_y = __uint_as_float(r5) / global_sum;
            out_val3_x = __uint_as_float(r6) / global_sum;
            out_val3_y = __uint_as_float(r7) / global_sum;
            
            int phys_col = ((tid % 8) ^ (col / 8)) * 8 + (col % 8);
            uint32_t o_packed0 = pack_fp32_to_bf16(out_val0_x, out_val0_y);
            uint32_t o_packed1 = pack_fp32_to_bf16(out_val1_x, out_val1_y);
            uint32_t o_packed2 = pack_fp32_to_bf16(out_val2_x, out_val2_y);
            uint32_t o_packed3 = pack_fp32_to_bf16(out_val3_x, out_val3_y);
            uint32_t dst = (uint32_t)__cvta_generic_to_shared(&smem_O[tid * 128 + phys_col]);
            st_shared_128_fn(dst, o_packed0, o_packed1, o_packed2, o_packed3);
        }
    } else {
        for (int col = 0; col < 128; col += 8) {
            int phys_col = ((tid % 8) ^ (col / 8)) * 8 + (col % 8);
            uint32_t dst = (uint32_t)__cvta_generic_to_shared(&smem_O[tid * 128 + phys_col]);
            st_shared_128_fn(dst, 0, 0, 0, 0);
        }
    }
    
    __syncthreads();
    fence_proxy_async_fn(); 
    
    if (tid == 0) {
        tma_store_2d_fn(&tma_O, smem_O, 0, bh * S + q_base);
        tma_store_2d_fn(&tma_O, smem_O + 8192, 64, bh * S + q_base);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }
    
    if (tid == 0) {
        if (q_base < S) {
            LSE[bh * S + q_base] = global_max + __logf(global_sum);
        }
    }
    
    if (tid == 0) {
        tmem_dealloc_fn(tmem_S, 192);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, 
         tvm::ffi::TensorView V, tvm::ffi::TensorView O, 
         tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    if (S == 0) return;
    
    int B_H = B * H;
    int smem_size = 115808; 
    
    CUtensorMap tma_Q0, tma_Q1, tma_K0, tma_K1, tma_V0, tma_V1, tma_P, tma_O;
    
    CUresult res;
    res = create_tma_2d_descriptor_2B(&tma_Q0, static_cast<const void*>(Q.data_ptr()), 
        D, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA Q failed\n"); exit(1); }
    
    res = create_tma_2d_descriptor_2B(&tma_Q1, static_cast<const void*>(Q.data_ptr()), 
        D, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA Q failed\n"); exit(1); }
    
    res = create_tma_2d_descriptor_2B(&tma_K0, static_cast<const void*>(K.data_ptr()), 
        D, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA K failed\n"); exit(1); }
    
    res = create_tma_2d_descriptor_2B(&tma_K1, static_cast<const void*>(K.data_ptr()), 
        D, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA K failed\n"); exit(1); }
    
    res = create_tma_2d_descriptor_2B(&tma_V0, static_cast<const void*>(V.data_ptr()), 
        D, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA V failed\n"); exit(1); }
    
    res = create_tma_2d_descriptor_2B(&tma_V1, static_cast<const void*>(V.data_ptr()), 
        D, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA V failed\n"); exit(1); }

    res = create_tma_2d_descriptor_2B(&tma_P, smem_P, 128, 128, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA P failed\n"); exit(1); }
    
    res = create_tma_2d_descriptor_2B(&tma_O, static_cast<const void*>(O.data_ptr()), 
        D, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA O failed\n"); exit(1); }
    
    CUDA_CHECK(cudaFuncSetAttribute(
        mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    dim3 grid((S + 127) / 128, 1, B * H);
    dim3 block(128);
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 1;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, mha_kernel,
        tma_Q0, tma_Q1, tma_K0, tma_K1, tma_V0, tma_V1, tma_P, tma_O,
        static_cast<float*>(LSE.data_ptr()),
        S, B_H));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha::run);

}  // namespace tvm_ffi_mha