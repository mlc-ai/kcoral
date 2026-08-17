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

__device__ __forceinline__ void st_shared_128_fn(uint32_t addr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
}

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

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
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

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
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
    constexpr uint32_t SBO = AT_N * 8; 
    constexpr uint32_t LBO = (AT_K / 8) * SBO;
    d |= (uint64_t)((LBO & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((SBO & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= (0u << 15);
    d |= (0u << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ int apply_swizzle_128B(int col, int row) {
    int x = col / 8;
    int seg = x / 8;
    int chunk = x % 8;
    int swizzled_x = seg * 8 + (chunk ^ (row % 8));
    return swizzled_x * 8 + (col % 8);
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

// -------------------------------------------------------------------------
// Main Kernel
// -------------------------------------------------------------------------

__global__ __launch_bounds__(128, 1) void mha_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O, float* LSE,
    int S, int B_H)
{
    int q_base = blockIdx.x;
    int bh = blockIdx.z;
    int tid = threadIdx.x;
    
    extern __shared__ __align__(128) char smem[];
    __nv_bfloat16* smem_Q0 = (__nv_bfloat16*)smem;
    __nv_bfloat16* smem_Q1 = (__nv_bfloat16*)(smem + 16384);
    __nv_bfloat16* smem_K0 = (__nv_bfloat16*)(smem + 32768);
    __nv_bfloat16* smem_K1 = (__nv_bfloat16*)(smem + 49152);
    __nv_bfloat16* smem_V0 = (__nv_bfloat16*)(smem + 65536);
    __nv_bfloat16* smem_V1 = (__nv_bfloat16*)(smem + 81920);
    __nv_bfloat16* smem_P  = (__nv_bfloat16*)(smem + 98304);
    
    uint64_t* bar_Q  = (uint64_t*)(smem + 114688);
    uint64_t* bar_K  = (uint64_t*)(smem + 114696);
    uint64_t* bar_QK = (uint64_t*)(smem + 114704);
    uint64_t* bar_PV = (uint64_t*)(smem + 114712);
    
    if (tid == 0) {
        init_smem_barrier_fn(bar_Q, 1);
        init_smem_barrier_fn(bar_K, 1);
        init_smem_barrier_fn(bar_QK, 1);
        init_smem_barrier_fn(bar_PV, 1);
    }
    __syncthreads();
    
    uint32_t tmem_S;
    if (tid == 0) {
        tmem_alloc_fn(&tmem_S, 192);
    }
    __syncthreads();
    
    uint32_t tmem_P = tmem_S + 128;
    uint32_t tmem_O = tmem_S;
    
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_Q, 32768);
        tma_load_2d_fn(&tma_Q, bar_Q, smem_Q0, 0, bh * S + q_base);
        tma_load_2d_fn(&tma_Q, bar_Q, smem_Q1, 64, bh * S + q_base);
    }
    mbarrier_wait_fn(bar_Q, 0);
    
    uint64_t desc_Q0 = make_smem_desc_k_major(smem_Q0);
    uint64_t desc_Q1 = make_smem_desc_k_major(smem_Q1);
    uint64_t desc_K0 = make_smem_desc_k_major(smem_K0);
    uint64_t desc_K1 = make_smem_desc_k_major(smem_K1);
    uint64_t desc_P = make_smem_desc_k_major(smem_P);
    uint64_t desc_V0 = make_smem_desc_n_major(smem_V0);
    uint64_t desc_V1 = make_smem_desc_n_major(smem_V1);
    
    uint32_t idesc_QK = make_instr_desc_fn(128, 128);
    uint32_t idesc_PV = make_instr_desc_fn(128, 64);
    idesc_PV |= (1u << 16);
    
    float scale = 1.0f / __sqrtf(128.0f);
    float global_max = -INFINITY;
    float global_sum = 0.0f;
    
    int num_steps = (S + 127) / 128;
    int phase_kv = 0, phase_qk = 0, phase_pv = 0;
    
    for (int step = 0; step < num_steps; ++step) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_K, 65536);
            tma_load_2d_fn(&tma_K, bar_K, smem_K0, 0, bh * S + step * 128);
            tma_load_2d_fn(&tma_K, bar_K, smem_K1, 64, bh * S + step * 128);
            tma_load_2d_fn(&tma_V, bar_K, smem_V0, 0, bh * S + step * 128);
            tma_load_2d_fn(&tma_V, bar_K, smem_V1, 64, bh * S + step * 128);
        }
        mbarrier_wait_fn(bar_K, phase_kv);
        phase_kv ^= 1;
        
        fence_proxy_async_fn();
        
        if (tid == 0) {
            for (int k = 0; k < 64; k += 16) {
                uint64_t desc_Q0_k = desc_Q0 + (k * 2);
                uint64_t desc_K0_k = desc_K0 + (k * 2);
                umma_f16_cg1_fn(tmem_S, desc_Q0_k, desc_K0_k, idesc_QK, k == 0 ? 0 : 1);
                
                uint64_t desc_Q1_k = desc_Q1 + (k * 2);
                uint64_t desc_K1_k = desc_K1 + (k * 2);
                umma_f16_cg1_fn(tmem_S, desc_Q1_k, desc_K1_k, idesc_QK, 1);
            }
            tcgen05_commit_fn(bar_QK);
        }
        mbarrier_wait_fn(bar_QK, phase_qk);
        phase_qk ^= 1;
        
        float row_max = -INFINITY;
        tmem_load_fence_fn();
        for (int col = 0; col < 64; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_S + col, &r0, &r1, &r2, &r3);
            float f0 = __uint_as_float(r0) * scale;
            float f1 = __uint_as_float(r1) * scale;
            float f2 = __uint_as_float(r2) * scale;
            float f3 = __uint_as_float(r3) * scale;
            
            if (step * 128 + col + 0 >= S) f0 = -INFINITY;
            if (step * 128 + col + 1 >= S) f1 = -INFINITY;
            if (step * 128 + col + 2 >= S) f2 = -INFINITY;
            if (step * 128 + col + 3 >= S) f3 = -INFINITY;
            
            row_max = max({row_max, f0, f1, f2, f3});
            
            smem_P[tid * 128 + col] = __float2bfloat16(f0);
            smem_P[tid * 128 + col + 1] = __float2bfloat16(f1);
            smem_P[tid * 128 + col + 2] = __float2bfloat16(f2);
            smem_P[tid * 128 + col + 3] = __float2bfloat16(f3);
        }
        
        tmem_load_fence_fn();
        for (int col = 64; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_S + col, &r0, &r1, &r2, &r3);
            float f0 = __uint_as_float(r0) * scale;
            float f1 = __uint_as_float(r1) * scale;
            float f2 = __uint_as_float(r2) * scale;
            float f3 = __uint_as_float(r3) * scale;
            
            if (step * 128 + col + 0 >= S) f0 = -INFINITY;
            if (step * 128 + col + 1 >= S) f1 = -INFINITY;
            if (step * 128 + col + 2 >= S) f2 = -INFINITY;
            if (step * 128 + col + 3 >= S) f3 = -INFINITY;
            
            row_max = max({row_max, f0, f1, f2, f3});
            
            smem_P[tid * 128 + col] = __float2bfloat16(f0);
            smem_P[tid * 128 + col + 1] = __float2bfloat16(f1);
            smem_P[tid * 128 + col + 2] = __float2bfloat16(f2);
            smem_P[tid * 128 + col + 3] = __float2bfloat16(f3);
        }
        
        float prev_max = global_max;
        float new_max = max(prev_max, row_max);
        
        if (new_max > prev_max) {
            float factor = __expf(prev_max - new_max);
            tmem_load_fence_fn();
            for (int col = 0; col < 128; col += 4) {
                uint32_t r0, r1, r2, r3;
                tmem_load_4x_fn(tmem_O + col, &r0, &r1, &r2, &r3);
                r0 = __float_as_uint(__uint_as_float(r0) * factor);
                r1 = __float_as_uint(__uint_as_float(r1) * factor);
                r2 = __float_as_uint(__uint_as_float(r2) * factor);
                r3 = __float_as_uint(__uint_as_float(r3) * factor);
                
                uint32_t addr = (uint32_t)__cvta_generic_to_shared(&smem_P[tid * 128 + apply_swizzle_128B(col, tid)]);
                st_shared_128_fn(addr, r0, r1, r2, r3);
            }
            global_sum *= __expf(prev_max - new_max);
        }
        global_max = new_max;
        
        float row_sum = 0;
        tmem_load_fence_fn();
        for (int col = 0; col < 64; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_S + col, &r0, &r1, &r2, &r3);
            float f0 = __uint_as_float(r0) * scale;
            float f1 = __uint_as_float(r1) * scale;
            float f2 = __uint_as_float(r2) * scale;
            float f3 = __uint_as_float(r3) * scale;
            
            if (step * 128 + col + 0 >= S) f0 = -INFINITY;
            if (step * 128 + col + 1 >= S) f1 = -INFINITY;
            if (step * 128 + col + 2 >= S) f2 = -INFINITY;
            if (step * 128 + col + 3 >= S) f3 = -INFINITY;
            
            float p0 = (f0 == -INFINITY) ? 0.0f : __expf(f0 - row_max);
            float p1 = (f1 == -INFINITY) ? 0.0f : __expf(f1 - row_max);
            float p2 = (f2 == -INFINITY) ? 0.0f : __expf(f2 - row_max);
            float p3 = (f3 == -INFINITY) ? 0.0f : __expf(f3 - row_max);
            
            p0 *= __expf(row_max - global_max);
            p1 *= __expf(row_max - global_max);
            p2 *= __expf(row_max - global_max);
            p3 *= __expf(row_max - global_max);
            
            row_sum += p0 + p1 + p2 + p3;
            
            int swizzled_col0 = apply_swizzle_128B(col, tid);
            int swizzled_col1 = apply_swizzle_128B(col + 1, tid);
            int swizzled_col2 = apply_swizzle_128B(col + 2, tid);
            int swizzled_col3 = apply_swizzle_128B(col + 3, tid);
            
            smem_P[tid * 128 + swizzled_col0] = __float2bfloat16(p0);
            smem_P[tid * 128 + swizzled_col1] = __float2bfloat16(p1);
            smem_P[tid * 128 + swizzled_col2] = __float2bfloat16(p2);
            smem_P[tid * 128 + swizzled_col3] = __float2bfloat16(p3);
        }
        
        tmem_load_fence_fn();
        for (int col = 64; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_S + col, &r0, &r1, &r2, &r3);
            float f0 = __uint_as_float(r0) * scale;
            float f1 = __uint_as_float(r1) * scale;
            float f2 = __uint_as_float(r2) * scale;
            float f3 = __uint_as_float(r3) * scale;
            
            if (step * 128 + col + 0 >= S) f0 = -INFINITY;
            if (step * 128 + col + 1 >= S) f1 = -INFINITY;
            if (step * 128 + col + 2 >= S) f2 = -INFINITY;
            if (step * 128 + col + 3 >= S) f3 = -INFINITY;
            
            float p0 = (f0 == -INFINITY) ? 0.0f : __expf(f0 - row_max);
            float p1 = (f1 == -INFINITY) ? 0.0f : __expf(f1 - row_max);
            float p2 = (f2 == -INFINITY) ? 0.0f : __expf(f2 - row_max);
            float p3 = (f3 == -INFINITY) ? 0.0f : __expf(f3 - row_max);
            
            p0 *= __expf(row_max - global_max);
            p1 *= __expf(row_max - global_max);
            p2 *= __expf(row_max - global_max);
            p3 *= __expf(row_max - global_max);
            
            row_sum += p0 + p1 + p2 + p3;
            
            int swizzled_col0 = apply_swizzle_128B(col, tid);
            int swizzled_col1 = apply_swizzle_128B(col + 1, tid);
            int swizzled_col2 = apply_swizzle_128B(col + 2, tid);
            int swizzled_col3 = apply_swizzle_128B(col + 3, tid);
            
            smem_P[tid * 128 + swizzled_col0] = __float2bfloat16(p0);
            smem_P[tid * 128 + swizzled_col1] = __float2bfloat16(p1);
            smem_P[tid * 128 + swizzled_col2] = __float2bfloat16(p2);
            smem_P[tid * 128 + swizzled_col3] = __float2bfloat16(p3);
        }
        global_sum += row_sum;
        
        __syncthreads();
        fence_proxy_async_fn();
        
        if (tid == 0) {
            for (int k = 0; k < 64; k += 16) {
                uint64_t desc_P_k = desc_P + (k * 2);
                uint64_t desc_V0_k = desc_V0 + (k * 16);
                umma_f16_cg1_fn(tmem_O, desc_P_k, desc_V0_k, idesc_PV, k == 0 ? 0 : 1);
            }
            for (int k = 0; k < 64; k += 16) {
                uint64_t desc_P_k = desc_P + (k * 2);
                uint64_t desc_V1_k = desc_V1 + (k * 16);
                umma_f16_cg1_fn(tmem_O + 64, desc_P_k, desc_V1_k, idesc_PV, 1);
            }
            tcgen05_commit_fn(bar_PV);
        }
        mbarrier_wait_fn(bar_PV, phase_pv);
        phase_pv ^= 1;
    }
    
    __nv_bfloat16* g_O = O + bh * S * 128;
    float* g_LSE = LSE + bh * S;
    
    tmem_load_fence_fn();
    for (int col = 0; col < 64; col += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(tmem_O + col, &r0, &r1, &r2, &r3);
        
        float out_val0 = __uint_as_float(r0);
        float out_val1 = __uint_as_float(r1);
        float out_val2 = __uint_as_float(r2);
        float out_val3 = __uint_as_float(r3);
        
        if (global_sum > 0.0f) {
            out_val0 /= global_sum;
            out_val1 /= global_sum;
            out_val2 /= global_sum;
            out_val3 /= global_sum;
        }
        
        __nv_bfloat16* out = g_O + q_base * 128 + tid * 128 + col;
        if (q_base + tid < S) {
            out[0] = __float2bfloat16(out_val0);
            out[1] = __float2bfloat16(out_val1);
            out[2] = __float2bfloat16(out_val2);
            out[3] = __float2bfloat16(out_val3);
        }
    }
    
    tmem_load_fence_fn();
    for (int col = 64; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(tmem_O + col, &r0, &r1, &r2, &r3);
        
        float out_val0 = __uint_as_float(r0);
        float out_val1 = __uint_as_float(r1);
        float out_val2 = __uint_as_float(r2);
        float out_val3 = __uint_as_float(r3);
        
        if (global_sum > 0.0f) {
            out_val0 /= global_sum;
            out_val1 /= global_sum;
            out_val2 /= global_sum;
            out_val3 /= global_sum;
        }
        
        __nv_bfloat16* out = g_O + q_base * 128 + tid * 128 + col;
        if (q_base + tid < S) {
            out[0] = __float2bfloat16(out_val0);
            out[1] = __float2bfloat16(out_val1);
            out[2] = __float2bfloat16(out_val2);
            out[3] = __float2bfloat16(out_val3);
        }
    }
    
    if (tid == 0) {
        if (q_base < S) {
            g_LSE[q_base] = global_max + __logf(global_sum);
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
    
    CUtensorMap tma_Q, tma_K, tma_V;
    
    CUresult res;
    res = create_tma_2d_descriptor_2B(&tma_Q, static_cast<const __nv_bfloat16*>(Q.data_ptr()), 
        D, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA Q failed\n"); exit(1); }
    
    res = create_tma_2d_descriptor_2B(&tma_K, static_cast<const __nv_bfloat16*>(K.data_ptr()), 
        D, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA K failed\n"); exit(1); }
    
    res = create_tma_2d_descriptor_2B(&tma_V, static_cast<const __nv_bfloat16*>(V.data_ptr()), 
        D, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA V failed\n"); exit(1); }
    
    int smem_size = 114720;
    CUDA_CHECK(cudaFuncSetAttribute(
        mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    dim3 grid(min(S, 8192), 1, B * H);
    dim3 block(128);
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    mha_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V,
        static_cast<__nv_bfloat16*>(O.data_ptr()),
        static_cast<float*>(LSE.data_ptr()),
        S, B * H);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha::run);

}  // namespace tvm_ffi_mha