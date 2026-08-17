#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>
#include <math.h>

#define CU_CHECK(call) do {                                      \
    CUresult _e = (call);                                        \
    if (_e != CUDA_SUCCESS) {                                    \
        fprintf(stderr, "CU error %d at %s:%d\n",               \
                _e, __FILE__, __LINE__);                         \
        exit(1);                                                 \
    }                                                            \
} while(0)

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

__device__ __forceinline__ uint32_t swizzle_128B(uint32_t row, uint32_t col) {
    uint32_t span_idx = col >> 3;
    uint32_t offset = col & 7;
    uint32_t swizzled_span = (row & 7) ^ span_idx;
    return ((row << 6) + (swizzled_span << 3)) + offset;
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, bool transpose_b) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (0u << 15);   
    d |= ((uint32_t)transpose_b << 16);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
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

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_store_fence_fn() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_store_4x_fn(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
       :: "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3));
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(phase));
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d), "r"((uint32_t)__cvta_generic_to_shared(bar)),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void umma_commit_1sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                 :: "r"(a));
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle, l2Promotion, oobFill
    );
}

struct SharedStorage {
    __nv_bfloat16 Q_0[4096]; 
    __nv_bfloat16 Q_1[4096]; 
    
    __nv_bfloat16 K_0[4096];     
    __nv_bfloat16 K_1[4096];     
    __nv_bfloat16 V_0[4096];     
    __nv_bfloat16 V_1[4096];     
    
    __nv_bfloat16 O_0[4096];     
    __nv_bfloat16 O_1[4096];     
    
    __nv_bfloat16 P[4096];       
};

__global__ __launch_bounds__(128) void flash_attention_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O, float* LSE,
    uint32_t S_len, uint32_t num_heads) 
{
    uint32_t bh_idx = blockIdx.y;
    uint32_t q_start = blockIdx.x * 128;
    
    if (q_start >= S_len) return;
    
    extern __shared__ char smem[];
    SharedStorage& s = *reinterpret_cast<SharedStorage*>(smem);
    uint32_t tid = threadIdx.x;
    
    __shared__ alignas(8) uint64_t bar_Q[2];
    __shared__ alignas(8) uint64_t bar_KV[4];
    __shared__ alignas(8) uint64_t bar_QK;
    __shared__ alignas(8) uint64_t bar_PV;
    
    if (tid == 0) {
        init_smem_barrier_fn(&bar_Q[0], 1);
        init_smem_barrier_fn(&bar_Q[1], 1);
        
        init_smem_barrier_fn(&bar_KV[0], 1);
        init_smem_barrier_fn(&bar_KV[1], 1);
        init_smem_barrier_fn(&bar_KV[2], 1);
        init_smem_barrier_fn(&bar_KV[3], 1);
        
        init_smem_barrier_fn(&bar_QK, 1);
        init_smem_barrier_fn(&bar_PV, 1);
        
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    __shared__ alignas(4) uint32_t tmem_base_ptr[1];
    if (tid == 0) {
        tmem_alloc_fn(&tmem_base_ptr[0], 256);
    }
    __syncthreads();
    uint32_t tmem_base = tmem_base_ptr[0];
    
    uint32_t phase_Q = 0;
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(&bar_Q[0], 8192);
        tma_load_2d_fn(&tma_Q, &bar_Q[0], s.Q_0, 0, bh_idx * S_len + q_start);
        
        mbarrier_arrive_and_expect_tx_fn(&bar_Q[1], 8192);
        tma_load_2d_fn(&tma_Q, &bar_Q[1], s.Q_1, 64, bh_idx * S_len + q_start);
    }
    mbarrier_wait_fn(&bar_Q[0], phase_Q);
    mbarrier_wait_fn(&bar_Q[1], phase_Q);
    phase_Q ^= 1;
    
    __shared__ float running_max_top[64];
    __shared__ float running_sum_top[64];
    __shared__ float running_max_bot[64];
    __shared__ float running_sum_bot[64];
    
    if (tid < 64) {
        running_max_top[tid] = -1e20f;
        running_sum_top[tid] = 0.0f;
        running_max_bot[tid] = -1e20f;
        running_sum_bot[tid] = 0.0f;
    }
    __syncthreads();
    
    uint32_t S_TMEM = tmem_base;
    uint32_t O0_TMEM = tmem_base + 64;
    uint32_t O1_TMEM = tmem_base + 128;
    
    uint32_t my_S_addr = S_TMEM + (tid << 16);
    uint32_t my_O0_addr = O0_TMEM + (tid << 16);
    uint32_t my_O1_addr = O1_TMEM + (tid << 16);
    
    // Initialize O to 0
    for (int col = 0; col < 64; col += 4) {
        uint32_t r0=0, r1=0, r2=0, r3=0;
        tmem_store_4x_fn(my_O0_addr + col, r0, r1, r2, r3);
        tmem_store_4x_fn(my_O1_addr + col, r0, r1, r2, r3);
    }
    tmem_store_fence_fn();
    
    float scale = 1.0f / sqrtf(128.0f);
    
    uint64_t desc_Q_0 = make_smem_desc_sm100_fn(s.Q_0, 1, 1024);
    uint64_t desc_Q_1 = make_smem_desc_sm100_fn(s.Q_1, 1, 1024);
    
    uint64_t desc_K_0 = make_smem_desc_sm100_fn(s.K_0, 1, 1024);
    uint64_t desc_K_1 = make_smem_desc_sm100_fn(s.K_1, 1, 1024);

    uint64_t desc_V_0 = make_smem_desc_sm100_fn(s.V_0, 8192, 1024);
    uint64_t desc_V_1 = make_smem_desc_sm100_fn(s.V_1, 8192, 1024);

    uint64_t desc_P = make_smem_desc_sm100_fn(s.P, 1, 1024);
    
    uint32_t idesc_Q = make_instr_desc_fn(64, 64, false);
    uint32_t idesc_P = make_instr_desc_fn(64, 128, true); // Transpose B to N-Major
    
    uint32_t phase_KV = 0;
    uint32_t phase_QK = 0;
    uint32_t phase_PV = 0;
    
    for (int c_start = 0; c_start < S_len; c_start += 64) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(&bar_KV[0], 8192);
            tma_load_2d_fn(&tma_K, &bar_KV[0], s.K_0, 0, bh_idx * S_len + c_start);
            
            mbarrier_arrive_and_expect_tx_fn(&bar_KV[1], 8192);
            tma_load_2d_fn(&tma_K, &bar_KV[1], s.K_1, 64, bh_idx * S_len + c_start);
            
            mbarrier_arrive_and_expect_tx_fn(&bar_KV[2], 8192);
            tma_load_2d_fn(&tma_V, &bar_KV[2], s.V_0, 0, bh_idx * S_len + c_start);
            
            mbarrier_arrive_and_expect_tx_fn(&bar_KV[3], 8192);
            tma_load_2d_fn(&tma_V, &bar_KV[3], s.V_1, 64, bh_idx * S_len + c_start);
        }
        mbarrier_wait_fn(&bar_KV[0], phase_KV);
        mbarrier_wait_fn(&bar_KV[1], phase_KV);
        mbarrier_wait_fn(&bar_KV[2], phase_KV);
        mbarrier_wait_fn(&bar_KV[3], phase_KV);
        phase_KV ^= 1;
        
        if (tid == 0) {
            for (int k = 0; k < 64; k += 16) {
                uint32_t accum = (k == 0 && c_start == 0) ? 0 : 1;
                uint64_t desc_A = desc_Q_0 + (k * 2);
                uint64_t desc_B = desc_K_0 + (k * 2);
                umma_f16_cg1_fn(S_TMEM, desc_A, desc_B, idesc_Q, accum);
            }
            for (int k = 0; k < 64; k += 16) {
                uint64_t desc_A = desc_Q_1 + (k * 2);
                uint64_t desc_B = desc_K_1 + (k * 2);
                umma_f16_cg1_fn(S_TMEM, desc_A, desc_B, idesc_Q, 1);
            }
        }
        
        if (tid == 0) {
            umma_commit_1sm_fn(&bar_QK);
        }
        mbarrier_wait_fn(&bar_QK, phase_QK);
        phase_QK ^= 1;
        
        tmem_load_fence_fn(); 
        
        float S_vals[64];
        for (int col = 0; col < 64; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(my_S_addr + col, &r0, &r1, &r2, &r3);
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            
            S_vals[col] = f0;
            S_vals[col+1] = f1;
            S_vals[col+2] = f2;
            S_vals[col+3] = f3;
        }
        
        uint32_t row = tid % 64;
        uint32_t is_top = tid < 64;
        
        if (q_start + tid >= S_len) {
            for(int i=0; i<64; ++i) S_vals[i] = -1e20f;
        } else {
            for(int i=0; i<64; ++i) {
                if (c_start + i >= S_len) S_vals[i] = -1e20f;
            }
        }
        
        float max_val = -1e20f;
        for(int i=0; i<64; ++i) {
            max_val = fmaxf(max_val, S_vals[i] * scale);
        }
        
        float r_max = is_top ? running_max_top[row] : running_max_bot[row];
        float new_max = fmaxf(r_max, max_val);
        
        float sum_val = 0.0f;
        for(int i=0; i<64; ++i) {
            float v = __expf(S_vals[i] * scale - new_max);
            sum_val += v;
            S_vals[i] = v; 
        }
        
        float r_sum = is_top ? running_sum_top[row] : running_sum_bot[row];
        if (r_max == -1e20f) {
            r_sum = sum_val;
        } else {
            r_sum = r_sum * __expf(r_max - new_max) + sum_val;
        }
        
        float scale_factor = (r_max == -1e20f) ? 0.0f : __expf(r_max - new_max);
        
        if (tid % 2 == 0) {
            if (is_top) {
                running_max_top[row] = new_max;
                running_sum_top[row] = r_sum;
            } else {
                running_max_bot[row] = new_max;
                running_sum_bot[row] = r_sum;
            }
        }
        
        for(int i=0; i<64; ++i) {
            __nv_bfloat16 val = __float2bfloat16(S_vals[i]);
            s.P[swizzle_128B(row, i)] = val;
        }
        
        __syncthreads(); 
        
        if (scale_factor != 1.0f) {
            for (int col = 0; col < 64; col += 4) {
                uint32_t r0, r1, r2, r3;
                tmem_load_4x_fn(my_O0_addr + col, &r0, &r1, &r2, &r3);
                float f0 = __uint_as_float(r0) * scale_factor;
                float f1 = __uint_as_float(r1) * scale_factor;
                float f2 = __uint_as_float(r2) * scale_factor;
                float f3 = __uint_as_float(r3) * scale_factor;
                
                tmem_store_4x_fn(my_O0_addr + col, __float_as_uint(f0), __float_as_uint(f1), __float_as_uint(f2), __float_as_uint(f3));
                
                tmem_load_4x_fn(my_O1_addr + col, &r0, &r1, &r2, &r3);
                f0 = __uint_as_float(r0) * scale_factor;
                f1 = __uint_as_float(r1) * scale_factor;
                f2 = __uint_as_float(r2) * scale_factor;
                f3 = __uint_as_float(r3) * scale_factor;
                
                tmem_store_4x_fn(my_O1_addr + col, __float_as_uint(f0), __float_as_uint(f1), __float_as_uint(f2), __float_as_uint(f3));
            }
            tmem_store_fence_fn();
        }
        __syncthreads(); 
        
        fence_proxy_async_fn(); 
        
        if (tid == 0) {
            for (int k = 0; k < 64; k += 16) {
                uint32_t accum = (c_start == 0 && k == 0) ? 0 : 1; 
                uint64_t desc_P_d = desc_P + (k * 2);
                uint64_t desc_V0_d = desc_V_0 + (k * 128); 
                uint64_t desc_V1_d = desc_V_1 + (k * 128); 
                
                uint64_t desc_O0 = O0_TMEM + (k * 128);
                umma_f16_cg1_fn(desc_O0, desc_P_d, desc_V0_d, idesc_P, accum);
                
                uint64_t desc_O1 = O1_TMEM + (k * 128);
                umma_f16_cg1_fn(desc_O1, desc_P_d, desc_V1_d, idesc_P, accum);
            }
        }
        
        if (tid == 0) {
            umma_commit_1sm_fn(&bar_PV);
        }
        mbarrier_wait_fn(&bar_PV, phase_PV);
        phase_PV ^= 1;
        
        __syncthreads();
    }
    
    tmem_load_fence_fn();
    
    for (int col = 0; col < 64; col += 4) {
        uint32_t r0, r1, r2, r3;
        
        tmem_load_4x_fn(my_O0_addr + col, &r0, &r1, &r2, &r3);
        float inv_sum_0 = (tid < 64) ? ((running_sum_top[tid] > 0.0f) ? 1.0f / running_sum_top[tid] : 0.0f) : 0.0f;
        s.O_0[swizzle_128B(tid, col)] = __float2bfloat16(__uint_as_float(r0) * inv_sum_0);
        s.O_0[swizzle_128B(tid, col+1)] = __float2bfloat16(__uint_as_float(r1) * inv_sum_0);
        s.O_0[swizzle_128B(tid, col+2)] = __float2bfloat16(__uint_as_float(r2) * inv_sum_0);
        s.O_0[swizzle_128B(tid, col+3)] = __float2bfloat16(__uint_as_float(r3) * inv_sum_0);
        
        tmem_load_4x_fn(my_O1_addr + col, &r0, &r1, &r2, &r3);
        float inv_sum_1 = (tid < 64) ? ((running_sum_top[tid] > 0.0f) ? 1.0f / running_sum_top[tid] : 0.0f) : 0.0f;
        s.O_1[swizzle_128B(tid, col)] = __float2bfloat16(__uint_as_float(r0) * inv_sum_1);
        s.O_1[swizzle_128B(tid, col+1)] = __float2bfloat16(__uint_as_float(r1) * inv_sum_1);
        s.O_1[swizzle_128B(tid, col+2)] = __float2bfloat16(__uint_as_float(r2) * inv_sum_1);
        s.O_1[swizzle_128B(tid, col+3)] = __float2bfloat16(__uint_as_float(r3) * inv_sum_1);
        
        tmem_load_4x_fn(my_O0_addr + ((tid+64) << 16) + col, &r0, &r1, &r2, &r3);
        inv_sum_0 = (tid >= 64) ? ((running_sum_bot[tid-64] > 0.0f) ? 1.0f / running_sum_bot[tid-64] : 0.0f) : 0.0f;
        s.O_0[swizzle_128B(tid-64, col)] = __float2bfloat16(__uint_as_float(r0) * inv_sum_0);
        s.O_0[swizzle_128B(tid-64, col+1)] = __float2bfloat16(__uint_as_float(r1) * inv_sum_0);
        s.O_0[swizzle_128B(tid-64, col+2)] = __float2bfloat16(__uint_as_float(r2) * inv_sum_0);
        s.O_0[swizzle_128B(tid-64, col+3)] = __float2bfloat16(__uint_as_float(r3) * inv_sum_0);
        
        tmem_load_4x_fn(my_O1_addr + ((tid+64) << 16) + col, &r0, &r1, &r2, &r3);
        inv_sum_1 = (tid >= 64) ? ((running_sum_bot[tid-64] > 0.0f) ? 1.0f / running_sum_bot[tid-64] : 0.0f) : 0.0f;
        s.O_1[swizzle_128B(tid-64, col)] = __float2bfloat16(__uint_as_float(r0) * inv_sum_1);
        s.O_1[swizzle_128B(tid-64, col+1)] = __float2bfloat16(__uint_as_float(r1) * inv_sum_1);
        s.O_1[swizzle_128B(tid-64, col+2)] = __float2bfloat16(__uint_as_float(r2) * inv_sum_1);
        s.O_1[swizzle_128B(tid-64, col+3)] = __float2bfloat16(__uint_as_float(r3) * inv_sum_1);
    }
    
    __syncthreads();
    
    __nv_bfloat16* my_O = O + (uint64_t)bh_idx * S_len * 128;
    
    for(int i = 0; i < 4096 / 4; ++i) {
        uint32_t elem_idx = tid + i * 128;
        uint32_t row = elem_idx / 64;
        uint32_t col = elem_idx % 64;
        
        __nv_bfloat16 val0 = s.O_0[swizzle_128B(row, col)];
        __nv_bfloat16 val1 = s.O_1[swizzle_128B(row, col)];
        
        uint32_t s_idx = q_start + row;
        uint32_t d_idx = col;
        
        if (s_idx < S_len) {
            uint64_t g_idx = (uint64_t)s_idx * 128 + d_idx;
            *reinterpret_cast<uint2*>(&my_O[g_idx]) = *reinterpret_cast<uint2*>(&val0);
            d_idx += 64;
            if (d_idx < 128) {
                *reinterpret_cast<__nv_bfloat16*>(&my_O[g_idx + 64]) = val1;
            }
        }
    }
    
    for(int i = 0; i < 4096 / 4; ++i) {
        uint32_t elem_idx = tid + i * 128;
        uint32_t row = elem_idx / 64;
        uint32_t col = elem_idx % 64;
        
        __nv_bfloat16 val0 = s.O_0[swizzle_128B(row, col)];
        __nv_bfloat16 val1 = s.O_1[swizzle_128B(row, col)];
        
        uint32_t s_idx = q_start + row + 64;
        uint32_t d_idx = col;
        
        if (s_idx < S_len) {
            uint64_t g_idx = (uint64_t)s_idx * 128 + d_idx;
            *reinterpret_cast<uint2*>(&my_O[g_idx]) = *reinterpret_cast<uint2*>(&val0);
            d_idx += 64;
            if (d_idx < 128) {
                *reinterpret_cast<__nv_bfloat16*>(&my_O[g_idx + 64]) = val1;
            }
        }
    }
    
    if (tid < 64) {
        uint32_t s_idx_0 = q_start + tid;
        if (s_idx_0 < S_len) {
            LSE[(uint64_t)bh_idx * S_len + s_idx_0] = running_max_top[tid] + logf(running_sum_top[tid]);
        }
        
        uint32_t s_idx_1 = q_start + tid + 64;
        if (s_idx_1 < S_len) {
            LSE[(uint64_t)bh_idx * S_len + s_idx_1] = running_max_bot[tid] + logf(running_sum_bot[tid]);
        }
    }
}

namespace tvm_ffi_mha_lse {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    uint32_t B = Q.size(0);
    uint32_t H = Q.size(1);
    uint32_t S_len = Q.size(2);
    uint32_t D = Q.size(3); 
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE.data_ptr());
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUtensorMap tma_Q, tma_K, tma_V;
    
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, (void*)Q_ptr, D, B * H * S_len, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, (void*)K_ptr, D, B * H * S_len, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, (void*)V_ptr, D, B * H * S_len, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    uint32_t num_blocks_x = (S_len + 127) / 128;
    dim3 grid(num_blocks_x, B * H);
    dim3 block(128);
    
    uint32_t smem_size = sizeof(SharedStorage) + 1024;
    CUDA_CHECK(cudaFuncSetAttribute(
        flash_attention_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    ));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(&config, flash_attention_kernel, tma_Q, tma_K, tma_V, O_ptr, LSE_ptr, S_len, H));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_mha_lse