#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <stdio.h>
#include <cmath>
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

__device__ __forceinline__ void tmem_alloc_fn_cta1(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn_cta1(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
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

__device__ __forceinline__ uint64_t make_smem_desc(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_mn_major(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_k_major(uint32_t M, uint32_t N) {
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

__device__ __forceinline__ uint32_t make_instr_desc_b_mn_major(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (0u << 15);   
    d |= (1u << 16);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ void umma_f16_cta1_fn(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.shared::cta.b64"
        " [%0];"
        :: "r"(a));
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

__device__ __forceinline__ int swizzle_128B_offset(int r, int c) {
    return r * 64 + (((r % 8) ^ (c / 8)) * 8 + (c % 8));
}

CUresult create_tma_4d_descriptor(CUtensorMap* d, void* globalAddress,
                                  uint64_t dim0, uint64_t dim1, uint64_t dim2, uint64_t dim3,
                                  uint32_t box0, uint32_t box1,
                                  CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[4] = {dim0, dim1, dim2, dim3};
    cuuint64_t globalStrides[3] = {dim0 * 2, dim0 * dim1 * 2, dim0 * dim1 * dim2 * 2};
    cuuint32_t boxDim[4] = {box0, box1, 1, 1};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
        globalAddress, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

__global__ void run_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O_ptr, float* LSE_ptr,
    int S, int num_heads, int batch_size) 
{
    extern __shared__ __align__(128) uint8_t smem_pool[];
    uintptr_t pool_addr = (uintptr_t)smem_pool;
    uint8_t* cur = smem_pool + (1024 - (pool_addr % 1024)) % 1024;
    
    uint64_t* mbar = (uint64_t*)cur;
    cur += 16;
    cur = (uint8_t*)((((uintptr_t)cur) + 1023) & ~1023);
    
    __nv_bfloat16* s_Q0 = (__nv_bfloat16*)cur; cur += 8192;
    __nv_bfloat16* s_Q1 = (__nv_bfloat16*)cur; cur += 8192;
    __nv_bfloat16* s_K0 = (__nv_bfloat16*)cur; cur += 8192;
    __nv_bfloat16* s_K1 = (__nv_bfloat16*)cur; cur += 8192;
    __nv_bfloat16* s_V0 = (__nv_bfloat16*)cur; cur += 8192;
    __nv_bfloat16* s_V1 = (__nv_bfloat16*)cur; cur += 8192;
    __nv_bfloat16* s_P  = (__nv_bfloat16*)cur; cur += 8192;
    
    int head_idx = blockIdx.y;
    int batch_idx = blockIdx.z;
    int q_base = blockIdx.x * 64;
    int flattened_head = head_idx + batch_idx * num_heads;
    int global_q_base = (flattened_head * S) + q_base;
    
    int q_idx = threadIdx.x; 
    int global_q_idx = q_base + q_idx;
    
    uint32_t tmem_S, tmem_O_left, tmem_O_right;
    if (threadIdx.x == 0) {
        tmem_alloc_fn_cta1(&tmem_S, 64);
        tmem_alloc_fn_cta1(&tmem_O_left, 64);
        tmem_alloc_fn_cta1(&tmem_O_right, 64);
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&mbar[0], 1);
        init_smem_barrier_fn(&mbar[1], 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar[0], 16384);
        tma_load_4d_fn(&tma_Q, &mbar[0], s_Q0, 0, q_base, head_idx, batch_idx);
        tma_load_4d_fn(&tma_Q, &mbar[0], s_Q1, 64, q_base, head_idx, batch_idx);
        
        mbarrier_arrive_and_expect_tx_fn(&mbar[1], 32768);
        tma_load_4d_fn(&tma_K, &mbar[1], s_K0, 0, 0, head_idx, batch_idx);
        tma_load_4d_fn(&tma_K, &mbar[1], s_K1, 64, 0, head_idx, batch_idx);
        tma_load_4d_fn(&tma_V, &mbar[1], s_V0, 0, 0, head_idx, batch_idx);
        tma_load_4d_fn(&tma_V, &mbar[1], s_V1, 64, 0, head_idx, batch_idx);
    }
    
    mbarrier_wait_fn(&mbar[0], 0);
    
    float O_scaled_left[64];
    float O_scaled_right[64];
    #pragma unroll
    for(int i=0; i<64; i++) {
        O_scaled_left[i] = 0.0f;
        O_scaled_right[i] = 0.0f;
    }
    
    float row_max_prev = -INFINITY;
    float row_sum_prev = 0;
    
    int max_step = (q_base + 63) / 64;
    if (max_step >= (S + 63) / 64) {
        max_step = (S + 63) / 64 - 1;
    }
    
    int next_barrier_phase = 0;
    float scale_factor = 1.0f / sqrtf(128.0f);
    
    for (int step = 0; step <= max_step; ++step) {
        int k_base = step * 64;
        int next_k_base = (step + 1) * 64;
        
        if (k_base > q_base) break; 
        
        if (next_k_base <= q_base && next_k_base < S) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&mbar[1], 32768);
                tma_load_4d_fn(&tma_K, &mbar[1], s_K0, 0, next_k_base, head_idx, batch_idx);
                tma_load_4d_fn(&tma_K, &mbar[1], s_K1, 64, next_k_base, head_idx, batch_idx);
                tma_load_4d_fn(&tma_V, &mbar[1], s_V0, 0, next_k_base, head_idx, batch_idx);
                tma_load_4d_fn(&tma_V, &mbar[1], s_V1, 64, next_k_base, head_idx, batch_idx);
            }
        }
        
        mbarrier_wait_fn(&mbar[1], next_barrier_phase & 1);
        
        if (threadIdx.x == 0) {
            for (int k = 0; k < 64; k += 16) {
                uint64_t desc_a = make_smem_desc(s_Q0 + k, 1, 1024);
                uint64_t desc_b = make_smem_desc(s_K0 + k, 1, 1024);
                uint32_t idesc = make_instr_desc_k_major(64, 64);
                umma_f16_cta1_fn(tmem_S, desc_a, desc_b, idesc, k == 0 ? 0 : 1);
            }
            for (int k = 0; k < 64; k += 16) {
                uint64_t desc_a = make_smem_desc(s_Q1 + k, 1, 1024);
                uint64_t desc_b = make_smem_desc(s_K1 + k, 1, 1024);
                uint32_t idesc = make_instr_desc_k_major(64, 64);
                umma_f16_cta1_fn(tmem_S, desc_a, desc_b, idesc, 1);
            }
            umma_commit_fn(&mbar[0]);
        }
        mbarrier_wait_fn(&mbar[0], next_barrier_phase & 1);
        
        uint32_t S_val_u[64];
        tmem_load_8x_fn(tmem_S + 0, &S_val_u[0], &S_val_u[1], &S_val_u[2], &S_val_u[3], &S_val_u[4], &S_val_u[5], &S_val_u[6], &S_val_u[7]);
        tmem_load_8x_fn(tmem_S + 8, &S_val_u[8], &S_val_u[9], &S_val_u[10], &S_val_u[11], &S_val_u[12], &S_val_u[13], &S_val_u[14], &S_val_u[15]);
        tmem_load_8x_fn(tmem_S + 16, &S_val_u[16], &S_val_u[17], &S_val_u[18], &S_val_u[19], &S_val_u[20], &S_val_u[21], &S_val_u[22], &S_val_u[23]);
        tmem_load_8x_fn(tmem_S + 24, &S_val_u[24], &S_val_u[25], &S_val_u[26], &S_val_u[27], &S_val_u[28], &S_val_u[29], &S_val_u[30], &S_val_u[31]);
        tmem_load_8x_fn(tmem_S + 32, &S_val_u[32], &S_val_u[33], &S_val_u[34], &S_val_u[35], &S_val_u[36], &S_val_u[37], &S_val_u[38], &S_val_u[39]);
        tmem_load_8x_fn(tmem_S + 40, &S_val_u[40], &S_val_u[41], &S_val_u[42], &S_val_u[43], &S_val_u[44], &S_val_u[45], &S_val_u[46], &S_val_u[47]);
        tmem_load_8x_fn(tmem_S + 48, &S_val_u[48], &S_val_u[49], &S_val_u[50], &S_val_u[51], &S_val_u[52], &S_val_u[53], &S_val_u[54], &S_val_u[55]);
        tmem_load_8x_fn(tmem_S + 56, &S_val_u[56], &S_val_u[57], &S_val_u[58], &S_val_u[59], &S_val_u[60], &S_val_u[61], &S_val_u[62], &S_val_u[63]);
        tmem_load_fence_fn();
        
        float current_row_max = -INFINITY;
        float S_val_f[64];
        for (int i = 0; i < 64; i++) {
            int global_k_idx = k_base + i;
            if (global_q_idx < global_k_idx || global_k_idx >= S) {
                S_val_f[i] = -INFINITY;
            } else {
                S_val_f[i] = __uint_as_float(S_val_u[i]) * scale_factor;
            }
            current_row_max = fmaxf(current_row_max, S_val_f[i]);
        }
        
        float scale_l = 1.0f;
        float scale_r = 1.0f;
        if (current_row_max > row_max_prev) {
            scale_l = fast_exp2f_fn((row_max_prev - current_row_max) * 1.4426950f);
            scale_r = fast_exp2f_fn((row_max_prev - current_row_max) * 1.4426950f);
            row_sum_prev *= scale_l;
        }
        
        for (int i = 0; i < 64; i++) {
            O_scaled_left[i] *= scale_l;
            O_scaled_right[i] *= scale_r;
        }
        
        float current_row_sum = 0;
        for (int i = 0; i < 64; i++) {
            float p = 0;
            if (S_val_f[i] != -INFINITY) {
                p = fast_exp2f_fn((S_val_f[i] - current_row_max) * 1.4426950f);
                current_row_sum += p;
                S_val_u[i] = __float_as_uint(p);
            } else {
                S_val_u[i] = __float_as_uint(0.0f);
            }
        }
        
        row_sum_prev += current_row_sum;
        row_max_prev = current_row_max;
        
        for (int c = 0; c < 64; c+=2) {
            float p0 = __uint_as_float(S_val_u[c]);
            float p1 = __uint_as_float(S_val_u[c+1]);
            
            uint32_t packed = pack_bf16_fn(__float_as_uint(p0), __float_as_uint(p1));
            int sc = swizzle_128B_offset(q_idx, c);
            *(uint32_t*)&s_P[sc] = packed;
        }
        
        __syncthreads();
        fence_proxy_async_fn();
        
        if (threadIdx.x == 0) {
            for (int p = 0; p < 64; p += 16) {
                uint64_t desc_a = make_smem_desc(s_P + p, 1, 1024); 
                uint64_t desc_b = make_smem_desc_mn_major(s_V0 + p * 64, 8192, 1024);
                uint32_t idesc = make_instr_desc_b_mn_major(64, 64);
                umma_f16_cta1_fn(tmem_O_left, desc_a, desc_b, idesc, p == 0 ? 0 : 1);
            }
            for (int p = 0; p < 64; p += 16) {
                uint64_t desc_a = make_smem_desc(s_P + p, 1, 1024); 
                uint64_t desc_b = make_smem_desc_mn_major(s_V1 + p * 64, 8192, 1024);
                uint32_t idesc = make_instr_desc_b_mn_major(64, 64);
                umma_f16_cta1_fn(tmem_O_right, desc_a, desc_b, idesc, p == 0 ? 0 : 1);
            }
            umma_commit_fn(&mbar[0]);
        }
        mbarrier_wait_fn(&mbar[0], next_barrier_phase & 1);
        
        uint32_t PV_val[64];
        tmem_load_8x_fn(tmem_O_left + 0, &PV_val[0], &PV_val[1], &PV_val[2], &PV_val[3], &PV_val[4], &PV_val[5], &PV_val[6], &PV_val[7]);
        tmem_load_8x_fn(tmem_O_left + 8, &PV_val[8], &PV_val[9], &PV_val[10], &PV_val[11], &PV_val[12], &PV_val[13], &PV_val[14], &PV_val[15]);
        tmem_load_8x_fn(tmem_O_left + 16, &PV_val[16], &PV_val[17], &PV_val[18], &PV_val[19], &PV_val[20], &PV_val[21], &PV_val[22], &PV_val[23]);
        tmem_load_8x_fn(tmem_O_left + 24, &PV_val[24], &PV_val[25], &PV_val[26], &PV_val[27], &PV_val[28], &PV_val[29], &PV_val[30], &PV_val[31]);
        
        tmem_load_8x_fn(tmem_O_left + 32, &PV_val[32], &PV_val[33], &PV_val[34], &PV_val[35], &PV_val[36], &PV_val[37], &PV_val[38], &PV_val[39]);
        tmem_load_8x_fn(tmem_O_left + 40, &PV_val[40], &PV_val[41], &PV_val[42], &PV_val[43], &PV_val[44], &PV_val[45], &PV_val[46], &PV_val[47]);
        tmem_load_8x_fn(tmem_O_left + 48, &PV_val[48], &PV_val[49], &PV_val[50], &PV_val[51], &PV_val[52], &PV_val[53], &PV_val[54], &PV_val[55]);
        tmem_load_8x_fn(tmem_O_left + 56, &PV_val[56], &PV_val[57], &PV_val[58], &PV_val[59], &PV_val[60], &PV_val[61], &PV_val[62], &PV_val[63]);
        
        tmem_load_8x_fn(tmem_O_right + 0, &PV_val[0], &PV_val[1], &PV_val[2], &PV_val[3], &PV_val[4], &PV_val[5], &PV_val[6], &PV_val[7]);
        tmem_load_8x_fn(tmem_O_right + 8, &PV_val[8], &PV_val[9], &PV_val[10], &PV_val[11], &PV_val[12], &PV_val[13], &PV_val[14], &PV_val[15]);
        tmem_load_8x_fn(tmem_O_right + 16, &PV_val[16], &PV_val[17], &PV_val[18], &PV_val[19], &PV_val[20], &PV_val[21], &PV_val[22], &PV_val[23]);
        tmem_load_8x_fn(tmem_O_right + 24, &PV_val[24], &PV_val[25], &PV_val[26], &PV_val[27], &PV_val[28], &PV_val[29], &PV_val[30], &PV_val[31]);
        
        tmem_load_8x_fn(tmem_O_right + 32, &PV_val[32], &PV_val[33], &PV_val[34], &PV_val[35], &PV_val[36], &PV_val[37], &PV_val[38], &PV_val[39]);
        tmem_load_8x_fn(tmem_O_right + 40, &PV_val[40], &PV_val[41], &PV_val[42], &PV_val[43], &PV_val[44], &PV_val[45], &PV_val[46], &PV_val[47]);
        tmem_load_8x_fn(tmem_O_right + 48, &PV_val[48], &PV_val[49], &PV_val[50], &PV_val[51], &PV_val[52], &PV_val[53], &PV_val[54], &PV_val[55]);
        tmem_load_8x_fn(tmem_O_right + 56, &PV_val[56], &PV_val[57], &PV_val[58], &PV_val[59], &PV_val[60], &PV_val[61], &PV_val[62], &PV_val[63]);
        tmem_load_fence_fn();
        
        for (int i = 0; i < 64; i++) {
            O_scaled_left[i] += __uint_as_float(PV_val[i]);
            O_scaled_right[i] += __uint_as_float(PV_val[i]);
        }
        
        next_barrier_phase++;
        __syncthreads();
    }
    
    if (global_q_idx < S) {
        float sum = row_sum_prev;
        for (int i = 0; i < 64; i++) {
            float out0 = O_scaled_left[i] / sum;
            float out1 = O_scaled_right[i] / sum;
            
            int global_d_idx_0 = i;
            int global_d_idx_1 = 64 + i;
            
            if (global_d_idx_0 < 128) {
                O_ptr[global_q_base * 128 + global_d_idx_0] = __float2bfloat16(out0);
            }
            if (global_d_idx_1 < 128) {
                O_ptr[global_q_base * 128 + global_d_idx_1] = __float2bfloat16(out1);
            }
        }
        
        LSE_ptr[global_q_base + global_q_idx] = row_max_prev + logf(sum);
    }
    
    if (threadIdx.x == 0) {
        tmem_dealloc_fn_cta1(tmem_S, 64);
        tmem_dealloc_fn_cta1(tmem_O_left, 64);
        tmem_dealloc_fn_cta1(tmem_O_right, 64);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3); 
    
    if (S == 0) return;
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUtensorMap tma_Q, tma_K, tma_V;
    create_tma_4d_descriptor(&tma_Q, Q.data_ptr(), D, S, H, B, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B);
    create_tma_4d_descriptor(&tma_K, K.data_ptr(), D, S, H, B, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B);
    create_tma_4d_descriptor(&tma_V, V.data_ptr(), D, S, H, B, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B);
    
    uint32_t smem_size = 57344 + 256;
    dim3 grid((S + 63) / 64, H, B);
    dim3 block(128);
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, run_kernel, tma_Q, tma_K, tma_V, O.data_ptr(), LSE.data_ptr(), S, H, B));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);