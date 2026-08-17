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

namespace tvm_ffi_example_cuda {

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ uint32_t pack_bf16_fn(float a, float b) {
    __nv_bfloat16 a_bf16 = __float2bfloat16(a);
    __nv_bfloat16 b_bf16 = __float2bfloat16(b);
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&a_bf16)),
          "h"(*reinterpret_cast<uint16_t*>(&b_bf16)));
    return result;
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_store_32x32b_x1_val(uint32_t col, uint32_t val) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 %0, [%1];"
   :: "r"(val), "r"(col));
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

__device__ __forceinline__ uint64_t make_smem_desc(const void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16; 
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32; 
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

template<int A_MAJOR, int B_MAJOR>
__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (A_MAJOR << 15);   
    d |= (B_MAJOR << 16);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ void load_tile_64x64_vec_swizzled(
    const __nv_bfloat16* gmem, uint8_t* smem, int S, int thread_idx, int row_base, int col_base, int offset) {
    int row = row_base + thread_idx;
    if (row < 64) {
        const __nv_bfloat16* gmem_row = gmem + offset * 64 * S + row * 128 + col_base;
        uint4 val0 = *(const uint4*)(gmem_row + 0 * 8);
        uint4 val1 = *(const uint4*)(gmem_row + 1 * 8);
        uint4 val2 = *(const uint4*)(gmem_row + 2 * 8);
        uint4 val3 = *(const uint4*)(gmem_row + 3 * 8);
        
        uint32_t swizzled_offset0 = ((0 ^ (row % 8)) * 8 + 0) * 2 + (row * 64 * 2);
        *(uint4*)((char*)smem + swizzled_offset0) = val0;
        uint32_t swizzled_offset1 = ((1 ^ (row % 8)) * 8 + 0) * 2 + (row * 64 * 2);
        *(uint4*)((char*)smem + swizzled_offset1) = val1;
        uint32_t swizzled_offset2 = ((2 ^ (row % 8)) * 8 + 0) * 2 + (row * 64 * 2);
        *(uint4*)((char*)smem + swizzled_offset2) = val2;
        uint32_t swizzled_offset3 = ((3 ^ (row % 8)) * 8 + 0) * 2 + (row * 64 * 2);
        *(uint4*)((char*)smem + swizzled_offset3) = val3;
    }
}

__device__ __forceinline__ void load_smem_to_tmem_64x16(const uint8_t* smem, int thread_idx, int k, uint32_t col) {
    for (int i = 0; i < 4; ++i) {
        int r = thread_idx + i * 32;
        int c_start = k * 16 + i * 8;
        uint32_t swizzled_offset = (((c_start / 8) ^ (r % 8)) * 8 + (c_start % 8)) * 2 + (r * 64 * 2);
        uint4 val = *(const uint4*)((const char*)smem + swizzled_offset);
        tmem_store_32x32b_x1_val(col + k * 16 + i * 8 + (thread_idx % 32), val.x);
        tmem_store_32x32b_x1_val(col + k * 16 + i * 8 + (thread_idx % 32) + 32, val.y);
        tmem_store_32x32b_x1_val(col + k * 16 + i * 8 + (thread_idx % 32) + 64, val.z);
        tmem_store_32x32b_x1_val(col + k * 16 + i * 8 + (thread_idx % 32) + 96, val.w);
    }
}

__device__ __forceinline__ void save_P_transposed(const uint32_t* tmem_base, uint32_t smem_addr) {
    int tid = threadIdx.x;
    uint32_t* smem = (uint32_t*)smem_addr;
    for (int c = 0; c < 64; c += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(c));
        uint32_t swizzled_offset = (((c / 8) ^ (tid % 8)) * 8 + (c % 8)) * 2 + (tid * 64 * 2);
        *(uint4*)((char*)smem + swizzled_offset) = *(uint4*)&r0;
    }
}

__device__ __forceinline__ void compute_softmax_causal(
    const uint32_t* S_tmem, uint32_t* P_tmem, float* s_D, int qb_idx, int kb_idx, const float* L, int b_idx, int h_idx, int S) {
    
    float scale_log2e = (1.0f / sqrt(128.0f)) * 1.4426950408889634f; 
    
    for (int c = 0; c < 64; c += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(c));
        tmem_load_fence();
        
        float S0 = __uint_as_float(r0) * scale_log2e;
        float S1 = __uint_as_float(r1) * scale_log2e;
        float S2 = __uint_as_float(r2) * scale_log2e;
        float S3 = __uint_as_float(r3) * scale_log2e;
        
        int q_idx = threadIdx.x;
        int k_idx0 = c;
        int k_idx1 = c + 1;
        int k_idx2 = c + 2;
        int k_idx3 = c + 3;
        
        int q_offset = b_idx * H * S + h_idx * S + qb_idx * 64 + q_idx;
        float lse0 = L[q_offset];
        float lse1 = L[q_offset];
        float lse2 = L[q_offset];
        float lse3 = L[q_offset];
        
        float p0 = fast_exp2f_fn(S0 - lse0);
        float p1 = fast_exp2f_fn(S1 - lse1);
        float p2 = fast_exp2f_fn(S2 - lse2);
        float p3 = fast_exp2f_fn(S3 - lse3);
        
        if (qb_idx * 64 + q_idx >= S || kb_idx * 64 + k_idx0 >= S || qb_idx * 64 + q_idx < kb_idx * 64 + k_idx0) p0 = 0;
        if (qb_idx * 64 + q_idx >= S || kb_idx * 64 + k_idx1 >= S || qb_idx * 64 + q_idx < kb_idx * 64 + k_idx1) p1 = 0;
        if (qb_idx * 64 + q_idx >= S || kb_idx * 64 + k_idx2 >= S || qb_idx * 64 + q_idx < kb_idx * 64 + k_idx2) p2 = 0;
        if (qb_idx * 64 + q_idx >= S || kb_idx * 64 + k_idx3 >= S || qb_idx * 64 + q_idx < kb_idx * 64 + k_idx3) p3 = 0;
        
        *(float*)&r0 = p0;
        *(float*)&r1 = p1;
        *(float*)&r2 = p2;
        *(float*)&r3 = p3;
        
        tmem_store_32x32b_x1_val(c + (threadIdx.x % 32), r0);
        tmem_store_32x32b_x1_val(c + (threadIdx.x % 32) + 32, r1);
        tmem_store_32x32b_x1_val(c + (threadIdx.x % 32) + 64, r2);
        tmem_store_32x32b_x1_val(c + (threadIdx.x % 32) + 96, r3);
    }
    
    for (int c = 0; c < 64; c += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(c));
        tmem_load_fence();
        
        float P0 = __uint_as_float(r0);
        float P1 = __uint_as_float(r1);
        float P2 = __uint_as_float(r2);
        float P3 = __uint_as_float(r3);
        
        int q_idx = threadIdx.x;
        int k_idx0 = c;
        int k_idx1 = c + 1;
        int k_idx2 = c + 2;
        int k_idx3 = c + 3;
        
        uint32_t swizzled_offset0 = (((k_idx0 / 8) ^ (q_idx % 8)) * 8 + (k_idx0 % 8)) * 2 + (q_idx * 64 * 2);
        uint32_t packed0 = pack_bf16_fn(P0, P1);
        *(uint32_t*)((char*)s_PT + swizzled_offset0) = packed0;
        
        uint32_t swizzled_offset1 = (((k_idx2 / 8) ^ (q_idx % 8)) * 8 + (k_idx2 % 8)) * 2 + (q_idx * 64 * 2);
        uint32_t packed1 = pack_bf16_fn(P2, P3);
        *(uint32_t*)((char*)s_PT + swizzled_offset1) = packed1;
    }
}

__device__ __forceinline__ void compute_dS_causal(
    const uint32_t* P_tmem, const uint32_t* dP_tmem, float* s_D, int qb_idx, int kb_idx, uint32_t* s_dS, uint32_t* s_dST) {
    
    for (int c = 0; c < 64; c += 4) {
        uint32_t r0_P, r1_P, r2_P, r3_P;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0_P),"=r"(r1_P),"=r"(r2_P),"=r"(r3_P) : "r"(c));
        tmem_load_fence();
        
        uint32_t r0_dP, r1_dP, r2_dP, r3_dP;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0_dP),"=r"(r1_dP),"=r"(r2_dP),"=r"(r3_dP) : "r"(c));
        tmem_load_fence();
        
        float P0 = __uint_as_float(r0_P);
        float P1 = __uint_as_float(r1_P);
        float P2 = __uint_as_float(r2_P);
        float P3 = __uint_as_float(r3_P);
        
        float dP0 = __uint_as_float(r0_dP);
        float dP1 = __uint_as_float(r1_dP);
        float dP2 = __uint_as_float(r2_dP);
        float dP3 = __uint_as_float(r3_dP);
        
        int q_idx = threadIdx.x;
        int k_idx0 = c;
        int k_idx1 = c + 1;
        int k_idx2 = c + 2;
        int k_idx3 = c + 3;
        
        float D0 = s_D[q_idx];
        float D1 = s_D[q_idx];
        float D2 = s_D[q_idx];
        float D3 = s_D[q_idx];
        
        float ds0 = P0 * (dP0 - D0);
        float ds1 = P1 * (dP1 - D1);
        float ds2 = P2 * (dP2 - D2);
        float ds3 = P3 * (dP3 - D3);
        
        if (qb_idx * 64 + q_idx >= S || kb_idx * 64 + k_idx0 >= S || qb_idx * 64 + q_idx < kb_idx * 64 + k_idx0) ds0 = 0;
        if (qb_idx * 64 + q_idx >= S || kb_idx * 64 + k_idx1 >= S || qb_idx * 64 + q_idx < kb_idx * 64 + k_idx1) ds1 = 0;
        if (qb_idx * 64 + q_idx >= S || kb_idx * 64 + k_idx2 >= S || qb_idx * 64 + q_idx < kb_idx * 64 + k_idx2) ds2 = 0;
        if (qb_idx * 64 + q_idx >= S || kb_idx * 64 + k_idx3 >= S || qb_idx * 64 + q_idx < kb_idx * 64 + k_idx3) ds3 = 0;
        
        uint32_t pr0, pr1, pr2, pr3;
        *(float*)&pr0 = ds0;
        *(float*)&pr1 = ds1;
        *(float*)&pr2 = ds2;
        *(float*)&pr3 = ds3;
        
        tmem_store_32x32b_x1_val(c + (threadIdx.x % 32), pr0);
        tmem_store_32x32b_x1_val(c + (threadIdx.x % 32) + 32, pr1);
        tmem_store_32x32b_x1_val(c + (threadIdx.x % 32) + 64, pr2);
        tmem_store_32x32b_x1_val(c + (threadIdx.x % 32) + 96, pr3);
        
        uint32_t swizzled_offset0 = (((k_idx0 / 8) ^ (q_idx % 8)) * 8 + (k_idx0 % 8)) * 2 + (q_idx * 64 * 2);
        uint32_t packed0 = pack_bf16_fn(ds0, ds1);
        *(uint32_t*)((char*)s_dS + swizzled_offset0) = packed0;
        *(uint32_t*)((char*)s_dST + swizzled_offset0) = packed0;
        
        uint32_t swizzled_offset1 = (((k_idx2 / 8) ^ (q_idx % 8)) * 8 + (k_idx2 % 8)) * 2 + (q_idx * 64 * 2);
        uint32_t packed1 = pack_bf16_fn(ds2, ds3);
        *(uint32_t*)((char*)s_dS + swizzled_offset1) = packed1;
        *(uint32_t*)((char*)s_dST + swizzled_offset1) = packed1;
    }
}

__device__ __forceinline__ void store_dQ_atomic_add(
    const uint32_t* tmem_base, __nv_bfloat16* dQ, int qb_idx, int b_idx, int h_idx, int S) {
    for (int c = 0; c < 64; c += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(c));
        tmem_load_fence();
        
        int q_idx = threadIdx.x;
        int k_idx0 = c;
        int k_idx1 = c + 1;
        int k_idx2 = c + 2;
        int k_idx3 = c + 3;
        
        float f0 = __uint_as_float(r0);
        float f1 = __uint_as_float(r1);
        float f2 = __uint_as_float(r2);
        float f3 = __uint_as_float(r3);
        
        __nv_bfloat16 bf0 = __float2bfloat16(f0);
        __nv_bfloat16 bf1 = __float2bfloat16(f1);
        __nv_bfloat16 bf2 = __float2bfloat16(f2);
        __nv_bfloat16 bf3 = __float2bfloat16(f3);
        
        int q_offset0 = b_idx * H * S * 128 + h_idx * S * 128 + (qb_idx * 64 + q_idx) * 128 + k_idx0;
        int q_offset1 = b_idx * H * S * 128 + h_idx * S * 128 + (qb_idx * 64 + q_idx) * 128 + k_idx1;
        int q_offset2 = b_idx * H * S * 128 + h_idx * S * 128 + (qb_idx * 64 + q_idx) * 128 + k_idx2;
        int q_offset3 = b_idx * H * S * 128 + h_idx * S * 128 + (qb_idx * 64 + q_idx) * 128 + k_idx3;
        
        if (qb_idx * 64 + q_idx < S && k_idx0 < 64) dQ[q_offset0] = bf0;
        if (qb_idx * 64 + q_idx < S && k_idx1 < 64) dQ[q_offset1] = bf1;
        if (qb_idx * 64 + q_idx < S && k_idx2 < 64) dQ[q_offset2] = bf2;
        if (qb_idx * 64 + q_idx < S && k_idx3 < 64) dQ[q_offset3] = bf3;
    }
}

__device__ __forceinline__ void store_dQ_half_atomic_add(
    const uint32_t* tmem_base, __nv_bfloat16* dQ, int qb_idx, int b_idx, int h_idx, int S, int offset) {
    for (int c = 0; c < 64; c += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(c));
        tmem_load_fence();
        
        int q_idx = threadIdx.x;
        int k_idx0 = c;
        int k_idx1 = c + 1;
        int k_idx2 = c + 2;
        int k_idx3 = c + 3;
        
        float f0 = __uint_as_float(r0);
        float f1 = __uint_as_float(r1);
        float f2 = __uint_as_float(r2);
        float f3 = __uint_as_float(r3);
        
        __nv_bfloat16 bf0 = __float2bfloat16(f0);
        __nv_bfloat16 bf1 = __float2bfloat16(f1);
        __nv_bfloat16 bf2 = __float2bfloat16(f2);
        __nv_bfloat16 bf3 = __float2bfloat16(f3);
        
        int q_offset0 = b_idx * H * S * 128 + h_idx * S * 128 + (qb_idx * 64 + q_idx) * 128 + offset * 64 + k_idx0;
        int q_offset1 = b_idx * H * S * 128 + h_idx * S * 128 + (qb_idx * 64 + q_idx) * 128 + offset * 64 + k_idx1;
        int q_offset2 = b_idx * H * S * 128 + h_idx * S * 128 + (qb_idx * 64 + q_idx) * 128 + offset * 64 + k_idx2;
        int q_offset3 = b_idx * H * S * 128 + h_idx * S * 128 + (qb_idx * 64 + q_idx) * 128 + offset * 64 + k_idx3;
        
        if (qb_idx * 64 + q_idx < S && k_idx0 < 64) dQ[q_offset0] = bf0;
        if (qb_idx * 64 + q_idx < S && k_idx1 < 64) dQ[q_offset1] = bf1;
        if (qb_idx * 64 + q_idx < S && k_idx2 < 64) dQ[q_offset2] = bf2;
        if (qb_idx * 64 + q_idx < S && k_idx3 < 64) dQ[q_offset3] = bf3;
    }
}

extern __shared__ __align__(128) uint8_t smem_pool[];

__global__ __launch_bounds__(128, 1) void bwd_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* O, const __nv_bfloat16* dO, const float* L,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    int S, int H) 
{
    int qb_idx = blockIdx.x % (S / 64);
    int h_idx = (blockIdx.x / (S / 64)) % H;
    int b_idx = blockIdx.x / (S / 64 * H);

    uint32_t tmem_base;
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_base, 512); 
    }
    __syncthreads();

    uint32_t col_S = tmem_base;
    uint32_t col_dP = tmem_base + 64;
    uint32_t col_dV = tmem_base + 128;
    uint32_t col_dK = tmem_base + 192;
    uint32_t col_dQ = tmem_base + 256;
    uint32_t col_dS = tmem_base + 320;

    extern __shared__ __align__(128) uint8_t smem_flex[];
    uint8_t* s_Q0 = smem_flex;                      
    uint8_t* s_Q1 = smem_flex + 8192;               
    uint8_t* s_K  = smem_flex + 16384;              
    uint8_t* s_V  = smem_flex + 24576;               
    uint8_t* s_dO = smem_flex + 32768;               
    uint8_t* s_PT = smem_flex + 40960;               
    uint8_t* s_DT = smem_flex + 49152;                
    uint8_t* s_dST= smem_flex + 57344;                
    uint8_t* s_dS = smem_flex + 65536;                
    uint8_t* s_dP = smem_flex + 73728;                
    float* s_D    = (float*)(smem_flex + 81920);     
    uint64_t* mbar = (uint64_t*)(smem_flex + 82176); 

    int thread_idx = threadIdx.x % 64;
    int offset = threadIdx.x / 64;
    
    if (threadIdx.x < 64) {
        float d_val = 0;
        int q_offset = b_idx * H * S + h_idx * S + qb_idx * 64 + threadIdx.x;
        if (threadIdx.x < 64) {
            d_val = 0; 
            for(int i=0; i<128; i++) {
                d_val += __bf162float(O[q_offset * 128 + i]) * __bf162float(dO[q_offset * 128 + i]);
            }
        }
        s_D[threadIdx.x] = d_val;
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    load_tile_64x64_vec_swizzled(Q, s_Q0, S, thread_idx, qb_idx * 64, 0, offset);
    load_tile_64x64_vec_swizzled(dO, s_dO, S, thread_idx, qb_idx * 64, 0, offset);
    
    uint32_t desc_dQ0 = make_smem_desc(s_dQ, 0, 1024);
    uint32_t desc_dQ1 = make_smem_desc(s_dQ, 0, 1024);
    umma_f16_cg1_fn(col_dQ, desc_dQ0, desc_dQ1, make_instr_desc_fn<0, 1>(64, 64), 0);
    umma_commit_1sm_fn(mbar);
    mbarrier_wait_fn(mbar, 0);
    __syncthreads();
    
    uint32_t phase = 0;

    for (int kb = 0; kb <= qb_idx; kb++) {
        load_tile_64x64_vec_swizzled(K, s_K, S, thread_idx, kb * 64, 0, offset);
        load_tile_64x64_vec_swizzled(V, s_V, S, thread_idx, kb * 64, 0, offset);
        __syncthreads();
        
        mbarrier_arrive_and_expect_tx_fn(mbar, 0);
        
        uint32_t desc_Q0 = make_smem_desc(s_Q0, 0, 1024);
        uint32_t desc_K0 = make_smem_desc(s_K, 0, 1024);
        umma_f16_cg1_fn(col_S, desc_Q0, desc_K0, make_instr_desc_fn<0, 0>(64, 64), 0);
        
        uint32_t desc_Q1 = make_smem_desc(s_Q1, 0, 1024);
        uint32_t desc_K1 = make_smem_desc(s_K + 8192, 0, 1024);
        umma_f16_cg1_fn(col_S, desc_Q1, desc_K1, make_instr_desc_fn<0, 0>(64, 64), 1);
        
        uint32_t desc_V0 = make_smem_desc(s_V, 0, 1024);
        uint32_t desc_dO0 = make_smem_desc(s_dO, 0, 1024);
        umma_f16_cg1_fn(col_dP, desc_V0, desc_dO0, make_instr_desc_fn<0, 1>(64, 64), 0);
        
        uint32_t desc_V1 = make_smem_desc(s_V + 8192, 0, 1024);
        uint32_t desc_dO1 = make_smem_desc(s_dO + 8192, 0, 1024);
        umma_f16_cg1_fn(col_dP, desc_V1, desc_dO1, make_instr_desc_fn<0, 1>(64, 64), 1);
        
        umma_commit_1sm_fn(mbar);
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        __syncthreads();
        
        compute_softmax_causal((uint32_t*)col_S, (uint32_t*)col_S, s_D, qb_idx, kb, L, b_idx, h_idx, S);
        compute_dS_causal((uint32_t*)col_S, (uint32_t*)col_dP, s_D, qb_idx, kb, (uint32_t*)s_dS, (uint32_t*)s_dST);
        
        uint32_t desc_PT = make_smem_desc(s_PT, 8192, 1024);
        uint32_t desc_dO0_v = make_smem_desc(s_dO, 0, 1024);
        umma_f16_cg1_fn(col_dV, desc_PT, desc_dO0_v, make_instr_desc_fn<1, 0>(64, 64), 0);
        
        uint32_t desc_dO1_v = make_smem_desc(s_dO + 8192, 0, 1024);
        umma_f16_cg1_fn(col_dV, desc_PT, desc_dO1_v, make_instr_desc_fn<1, 0>(64, 64), 1);
        
        uint32_t desc_dST = make_smem_desc(s_dST, 8192, 1024);
        uint32_t desc_Q0_k = make_smem_desc(s_Q0, 0, 1024);
        umma_f16_cg1_fn(col_dK, desc_dST, desc_Q0_k, make_instr_desc_fn<1, 1>(64, 64), 0);
        
        uint32_t desc_Q1_k = make_smem_desc(s_Q1, 0, 1024);
        umma_f16_cg1_fn(col_dK, desc_dST, desc_Q1_k, make_instr_desc_fn<1, 1>(64, 64), 1);
        
        uint32_t desc_dS = make_smem_desc(s_dS, 0, 1024);
        uint32_t desc_K0_q = make_smem_desc(s_K, 8192, 1024);
        umma_f16_cg1_fn(col_dQ, desc_dS, desc_K0_q, make_instr_desc_fn<0, 1>(64, 64), 1);
        
        uint32_t desc_K1_q = make_smem_desc(s_K + 8192, 8192, 1024);
        umma_f16_cg1_fn(col_dQ, desc_dS, desc_K1_q, make_instr_desc_fn<0, 1>(64, 64), 1);
        
        umma_commit_1sm_fn(mbar);
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        __syncthreads();
        
        store_dQ_half_atomic_add((uint32_t*)col_dV, dV, kb, b_idx, h_idx, S, 0);
        store_dQ_half_atomic_add((uint32_t*)col_dV, dV, kb, b_idx, h_idx, S, 1);
        store_dQ_half_atomic_add((uint32_t*)col_dK, dK, kb, b_idx, h_idx, S, 0);
        store_dQ_half_atomic_add((uint32_t*)col_dK, dK, kb, b_idx, h_idx, S, 1);
    }
    
    store_dQ_half_atomic_add((uint32_t*)col_dQ, dQ, qb_idx, b_idx, h_idx, S, 0);
    store_dQ_half_atomic_add((uint32_t*)col_dQ, dQ, qb_idx, b_idx, h_idx, S, 1);
    
    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_base, 512);
    }
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

    dim3 grid(B * H * (S + 63) / 64, 1, 1);
    dim3 block(128, 1, 1);

    CUDA_CHECK(cudaFuncSetAttribute(
        bwd_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        85000));

    bwd_kernel<<<grid, block, 85000, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(O.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        S, H);
        
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda