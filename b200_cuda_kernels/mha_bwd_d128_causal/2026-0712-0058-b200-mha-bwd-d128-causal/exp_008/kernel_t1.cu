#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_tmem.h>
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

__device__ __forceinline__ void umma_f16_cg2_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_2sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(a), "h"((uint16_t)0x3));
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

__device__ __forceinline__ void to_float_vec(const uint8_t* smem, int r, float4& vals) {
    int c_chunk = 0;
    int r_chunk = r % 8;
    int swizzled_c_chunk = c_chunk ^ r_chunk;
    int swizzled_c = swizzled_c_chunk * 8;
    int swizzled_idx = r * 64 + swizzled_c;
    float4 val = *(const float4*)((const char*)smem + swizzled_idx * 2);
    vals.x = __bfloat162float(*(const __nv_bfloat16*)&val.x);
    vals.y = __bfloat162float(*(const __nv_bfloat16*)&val.y);
    vals.z = __bfloat162float(*(const __nv_bfloat16*)&val.z);
    vals.w = __bfloat162float(*(const __nv_bfloat16*)&val.w);
}

__device__ __forceinline__ void compute_D_from_O_dO_vec(
    const uint8_t* s_O0, const uint8_t* s_dO0, 
    const uint8_t* s_O1, const uint8_t* s_dO1, 
    float* s_D, int num_rows) 
{
    for (int r = threadIdx.x; r < num_rows; r += blockDim.x) {
        float d_val = 0;
        float4 v0, v1, v2, v3;
        to_float_vec(s_O0, r, v0);
        to_float_vec(s_dO0, r, v1);
        d_val += v0.x * v1.x + v0.y * v1.y + v0.z * v1.z + v0.w * v1.w;
        
        to_float_vec(s_O1, r, v2);
        to_float_vec(s_dO1, r, v3);
        d_val += v2.x * v3.x + v2.y * v3.y + v2.z * v3.z + v2.w * v3.w;
        
        s_D[r] = d_val;
    }
}

__device__ __forceinline__ void compute_P_from_S(
    const uint32_t* S_tmem, uint8_t* s_PT, float* s_D, int qb_idx, int kb_idx, const float* L, int b_idx, int h_idx, int S) {
    
    float scale_log2e = (1.0f / sqrt(128.0f)) * 1.4426950408889634f; 
    
    for (int c = 0; c < 64; c += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(c));
        tmem_load_fence_fn();
        
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
        float lse0 = L[q_offset] * 1.4426950408889634f;
        float lse1 = L[q_offset] * 1.4426950408889634f;
        float lse2 = L[q_offset] * 1.4426950408889634f;
        float lse3 = L[q_offset] * 1.4426950408889634f;
        
        float p0 = fast_exp2f_fn(S0 - lse0);
        float p1 = fast_exp2f_fn(S1 - lse1);
        float p2 = fast_exp2f_fn(S2 - lse2);
        float p3 = fast_exp2f_fn(S3 - lse3);
        
        int global_q_idx = qb_idx * 64 + q_idx;
        int global_k_idx0 = kb_idx * 64 + k_idx0;
        int global_k_idx1 = kb_idx * 64 + k_idx1;
        int global_k_idx2 = kb_idx * 64 + k_idx2;
        int global_k_idx3 = kb_idx * 64 + k_idx3;
        
        if (global_q_idx >= S || global_k_idx0 >= S || global_q_idx < global_k_idx0) p0 = 0;
        if (global_q_idx >= S || global_k_idx1 >= S || global_q_idx < global_k_idx1) p1 = 0;
        if (global_q_idx >= S || global_k_idx2 >= S || global_q_idx < global_k_idx2) p2 = 0;
        if (global_q_idx >= S || global_k_idx3 >= S || global_q_idx < global_k_idx3) p3 = 0;
        
        uint32_t packed0 = pack_bf16_fn(p0, p1);
        uint32_t packed1 = pack_bf16_fn(p2, p3);
        
        int c_chunk0 = k_idx0 / 8;
        int r_chunk0 = q_idx % 8;
        int swizzled_c_chunk0 = c_chunk0 ^ r_chunk0;
        int swizzled_c0 = swizzled_c_chunk0 * 8 + (k_idx0 % 8);
        int swizzled_idx0 = q_idx * 64 + swizzled_c0;
        *(uint32_t*)((char*)s_PT + swizzled_idx0 * 2) = packed0;
        
        int c_chunk1 = k_idx2 / 8;
        int r_chunk1 = q_idx % 8;
        int swizzled_c_chunk1 = c_chunk1 ^ r_chunk1;
        int swizzled_c1 = swizzled_c_chunk1 * 8 + (k_idx2 % 8);
        int swizzled_idx1 = q_idx * 64 + swizzled_c1;
        *(uint32_t*)((char*)s_PT + swizzled_idx1 * 2) = packed1;
    }
}

__device__ __forceinline__ void compute_dS_from_P_dP(
    const uint8_t* s_PT, const uint32_t* dP_tmem, float* s_D, int qb_idx, int kb_idx, uint8_t* s_dS, uint8_t* s_dST) {
    
    for (int c = 0; c < 64; c += 4) {
        uint32_t r0_dP, r1_dP, r2_dP, r3_dP;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0_dP),"=r"(r1_dP),"=r"(r2_dP),"=r"(r3_dP) : "r"(c));
        tmem_load_fence_fn();
        
        int q_idx = threadIdx.x;
        int k_idx0 = c;
        int k_idx1 = c + 1;
        int k_idx2 = c + 2;
        int k_idx3 = c + 3;
        
        int c_chunk0 = k_idx0 / 8;
        int r_chunk0 = q_idx % 8;
        int swizzled_c_chunk0 = c_chunk0 ^ r_chunk0;
        int swizzled_c0 = swizzled_c_chunk0 * 8 + (k_idx0 % 8);
        int swizzled_idx0 = q_idx * 64 + swizzled_c0;
        uint32_t packed0 = *(const uint32_t*)((const char*)s_PT + swizzled_idx0 * 2);
        
        int c_chunk1 = k_idx2 / 8;
        int r_chunk1 = q_idx % 8;
        int swizzled_c_chunk1 = c_chunk1 ^ r_chunk1;
        int swizzled_c1 = swizzled_c_chunk1 * 8 + (k_idx2 % 8);
        int swizzled_idx1 = q_idx * 64 + swizzled_c1;
        uint32_t packed1 = *(const uint32_t*)((const char*)s_PT + swizzled_idx1 * 2);
        
        __nv_bfloat16 bf_p0 = *reinterpret_cast<__nv_bfloat16*>(&packed0);
        __nv_bfloat16 bf_p1 = *reinterpret_cast<__nv_bfloat16*>(&packed0 + 1);
        __nv_bfloat16 bf_p2 = *reinterpret_cast<__nv_bfloat16*>(&packed1);
        __nv_bfloat16 bf_p3 = *reinterpret_cast<__nv_bfloat16*>(&packed1 + 1);
        
        float P0 = __bfloat162float(bf_p0);
        float P1 = __bfloat162float(bf_p1);
        float P2 = __bfloat162float(bf_p2);
        float P3 = __bfloat162float(bf_p3);
        
        float dP0 = __uint_as_float(r0_dP);
        float dP1 = __uint_as_float(r1_dP);
        float dP2 = __uint_as_float(r2_dP);
        float dP3 = __uint_as_float(r3_dP);
        
        float D0 = s_D[q_idx];
        float D1 = s_D[q_idx];
        float D2 = s_D[q_idx];
        float D3 = s_D[q_idx];
        
        float ds0 = P0 * (dP0 - D0);
        float ds1 = P1 * (dP1 - D1);
        float ds2 = P2 * (dP2 - D2);
        float ds3 = P3 * (dP3 - D3);
        
        int global_q_idx = qb_idx * 64 + q_idx;
        int global_k_idx0 = kb_idx * 64 + k_idx0;
        int global_k_idx1 = kb_idx * 64 + k_idx1;
        int global_k_idx2 = kb_idx * 64 + k_idx2;
        int global_k_idx3 = kb_idx * 64 + k_idx3;
        
        if (global_q_idx >= S || global_k_idx0 >= S || global_q_idx < global_k_idx0) ds0 = 0;
        if (global_q_idx >= S || global_k_idx1 >= S || global_q_idx < global_k_idx1) ds1 = 0;
        if (global_q_idx >= S || global_k_idx2 >= S || global_q_idx < global_k_idx2) ds2 = 0;
        if (global_q_idx >= S || global_k_idx3 >= S || global_q_idx < global_k_idx3) ds3 = 0;
        
        uint32_t pr0 = pack_bf16_fn(ds0, ds1);
        uint32_t pr1 = pack_bf16_fn(ds2, ds3);
        
        *(uint32_t*)((char*)s_dS + swizzled_idx0 * 2) = pr0;
        *(uint32_t*)((char*)s_dST + swizzled_idx0 * 2) = pr0;
        
        *(uint32_t*)((char*)s_dS + swizzled_idx1 * 2) = pr1;
        *(uint32_t*)((char*)s_dST + swizzled_idx1 * 2) = pr1;
    }
}

__device__ __forceinline__ void store_tmem_to_gmem(
    __nv_bfloat16* gmem, uint32_t tmem_col_base,
    int row_base, int col_base, int stride_row, int S) {
    int tid = threadIdx.x;
    for (int col = 0; col < 64; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_col_base + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        int row = tid;
        int g_row = row_base + row;
        int g_col = col_base + col;
        
        if (g_row < S && g_col + 3 < 128) {
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            
            __nv_bfloat16 bf0 = __float2bfloat16(f0);
            __nv_bfloat16 bf1 = __float2bfloat16(f1);
            __nv_bfloat16 bf2 = __float2bfloat16(f2);
            __nv_bfloat16 bf3 = __float2bfloat16(f3);
            
            uint4 val;
            val.x = r0; val.y = r1; val.z = r2; val.w = r3;
            *reinterpret_cast<uint4*>(gmem + (uint64_t)row * stride_row + g_col) = val;
        }
    }
}

extern __shared__ __align__(128) uint8_t smem_pool[];

__global__ __launch_bounds__(128, 1) void bwd_kernel_dvk(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_dV,
    const __grid_constant__ CUtensorMap tma_dK,
    const float* L, __nv_bfloat16* dV, __nv_bfloat16* dK, int S, int H) 
{
    int kb_idx = blockIdx.x % (S / 64);
    int h_idx = (blockIdx.x / (S / 64)) % H;
    int b_idx = blockIdx.x / (S / 64 * H);

    uint32_t tmem_base;
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_base, 512);
    }
    __syncthreads();

    uint32_t col_S = tmem_base;
    uint32_t col_dP = tmem_base + 64;
    uint32_t col_dV0 = tmem_base + 128;
    uint32_t col_dV1 = tmem_base + 192;
    uint32_t col_dK0 = tmem_base + 256;
    uint32_t col_dK1 = tmem_base + 320;

    extern __shared__ __align__(128) uint8_t smem_flex[];
    uint8_t* s_Q0 = smem_flex;                      
    uint8_t* s_Q1 = smem_flex + 8192;               
    uint8_t* s_K0 = smem_flex + 16384;              
    uint8_t* s_K1 = smem_flex + 24576;               
    uint8_t* s_V0 = smem_flex + 32768;                
    uint8_t* s_V1 = smem_flex + 40960;                
    uint8_t* s_O0 = smem_flex + 49152;                
    uint8_t* s_O1 = smem_flex + 57344;                
    uint8_t* s_dO0= smem_flex + 65536;                
    uint8_t* s_dO1= smem_flex + 73728;                
    uint8_t* s_PT = smem_flex + 81920;                
    uint8_t* s_dST= smem_flex + 90112;                
    float* s_D    = (float*)(smem_flex + 98304);     
    uint64_t* mbar = (uint64_t*)(smem_flex + 98560); 

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar, 16384); 
        tma_load_2d_fn(&tma_K, mbar, s_K0, 0, b_idx * H * S + h_idx * S + kb_idx * 64);
        tma_load_2d_fn(&tma_K, mbar, s_K1, 64, b_idx * H * S + h_idx * S + kb_idx * 64);
        tma_load_2d_fn(&tma_V, mbar, s_V0, 0, b_idx * H * S + h_idx * S + kb_idx * 64);
        tma_load_2d_fn(&tma_V, mbar, s_V1, 64, b_idx * H * S + h_idx * S + kb_idx * 64);
    }
    mbarrier_wait_fn(mbar, 0);
    __syncthreads();

    uint32_t phase = 0;

    for (int qb = kb_idx; qb < S / 64; qb++) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar, 16384 * 3); 
            tma_load_2d_fn(&tma_Q, mbar, s_Q0, 0, b_idx * H * S + h_idx * S + qb * 64);
            tma_load_2d_fn(&tma_Q, mbar, s_Q1, 64, b_idx * H * S + h_idx * S + qb * 64);
            tma_load_2d_fn(&tma_O, mbar, s_O0, 0, b_idx * H * S + h_idx * S + qb * 64);
            tma_load_2d_fn(&tma_O, mbar, s_O1, 64, b_idx * H * S + h_idx * S + qb * 64);
            tma_load_2d_fn(&tma_dO, mbar, s_dO0, 0, b_idx * H * S + h_idx * S + qb * 64);
            tma_load_2d_fn(&tma_dO, mbar, s_dO1, 64, b_idx * H * S + h_idx * S + qb * 64);
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        __syncthreads();
        
        compute_D_from_O_dO_vec(s_O0, s_dO0, s_O1, s_dO1, s_D, 64);
        __syncthreads();
        
        mbarrier_arrive_and_expect_tx_fn(mbar, 0);
        
        for (int k=0; k<4; k++) {
            uint32_t desc_K = make_smem_desc((char*)s_K0 + k * 128, 0, 1024);
            uint32_t desc_Q = make_smem_desc((char*)s_Q0 + k * 128, 0, 1024);
            umma_f16_cg2_fn(col_S, desc_K, desc_Q, make_instr_desc_fn<0, 0>(64, 64), k==0 ? 0 : 1);
        }
        for (int k=0; k<4; k++) {
            uint32_t desc_K = make_smem_desc((char*)s_K1 + k * 128, 0, 1024);
            uint32_t desc_Q = make_smem_desc((char*)s_Q1 + k * 128, 0, 1024);
            umma_f16_cg2_fn(col_S, desc_K, desc_Q, make_instr_desc_fn<0, 0>(64, 64), 1);
        }
        
        for (int k=0; k<4; k++) {
            uint32_t desc_V = make_smem_desc((char*)s_V0 + k * 128, 0, 1024);
            uint32_t desc_dO = make_smem_desc((char*)s_dO0 + k * 128, 0, 1024);
            umma_f16_cg2_fn(col_dP, desc_V, desc_dO, make_instr_desc_fn<0, 0>(64, 64), k==0 ? 0 : 1);
        }
        for (int k=0; k<4; k++) {
            uint32_t desc_V = make_smem_desc((char*)s_V1 + k * 128, 0, 1024);
            uint32_t desc_dO = make_smem_desc((char*)s_dO1 + k * 128, 0, 1024);
            umma_f16_cg2_fn(col_dP, desc_V, desc_dO, make_instr_desc_fn<0, 0>(64, 64), 1);
        }
        
        umma_commit_2sm_fn(mbar);
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        __syncthreads();
        
        compute_P_from_S((uint32_t*)col_S, s_PT, s_D, qb, kb_idx, L, b_idx, h_idx, S);
        compute_dS_from_P_dP(s_PT, (uint32_t*)col_dP, s_D, qb, kb_idx, s_Q0, s_Q1);
        __syncthreads();
        
        mbarrier_arrive_and_expect_tx_fn(mbar, 0);
        
        for (int k=0; k<4; k++) {
            uint32_t desc_PT = make_smem_desc((char*)s_PT + k * 128, 1024, 0);
            uint32_t desc_dO = make_smem_desc((char*)s_dO0 + k * 128, 0, 1024);
            umma_f16_cg2_fn(col_dV0, desc_PT, desc_dO, make_instr_desc_fn<1, 0>(64, 64), 1);
        }
        for (int k=0; k<4; k++) {
            uint32_t desc_PT = make_smem_desc((char*)s_PT + k * 128, 1024, 0);
            uint32_t desc_dO = make_smem_desc((char*)s_dO1 + k * 128, 0, 1024);
            umma_f16_cg2_fn(col_dV1, desc_PT, desc_dO, make_instr_desc_fn<1, 0>(64, 64), 1);
        }
        
        for (int k=0; k<4; k++) {
            uint32_t desc_dST = make_smem_desc((char*)s_dST + k * 128, 1024, 0);
            uint32_t desc_Q = make_smem_desc((char*)s_Q0 + k * 128, 0, 1024);
            umma_f16_cg2_fn(col_dK0, desc_dST, desc_Q, make_instr_desc_fn<1, 0>(64, 64), 1);
        }
        for (int k=0; k<4; k++) {
            uint32_t desc_dST = make_smem_desc((char*)s_dST + k * 128, 1024, 0);
            uint32_t desc_Q = make_smem_desc((char*)s_Q1 + k * 128, 0, 1024);
            umma_f16_cg2_fn(col_dK1, desc_dST, desc_Q, make_instr_desc_fn<1, 0>(64, 64), 1);
        }
        
        umma_commit_2sm_fn(mbar);
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        __syncthreads();
    }
    
    __syncthreads();
    store_tmem_to_gmem(dV, tmem_base + col_dV0, b_idx * H * S + h_idx * S + kb_idx * 64, 0, 128, S);
    store_tmem_to_gmem(dV, tmem_base + col_dV1, b_idx * H * S + h_idx * S + kb_idx * 64, 64, 128, S);
    store_tmem_to_gmem(dK, tmem_base + col_dK0, b_idx * H * S + h_idx * S + kb_idx * 64, 0, 128, S);
    store_tmem_to_gmem(dK, tmem_base + col_dK1, b_idx * H * S + h_idx * S + kb_idx * 64, 64, 128, S);
    
    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_base, 512);
    }
}

__global__ __launch_bounds__(128, 1) void bwd_kernel_dq(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_dQ,
    const float* L, __nv_bfloat16* dQ, int S, int H) 
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
    uint32_t col_dQ0 = tmem_base + 128;
    uint32_t col_dQ1 = tmem_base + 192;

    extern __shared__ __align__(128) uint8_t smem_flex[];
    uint8_t* s_Q0 = smem_flex;                      
    uint8_t* s_Q1 = smem_flex + 8192;               
    uint8_t* s_K0 = smem_flex + 16384;              
    uint8_t* s_K1 = smem_flex + 24576;               
    uint8_t* s_V0 = smem_flex + 32768;                
    uint8_t* s_V1 = smem_flex + 40960;                
    uint8_t* s_O0 = smem_flex + 49152;                
    uint8_t* s_O1 = smem_flex + 57344;                
    uint8_t* s_dO0= smem_flex + 65536;                
    uint8_t* s_dO1= smem_flex + 73728;                
    uint8_t* s_PT = smem_flex + 81920;                
    uint8_t* s_dS = smem_flex + 90112;                
    float* s_D    = (float*)(smem_flex + 98304);     
    uint64_t* mbar = (uint64_t*)(smem_flex + 98560); 

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar, 16384 * 3); 
        tma_load_2d_fn(&tma_Q, mbar, s_Q0, 0, b_idx * H * S + h_idx * S + qb_idx * 64);
        tma_load_2d_fn(&tma_Q, mbar, s_Q1, 64, b_idx * H * S + h_idx * S + qb_idx * 64);
        tma_load_2d_fn(&tma_O, mbar, s_O0, 0, b_idx * H * S + h_idx * S + qb_idx * 64);
        tma_load_2d_fn(&tma_O, mbar, s_O1, 64, b_idx * H * S + h_idx * S + qb_idx * 64);
        tma_load_2d_fn(&tma_dO, mbar, s_dO0, 0, b_idx * H * S + h_idx * S + qb_idx * 64);
        tma_load_2d_fn(&tma_dO, mbar, s_dO1, 64, b_idx * H * S + h_idx * S + qb_idx * 64);
    }
    mbarrier_wait_fn(mbar, 0);
    __syncthreads();
    
    compute_D_from_O_dO_vec(s_O0, s_dO0, s_O1, s_O1, s_D, 64);
    __syncthreads();

    uint32_t phase = 0;

    for (int kb = 0; kb <= qb_idx && kb < S / 64; kb++) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar, 16384); 
            tma_load_2d_fn(&tma_K, mbar, s_K0, 0, b_idx * H * S + h_idx * S + kb * 64);
            tma_load_2d_fn(&tma_K, mbar, s_K1, 64, b_idx * H * S + h_idx * S + kb * 64);
            tma_load_2d_fn(&tma_V, mbar, s_V0, 0, b_idx * H * S + h_idx * S + kb * 64);
            tma_load_2d_fn(&tma_V, mbar, s_V1, 64, b_idx * H * S + h_idx * S + kb * 64);
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        __syncthreads();
        
        mbarrier_arrive_and_expect_tx_fn(mbar, 0);
        
        for (int k=0; k<4; k++) {
            uint32_t desc_K = make_smem_desc((char*)s_K0 + k * 128, 0, 1024);
            uint32_t desc_Q = make_smem_desc((char*)s_Q0 + k * 128, 0, 1024);
            umma_f16_cg2_fn(col_S, desc_K, desc_Q, make_instr_desc_fn<0, 0>(64, 64), k==0 ? 0 : 1);
        }
        for (int k=0; k<4; k++) {
            uint32_t desc_K = make_smem_desc((char*)s_K1 + k * 128, 0, 1024);
            uint32_t desc_Q = make_smem_desc((char*)s_Q1 + k * 128, 0, 1024);
            umma_f16_cg2_fn(col_S, desc_K, desc_Q, make_instr_desc_fn<0, 0>(64, 64), 1);
        }
        
        for (int k=0; k<4; k++) {
            uint32_t desc_V = make_smem_desc((char*)s_V0 + k * 128, 0, 1024);
            uint32_t desc_dO = make_smem_desc((char*)s_dO0 + k * 128, 0, 1024);
            umma_f16_cg2_fn(col_dP, desc_V, desc_dO, make_instr_desc_fn<0, 0>(64, 64), k==0 ? 0 : 1);
        }
        for (int k=0; k<4; k++) {
            uint32_t desc_V = make_smem_desc((char*)s_V1 + k * 128, 0, 1024);
            uint32_t desc_dO = make_smem_desc((char*)s_dO1 + k * 128, 0, 1024);
            umma_f16_cg2_fn(col_dP, desc_V, desc_dO, make_instr_desc_fn<0, 0>(64, 64), 1);
        }
        
        umma_commit_2sm_fn(mbar);
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        __syncthreads();
        
        compute_P_from_S((uint32_t*)col_S, s_PT, s_D, qb_idx, kb, L, b_idx, h_idx, S);
        compute_dS_from_P_dP(s_PT, (uint32_t*)col_dP, s_D, qb_idx, kb, s_dS, s_PT);
        __syncthreads();
        
        mbarrier_arrive_and_expect_tx_fn(mbar, 0);
        
        for (int k=0; k<4; k++) {
            uint32_t desc_dS = make_smem_desc((char*)s_dS + k * 128, 0, 1024);
            uint32_t desc_K = make_smem_desc((char*)s_K0 + k * 128, 1024, 0);
            umma_f16_cg2_fn(col_dQ0, desc_dS, desc_K, make_instr_desc_fn<0, 1>(64, 64), 1);
        }
        for (int k=0; k<4; k++) {
            uint32_t desc_dS = make_smem_desc((char*)s_dS + k * 128, 0, 1024);
            uint32_t desc_K = make_smem_desc((char*)s_K1 + k * 128, 1024, 0);
            umma_f16_cg2_fn(col_dQ1, desc_dS, desc_K, make_instr_desc_fn<0, 1>(64, 64), 1);
        }
        
        umma_commit_2sm_fn(mbar);
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        __syncthreads();
    }
    
    __syncthreads();
    store_tmem_to_gmem(dQ, tmem_base + col_dQ0, b_idx * H * S + h_idx * S + qb_idx * 64, 0, 128, S);
    store_tmem_to_gmem(dQ, tmem_base + col_dQ1, b_idx * H * S + h_idx * S + qb_idx * 64, 64, 128, S);
    
    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_base, 512);
    }
}

CUresult create_tma_descriptor(CUtensorMap* d, void* globalAddress, uint64_t BHS, uint64_t d_dim, uint32_t BLOCK_M) {
    cuuint64_t globalDim[2] = {d_dim, BHS};
    cuuint64_t globalStrides[1] = {d_dim * 2};
    cuuint32_t boxDim[2] = {64, BLOCK_M};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, globalAddress, globalDim, globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
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

    CUtensorMap tma_Q, tma_K, tma_V, tma_O, tma_dO, tma_dQ, tma_dK, tma_dV;
    uint64_t BHS = B * H * S;
    create_tma_descriptor(&tma_Q, (void*)Q.data_ptr(), BHS, d, 64);
    create_tma_descriptor(&tma_K, (void*)K.data_ptr(), BHS, d, 64);
    create_tma_descriptor(&tma_V, (void*)V.data_ptr(), BHS, d, 64);
    create_tma_descriptor(&tma_O, (void*)O.data_ptr(), BHS, d, 64);
    create_tma_descriptor(&tma_dO, (void*)dO.data_ptr(), BHS, d, 64);
    create_tma_descriptor(&tma_dQ, (void*)dQ.data_ptr(), BHS, d, 64);
    create_tma_descriptor(&tma_dK, (void*)dK.data_ptr(), BHS, d, 64);
    create_tma_descriptor(&tma_dV, (void*)dV.data_ptr(), BHS, d, 64);

    dim3 grid(B * H * (S + 63) / 64, 1, 1);
    dim3 block(128, 1, 1);

    CUDA_CHECK(cudaFuncSetAttribute(
        bwd_kernel_dvk,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        100000));
    
    CUDA_CHECK(cudaFuncSetAttribute(
        bwd_kernel_dq,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        100000));

    cudaLaunchConfig_t config_dvk = {};
    config_dvk.gridDim = grid;
    config_dvk.blockDim = block;
    config_dvk.dynamicSmemBytes = 100000;
    config_dvk.stream = stream;
    cudaLaunchAttribute attrs_dvk[1];
    attrs_dvk[0].id = cudaLaunchAttributeClusterDimension;
    attrs_dvk[0].val.clusterDim.x = 2;
    attrs_dvk[0].val.clusterDim.y = 1;
    attrs_dvk[0].val.clusterDim.z = 1;
    config_dvk.attrs = attrs_dvk;
    config_dvk.numAttrs = 1;
    
    CUresult res_dvk = cudaLaunchKernelEx(&config_dvk, bwd_kernel_dvk, tma_Q, tma_K, tma_V, tma_O, tma_dO, tma_dV, tma_dK, static_cast<const float*>(L.data_ptr()), static_cast<__nv_bfloat16*>(dV.data_ptr()), static_cast<__nv_bfloat16*>(dK.data_ptr()), S, H);
    if (res_dvk != CUDA_SUCCESS) {
        fprintf(stderr, "dvk kernel failed\n");
    }

    cudaLaunchConfig_t config_dq = {};
    config_dq.gridDim = grid;
    config_dq.blockDim = block;
    config_dq.dynamicSmemBytes = 100000;
    config_dq.stream = stream;
    cudaLaunchAttribute attrs_dq[1];
    attrs_dq[0].id = cudaLaunchAttributeClusterDimension;
    attrs_dq[0].val.clusterDim.x = 2;
    attrs_dq[0].val.clusterDim.y = 1;
    attrs_dq[0].val.clusterDim.z = 1;
    config_dq.attrs = attrs_dq;
    config_dq.numAttrs = 1;
    
    CUresult res_dq = cudaLaunchKernelEx(&config_dq, bwd_kernel_dq, tma_Q, tma_K, tma_V, tma_O, tma_dO, tma_dQ, static_cast<const float*>(L.data_ptr()), static_cast<__nv_bfloat16*>(dQ.data_ptr()), S, H);
    if (res_dq != CUDA_SUCCESS) {
        fprintf(stderr, "dq kernel failed\n");
    }
        
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda