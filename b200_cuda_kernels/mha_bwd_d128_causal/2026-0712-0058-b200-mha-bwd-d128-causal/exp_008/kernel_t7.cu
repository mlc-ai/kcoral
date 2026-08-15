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
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
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

template<uint32_t M, uint32_t N, int A_MAJOR, int B_MAJOR>
__device__ __forceinline__ uint32_t make_instr_desc_fn() {
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

__device__ __forceinline__ void umma_f16_cg1_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_1sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.shared::cta.b64"
        " [%0];"
        :: "r"(a));
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
        uint4 val4 = *(const uint4*)(gmem_row + 4 * 8);
        uint4 val5 = *(const uint4*)(gmem_row + 5 * 8);
        uint4 val6 = *(const uint4*)(gmem_row + 6 * 8);
        uint4 val7 = *(const uint4*)(gmem_row + 7 * 8);
        
        uint32_t swizzled_offset0 = ((0 ^ (row % 8)) * 8 + 0) * 2 + (row * 64 * 2);
        *(uint4*)((char*)smem + swizzled_offset0) = val0;
        uint32_t swizzled_offset1 = ((1 ^ (row % 8)) * 8 + 0) * 2 + (row * 64 * 2);
        *(uint4*)((char*)smem + swizzled_offset1) = val1;
        uint32_t swizzled_offset2 = ((2 ^ (row % 8)) * 8 + 0) * 2 + (row * 64 * 2);
        *(uint4*)((char*)smem + swizzled_offset2) = val2;
        uint32_t swizzled_offset3 = ((3 ^ (row % 8)) * 8 + 0) * 2 + (row * 64 * 2);
        *(uint4*)((char*)smem + swizzled_offset3) = val3;
        uint32_t swizzled_offset4 = ((4 ^ (row % 8)) * 8 + 0) * 2 + (row * 64 * 2);
        *(uint4*)((char*)smem + swizzled_offset4) = val4;
        uint32_t swizzled_offset5 = ((5 ^ (row % 8)) * 8 + 0) * 2 + (row * 64 * 2);
        *(uint4*)((char*)smem + swizzled_offset5) = val5;
        uint32_t swizzled_offset6 = ((6 ^ (row % 8)) * 8 + 0) * 2 + (row * 64 * 2);
        *(uint4*)((char*)smem + swizzled_offset6) = val6;
        uint32_t swizzled_offset7 = ((7 ^ (row % 8)) * 8 + 0) * 2 + (row * 64 * 2);
        *(uint4*)((char*)smem + swizzled_offset7) = val7;
    }
}

__device__ __forceinline__ void compute_D_from_O_dO_vec(
    const uint8_t* s_O0, const uint8_t* s_dO0, 
    const uint8_t* s_O1, const uint8_t* s_dO1, 
    float* s_D) 
{
    int r = threadIdx.x;
    if (r < 64) {
        float d_val = 0;
        
        for (int c = 0; c < 64; c++) {
            int c_chunk = c / 8;
            int r_chunk = r % 8;
            int swizzled_c_chunk = c_chunk ^ r_chunk;
            int swizzled_c = swizzled_c_chunk * 8 + (c % 8);
            int swizzled_idx = r * 64 + swizzled_c;
            
            __nv_bfloat16 o0 = *reinterpret_cast<const __nv_bfloat16*>((const char*)s_O0 + swizzled_idx * 2);
            __nv_bfloat16 do0 = *reinterpret_cast<const __nv_bfloat16*>((const char*)s_dO0 + swizzled_idx * 2);
            
            d_val += __bfloat162float(o0) * __bfloat162float(do0);
        }
        
        for (int c = 0; c < 64; c++) {
            int c_chunk = c / 8;
            int r_chunk = r % 8;
            int swizzled_c_chunk = c_chunk ^ r_chunk;
            int swizzled_c = swizzled_c_chunk * 8 + (c % 8);
            int swizzled_idx = r * 64 + swizzled_c;
            
            __nv_bfloat16 o1 = *reinterpret_cast<const __nv_bfloat16*>((const char*)s_O1 + swizzled_idx * 2);
            __nv_bfloat16 do1 = *reinterpret_cast<const __nv_bfloat16*>((const char*)s_dO1 + swizzled_idx * 2);
            
            d_val += __bfloat162float(o1) * __bfloat162float(do1);
        }
        
        s_D[r] = d_val;
    }
}

__device__ __forceinline__ void compute_P_and_dS(
    const uint32_t* S_tmem, const uint32_t* dP_tmem, float* s_D, 
    int qb_idx, int kb_idx, const float* L, int b_idx, int h_idx, int H, int S, 
    uint8_t* s_P, uint8_t* s_dS, bool transpose) {
    
    float scale_log2e = (1.0f / sqrt(128.0f)) * 1.4426950408889634f;
    int q_idx = threadIdx.x;
    
    for (int c = 0; c < 64; c += 4) {
        uint32_t r0_S, r1_S, r2_S, r3_S;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" 
            : "=r"(r0_S),"=r"(r1_S),"=r"(r2_S),"=r"(r3_S) : "r"(c));
        tmem_load_fence_fn();
        
        uint32_t r0_dP, r1_dP, r2_dP, r3_dP;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" 
            : "=r"(r0_dP),"=r"(r1_dP),"=r"(r2_dP),"=r"(r3_dP) : "r"(c));
        tmem_load_fence_fn();
        
        float S0 = __uint_as_float(r0_S) * scale_log2e;
        float S1 = __uint_as_float(r1_S) * scale_log2e;
        float S2 = __uint_as_float(r2_S) * scale_log2e;
        float S3 = __uint_as_float(r3_S) * scale_log2e;
        
        float dP0 = __uint_as_float(r0_dP);
        float dP1 = __uint_as_float(r1_dP);
        float dP2 = __uint_as_float(r2_dP);
        float dP3 = __uint_as_float(r3_dP);
        
        int k_idx0 = c;
        int k_idx1 = c + 1;
        int k_idx2 = c + 2;
        int k_idx3 = c + 3;
        
        float lse = (qb_idx * 64 + q_idx < S) ? L[b_idx * H * S + h_idx * S + qb_idx * 64 + q_idx] * 1.4426950408889634f : 0;
        float D0 = s_D[q_idx];
        
        float p0 = fast_exp2f_fn(S0 - lse);
        float p1 = fast_exp2f_fn(S1 - lse);
        float p2 = fast_exp2f_fn(S2 - lse);
        float p3 = fast_exp2f_fn(S3 - lse);
        
        int global_q_idx = qb_idx * 64 + q_idx;
        int global_k_idx0 = kb_idx * 64 + k_idx0;
        int global_k_idx1 = kb_idx * 64 + k_idx1;
        int global_k_idx2 = kb_idx * 64 + k_idx2;
        int global_k_idx3 = kb_idx * 64 + k_idx3;
        
        if (global_q_idx >= S || global_k_idx0 >= S || global_q_idx < global_k_idx0) p0 = 0;
        if (global_q_idx >= S || global_k_idx1 >= S || global_q_idx < global_k_idx1) p1 = 0;
        if (global_q_idx >= S || global_k_idx2 >= S || global_q_idx < global_k_idx2) p2 = 0;
        if (global_q_idx >= S || global_k_idx3 >= S || global_q_idx < global_k_idx3) p3 = 0;
        
        float ds0 = p0 * (dP0 - D0);
        float ds1 = p1 * (dP1 - D0);
        float ds2 = p2 * (dP2 - D0);
        float ds3 = p3 * (dP3 - D0);
        
        if (transpose) {
            if (s_P) {
                *( (__nv_bfloat16*)((char*)s_P + k_idx0 * 128 + (((q_idx / 8) ^ (k_idx0 % 8)) * 16 + (q_idx % 8) * 2)) ) = __float2bfloat16(p0);
                *( (__nv_bfloat16*)((char*)s_P + k_idx1 * 128 + (((q_idx / 8) ^ (k_idx1 % 8)) * 16 + (q_idx % 8) * 2)) ) = __float2bfloat16(p1);
                *( (__nv_bfloat16*)((char*)s_P + k_idx2 * 128 + (((q_idx / 8) ^ (k_idx2 % 8)) * 16 + (q_idx % 8) * 2)) ) = __float2bfloat16(p2);
                *( (__nv_bfloat16*)((char*)s_P + k_idx3 * 128 + (((q_idx / 8) ^ (k_idx3 % 8)) * 16 + (q_idx % 8) * 2)) ) = __float2bfloat16(p3);
            }
            if (s_dS) {
                *( (__nv_bfloat16*)((char*)s_dS + k_idx0 * 128 + (((q_idx / 8) ^ (k_idx0 % 8)) * 16 + (q_idx % 8) * 2)) ) = __float2bfloat16(ds0);
                *( (__nv_bfloat16*)((char*)s_dS + k_idx1 * 128 + (((q_idx / 8) ^ (k_idx1 % 8)) * 16 + (q_idx % 8) * 2)) ) = __float2bfloat16(ds1);
                *( (__nv_bfloat16*)((char*)s_dS + k_idx2 * 128 + (((q_idx / 8) ^ (k_idx2 % 8)) * 16 + (q_idx % 8) * 2)) ) = __float2bfloat16(ds2);
                *( (__nv_bfloat16*)((char*)s_dS + k_idx3 * 128 + (((q_idx / 8) ^ (k_idx3 % 8)) * 16 + (q_idx % 8) * 2)) ) = __float2bfloat16(ds3);
            }
        } else {
            if (s_dS) {
                uint32_t packed0 = pack_bf16_fn(ds0, ds1);
                uint32_t packed1 = pack_bf16_fn(ds2, ds3);
                int swizzled_idx0 = q_idx * 64 + (((k_idx0 / 8) ^ (q_idx % 8)) * 8 + (k_idx0 % 8));
                int swizzled_idx1 = q_idx * 64 + (((k_idx2 / 8) ^ (q_idx % 8)) * 8 + (k_idx2 % 8));
                *(uint32_t*)((char*)s_dS + swizzled_idx0 * 2) = packed0;
                *(uint32_t*)((char*)s_dS + swizzled_idx1 * 2) = packed1;
            }
        }
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
        
        if (row < 64 && g_row < S && g_col + 3 < 128) {
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            
            __nv_bfloat16* out = gmem + (uint64_t)g_row * stride_row + g_col;
            out[0] = __float2bfloat16(f0);
            out[1] = __float2bfloat16(f1);
            out[2] = __float2bfloat16(f2);
            out[3] = __float2bfloat16(f3);
        }
    }
}

extern __shared__ __align__(1024) uint8_t smem_pool[];

__global__ __launch_bounds__(128, 1) void bwd_kernel_dq(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* O, const __nv_bfloat16* dO, const float* L,
    __nv_bfloat16* dQ, int S, int H) 
{
    int qb_idx = blockIdx.x;
    int h_idx = blockIdx.y % H;
    int b_idx = blockIdx.y / H;

    uint32_t tmem_base;
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_base, 256);
    }
    __syncthreads();

    uint32_t col_dQ0 = tmem_base;
    uint32_t col_dQ1 = tmem_base + 64;
    uint32_t col_S = tmem_base + 128;
    uint32_t col_dP = tmem_base + 192;

    extern __shared__ __align__(1024) uint8_t smem_flex[];
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
    uint8_t* s_dS = smem_flex + 81920;                
    float* s_D    = (float*)(smem_flex + 90112);     
    uint64_t* mbar = (uint64_t*)(smem_flex + 90368); 

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    int stride_y = (h_idx + b_idx * H) * S * 128;
    uint32_t outer_offset = b_idx * H * S + h_idx * S;
    
    if (threadIdx.x < 64) {
        load_tile_64x64_vec_swizzled(Q + stride_y, s_Q0, S, threadIdx.x, qb_idx * 64, 0, 0);
        load_tile_64x64_vec_swizzled(Q + stride_y, s_Q1, S, threadIdx.x, qb_idx * 64, 64, 0);
        load_tile_64x64_vec_swizzled(O + stride_y, s_O0, S, threadIdx.x, qb_idx * 64, 0, 0);
        load_tile_64x64_vec_swizzled(O + stride_y, s_O1, S, threadIdx.x, qb_idx * 64, 64, 0);
        load_tile_64x64_vec_swizzled(dO + stride_y, s_dO0, S, threadIdx.x, qb_idx * 64, 0, 0);
        load_tile_64x64_vec_swizzled(dO + stride_y, s_dO1, S, threadIdx.x, qb_idx * 64, 64, 0);
    }
    __syncthreads();
    
    compute_D_from_O_dO_vec(s_O0, s_dO0, s_O1, s_dO1, s_D);
    __syncthreads();

    int phase = 0;
    int max_k = min(qb_idx, S / 64 - 1);

    for (int kb = 0; kb <= max_k; kb++) {
        if (threadIdx.x < 64) {
            load_tile_64x64_vec_swizzled(K + stride_y, s_K0, S, threadIdx.x, kb * 64, 0, 0);
            load_tile_64x64_vec_swizzled(K + stride_y, s_K1, S, threadIdx.x, kb * 64, 64, 0);
            load_tile_64x64_vec_swizzled(V + stride_y, s_V0, S, threadIdx.x, kb * 64, 0, 0);
            load_tile_64x64_vec_swizzled(V + stride_y, s_V1, S, threadIdx.x, kb * 64, 64, 0);
        }
        __syncthreads();
        
        for (int k=0; k<4; k++) {
            uint32_t desc_K0 = make_smem_desc(s_K0 + k * 32, 1, 1024);
            uint32_t desc_Q0 = make_smem_desc(s_Q0 + k * 32, 1, 1024);
            umma_f16_cg1_fn(col_S, desc_Q0, desc_K0, make_instr_desc_fn<64, 64, 0, 0>(), k==0 ? 0 : 1);
        }
        for (int k=0; k<4; k++) {
            uint32_t desc_K1 = make_smem_desc(s_K1 + k * 32, 1, 1024);
            uint32_t desc_Q1 = make_smem_desc(s_Q1 + k * 32, 1, 1024);
            umma_f16_cg1_fn(col_S, desc_Q1, desc_K1, make_instr_desc_fn<64, 64, 0, 0>(), 1);
        }
        
        for (int k=0; k<4; k++) {
            uint32_t desc_V0 = make_smem_desc(s_V0 + k * 32, 1, 1024);
            uint32_t desc_dO0 = make_smem_desc(s_dO0 + k * 32, 1, 1024);
            umma_f16_cg1_fn(col_dP, desc_V0, desc_dO0, make_instr_desc_fn<64, 64, 0, 0>(), k==0 ? 0 : 1);
        }
        for (int k=0; k<4; k++) {
            uint32_t desc_V1 = make_smem_desc(s_V1 + k * 32, 1, 1024);
            uint32_t desc_dO1 = make_smem_desc(s_dO1 + k * 32, 1, 1024);
            umma_f16_cg1_fn(col_dP, desc_V1, desc_dO1, make_instr_desc_fn<64, 64, 0, 0>(), 1);
        }
        
        if (threadIdx.x == 0) umma_commit_1sm_fn(mbar);
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        __syncthreads();
        
        compute_P_and_dS((uint32_t*)col_S, (uint32_t*)col_dP, s_D, qb_idx, kb, L, b_idx, h_idx, H, S, NULL, s_dS, false);
        asm volatile("fence.proxy.async;" ::: "memory");
        __syncthreads();
        
        for (int k=0; k<4; k++) {
            uint32_t desc_dS = make_smem_desc(s_dS + k * 32, 1, 1024);
            uint32_t desc_K0 = make_smem_desc(s_K0 + k * 2048, 8192, 1024);
            umma_f16_cg1_fn(col_dQ0, desc_dS, desc_K0, make_instr_desc_fn<64, 64, 0, 1>(), 1);
        }
        for (int k=0; k<4; k++) {
            uint32_t desc_dS = make_smem_desc(s_dS + k * 32, 1, 1024);
            uint32_t desc_K1 = make_smem_desc(s_K1 + k * 2048, 8192, 1024);
            umma_f16_cg1_fn(col_dQ1, desc_dS, desc_K1, make_instr_desc_fn<64, 64, 0, 1>(), 1);
        }
        
        if (threadIdx.x == 0) umma_commit_1sm_fn(mbar);
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        __syncthreads();
    }
    
    __syncthreads();
    store_tmem_to_gmem(dQ + stride_y, col_dQ0, qb_idx * 64, 0, 128, S);
    store_tmem_to_gmem(dQ + stride_y, col_dQ1, qb_idx * 64, 64, 128, S);
    
    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_base, 256);
    }
}

__global__ __launch_bounds__(128, 1) void bwd_kernel_dvk(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* O, const __nv_bfloat16* dO, const float* L,
    __nv_bfloat16* dV, __nv_bfloat16* dK, int S, int H) 
{
    int kb_idx = blockIdx.x;
    int h_idx = blockIdx.y % H;
    int b_idx = blockIdx.y / H;

    uint32_t tmem_base;
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_base, 384);
    }
    __syncthreads();

    uint32_t col_S = tmem_base;
    uint32_t col_dP = tmem_base + 64;
    uint32_t col_dV0 = tmem_base + 128;
    uint32_t col_dV1 = tmem_base + 192;
    uint32_t col_dK0 = tmem_base + 256;
    uint32_t col_dK1 = tmem_base + 320;

    extern __shared__ __align__(1024) uint8_t smem_flex[];
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

    int stride_y = (h_idx + b_idx * H) * S * 128;
    uint32_t outer_offset = b_idx * H * S + h_idx * S;

    if (threadIdx.x < 64) {
        load_tile_64x64_vec_swizzled(K + stride_y, s_K0, S, threadIdx.x, kb_idx * 64, 0, 0);
        load_tile_64x64_vec_swizzled(K + stride_y, s_K1, S, threadIdx.x, kb_idx * 64, 64, 0);
        load_tile_64x64_vec_swizzled(V + stride_y, s_V0, S, threadIdx.x, kb_idx * 64, 0, 0);
        load_tile_64x64_vec_swizzled(V + stride_y, s_V1, S, threadIdx.x, kb_idx * 64, 64, 0);
    }
    __syncthreads();

    int phase = 0;
    int num_blocks = S / 64;

    for (int qb = kb_idx; qb < num_blocks; qb++) {
        if (threadIdx.x < 64) {
            load_tile_64x64_vec_swizzled(Q + stride_y, s_Q0, S, threadIdx.x, qb * 64, 0, 0);
            load_tile_64x64_vec_swizzled(Q + stride_y, s_Q1, S, threadIdx.x, qb * 64, 64, 0);
            load_tile_64x64_vec_swizzled(O + stride_y, s_O0, S, threadIdx.x, qb * 64, 0, 0);
            load_tile_64x64_vec_swizzled(O + stride_y, s_O1, S, threadIdx.x, qb * 64, 64, 0);
            load_tile_64x64_vec_swizzled(dO + stride_y, s_dO0, S, threadIdx.x, qb * 64, 0, 0);
            load_tile_64x64_vec_swizzled(dO + stride_y, s_dO1, S, threadIdx.x, qb * 64, 64, 0);
        }
        __syncthreads();
        
        compute_D_from_O_dO_vec(s_O0, s_dO0, s_O1, s_dO1, s_D);
        __syncthreads();
        
        for (int k=0; k<4; k++) {
            uint32_t desc_K0 = make_smem_desc(s_K0 + k * 32, 1, 1024);
            uint32_t desc_Q0 = make_smem_desc(s_Q0 + k * 32, 1, 1024);
            umma_f16_cg1_fn(col_S, desc_Q0, desc_K0, make_instr_desc_fn<64, 64, 0, 0>(), k==0 ? 0 : 1);
        }
        for (int k=0; k<4; k++) {
            uint32_t desc_K1 = make_smem_desc(s_K1 + k * 32, 1, 1024);
            uint32_t desc_Q1 = make_smem_desc(s_Q1 + k * 32, 1, 1024);
            umma_f16_cg1_fn(col_S, desc_Q1, desc_K1, make_instr_desc_fn<64, 64, 0, 0>(), 1);
        }
        
        for (int k=0; k<4; k++) {
            uint32_t desc_V0 = make_smem_desc(s_V0 + k * 32, 1, 1024);
            uint32_t desc_dO0 = make_smem_desc(s_dO0 + k * 32, 1, 1024);
            umma_f16_cg1_fn(col_dP, desc_V0, desc_dO0, make_instr_desc_fn<64, 64, 0, 0>(), k==0 ? 0 : 1);
        }
        for (int k=0; k<4; k++) {
            uint32_t desc_V1 = make_smem_desc(s_V1 + k * 32, 1, 1024);
            uint32_t desc_dO1 = make_smem_desc(s_dO1 + k * 32, 1, 1024);
            umma_f16_cg1_fn(col_dP, desc_V1, desc_dO1, make_instr_desc_fn<64, 64, 0, 0>(), 1);
        }
        
        if (threadIdx.x == 0) umma_commit_1sm_fn(mbar);
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        __syncthreads();
        
        compute_P_and_dS((uint32_t*)col_S, (uint32_t*)col_dP, s_D, qb, kb_idx, L, b_idx, h_idx, H, S, s_PT, s_dST, true);
        asm volatile("fence.proxy.async;" ::: "memory");
        __syncthreads();
        
        for (int k=0; k<4; k++) {
            uint32_t desc_PT0 = make_smem_desc(s_PT + k * 32, 1, 1024);
            uint32_t desc_dO0 = make_smem_desc(s_dO0 + k * 2048, 8192, 1024);
            umma_f16_cg1_fn(col_dV0, desc_PT0, desc_dO0, make_instr_desc_fn<64, 64, 0, 1>(), 1);
        }
        for (int k=0; k<4; k++) {
            uint32_t desc_PT1 = make_smem_desc(s_PT + k * 32, 1, 1024);
            uint32_t desc_dO1 = make_smem_desc(s_dO1 + k * 2048, 8192, 1024);
            umma_f16_cg1_fn(col_dV1, desc_PT1, desc_dO1, make_instr_desc_fn<64, 64, 0, 1>(), 1);
        }
        
        for (int k=0; k<4; k++) {
            uint32_t desc_dST0 = make_smem_desc(s_dST + k * 32, 1, 1024);
            uint32_t desc_Q0 = make_smem_desc(s_Q0 + k * 2048, 8192, 1024);
            umma_f16_cg1_fn(col_dK0, desc_dST0, desc_Q0, make_instr_desc_fn<64, 64, 0, 1>(), 1);
        }
        for (int k=0; k<4; k++) {
            uint32_t desc_dST1 = make_smem_desc(s_dST + k * 32, 1, 1024);
            uint32_t desc_Q1 = make_smem_desc(s_Q1 + k * 2048, 8192, 1024);
            umma_f16_cg1_fn(col_dK1, desc_dST1, desc_Q1, make_instr_desc_fn<64, 64, 0, 1>(), 1);
        }
        
        if (threadIdx.x == 0) umma_commit_1sm_fn(mbar);
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        __syncthreads();
    }
    
    __syncthreads();
    store_tmem_to_gmem(dV + stride_y, col_dV0, kb_idx * 64, 0, 128, S);
    store_tmem_to_gmem(dV + stride_y, col_dV1, kb_idx * 64, 64, 128, S);
    store_tmem_to_gmem(dK + stride_y, col_dK0, kb_idx * 64, 0, 128, S);
    store_tmem_to_gmem(dK + stride_y, col_dK1, kb_idx * 64, 64, 128, S);
    
    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_base, 384);
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

    dim3 grid(S / 64, B * H);
    dim3 block(128, 1, 1);

    CUDA_CHECK(cudaFuncSetAttribute(
        bwd_kernel_dvk,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        100000));
    
    CUDA_CHECK(cudaFuncSetAttribute(
        bwd_kernel_dq,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        96000));

    bwd_kernel_dvk<<<grid, block, 100000, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(O.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        S, H);

    bwd_kernel_dq<<<grid, block, 96000, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(O.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        S, H);
        
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda