#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>
#include <math.h>

#define CU_CHECK(call) do { \
    CUresult _e = (call); \
    if (_e != CUDA_SUCCESS) { \
        fprintf(stderr, "CU error %d at %s:%d\n", (int)_e, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

__device__ __forceinline__ bool elect_one_sync_fn() {
    uint32_t pred;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "elect.sync _|p, 0xFFFFFFFF;\n"
        "selp.b32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(pred));
    return pred != 0;
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ uint32_t pack_bf16_fn(float fp32_a, float fp32_b) {
    __nv_bfloat16 a = __float2bfloat16(fp32_a);
    __nv_bfloat16 b = __float2bfloat16(fp32_b);
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

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

CUresult create_tma_4d_descriptor(CUtensorMap* d, void* globalAddress, 
                                  uint64_t g_D, uint64_t g_S, uint64_t g_H, uint64_t g_B,
                                  uint32_t b_D, uint32_t b_S) {
    cuuint64_t globalDim[4] = {g_D, g_S, g_H, g_B};
    cuuint64_t globalStrides[3] = {g_D * 2, g_D * g_S * 2, g_D * g_S * g_H * 2};
    cuuint32_t boxDim[4] = {b_D, b_S, 1, 1};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
        globalAddress, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

__device__ __forceinline__ void tmem_alloc_fn_cg1(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
       :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn_cg1(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
       :: "r"(addr), "r"(ncols));
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

__device__ __forceinline__ void umma_f16_cg1(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_f16_cg1_tmem_a(
    uint32_t tmem_c, uint32_t tmem_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
        :: "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_1sm(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];"
        :: "r"(a));
}


__global__ __launch_bounds__(128)
void causal_attention_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O, float* LSE, int S_len)
{
    extern __shared__ char smem_raw[];
    char* smem = (char*)(((uintptr_t)smem_raw + 1023) & ~1023); 
    
    char* Q_0 = smem;            
    char* K_0 = smem + 32768;   
    char* V_0 = smem + 65536;   

    uint32_t tmem_S_0, tmem_P_0, tmem_O_0;
    int tid = threadIdx.x;
    
    __shared__ uint64_t bar[1];
    if (tid == 0) {
        init_smem_barrier_fn(bar, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    if (tid == 0) {
        tmem_alloc_fn_cg1(&tmem_S_0, 128);
        tmem_alloc_fn_cg1(&tmem_P_0, 128);
        tmem_alloc_fn_cg1(&tmem_O_0, 128);
    }
    __syncthreads();
    
    int q_blk = blockIdx.x;
    int b_h = blockIdx.y;
    int q_off = q_blk * 128;
    uint32_t phase = 0;

    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar, 32768);
        tma_load_4d_fn(&tma_Q, bar, Q_0, 0, q_off, b_h % 48, b_h / 48);
        tma_load_4d_fn(&tma_Q, bar, Q_0 + 16384, 64, q_off, b_h % 48, b_h / 48);
    }
    if (q_off >= S_len) return;
    
    float row_max_0 = -1e20f;
    float row_sum_0 = 0.0f;
    float prev_max_0 = -1e20f;
    
    uint32_t idesc_QKT = 0;
    idesc_QKT |= (1u << 4);     
    idesc_QKT |= (1u << 7);     
    idesc_QKT |= (1u << 10);    
    idesc_QKT |= (0u << 15);    
    idesc_QKT |= (1u << 16);    
    idesc_QKT |= ((128 / 8) << 17);  
    idesc_QKT |= ((128 / 16) << 24); 

    uint32_t idesc_PV = 0;
    idesc_PV |= (1u << 4);      
    idesc_PV |= (1u << 7);      
    idesc_PV |= (1u << 10);     
    idesc_PV |= (0u << 15);     
    idesc_PV |= (0u << 16);     
    idesc_PV |= ((128 / 8) << 17);  
    idesc_PV |= ((128 / 16) << 24); 

    int num_S_blocks = (S_len + 127) / 128;

    for (int k_blk = 0; k_blk <= q_blk && k_blk < num_S_blocks; k_blk++) {
        int k_off = k_blk * 128;
        
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar, 32768 * 2);
            tma_load_4d_fn(&tma_K, bar, K_0, 0, k_off, b_h % 48, b_h / 48);
            tma_load_4d_fn(&tma_K, bar, K_0 + 16384, 64, k_off, b_h % 48, b_h / 48);
            tma_load_4d_fn(&tma_V, bar, V_0, 0, k_off, b_h % 48, b_h / 48);
            tma_load_4d_fn(&tma_V, bar, V_0 + 16384, 64, k_off, b_h % 48, b_h / 48);
        }
        mbarrier_wait_fn(bar, phase);
        phase ^= 1;
        __syncthreads();
        
        float alpha_0 = 1.0f;
        float rmax_0 = -1e20f;
        
        for (uint32_t col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            uint32_t addr = (tid << 16) | col;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            
            if (k_off + col > q_off + tid || k_off + col >= S_len) f0 = -1e20f;
            if (k_off + col + 1 > q_off + tid || k_off + col + 1 >= S_len) f1 = -1e20f;
            if (k_off + col + 2 > q_off + tid || k_off + col + 2 >= S_len) f2 = -1e20f;
            if (k_off + col + 3 > q_off + tid || k_off + col + 3 >= S_len) f3 = -1e20f;
            
            rmax_0 = fmaxf(rmax_0, fmaxf(fmaxf(f0, f1), fmaxf(f2, f3)));
        }
        
        float max_jump_0 = rmax_0 - prev_max_0;
        if (max_jump_0 > 0.01f) {
            alpha_0 = fast_exp2f_fn((prev_max_0 - rmax_0) * 1.4426950408889634f);
            prev_max_0 = rmax_0;
        } else {
            alpha_0 = 1.0f;
            rmax_0 = prev_max_0;
        }
        
        if (alpha_0 < 1.0f) {
            for (uint32_t col = 0; col < 128; col += 4) {
                uint32_t r0, r1, r2, r3;
                uint32_t addr = (tid << 16) | col;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                
                float f0 = __uint_as_float(r0) * alpha_0;
                float f1 = __uint_as_float(r1) * alpha_0;
                float f2 = __uint_as_float(r2) * alpha_0;
                float f3 = __uint_as_float(r3) * alpha_0;
                
                uint32_t p0 = __float_as_uint(f0);
                uint32_t p1 = __float_as_uint(f1);
                uint32_t p2 = __float_as_uint(f2);
                uint32_t p3 = __float_as_uint(f3);
                
                asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    :: "r"(p0), "r"(p1), "r"(p2), "r"(p3), "r"(addr));
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        }
        __syncthreads();
        
        if (tid == 0) {
            uint64_t desc_q0_h0_step0 = make_smem_desc_sm100_fn(Q_0, 1, 1024);
            uint64_t desc_q0_h0_step1 = make_smem_desc_sm100_fn((char*)Q_0 + 256, 1, 1024);
            uint64_t desc_q0_h0_step2 = make_smem_desc_sm100_fn((char*)Q_0 + 512, 1, 1024);
            uint64_t desc_q0_h0_step3 = make_smem_desc_sm100_fn((char*)Q_0 + 768, 1, 1024);
            
            uint64_t desc_k0_h0_step0 = make_smem_desc_sm100_fn(K_0, 1, 1024);
            uint64_t desc_k0_h0_step1 = make_smem_desc_sm100_fn((char*)K_0 + 256, 1, 1024);
            uint64_t desc_k0_h0_step2 = make_smem_desc_sm100_fn((char*)K_0 + 512, 1, 1024);
            uint64_t desc_k0_h0_step3 = make_smem_desc_sm100_fn((char*)K_0 + 768, 1, 1024);
            
            uint64_t desc_q0_h1_step0 = make_smem_desc_sm100_fn(Q_0 + 16384, 1, 1024);
            uint64_t desc_q0_h1_step1 = make_smem_desc_sm100_fn((char*)Q_0 + 16384 + 256, 1, 1024);
            uint64_t desc_q0_h1_step2 = make_smem_desc_sm100_fn((char*)Q_0 + 16384 + 512, 1, 1024);
            uint64_t desc_q0_h1_step3 = make_smem_desc_sm100_fn((char*)Q_0 + 16384 + 768, 1, 1024);
            
            uint64_t desc_k0_h1_step0 = make_smem_desc_sm100_fn(K_0 + 16384, 1, 1024);
            uint64_t desc_k0_h1_step1 = make_smem_desc_sm100_fn((char*)K_0 + 16384 + 256, 1, 1024);
            uint64_t desc_k0_h1_step2 = make_smem_desc_sm100_fn((char*)K_0 + 16384 + 512, 1, 1024);
            uint64_t desc_k0_h1_step3 = make_smem_desc_sm100_fn((char*)K_0 + 16384 + 768, 1, 1024);
            
            asm volatile(
                "{\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 0;\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %4, %5, %3, 1;\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %6, %7, %3, 1;\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %8, %9, %3, 1;\n"
                "}\n"
                :: "r"(tmem_S_0), 
                   "l"(desc_q0_h0_step0), "l"(desc_k0_h0_step0), "r"(idesc_QKT),
                   "l"(desc_q0_h0_step1), "l"(desc_k0_h0_step1),
                   "l"(desc_q0_h0_step2), "l"(desc_k0_h0_step2),
                   "l"(desc_q0_h0_step3), "l"(desc_k0_h0_step3));
                   
            asm volatile(
                "{\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 1;\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %4, %5, %3, 1;\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %6, %7, %3, 1;\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %8, %9, %3, 1;\n"
                "}\n"
                :: "r"(tmem_S_0), 
                   "l"(desc_q0_h1_step0), "l"(desc_k0_h1_step0), "r"(idesc_QKT),
                   "l"(desc_q0_h1_step1), "l"(desc_k0_h1_step1),
                   "l"(desc_q0_h1_step2), "l"(desc_k0_h1_step2),
                   "l"(desc_q0_h1_step3), "l"(desc_k0_h1_step3));
                   
            umma_commit_1sm(bar);
        }
        mbarrier_wait_fn(bar, phase);
        phase ^= 1;
        __syncthreads();
        
        float rsum_local_0 = 0.0f;
        for (uint32_t col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            uint32_t addr = (tid << 16) | col;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float f0 = __uint_as_float(r0) * (1.0f / sqrt(128.0f));
            float f1 = __uint_as_float(r1) * (1.0f / sqrt(128.0f));
            float f2 = __uint_as_float(r2) * (1.0f / sqrt(128.0f));
            float f3 = __uint_as_float(r3) * (1.0f / sqrt(128.0f));
            
            if (k_off + col > q_off + tid || k_off + col >= S_len) f0 = -1e20f;
            if (k_off + col + 1 > q_off + tid || k_off + col + 1 >= S_len) f1 = -1e20f;
            if (k_off + col + 2 > q_off + tid || k_off + col + 2 >= S_len) f2 = -1e20f;
            if (k_off + col + 3 > q_off + tid || k_off + col + 3 >= S_len) f3 = -1e20f;
            
            float e0 = fast_exp2f_fn((f0 - rmax_0) * 1.4426950408889634f);
            float e1 = fast_exp2f_fn((f1 - rmax_0) * 1.4426950408889634f);
            float e2 = fast_exp2f_fn((f2 - rmax_0) * 1.4426950408889634f);
            float e3 = fast_exp2f_fn((f3 - rmax_0) * 1.4426950408889634f);
            
            rsum_local_0 += e0 + e1 + e2 + e3;
            
            uint32_t p0 = pack_bf16_fn(e0, e1);
            uint32_t p1 = pack_bf16_fn(e2, e3);
            
            uint32_t p_col = col / 2;
            uint32_t p_addr = (tid << 16) | p_col;
            asm volatile("tcgen05.st.sync.aligned.32x32b.x2.b32 {%0,%1}, [%2];"
                :: "r"(p0), "r"(p1), "r"(p_addr));
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        __syncthreads();
        
        row_sum_0 = row_sum_0 * alpha_0 + rsum_local_0;
        
        if (tid == 0) {
            uint64_t desc_v0_h0_step0 = make_smem_desc_sm100_fn(V_0, 1, 1024);
            uint64_t desc_v0_h0_step1 = make_smem_desc_sm100_fn((char*)V_0 + 256, 1, 1024);
            uint64_t desc_v0_h0_step2 = make_smem_desc_sm100_fn((char*)V_0 + 512, 1, 1024);
            uint64_t desc_v0_h0_step3 = make_smem_desc_sm100_fn((char*)V_0 + 768, 1, 1024);
            
            uint64_t desc_v0_h1_step0 = make_smem_desc_sm100_fn(V_0 + 16384, 1, 1024);
            uint64_t desc_v0_h1_step1 = make_smem_desc_sm100_fn((char*)V_0 + 16384 + 256, 1, 1024);
            uint64_t desc_v0_h1_step2 = make_smem_desc_sm100_fn((char*)V_0 + 16384 + 512, 1, 1024);
            uint64_t desc_v0_h1_step3 = make_smem_desc_sm100_fn((char*)V_0 + 16384 + 768, 1, 1024);
            
            asm volatile(
                "{\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, 1;\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1 + 8], %4, %3, 1;\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1 + 16], %5, %3, 1;\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1 + 24], %6, %3, 1;\n"
                "}\n"
                :: "r"(tmem_O_0), "r"(tmem_P_0), 
                   "l"(desc_v0_h0_step0), "r"(idesc_PV),
                   "l"(desc_v0_h0_step1), "l"(desc_v0_h0_step2), "l"(desc_v0_h0_step3));
                   
            asm volatile(
                "{\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1 + 32], %2, %3, 1;\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1 + 40], %4, %3, 1;\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1 + 48], %5, %3, 1;\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1 + 56], %6, %3, 1;\n"
                "}\n"
                :: "r"(tmem_O_0), "r"(tmem_P_0), 
                   "l"(desc_v0_h1_step0), "r"(idesc_PV),
                   "l"(desc_v0_h1_step1), "l"(desc_v0_h1_step2), "l"(desc_v0_h1_step3));
                   
            umma_commit_1sm(bar);
        }
        mbarrier_wait_fn(bar, phase);
        phase ^= 1;
        __syncthreads();
    }
    
    for (uint32_t col = 0; col < 128; col += 2) {
        uint32_t r0, r1;
        uint32_t addr = (tid << 16) | col;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x2.b32 {%0,%1}, [%2];"
            : "=r"(r0), "=r"(r1) : "r"(addr));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float f0 = __uint_as_float(r0) / row_sum_0;
        float f1 = __uint_as_float(r1) / row_sum_0;
        
        if (q_off + tid < S_len) {
            uint32_t packed = pack_bf16_fn(f0, f1);
            uint64_t idx = ((uint64_t)b_h * S_len + q_off + tid) * 128 + col;
            *(uint32_t*)&O[idx] = packed;
        }
    }
    
    if (tid < 128) {
        if (q_off + tid < S_len) {
            LSE[(uint64_t)b_h * S_len + q_off + tid] = prev_max_0 + logf(row_sum_0);
        }
    }
    
    if (tid == 0) {
        tmem_dealloc_fn_cg1(tmem_S_0, 128);
        tmem_dealloc_fn_cg1(tmem_P_0, 128);
        tmem_dealloc_fn_cg1(tmem_O_0, 128);
    }
}

namespace tvm_ffi_mha_with_lse_d128_causal {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    if (S == 0) return;
    
    CUtensorMap tma_Q, tma_K, tma_V;
    void* Q_ptr = Q.data_ptr();
    void* K_ptr = K.data_ptr();
    void* V_ptr = V.data_ptr();
    
    CU_CHECK(create_tma_4d_descriptor(&tma_Q, Q_ptr, D, S, H, B, 64, 128));
    CU_CHECK(create_tma_4d_descriptor(&tma_K, K_ptr, D, S, H, B, 64, 128));
    CU_CHECK(create_tma_4d_descriptor(&tma_V, V_ptr, D, S, H, B, 64, 128));
    
    int64_t S_blocks = (S + 127) / 128;
    dim3 grid(S_blocks, B * H);
    dim3 block(128);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int smem_size = 65536 * 3; // 196 KB
    CUDA_CHECK(cudaFuncSetAttribute(
        causal_attention_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size));
        
    causal_attention_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, 
        static_cast<__nv_bfloat16*>(O.data_ptr()), 
        static_cast<float*>(LSE.data_ptr()), 
        S);
        
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}  // namespace tvm_ffi_mha_with_lse_d128_causal