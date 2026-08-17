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
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(bar)),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

CUresult create_tma_2d_descriptor(CUtensorMap* d, void* globalAddress, 
                                  uint64_t g_D, uint64_t g_total_S,
                                  uint32_t b_D, uint32_t b_S,
                                  CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[2] = {g_D, g_total_S};
    cuuint64_t globalStrides[1] = {g_D * 2};
    cuuint32_t boxDim[2] = {b_D, b_S};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        globalAddress, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
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

__device__ __forceinline__ void umma_commit_1sm(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];"
        :: "r"(a));
}

__device__ __forceinline__ void read_packed_float4_from_tmem(int tid, int col, float4* r, uint32_t tmem_base) {
    uint32_t addr = tmem_base + (tid << 16) + col;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(r->x),"=r"(r->y),"=r"(r->z),"=r"(r->w) : "r"(addr));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void write_packed_float4_to_tmem(int tid, int col, float4* r, uint32_t tmem_base) {
    uint32_t addr = tmem_base + (tid << 16) + col;
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        :: "r"(__float_as_uint(r->x)), "r"(__float_as_uint(r->y)), 
           "r"(__float_as_uint(r->z)), "r"(__float_as_uint(r->w)), "r"(addr));
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
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
    char* Q_1 = smem + 8192;    
    char* K_0 = smem + 16384;   
    char* K_1 = smem + 24576;   
    char* V_0 = smem + 32768;   
    char* V_1 = smem + 40960;   
    char* P_0 = smem + 49152;   

    uint32_t tmem_S_0, tmem_O_0, tmem_O_1;
    int tid = threadIdx.x;
    
    __shared__ uint64_t bar[1];
    if (tid == 0) {
        init_smem_barrier_fn(bar, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    if (tid == 0) {
        tmem_alloc_fn_cg1(&tmem_S_0, 128);
        tmem_alloc_fn_cg1(&tmem_O_0, 128);
        tmem_alloc_fn_cg1(&tmem_O_1, 128);
    }
    __syncthreads();
    
    int q_blk = blockIdx.x;
    int b_h = blockIdx.y;
    int q_off = q_blk * 64;
    
    uint32_t phase = 0;

    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar, 16384);
        tma_load_2d_fn(&tma_Q, bar, Q_0, 0, b_h * S_len + q_off);
        tma_load_2d_fn(&tma_Q, bar, Q_1, 64, b_h * S_len + q_off);
    }
    if (q_off >= S_len) return;
    
    mbarrier_wait_fn(bar, phase);
    phase ^= 1;
    __syncthreads();
    
    float prev_max_0 = -1e20f;
    float row_sum_0 = 0.0f;
    
    uint32_t idesc_QKT = 0;
    idesc_QKT |= (1u << 4);     
    idesc_QKT |= (1u << 7);     
    idesc_QKT |= (1u << 10);    
    idesc_QKT |= (0u << 15);    
    idesc_QKT |= (0u << 16);    
    idesc_QKT |= ((64 / 8) << 17);  
    idesc_QKT |= ((64 / 16) << 24); 

    uint32_t idesc_PV = 0;
    idesc_PV |= (1u << 4);      
    idesc_PV |= (1u << 7);      
    idesc_PV |= (1u << 10);     
    idesc_PV |= (0u << 15);     
    idesc_PV |= (1u << 16);     
    idesc_PV |= ((64 / 8) << 17);  
    idesc_PV |= ((64 / 16) << 24); 

    int num_S_blocks = (S_len + 63) / 64;

    for (int k_blk = 0; k_blk <= q_blk && k_blk < num_S_blocks; k_blk++) {
        int k_off = k_blk * 64;
        
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar, 32768);
            tma_load_2d_fn(&tma_K, bar, K_0, 0, b_h * S_len + k_off);
            tma_load_2d_fn(&tma_K, bar, K_1, 64, b_h * S_len + k_off);
            tma_load_2d_fn(&tma_V, bar, V_0, 0, b_h * S_len + k_off);
            tma_load_2d_fn(&tma_V, bar, V_1, 64, b_h * S_len + k_off);
        }
        mbarrier_wait_fn(bar, phase);
        phase ^= 1;
        __syncthreads();
        
        float alpha_0 = 1.0f;
        float rmax_0 = -1e20f;
        
        for (int col = 0; col < 64; col += 4) {
            float4 r;
            read_packed_float4_from_tmem(tid, col, &r, tmem_S_0);
            
            r.x = (k_off + col > q_off + tid || k_off + col >= S_len) ? -1e20f : r.x;
            r.y = (k_off + col + 1 > q_off + tid || k_off + col + 1 >= S_len) ? -1e20f : r.y;
            r.z = (k_off + col + 2 > q_off + tid || k_off + col + 2 >= S_len) ? -1e20f : r.z;
            r.w = (k_off + col + 3 > q_off + tid || k_off + col + 3 >= S_len) ? -1e20f : r.w;
            
            rmax_0 = fmaxf(rmax_0, fmaxf(fmaxf(fmaxf(r.x, r.y), fmaxf(r.z, r.w)), rmax_0));
        }
        
        float new_max_0 = fmaxf(prev_max_0, rmax_0);
        alpha_0 = fast_exp2f_fn((prev_max_0 - new_max_0) * 1.4426950408889634f);
        
        if (alpha_0 < 1.0f) {
            for (int col = 0; col < 64; col += 4) {
                float4 r0, r1;
                read_packed_float4_from_tmem(tid, col, &r0, tmem_O_0);
                read_packed_float4_from_tmem(tid, col, &r1, tmem_O_1);
                
                r0.x *= alpha_0; r0.y *= alpha_0; r0.z *= alpha_0; r0.w *= alpha_0;
                r1.x *= alpha_0; r1.y *= alpha_0; r1.z *= alpha_0; r1.w *= alpha_0;
                
                write_packed_float4_to_tmem(tid, col, &r0, tmem_O_0);
                write_packed_float4_to_tmem(tid, col, &r1, tmem_O_1);
            }
        }
        prev_max_0 = new_max_0;
        __syncthreads();
        
        if (tid == 0) {
            asm volatile(
                "{\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 0;\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %4, %5, %3, 1;\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %6, %7, %3, 1;\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %8, %9, %3, 1;\n"
                "}\n"
                :: "r"(tmem_S_0), 
                   "l"(make_smem_desc_sm100_fn(Q_0, 1, 1024)), "l"(make_smem_desc_sm100_fn(K_0, 1, 1024)), "r"(idesc_QKT),
                   "l"(make_smem_desc_sm100_fn((char*)Q_0 + 256, 1, 1024)), "l"(make_smem_desc_sm100_fn((char*)K_0 + 256, 1, 1024)),
                   "l"(make_smem_desc_sm100_fn((char*)Q_0 + 512, 1, 1024)), "l"(make_smem_desc_sm100_fn((char*)K_0 + 512, 1, 1024)),
                   "l"(make_smem_desc_sm100_fn((char*)Q_0 + 768, 1, 1024)), "l"(make_smem_desc_sm100_fn((char*)K_0 + 768, 1, 1024)));
                   
            asm volatile(
                "{\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 1;\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %4, %5, %3, 1;\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %6, %7, %3, 1;\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %8, %9, %3, 1;\n"
                "}\n"
                :: "r"(tmem_S_0), 
                   "l"(make_smem_desc_sm100_fn(Q_1, 1, 1024)), "l"(make_smem_desc_sm100_fn(K_1, 1, 1024)), "r"(idesc_QKT),
                   "l"(make_smem_desc_sm100_fn((char*)Q_1 + 256, 1, 1024)), "l"(make_smem_desc_sm100_fn((char*)K_1 + 256, 1, 1024)),
                   "l"(make_smem_desc_sm100_fn((char*)Q_1 + 512, 1, 1024)), "l"(make_smem_desc_sm100_fn((char*)K_1 + 512, 1, 1024)),
                   "l"(make_smem_desc_sm100_fn((char*)Q_1 + 768, 1, 1024)), "l"(make_smem_desc_sm100_fn((char*)K_1 + 768, 1, 1024)));
                   
            umma_commit_1sm(bar);
        }
        mbarrier_wait_fn(bar, phase);
        phase ^= 1;
        __syncthreads();
        
        float rsum_local_0 = 0.0f;
        for (int c = 0; c < 64; c += 2) {
            float4 r;
            read_packed_float4_from_tmem(tid, c, &r, tmem_S_0);
            
            float f0 = r.x * (1.0f / sqrt(128.0f));
            float f1 = r.y * (1.0f / sqrt(128.0f));
            float f2 = r.z * (1.0f / sqrt(128.0f));
            float f3 = r.w * (1.0f / sqrt(128.0f));
            
            f0 = (k_off + c > q_off + tid || k_off + c >= S_len) ? -1e20f : f0;
            f1 = (k_off + c + 1 > q_off + tid || k_off + c + 1 >= S_len) ? -1e20f : f1;
            f2 = (k_off + c + 2 > q_off + tid || k_off + c + 2 >= S_len) ? -1e20f : f2;
            f3 = (k_off + c + 3 > q_off + tid || k_off + c + 3 >= S_len) ? -1e20f : f3;
            
            float e0 = fast_exp2f_fn((f0 - prev_max_0) * 1.4426950408889634f);
            float e1 = fast_exp2f_fn((f1 - prev_max_0) * 1.4426950408889634f);
            float e2 = fast_exp2f_fn((f2 - prev_max_0) * 1.4426950408889634f);
            float e3 = fast_exp2f_fn((f3 - prev_max_0) * 1.4426950408889634f);
            
            rsum_local_0 += e0 + e1 + e2 + e3;
            
            int x_chunk0 = c / 8;
            int y_rem0 = tid % 8;
            int swizzled_x0 = y_rem0 ^ x_chunk0;
            int offset0 = tid * 128 + swizzled_x0 * 16 + (c % 8) * 2;
            *(uint32_t*)&P_0[offset0] = pack_bf16_fn(e0, e1);

            int x_chunk1 = (c+2) / 8;
            int y_rem1 = tid % 8;
            int swizzled_x1 = y_rem1 ^ x_chunk1;
            int offset1 = tid * 128 + swizzled_x1 * 16 + ((c+2) % 8) * 2;
            *(uint32_t*)&P_0[offset1] = pack_bf16_fn(e2, e3);
        }
        
        row_sum_0 = row_sum_0 * alpha_0 + rsum_local_0;
        
        __syncthreads();
        fence_async_shared_fn();
        
        if (tid == 0) {
            asm volatile(
                "{\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 0;\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %4, %5, %3, 1;\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %6, %7, %3, 1;\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %8, %9, %3, 1;\n"
                "}\n"
                :: "r"(tmem_O_0), 
                   "l"(make_smem_desc_sm100_fn(P_0, 1, 1024)), "l"(make_smem_desc_sm100_fn(V_0, 1024, 1024)), "r"(idesc_PV),
                   "l"(make_smem_desc_sm100_fn((char*)P_0 + 256, 1, 1024)), "l"(make_smem_desc_sm100_fn((char*)V_0 + 1024, 1024, 1024)),
                   "l"(make_smem_desc_sm100_fn((char*)P_0 + 512, 1, 1024)), "l"(make_smem_desc_sm100_fn((char*)V_0 + 2048, 1024, 1024)),
                   "l"(make_smem_desc_sm100_fn((char*)P_0 + 768, 1, 1024)), "l"(make_smem_desc_sm100_fn((char*)V_0 + 3072, 1024, 1024)));
                   
            asm volatile(
                "{\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 0;\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %4, %5, %3, 1;\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %6, %7, %3, 1;\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %8, %9, %3, 1;\n"
                "}\n"
                :: "r"(tmem_O_1), 
                   "l"(make_smem_desc_sm100_fn(P_0, 1, 1024)), "l"(make_smem_desc_sm100_fn(V_1, 1024, 1024)), "r"(idesc_PV),
                   "l"(make_smem_desc_sm100_fn((char*)P_0 + 256, 1, 1024)), "l"(make_smem_desc_sm100_fn((char*)V_1 + 1024, 1024, 1024)),
                   "l"(make_smem_desc_sm100_fn((char*)P_0 + 512, 1, 1024)), "l"(make_smem_desc_sm100_fn((char*)V_1 + 2048, 1024, 1024)),
                   "l"(make_smem_desc_sm100_fn((char*)P_0 + 768, 1, 1024)), "l"(make_smem_desc_sm100_fn((char*)V_1 + 3072, 1024, 1024)));
                   
            umma_commit_1sm(bar);
        }
        mbarrier_wait_fn(bar, phase);
        phase ^= 1;
        __syncthreads();
    }
    
    for (int col = 0; col < 64; col += 2) {
        uint32_t r0, r1;
        uint32_t addr0 = tmem_O_0 + (tid << 16) + col;
        uint32_t addr1 = tmem_O_1 + (tid << 16) + col;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x2.b32 {%0,%1}, [%2];"
            : "=r"(r0), "=r"(r1) : "r"(addr0), "r"(addr1));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float f0_0 = __uint_as_float(r0) / row_sum_0;
        float f1_0 = __uint_as_float(r1) / row_sum_0;
        
        float f0_1 = __uint_as_float(r0) / row_sum_0;
        float f1_1 = __uint_as_float(r1) / row_sum_0;
        
        if (q_off + tid < S_len) {
            uint32_t packed0 = pack_bf16_fn(f0_0, f1_0);
            uint64_t idx0 = ((uint64_t)b_h * S_len + q_off + tid) * 128 + col;
            *(uint32_t*)&O[idx0] = packed0;
            
            uint32_t packed1 = pack_bf16_fn(f0_1, f1_1);
            uint64_t idx1 = ((uint64_t)b_h * S_len + q_off + tid) * 128 + col + 64;
            *(uint32_t*)&O[idx1] = packed1;
        }
    }
    
    if (tid < 64) {
        if (q_off + tid < S_len) {
            LSE[(uint64_t)b_h * S_len + q_off + tid] = prev_max_0 + logf(row_sum_0);
        }
    }
    
    if (tid == 0) {
        tmem_dealloc_fn_cg1(tmem_S_0, 128);
        tmem_dealloc_fn_cg1(tmem_O_0, 128);
        tmem_dealloc_fn_cg1(tmem_O_1, 128);
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
    
    CU_CHECK(create_tma_2d_descriptor(&tma_Q, Q_ptr, D, B * H * S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_2d_descriptor(&tma_K, K_ptr, D, B * H * S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_2d_descriptor(&tma_V, V_ptr, D, B * H * S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B));
    
    int64_t S_blocks = (S + 63) / 64;
    dim3 grid(S_blocks, B * H);
    dim3 block(128);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int smem_size = 57344 + 1024; // 56 KB tensors + dynamic offsets + barrier alignment
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