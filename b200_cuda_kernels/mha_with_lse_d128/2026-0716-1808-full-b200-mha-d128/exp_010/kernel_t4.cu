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

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        fprintf(stderr, "CU error %d at %s:%d\n",                 \
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)


namespace tvm_ffi_optimized_cuda {

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

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tma_load_2d_cta_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ uint64_t add_base_offset(uint64_t desc, uint32_t offset_bytes) {
    uint32_t base = desc & 0x3FFF;
    base += offset_bytes / 16;
    if ((base >> 14) != 0) {
        base &= 0x3FFF;
    }
    desc &= ~0x3FFF;
    desc |= base;
    return desc;
}

__device__ __forceinline__ uint64_t make_smem_desc_128B(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16; 
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32; 
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
    d |= ((N >> 3) & 0x3F);     
    d |= ((M >> 4) & 0x1F) << 24;    
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn_trans_B(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (0u << 15);   
    d |= (1u << 16);   
    d |= ((N >> 3) & 0x3F);     
    d |= ((M >> 4) & 0x1F) << 24;    
    return d;
}

__device__ __forceinline__ void umma_f16_cg2(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum, uint32_t scale) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p, %5;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum), "r"(scale));
}

__device__ __forceinline__ void umma_commit_2sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(a), "h"((uint16_t)0x3));
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

__global__ void __launch_bounds__(128) mha_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* gmem_O,
    float* gmem_LSE,
    uint32_t S)
{
    extern __shared__ __align__(1024) char smem_pool[];
    
    __nv_bfloat16* smem_Q0 = (__nv_bfloat16*)smem_pool;                
    __nv_bfloat16* smem_Q1 = (__nv_bfloat16*)(smem_pool + 8192);      
    __nv_bfloat16* smem_K0[2] = {(__nv_bfloat16*)(smem_pool + 16384), (__nv_bfloat16*)(smem_pool + 24576)};      
    __nv_bfloat16* smem_K1[2] = {(__nv_bfloat16*)(smem_pool + 32768), (__nv_bfloat16*)(smem_pool + 40960)};      
    __nv_bfloat16* smem_V0[2] = {(__nv_bfloat16*)(smem_pool + 49152), (__nv_bfloat16*)(smem_pool + 57344)};      
    __nv_bfloat16* smem_V1[2] = {(__nv_bfloat16*)(smem_pool + 65536), (__nv_bfloat16*)(smem_pool + 73728)};      
    __nv_bfloat16* smem_S = (__nv_bfloat16*)(smem_pool + 81920);     
    
    uint64_t* mbar_Q = (uint64_t*)(smem_pool + 90112);                  
    uint64_t* mbar_KV = (uint64_t*)(smem_pool + 90120);                 

    uint32_t tid = threadIdx.x;
    uint32_t s_blk = blockIdx.y;
    uint32_t bh = blockIdx.x;
    uint32_t row_base = s_blk * 64;

    if (tid == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(&mbar_KV[0], 1);
        init_smem_barrier_fn(&mbar_KV[1], 1);
    }
    __syncthreads();
    
    uint32_t tmem_base;
    if (tid == 0) {
        tmem_alloc_fn(&tmem_base, 256); 
    }
    __syncthreads();
    
    uint32_t tmem_P = tmem_base;
    uint32_t tmem_O = tmem_base + 4096;
    
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 16384);
        tma_load_2d_cta_fn(&tma_Q, mbar_Q, smem_Q0, 0, bh * S + row_base);
        tma_load_2d_cta_fn(&tma_Q, mbar_Q, smem_Q1, 64, bh * S + row_base);
        
        if (S > 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_KV[0], 32768);
            tma_load_2d_cta_fn(&tma_K, &mbar_KV[0], smem_K0[0], 0, bh * S);
            tma_load_2d_cta_fn(&tma_K, &mbar_KV[0], smem_K1[0], 64, bh * S);
            tma_load_2d_cta_fn(&tma_V, &mbar_KV[0], smem_V0[0], 0, bh * S);
            tma_load_2d_cta_fn(&tma_V, &mbar_KV[0], smem_V1[0], 64, bh * S);
        }
    }
    mbarrier_wait_fn(mbar_Q, 0);
    fence_proxy_async_fn();

    float global_max = -INFINITY;
    float global_sum = 0.0f;
    
    uint32_t phase[2] = {0, 0};
    uint32_t idx = 0;
    int cur_kv = 0;

    uint64_t desc_Q0 = make_smem_desc_128B(smem_Q0, 1, 1024);
    uint64_t desc_Q1 = make_smem_desc_128B(smem_Q1, 1, 1024);
    uint64_t desc_S = make_smem_desc_128B(smem_S, 1, 1024);
    
    uint32_t idesc_QK = make_instr_desc_fn(64, 64);
    uint32_t idesc_SV = make_instr_desc_fn_trans_B(64, 64);

    for (; idx < S/64; idx++) {
        mbarrier_wait_fn(&mbar_KV[cur_kv], phase[cur_kv]);
        fence_proxy_async_fn();
        phase[cur_kv] ^= 1;
        
        if (idx + 1 < S/64) {
            if (tid == 0) {
                mbarrier_arrive_and_expect_tx_fn(&mbar_KV[cur_kv ^ 1], 32768);
                tma_load_2d_cta_fn(&tma_K, &mbar_KV[cur_kv ^ 1], smem_K0[cur_kv ^ 1], 0, bh * S + (idx + 1) * 64);
                tma_load_2d_cta_fn(&tma_K, &mbar_KV[cur_kv ^ 1], smem_K1[cur_kv ^ 1], 64, bh * S + (idx + 1) * 64);
                tma_load_2d_cta_fn(&tma_V, &mbar_KV[cur_kv ^ 1], smem_V0[cur_kv ^ 1], 0, bh * S + (idx + 1) * 64);
                tma_load_2d_cta_fn(&tma_V, &mbar_KV[cur_kv ^ 1], smem_V1[cur_kv ^ 1], 64, bh * S + (idx + 1) * 64);
            }
        }
        cur_kv ^= 1;
        
        __syncthreads(); 
        
        float local_max = -INFINITY;
        uint32_t r_idx = tid % 128;
        
        for (uint32_t k_step = 0; k_step < 4; ++k_step) {
            umma_f16_cg2(tmem_P, add_base_offset(desc_Q0, k_step * 16 * 2), add_base_offset(desc_K0[cur_kv], k_step * 16 * 2), idesc_QK, (k_step == 0) ? 0 : 1, 1);
        }
        for (uint32_t k_step = 0; k_step < 4; ++k_step) {
            umma_f16_cg2(tmem_P, add_base_offset(desc_Q1, k_step * 16 * 2), add_base_offset(desc_K1[cur_kv], k_step * 16 * 2), idesc_QK, 1, 1);
        }
        
        umma_commit_2sm_fn(mbar_Q);
        mbarrier_wait_fn(mbar_Q, 0); 
        
        for (uint32_t col = 0; col < 64; col += 4) {
            uint32_t val = tmem_P + (r_idx << 16) + col;
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(val));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float f0 = __uint_as_float(r0) * (1.0f / sqrtf(128.0f));
            float f1 = __uint_as_float(r1) * (1.0f / sqrtf(128.0f));
            float f2 = __uint_as_float(r2) * (1.0f / sqrtf(128.0f));
            float f3 = __uint_as_float(r3) * (1.0f / sqrtf(128.0f));
            
            local_max = fmaxf(local_max, fmaxf(fmaxf(f0, f1), fmaxf(f2, f3)));
        }
        
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, 1));
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, 2));
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, 4));
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, 8));
        
        float prev_max = global_max;
        float new_max = fmaxf(prev_max, local_max);
        float scale = __expf(prev_max - new_max);
        global_sum *= scale;
        global_max = new_max;
        
        float local_sum = 0;
        for (uint32_t col = 0; col < 64; col += 4) {
            uint32_t val = tmem_P + (r_idx << 16) + col;
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(val));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float f0 = __uint_as_float(r0) * (1.0f / sqrtf(128.0f));
            float f1 = __uint_as_float(r1) * (1.0f / sqrtf(128.0f));
            float f2 = __uint_as_float(r2) * (1.0f / sqrtf(128.0f));
            float f3 = __uint_as_float(r3) * (1.0f / sqrtf(128.0f));
            
            f0 = __expf(f0 - new_max);
            f1 = __expf(f1 - new_max);
            f2 = __expf(f2 - new_max);
            f3 = __expf(f3 - new_max);
            
            local_sum += (f0 + f1 + f2 + f3);
            
            int x_int4 = col >> 3;
            int swizzled_x = (r_idx & 7) ^ x_int4;
            int swizzled_col = (swizzled_x << 3) | (col & 7);
            smem_S[(r_idx << 6) | swizzled_col] = __float2bfloat16(f0);
            smem_S[(r_idx << 6) | swizzled_col + 1] = __float2bfloat16(f1);
            smem_S[(r_idx << 6) | swizzled_col + 2] = __float2bfloat16(f2);
            smem_S[(r_idx << 6) | swizzled_col + 3] = __float2bfloat16(f3);
        }
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, 1);
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, 2);
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, 4);
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, 8);
        
        global_sum += local_sum;
        
        for (uint32_t col = 0; col < 64; col += 4) {
            uint32_t val_O = tmem_O + (r_idx << 16) + col;
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(val_O));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float f0 = __uint_as_float(r0) * scale;
            float f1 = __uint_as_float(r1) * scale;
            float f2 = __uint_as_float(r2) * scale;
            float f3 = __uint_as_float(r3) * scale;
            
            uint32_t c_u32[4];
            c_u32[0] = __float_as_uint(f0);
            c_u32[1] = __float_as_uint(f1);
            c_u32[2] = __float_as_uint(f2);
            c_u32[3] = __float_as_uint(f3);
            
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
                   :: "r"(val_O), "r"(c_u32[0]), "r"(c_u32[1]), "r"(c_u32[2]), "r"(c_u32[3]) : "memory");
        }
        
        __syncthreads(); // Wait for TMEM O scaling finishes before issuing S @ V
        
        for (uint32_t n_step = 0; n_step < 2; ++n_step) {
            uint64_t desc_V_n = (n_step == 0) ? make_smem_desc_128B(smem_V0[cur_kv], 8192, 1024) : make_smem_desc_128B(smem_V1[cur_kv], 8192, 1024);
            for (uint32_t k_step = 0; k_step < 4; ++k_step) {
                umma_f16_cg2(tmem_O, add_base_offset(desc_S, k_step * 16 * 2), add_base_offset(desc_V_n, k_step * 16 * 128), idesc_SV, (k_step == 0) ? 0 : 1, 1);
            }
        }
        
        umma_commit_2sm_fn(&mbar_KV[cur_kv ^ 1]); 
        mbarrier_wait_fn(&mbar_KV[cur_kv ^ 1], phase[cur_kv ^ 1]); 
        
        __syncthreads(); 
    }
    
    for (uint32_t col = 0; col < 64; col += 4) {
        uint32_t val_O = tmem_O + (tid << 16) + col;
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
           : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(val_O));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float f0 = __uint_as_float(r0) / global_sum;
        float f1 = __uint_as_float(r1) / global_sum;
        float f2 = __uint_as_float(r2) / global_sum;
        float f3 = __uint_as_float(r3) / global_sum;
        
        uint32_t nc = col;
        __nv_bfloat16* out = gmem_O + bh * S * 128 + row_base * 128 + nc;
        if (row_base < S && nc < 128) out[0] = __float2bfloat16(f0);
        if (row_base < S && nc + 1 < 128) out[1] = __float2bfloat16(f1);
        if (row_base < S && nc + 2 < 128) out[2] = __float2bfloat16(f2);
        if (row_base < S && nc + 3 < 128) out[3] = __float2bfloat16(f3);
    }
    
    for (uint32_t col = 0; col < 64; col += 4) {
        uint32_t val_O = tmem_O + 4096 + (tid << 16) + col;
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
           : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(val_O));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float f0 = __uint_as_float(r0) / global_sum;
        float f1 = __uint_as_float(r1) / global_sum;
        float f2 = __uint_as_float(r2) / global_sum;
        float f3 = __uint_as_float(r3) / global_sum;
        
        uint32_t nc = col + 64;
        __nv_bfloat16* out = gmem_O + bh * S * 128 + row_base * 128 + nc;
        if (row_base < S && nc < 128) out[0] = __float2bfloat16(f0);
        if (row_base < S && nc + 1 < 128) out[1] = __float2bfloat16(f1);
        if (row_base < S && nc + 2 < 128) out[2] = __float2bfloat16(f2);
        if (row_base < S && nc + 3 < 128) out[3] = __float2bfloat16(f3);
    }
    
    if (tid == 0) {
        *(gmem_LSE + bh * S + row_base) = global_max + __logf(global_sum);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    uint32_t B = Q.size(0);
    uint32_t H = Q.size(1);
    uint32_t S = Q.size(2);
    uint32_t D = Q.size(3); 

    CUtensorMap tma_Q, tma_K, tma_V;
    __nv_bfloat16* q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* k_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* v_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());

    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, q_ptr, D, B * H * S, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, k_ptr, D, B * H * S, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, v_ptr, D, B * H * S, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    __nv_bfloat16* o_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_data = static_cast<float*>(LSE.data_ptr());

    dim3 grid(B * H, S / 64);
    dim3 block(128); 

    uint32_t smem_bytes = 96 * 1024;
    CUDA_CHECK(cudaFuncSetAttribute(mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_bytes;
    config.stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(&config, mha_kernel, tma_Q, tma_K, tma_V, o_data, lse_data, S));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id))));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_optimized_cuda::run);

} // namespace tvm_ffi_optimized_cuda