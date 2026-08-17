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
        const char* err_str;                                       \
        cuGetErrorString(_e, &err_str);                            \
        fprintf(stderr, "CU error %s at %s:%d\n",                 \
                err_str, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

// -------------------------------------------------------------------------
// PTX Helper Functions
// -------------------------------------------------------------------------

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float fast_expf(float x) {
    return fast_exp2f_fn(x * 1.4426950408889634f);
}

__device__ __forceinline__ void init_smem_barrier(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase));
}

__device__ __forceinline__ void tma_load_2d(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tmem_alloc(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ uint64_t make_smem_desc_k_major(void* ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    d |= (addr & 0x3FFFF) >> 4;
    uint32_t base_offset = (addr >> 7) & 7;
    d |= ((uint64_t)base_offset) << 49;
    constexpr uint32_t LBO = 1;
    constexpr uint32_t SBO = 1024; 
    d |= ((uint64_t)(LBO & 0x3FFFF) >> 4) << 16;
    d |= ((uint64_t)(SBO & 0x3FFFF) >> 4) << 32;
    d |= (1ULL << 46); 
    d |= (2ULL << 61); // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_mn_major(void* ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    d |= (addr & 0x3FFFF) >> 4;
    uint32_t base_offset = (addr >> 7) & 7;
    d |= ((uint64_t)base_offset) << 49;
    constexpr uint32_t SBO = 1024; 
    constexpr uint32_t LBO = 16384; // (128 / 8U) * 1024 mapped dynamically
    d |= ((uint64_t)(LBO & 0x3FFFF) >> 4) << 16;
    d |= ((uint64_t)(SBO & 0x3FFFF) >> 4) << 32;
    d |= (1ULL << 46); 
    d |= (2ULL << 61); // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
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
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(a));
}

__device__ __forceinline__ __nv_bfloat16* write_smem_swizzle(__nv_bfloat16* ptr, uint32_t r, uint32_t c) {
    uint32_t c_x = c / 8;
    uint32_t s_x = (r % 8) ^ c_x;
    return &(ptr[r * 128 + s_x * 8 + (c % 8)]);
}

__device__ __forceinline__ __nv_bfloat16 read_smem_swizzle(__nv_bfloat16* ptr, uint32_t r, uint32_t c) {
    uint32_t c_x = c / 8;
    uint32_t s_x = (r % 8) ^ c_x;
    return ptr[r * 128 + s_x * 8 + (c % 8)];
}

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    CUtensorMapDataType dataType = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, dataType, 2, globalAddress, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill
    );
}

// -------------------------------------------------------------------------
// Attention Kernel
// -------------------------------------------------------------------------

__global__ void attn_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* D, float* lse_out, uint32_t S_len, uint32_t D_dim)
{
    uint32_t bh = blockIdx.y;
    uint32_t q_tile = blockIdx.x;
    uint32_t q_offset = q_tile * 128;
    uint32_t bh_offset = bh * S_len;
    
    uint32_t tid = threadIdx.x;
    if (q_offset + tid >= S_len) return;
    
    extern __shared__ __align__(1024) uint8_t smem_pool[];
    __nv_bfloat16* smem_Q0_0 = (__nv_bfloat16*)(smem_pool + 0);       
    __nv_bfloat16* smem_Q0_1 = (__nv_bfloat16*)(smem_pool + 16384);    
    __nv_bfloat16* smem_K_0 = (__nv_bfloat16*)(smem_pool + 32768);     
    __nv_bfloat16* smem_K_1 = (__nv_bfloat16*)(smem_pool + 49152);     
    __nv_bfloat16* smem_V_0 = (__nv_bfloat16*)(smem_pool + 65536);     
    __nv_bfloat16* smem_V_1 = (__nv_bfloat16*)(smem_pool + 81920);     
    __nv_bfloat16* smem_P    = (__nv_bfloat16*)(smem_pool + 98304);     
    
    __shared__ alignas(128) uint64_t bar_Q;
    __shared__ alignas(128) uint64_t bar_KV;
    __shared__ alignas(128) uint32_t tmem_addrs[3];
    
    if (tid == 0) {
        init_smem_barrier(&bar_Q, 1);
        init_smem_barrier(&bar_KV, 1);
        tmem_alloc(&tmem_addrs[0], 128);
        tmem_alloc(&tmem_addrs[1], 128);
        tmem_alloc(&tmem_addrs[2], 128);
    }
    __syncthreads();
    
    uint32_t tmem_S_addr = tmem_addrs[0];
    uint32_t tmem_O0_addr = tmem_addrs[1];
    uint32_t tmem_O1_addr = tmem_addrs[2];
    
    uint64_t desc_Q0_0 = make_smem_desc_k_major(smem_Q0_0);
    uint64_t desc_Q0_1 = make_smem_desc_k_major(smem_Q0_1);
    uint64_t desc_K_0 = make_smem_desc_k_major(smem_K_0);
    uint64_t desc_K_1 = make_smem_desc_k_major(smem_K_1);
    uint64_t desc_V_0 = make_smem_desc_mn_major(smem_V_0);
    uint64_t desc_V_1 = make_smem_desc_mn_major(smem_V_1);
    uint64_t desc_P_0 = make_smem_desc_k_major(smem_P);
    uint64_t desc_P_1 = make_smem_desc_k_major((__nv_bfloat16*)smem_P + 8192);
    
    uint32_t idesc_QK = (1u << 4) | (1u << 7) | (1u << 10) | ((128 / 8) << 17) | ((128 / 16) << 24);
    uint32_t idesc_PV = (1u << 4) | (1u << 7) | (1u << 10) | (1u << 16) | ((64 / 8) << 17) | ((128 / 16) << 24);
    
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx(&bar_Q, 32768); 
        
        tma_load_2d(&tma_Q, &bar_Q, smem_Q0_0, 0, bh_offset + q_offset);
        tma_load_2d(&tma_Q, &bar_Q, smem_Q0_1, 64, bh_offset + q_offset);
    }
    mbarrier_wait(&bar_Q, 0);
    
    float old_max = -INFINITY;
    float l_sum = 0.0f;
    
    uint32_t block_stop = q_tile + 1; 
    if (block_stop > (S_len + 127) / 128) {
        block_stop = (S_len + 127) / 128;
    }
    
    uint32_t phase_KV = 0;
    uint32_t phase = 0;
    
    for (uint32_t j = 0; j < block_stop; ++j) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx(&bar_KV, 65536);
            tma_load_2d(&tma_K, &bar_KV, smem_K_0, 0, bh_offset + j * 128);
            tma_load_2d(&tma_K, &bar_KV, smem_K_1, 64, bh_offset + j * 128);
            tma_load_2d(&tma_V, &bar_KV, smem_V_0, 0, bh_offset + j * 128);
            tma_load_2d(&tma_V, &bar_KV, smem_V_1, 64, bh_offset + j * 128);
        }
        
        mbarrier_wait(&bar_KV, phase_KV);
        phase_KV ^= 1;
        
        fence_proxy_async_fn(); 
        
        if (tid == 0) {
            for (uint32_t k_step = 0; k_step < 4; ++k_step) {
                uint64_t desc_a = desc_Q0_0 + (k_step * 2);
                uint64_t desc_b = desc_K_0 + (k_step * 2);
                uint32_t accum = (phase == 0 && k_step == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_S_addr, desc_a, desc_b, idesc_QK, accum);
            }
            for (uint32_t k_step = 0; k_step < 4; ++k_step) {
                uint64_t desc_a = desc_Q0_1 + (k_step * 2);
                uint64_t desc_b = desc_K_1 + (k_step * 2);
                umma_f16_cg1_fn(tmem_S_addr, desc_a, desc_b, idesc_QK, 1);
            }
            umma_commit_1sm_fn(&bar_KV);
        }
        mbarrier_wait(&bar_KV, phase_KV);
        phase_KV ^= 1;
        
        float local_max = -INFINITY;
        uint32_t row = tid;
        uint32_t cur_q_offset = q_offset;
        
        for (uint32_t c_chunk = 0; c_chunk < 16; ++c_chunk) {
            uint32_t col_base = c_chunk * 8;
            uint32_t addr = tmem_S_addr + row * 128 + col_base;
            uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(addr));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float val[8];
            val[0] = __uint_as_float(r0);
            val[1] = __uint_as_float(r1);
            val[2] = __uint_as_float(r2);
            val[3] = __uint_as_float(r3);
            val[4] = __uint_as_float(r4);
            val[5] = __uint_as_float(r5);
            val[6] = __uint_as_float(r6);
            val[7] = __uint_as_float(r7);
            
            uint32_t swizzled_col = (((row % 8) ^ (col_base / 8)) * 8) + (col_base % 8);
            uint32_t final_col = swizzled_col;
            
            for(int c = 0; c < 8; ++c) {
                uint32_t global_col = j * 128 + final_col + c;
                uint32_t global_row = cur_q_offset + row;
                if (global_col <= global_row && global_col < S_len) {
                    val[c] *= 0.08838834764f;
                    val[c] = fast_expf(val[c]);
                } else {
                    val[c] = 0.0f;
                }
                local_max = fmaxf(local_max, val[c]);
            }
        }
        
        float max_val = local_max;
        max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, local_max, 1));
        max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, local_max, 2));
        max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, local_max, 4));
        
        float new_max = fmaxf(old_max, max_val);
        
        l_sum *= fast_expf(max_val - new_max);
        
        float local_sum = 0.0f;
        for (uint32_t c_chunk = 0; c_chunk < 16; ++c_chunk) {
            uint32_t col_base = c_chunk * 8;
            uint32_t addr = tmem_S_addr + row * 128 + col_base;
            uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(addr));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float val[8];
            val[0] = __uint_as_float(r0);
            val[1] = __uint_as_float(r1);
            val[2] = __uint_as_float(r2);
            val[3] = __uint_as_float(r3);
            val[4] = __uint_as_float(r4);
            val[5] = __uint_as_float(r5);
            val[6] = __uint_as_float(r6);
            val[7] = __uint_as_float(r7);
            
            uint32_t swizzled_col = (((row % 8) ^ (col_base / 8)) * 8) + (col_base % 8);
            uint32_t final_col = swizzled_col;
            
            for(int c = 0; c < 8; ++c) {
                uint32_t global_col = j * 128 + final_col + c;
                uint32_t global_row = cur_q_offset + row;
                float v = 0.0f;
                if (global_col <= global_row && global_col < S_len) {
                    v = fast_expf(val[c] * 0.08838834764f - new_max);
                }
                local_sum += v;
                
                if (final_col + c < 64) {
                    *write_smem_swizzle(((__nv_bfloat16*)smem_P), tid, final_col + c) = __float2bfloat16(v);
                } else {
                    *write_smem_swizzle(((__nv_bfloat16*)smem_P) + 8192, tid, final_col + c) = __float2bfloat16(v);
                }
            }
        }
        
        float new_l_sum = local_sum;
        new_l_sum += __shfl_xor_sync(0xFFFFFFFF, new_l_sum, 1);
        new_l_sum += __shfl_xor_sync(0xFFFFFFFF, new_l_sum, 2);
        new_l_sum += __shfl_xor_sync(0xFFFFFFFF, new_l_sum, 4);
        
        if (new_max - old_max > 2.0f) {
            l_sum = l_sum * fast_expf(old_max - new_max) + new_l_sum * fast_expf(max_val - new_max);
        } else {
            l_sum = l_sum + new_l_sum * fast_expf(max_val - new_max);
        }
        old_max = new_max;
        
        __syncthreads(); 
        fence_proxy_async_fn(); 
        
        if (tid == 0) {
            for (uint32_t k_step = 0; k_step < 4; ++k_step) {
                uint64_t desc_a = desc_P_0 + (k_step * 2);
                
                uint64_t desc_b_0 = desc_V_0 + (k_step * 128);
                uint64_t desc_b_1 = desc_V_1 + (k_step * 128);
                
                umma_f16_cg1_fn(tmem_O0_addr, desc_a, desc_b_0, idesc_PV, (k_step == 0 && phase == 0) ? 0 : 1);
                umma_f16_cg1_fn(tmem_O1_addr, desc_a, desc_b_1, idesc_PV, (k_step == 0 && phase == 0) ? 0 : 1);
            }
            for (uint32_t k_step = 0; k_step < 4; ++k_step) {
                uint64_t desc_a = desc_P_1 + (k_step * 2);
                
                uint64_t desc_b_0 = desc_V_0 + ((k_step + 4) * 128);
                uint64_t desc_b_1 = desc_V_1 + ((k_step + 4) * 128);
                
                umma_f16_cg1_fn(tmem_O0_addr, desc_a, desc_b_0, idesc_PV, 1);
                umma_f16_cg1_fn(tmem_O1_addr, desc_a, desc_b_1, idesc_PV, 1);
            }
            umma_commit_1sm_fn(&bar_KV);
        }
        mbarrier_wait(&bar_KV, phase_KV);
        phase_KV ^= 1;
        
        __syncthreads(); 
        
        phase ^= 1;
    }
    
    if (tid < 128) {
        __syncthreads(); 
        
        for (uint32_t col = 0; col < 64; col += 4) {
            uint32_t addr = tmem_O0_addr + tid * 128 + col;
            uint32_t r0, r1, r2, r3;
            
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(addr));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float out_val[4];
            out_val[0] = __uint_as_float(r0);
            out_val[1] = __uint_as_float(r1);
            out_val[2] = __uint_as_float(r2);
            out_val[3] = __uint_as_float(r3);
            
            for(int i = 0; i < 4; ++i) {
                float v = out_val[i];
                v /= l_sum;
                if (col + i >= 64 || q_offset + tid >= S_len) {
                    v = 0.0f;
                }
                *write_smem_swizzle(smem_Q0_0, tid, col + i) = __float2bfloat16(v);
            }
        }
        
        for (uint32_t col = 0; col < 64; col += 4) {
            uint32_t addr = tmem_O1_addr + tid * 128 + col;
            uint32_t r0, r1, r2, r3;
            
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(addr));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float out_val[4];
            out_val[0] = __uint_as_float(r0);
            out_val[1] = __uint_as_float(r1);
            out_val[2] = __uint_as_float(r2);
            out_val[3] = __uint_as_float(r3);
            
            for(int i = 0; i < 4; ++i) {
                float v = out_val[i];
                v /= l_sum;
                if (col + i + 64 >= 128 || q_offset + tid >= S_len) {
                    v = 0.0f;
                }
                *write_smem_swizzle(smem_Q0_1, tid, col + i) = __float2bfloat16(v);
            }
        }
        
        __syncthreads(); 
        
        for (uint32_t i = tid; i < 128 * 128; i += 128) {
            uint32_t row = i / 128;
            uint32_t col = i % 128;
            if (q_offset + row < S_len) {
                D[(bh_offset + q_offset + row) * 128 + col] = read_smem_swizzle(smem_Q0_0, row, col);
                D[(bh_offset + q_offset + row) * 128 + col + 64] = read_smem_swizzle(smem_Q0_1, row, col);
            }
        }
        
        if (tid < 128) {
            uint32_t row = tid;
            if (q_offset + row < S_len) {
                lse_out[bh_offset + q_offset + row] = old_max + __logf(l_sum);
            }
        }
    }
}

// -------------------------------------------------------------------------
// TVM-FFI Binding
// -------------------------------------------------------------------------

namespace tvm_ffi_attention {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0); 
    int64_t H = Q.size(1); 
    int64_t S_len = Q.size(2); 
    int64_t D = Q.size(3); 
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE.data_ptr());
    
    CUtensorMap tma_Q, tma_K, tma_V;
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, (void*)Q_ptr, D, B*H*S_len, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, (void*)K_ptr, D, B*H*S_len, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, (void*)V_ptr, D, B*H*S_len, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    uint32_t num_q_tiles = (S_len + 127) / 128;
    dim3 grid(num_q_tiles, B * H);
    dim3 block(128);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUDA_CHECK(cudaFuncSetAttribute(attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 160 * 1024));
    
    attn_kernel<<<grid, block, 160 * 1024, stream>>>(tma_Q, tma_K, tma_V, O_ptr, LSE_ptr, S_len, D);
    CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_attention