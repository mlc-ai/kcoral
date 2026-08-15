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

__device__ __forceinline__ float expf_emulated(float x) {
    // Cody-Waite style range reduction mapped to powers of 2.
    // Hardware ex2.approx typically expects input in [-128, 128]. 
    // We strictly clamp x to <= 127.99 to ensure integer part n <= 127.
    if (x > 127.99f) x = 127.99f;
    
    int n = __floorf(x);
    float f = x - n;
    
    // Approximation of 2^f using optimized Horner's Method polynomials.
    // Minimal relative error bounded tightly over interval [0, 1).
    float f2 = f * f;
    float f3 = f2 * f;
    float approx = 1.0f + f * (0.6951f + f * (0.2276f + f * 0.0771f));
    
    // Apply integer offset N directly inside IEEE 754 exponent field limits.
    uint32_t int_part = ((n + 127) & 0xFF) << 23;
    uint32_t frac_bits = __float_as_uint(approx);
    frac_bits &= 0x7FFFFF; // Retain purely the mantissa fraction component bits
    
    return __uint_as_float(int_part | frac_bits);
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
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
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
    constexpr uint32_t LBO = 1;
    constexpr uint32_t SBO = 1024; // 8U * 128U
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
    constexpr uint32_t SBO = 1024; // 128 * 8U = 1024
    constexpr uint32_t LBO = 16384; // (64 / 8U) * SBO
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

__device__ __forceinline__ void umma_f16_cg1_tmem_a_fn(
    uint32_t tmem_d, uint32_t tmem_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
        :: "r"(tmem_d), "r"(tmem_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_1sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];" :: "r"(a));
}

__device__ __forceinline__ void cp_async_cg(uint32_t tmem_addr, void* smem_ptr, int bytes) {
    uint32_t smem_int = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.32b.shared::cta.tm [%0], [%1], 16384;\n"
        :: "r"(tmem_addr), "r"(smem_int));
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
    if (q_offset >= S_len) return;
    
    extern __shared__ __align__(1024) uint8_t smem_pool[];
    __nv_bfloat16* smem_Q0 = (__nv_bfloat16*)(smem_pool + 0);       
    __nv_bfloat16* smem_Q1 = (__nv_bfloat16*)(smem_pool + 32768);    
    __nv_bfloat16* smem_K0 = (__nv_bfloat16*)(smem_pool + 65536);    
    __nv_bfloat16* smem_K1 = (__nv_bfloat16*)(smem_pool + 98304);    
    __nv_bfloat16* smem_V0 = (__nv_bfloat16*)(smem_pool + 131072);   
    __nv_bfloat16* smem_V1 = (__nv_bfloat16*)(smem_pool + 163840);   
    __nv_bfloat16* smem_P  = (__nv_bfloat16*)(smem_pool + 196608);    
    
    uint64_t* bar_Q = (uint64_t*)(smem_pool + 229376);
    uint64_t* bar_KV0 = (uint64_t*)(smem_pool + 229384);
    uint64_t* bar_KV1 = (uint64_t*)(smem_pool + 229392);
    uint32_t* tmem_addrs = (uint32_t*)(smem_pool + 229400);
    
    uint32_t tid = threadIdx.x;
    uint32_t wg_id = tid / 128; // Identifies execution responsibility within the 512 thread block
    uint32_t tid_wg = tid % 128;
    
    if (tid == 0) {
        init_smem_barrier(bar_Q, 1);
        init_smem_barrier(bar_KV0, 1);
        init_smem_barrier(bar_KV1, 1);
        tmem_alloc(&tmem_addrs[0], 128);
        tmem_alloc(&tmem_addrs[1], 128);
        tmem_alloc(&tmem_addrs[2], 128);
        tmem_alloc(&tmem_addrs[3], 128);
    }
    __syncthreads();
    
    uint32_t tmem_S0_addr = tmem_addrs[0];
    uint32_t tmem_S1_addr = tmem_addrs[1];
    uint32_t tmem_O0_addr = tmem_addrs[2];
    uint32_t tmem_O1_addr = tmem_addrs[3];
    uint32_t tmem_P0_addr = tmem_S0_addr; // Reuse allocated SMEM space post-calculation
    uint32_t tmem_P1_addr = tmem_S1_addr;
    
    uint64_t desc_Q0 = make_smem_desc_k_major(smem_Q0);
    uint64_t desc_Q1 = make_smem_desc_k_major(smem_Q1);
    uint64_t desc_K0 = make_smem_desc_k_major(smem_K0);
    uint64_t desc_K1 = make_smem_desc_k_major(smem_K1);
    uint64_t desc_V0 = make_smem_desc_mn_major(smem_V0);
    uint64_t desc_V1 = make_smem_desc_mn_major(smem_V1);
    
    uint32_t idesc_QK = (1u << 4) | (1u << 7) | (1u << 10) | ((128 / 8) << 17) | ((128 / 16) << 24);
    uint32_t idesc_PV = (1u << 4) | (1u << 7) | (1u << 10) | (1u << 16) | ((128 / 8) << 17) | ((128 / 16) << 24);
    
    if (wg_id == 3) {
        if (tid_wg == 0) {
            mbarrier_arrive_and_expect_tx(bar_Q, 65536); 
            tma_load_2d(&tma_Q, bar_Q, smem_Q0, 0, bh_offset + q_offset);
            tma_load_2d(&tma_Q, bar_Q, smem_Q0 + 8192, 64, bh_offset + q_offset);
            
            tma_load_2d(&tma_Q, bar_Q, smem_Q1, 0, bh_offset + q_offset + 128);
            tma_load_2d(&tma_Q, bar_Q, smem_Q1 + 8192, 64, bh_offset + q_offset + 128);
            
            mbarrier_arrive_and_expect_tx(bar_KV0, 32768);
            tma_load_2d(&tma_K, bar_KV0, smem_K0, 0, bh_offset + 0 * 128);
            tma_load_2d(&tma_K, bar_KV0, smem_K0 + 8192, 64, bh_offset + 0 * 128);
            tma_load_2d(&tma_V, bar_KV0, smem_V0, 0, bh_offset + 0 * 128);
            tma_load_2d(&tma_V, bar_KV0, smem_V0 + 8192, 64, bh_offset + 0 * 128);
        }
    }
    mbarrier_wait(bar_Q, 0);
    
    // Accumulators properly mapped across wg_id for accurate online normalization updates
    float old_max[2] = {-INFINITY, -INFINITY};
    float l_sum[2] = {0.0f, 0.0f};
    
    // Limit search blocks dynamically enforcing causal mask constraints bounding box logic
    uint32_t block_stop = (q_offset + 255) / 128;
    if (block_stop > (S_len + 127) / 128) {
        block_stop = (S_len + 127) / 128;
    }
    
    uint32_t kv_phase = 0;
    uint32_t phase = 0;
    
    for (uint32_t j = 0; j < block_stop; ++j) {
        uint32_t next_j = j + 1;
        uint32_t buffer_idx = kv_phase;
        uint32_t next_buffer_idx = kv_phase ^ 1;
        
        __nv_bfloat16* cur_K = (buffer_idx == 0) ? smem_K0 : smem_K1;
        __nv_bfloat16* cur_V = (buffer_idx == 0) ? smem_V0 : smem_V1;
        __nv_bfloat16* next_K = (next_buffer_idx == 0) ? smem_K0 : smem_K1;
        __nv_bfloat16* next_V = (next_buffer_idx == 0) ? smem_V0 : smem_V1;
        
        if (next_j < block_stop) {
            if (wg_id == 3) {
                if (tid_wg == 0) {
                    mbarrier_arrive_and_expect_tx(bar_KV0 + next_buffer_idx, 32768);
                    tma_load_2d(&tma_K, bar_KV0 + next_buffer_idx, next_K, 0, bh_offset + next_j * 128);
                    tma_load_2d(&tma_K, bar_KV0 + next_buffer_idx, next_K + 8192, 64, bh_offset + next_j * 128);
                    tma_load_2d(&tma_V, bar_KV0 + next_buffer_idx, next_V, 0, bh_offset + next_j * 128);
                    tma_load_2d(&tma_V, bar_KV0 + next_buffer_idx, next_V + 8192, 64, bh_offset + next_j * 128);
                }
            }
        }
        
        mbarrier_wait(bar_KV0 + buffer_idx, kv_phase);
        kv_phase ^= 1;
        
        // Explicit asynchronous proxy memory fence resolving generic-to-async SMEM coherency delay 
        fence_proxy_async_fn(); 
        
        if (wg_id == 0) {
            if (tid_wg == 0) {
                for (uint32_t i = 0; i < 2; ++i) {
                    uint64_t desc_Q_i = (i == 0) ? desc_Q0 : desc_Q1;
                    uint64_t desc_K_i = (i == 0) ? desc_K0 : desc_K1;
                    uint32_t tmem_S_i = (i == 0) ? tmem_S0_addr : tmem_S1_addr;
                    
                    __nv_bfloat16* cur_K_ptr = (buffer_idx == 0) ? smem_K0 : smem_K1;
                    uint64_t desc_K_cur = (buffer_idx == 0) ? desc_K0 : desc_K1;
                    
                    for (uint32_t k_step = 0; k_step < 4; ++k_step) {
                        uint64_t desc_a = desc_Q_i + (k_step * 2);
                        uint64_t desc_b = desc_K_cur + (k_step * 2);
                        
                        // Accumulation predicate mapping logic incorporating outer phases tracking 
                        uint32_t accum = (phase == 0 && i == 0 && k_step == 0) ? 0 : 1;
                        
                        umma_f16_cg1_fn(tmem_S_i, desc_a, desc_b, idesc_QK, accum);
                    }
                }
            }
            umma_commit_1sm_fn(bar_KV0 + buffer_idx);
        }
        mbarrier_wait(bar_KV0 + buffer_idx, kv_phase);
        kv_phase ^= 1;
        
        float local_max[2] = {-INFINITY, -INFINITY};
        
        if (wg_id == 0 || wg_id == 1) {
            uint32_t i = wg_id;
            uint32_t row = tid_wg;
            uint32_t tmem_S_i = (i == 0) ? tmem_S0_addr : tmem_S1_addr;
            __nv_bfloat16* base_ptr = (i == 0) ? smem_P : (smem_P + 16384);
            
            for (uint32_t c_chunk = 0; c_chunk < 8; ++c_chunk) {
                uint32_t col_base = c_chunk * 8;
                uint32_t addr = tmem_S_i + row * 128 + col_base;
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
                
                // Software emulated exponentials hybridized natively increasing aggregate MUFU throughput capacity
                for(int c = 0; c < 8; ++c) {
                    uint32_t global_col = j * 128 + final_col + c;
                    uint32_t global_row = q_offset + i * 128 + row;
                    if (global_col <= global_row && global_col < S_len) {
                        val[c] *= 0.08838834764f;
                        if (final_col + c < 32) {
                            val[c] = fast_expf(val[c]);
                        } else {
                            val[c] = expf_emulated(val[c] * 1.4426950408889634f);
                        }
                    } else {
                        val[c] = 0.0f;
                    }
                    local_max[i] = fmaxf(local_max[i], val[c]);
                }
                
                for(int c = 0; c < 8; ++c) {
                    *write_smem_swizzle(base_ptr, row, final_col + c) = __float2bfloat16(val[c]);
                }
            }
        }
        
        // Reduction over localized sub-blocks utilizing warp shuffle intrinsics across participating WG's
        float max_val[2];
        if (wg_id == 0 || wg_id == 1) {
            uint32_t i = wg_id;
            max_val[i] = local_max[i];
            max_val[i] = fmaxf(max_val[i], __shfl_xor_sync(0xFFFFFFFF, local_max[i], 1));
            max_val[i] = fmaxf(max_val[i], __shfl_xor_sync(0xFFFFFFFF, local_max[i], 2));
            max_val[i] = fmaxf(max_val[i], __shfl_xor_sync(0xFFFFFFFF, local_max[i], 4));
        }
        
        float new_max[2];
        new_max[0] = fmaxf(old_max[0], max_val[0]);
        new_max[1] = fmaxf(old_max[1], max_val[1]);
        
        float local_sum[2] = {0.0f, 0.0f};
        if (wg_id == 0 || wg_id == 1) {
            uint32_t i = wg_id;
            uint32_t row = tid_wg;
            uint32_t tmem_S_i = (i == 0) ? tmem_S0_addr : tmem_S1_addr;
            __nv_bfloat16* base_ptr = (i == 0) ? smem_P : (smem_P + 16384);
            
            for (uint32_t c_chunk = 0; c_chunk < 8; ++c_chunk) {
                uint32_t col_base = c_chunk * 8;
                uint32_t swizzled_col = (((row % 8) ^ (col_base / 8)) * 8) + (col_base % 8);
                uint32_t final_col = swizzled_col;
                
                float val[8];
                for(int c = 0; c < 8; ++c) {
                    val[c] = read_smem_swizzle(base_ptr, row, final_col + c);
                }
                
                for(int c = 0; c < 8; ++c) {
                    if (val[c] != 0.0f) {
                        val[c] = __float2bfloat16(__uint_as_float(val[c]));
                        local_sum[i] += val[c];
                    }
                }
            }
            local_sum[i] *= fast_expf(max_val[i] - new_max[i]);
        }
        
        float new_l_sum[2];
        new_l_sum[0] = local_sum[0];
        new_l_sum[1] = local_sum[1];
        
        new_l_sum[0] += __shfl_xor_sync(0xFFFFFFFF, new_l_sum[0], 1);
        new_l_sum[0] += __shfl_xor_sync(0xFFFFFFFF, new_l_sum[0], 2);
        new_l_sum[0] += __shfl_xor_sync(0xFFFFFFFF, new_l_sum[0], 4);
        
        new_l_sum[1] += __shfl_xor_sync(0xFFFFFFFF, new_l_sum[1], 1);
        new_l_sum[1] += __shfl_xor_sync(0xFFFFFFFF, new_l_sum[1], 2);
        new_l_sum[1] += __shfl_xor_sync(0xFFFFFFFF, new_l_sum[1], 4);
        
        bool rescale_flag[2];
        rescale_flag[0] = (new_max[0] - old_max[0]) > 2.0f;
        rescale_flag[1] = (new_max[1] - old_max[1]) > 2.0f;
        
        float factor[2];
        factor[0] = fast_expf(old_max[0] - new_max[0]);
        factor[1] = fast_expf(old_max[1] - new_max[1]);
        
        l_sum[0] = (rescale_flag[0]) ? (l_sum[0] * factor[0] + new_l_sum[0]) : (l_sum[0] + new_l_sum[0]);
        l_sum[1] = (rescale_flag[1]) ? (l_sum[1] * factor[1] + new_l_sum[1]) : (l_sum[1] + new_l_sum[1]);
        
        old_max[0] = new_max[0];
        old_max[1] = new_max[1];
        
        // Out of band asynchronous multi-source TMEM copies leveraging background commit pipelines latency hiding
        if (wg_id == 0) {
            for (int i = 0; i < 2; ++i) {
                uint32_t tmem_S_i = (i == 0) ? tmem_S0_addr : tmem_S1_addr;
                uint32_t tmem_P_i = (i == 0) ? tmem_P0_addr : tmem_P1_addr;
                __nv_bfloat16* smem_P_ptr = (i == 0) ? smem_P : (smem_P + 16384);
                cp_async_cg(tmem_P_i, smem_P_ptr, 16384);
            }
            asm volatile("cp.async.commit_group;" ::: "memory");
            asm volatile("cp.async.wait_group 0;" ::: "memory");
        }
        
        // Subsequent multi-stage fence synchronizing prior generic SMEM writes visibility to subsequent WGMMA passes
        fence_proxy_async_fn(); 
        
        if (wg_id == 0) {
            if (tid_wg == 0) {
                for (uint32_t i = 0; i < 2; ++i) {
                    uint32_t tmem_O_i = (i == 0) ? tmem_O0_addr : tmem_O1_addr;
                    uint32_t tmem_P_i = (i == 0) ? tmem_P0_addr : tmem_P1_addr;
                    __nv_bfloat16* cur_V_ptr = (buffer_idx == 0) ? smem_V0 : smem_V1;
                    uint64_t desc_V_cur = (buffer_idx == 0) ? desc_V0 : desc_V1;
                    
                    for (uint32_t k_step = 0; k_step < 4; ++k_step) {
                        uint32_t desc_a = tmem_P_i + (k_step * 2);
                        uint64_t desc_b = desc_V_cur + (k_step * 128); 
                        
                        // Consistently incorporate previously computed unnormalized partial product accumulator sequences
                        umma_f16_cg1_tmem_a_fn(tmem_O_i, desc_a, desc_b, idesc_PV, 1);
                    }
                }
            }
            umma_commit_1sm_fn(bar_KV0 + buffer_idx);
        }
        mbarrier_wait(bar_KV0 + buffer_idx, kv_phase);
        kv_phase ^= 1;
        
        phase ^= 1;
    }
    
    // Epilogue phase - Coalesced Vectorization mapping
    if (wg_id == 0) {
        for (uint32_t col = 0; col < 128; col += 4) {
            uint32_t addr = tmem_O0_addr + tid_wg * 128 + col;
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
                v /= l_sum[0];
                if (col + i >= 128 || q_offset + tid_wg >= S_len) {
                    v = 0.0f;
                }
                *write_smem_swizzle(smem_Q0, tid_wg, col + i) = __float2bfloat16(v);
            }
        }
        
        for (uint32_t col = 0; col < 128; col += 4) {
            uint32_t addr = tmem_O1_addr + tid_wg * 128 + col;
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
                v /= l_sum[1];
                if (col + i >= 128 || q_offset + 128 + tid_wg >= S_len) {
                    v = 0.0f;
                }
                *write_smem_swizzle(smem_Q1, tid_wg, col + i) = __float2bfloat16(v);
            }
        }
        
        __syncthreads(); 
        
        for (uint32_t i = tid_wg; i < 128 * 128; i += 128) {
            uint32_t row = i / 128;
            uint32_t col = i % 128;
            if (q_offset + row < S_len) {
                D[(bh_offset + q_offset + row) * 128 + col] = read_smem_swizzle(smem_Q0, row, col);
                D[(bh_offset + q_offset + row) * 128 + col + 64] = read_smem_swizzle(smem_Q1, row, col);
            }
        }
        
        if (tid_wg < 128) {
            uint32_t row = tid_wg;
            if (q_offset + row < S_len) {
                lse_out[bh_offset + q_offset + row] = old_max[0] + __logf(l_sum[0]);
            }
            if (q_offset + 128 + row < S_len) {
                lse_out[bh_offset + q_offset + 128 + row] = old_max[1] + __logf(l_sum[1]);
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
    
    uint32_t num_q_tiles = (S_len + 255) / 256;
    dim3 grid(num_q_tiles, B * H);
    dim3 block(512);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUDA_CHECK(cudaFuncSetAttribute(attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 228 * 1024));
    
    attn_kernel<<<grid, block, 228 * 1024, stream>>>(tma_Q, tma_K, tma_V, O_ptr, LSE_ptr, S_len, D);
    CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_attention