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

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_load_2d_cg2_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t ba = (uint32_t)__cvta_generic_to_shared(&bar[0]) & 0xFEFFFFFF; 
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(sa), "l"((uint64_t)d), "r"(c0), "r"(c1), "r"(ba) : "memory");
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

__device__ __forceinline__ uint64_t make_smem_desc_128B_at_y_offset(void* smem_ptr, uint32_t y_offset) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    addr += y_offset * 128; 
    return make_smem_desc_128B((void*)addr, 1, 1024);
}

__device__ __forceinline__ uint64_t make_smem_desc_mnmaj_128B(void* smem_ptr, void* pattern_start_ptr, uint32_t sbo, uint32_t lbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint32_t pattern_addr = (uint32_t)__cvta_generic_to_shared(pattern_start_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16; 
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32; 
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    uint32_t base_offset = (pattern_addr >> 7) & 0x7;
    d |= (uint64_t)base_offset << 49;
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

__device__ __forceinline__ void umma_f16_cg2_scale(
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

struct Tile128x128Swizzled {
    uint4 row[64][128]; // 32768 bytes
};

__global__ void mha_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* gmem_O,
    float* gmem_LSE,
    uint32_t S)
{
    extern __shared__ __align__(1024) char smem_pool[];
    Tile128x128Swizzled* smem_Q = (Tile128x128Swizzled*)smem_pool;                
    Tile128x128Swizzled* smem_K[2] = {(Tile128x128Swizzled*)(smem_pool + 32768), (Tile128x128Swizzled*)(smem_pool + 65536)};      
    Tile128x128Swizzled* smem_V[2] = {(Tile128x128Swizzled*)(smem_pool + 98304), (Tile128x128Swizzled*)(smem_pool + 131072)};      
    Tile128x128Swizzled* smem_P = (Tile128x128Swizzled*)(smem_pool + 163840);     

    uint64_t* mbar_Q = (uint64_t*)(smem_pool + 196608);                  
    uint64_t* mbar_KV = (uint64_t*)(smem_pool + 196616);                 

    uint32_t tid = threadIdx.x;
    uint32_t s_blk = blockIdx.y;
    uint32_t bh = blockIdx.x;
    uint32_t row_base = s_blk * 128;
    uint32_t wg_id = tid / 128;
    uint32_t row = (tid % 128);

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(&mbar_KV[0], 1);
        init_smem_barrier_fn(&mbar_KV[1], 1);
    }
    __syncthreads();

    uint32_t tmem_base;
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_base, 256); 
    }
    __syncthreads();
    
    uint32_t tmem_P = tmem_base;
    uint32_t tmem_O = tmem_base + 8192;
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 32768);
        tma_load_2d_cg2_fn(&tma_Q, mbar_Q, smem_Q, 0, bh * S + row_base);
        tma_load_2d_cg2_fn(&tma_Q, mbar_Q, (char*)smem_Q + 16384, 64, bh * S + row_base);
    }
    if (threadIdx.x == 0 && S > 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar_KV[0], 65536);
        tma_load_2d_cg2_fn(&tma_K, &mbar_KV[0], smem_K[0], 0, bh * S);
        tma_load_2d_cg2_fn(&tma_K, &mbar_KV[0], (char*)smem_K[0] + 16384, 64, bh * S);
        tma_load_2d_cg2_fn(&tma_V, &mbar_KV[0], smem_V[0], 0, bh * S);
        tma_load_2d_cg2_fn(&tma_V, &mbar_KV[0], (char*)smem_V[0] + 16384, 64, bh * S);
    }
    
    mbarrier_wait_fn(mbar_Q, 0);
    fence_proxy_async_fn();

    float global_max[2] = {-INFINITY, -INFINITY};
    float global_sum[2] = {0.0f, 0.0f};
    float block_P_packed[2][64/4][4]; 
    
    uint32_t phase[2] = {0, 0};
    uint32_t idx = 0;
    int kv_idx = 1;

    uint32_t idesc_QK = make_instr_desc_fn(64, 64);
    uint32_t idesc_SV = make_instr_desc_fn_trans_B(64, 64);

    for (; idx < S/128; idx++) {
        int cur_kv = kv_idx ^ 1;
        
        mbarrier_wait_fn(&mbar_KV[cur_kv], phase[cur_kv]);
        fence_proxy_async_fn();
        phase[cur_kv] ^= 1;
        
        if (idx + 1 < S/128) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&mbar_KV[kv_idx], 65536);
                tma_load_2d_cg2_fn(&tma_K, &mbar_KV[kv_idx], smem_K[kv_idx], 0, bh * S + (idx + 1) * 128);
                tma_load_2d_cg2_fn(&tma_K, &mbar_KV[kv_idx], (char*)smem_K[kv_idx] + 16384, 64, bh * S + (idx + 1) * 128);
                tma_load_2d_cg2_fn(&tma_V, &mbar_KV[kv_idx], smem_V[kv_idx], 0, bh * S + (idx + 1) * 128);
                tma_load_2d_cg2_fn(&tma_V, &mbar_KV[kv_idx], (char*)smem_V[kv_idx] + 16384, 64, bh * S + (idx + 1) * 128);
            }
        }
        kv_idx ^= 1;
        
        __syncthreads(); 
        
        float local_max = -INFINITY;
        
        for (uint32_t k_step = 0; k_step < 4; ++k_step) {
            uint64_t desc_Q_k = make_smem_desc_128B_at_y_offset(smem_Q, k_step * 16);
            uint64_t desc_K_k = make_smem_desc_128B_at_y_offset(smem_K[cur_kv], k_step * 16);
            
            if (wg_id == 0) {
                if (k_step == 0) {
                    umma_f16_cg2_scale(tmem_P, desc_Q_k, desc_K_k, idesc_QK, 0, 128);
                } else {
                    umma_f16_cg2_scale(tmem_P, desc_Q_k, desc_K_k, idesc_QK, 1, 128);
                }
            } else {
                if (k_step == 0) {
                    umma_f16_cg2_scale(tmem_P + 4096, desc_Q_k, desc_K_k, idesc_QK, 0, 128);
                } else {
                    umma_f16_cg2_scale(tmem_P + 4096, desc_Q_k, desc_K_k, idesc_QK, 1, 128);
                }
            }
        }
        
        for (uint32_t k_step = 0; k_step < 4; ++k_step) {
            uint64_t desc_Q_k = make_smem_desc_128B_at_y_offset((char*)smem_Q + 16384, k_step * 16);
            uint64_t desc_K_k = make_smem_desc_128B_at_y_offset((char*)smem_K[cur_kv] + 16384, k_step * 16);
            
            if (wg_id == 0) {
                umma_f16_cg2_scale(tmem_P, desc_Q_k, desc_K_k, idesc_QK, 1, 128);
            } else {
                umma_f16_cg2_scale(tmem_P + 4096, desc_Q_k, desc_K_k, idesc_QK, 1, 128);
            }
        }
        
        umma_commit_2sm_fn(mbar_Q); 
        mbarrier_wait_fn(mbar_Q, 0);
        
        for (uint32_t col = 0; col < 64; col += 4) {
            uint32_t r0, r1, r2, r3;
            if (wg_id == 0) {
                tmem_load_4x_fn(col, &r0, &r1, &r2, &r3);
            } else {
                tmem_load_4x_fn(col + 64, &r0, &r1, &r2, &r3);
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            
            block_P_packed[wg_id][col/4][0] = f0;
            block_P_packed[wg_id][col/4][1] = f1;
            block_P_packed[wg_id][col/4][2] = f2;
            block_P_packed[wg_id][col/4][3] = f3;
            
            local_max = fmaxf(local_max, fmaxf(fmaxf(f0, f1), fmaxf(f2, f3)));
        }
        
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, 1));
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, 2));
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, 4));
        
        float prev_max = global_max[wg_id];
        float new_max = fmaxf(prev_max, local_max);
        float scale = __expf(prev_max - new_max);
        global_sum[wg_id] *= scale;
        global_max[wg_id] = new_max;
        
        float local_sum = 0;
        for (uint32_t col = 0; col < 64; col += 4) {
            float f0 = block_P_packed[wg_id][col/4][0];
            float f1 = block_P_packed[wg_id][col/4][1];
            float f2 = block_P_packed[wg_id][col/4][2];
            float f3 = block_P_packed[wg_id][col/4][3];
            
            f0 = __expf(f0 - local_max);
            f1 = __expf(f1 - local_max);
            f2 = __expf(f2 - local_max);
            f3 = __expf(f3 - local_max);
            
            block_P_packed[wg_id][col/4][0] = f0;
            block_P_packed[wg_id][col/4][1] = f1;
            block_P_packed[wg_id][col/4][2] = f2;
            block_P_packed[wg_id][col/4][3] = f3;
            
            local_sum += (f0 + f1 + f2 + f3);
        }
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, 1);
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, 2);
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, 4);
        
        global_sum[wg_id] += local_sum;
        
        for (uint32_t col = 0; col < 64; col += 4) {
            float f0 = block_P_packed[wg_id][col/4][0];
            float f1 = block_P_packed[wg_id][col/4][1];
            float f2 = block_P_packed[wg_id][col/4][2];
            float f3 = block_P_packed[wg_id][col/4][3];
            
            f0 *= scale;
            f1 *= scale;
            f2 *= scale;
            f3 *= scale;
            
            block_P_packed[wg_id][col/4][0] = f0;
            block_P_packed[wg_id][col/4][1] = f1;
            block_P_packed[wg_id][col/4][2] = f2;
            block_P_packed[wg_id][col/4][3] = f3;
        }
        
        __syncthreads(); 
        
        for (uint32_t c = 0; c < 64; c += 4) {
            uint32_t col_u32[4];
            col_u32[0] = __float_as_uint(__bfloat162float(__float2bfloat16(block_P_packed[wg_id][c/4][0])));
            col_u32[1] = __float_as_uint(__bfloat162float(__float2bfloat16(block_P_packed[wg_id][c/4][1])));
            col_u32[2] = __float_as_uint(__bfloat162float(__float2bfloat16(block_P_packed[wg_id][c/4][2])));
            col_u32[3] = __float_as_uint(__bfloat162float(__float2bfloat16(block_P_packed[wg_id][c/4][3])));
            
            uint32_t phys_x = ((row % 8) ^ (c / 8)) * 8 + (c % 8);
            uint32_t addr = (uint32_t)__cvta_generic_to_shared(&smem_P->row[row][phys_x]);
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                         :: "r"(addr), "r"(col_u32[0]), "r"(col_u32[1]), "r"(col_u32[2]), "r"(col_u32[3]) : "memory");
        }
        __syncthreads();
        
        for (uint32_t k_step = 0; k_step < 4; ++k_step) {
            uint64_t desc_S_k = make_smem_desc_128B_at_y_offset(smem_P, k_step * 16);
            uint64_t desc_V0_k = make_smem_desc_mnmaj_128B(smem_V[cur_kv], smem_V[cur_kv], 1024, 16384);
            desc_V0_k = make_smem_desc_128B_at_y_offset(smem_V[cur_kv], k_step * 16);
            
            if (wg_id == 0) {
                if (k_step == 0) {
                    umma_f16_cg2_scale(tmem_O, desc_S_k, desc_V0_k, idesc_SV, 0, 128);
                } else {
                    umma_f16_cg2_scale(tmem_O, desc_S_k, desc_V0_k, idesc_SV, 1, 128);
                }
            } else {
                if (k_step == 0) {
                    umma_f16_cg2_scale(tmem_O + 4096, desc_S_k, desc_V0_k, idesc_SV, 0, 128);
                } else {
                    umma_f16_cg2_scale(tmem_O + 4096, desc_S_k, desc_V0_k, idesc_SV, 1, 128);
                }
            }
        }
        
        for (uint32_t k_step = 0; k_step < 4; ++k_step) {
            uint64_t desc_S_k = make_smem_desc_128B_at_y_offset(smem_P, k_step * 16);
            uint64_t desc_V1_k = make_smem_desc_128B_at_y_offset((char*)smem_V[cur_kv] + 16384, k_step * 16);
            
            if (wg_id == 0) {
                umma_f16_cg2_scale(tmem_O, desc_S_k, desc_V1_k, idesc_SV, 1, 128);
            } else {
                umma_f16_cg2_scale(tmem_O + 4096, desc_S_k, desc_V1_k, idesc_SV, 1, 128);
            }
        }
        
        umma_commit_2sm_fn(&mbar_KV[cur_kv]);
        mbarrier_wait_fn(&mbar_KV[cur_kv], phase[cur_kv]); // Wait for S @ V to conclude
        
        __syncthreads(); 
    }
    
    for (uint32_t col = 0; col < 64; col += 4) {
        uint32_t r0, r1, r2, r3;
        if (wg_id == 0) {
            tmem_load_4x_fn(col, &r0, &r1, &r2, &r3);
        } else {
            tmem_load_4x_fn(col + 64, &r0, &r1, &r2, &r3);
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float f0 = __uint_as_float(r0) / global_sum[wg_id];
        float f1 = __uint_as_float(r1) / global_sum[wg_id];
        float f2 = __uint_as_float(r2) / global_sum[wg_id];
        float f3 = __uint_as_float(r3) / global_sum[wg_id];
        
        uint32_t nc = col;
        __nv_bfloat16* out = gmem_O + bh * S * 128 + row * 128 + nc;
        if (row_base + row < S && nc < 128) out[0] = __float2bfloat16(f0);
        if (row_base + row < S && nc + 1 < 128) out[1] = __float2bfloat16(f1);
        if (row_base + row < S && nc + 2 < 128) out[2] = __float2bfloat16(f2);
        if (row_base + row < S && nc + 3 < 128) out[3] = __float2bfloat16(f3);
    }
    
    for (uint32_t col = 0; col < 64; col += 4) {
        uint32_t r0, r1, r2, r3;
        if (wg_id == 0) {
            tmem_load_4x_fn(64 + col, &r0, &r1, &r2, &r3);
        } else {
            tmem_load_4x_fn(128 + col, &r0, &r1, &r2, &r3);
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float f0 = __uint_as_float(r0) / global_sum[wg_id];
        float f1 = __uint_as_float(r1) / global_sum[wg_id];
        float f2 = __uint_as_float(r2) / global_sum[wg_id];
        float f3 = __uint_as_float(r3) / global_sum[wg_id];
        
        uint32_t nc = col + 64;
        __nv_bfloat16* out = gmem_O + bh * S * 128 + row * 128 + nc;
        if (row_base + row < S && nc < 128) out[0] = __float2bfloat16(f0);
        if (row_base + row < S && nc + 1 < 128) out[1] = __float2bfloat16(f1);
        if (row_base + row < S && nc + 2 < 128) out[2] = __float2bfloat16(f2);
        if (row_base + row < S && nc + 3 < 128) out[3] = __float2bfloat16(f3);
    }
    
    if (tid == 128 * wg_id) {
        float lse = global_max[wg_id] + __logf(global_sum[wg_id]);
        *(gmem_LSE + bh * S + row_base + row) = lse;
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

    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, q_ptr, D, B * H * S, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, k_ptr, D, B * H * S, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, v_ptr, D, B * H * S, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    __nv_bfloat16* o_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_data = static_cast<float*>(LSE.data_ptr());

    dim3 grid(B * H, S / 128);
    dim3 block(256); 

    uint32_t smem_bytes = 197 * 1024;
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