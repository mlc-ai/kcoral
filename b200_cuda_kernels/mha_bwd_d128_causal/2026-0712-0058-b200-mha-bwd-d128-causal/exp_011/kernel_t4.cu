#include <cuda_bf16.h>
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

namespace tvm_ffi_attention_bwd {

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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

__device__ __forceinline__ void tmdb_wait_fn(uint64_t* bar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared.b64 [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])) : "memory");
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ uint64_t make_smem_desc_128B_lbo_sbo(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16; 
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32; 
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint32_t create_desc(uint32_t M, uint32_t N, uint32_t a_major, uint32_t b_major) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (a_major << 15);  
    d |= (b_major << 16);  
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
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

#define tmem_addr(col, row) ((uint32_t)((row << 16) | (col & 0xFFFF)))
__device__ __forceinline__ float tmem_read_f32(uint32_t col, int row) {
    float val;
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(val) : "r"(tmem_addr(col, row)));
    return val;
}

__device__ __forceinline__ void tmem_write_f32(uint32_t col, int row, float val) {
    asm volatile("st.shared.f32 [%0], %1;" :: "r"(tmem_addr(col, row)), "f"(val));
}

__device__ __forceinline__ uint64_t advance_desc(uint64_t desc, int step) {
    return desc + (step * 2);
}

__device__ __forceinline__ uint64_t advance_desc_mn(uint64_t desc, int step) {
    return desc + (step * 128);
}

extern __shared__ __align__(1024) uint8_t smem_pool[];

__device__ __forceinline__ void write_swizzled(__nv_bfloat16* dst, int r, int c, float val) {
    int x_chunk = c / 8;
    int x_rem = c % 8;
    int swizzled_x = ((r % 8) ^ x_chunk) * 8 + x_rem;
    dst[r * 64 + swizzled_x] = __float2bfloat16(val);
}

__device__ __forceinline__ __nv_bfloat16 read_swizzled(const __nv_bfloat16* src, int r, int c) {
    int x_chunk = c / 8;
    int x_rem = c % 8;
    int swizzled_x = ((r % 8) ^ x_chunk) * 8 + x_rem;
    return src[r * 64 + swizzled_x];
}

__device__ __forceinline__ void transpose_64x64(__nv_bfloat16* src, __nv_bfloat16* dst) {
    int tid = threadIdx.x;
    for (int idx = tid; idx < 4096; idx += 128) {
        int r = idx / 64;
        int c = idx % 64;
        
        int x_chunk = c / 8;
        int x_rem = c % 8;
        int swizzled_x = ((r % 8) ^ x_chunk) * 8 + x_rem;
        __nv_bfloat16 val = src[r * 64 + swizzled_x];
        
        int dst_r = c;
        int dst_c = r;
        int dst_x_chunk = dst_c / 8;
        int dst_x_rem = dst_c % 8;
        int dst_swizzled_x = ((dst_r % 8) ^ dst_x_chunk) * 8 + dst_x_rem;
        
        dst[dst_r * 64 + dst_swizzled_x] = val;
    }
}

__global__ void __launch_bounds__(128, 1) bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* L, float* dK_fp32, float* dV_fp32,
    __nv_bfloat16* dQ, int S_len, float scale)
{
    int q_tile = blockIdx.x;
    int head_idx = blockIdx.y;
    int num_tiles = (S_len + 63) / 64;
    if (q_tile >= num_tiles) return;
    
    int q_base = q_tile * 64;
    uint64_t head_offset = head_idx * S_len * 128;
    int tid = threadIdx.x;
    
    // Explicit spacing of 16KB, 16KB, 16KB ... to guarantee 1024B alignment for every slice
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)smem_pool;                
    __nv_bfloat16* smem_Q_T = smem_Q + 8192;                          
    __nv_bfloat16* smem_K = smem_Q_T + 8192;                           
    __nv_bfloat16* smem_V = smem_K + 8192;                             
    __nv_bfloat16* smem_V_T = smem_V + 8192;                            
    __nv_bfloat16* smem_O = smem_V_T + 8192;                            
    __nv_bfloat16* smem_dO = smem_O + 8192;                             
    __nv_bfloat16* smem_P_T = smem_dO + 8192;                           
    __nv_bfloat16* smem_dS_T = smem_P_T + 4096;                         

    float* smem_D = (float*)(smem_dS_T + 4096);                        
    float* smem_LSE = smem_D + 64;                                     
    uint64_t* mbar_Q = (uint64_t*)(smem_LSE + 64);                      
    uint64_t* mbar_K = mbar_Q + 1;                                      
    uint64_t* mbar_V = mbar_K + 1;                                      
    uint64_t* mbar_O = mbar_V + 1;                                      
    uint64_t* mbar_dO = mbar_O + 1;                                     
    uint64_t* mbar_S_T = mbar_dO + 1;                                   
    uint64_t* mbar_dP_T = mbar_S_T + 1;                                 
    uint64_t* mbar_dV = mbar_dP_T + 1;                                  
    uint64_t* mbar_dK = mbar_dV + 1;                                    
    uint64_t* mbar_dQ = mbar_dK + 1;                                    

    if (tid == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_K, 1);
        init_smem_barrier_fn(mbar_V, 1);
        init_smem_barrier_fn(mbar_O, 1);
        init_smem_barrier_fn(mbar_dO, 1);
        init_smem_barrier_fn(mbar_S_T, 1);
        init_smem_barrier_fn(mbar_dP_T, 1);
        init_smem_barrier_fn(mbar_dV, 1);
        init_smem_barrier_fn(mbar_dK, 1);
        init_smem_barrier_fn(mbar_dQ, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    uint32_t tmem_S_T, tmem_dP_T, tmem_dV_0, tmem_dV_1, tmem_dK_0, tmem_dK_1, tmem_dQ_0, tmem_dQ_1;
    
    if (tid == 0) {
        tmem_alloc_fn(&tmem_S_T, 64);
        tmem_alloc_fn(&tmem_dP_T, 64);
        tmem_alloc_fn(&tmem_dV_0, 64);
        tmem_alloc_fn(&tmem_dV_1, 64);
        tmem_alloc_fn(&tmem_dK_0, 64);
        tmem_alloc_fn(&tmem_dK_1, 64);
        tmem_alloc_fn(&tmem_dQ_0, 64);
        tmem_alloc_fn(&tmem_dQ_1, 64);
    }
    __syncthreads();
    
    uint32_t phase_Q = 0, phase_K = 0, phase_V = 0, phase_O = 0, phase_dO = 0;
    uint32_t phase_S_T = 0, phase_dP_T = 0, phase_dV = 0, phase_dK = 0, phase_dQ = 0;
    
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 16384);
        tma_load_2d_fn(&tma_Q, mbar_Q, smem_Q, 0, head_idx * S_len + q_base);
        tma_load_2d_fn(&tma_Q, mbar_Q, (char*)smem_Q + 8192, 64, head_idx * S_len + q_base);
        
        mbarrier_arrive_and_expect_tx_fn(mbar_O, 16384);
        tma_load_2d_fn(&tma_O, mbar_O, smem_O, 0, head_idx * S_len + q_base);
        tma_load_2d_fn(&tma_O, mbar_O, (char*)smem_O + 8192, 64, head_idx * S_len + q_base);
        
        mbarrier_arrive_and_expect_tx_fn(mbar_dO, 16384);
        tma_load_2d_fn(&tma_dO, mbar_dO, smem_dO, 0, head_idx * S_len + q_base);
        tma_load_2d_fn(&tma_dO, mbar_dO, (char*)smem_dO + 8192, 64, head_idx * S_len + q_base);
    }
    
    mbarrier_wait_fn(mbar_Q, phase_Q);
    mbarrier_wait_fn(mbar_O, phase_O);
    mbarrier_wait_fn(mbar_dO, phase_dO);
    phase_Q ^= 1; phase_O ^= 1; phase_dO ^= 1;
    
    if (tid < 64) {
        float sum = 0;
        if (q_base + tid < S_len) {
            for (int d = 0; d < 64; d++) {
                sum += __bfloat162float(read_swizzled(smem_dO, tid, d)) * __bfloat162float(read_swizzled(smem_O, tid, d));
                sum += __bfloat162float(read_swizzled(smem_dO, tid, 64 + d)) * __bfloat162float(read_swizzled(smem_O, tid, 64 + d));
            }
        }
        smem_D[tid] = sum;
        smem_LSE[tid] = (q_base + tid < S_len) ? L[head_idx * S_len + q_base + tid] : 0;
    }
    __syncthreads();
    
    transpose_64x64(smem_Q, smem_Q_T);
    __syncthreads();
    
    // Ensure dQ TMEM tracks are zero-initialized for accumulation across k_tiles
    if (tid < 64) {
        tmem_write_f32(tmem_dQ_0, tid, 0);
        tmem_write_f32(tmem_dQ_1, tid, 0);
    }
    
    uint32_t idesc_S_T = create_desc(64, 64, 0, 0);
    uint32_t idesc_dP_T = create_desc(64, 64, 0, 0);
    uint32_t idesc_dV = create_desc(64, 128, 0, 1);
    uint32_t idesc_dK = create_desc(64, 128, 0, 1);
    uint32_t idesc_dQ = create_desc(64, 128, 0, 1);
    
    for (int k_tile = 0; k_tile <= q_tile && k_tile < num_tiles; k_tile++) {
        int k_base = k_tile * 64;
        
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_K, 16384);
            tma_load_2d_fn(&tma_K, mbar_K, smem_K, 0, head_idx * S_len + k_base);
            tma_load_2d_fn(&tma_K, mbar_K, (char*)smem_K + 8192, 64, head_idx * S_len + k_base);
            
            mbarrier_arrive_and_expect_tx_fn(mbar_V, 16384);
            tma_load_2d_fn(&tma_V, mbar_V, smem_V, 0, head_idx * S_len + k_base);
            tma_load_2d_fn(&tma_V, mbar_V, (char*)smem_V + 8192, 64, head_idx * S_len + k_base);
        }
        
        mbarrier_wait_fn(mbar_K, phase_K);
        mbarrier_wait_fn(mbar_V, phase_V);
        phase_K ^= 1; phase_V ^= 1;
        
        __syncthreads();
        transpose_64x64(smem_V, smem_V_T);
        __syncthreads();
        
        if (tid == 0) {
            uint64_t desc_K = make_smem_desc_128B_lbo_sbo(smem_K, 1, 1024);
            uint64_t desc_Q_T = make_smem_desc_128B_lbo_sbo(smem_Q_T, 8192, 1024);
            uint64_t desc_K_1 = make_smem_desc_128B_lbo_sbo((char*)smem_K + 8192, 1, 1024);
            uint64_t desc_Q_T_1 = make_smem_desc_128B_lbo_sbo((char*)smem_Q_T + 8192, 8192, 1024);
            
            for (int k_step = 0; k_step < 4; k_step++) {
                uint64_t da0 = advance_desc(desc_K, k_step);
                uint64_t da1 = advance_desc(desc_K_1, k_step);
                uint64_t db0 = advance_desc(desc_Q_T, k_step);
                uint64_t db1 = advance_desc(desc_Q_T_1, k_step);
                
                uint32_t accum = (k_step == 0) ? 0 : 1;
                umma_f16_cg1(tmem_S_T, da0, db0, idesc_S_T, accum);
                umma_f16_cg1(tmem_S_T, da1, db1, idesc_S_T, accum);
            }
        }
        tmdb_wait_fn(mbar_S_T);
        mbarrier_wait_fn(mbar_S_T, phase_S_T);
        phase_S_T ^= 1;
        tmem_load_fence_fn();
        
        // Compute Softmax directly on TMEM elements and update matrix to P_T
        if (tid < 64) {
            for (int col = 0; col < 64; col++) {
                float val = tmem_read_f32(tmem_S_T + col, tid);
                float s = val * scale;
                float p = fast_exp2f_fn((s - smem_LSE[tid]) * 1.4426950408889634f);
                if (q_base + col < k_base + tid || q_base + col >= S_len || k_base + tid >= S_len) {
                    p = 0;
                }
                tmem_write_f32(tmem_S_T + col, tid, p);
            }
        }
        tmem_load_fence_fn();
        __syncthreads(); 
        
        // Write P_T to Shared Memory
        if (tid < 4096) {
            int r = tid / 64;
            int c = tid % 64;
            float p_val = tmem_read_f32(tmem_S_T + c, r);
            write_swizzled(smem_P_T, r, c, p_val);
        }
        __syncthreads();
        
        if (tid == 0) {
            uint64_t desc_dO = make_smem_desc_128B_lbo_sbo(smem_dO, 1, 1024);
            uint64_t desc_V_T = make_smem_desc_128B_lbo_sbo(smem_V_T, 1, 1024);
            uint64_t desc_dO_1 = make_smem_desc_128B_lbo_sbo((char*)smem_dO + 8192, 1, 1024);
            uint64_t desc_V_T_1 = make_smem_desc_128B_lbo_sbo((char*)smem_V_T + 8192, 1, 1024);
            
            for (int k_step = 0; k_step < 4; k_step++) {
                uint64_t da0 = advance_desc(desc_dO, k_step);
                uint64_t da1 = advance_desc(desc_dO_1, k_step);
                uint64_t db0 = advance_desc(desc_V_T, k_step);
                uint64_t db1 = advance_desc(desc_V_T_1, k_step);
                
                uint32_t accum = (k_step == 0) ? 0 : 1;
                umma_f16_cg1(tmem_dP_T, da0, db0, idesc_dP_T, accum);
                umma_f16_cg1(tmem_dP_T, da1, db1, idesc_dP_T, accum);
            }
        }
        tmdb_wait_fn(mbar_dP_T);
        mbarrier_wait_fn(mbar_dP_T, phase_dP_T);
        phase_dP_T ^= 1;
        tmem_load_fence_fn();
        
        // Compute dS_T directly on TMEM elements
        if (tid < 64) {
            for (int col = 0; col < 64; col++) {
                float p = tmem_read_f32(tmem_S_T + col, tid);
                float dp = tmem_read_f32(tmem_dP_T + col, tid);
                float ds = p * (dp - smem_D[col]);
                
                if (q_base + col < k_base + tid || q_base + col >= S_len || k_base + tid >= S_len) {
                    ds = 0;
                }
                tmem_write_f32(tmem_dP_T + col, tid, ds);
            }
        }
        tmem_load_fence_fn();
        __syncthreads();
        
        // Write dS_T to Shared Memory
        if (tid < 4096) {
            int r = tid / 64;
            int c = tid % 64;
            float ds_val = tmem_read_f32(tmem_dP_T + c, r);
            write_swizzled(smem_dS_T, r, c, ds_val);
        }
        __syncthreads();
        
        if (tid == 0) {
            uint64_t desc_P_T = make_smem_desc_128B_lbo_sbo(smem_P_T, 1, 1024);
            uint64_t desc_dO = make_smem_desc_128B_lbo_sbo(smem_dO, 8192, 1024);
            uint64_t desc_dO_1 = make_smem_desc_128B_lbo_sbo((char*)smem_dO + 8192, 8192, 1024);
            
            for (int k_step = 0; k_step < 4; k_step++) {
                uint64_t da = advance_desc(desc_P_T, k_step);
                uint64_t db0 = advance_desc_mn(desc_dO, k_step);
                uint64_t db1 = advance_desc_mn(desc_dO_1, k_step);
                
                uint32_t accum = (k_step == 0) ? 0 : 1;
                umma_f16_cg1(tmem_dV_0, da, db0, idesc_dV, accum);
                umma_f16_cg1(tmem_dV_1, da, db1, idesc_dV, accum);
            }
        }
        tmdb_wait_fn(mbar_dV);
        mbarrier_wait_fn(mbar_dV, phase_dV);
        phase_dV ^= 1;
        
        if (tid < 64) {
            float dv0 = tmem_read_f32(tmem_dV_0 + tid, 0);
            float dv1 = tmem_read_f32(tmem_dV_1 + tid, 0);
            if (k_base + tid < S_len) {
                dV_fp32[head_offset + (k_base + tid) * 128 + 0] = dv0;
                dV_fp32[head_offset + (k_base + tid) * 128 + 64] = dv1;
            }
        }
        
        if (tid == 0) {
            uint64_t desc_dS_T = make_smem_desc_128B_lbo_sbo(smem_dS_T, 1, 1024);
            uint64_t desc_Q = make_smem_desc_128B_lbo_sbo(smem_Q, 8192, 1024);
            uint64_t desc_Q_1 = make_smem_desc_128B_lbo_sbo((char*)smem_Q + 8192, 8192, 1024);
            
            for (int k_step = 0; k_step < 4; k_step++) {
                uint64_t da = advance_desc(desc_dS_T, k_step);
                uint64_t db0 = advance_desc_mn(desc_Q, k_step);
                uint64_t db1 = advance_desc_mn(desc_Q_1, k_step);
                
                uint32_t accum = (k_step == 0) ? 0 : 1;
                umma_f16_cg1(tmem_dK_0, da, db0, idesc_dK, accum);
                umma_f16_cg1(tmem_dK_1, da, db1, idesc_dK, accum);
            }
        }
        tmdb_wait_fn(mbar_dK);
        mbarrier_wait_fn(mbar_dK, phase_dK);
        phase_dK ^= 1;
        
        if (tid < 64) {
            float dk0 = tmem_read_f32(tmem_dK_0 + tid, 0);
            float dk1 = tmem_read_f32(tmem_dK_1 + tid, 0);
            if (k_base + tid < S_len) {
                dK_fp32[head_offset + (k_base + tid) * 128 + 0] = dk0;
                dK_fp32[head_offset + (k_base + tid) * 128 + 64] = dk1;
            }
        }
        
        __syncthreads();
        // Transpose dS_T -> dS 
        transpose_64x64(smem_dS_T, smem_P_T); 
        __syncthreads();
        
        if (tid == 0) {
            uint64_t desc_dS = make_smem_desc_128B_lbo_sbo(smem_P_T, 1, 1024);
            uint64_t desc_K = make_smem_desc_128B_lbo_sbo(smem_K, 8192, 1024);
            uint64_t desc_K_1 = make_smem_desc_128B_lbo_sbo((char*)smem_K + 8192, 8192, 1024);
            
            for (int k_step = 0; k_step < 4; k_step++) {
                uint64_t da = advance_desc(desc_dS, k_step);
                uint64_t db0 = advance_desc_mn(desc_K, k_step);
                uint64_t db1 = advance_desc_mn(desc_K_1, k_step);
                
                uint32_t accum_dQ = 1;
                umma_f16_cg1(tmem_dQ_0, da, db0, idesc_dQ, accum_dQ);
                umma_f16_cg1(tmem_dQ_1, da, db1, idesc_dQ, accum_dQ);
            }
        }
        tmdb_wait_fn(mbar_dQ);
        mbarrier_wait_fn(mbar_dQ, phase_dQ);
        phase_dQ ^= 1;
        
        __syncthreads();
    }
    
    // Store final dQ to shared memory
    if (tid < 64) {
        float dq0 = tmem_read_f32(tmem_dQ_0 + tid, 0);
        float dq1 = tmem_read_f32(tmem_dQ_1 + tid, 0);
        
        int row = tid;
        if (q_base + row < S_len) {
            int col0 = 0;
            int col1 = 64;
            int swizzled_x0 = ((row % 8) ^ (col0 / 8)) * 8 + (col0 % 8);
            int swizzled_x1 = ((row % 8) ^ (col1 / 8)) * 8 + (col1 % 8);
            smem_O[row * 128 + swizzled_x0] = __float2bfloat16(dq0);
            smem_O[row * 128 + swizzled_x1] = __float2bfloat16(dq1);
        }
    }
    __syncthreads();
    
    for (int idx = threadIdx.x; idx < 128*128/8; idx += 128) {
        int row = (idx * 8) / 128;
        if (q_base + row < S_len) {
            int col = (idx * 8) % 128;
            *reinterpret_cast<float4*>(&dQ[head_offset + (q_base + row) * 128 + col]) = *reinterpret_cast<float4*>(&smem_O[row * 128 + col]);
        }
    }
}

__global__ void fp32_to_bf16_kernel(const float* in, __nv_bfloat16* out, size_t n) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(in[idx]);
    }
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

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3);
    
    float scale = 1.0f / sqrtf((float)d);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    size_t num_elements = B * H * S * d;
    float* dV_fp32 = nullptr;
    float* dK_fp32 = nullptr;
    
    CUDA_CHECK(cudaMallocAsync(&dV_fp32, num_elements * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&dK_fp32, num_elements * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dV_fp32, 0, num_elements * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dK_fp32, 0, num_elements * sizeof(float), stream));
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O, tma_dO;
    constexpr uint32_t ATOM_KMODE_DIM = 64;
    constexpr uint32_t ATOM_MMODE_DIM = 64;
    
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), d, B*H*S, ATOM_KMODE_DIM, ATOM_MMODE_DIM, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), d, B*H*S, ATOM_KMODE_DIM, ATOM_MMODE_DIM, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), d, B*H*S, ATOM_KMODE_DIM, ATOM_MMODE_DIM, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_O, O.data_ptr(), d, B*H*S, ATOM_KMODE_DIM, ATOM_MMODE_DIM, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_dO, dO.data_ptr(), d, B*H*S, ATOM_KMODE_DIM, ATOM_MMODE_DIM, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    CUDA_CHECK(cudaFuncSetAttribute(bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 140000));
    
    dim3 grid((S + 63) / 64, B * H);
    dim3 block(128);
    
    bwd_kernel<<<grid, block, 140000, stream>>>(
        tma_Q, tma_K, tma_V, tma_O, tma_dO,
        static_cast<const float*>(L.data_ptr()),
        dV_fp32, dK_fp32,
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        S, scale);
    
    CUDA_CHECK(cudaGetLastError());
    
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    fp32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(dV_fp32, static_cast<__nv_bfloat16*>(dV.data_ptr()), num_elements);
    fp32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(dK_fp32, static_cast<__nv_bfloat16*>(dK.data_ptr()), num_elements);
    
    CUDA_CHECK(cudaFreeAsync(dV_fp32, stream));
    CUDA_CHECK(cudaFreeAsync(dK_fp32, stream));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_attention_bwd::run);

}  // namespace tvm_ffi_attention_bwd