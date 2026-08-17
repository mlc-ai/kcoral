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
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)


// ===================================================================
// PTX Helper Functions
// ===================================================================

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

__device__ __forceinline__ void tma_store_fence_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
}

template<typename T, bool major>
__device__ uint64_t make_smem_desc_wgmma_fn(const T smem_ptr, int stage) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    
    constexpr uint32_t ATOM_KMODE_DIM = 64; 
    
    constexpr uint32_t LBO_major = (ATOM_KMODE_DIM / 8U) * (8U * 128U); // For K-major (A)
    constexpr uint32_t LBO_minor = (ATOM_KMODE_DIM / 8U) * (128U * 8U); // For MN-major (B)
    
    bool is_A = (stage == 0);
    constexpr uint32_t SBO = is_A ? (8U * 128U) : (128U * 8U);
    constexpr uint32_t LBO = is_A ? LBO_major : LBO_minor;
    
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((LBO & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((SBO & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ void gemm_64x64x64(__nv_bfloat16* smem_C, const __nv_bfloat16* smem_A, const __nv_bfloat16* smem_B) {
    for (int k = 0; k < 4; ++k) {
        uint64_t desc_a = make_smem_desc_wgmma_fn<const __nv_bfloat16*, true>(smem_A + k * 1024, 0);
        uint64_t desc_b = make_smem_desc_wgmma_fn<const __nv_bfloat16*, true>(smem_B + k * 1024, 1);
        
        for (int n = 0; n < 8; ++n) {
            for (int m = 0; m < 4; ++m) {
                uint64_t desc_c = make_smem_desc_wgmma_fn<__nv_bfloat16*, true>(smem_C + (m * 256) + (n * 16), 0); 
                uint32_t accum_flag = (k == 0) ? 0 : 1;
                asm volatile("mma.sync.aligned.m16n8k16.shared.b16 %0, %1, %2, %3;" 
                    : : "r"(desc_c), "r"(desc_a), "r"(desc_b), "r"(accum_flag));
            }
        }
    }
}

__device__ void compute_QK_T_scaled_64x64_fn(__nv_bfloat16* p_half_0, __nv_bfloat16* p_half_1, const __nv_bfloat16* q_tile, const __nv_bfloat16* k_tile, float scale) {
    int tid = threadIdx.x;
    
    for (int i = tid; i < 4096; i += blockDim.x) {
        p_half_0[i] = __float2bfloat16(0);
        p_half_1[i] = __float2bfloat16(0);
    }
    
    gemm_64x64x64(p_half_0, q_tile, k_tile);
    gemm_64x64x64(p_half_1, q_tile + 4096, k_tile + 4096);
    
    for (int idx = tid; idx < 4096; idx += blockDim.x) {
        float sum = __bfloat162float(p_half_0[idx]) + __bfloat162float(p_half_1[idx]);
        p_half_0[idx] = __float2bfloat16(sum * scale);
    }
}

__device__ void compute_D_V_T_64x64_fn(__nv_bfloat16* dp_half_0, __nv_bfloat16* dp_half_1, const __nv_bfloat16* d_flat_0, const __nv_bfloat16* d_flat_1, const __nv_bfloat16* v_tile) {
    int tid = threadIdx.x;
    
    for (int i = tid; i < 4096; i += blockDim.x) {
        dp_half_0[i] = __float2bfloat16(0);
        dp_half_1[i] = __float2bfloat16(0);
    }
    
    gemm_64x64x64(dp_half_0, d_flat_0, v_tile);
    gemm_64x64x64(dp_half_1, d_flat_1, v_tile + 4096);
    
    for (int idx = tid; idx < 4096; idx += blockDim.x) {
        float sum = __bfloat162float(dp_half_0[idx]) + __bfloat162float(dp_half_1[idx]);
        dp_half_0[idx] = __float2bfloat16(sum);
    }
}

__device__ void compute_dQ_contribution_64x64_fn(__nv_bfloat16* dst_accum_0, __nv_bfloat16* dst_accum_1, const __nv_bfloat16* src_A, const __nv_bfloat16* src_B) {
    for (int k_iter = 0; k_iter < 4; ++k_iter) {
        uint64_t desc_a = make_smem_desc_wgmma_fn<const __nv_bfloat16*, true>(src_A + k_iter * 1024, 0);
        uint64_t desc_b = make_smem_desc_wgmma_fn<const __nv_bfloat16*, true>(src_B + k_iter * 1024, 1);
        
        for (int n = 0; n < 8; ++n) {
            for (int m = 0; m < 4; ++m) {
                uint64_t desc_c_0 = make_smem_desc_wgmma_fn<__nv_bfloat16*, true>(dst_accum_0 + (m * 256) + (n * 16), 0);
                uint64_t desc_c_1 = make_smem_desc_wgmma_fn<__nv_bfloat16*, true>(dst_accum_1 + (m * 256) + (n * 16), 0);
                
                asm volatile("mma.sync.aligned.m16n8k16.shared.b16 %0, %1, %2, 1;" : : "r"(desc_c_0), "r"(desc_a), "r"(desc_b));
                asm volatile("mma.sync.aligned.m16n8k16.shared.b16 %0, %1, %2, 1;" : : "r"(desc_c_1), "r"(desc_a), "r"(desc_b));
            }
        }
    }
}

__device__ void compute_dK_contribution_64x64_fn(__nv_bfloat16* dst_accum_0, __nv_bfloat16* dst_accum_1, const __nv_bfloat16* src_A_trans, const __nv_bfloat16* src_B) {
    for (int k_iter = 0; k_iter < 4; ++k_iter) {
        uint64_t desc_a = make_smem_desc_wgmma_fn<const __nv_bfloat16*, true>(src_A_trans + k_iter * 1024, 0);
        uint64_t desc_b = make_smem_desc_wgmma_fn<const __nv_bfloat16*, true>(src_B + k_iter * 1024, 1);
        
        for (int n = 0; n < 8; ++n) {
            for (int m = 0; m < 4; ++m) {
                uint64_t desc_c_0 = make_smem_desc_wgmma_fn<__nv_bfloat16*, true>(dst_accum_0 + (m * 256) + (n * 16), 0);
                uint64_t desc_c_1 = make_smem_desc_wgmma_fn<__nv_bfloat16*, true>(dst_accum_1 + (m * 256) + (n * 16), 0);
                
                asm volatile("mma.sync.aligned.m16n8k16.shared.b16 %0, %1, %2, 1;" : : "r"(desc_c_0), "r"(desc_a), "r"(desc_b));
                asm volatile("mma.sync.aligned.m16n8k16.shared.b16 %0, %1, %2, 1;" : : "r"(desc_c_1), "r"(desc_a), "r"(desc_b));
            }
        }
    }
}

__device__ void compute_dV_contribution_64x64_fn(__nv_bfloat16* dst_accum_0, __nv_bfloat16* dst_accum_1, const __nv_bfloat16* src_A_trans, const __nv_bfloat16* src_B) {
    for (int k_iter = 0; k_iter < 4; ++k_iter) {
        uint64_t desc_a = make_smem_desc_wgmma_fn<const __nv_bfloat16*, true>(src_A_trans + k_iter * 1024, 0);
        uint64_t desc_b = make_smem_desc_wgmma_fn<const __nv_bfloat16*, true>(src_B + k_iter * 1024, 1);
        
        for (int n = 0; n < 8; ++n) {
            for (int m = 0; m < 4; ++m) {
                uint64_t desc_c_0 = make_smem_desc_wgmma_fn<__nv_bfloat16*, true>(dst_accum_0 + (m * 256) + (n * 16), 0);
                uint64_t desc_c_1 = make_smem_desc_wgmma_fn<__nv_bfloat16*, true>(dst_accum_1 + (m * 256) + (n * 16), 0);
                
                asm volatile("mma.sync.aligned.m16n8k16.shared.b16 %0, %1, %2, 1;" : : "r"(desc_c_0), "r"(desc_a), "r"(desc_b));
                asm volatile("mma.sync.aligned.m16n8k16.shared.b16 %0, %1, %2, 1;" : : "r"(desc_c_1), "r"(desc_a), "r"(desc_b));
            }
        }
    }
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ __forceinline__ void load_64x128(
    const CUtensorMap* tma, uint64_t* bar, __nv_bfloat16* smem,
    int32_t global_offset, uint32_t& phase) {
    int tid = threadIdx.x;
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar, 16384);
        tma_load_2d_fn(tma, bar, smem, 0, global_offset);
        tma_load_2d_fn(tma, bar, smem + 4096, 64, global_offset);
    }
    mbarrier_wait_fn(bar, phase);
    __syncthreads();
    phase ^= 1;
}

__device__ __forceinline__ void store_64x128(
    const CUtensorMap* tma, __nv_bfloat16* smem, int32_t global_offset) {
    int tid = threadIdx.x;
    if (tid == 0) {
        tma_store_2d_fn(tma, smem, 0, global_offset);
        tma_store_2d_fn(tma, smem + 4096, 64, global_offset);
        tma_store_commit_fn();
    }
    tma_store_wait_fn<0>();
    __syncthreads();
}

// Epilogue reading from TMEM, converting FP32 -> BF16, staging in SMEM
__device__ __forceinline__ void epilogue_64x128(__nv_bfloat16* smem_out) {
    int tid = threadIdx.x;
    for (uint32_t col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t base = tid * 128 + col;
        smem_out[base + 0] = __float2bfloat16(__uint_as_float(r0));
        smem_out[base + 1] = __float2bfloat16(__uint_as_float(r1));
        smem_out[base + 2] = __float2bfloat16(__uint_as_float(r2));
        smem_out[base + 3] = __float2bfloat16(__uint_as_float(r3));
    }
    __syncthreads();
}


// ===================================================================
// Kernels
// ===================================================================

__global__ void __launch_bounds__(128) kernel_1_dq(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_dQ,
    const float* L, int S, float scale) 
{
    extern __shared__ __align__(1024) uint8_t smem_pool[];
    __nv_bfloat16* q_tile = (__nv_bfloat16*)smem_pool;                    
    __nv_bfloat16* k_tile = (__nv_bfloat16*)(smem_pool + 16384);           
    __nv_bfloat16* v_tile = (__nv_bfloat16*)(smem_pool + 32768);           
    __nv_bfloat16* o_tile = (__nv_bfloat16*)(smem_pool + 49152);           
    __nv_bfloat16* do_tile = (__nv_bfloat16*)(smem_pool + 65536);          
    __nv_bfloat16* p_half_0 = (__nv_bfloat16*)(smem_pool + 81920);         
    __nv_bfloat16* p_half_1 = (__nv_bfloat16*)(smem_pool + 90112);         
    __nv_bfloat16* ds_half_0 = (__nv_bfloat16*)(smem_pool + 98304);        
    __nv_bfloat16* ds_half_1 = (__nv_bfloat16*)(smem_pool + 102400);       
    __nv_bfloat16* d_f_flat = (__nv_bfloat16*)(smem_pool + 106496);        
    float* d_sum = (float*)(smem_pool + 110592);                           
    float* l_exp = (float*)(smem_pool + 110848);                           

    int s_off = blockIdx.x * 64;
    int s_bh = blockIdx.y;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    uint64_t bar_q, bar_k, bar_v, bar_o, bar_do;
    if (tid == 0) {
        init_smem_barrier_fn(&bar_q, 1);
        init_smem_barrier_fn(&bar_k, 1);
        init_smem_barrier_fn(&bar_v, 1);
        init_smem_barrier_fn(&bar_o, 1);
        init_smem_barrier_fn(&bar_do, 1);
    }
    __syncthreads();
    
    int phase_q = 0, phase_k = 0, phase_v = 0, phase_o = 0, phase_do = 0;
    
    int s_off_global = s_bh * S + s_off;
    load_64x128(&tma_Q, &bar_q, q_tile, s_off_global, phase_q);
    load_64x128(&tma_O, &bar_o, o_tile, s_off_global, phase_o);
    load_64x128(&tma_dO, &bar_do, do_tile, s_off_global, phase_do);
    
    float2 dq_half_0 = {0, 0};
    float2 dq_half_1 = {0, 0};
    
    float sum_0 = 0, sum_1 = 0;
    float local_o_0[64], local_do_0[64];
    float local_o_1[64], local_do_1[64];
    
    for (int col = 0; col < 64; ++col) {
        local_o_0[col] = __bfloat162float(o_tile[tid * 64 + col]);
        local_do_0[col] = __bfloat162float(do_tile[tid * 64 + col]);
        sum_0 += local_o_0[col] * local_do_0[col];
        
        local_o_1[col] = __bfloat162float(o_tile[4096 + tid * 64 + col]);
        local_do_1[col] = __bfloat162float(do_tile[4096 + tid * 64 + col]);
        sum_1 += local_o_1[col] * local_do_1[col];
    }
    
    sum_0 = warp_reduce_sum(sum_0);
    sum_1 = warp_reduce_sum(sum_1);
    float d_sum_val = sum_0 + sum_1;
    
    if (tid == 0) {
        for (int i = 0; i < 64; ++i) {
            d_sum[i] = d_sum_val;
            l_exp[i] = expf(L[s_bh * S + s_off + i]);
        }
    }
    __syncthreads();
    
    for (int j_off = 0; j_off <= s_off; j_off += 64) {
        int j_off_global = s_bh * S + j_off;
        load_64x128(&tma_K, &bar_k, k_tile, j_off_global, phase_k);
        load_64x128(&tma_V, &bar_v, v_tile, j_off_global, phase_v);
        
        compute_QK_T_scaled_64x64_fn(p_half_0, p_half_1, q_tile, k_tile, scale);
        
        for (int i = tid; i < 4096; i += blockDim.x) {
            int row = i / 64;
            int col = i % 64;
            int global_row_idx = s_off + row;
            int global_col_idx = j_off + col;
            
            float f_val = __bfloat162float(p_half_0[i]);
            float p_val = (global_col_idx <= global_row_idx) ? (expf(f_val) / l_exp[row]) : 0.0f;
            
            if (global_row_idx >= S || global_col_idx >= S) p_val = 0.0f;
            
            p_half_0[i] = __float2bfloat16(p_val);
        }
        __syncthreads(); 
        
        for (int i = tid; i < 4096; i += blockDim.x) {
            ds_half_0[i] = p_half_0[i]; 
            ds_half_1[i] = p_half_1[i];
        }
        __syncthreads();
        
        compute_D_V_T_64x64_fn(ds_half_0, ds_half_1, 
            (q_tile), (q_tile + 4096), v_tile);
        
        for (int i = tid; i < 4096; i += blockDim.x) {
            int row = i / 64;
            int col = i % 64;
            int global_row_idx = s_off + row;
            int global_col_idx = j_off + col;
            
            float p_val = __bfloat162float(p_half_0[i]);
            float dp_val = __bfloat162float(ds_half_0[i]);
            
            float val = p_val * (dp_val - d_sum_val) * scale;
            
            if (global_row_idx >= S || global_col_idx >= S) val = 0.0f;
            
            ds_half_0[i] = __float2bfloat16(val);
        }
        __syncthreads();
        
        for (int i = tid; i < 4096; i += blockDim.x) {
            d_f_flat[i] = ds_half_0[i];
        }
        __syncthreads();
        
        compute_dQ_contribution_64x64_fn(q_tile, q_tile + 4096, d_f_flat, k_tile);
        __syncthreads();
    }
    
    epilogue_64x128(q_tile);
    
    tma_store_fence_fn();
    store_64x128(&tma_dQ, q_tile, s_off_global);
}

__global__ void __launch_bounds__(128) kernel_2_dkv(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_dK,
    const __grid_constant__ CUtensorMap tma_dV,
    const float* L, int S, float scale) 
{
    extern __shared__ __align__(1024) uint8_t smem_pool[];
    __nv_bfloat16* k_tile = (__nv_bfloat16*)smem_pool;                    
    __nv_bfloat16* v_tile = (__nv_bfloat16*)(smem_pool + 16384);           
    __nv_bfloat16* q_tile = (__nv_bfloat16*)(smem_pool + 32768);           
    __nv_bfloat16* o_tile = (__nv_bfloat16*)(smem_pool + 49152);           
    __nv_bfloat16* do_tile = (__nv_bfloat16*)(smem_pool + 65536);          
    __nv_bfloat16* p_half_0 = (__nv_bfloat16*)(smem_pool + 81920);         
    __nv_bfloat16* p_half_1 = (__nv_bfloat16*)(smem_pool + 90112);         
    __nv_bfloat16* ds_half_0 = (__nv_bfloat16*)(smem_pool + 98304);        
    __nv_bfloat16* ds_half_1 = (__nv_bfloat16*)(smem_pool + 102400);       
    __nv_bfloat16* dk_half_0 = (__nv_bfloat16*)(smem_pool + 106496);       
    __nv_bfloat16* dk_half_1 = (__nv_bfloat16*)(smem_pool + 110592);       
    __nv_bfloat16* dv_half_0 = (__nv_bfloat16*)(smem_pool + 114688);       
    __nv_bfloat16* dv_half_1 = (__nv_bfloat16*)(smem_pool + 118784);       
    float* d_sum = (float*)(smem_pool + 122880);                           
    float* l_exp = (float*)(smem_pool + 123136);                           

    int s_off = blockIdx.x * 64;
    int s_bh = blockIdx.y;
    int tid = threadIdx.x;
    
    uint64_t bar_q, bar_k, bar_v, bar_o, bar_do;
    if (tid == 0) {
        init_smem_barrier_fn(&bar_q, 1);
        init_smem_barrier_fn(&bar_k, 1);
        init_smem_barrier_fn(&bar_v, 1);
        init_smem_barrier_fn(&bar_o, 1);
        init_smem_barrier_fn(&bar_do, 1);
    }
    __syncthreads();
    
    int phase_q = 0, phase_k = 0, phase_v = 0, phase_o = 0, phase_do = 0;
    
    int s_off_global = s_bh * S + s_off;
    load_64x128(&tma_K, &bar_k, k_tile, s_off_global, phase_k);
    load_64x128(&tma_V, &bar_v, v_tile, s_off_global, phase_v);
    
    for (int i = tid; i < 4096; ++i) {
        dk_half_0[i] = __float2bfloat16(0);
        dk_half_1[i] = __float2bfloat16(0);
        dv_half_0[i] = __float2bfloat16(0);
        dv_half_1[i] = __float2bfloat16(0);
    }
    __syncthreads();
    
    for (int i_off = s_off; i_off < S; i_off += 64) {
        int i_off_global = s_bh * S + i_off;
        load_64x128(&tma_Q, &bar_q, q_tile, i_off_global, phase_q);
        load_64x128(&tma_O, &bar_o, o_tile, i_off_global, phase_o);
        load_64x128(&tma_dO, &bar_do, do_tile, i_off_global, phase_do);
        
        float sum_0 = 0, sum_1 = 0;
        for (int col = 0; col < 64; ++col) {
            sum_0 += __bfloat162float(o_tile[tid * 64 + col]) * __bfloat162float(do_tile[tid * 64 + col]);
            sum_1 += __bfloat162float(o_tile[4096 + tid * 64 + col]) * __bfloat162float(do_tile[4096 + tid * 64 + col]);
        }
        sum_0 = warp_reduce_sum(sum_0);
        sum_1 = warp_reduce_sum(sum_1);
        float d_sum_val = sum_0 + sum_1;
        
        if (tid == 0) {
            for (int i = 0; i < 64; ++i) {
                d_sum[i] = d_sum_val;
                l_exp[i] = expf(L[s_bh * S + i_off + i]);
            }
        }
        __syncthreads();
        
        compute_QK_T_scaled_64x64_fn(p_half_0, p_half_1, q_tile, k_tile, scale);
        
        for (int i = tid; i < 4096; i += blockDim.x) {
            int row = i / 64;
            int col = i % 64;
            int global_row_idx = i_off + row;
            int global_col_idx = s_off + col;
            
            float f_val = __bfloat162float(p_half_0[i]);
            float p_val = (global_col_idx <= global_row_idx) ? (expf(f_val) / l_exp[row]) : 0.0f;
            
            if (global_row_idx >= S || global_col_idx >= S) p_val = 0.0f;
            
            p_half_0[i] = __float2bfloat16(p_val);
        }
        __syncthreads();
        
        compute_D_V_T_64x64_fn(ds_half_0, ds_half_1, 
            (q_tile), (q_tile + 4096), v_tile);
        
        for (int i = tid; i < 4096; i += blockDim.x) {
            int row = i / 64;
            int col = i % 64;
            int global_row_idx = i_off + row;
            int global_col_idx = s_off + col;
            
            float p_val = __bfloat162float(p_half_0[i]);
            float dp_val = __bfloat162float(ds_half_0[i]);
            
            float val = p_val * (dp_val - d_sum_val) * scale;
            
            if (global_row_idx >= S || global_col_idx >= S) val = 0.0f;
            
            ds_half_0[i] = __float2bfloat16(val);
        }
        __syncthreads();
        
        compute_dK_contribution_64x64_fn(dk_half_0, dk_half_1, ds_half_0, q_tile);
        compute_dV_contribution_64x64_fn(dv_half_0, dv_half_1, p_half_0, do_tile);
        __syncthreads();
    }
    
    epilogue_64x128(dk_half_0);
    epilogue_64x128(dv_half_0);
    
    tma_store_fence_fn();
    store_64x128(&tma_dK, dk_half_0, s_off_global);
    store_64x128(&tma_dV, dv_half_0, s_off_global);
}

// ===================================================================
// TVM-FFI Binding
// ===================================================================

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, globalAddress, globalDim,
        globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle, l2Promotion, oobFill
    );
}

namespace tvm_ffi_mha_bwd {

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
    auto make_tma = [&](CUtensorMap* tma, void* ptr) {
        CU_CHECK(create_tma_2d_descriptor_2B(
            tma, ptr, 128, B * H * S, 64, 64,
            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    };
    
    make_tma(&tma_Q, Q.data_ptr());
    make_tma(&tma_K, K.data_ptr());
    make_tma(&tma_V, V.data_ptr());
    make_tma(&tma_O, O.data_ptr());
    make_tma(&tma_dO, dO.data_ptr());
    make_tma(&tma_dQ, dQ.data_ptr());
    make_tma(&tma_dK, dK.data_ptr());
    make_tma(&tma_dV, dV.data_ptr());
    
    dim3 grid((S + 63) / 64, B * H);
    dim3 block(128);
    
    CUDA_CHECK(cudaFuncSetAttribute(kernel_1_dq, cudaFuncAttributeMaxDynamicSharedMemorySize, 102400));
    kernel_1_dq<<<grid, block, 102400, stream>>>(
        tma_Q, tma_K, tma_V, tma_O, tma_dO, tma_dQ, 
        static_cast<const float*>(L.data_ptr()), S, 1.0f / sqrtf(128.0f));
        
    CUDA_CHECK(cudaFuncSetAttribute(kernel_2_dkv, cudaFuncAttributeMaxDynamicSharedMemorySize, 102400));
    kernel_2_dkv<<<grid, block, 102400, stream>>>(
        tma_Q, tma_K, tma_V, tma_O, tma_dO, tma_dK, tma_dV, 
        static_cast<const float*>(L.data_ptr()), S, 1.0f / sqrtf(128.0f));
        
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_mha_bwd