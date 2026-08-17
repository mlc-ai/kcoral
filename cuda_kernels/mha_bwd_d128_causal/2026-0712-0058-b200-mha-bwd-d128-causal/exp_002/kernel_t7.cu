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

__device__ __forceinline__ uint64_t encode_mma_sync_A(void* ptr, int ldm_bytes) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    return (addr & 0x1FF) | (((uint32_t)(ldm_bytes / 16) & 0x3F) << 9) | (1ULL << 63);
}

__device__ __forceinline__ uint64_t encode_mma_sync_B(void* ptr, int ldm_bytes) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    return (addr & 0x1FF) | (((uint32_t)(ldm_bytes / 16) & 0x3F) << 9) | (1ULL << 63);
}

__device__ __forceinline__ uint64_t encode_mma_sync_C(void* ptr, int ldm_bytes) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    return (addr & 0x1FF) | (((uint32_t)(ldm_bytes / 16) & 0x3F) << 9) | (1ULL << 63);
}

__device__ __forceinline__ void gemm_64x64x64(__nv_bfloat16* C, const __nv_bfloat16* A, const __nv_bfloat16* B) {
    for (int k = 0; k < 4; ++k) {
        uint64_t desc_a = encode_mma_sync_A((const void*)(A + k * 64), 128);
        for (int n = 0; n < 8; ++n) {
            for (int m = 0; m < 4; ++m) {
                uint64_t desc_c = encode_mma_sync_C((void*)(C + m * 64 * 2 + n * 16), 128);
                uint64_t desc_b = encode_mma_sync_B((const void*)(B + k * 64), 128);
                uint32_t accum = (k == 0) ? 0 : 1;
                asm volatile("mma.sync.aligned.m16n8k16.shared.b16 %0, %1, %2, %3;" 
                    : : "r"(desc_c), "r"(desc_a), "r"(desc_b), "r"(accum));
            }
        }
    }
}

__device__ __forceinline__ void gemm_64x64x64_cta_g2(__nv_bfloat16* C, const __nv_bfloat16* A, const __nv_bfloat16* B) {
    for (int k = 0; k < 4; ++k) {
        for (int n = 0; n < 8; ++n) {
            for (int m = 0; m < 2; ++m) {
                uint64_t desc_a = encode_mma_sync_A((const void*)(A + k * 128 + n * 16 + m * 64 * 128), 128);
                uint64_t desc_b = encode_mma_sync_B((const void*)(B + k * 128 + n * 16 + m * 64 * 128), 128);
                uint64_t desc_c = encode_mma_sync_C((void*)(C + n * 16 + m * 64 * 128), 128);
                uint32_t accum = (k == 0) ? 0 : 1;
                asm volatile("mma.sync.aligned.m16n8k16.shared.b16 %0, %1, %2, %3;" 
                    : : "r"(desc_c), "r"(desc_a), "r"(desc_b), "r"(accum));
            }
        }
    }
}

__device__ __forceinline__ void transpose_64x64(__nv_bfloat16* dst, const __nv_bfloat16* src) {
    int tid = threadIdx.x;
    __syncthreads();
    for (int i = tid; i < 4096; i += blockDim.x) {
        int row = i / 64;
        int col = i % 64;
        dst[col * 64 + row] = src[row * 64 + col];
    }
    __syncthreads();
}

__device__ __forceinline__ void load_64x128(
    const CUtensorMap* tma, uint64_t* bar, __nv_bfloat16* smem_0, __nv_bfloat16* smem_1,
    int32_t global_offset, uint32_t& phase) {
    int tid = threadIdx.x;
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar, 16384);
        tma_load_2d_fn(tma, bar, smem_0, 0, global_offset);
        tma_load_2d_fn(tma, bar, smem_1, 64, global_offset);
    }
    mbarrier_wait_fn(bar, phase);
    __syncthreads();
    phase ^= 1;
}

__device__ __forceinline__ void store_64x128(
    const CUtensorMap* tma, __nv_bfloat16* smem_0, __nv_bfloat16* smem_1, int32_t global_offset) {
    int tid = threadIdx.x;
    if (tid == 0) {
        tma_store_2d_fn(tma, smem_0, 0, global_offset);
        tma_store_2d_fn(tma, smem_1, 64, global_offset);
        tma_store_commit_fn();
    }
    tma_store_wait_fn<0>();
    __syncthreads();
}

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

// ===================================================================
// Kernels
// ===================================================================

__global__ void __launch_bounds__(128, 2) kernel_fused(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_dQ,
    const __grid_constant__ CUtensorMap tma_dK,
    const __grid_constant__ CUtensorMap tma_dV,
    const float* L, int S, float scale) 
{
    extern __shared__ __align__(1024) uint8_t smem_pool[];
    uint32_t cta_offset = cluster_rank_fn() * 73728;

    __nv_bfloat16* q_tile_0 = (__nv_bfloat16*)(smem_pool + cta_offset + 0);
    __nv_bfloat16* q_tile_1 = (__nv_bfloat16*)(smem_pool + cta_offset + 8192);
    __nv_bfloat16* do_tile_0 = (__nv_bfloat16*)(smem_pool + cta_offset + 16384);
    __nv_bfloat16* do_tile_1 = (__nv_bfloat16*)(smem_pool + cta_offset + 24576);
    __nv_bfloat16* o_tile_0 = (__nv_bfloat16*)(smem_pool + cta_offset + 32768);
    __nv_bfloat16* o_tile_1 = (__nv_bfloat16*)(smem_pool + cta_offset + 40960);
    __nv_bfloat16* k_tile_0 = (__nv_bfloat16*)(smem_pool + cta_offset + 49152);
    __nv_bfloat16* k_tile_1 = (__nv_bfloat16*)(smem_pool + cta_offset + 57344);
    __nv_bfloat16* v_tile_0 = (__nv_bfloat16*)(smem_pool + cta_offset + 65536);
    __nv_bfloat16* v_tile_1 = (__nv_bfloat16*)(smem_pool + cta_offset + 73728);
    
    __nv_bfloat16* p_flat = (__nv_bfloat16*)(smem_pool + 147456);
    __nv_bfloat16* dp_flat = (__nv_bfloat16*)(smem_pool + 155648);
    __nv_bfloat16* d_flat = (__nv_bfloat16*)(smem_pool + 163840);
    __nv_bfloat16* dq_flat_0 = (__nv_bfloat16*)(smem_pool + 171040);
    __nv_bfloat16* dq_flat_1 = (__nv_bfloat16*)(smem_pool + 179232);
    
    uint64_t* bar_q = (uint64_t*)(smem_pool + 187424);
    uint64_t* bar_k = (uint64_t*)(smem_pool + 187432);
    uint64_t* bar_v = (uint64_t*)(smem_pool + 187440);
    uint64_t* bar_o = (uint64_t*)(smem_pool + 187448);
    uint64_t* bar_do = (uint64_t*)(smem_pool + 187456);
    
    float* d_sum = (float*)(smem_pool + 187488);
    float* l_exp = d_sum + 64;

    int s_off = (blockIdx.x * 128) + (cluster_rank_fn() * 64);
    int s_bh = blockIdx.y;
    int tid = threadIdx.x;
    
    uint32_t phase_q = 0, phase_k = 0, phase_v = 0, phase_o = 0, phase_do = 0;
    
    if (tid == 0) {
        init_smem_barrier_fn(bar_q, 1);
        init_smem_barrier_fn(bar_k, 1);
        init_smem_barrier_fn(bar_v, 1);
        init_smem_barrier_fn(bar_o, 1);
        init_smem_barrier_fn(bar_do, 1);
    }
    __syncthreads();
    
    int s_off_global = s_bh * S + s_off;
    load_64x128(&tma_Q, bar_q, q_tile_0, q_tile_1, s_off_global, phase_q);
    load_64x128(&tma_O, bar_o, o_tile_0, o_tile_1, s_off_global, phase_o);
    load_64x128(&tma_dO, bar_do, do_tile_0, do_tile_1, s_off_global, phase_do);
    
    if (tid < 64) {
        float sum_0 = 0, sum_1 = 0;
        for (int col = 0; col < 64; ++col) {
            float o_0 = __bfloat162float(o_tile_0[tid * 64 + col]);
            float do_0 = __bfloat162float(do_tile_0[tid * 64 + col]);
            sum_0 += o_0 * do_0;
            
            float o_1 = __bfloat162float(o_tile_1[tid * 64 + col]);
            float do_1 = __bfloat162float(do_tile_1[tid * 64 + col]);
            sum_1 += o_1 * do_1;
        }
        d_sum[tid] = sum_0 + sum_1;
        l_exp[tid] = (s_off + tid < S) ? expf(L[s_bh * S + s_off + tid]) : 1.0f;
    }
    __syncthreads();

    if (tid == 0) {
        memset(smem_pool + 171040, 0, 8192);
        memset(smem_pool + 179232, 0, 8192);
    }
    __syncthreads();
    
    for (int j_off = 0; j_off <= s_off; j_off += 64) {
        int j_off_global = s_bh * S + j_off;
        load_64x128(&tma_K, bar_k, k_tile_0, k_tile_1, j_off_global, phase_k);
        load_64x128(&tma_V, bar_v, v_tile_0, v_tile_1, j_off_global, phase_v);
        
        __nv_bfloat16* a_q_0 = p_flat;
        __nv_bfloat16* a_q_1 = dp_flat;
        transpose_64x64(a_q_0, q_tile_0);
        transpose_64x64(a_q_1, q_tile_1);
        
        __nv_bfloat16* a_k_0 = o_tile_0;
        __nv_bfloat16* a_k_1 = o_tile_1;
        transpose_64x64(a_k_0, k_tile_0);
        transpose_64x64(a_k_1, k_tile_1);
        
        memset(smem_pool + 81920, 0, 8192);
        memset(smem_pool + 90112, 0, 8192);
        
        gemm_64x64x64_cta_g2(p_flat, a_q_0, a_k_0);
        gemm_64x64x64_cta_g2(p_flat + 4096, a_q_1, a_k_1);
        
        __nv_bfloat16* a_do_0 = k_tile_0;
        __nv_bfloat16* a_do_1 = k_tile_1;
        transpose_64x64(a_do_0, do_tile_0);
        transpose_64x64(a_do_1, do_tile_1);
        
        __nv_bfloat16* a_v_0 = do_tile_0;
        __nv_bfloat16* a_v_1 = do_tile_1;
        transpose_64x64(a_v_0, v_tile_0);
        transpose_64x64(a_v_1, v_tile_1);
        
        memset(smem_pool + 90112, 0, 8192);
        memset(smem_pool + 98304, 0, 8192);
        
        gemm_64x64x64_cta_g2(dp_flat, a_do_0, a_v_0);
        gemm_64x64x64_cta_g2(dp_flat + 4096, a_do_1, a_v_1);
        
        for (int i = tid; i < 4096; i += blockDim.x) {
            int row = i / 64;
            int col = i % 64;
            
            float p_val_0 = __bfloat162float(p_flat[i]);
            float dp_val_0 = __bfloat162float(dp_flat[i]);
            float val_0 = p_val_0 * (dp_val_0 - d_sum[row]) * scale;
            if (s_off + row >= S || j_off + col >= S) val_0 = 0.0f;
            d_flat[i] = __float2bfloat16(val_0);
            
            float p_val_1 = __bfloat162float(p_flat[i + 4096]);
            float dp_val_1 = __bfloat162float(dp_flat[i + 4096]);
            float val_1 = p_val_1 * (dp_val_1 - d_sum[row]) * scale;
            if (s_off + row >= S || j_off + col >= S) val_1 = 0.0f;
            d_flat[i + 4096] = __float2bfloat16(val_1);
        }
        __syncthreads();
        
        __nv_bfloat16* a_dqkt_0 = p_flat;
        __nv_bfloat16* a_dqkt_1 = p_flat + 4096;
        transpose_64x64(a_dqkt_0, d_flat);
        transpose_64x64(a_dqkt_1, d_flat + 4096);
        
        __nv_bfloat16* a_k_0_dq = o_tile_0;
        __nv_bfloat16* a_k_1_dq = o_tile_1;
        transpose_64x64(a_k_0_dq, k_tile_0);
        transpose_64x64(a_k_1_dq, k_tile_1);
        
        gemm_64x64x64_cta_g2(dq_flat_0, a_dqkt_0, a_k_0_dq);
        gemm_64x64x64_cta_g2(dq_flat_1, a_dqkt_1, a_k_1_dq);
        
        __syncthreads();
    }
    
    tma_store_fence_fn();
    store_64x128(&tma_dQ, dq_flat_0, dq_flat_1, s_off_global);
    
    // -------------------------------------------------------------------------
    // Pass 2: dK and dV Computations
    // -------------------------------------------------------------------------
    
    load_64x128(&tma_K, bar_k, k_tile_0, k_tile_1, s_off_global, phase_k);
    load_64x128(&tma_V, bar_v, v_tile_0, v_tile_1, s_off_global, phase_v);
    
    __nv_bfloat16* dk_flat_0 = p_flat;
    __nv_bfloat16* dk_flat_1 = dp_flat;
    __nv_bfloat16* dv_flat_0 = d_flat;
    __nv_bfloat16* dv_flat_1 = o_tile_0;
    
    if (tid == 0) {
        memset(smem_pool + 81920, 0, 8192);
        memset(smem_pool + 90112, 0, 8192);
        memset(smem_pool + 98304, 0, 8192);
        memset(smem_pool + 49152, 0, 8192);
    }
    __syncthreads();
    
    for (int i_off = s_off; i_off < S; i_off += 64) {
        int i_off_global = s_bh * S + i_off;
        load_64x128(&tma_Q, bar_q, q_tile_0, q_tile_1, i_off_global, phase_q);
        load_64x128(&tma_O, bar_o, o_tile_0, o_tile_1, i_off_global, phase_o);
        load_64x128(&tma_dO, bar_do, do_tile_0, do_tile_1, i_off_global, phase_do);
        
        if (tid < 64) {
            float sum_0 = 0, sum_1 = 0;
            for (int col = 0; col < 64; ++col) {
                float o_0 = __bfloat162float(o_tile_0[tid * 64 + col]);
                float do_0 = __bfloat162float(do_tile_0[tid * 64 + col]);
                sum_0 += o_0 * do_0;
                
                float o_1 = __bfloat162float(o_tile_1[tid * 64 + col]);
                float do_1 = __bfloat162float(do_tile_1[tid * 64 + col]);
                sum_1 += o_1 * do_1;
            }
            d_sum[tid] = sum_0 + sum_1;
            l_exp[tid] = (i_off + tid < S) ? expf(L[s_bh * S + i_off + tid]) : 1.0f;
        }
        __syncthreads();
        
        __nv_bfloat16* a_q_0 = p_flat;
        __nv_bfloat16* a_q_1 = dp_flat;
        transpose_64x64(a_q_0, q_tile_0);
        transpose_64x64(a_q_1, q_tile_1);
        
        __nv_bfloat16* a_k_0 = o_tile_0;
        __nv_bfloat16* a_k_1 = o_tile_1;
        transpose_64x64(a_k_0, k_tile_0);
        transpose_64x64(a_k_1, k_tile_1);
        
        memset(smem_pool + 81920, 0, 8192);
        memset(smem_pool + 90112, 0, 8192);
        
        gemm_64x64x64_cta_g2(p_flat, a_q_0, a_k_0);
        gemm_64x64x64_cta_g2(p_flat + 4096, a_q_1, a_k_1);
        
        __nv_bfloat16* a_do_0 = k_tile_0;
        __nv_bfloat16* a_do_1 = k_tile_1;
        transpose_64x64(a_do_0, do_tile_0);
        transpose_64x64(a_do_1, do_tile_1);
        
        __nv_bfloat16* a_v_0 = do_tile_0;
        __nv_bfloat16* a_v_1 = do_tile_1;
        transpose_64x64(a_v_0, v_tile_0);
        transpose_64x64(a_v_1, v_tile_1);
        
        memset(smem_pool + 90112, 0, 8192);
        memset(smem_pool + 98304, 0, 8192);
        
        gemm_64x64x64_cta_g2(dp_flat, a_do_0, a_v_0);
        gemm_64x64x64_cta_g2(dp_flat + 4096, a_do_1, a_v_1);
        
        for (int i = tid; i < 4096; i += blockDim.x) {
            int row = i / 64;
            int col = i % 64;
            
            float p_val_0 = __bfloat162float(p_flat[i]);
            float dp_val_0 = __bfloat162float(dp_flat[i]);
            float val_0 = p_val_0 * (dp_val_0 - d_sum[row]) * scale;
            if (i_off + row >= S || s_off + col >= S) val_0 = 0.0f;
            d_flat[i] = __float2bfloat16(val_0);
            
            float p_val_1 = __bfloat162float(p_flat[i + 4096]);
            float dp_val_1 = __bfloat162float(dp_flat[i + 4096]);
            float val_1 = p_val_1 * (dp_val_1 - d_sum[row]) * scale;
            if (i_off + row >= S || s_off + col >= S) val_1 = 0.0f;
            d_flat[i + 4096] = __float2bfloat16(val_1);
        }
        __syncthreads();
        
        __nv_bfloat16* a_dkt_0 = p_flat;
        __nv_bfloat16* a_dkt_1 = p_flat + 4096;
        transpose_64x64(a_dkt_0, d_flat);
        transpose_64x64(a_dkt_1, d_flat + 4096);
        
        __nv_bfloat16* a_q_0_dk = o_tile_0;
        __nv_bfloat16* a_q_1_dk = o_tile_1;
        transpose_64x64(a_q_0_dk, q_tile_0);
        transpose_64x64(a_q_1_dk, q_tile_1);
        
        gemm_64x64x64_cta_g2(dk_flat_0, a_dkt_0, a_q_0_dk);
        gemm_64x64x64_cta_g2(dk_flat_1, a_dkt_1, a_q_1_dk);
        
        __nv_bfloat16* a_pt_0 = dp_flat;
        __nv_bfloat16* a_pt_1 = dp_flat + 4096;
        transpose_64x64(a_pt_0, p_flat);
        transpose_64x64(a_pt_1, p_flat + 4096);
        
        __nv_bfloat16* a_do_0_dv = k_tile_0;
        __nv_bfloat16* a_do_1_dv = k_tile_1;
        transpose_64x64(a_do_0_dv, do_tile_0);
        transpose_64x64(a_do_1_dv, do_tile_1);
        
        gemm_64x64x64_cta_g2(dv_flat_0, a_pt_0, a_do_0_dv);
        gemm_64x64x64_cta_g2(dv_flat_1, a_pt_1, a_do_1_dv);
        
        __syncthreads();
    }
    
    tma_store_fence_fn();
    store_64x128(&tma_dK, dk_flat_0, dk_flat_1, s_off_global);
    store_64x128(&tma_dV, dv_flat_0, dv_flat_1, s_off_global);
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
            CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    };
    
    make_tma(&tma_Q, Q.data_ptr());
    make_tma(&tma_K, K.data_ptr());
    make_tma(&tma_V, V.data_ptr());
    make_tma(&tma_O, O.data_ptr());
    make_tma(&tma_dO, dO.data_ptr());
    make_tma(&tma_dQ, dQ.data_ptr());
    make_tma(&tma_dK, dK.data_ptr());
    make_tma(&tma_dV, dV.data_ptr());
    
    dim3 grid((S + 127) / 128, B * H);
    dim3 block(128);
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 186 * 1024;
    config.stream = stream;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    CUDA_CHECK(cudaFuncSetAttribute(kernel_fused, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes));

    CUDA_CHECK(cudaLaunchKernelEx(&config, kernel_fused, 
        tma_Q, tma_K, tma_V, tma_O, tma_dO, tma_dQ, tma_dK, tma_dV, 
        static_cast<const float*>(L.data_ptr()), S, 1.0f / sqrtf(128.0f)));
        
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_mha_bwd