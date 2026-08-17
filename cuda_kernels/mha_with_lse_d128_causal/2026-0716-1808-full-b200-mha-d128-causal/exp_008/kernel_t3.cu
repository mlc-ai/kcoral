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
                (int)_e, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_causal_attention {

// -------------------------------------------------------------------------------------
// Helper Functions for SM100
// -------------------------------------------------------------------------------------

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.expect_tx.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_fn(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])) : "memory");
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

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ uint64_t make_smem_desc_swizzle_k_major(void* smem_ptr, uint32_t lbo, uint32_t k_step) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    addr += k_step * 2; 
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16; 
    d |= ((uint64_t)(1024 >> 4)) << 32; // SBO = 1024
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_swizzle_n_major(void* smem_ptr, uint32_t lbo, uint32_t k_step) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    addr += k_step * 2048; // Advance row by k_step * 16 spans
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16; 
    d |= ((uint64_t)(1024 >> 4)) << 32; // SBO = 1024
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_k_major(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= (0u << 15);   // a_major = 0 (A is K-Major)
    d |= (0u << 16);   // b_major = 0 (B is K-Major)
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_k_n_major(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= (0u << 15);   // a_major = 0 (A is K-Major)
    d |= (1u << 16);   // b_major = 1 (B is N-Major)
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

__device__ __forceinline__ void gemm_q_k(
    __nv_bfloat16* s_Q_flat, __nv_bfloat16* cur_K, uint32_t tmem_c) {
    for (uint32_t k_step = 0; k_step < 128; k_step += 16) {
        uint64_t desc_A = make_smem_desc_swizzle_k_major(s_Q_flat, 1, k_step);
        uint64_t desc_B = make_smem_desc_swizzle_k_major(cur_K, 1, k_step);
        uint32_t idesc = make_instr_desc_k_major(128, 128);
        
        if (k_step == 0) {
            umma_f16_cg1_fn(tmem_c, desc_A, desc_B, idesc, 0);
        } else {
            umma_f16_cg1_fn(tmem_c, desc_A, desc_B, idesc, 1);
        }
    }
}

__device__ __forceinline__ void gemm_p_v(
    __nv_bfloat16* s_P, __nv_bfloat16* cur_V, uint32_t tmem_o) {
    for (uint32_t k_step = 0; k_step < 128; k_step += 16) {
        uint64_t desc_P = make_smem_desc_swizzle_k_major(s_P, 1, k_step);
        uint64_t desc_V = make_smem_desc_swizzle_n_major(cur_V, 16384, k_step);
        uint32_t idesc_P_V = make_instr_desc_k_n_major(128, 128);
        umma_f16_cg1_fn(tmem_o, desc_P, desc_V, idesc_P_V, 1);
    }
}

__device__ __forceinline__ void commit_and_wait(uint64_t* bar, uint32_t phase) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])) : "memory");
    mbarrier_wait_fn(bar, phase);
    fence_proxy_async_fn();
}

// -------------------------------------------------------------------------------------
// Causal Attention Kernel
// -------------------------------------------------------------------------------------

__global__ void causal_attention_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S_len, int B, int H)
{
    extern __shared__ char smem_raw[];
    size_t smem_addr = (size_t)smem_raw;
    smem_addr = (smem_addr + 1023) & ~1023;
    char* smem = (char*)smem_addr;

    __nv_bfloat16* s_Q_flat = (__nv_bfloat16*)smem;                   // 32768 Bytes
    __nv_bfloat16* s_K_flat = (__nv_bfloat16*)(smem + 32768);         // 65536 Bytes
    __nv_bfloat16* s_V_flat = (__nv_bfloat16*)(smem + 98304);         // 65536 Bytes
    __nv_bfloat16* s_P      = (__nv_bfloat16*)(smem + 163840);        // 32768 Bytes
    
    char* barrier_smem = smem + 196608;
    barrier_smem = (char*)(((size_t)barrier_smem + 1023) & ~1023);
    uint64_t* mbar_Q = (uint64_t*)barrier_smem;
    uint64_t* mbar_K = (uint64_t*)(barrier_smem + 8);
    uint64_t* mbar_V = (uint64_t*)(barrier_smem + 16);
    uint64_t* mbar_C = (uint64_t*)(barrier_smem + 40);
    uint64_t* mbar_O = (uint64_t*)(barrier_smem + 48);
    
    uint32_t* tmem_c_addr = (uint32_t*)(barrier_smem + 64);
    uint32_t* tmem_o_addr = (uint32_t*)(barrier_smem + 72);

    int bh = blockIdx.x * gridDim.y + blockIdx.y;
    int batch_idx = bh / H;
    int head_idx = bh % H;
    int bid = blockIdx.z;
    int q_blk = bid * 128;
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    
    float* LSE_ptr = LSE + batch_idx * H * S_len + head_idx * S_len;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(&mbar_K[0], 1);
        init_smem_barrier_fn(&mbar_K[1], 1);
        init_smem_barrier_fn(&mbar_V[0], 1);
        init_smem_barrier_fn(&mbar_V[1], 1);
        init_smem_barrier_fn(mbar_C, 1);
        init_smem_barrier_fn(mbar_O, 1);
        
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 32768); 
        
        tma_load_2d_fn(&tma_Q, mbar_Q, s_Q_flat, 0, bh * S_len + q_blk);
        tma_load_2d_fn(&tma_Q, mbar_Q, s_Q_flat + 8192, 64, bh * S_len + q_blk);
        
        tmem_alloc_fn(tmem_c_addr, 128);
        tmem_alloc_fn(tmem_o_addr, 128);
    }
    
    __syncthreads();
    mbarrier_wait_fn(mbar_Q, 0);
    fence_proxy_async_fn();
    
    uint32_t tmem_c = tmem_c_addr[0];
    uint32_t tmem_o = tmem_o_addr[0];
    
    float prev_max = -1e20f;
    float curr_max = -1e20f;
    float curr_sum = 0.0f;
    
    int num_blocks = (S_len + 127) / 128;
    int max_j = min(num_blocks, (q_blk / 128) + 1);
    
    for (int j = 0; j < max_j; ++j) {
        int phase = j % 2;
        int next_phase = (j + 1) % 2;
        
        __nv_bfloat16* cur_K = &s_K_flat[phase * 16384];
        __nv_bfloat16* cur_V = &s_V_flat[phase * 16384];
        __nv_bfloat16* nxt_K = &s_K_flat[next_phase * 16384];
        __nv_bfloat16* nxt_V = &s_V_flat[next_phase * 16384];
        
        int k_blk = j * 128;
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_K[phase], 32768);
            tma_load_2d_fn(&tma_K, &mbar_K[phase], cur_K, 0, bh * S_len + k_blk);
            tma_load_2d_fn(&tma_K, &mbar_K[phase], cur_K + 8192, 64, bh * S_len + k_blk);
            
            mbarrier_arrive_and_expect_tx_fn(&mbar_V[phase], 32768);
            tma_load_2d_fn(&tma_V, &mbar_V[phase], cur_V, 0, bh * S_len + k_blk);
            tma_load_2d_fn(&tma_V, &mbar_V[phase], cur_V + 8192, 64, bh * S_len + k_blk);
            
            if (j + 1 < max_j) {
                int next_k_blk = (j + 1) * 128;
                mbarrier_arrive_and_expect_tx_fn(&mbar_K[next_phase], 32768);
                tma_load_2d_fn(&tma_K, &mbar_K[next_phase], nxt_K, 0, bh * S_len + next_k_blk);
                tma_load_2d_fn(&tma_K, &mbar_K[next_phase], nxt_K + 8192, 64, bh * S_len + next_k_blk);
                
                mbarrier_arrive_and_expect_tx_fn(&mbar_V[next_phase], 32768);
                tma_load_2d_fn(&tma_V, &mbar_V[next_phase], nxt_V, 0, bh * S_len + next_k_blk);
                tma_load_2d_fn(&tma_V, &mbar_V[next_phase], nxt_V + 8192, 64, bh * S_len + next_k_blk);
            }
        }
        
        __syncthreads(); 
        mbarrier_wait_fn(&mbar_K[phase], (j / 2) % 2);
        mbarrier_wait_fn(&mbar_V[phase], (j / 2) % 2);
        fence_proxy_async_fn();
        
        if (threadIdx.x == 0) {
            fence_proxy_async_fn();
            
            gemm_q_k(s_Q_flat, cur_K, tmem_c);
        }
        
        commit_and_wait(mbar_C, (j * 2) % 2); 
        
        float thread_max[32][16];
        float thread_sum[32][16];
        
        for (int i = 0; i < 16; ++i) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(i * 4));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            
            int q_idx = q_blk + lane_id + i * 4 * 32;
            int k_idx_0 = k_blk + i * 4;
            
            if (k_idx_0 >= S_len || k_idx_0 > q_idx) f0 = -1e20f;
            if (k_idx_0 + 1 >= S_len || k_idx_0 + 1 > q_idx) f1 = -1e20f;
            if (k_idx_0 + 2 >= S_len || k_idx_0 + 2 > q_idx) f2 = -1e20f;
            if (k_idx_0 + 3 >= S_len || k_idx_0 + 3 > q_idx) f3 = -1e20f;
            
            thread_max[lane_id][i] = fmaxf(fmaxf(f0, f1), fmaxf(f2, f3));
            thread_sum[lane_id][i] = 0.0f;
        }
        
        float max_val = -1e20f;
        for (int i = 0; i < 16; ++i) {
            max_val = fmaxf(max_val, thread_max[lane_id][i]);
        }
        
        float local_max[4][16];
        for(int i = 0; i < 16; ++i) local_max[warp_id][i] = max_val;
        
        for(int i = 0; i < 16; ++i) {
            local_max[warp_id][i] = fmaxf(local_max[warp_id][i], __shfl_xor_sync(0xFFFFFFFF, local_max[warp_id][i], 16));
            local_max[warp_id][i] = fmaxf(local_max[warp_id][i], __shfl_xor_sync(0xFFFFFFFF, local_max[warp_id][i], 8));
            local_max[warp_id][i] = fmaxf(local_max[warp_id][i], __shfl_xor_sync(0xFFFFFFFF, local_max[warp_id][i], 4));
            local_max[warp_id][i] = fmaxf(local_max[warp_id][i], __shfl_xor_sync(0xFFFFFFFF, local_max[warp_id][i], 2));
            local_max[warp_id][i] = fmaxf(local_max[warp_id][i], __shfl_xor_sync(0xFFFFFFFF, local_max[warp_id][i], 1));
        }
        
        curr_max = fmaxf(prev_max, local_max[warp_id][0]);
        float scale = expf(prev_max - curr_max);
        
        float sum_val = 0.0f;
        for (int i = 0; i < 16; ++i) {
            float exp_scale = expf(local_max[warp_id][i] - curr_max);
            thread_sum[lane_id][i] *= exp_scale;
            sum_val += thread_sum[lane_id][i] * exp_scale;
        }
        
        float local_sum[4][16];
        for(int i = 0; i < 16; ++i) local_sum[warp_id][i] = sum_val;
        
        for(int i = 0; i < 16; ++i) {
            local_sum[warp_id][i] += __shfl_xor_sync(0xFFFFFFFF, local_sum[warp_id][i], 16);
            local_sum[warp_id][i] += __shfl_xor_sync(0xFFFFFFFF, local_sum[warp_id][i], 8);
            local_sum[warp_id][i] += __shfl_xor_sync(0xFFFFFFFF, local_sum[warp_id][i], 4);
            local_sum[warp_id][i] += __shfl_xor_sync(0xFFFFFFFF, local_sum[warp_id][i], 2);
            local_sum[warp_id][i] += __shfl_xor_sync(0xFFFFFFFF, local_sum[warp_id][i], 1);
        }
        
        curr_sum = local_sum[warp_id][0] * scale;
        
        for (int i = 0; i < 16; ++i) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(i * 4));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            
            int q_idx = q_blk + lane_id + i * 4 * 32;
            int k_idx_0 = k_blk + i * 4;
            
            float exp_scale = expf(local_max[warp_id][i] - curr_max);
            
            float f0_exp = 0.0f, f1_exp = 0.0f, f2_exp = 0.0f, f3_exp = 0.0f;
            
            if (k_idx_0 >= S_len || k_idx_0 > q_idx) f0_exp = 0.0f; else f0_exp = expf(f0 - local_max[warp_id][i]) * exp_scale;
            if (k_idx_0 + 1 >= S_len || k_idx_0 + 1 > q_idx) f1_exp = 0.0f; else f1_exp = expf(f1 - local_max[warp_id][i]) * exp_scale;
            if (k_idx_0 + 2 >= S_len || k_idx_0 + 2 > q_idx) f2_exp = 0.0f; else f2_exp = expf(f2 - local_max[warp_id][i]) * exp_scale;
            if (k_idx_0 + 3 >= S_len || k_idx_0 + 3 > q_idx) f3_exp = 0.0f; else f3_exp = expf(f3 - local_max[warp_id][i]) * exp_scale;
            
            int col = i * 4;
            int row = lane_id + warp_id * 32;
            
            int chunk_x = col / 8;
            int swizzled_chunk_x = chunk_x ^ (row % 8);
            int swizzled_col = swizzled_chunk_x * 8 + (col % 8);
            int swizzled_idx = row * 128 + swizzled_col;
            
            s_P[swizzled_idx] = __float2bfloat16(f0_exp);
            s_P[swizzled_idx + 1] = __float2bfloat16(f1_exp);
            s_P[swizzled_idx + 2] = __float2bfloat16(f2_exp);
            s_P[swizzled_idx + 3] = __float2bfloat16(f3_exp);
        }
        
        __syncthreads(); 
        
        if (threadIdx.x == 0) {
            fence_proxy_async_fn();
            
            gemm_p_v(s_P, cur_V, tmem_o);
        }
        
        __syncthreads(); 
        commit_and_wait(mbar_O, (j * 2 + 1) % 2); 
        
        prev_max = curr_max;
        float safe_final_sum = curr_sum < 1e-20f ? 1e-20f : curr_sum;
        float lse_val = curr_max + logf(safe_final_sum);
        
        if (q_blk + lane_id < S_len) {
            LSE_ptr[q_blk + lane_id] = lse_val;
        }
    }
    
    __syncthreads(); 
    
    for (int i = 0; i < 16; ++i) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
           : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(i * 4));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float f0 = __uint_as_float(r0);
        float f1 = __uint_as_float(r1);
        float f2 = __uint_as_float(r2);
        float f3 = __uint_as_float(r3);
        
        int q_idx = q_blk + lane_id + i * 4 * 32;
        int k_idx_0 = i * 4;
        
        if (q_idx < S_len) {
            __nv_bfloat16* out = O + batch_idx * H * S_len * 128 + head_idx * S_len * 128 + q_idx * 128;
            if (k_idx_0 < 128) out[k_idx_0] = __float2bfloat16(f0);
            if (k_idx_0 + 1 < 128) out[k_idx_0 + 1] = __float2bfloat16(f1);
            if (k_idx_0 + 2 < 128) out[k_idx_0 + 2] = __float2bfloat16(f2);
            if (k_idx_0 + 3 < 128) out[k_idx_0 + 3] = __float2bfloat16(f3);
        }
    }
    
    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_c, 128);
        tmem_dealloc_fn(tmem_o, 128);
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
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  
  int64_t B = Q.size(0);
  int64_t H = Q.size(1);
  int64_t S_len = Q.size(2);
  int64_t D = Q.size(3);
  
  if (D != 128) {
    fprintf(stderr, "Error: D must be 128\n");
    exit(1);
  }
  
  const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
  
  __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
  float* LSE_data = static_cast<float*>(LSE.data_ptr());
  
  CUtensorMap tma_Q, tma_K, tma_V;
  
  CU_CHECK(create_tma_2d_descriptor_2B(
      &tma_Q, (void*)Q_data, 128, B * H * S_len, 64, 128,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  ));
  CU_CHECK(create_tma_2d_descriptor_2B(
      &tma_K, (void*)K_data, 128, B * H * S_len, 64, 128,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  ));
  CU_CHECK(create_tma_2d_descriptor_2B(
      &tma_V, (void*)V_data, 128, B * H * S_len, 64, 128,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  ));
  
  int threads = 128;
  dim3 grid(B, H, (S_len + 127) / 128);
  
  int smem_size = 197632; 
  cudaFuncSetAttribute(causal_attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  
  causal_attention_kernel<<<grid, threads, smem_size, static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id))>>>(
      tma_Q, tma_K, tma_V, O_data, LSE_data, S_len, B, H);
      
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id))));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_causal_attention::run);

} // namespace tvm_ffi_causal_attention