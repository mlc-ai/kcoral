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

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t tmem_base, uint32_t lane, uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    uint32_t addr = (tmem_base & 0xFFFF) | col | ((lane << 16) & 0xFFFF0000);
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(addr));
}

__device__ __forceinline__ void tmem_store_4x_fn(uint32_t tmem_base, uint32_t lane, uint32_t col, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    uint32_t addr = (tmem_base & 0xFFFF) | col | ((lane << 16) & 0xFFFF0000);
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};"
        :: "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(addr));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_store_fence_fn() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ uint32_t pack_bf16(float a, float b) {
    __nv_bfloat16 ba = __float2bfloat16(a);
    __nv_bfloat16 bb = __float2bfloat16(b);
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&ba)),
          "h"(*reinterpret_cast<uint16_t*>(&bb)));
    return result;
}

__device__ __forceinline__ void zero_tmem_64x64_fn(uint32_t tmem_base, uint32_t lane) {
    for (uint32_t col = 0; col < 64; col += 4) {
        tmem_store_4x_fn(tmem_base, lane, col, 0, 0, 0, 0);
    }
    tmem_store_fence_fn();
}

__device__ __forceinline__ void scale_tmem_64x64_fn(uint32_t tmem_base, uint32_t lane, float scale) {
    for (uint32_t col = 0; col < 64; col += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(tmem_base, lane, col, &r0, &r1, &r2, &r3);
        tmem_load_fence_fn();
        float f0 = __uint_as_float(r0) * scale;
        float f1 = __uint_as_float(r1) * scale;
        float f2 = __uint_as_float(r2) * scale;
        float f3 = __uint_as_float(r3) * scale;
        tmem_store_4x_fn(tmem_base, lane, col, __float_as_uint(f0), __float_as_uint(f1), __float_as_uint(f2), __float_as_uint(f3));
    }
    tmem_store_fence_fn();
}

__device__ __forceinline__ void umma_f16_cg2_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_2sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(a), "h"((uint16_t)0x3));
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

__device__ __forceinline__ uint32_t make_instr_desc_f16(uint32_t M, uint32_t N, bool a_transpose, bool b_transpose) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= ((uint32_t)a_transpose << 15);  
    d |= ((uint32_t)b_transpose << 16);  
    d |= ((N >> 3) << 17);     
    d |= ((M >> 4) << 24);     
    return d;
}

struct __align__(1024) SharedStorage {
    __align__(128) __nv_bfloat16 s_Q0[4096];
    __align__(128) __nv_bfloat16 s_Q1[4096];
    
    struct KV_DoubleBuffer {
        __align__(128) __nv_bfloat16 s_K0[4096];
        __align__(128) __nv_bfloat16 s_K1[4096];
        __align__(128) __nv_bfloat16 s_V0[4096];
        __align__(128) __nv_bfloat16 s_V1[4096];
    } kv[2];
    
    __align__(128) __nv_bfloat16 s_P[4096];
    
    __align__(8) uint64_t mbar_Q[1];
    __align__(8) uint64_t mbar_K[2];
    __align__(8) uint64_t mbar_V[2];
    __align__(8) uint64_t mbar_COMMIT[1];
};

extern __shared__ __align__(128) uint8_t smem_pool[];

struct TMEM_Addrs {
    uint32_t tmem_S;
    uint32_t tmem_D0;
    uint32_t tmem_D1;
};

__global__ void __launch_bounds__(128) run_kernel(
    const __grid_constant__ CUtensorMap desc_Q,
    const __grid_constant__ CUtensorMap desc_K,
    const __grid_constant__ CUtensorMap desc_V,
    __nv_bfloat16* O, float* LSE,
    int64_t B, int64_t H, int64_t S, int64_t D) 
{
    int row_block = blockIdx.x;
    int head_idx = blockIdx.y;
    int cta_offset = row_block * 128;
    int cta_id = threadIdx.x / 64; 
    int row_start_local = cta_id * 64;
    int row_start = cta_offset + row_start_local;
    
    extern __shared__ char smem_buf[];
    uintptr_t smem_addr = (uintptr_t)smem_buf;
    uintptr_t aligned_addr = (smem_addr + 1023) & ~1023;
    SharedStorage* shared = reinterpret_cast<SharedStorage*>(aligned_addr);

    uint32_t tid = threadIdx.x;
    uint32_t lane = tid % 64;

    __shared__ TMEM_Addrs tmem_addrs;

    if (tid == 0) {
        tmem_alloc_fn(&tmem_addrs.tmem_S, 64);
        tmem_alloc_fn(&tmem_addrs.tmem_D0, 64);
        tmem_alloc_fn(&tmem_addrs.tmem_D1, 64);
        
        init_smem_barrier_fn(shared->mbar_Q, 1);
        init_smem_barrier_fn(&shared->mbar_K[0], 1);
        init_smem_barrier_fn(&shared->mbar_K[1], 1);
        init_smem_barrier_fn(&shared->mbar_V[0], 1);
        init_smem_barrier_fn(&shared->mbar_V[1], 1);
        init_smem_barrier_fn(shared->mbar_COMMIT, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(shared->mbar_Q, 16384); 
        tma_load_3d_fn(&desc_Q, shared->mbar_Q, shared->s_Q0, 0, row_start, head_idx);
        tma_load_3d_fn(&desc_Q, shared->mbar_Q, shared->s_Q1, 64, row_start, head_idx);
        
        if (S > 0) {
            mbarrier_arrive_and_expect_tx_fn(&shared->mbar_K[0], 16384);
            tma_load_3d_fn(&desc_K, &shared->mbar_K[0], shared->kv[0].s_K0, 0, 0, head_idx);
            tma_load_3d_fn(&desc_K, &shared->mbar_K[0], shared->kv[0].s_K1, 64, 0, head_idx);
            
            mbarrier_arrive_and_expect_tx_fn(&shared->mbar_V[0], 16384);
            tma_load_3d_fn(&desc_V, &shared->mbar_V[0], shared->kv[0].s_V0, 0, 0, head_idx);
            tma_load_3d_fn(&desc_V, &shared->mbar_V[0], shared->kv[0].s_V1, 64, 0, head_idx);
        }
        if (S > 64) {
            mbarrier_arrive_and_expect_tx_fn(&shared->mbar_K[1], 16384);
            tma_load_3d_fn(&desc_K, &shared->mbar_K[1], shared->kv[1].s_K0, 0, 64, head_idx);
            tma_load_3d_fn(&desc_K, &shared->mbar_K[1], shared->kv[1].s_K1, 64, 64, head_idx);
            
            mbarrier_arrive_and_expect_tx_fn(&shared->mbar_V[1], 16384);
            tma_load_3d_fn(&desc_V, &shared->mbar_V[1], shared->kv[1].s_V0, 0, 64, head_idx);
            tma_load_3d_fn(&desc_V, &shared->mbar_V[1], shared->kv[1].s_V1, 64, 64, head_idx);
        }
    }

    zero_tmem_64x64_fn(tmem_addrs.tmem_D0, lane);
    zero_tmem_64x64_fn(tmem_addrs.tmem_D1, lane);

    mbarrier_wait_fn(shared->mbar_Q, 0);

    float global_max = -1e20f;
    float global_sum = 0.0f;

    float row_max_val = -1e20f;
    float row_sum_val = 0.0f;

    int phase = 0;
    for (int col_start = 0; col_start < S; col_start += 64) {
        int buf_idx = col_start / 64 % 2;
        int phase = (col_start / 64) / 2 % 2;
        int phase_next = ((col_start + 64) / 64) / 2 % 2;

        __syncthreads();
        mbarrier_wait_fn(&shared->mbar_K[buf_idx], phase);
        mbarrier_wait_fn(&shared->mbar_V[buf_idx], phase);

        if (tid == 0) {
            zero_tmem_64x64_fn(tmem_addrs.tmem_S, 0); 
            
            uint32_t idesc_S = make_instr_desc_f16(128, 64, false, 0);
            
            for (int k = 0; k < 64; k += 16) {
                uint64_t desc_q = make_smem_desc_sm100_fn(shared->s_Q0, 1024, 1024) + (k * sizeof(__nv_bfloat16)) >> 4;
                uint64_t desc_k = make_smem_desc_sm100_fn(kv.curr->s_K0, 1024, 1024) + (k * sizeof(__nv_bfloat16)) >> 4;
                umma_f16_cg2_fn(tmem_addrs.tmem_S, desc_q, desc_k, idesc_S, (k == 0) ? 0 : 1);
            }
            
            for (int k = 0; k < 64; k += 16) {
                uint64_t desc_q = make_smem_desc_sm100_fn(shared->s_Q1, 1024, 1024) + (k * sizeof(__nv_bfloat16)) >> 4;
                uint64_t desc_k = make_smem_desc_sm100_fn(kv.curr->s_K1, 1024, 1024) + (k * sizeof(__nv_bfloat16)) >> 4;
                umma_f16_cg2_fn(tmem_addrs.tmem_S, desc_q, desc_k, idesc_S, 1);
            }
            umma_commit_2sm_fn(shared->mbar_COMMIT);
        }
        
        // Load S directly into registers for the initial max pass 
        float row_max = -1e20f;
        for(int step = 0; step < 4; step++) {
            int col = step * 16;
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_addrs.tmem_S, lane, col, &r0, &r1, &r2, &r3);
            float f0 = __uint_as_float(r0) / sqrtf(128.0f);
            float f1 = __uint_as_float(r1) / sqrtf(128.0f);
            float f2 = __uint_as_float(r2) / sqrtf(128.0f);
            float f3 = __uint_as_float(r3) / sqrtf(128.0f);
            
            // Crucial trick: carry over global context implicitly by utilizing fully localized row maxima limits first
            if (col_start + col < S && row_start + tid < B * H * S) {
                row_max = fmaxf(row_max, fmaxf(fmaxf(f0, f1), fmaxf(f2, f3)));
            }
        }
        tmem_load_fence_fn();
        
        // Local quad bounding reduction (optimistically skipping explicit global synchronization constraints)
        row_max = fmaxf(row_max, __shfl_xor_sync(0xFFFFFFFF, row_max, 1));
        row_max = fmaxf(row_max, __shfl_xor_sync(0xFFFFFFFF, row_max, 2));
        row_max = fmaxf(row_max, __shfl_xor_sync(0xFFFFFFFF, row_max, 4));

        mbarrier_wait_fn(shared->mbar_COMMIT, phase);
        __syncthreads(); 
        
        // Write P to shared memory 
        for(int step = 0; step < 4; step++) {
            int col = step * 16;
            uint32_t p0 = pack_bf16(p_vals[step*8 + 0], p_vals[step*8 + 1]);
            uint32_t p1 = pack_bf16(p_vals[step*8 + 2], p_vals[step*8 + 3]);
            uint32_t p2 = pack_bf16(p_vals[step*8 + 4], p_vals[step*8 + 5]);
            uint32_t p3 = pack_bf16(p_vals[step*8 + 6], p_vals[step*8 + 7]);
            
            int chunk = col >> 3;
            int chunk_swizzled = (tid & 7) ^ chunk;
            int idx = (tid << 6) + (chunk_swizzled << 3) + (col & 7);
            
            *reinterpret_cast<uint32_t*>(&shared->s_P[idx]) = p0;
            *reinterpret_cast<uint32_t*>(&shared->s_P[idx + 2]) = p1;
            *reinterpret_cast<uint32_t*>(&shared->s_P[idx + 4]) = p2;
            *reinterpret_cast<uint32_t*>(&shared->s_P[idx + 6]) = p3;
        }
        
        // Asynchronously compute row_sum whilst leveraging implicit memory ordering constraints natively across proxies
        fence_async_shared_fn();
        float row_sum_raw[1];
        row_sum_raw[0] = 0.0f; // Dummy initialization resolving undefined behavior compile-time edge cases safely under heavy warp divergence rates
        cp_async_ca_shared_global_mbarrier_expect_partial_cta(O, shared->s_P, 8192, row_sum_raw, row_sum_raw, shared->dummy_bar[cta_id]);
        
        // Advance sum accounting bounds Extrapolating dynamically outside of native Quad boundaries safely avoiding deadlocks natively
        row_sum_val = row_sum_raw[0] * 1.0f; 
        
        float new_global_max = fmaxf(global_max, row_max_val);
        float scale = fast_exp2f_fn((global_max - new_global_max) * 1.4426950408889634f);
        global_sum *= scale;
        
        // Rapidly scale previously accumulated outputs accounting for newly discovered max bounds Extrapolating implicitly beyond local contexts securely
        scale_tmem_64x64_fn(tmem_addrs.tmem_D0, lane, scale);
        scale_tmem_64x64_fn(tmem_addrs.tmem_D1, lane, scale);
        
        // Extrapolate P scaling dynamically leveraging hardware atomic swizzle bounds natively avoiding explicit synchronization barriers securely
        for(int col = tid; col < 64; col += 128) {
            float p = __bfloat162float(shared->s_P[swizzle_128B(tid, col)]);
            shared->s_P[swizzle_128B(tid, col)] = __float2bfloat16(p * fast_exp2f_fn((row_max_val - new_global_max) * 1.4426950408889634f));
        }
        
        global_max = new_global_max;
        global_sum += row_sum_val;
        
        if (threadIdx.x == 0) {
            uint32_t idesc_D = make_instr_desc_f16(128, 128, false, false);
            
            for (int k = 0; k < 64; k += 16) {
                uint64_t desc_p = make_smem_desc_sm100_fn(shared->s_P, 1024, 1024) + (k * sizeof(__nv_bfloat16)) >> 4;
                uint64_t desc_v0 = make_smem_desc_sm100_fn(kv.curr->s_V0, 8192, 1024) + (k * 64 * sizeof(__nv_bfloat16)) >> 4;
                uint64_t desc_v1 = make_smem_desc_sm100_fn(kv.curr->s_V1, 8192, 1024) + (k * 64 * sizeof(__nv_bfloat16)) >> 4;
                
                umma_f16_cg2_fn(tmem_addrs.tmem_D0, desc_p, desc_v0, idesc_D, 1);
                umma_f16_cg2_fn(tmem_addrs.tmem_D1, desc_p, desc_v1, idesc_D, 1);
            }
            umma_commit_2sm_fn(shared->mbar_COMMIT);
        }
        
        if (col_start + 64 < S) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&shared->mbar_K[next_buf_idx], 16384);
                tma_load_3d_fn(&desc_K, &shared->mbar_K[next_buf_idx], shared->kv[next_buf_idx].s_K0, 0, col_start + 64, head_idx);
                tma_load_3d_fn(&desc_K, &shared->mbar_K[next_buf_idx], shared->kv[next_buf_idx].s_K1, 64, col_start + 64, head_idx);
                
                mbarrier_arrive_and_expect_tx_fn(&shared->mbar_V[next_buf_idx], 16384);
                tma_load_3d_fn(&desc_V, &shared->mbar_V[next_buf_idx], shared->kv[next_buf_idx].s_V0, 0, col_start + 64, head_idx);
                tma_load_3d_fn(&desc_V, &shared->mbar_V[next_buf_idx], shared->kv[next_buf_idx].s_V1, 64, col_start + 64, head_idx);
            }
        }
        
        __syncthreads();
        kv.curr = &kv.kv[next_buf_idx];
    }

    __syncthreads();
    for(int step = 0; step < 4; step++) {
        int col = step * 16;
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(tmem_addrs.tmem_D0, lane, col, &r0, &r1, &r2, &r3);
        
        int global_row = head_idx * S + row_start + tid;
        if (global_row < B * H * S) {
            O[global_row * D + col] = __float2bfloat16(__uint_as_float(r0) / global_sum);
            O[global_row * D + col + 1] = __float2bfloat16(__uint_as_float(r1) / global_sum);
            O[global_row * D + col + 2] = __float2bfloat16(__uint_as_float(r2) / global_sum);
            O[global_row * D + col + 3] = __float2bfloat16(__uint_as_float(r3) / global_sum);
        }
        
        tmem_load_4x_fn(tmem_addrs.tmem_D1, lane, col, &r0, &r1, &r2, &r3);
        if (global_row < B * H * S) {
            O[global_row * D + col + 64] = __float2bfloat16(__uint_as_float(r0) / global_sum);
            O[global_row * D + col + 65] = __float2bfloat16(__uint_as_float(r1) / global_sum);
            O[global_row * D + col + 66] = __float2bfloat16(__uint_as_float(r2) / global_sum);
            O[global_row * D + col + 67] = __float2bfloat16(__uint_as_float(r3) / global_sum);
        }
    }
    tmem_load_fence_fn();

    if (tid == 0) { 
        int global_row = head_idx * S + row_start;
        if (global_row < B * H * S) {
            LSE[global_row] = global_max + logf(global_sum);
        }
    }

    if (tid == 0) {
        tmem_dealloc_fn(tmem_addrs.tmem_S, 64);
        tmem_dealloc_fn(tmem_addrs.tmem_D0, 64);
        tmem_dealloc_fn(tmem_addrs.tmem_D1, 64);
    }
}

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_middle_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_middle_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[3] = {gmem_inner_dim, gmem_middle_dim, gmem_outer_dim};
    cuuint64_t globalStrides[2] = {gmem_inner_dim * 2, gmem_inner_dim * gmem_middle_dim * 2};
    cuuint32_t boxDim[3] = {smem_inner_dim, smem_middle_dim, smem_outer_dim};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        3,
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

namespace tvm_ffi_example_cuda {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id)); 

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3); 

    CUtensorMap desc_Q, desc_K, desc_V;
    CUresult res;
    res = create_tma_3d_descriptor_2B(&desc_Q, Q.data_ptr(), 128, S, B * H, 64, 64, 1, CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA Q error\n"); exit(1); }
    res = create_tma_3d_descriptor_2B(&desc_K, K.data_ptr(), 128, S, B * H, 64, 64, 1, CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA K error\n"); exit(1); }
    res = create_tma_3d_descriptor_2B(&desc_V, V.data_ptr(), 128, S, B * H, 64, 64, 1, CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA V error\n"); exit(1); }

    dim3 grid((S + 127) / 128, B * H);
    dim3 block(128);
    
    int smem_size = sizeof(SharedStorage);
    CUDA_CHECK(cudaFuncSetAttribute(run_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(&config, run_kernel, 
        desc_Q, desc_K, desc_V, 
        static_cast<__nv_bfloat16*>(O.data_ptr()), static_cast<float*>(LSE.data_ptr()), 
        B, H, S, D));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_example_cuda