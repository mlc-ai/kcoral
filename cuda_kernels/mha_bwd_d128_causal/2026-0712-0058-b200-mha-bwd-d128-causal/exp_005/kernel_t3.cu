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
#include <algorithm>
#include <vector>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_fa4 {

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

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
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

__device__ __forceinline__ uint64_t make_smem_desc_k_major_128b(void* smem_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((1 & 0x3FFFF) >> 4) << 16; 
    d |= (uint64_t)((1024 & 0x3FFFF) >> 4) << 32; 
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
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ void tmem_wait_fn(uint64_t* bar, uint32_t phase) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared.b64 [%0];" :: "r"(a));
    mbarrier_wait_fn(bar, phase);
}

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ __nv_bfloat16 read_swizzled(const char* smem, int row, int col) {
    int col_bytes = col * 2;
    int chunk = col_bytes / 16;
    int chunk_swizzled = chunk ^ (row % 8);
    int final_col_bytes = (chunk_swizzled * 16) + (col_bytes % 16);
    int idx = row * 128 + final_col_bytes;
    return *( (__nv_bfloat16*) (&smem[idx]) );
}

__device__ __forceinline__ void write_swizzled(char* smem, int row, int col, float val) {
    int col_bytes = col * 2;
    int chunk = col_bytes / 16;
    int chunk_swizzled = chunk ^ (row % 8);
    int final_col_bytes = (chunk_swizzled * 16) + (col_bytes % 16);
    int idx = row * 128 + final_col_bytes;
    __nv_bfloat16 a = __float2bfloat16(val);
    *( (__nv_bfloat16*) (&smem[idx]) ) = a;
}

__device__ void transpose_64x64_swizzled(char* src, char* dst) {
    for (int i = threadIdx.x; i < 4096; i += blockDim.x) {
        int row = i / 64;
        int col = i % 64;
        float val = __bfloat162float(read_swizzled(src, row, col));
        write_swizzled(dst, col, row, val);
    }
}

__device__ void gemm_tmem(uint32_t tmem_c_ptr, char* smem_A0, char* smem_A1, char* smem_B0, char* smem_B1, uint32_t idesc, uint32_t accum) {
    uint64_t desc_a0 = make_smem_desc_k_major_128b(smem_A0);
    uint64_t desc_b0 = make_smem_desc_k_major_128b(smem_B0);
    uint32_t accum_i = accum;
    umma_f16_cg2_fn(tmem_c_ptr, desc_a0, desc_b0, idesc, accum_i);
    
    uint64_t desc_a1 = make_smem_desc_k_major_128b(smem_A1);
    uint64_t desc_b1 = make_smem_desc_k_major_128b(smem_B1);
    umma_f16_cg2_fn(tmem_c_ptr, desc_a1, desc_b1, idesc, 1);
}

__device__ void compute_D_local(char* s_O0, char* s_dO0, char* s_O1, char* s_dO1, float* s_DT, float* global_s_DT, int q_start_local, int tid) {
    float local_D[2] = {0, 0};
    for(int k=0; k<64; ++k) {
        local_D[0] += __bfloat162float(read_swizzled(s_O0, tid, k)) * __bfloat162float(read_swizzled(s_dO0, tid, k));
        local_D[1] += __bfloat162float(read_swizzled(s_O1, tid, k)) * __bfloat162float(read_swizzled(s_dO1, tid, k));
    }
    
    int row_idx = (q_start_local / 64) * 64 + (tmem_base_cta0 % 128);
    s_DT[tid] = (tmem_base_cta0 == 0) ? local_D[1] : local_D[0];
    
    if (tmem_base_cta0 == 0) {
        atomicAdd(global_s_DT + ((q_start_local / 64) * 64 + tid), local_D[1]);
    } else {
        atomicAdd(global_s_DT + ((q_start_local / 64) * 64 + tid), local_D[0]);
    }
}

__device__ void apply_softmax_local(char* s_PT, char* s_S_T, const float* L_data, int bh, int q_start_local, int kv_start, float p_scale, float lse_scale, int S_val) {
    int tid = threadIdx.x;
    for (int col = 0; col < 64; col++) {
        float s = __bfloat162float(read_swizzled(s_S_T, tid, col));
        float p = 0;
        const float* L_bh = L_data + bh * S_val + q_start_local + tid;
        if (kv_start + col <= q_start_local + tid && q_start_local + tid < S_val && kv_start + col < S_val) {
            float lse = L_bh[0];
            p = fast_exp2f_fn(s * p_scale - lse * lse_scale);
        }
        write_swizzled(s_PT, tid, col, p);
    }
}

__device__ void compute_dS_local(char* s_dST, char* s_PT, char* s_dPT, float* s_DT, float* global_s_DT, int q_start_local, int kv_start) {
    int tid = threadIdx.x;
    for (int col = 0; col < 64; col++) {
        float p = __bfloat162float(read_swizzled(s_PT, tid, col));
        float dp = __bfloat162float(read_swizzled(s_dPT, tid, col));
        
        int row_idx = (q_start_local / 64) * 64 + (tmem_base_cta0 % 128);
        float s_DT_final = s_DT[tid] + global_s_DT[(q_start_local / 64) * 64 + tid];
        
        if (kv_start + col <= q_start_local + tid && q_start_local + tid < S_val && kv_start + col < S_val) {
            float ds = p * (dp - s_DT_final);
            write_swizzled(s_dST, tid, col, ds);
        } else {
            write_swizzled(s_dST, tid, col, 0.0f);
        }
    }
}

__device__ void store_gemm_atomic_add(uint32_t tmem_ptr, __nv_bfloat16* global_D, int64_t row_base, int S_val) {
    int tid = threadIdx.x;
    for (int col = 0; col < 64; col += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(tmem_ptr + (tid * 64) + col, &r0, &r1, &r2, &r3);
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float f0 = __uint_as_float(r0);
        float f1 = __uint_as_float(r1);
        float f2 = __uint_as_float(r2);
        float f3 = __uint_as_float(r3);
        
        int g_row = row_base + tid;
        if (g_row < S_val) {
            atomicAdd(&global_D[(uint64_t)g_row * 128 + col + 0], __float2bfloat16(f0));
            atomicAdd(&global_D[(uint64_t)g_row * 128 + col + 1], __float2bfloat16(f1));
            atomicAdd(&global_D[(uint64_t)g_row * 128 + col + 2], __float2bfloat16(f2));
            atomicAdd(&global_D[(uint64_t)g_row * 128 + col + 3], __float2bfloat16(f3));
        }
    }
}

__device__ __forceinline__ void tmem_epilogue_coalesced_4w_fn(
    __nv_bfloat16* D, uint32_t tmem_base_ptr,
    uint32_t M, uint32_t N, uint32_t m_block, uint32_t n_block,
    uint32_t BM, uint32_t BN) {
    
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = (BM + 3) / 4;
    
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        if (row >= BM) continue;
        
        uint32_t global_row = m_block + row;
        if (global_row >= M) continue;
        
        uint32_t col_start = lane_id * 4;
        if (n_block + col_start >= N) continue;
        
        uint32_t tmem_col = (row * 64) + col_start;
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(tmem_base_ptr + tmem_col, &r0, &r1, &r2, &r3);
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float f0 = __uint_as_float(r0);
        float f1 = __uint_as_float(r1);
        float f2 = __uint_as_float(r2);
        float f3 = __uint_as_float(r3);
        
        __nv_bfloat16 bf0 = __float2bfloat16(f0);
        __nv_bfloat16 bf1 = __float2bfloat16(f1);
        __nv_bfloat16 bf2 = __float2bfloat16(f2);
        __nv_bfloat16 bf3 = __float2bfloat16(f3);
        
        uint32_t val0 = *(uint32_t*)&bf0;
        uint32_t val1 = *(uint32_t*)&bf1;
        uint32_t val2 = *(uint32_t*)&bf2;
        uint32_t val3 = *(uint32_t*)&bf3;
        
        uint4 data;
        data.x = val0;
        data.y = val1;
        data.z = val2;
        data.w = val3;
        
        uint32_t g_addr = (uint32_t)(&D[(uint64_t)global_row * N + n_block + col_start]);
        *(uint4*)g_addr = data;
    }
}

struct SharedStorage {
    char s_K0[4096 * 2];
    char s_K1[4096 * 2];
    char s_K2[4096 * 2];
    char s_K3[4096 * 2];
    char s_V0[4096 * 2];
    char s_V1[4096 * 2];
    char s_V2[4096 * 2];
    char s_V3[4096 * 2];
    
    char s_Q0[4096 * 2];
    char s_Q1[4096 * 2];
    char s_Q2[4096 * 2];
    char s_Q3[4096 * 2];
    char s_Q0_T[4096 * 2];
    char s_Q1_T[4096 * 2];
    char s_Q2_T[4096 * 2];
    char s_Q3_T[4096 * 2];
    
    char s_dO0[4096 * 2];
    char s_dO1[4096 * 2];
    char s_dO2[4096 * 2];
    char s_dO3[4096 * 2];
    char s_O0[4096 * 2];
    char s_O1[4096 * 2];
    char s_O2[4096 * 2];
    char s_O3[4096 * 2];
    
    char s_PT[4096 * 2];
    char s_dPT[4096 * 2];
    char s_dST[4096 * 2];
    float s_DT[128];
    float global_s_DT[128];
    uint64_t bar[1];
};

__global__ void bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* __restrict__ L_data,
    __nv_bfloat16* __restrict__ dQ_bh,
    __nv_bfloat16* __restrict__ dK_bh,
    __nv_bfloat16* __restrict__ dV_bh,
    int64_t S_val, float scale)
{
    int bh = blockIdx.y;
    int kv_start = blockIdx.x * 128;
    uint32_t cluster_rank = cluster_rank_fn();
    
    extern __shared__ char smem_buf[];
    SharedStorage* smem = (SharedStorage*)smem_buf;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(smem->bar, 1);
    }
    __syncthreads();

    uint32_t tmem_base_cta0;
    uint32_t tmem_base_cta1 = 0;
    if (cluster_rank == 0) {
        tmem_alloc_fn(&tmem_base_cta0, 256);
    } else {
        tmem_base_cta0 = 0; // Block 1 logically continues from Block 0's allocation base
    }
    __syncthreads();
    
    uint32_t tmem_S_ptr = tmem_base_cta0;
    uint32_t tmem_P_ptr = tmem_base_cta0 + 8192;
    uint32_t tmem_dP_ptr = tmem_base_cta0 + 16384;
    uint32_t tmem_dS_ptr = tmem_base_cta0 + 24576;
    uint32_t tmem_dV_ptr = tmem_P_ptr;
    uint32_t tmem_dK_ptr = tmem_dS_ptr;
    uint32_t tmem_dQ_ptr = tmem_dP_ptr;
    
    uint32_t phase = 0;

    if (kv_start < S_val) {
        mbarrier_arrive_and_expect_tx_fn(smem->bar, 8192 * 8);
        tma_load_2d_fn(&tma_K, smem->bar, smem->s_K0, 0, bh * S_val + kv_start);
        tma_load_2d_fn(&tma_K, smem->bar, smem->s_K1, 64, bh * S_val + kv_start);
        tma_load_2d_fn(&tma_K, smem->bar, smem->s_K2, 0, bh * S_val + kv_start + 64);
        tma_load_2d_fn(&tma_K, smem->bar, smem->s_K3, 64, bh * S_val + kv_start + 64);

        tma_load_2d_fn(&tma_V, smem->bar, smem->s_V0, 0, bh * S_val + kv_start);
        tma_load_2d_fn(&tma_V, smem->bar, smem->s_V1, 64, bh * S_val + kv_start);
        tma_load_2d_fn(&tma_V, smem->bar, smem->s_V2, 0, bh * S_val + kv_start + 64);
        tma_load_2d_fn(&tma_V, smem->bar, smem->s_V3, 64, bh * S_val + kv_start + 64);
        mbarrier_wait_fn(smem->bar, phase);
        fence_proxy_async_fn();
        phase ^= 1;
    }
    __syncthreads();

    uint32_t idesc_S = make_instr_desc_fn(128, 64);
    uint32_t idesc_dP = make_instr_desc_fn(128, 64);
    uint32_t idesc_dV = make_instr_desc_fn(128, 64);
    uint32_t idesc_dK = make_instr_desc_fn(128, 64);
    uint32_t idesc_dQ = make_instr_desc_fn(128, 64);

    // ==== Phase 1: Accumulate dV and calculate preliminary context ====
    for (int q_start = 0; q_start <= kv_start; q_start += 64) {
        int q_start_local = q_start + cluster_rank * 64;
        
        if (threadIdx.x < 128) {
            smem->global_s_DT[threadIdx.x] = 0;
        }
        __syncthreads();

        if (q_start_local < S_val) {
            mbarrier_arrive_and_expect_tx_fn(smem->bar, 8192 * 12);
            tma_load_2d_fn(&tma_Q, smem->bar, smem->s_Q0, 0, bh * S_val + q_start_local);
            tma_load_2d_fn(&tma_Q, smem->bar, smem->s_Q1, 64, bh * S_val + q_start_local);
            tma_load_2d_fn(&tma_Q, smem->bar, smem->s_Q2, 0, bh * S_val + q_start_local + 64);
            tma_load_2d_fn(&tma_Q, smem->bar, smem->s_Q3, 64, bh * S_val + q_start_local + 64);

            tma_load_2d_fn(&tma_dO, smem->bar, smem->s_dO0, 0, bh * S_val + q_start_local);
            tma_load_2d_fn(&tma_dO, smem->bar, smem->s_dO1, 64, bh * S_val + q_start_local);
            tma_load_2d_fn(&tma_dO, smem->bar, smem->s_dO2, 0, bh * S_val + q_start_local + 64);
            tma_load_2d_fn(&tma_dO, smem->bar, smem->s_dO3, 64, bh * S_val + q_start_local + 64);

            tma_load_2d_fn(&tma_O, smem->bar, smem->s_O0, 0, bh * S_val + q_start_local);
            tma_load_2d_fn(&tma_O, smem->bar, smem->s_O1, 64, bh * S_val + q_start_local);
            tma_load_2d_fn(&tma_O, smem->bar, smem->s_O2, 0, bh * S_val + q_start_local + 64);
            tma_load_2d_fn(&tma_O, smem->bar, smem->s_O3, 64, bh * S_val + q_start_local + 64);
            
            mbarrier_wait_fn(smem->bar, phase);
            fence_proxy_async_fn();
            phase ^= 1;
        }
        __syncthreads();

        if (q_start_local < S_val) {
            compute_D_local(smem->s_O0, smem->s_dO0, smem->s_O1, smem->s_dO1, smem->s_DT, smem->global_s_DT, q_start_local, threadIdx.x);
        }
        __syncthreads();

        if (q_start_local < S_val) {
            transpose_64x64_swizzled(smem->s_Q0, smem->s_Q0_T);
            transpose_64x64_swizzled(smem->s_Q1, smem->s_Q1_T);
            transpose_64x64_swizzled(smem->s_Q2, smem->s_Q2_T);
            transpose_64x64_swizzled(smem->s_Q3, smem->s_Q3_T);
        }
        __syncthreads();

        if (q_start_local < S_val) {
            gemm_tmem(tmem_S_ptr, smem->s_K0, smem->s_K1, smem->s_Q0_T, smem->s_Q1_T, idesc_S, 0);
            tmem_wait_fn(smem->bar, phase);
            phase ^= 1;
        }
        __syncthreads();

        if (q_start_local < S_val) {
            apply_softmax_local(smem->s_PT, smem->s_Q0_T, L_data, bh, q_start_local, kv_start, scale, scale, S_val);
        }
        __syncthreads();

        if (q_start_local < S_val) {
            gemm_tmem(tmem_dP_ptr, smem->s_dO0, smem->s_dO1, smem->s_V0, s_V1, idesc_dP, 0);
            tmem_wait_fn(smem->bar, phase);
            phase ^= 1;
        }
        __syncthreads();

        if (q_start_local < S_val) {
            compute_dS_local(smem->s_dST, smem->s_PT, smem->s_dPT, smem->s_DT, smem->global_s_DT, q_start_local, kv_start);
        }
        __syncthreads();

        if (q_start_local < S_val) {
            gemm_tmem(tmem_dV_ptr, smem->s_PT, smem->s_PT, smem->s_dO0, smem->s_dO1, idesc_dP, 1);
            tmem_wait_fn(smem->bar, phase);
            phase ^= 1;
        }
        __syncthreads();
    }

    // ==== Phase 2: Accumulate dK and compute dQ ====
    for (int q_start = kv_start; q_start < S_val; q_start += 64) {
        int q_start_local = q_start + cluster_rank * 64;
        
        if (threadIdx.x < 128) {
            smem->global_s_DT[threadIdx.x] = 0;
        }
        __syncthreads();

        if (q_start_local < S_val) {
            mbarrier_arrive_and_expect_tx_fn(smem->bar, 8192 * 12);
            tma_load_2d_fn(&tma_Q, smem->bar, smem->s_Q0, 0, bh * S_val + q_start_local);
            tma_load_2d_fn(&tma_Q, smem->bar, smem->s_Q1, 64, bh * S_val + q_start_local);
            tma_load_2d_fn(&tma_Q, smem->bar, smem->s_Q2, 0, bh * S_val + q_start_local + 64);
            tma_load_2d_fn(&tma_Q, smem->bar, smem->s_Q3, 64, bh * S_val + q_start_local + 64);

            tma_load_2d_fn(&tma_dO, smem->bar, smem->s_dO0, 0, bh * S_val + q_start_local);
            tma_load_2d_fn(&tma_dO, smem->bar, smem->s_dO1, 64, bh * S_val + q_start_local);
            tma_load_2d_fn(&tma_dO, smem->bar, smem->s_dO2, 0, bh * S_val + q_start_local + 64);
            tma_load_2d_fn(&tma_dO, smem->bar, smem->s_dO3, 64, bh * S_val + q_start_local + 64);

            tma_load_2d_fn(&tma_O, smem->bar, smem->s_O0, 0, bh * S_val + q_start_local);
            tma_load_2d_fn(&tma_O, smem->bar, smem->s_O1, 64, bh * S_val + q_start_local);
            tma_load_2d_fn(&tma_O, smem->bar, smem->s_O2, 0, bh * S_val + q_start_local + 64);
            tma_load_2d_fn(&tma_O, smem->bar, smem->s_O3, 64, bh * S_val + q_start_local + 64);
            
            mbarrier_wait_fn(smem->bar, phase);
            fence_proxy_async_fn();
            phase ^= 1;
        }
        __syncthreads();

        if (q_start_local < S_val) {
            compute_D_local(smem->s_O0, smem->s_dO0, smem->s_O1, smem->s_dO1, smem->s_DT, smem->global_s_DT, q_start_local, threadIdx.x);
        }
        __syncthreads();

        if (q_start_local < S_val) {
            transpose_64x64_swizzled(smem->s_Q0, smem->s_Q0_T);
            transpose_64x64_swizzled(smem->s_Q1, smem->s_Q1_T);
            transpose_64x64_swizzled(smem->s_Q2, smem->s_Q2_T);
            transpose_64x64_swizzled(smem->s_Q3, smem->s_Q3_T);
        }
        __syncthreads();

        if (q_start_local < S_val) {
            gemm_tmem(tmem_S_ptr, smem->s_K0, smem->s_K1, smem->s_Q0_T, smem->s_Q1_T, idesc_S, 0);
            tmem_wait_fn(smem->bar, phase);
            phase ^= 1;
        }
        __syncthreads();

        if (q_start_local < S_val) {
            apply_softmax_local(smem->s_PT, smem->s_Q0_T, L_data, bh, q_start_local, kv_start, scale, scale, S_val);
        }
        __syncthreads();

        if (q_start_local < S_val) {
            gemm_tmem(tmem_dP_ptr, smem->s_dO0, smem->s_dO1, smem->s_V0, s_V1, idesc_dP, 0);
            tmem_wait_fn(smem->bar, phase);
            phase ^= 1;
        }
        __syncthreads();

        if (q_start_local < S_val) {
            compute_dS_local(smem->s_dST, smem->s_PT, smem->s_dPT, smem->s_DT, smem->global_s_DT, q_start_local, kv_start);
        }
        __syncthreads();

        if (q_start_local < S_val) {
            gemm_tmem(tmem_dK_ptr, smem->s_dST, smem->s_dST, smem->s_Q0, smem->s_Q1, idesc_dK, 1);
            tmem_wait_fn(smem->bar, phase);
            phase ^= 1;
        }
        __syncthreads();
        
        if (q_start_local < S_val) {
            gemm_tmem(tmem_dQ_ptr, smem->s_dST, smem->s_dST, smem->s_K0, smem->s_K1, idesc_dQ, 1);
            tmem_wait_fn(smem->bar, phase);
            phase ^= 1;
            
            store_gemm_atomic_add(tmem_dQ_ptr, dQ_bh + bh * S_val * 128, q_start_local, S_val);
        }
        __syncthreads();
    }
    
    tmem_epilogue_coalesced_4w_fn(dV_bh + bh * S_val * 128, tmem_dV_ptr, S_val, 128, kv_start, 0, 128, 128);
    tmem_epilogue_coalesced_4w_fn(dK_bh + bh * S_val * 128, tmem_dK_ptr, S_val, 128, kv_start, 0, 128, 128);

    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_base_cta0, 256);
    }
}

CUresult create_tma_2d_descriptor_2B(
    CUtensorMap* d, void* globalAddress, 
    uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, 
    uint32_t smem_inner_dim, uint32_t smem_outer_dim, 
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) 
{
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
    int64_t S_val = Q.size(2);
    int64_t d = Q.size(3);
    
    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_data = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_data = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_data = static_cast<const float*>(L.data_ptr());
    
    __nv_bfloat16* dQ_data = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_data = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_data = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    float scale = 1.0f / sqrtf((float)d);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUDA_CHECK(cudaMemsetAsync(dQ_data, 0, B * H * S_val * d * sizeof(__nv_bfloat16), stream));
    CUDA_CHECK(cudaMemsetAsync(dK_data, 0, B * H * S_val * d * sizeof(__nv_bfloat16), stream));
    CUDA_CHECK(cudaMemsetAsync(dV_data, 0, B * H * S_val * d * sizeof(__nv_bfloat16), stream));
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O, tma_dO;
    create_tma_2d_descriptor_2B(&tma_Q, (void*)Q_data, d, B * H * S_val, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_K, (void*)K_data, d, B * H * S_val, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_V, (void*)V_data, d, B * H * S_val, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_O, (void*)O_data, d, B * H * S_val, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_dO, (void*)dO_data, d, B * H * S_val, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    int64_t threads = 128;
    dim3 grid((S_val + 127) / 128, B * H);
    
    CUDA_CHECK(cudaFuncSetAttribute(bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedStorage)));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = dim3(threads);
    config.dynamicSmemBytes = sizeof(SharedStorage);
    config.stream = stream;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(&config, bwd_kernel, 
        tma_Q, tma_K, tma_V, tma_O, tma_dO, L_data, dQ_data, dK_data, dV_data, S_val, scale));
        
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_fa4::run);

} // namespace tvm_ffi_fa4