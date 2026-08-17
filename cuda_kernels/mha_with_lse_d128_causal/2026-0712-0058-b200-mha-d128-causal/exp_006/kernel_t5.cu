#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <stdio.h>
#include <cmath>
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

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

__device__ __forceinline__ void tmem_alloc_fn_cta1(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn_cta1(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void named_barrier_sync_fn(int bar_id, int count) {
    asm volatile("barrier.sync.aligned %0, %1;" :: "r"(bar_id), "r"(count));
}

__device__ __forceinline__ uint64_t make_smem_desc(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_mn_major(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_k_major(uint32_t M, uint32_t N) {
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

__device__ __forceinline__ uint32_t make_instr_desc_b_mn_major(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (0u << 15);   
    d |= (1u << 16);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ void umma_f16_cta1_fn(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.shared::cta.b64"
        " [%0];"
        :: "r"(a));
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ uint32_t pack_bf16_fn(uint32_t fp32_a, uint32_t fp32_b) {
    __nv_bfloat16 a = __float2bfloat16(__uint_as_float(fp32_a));
    __nv_bfloat16 b = __float2bfloat16(__uint_as_float(fp32_b));
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&a)),
          "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}

__device__ __forceinline__ int swizzle_128B_offset(int r, int c) {
    return r * 64 + (((r % 8) ^ (c / 8)) * 8 + (c % 8));
}

CUresult create_tma_4d_descriptor(CUtensorMap* d, void* globalAddress,
                                  uint64_t dim0, uint64_t dim1, uint64_t dim2, uint64_t dim3,
                                  uint32_t box0, uint32_t box1,
                                  CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[4] = {dim0, dim1, dim2, dim3};
    cuuint64_t globalStrides[3] = {dim0 * 2, dim0 * dim1 * 2, dim0 * dim1 * dim2 * 2};
    cuuint32_t boxDim[4] = {box0, box1, 1, 1};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
        globalAddress, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

__device__ __forceinline__ void tmem_epilogue_coalesced_4w_fn(
    __nv_bfloat16* D, __nv_bfloat16* smem_out,
    uint32_t M, uint32_t N, uint32_t m_block, uint32_t n_block,
    uint32_t BM, uint32_t BN) {
    __syncthreads(); 
    
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    
    uint32_t num_steps = (BM + 3) / 4;
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        if (row >= BM) continue;
        
        uint32_t global_row = m_block + row;
        
        // Process Left half (Cols 0..63)
        uint32_t col_start_L = lane_id * 4;
        uint32_t global_col_L = n_block + col_start_L;
        if (global_row < M && global_col_L + 3 < N) {
            uint2 data_L = *reinterpret_cast<uint2*>(&smem_out[row * BN + col_start_L]);
            *reinterpret_cast<uint2*>(D + (uint64_t)global_row * N + global_col_L) = data_L;
        }
        
        // Process Right half (Cols 64..127)
        uint32_t col_start_R = lane_id * 4;
        uint32_t global_col_R = n_block + 64 + col_start_R;
        if (global_row < M && global_col_R + 3 < N) {
            uint2 data_R = *reinterpret_cast<uint2*>(&smem_out[row * BN + col_start_R]);
            *reinterpret_cast<uint2*>(D + (uint64_t)global_row * N + global_col_R) = data_R;
        }
    }
}

__global__ void run_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O_ptr, float* LSE_ptr,
    int S, int num_heads, int batch_size) 
{
    extern __shared__ __align__(128) uint8_t smem_pool[];
    uintptr_t pool_addr = (uintptr_t)smem_pool;
    uint8_t* cur = smem_pool + (1024 - (pool_addr % 1024)) % 1024;
    
    uint64_t* mbar = (uint64_t*)cur;
    cur += 32; 
    cur = (uint8_t*)((((uintptr_t)cur) + 1023) & ~1023);
    
    __nv_bfloat16* s_Q0 = (__nv_bfloat16*)cur; cur += 8192;
    __nv_bfloat16* s_Q1 = (__nv_bfloat16*)cur; cur += 8192;
    __nv_bfloat16* s_Q2 = (__nv_bfloat16*)cur; cur += 8192;
    __nv_bfloat16* s_Q3 = (__nv_bfloat16*)cur; cur += 8192;
    __nv_bfloat16* s_K0 = (__nv_bfloat16*)cur; cur += 8192;
    __nv_bfloat16* s_K1 = (__nv_bfloat16*)cur; cur += 8192;
    __nv_bfloat16* s_V0 = (__nv_bfloat16*)cur; cur += 8192;
    __nv_bfloat16* s_V1 = (__nv_bfloat16*)cur; cur += 8192;
    __nv_bfloat16* s_P  = (__nv_bfloat16*)cur; cur += 8192;
    __nv_bfloat16* smem_out_L = (__nv_bfloat16*)cur; cur += 8192;
    __nv_bfloat16* smem_out_R = (__nv_bfloat16*)cur; cur += 8192;
    
    int head_idx = blockIdx.y;
    int batch_idx = blockIdx.z;
    int q_base_block = blockIdx.x * 128;
    int flattened_head = head_idx + batch_idx * num_heads;
    int global_q_base = (flattened_head * S);
    
    uint32_t tmem_S, tmem_O_left, tmem_O_right;
    if (threadIdx.x == 0) {
        tmem_alloc_fn_cta1(&tmem_S, 64);
        tmem_alloc_fn_cta1(&tmem_O_left, 64);
        tmem_alloc_fn_cta1(&tmem_O_right, 64);
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&mbar[0], 1);
        init_smem_barrier_fn(&mbar[1], 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    int wg_id = (threadIdx.x / 128) % 2;
    int q_base_wg = q_base_block + wg_id * 64;
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar[0], 16384);
        tma_load_4d_fn(&tma_Q, &mbar[0], s_Q0, 0, q_base_block, head_idx, batch_idx);
        tma_load_4d_fn(&tma_Q, &mbar[0], s_Q1, 64, q_base_block, head_idx, batch_idx);
        tma_load_4d_fn(&tma_Q, &mbar[0], s_Q2, 0, q_base_block + 64, head_idx, batch_idx);
        tma_load_4d_fn(&tma_Q, &mbar[0], s_Q3, 64, q_base_block + 64, head_idx, batch_idx);
        
        mbarrier_arrive_and_expect_tx_fn(&mbar[1], 32768);
        tma_load_4d_fn(&tma_K, &mbar[1], s_K0, 0, 0, head_idx, batch_idx);
        tma_load_4d_fn(&tma_K, &mbar[1], s_K1, 64, 0, head_idx, batch_idx);
        tma_load_4d_fn(&tma_V, &mbar[1], s_V0, 0, 0, head_idx, batch_idx);
        tma_load_4d_fn(&tma_V, &mbar[1], s_V1, 64, 0, head_idx, batch_idx);
    }
    
    mbarrier_wait_fn(&mbar[0], 0);
    
    float O_acc_left[64];
    float O_acc_right[64];
    #pragma unroll
    for(int i = 0; i < 64; i++) {
        O_acc_left[i] = 0.0f;
        O_acc_right[i] = 0.0f;
    }
    
    float row_max_prev = -INFINITY;
    float row_sum_prev = 0.0f;
    
    int next_barrier_phase = 0;
    float scale_factor = 1.0f / sqrtf(128.0f);
    
    for (int k_base = 0; k_base <= q_base_wg; k_base += 64) {
        int next_k_base = k_base + 64;
        
        if (next_k_base <= q_base_wg && next_k_base < S) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&mbar[1], 32768);
                tma_load_4d_fn(&tma_K, &mbar[1], s_K0, 0, next_k_base, head_idx, batch_idx);
                tma_load_4d_fn(&tma_K, &mbar[1], s_K1, 64, next_k_base, head_idx, batch_idx);
                tma_load_4d_fn(&tma_V, &mbar[1], s_V0, 0, next_k_base, head_idx, batch_idx);
                tma_load_4d_fn(&tma_V, &mbar[1], s_V1, 64, next_k_base, head_idx, batch_idx);
            }
        }
        
        mbarrier_wait_fn(&mbar[1], next_barrier_phase & 1);
        
        uint32_t r_base = (threadIdx.x % 128) / 32 * 16;
        uint32_t addr_S0 = tmem_S + (0 + (r_base + 0) * 64);
        uint32_t addr_S1 = tmem_S + (4 + (r_base + 0) * 64);
        uint32_t addr_S2 = tmem_S + (8 + (r_base + 0) * 64);
        uint32_t addr_S3 = tmem_S + (12 + (r_base + 0) * 64);

        uint32_t S0[4], S1[4], S2[4], S3[4];
        
        if (wg_id == 0) {
            asm volatile(
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(S0[0]), "=r"(S0[1]), "=r"(S0[2]), "=r"(S0[3]) : "r"(addr_S0));
            asm volatile(
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(S1[0]), "=r"(S1[1]), "=r"(S1[2]), "=r"(S1[3]) : "r"(addr_S1));
            asm volatile(
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(S2[0]), "=r"(S2[1]), "=r"(S2[2]), "=r"(S2[3]) : "r"(addr_S2));
            asm volatile(
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(S3[0]), "=r"(S3[1]), "=r"(S3[2]), "=r"(S3[3]) : "r"(addr_S3));
        } else {
            asm volatile(
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(S0[0]), "=r"(S0[1]), "=r"(S0[2]), "=r"(S0[3]) : "r"(addr_S0 + 1024));
            asm volatile(
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(S1[0]), "=r"(S1[1]), "=r"(S1[2]), "=r"(S1[3]) : "r"(addr_S1 + 1024));
            asm volatile(
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(S2[0]), "=r"(S2[1]), "=r"(S2[2]), "=r"(S2[3]) : "r"(addr_S2 + 1024));
            asm volatile(
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(S3[0]), "=r"(S3[1]), "=r"(S3[2]), "=r"(S3[3]) : "r"(addr_S3 + 1024));
        }
        
        float S_val_f[64];
        for(int i = 0; i < 4; i++) {
            S_val_f[r_base + 0 + i * 16] = __uint_as_float(S0[i]);
            S_val_f[r_base + 1 + i * 16] = __uint_as_float(S1[i]);
            S_val_f[r_base + 2 + i * 16] = __uint_as_float(S2[i]);
            S_val_f[r_base + 3 + i * 16] = __uint_as_float(S3[i]);
        }
        
        int my_c[4] = {0, 16, 32, 48};
        int my_r[4] = {0, 1, 2, 3};
        float current_row_max[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
        
        for(int i = 0; i < 4; i++) {
            for(int j = 0; j < 4; j++) {
                int col = my_c[j] + (threadIdx.x % 32) * 2;
                int row = r_base + my_r[i];
                int global_k_idx = k_base + col;
                int global_q_idx = q_base_wg + row;
                if (global_q_idx < global_k_idx || global_k_idx >= S || global_q_idx >= S) {
                    S_val_f[row + j * 16] = -INFINITY;
                } else {
                    S_val_f[row + j * 16] *= scale_factor;
                }
                current_row_max[i] = fmaxf(current_row_max[i], S_val_f[row + j * 16]);
            }
        }
        
        float final_max[4];
        for(int i = 0; i < 4; i++) {
            float my_max = current_row_max[i];
            for (int offset = 1; offset < 32; offset *= 2) {
                my_max = fmaxf(my_max, __shfl_xor_sync(0xFFFFFFFF, my_max, offset));
            }
            final_max[i] = my_max;
        }
        
        float scale_l[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        float scale_r[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        
        for(int i = 0; i < 4; i++) {
            if (final_max[i] > row_max_prev) {
                float s = fast_exp2f_fn((row_max_prev - final_max[i]) * 1.4426950f);
                scale_l[i] = s;
                scale_r[i] = s;
                row_sum_prev *= s;
            }
        }
        
        for(int i = 0; i < 4; i++) {
            for(int j = 0; j < 4; j++) {
                O_acc_left[r_base + my_r[i] + j * 16] *= scale_l[i];
                O_acc_right[r_base + my_r[i] + j * 16] *= scale_r[i];
            }
        }
        
        float current_row_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for(int i = 0; i < 4; i++) {
            for(int j = 0; j < 4; j++) {
                float p = 0.0f;
                if (S_val_f[r_base + my_r[i] + j * 16] != -INFINITY) {
                    p = fast_exp2f_fn((S_val_f[r_base + my_r[i] + j * 16] - final_max[i]) * 1.4426950f);
                }
                current_row_sum[i] += p;
                S_val_f[r_base + my_r[i] + j * 16] = p;
            }
        }
        
        for(int i = 0; i < 4; i++) {
            float my_sum = current_row_sum[i];
            for (int offset = 1; offset < 32; offset *= 2) {
                my_sum += __shfl_xor_sync(0xFFFFFFFF, my_sum, offset);
            }
            current_row_sum[i] = my_sum;
            row_sum_prev += my_sum;
            row_max_prev = final_max[i];
        }
        
        for(int r = 0; r < 64; r++) {
            int c = (threadIdx.x % 32) * 2;
            int my_r_idx = r / 8; 
            float p0 = S_val_f[r];
            float p1 = (c + 1 < 64) ? S_val_f[r] : 0.0f; 
            
            // Force NaN values to 0.0f to avoid polluting outputs with NaN
            if (p0 != p0) p0 = 0.0f;
            if (p1 != p1) p1 = 0.0f;
            
            uint32_t packed = pack_bf16_fn(__float_as_uint(p0), __float_as_uint(p1));
            int sc = swizzle_128B_offset(r, c);
            *(uint32_t*)&s_P[sc] = packed;
        }
        
        __syncthreads();
        fence_proxy_async_fn();
        
        if (wg_id == 0) {
            for (int p = 0; p < 64; p += 16) {
                uint64_t desc_a = make_smem_desc(s_P + p, 1, 1024); 
                uint64_t desc_b = make_smem_desc_mn_major(s_V0 + p * 64, 8192, 1024);
                uint32_t idesc = make_instr_desc_b_mn_major(64, 64);
                umma_f16_cta1_fn(tmem_O_left, desc_a, desc_b, idesc, p == 0 ? 0 : 1);
            }
            for (int p = 0; p < 64; p += 16) {
                uint64_t desc_a = make_smem_desc(s_P + p, 1, 1024); 
                uint64_t desc_b = make_smem_desc_mn_major(s_V1 + p * 64, 8192, 1024);
                uint32_t idesc = make_instr_desc_b_mn_major(64, 64);
                umma_f16_cta1_fn(tmem_O_right, desc_a, desc_b, idesc, p == 0 ? 0 : 1);
            }
        } else {
            for (int p = 0; p < 64; p += 16) {
                uint64_t desc_a = make_smem_desc(s_P + p, 1, 1024); 
                uint64_t desc_b = make_smem_desc_mn_major(s_V0 + p * 64, 8192, 1024);
                uint32_t idesc = make_instr_desc_b_mn_major(64, 64);
                umma_f16_cta1_fn(tmem_O_left + 1024, desc_a, desc_b, idesc, p == 0 ? 0 : 1);
            }
            for (int p = 0; p < 64; p += 16) {
                uint64_t desc_a = make_smem_desc(s_P + p, 1, 1024); 
                uint64_t desc_b = make_smem_desc_mn_major(s_V1 + p * 64, 8192, 1024);
                uint32_t idesc = make_instr_desc_b_mn_major(64, 64);
                umma_f16_cta1_fn(tmem_O_right + 1024, desc_a, desc_b, idesc, p == 0 ? 0 : 1);
            }
        }
        umma_commit_fn(&mbar[0]);
        mbarrier_wait_fn(&mbar[0], next_barrier_phase & 1);
        
        uint32_t addr_O0 = tmem_O_left + (0 + (r_base + 0) * 64);
        uint32_t addr_O1 = tmem_O_left + (4 + (r_base + 0) * 64);
        uint32_t addr_O2 = tmem_O_left + (8 + (r_base + 0) * 64);
        uint32_t addr_O3 = tmem_O_left + (12 + (r_base + 0) * 64);

        uint32_t O0[4], O1[4], O2[4], O3[4];
        
        if (wg_id == 0) {
            asm volatile(
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(O0[0]), "=r"(O0[1]), "=r"(O0[2]), "=r"(O0[3]) : "r"(addr_O0));
            asm volatile(
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(O1[0]), "=r"(O1[1]), "=r"(O1[2]), "=r"(O1[3]) : "r"(addr_O1));
            asm volatile(
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(O2[0]), "=r"(O2[1]), "=r"(O2[2]), "=r"(O2[3]) : "r"(addr_O2));
            asm volatile(
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(O3[0]), "=r"(O3[1]), "=r"(O3[2]), "=r"(O3[3]) : "r"(addr_O3));
        } else {
            asm volatile(
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(O0[0]), "=r"(O0[1]), "=r"(O0[2]), "=r"(O0[3]) : "r"(addr_O0 + 1024));
            asm volatile(
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(O1[0]), "=r"(O1[1]), "=r"(O1[2]), "=r"(O1[3]) : "r"(addr_O1 + 1024));
            asm volatile(
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(O2[0]), "=r"(O2[1]), "=r"(O2[2]), "=r"(O2[3]) : "r"(addr_O2 + 1024));
            asm volatile(
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(O3[0]), "=r"(O3[1]), "=r"(O3[2]), "=r"(O3[3]) : "r"(addr_O3 + 1024));
        }
        
        for(int i = 0; i < 4; i++) {
            O_acc_left[r_base + my_r[i]] += __uint_as_float(O0[i]);
            O_acc_left[r_base + my_r[i] + 16] += __uint_as_float(O1[i]);
            O_acc_left[r_base + my_r[i] + 32] += __uint_as_float(O2[i]);
            O_acc_left[r_base + my_r[i] + 48] += __uint_as_float(O3[i]);
        }
        
        uint32_t addr_O4 = tmem_O_right + (0 + (r_base + 0) * 64);
        uint32_t addr_O5 = tmem_O_right + (4 + (r_base + 0) * 64);
        uint32_t addr_O6 = tmem_O_right + (8 + (r_base + 0) * 64);
        uint32_t addr_O7 = tmem_O_right + (12 + (r_base + 0) * 64);

        uint32_t O4[4], O5[4], O6[4], O7[4];
        
        if (wg_id == 0) {
            asm volatile(
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(O4[0]), "=r"(O4[1]), "=r"(O4[2]), "=r"(O4[3]) : "r"(addr_O4));
            asm volatile(
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(O5[0]), "=r"(O5[1]), "=r"(O5[2]), "=r"(O5[3]) : "r"(addr_O5));
            asm volatile(
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(O6[0]), "=r"(O6[1]), "=r"(O6[2]), "=r"(O6[3]) : "r"(addr_O6));
            asm volatile(
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(O7[0]), "=r"(O7[1]), "=r"(O7[2]), "=r"(O7[3]) : "r"(addr_O7));
        } else {
            asm volatile(
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(O4[0]), "=r"(O4[1]), "=r"(O4[2]), "=r"(O4[3]) : "r"(addr_O4 + 1024));
            asm volatile(
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(O5[0]), "=r"(O5[1]), "=r"(O5[2]), "=r"(O5[3]) : "r"(addr_O5 + 1024));
            asm volatile(
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(O6[0]), "=r"(O6[1]), "=r"(O6[2]), "=r"(O6[3]) : "r"(addr_O6 + 1024));
            asm volatile(
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];\n"
                "tcgen05.ld.sync.aligned.16x128b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(O7[0]), "=r"(O7[1]), "=r"(O7[2]), "=r"(O7[3]) : "r"(addr_O7 + 1024));
        }
        
        for(int i = 0; i < 4; i++) {
            O_acc_right[r_base + my_r[i]] += __uint_as_float(O4[i]);
            O_acc_right[r_base + my_r[i] + 16] += __uint_as_float(O5[i]);
            O_acc_right[r_base + my_r[i] + 32] += __uint_as_float(O6[i]);
            O_acc_right[r_base + my_r[i] + 48] += __uint_as_float(O7[i]);
        }
        
        next_barrier_phase++;
        __syncthreads();
    }
    
    if (wg_id == 0) {
        for(int r = 0; r < 64; r++) {
            int c = (threadIdx.x % 32) * 2;
            float out0 = O_acc_left[r] / row_sum_prev;
            float out1 = O_acc_right[r] / row_sum_prev;
            
            uint32_t packed = pack_bf16_fn(__float_as_uint(out0), __float_as_uint(out1));
            int sc = swizzle_128B_offset(r, c);
            *(uint32_t*)&smem_out_L[sc] = packed;
            *(uint32_t*)&smem_out_R[sc] = packed;
        }
        
        tmem_epilogue_coalesced_4w_fn(O_ptr, smem_out_L, S, 128, q_base_block, 0, 64, 64);
        tmem_epilogue_coalesced_4w_fn(O_ptr, smem_out_R, S, 128, q_base_block, 64, 64, 64);
        
        if ((threadIdx.x % 128) < 64) {
            int tid = threadIdx.x % 128;
            int global_q_idx = q_base_block + tid;
            if (global_q_idx < S) {
                LSE_ptr[global_q_base + global_q_idx] = row_max_prev + logf(row_sum_prev);
            }
        }
    } else {
        for(int r = 0; r < 64; r++) {
            int c = (threadIdx.x % 32) * 2;
            float out0 = O_acc_left[r] / row_sum_prev;
            float out1 = O_acc_right[r] / row_sum_prev;
            
            uint32_t packed = pack_bf16_fn(__float_as_uint(out0), __float_as_uint(out1));
            int sc = swizzle_128B_offset(r, c);
            *(uint32_t*)&smem_out_L[sc] = packed;
            *(uint32_t*)&smem_out_R[sc] = packed;
        }
        
        tmem_epilogue_coalesced_4w_fn(O_ptr, smem_out_L, S, 128, q_base_block + 64, 0, 64, 64);
        tmem_epilogue_coalesced_4w_fn(O_ptr, smem_out_R, S, 128, q_base_block + 64, 64, 64, 64);
        
        if ((threadIdx.x % 128) < 64) {
            int tid = threadIdx.x % 128;
            int global_q_idx = q_base_block + 64 + tid;
            if (global_q_idx < S) {
                LSE_ptr[global_q_base + global_q_idx] = row_max_prev + logf(row_sum_prev);
            }
        }
    }
    
    if (threadIdx.x == 0) {
        tmem_dealloc_fn_cta1(tmem_S, 64);
        tmem_dealloc_fn_cta1(tmem_O_left, 64);
        tmem_dealloc_fn_cta1(tmem_O_right, 64);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3); 
    
    if (S == 0) return;
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUtensorMap tma_Q, tma_K, tma_V;
    create_tma_4d_descriptor(&tma_Q, Q.data_ptr(), D, S, H, B, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B);
    create_tma_4d_descriptor(&tma_K, K.data_ptr(), D, S, H, B, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B);
    create_tma_4d_descriptor(&tma_V, V.data_ptr(), D, S, H, B, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B);
    
    uint32_t smem_size = 92160;
    dim3 grid((S + 127) / 128, H, B);
    dim3 block(256);
    
    cudaFuncSetAttribute(run_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, run_kernel, tma_Q, tma_K, tma_V, 
        reinterpret_cast<__nv_bfloat16*>(O.data_ptr()), 
        reinterpret_cast<float*>(LSE.data_ptr()), 
        static_cast<int>(S), static_cast<int>(H), static_cast<int>(B)));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);