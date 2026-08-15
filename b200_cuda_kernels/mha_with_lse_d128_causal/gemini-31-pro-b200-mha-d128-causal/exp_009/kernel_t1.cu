#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
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
        cuGetErrorName(_e, &err_str);                              \
        fprintf(stderr, "CU error %s at %s:%d\n",                 \
                err_str, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_mha {

CUresult create_tma_desc(CUtensorMap* tma, void* ptr, int64_t outer_dim) {
    cuuint64_t g_dim[2] = {128, (cuuint64_t)outer_dim};
    cuuint64_t g_strides[1] = {256}; // Stride in bytes: 128 elements * 2 bytes = 256
    cuuint32_t b_dim[2] = {64, 128};
    cuuint32_t e_strides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        tma, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ptr, 
        g_dim, g_strides, b_dim, e_strides, 
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
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

__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols) : "memory");
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols) : "memory");
}

__device__ __forceinline__ void tmem_load_8x_fn(uint32_t col,
    uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
    uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),
     "=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(col));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
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

__device__ __forceinline__ void tcgen05_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(a) : "memory");
}

__device__ __forceinline__ uint64_t my_make_smem_desc(void* smem_ptr, uint32_t lbo, uint32_t sbo, int swizzle) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    if (swizzle != 0) {
        uint32_t base_offset = (addr >> 7) & 0x7;
        d |= (uint64_t)base_offset << 49;
    }
    d |= (uint64_t)swizzle << 61;
    return d;
}


__global__ __launch_bounds__(128) void mha_fwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    float* __restrict__ LSE,
    int S, int H) 
{
    int tid = threadIdx.x;
    int b = blockIdx.y / H;
    int h = blockIdx.y % H;
    int q_idx = blockIdx.x * 128;
    if (q_idx >= S) return;
    
    // Shared memory layout accommodating Double-Buffered Async TMA Operations
    extern __shared__ __align__(128) uint8_t smem_pool[];
    uint16_t* smem_Q = (uint16_t*)smem_pool;                            // 32 KB
    uint16_t* smem_K[2];
    smem_K[0] = (uint16_t*)(smem_pool + 32768);                         // 32 KB
    smem_K[1] = (uint16_t*)(smem_pool + 65536);                         // 32 KB
    uint16_t* smem_V[2];
    smem_V[0] = (uint16_t*)(smem_pool + 98304);                         // 32 KB
    smem_V[1] = (uint16_t*)(smem_pool + 131072);                        // 32 KB
    uint16_t* smem_P = (uint16_t*)(smem_pool + 163840);                 // 32 KB
    
    // TMA Barriers mapping
    uint64_t* mbar_Q = (uint64_t*)(smem_pool + 196608);
    uint64_t* mbar_K = (uint64_t*)(smem_pool + 196616);
    uint64_t* mbar_V = (uint64_t*)(smem_pool + 196624);
    uint64_t* mbar_umma = (uint64_t*)(smem_pool + 196632);
    
    if (tid == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_K, 1);
        init_smem_barrier_fn(mbar_V, 1);
        init_smem_barrier_fn(mbar_umma, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    __shared__ uint32_t tmem_addr;
    if (tid < 32) tmem_alloc_cg1_fn(&tmem_addr, 128); // 128 cols
    __syncthreads();
    uint32_t tmem_base = tmem_addr;
    
    int start_row = (b * H + h) * S;
    
    // Q is block invariant, Prefetch K and V early
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 32768);
        tma_load_2d_fn(&tma_Q, mbar_Q, smem_Q, 0, start_row + q_idx);
        tma_load_2d_fn(&tma_Q, mbar_Q, smem_Q + 8192, 64, start_row + q_idx); // Start at +64 columns elements (128x64 block)
        
        mbarrier_arrive_and_expect_tx_fn(mbar_K, 32768);
        tma_load_2d_fn(&tma_K, mbar_K, smem_K[0], 0, start_row);
        tma_load_2d_fn(&tma_K, mbar_K, smem_K[0] + 8192, 64, start_row);
        
        mbarrier_arrive_and_expect_tx_fn(mbar_V, 32768);
        tma_load_2d_fn(&tma_V, mbar_V, smem_V[0], 0, start_row);
        tma_load_2d_fn(&tma_V, mbar_V, smem_V[0] + 8192, 64, start_row);
    }
    
    float O_acc[128];
    for (int i = 0; i < 128; i++) O_acc[i] = 0.0f;
    float m_val = -1e20f;
    float l_val = 0.0f;
    
    int phase_Q = 0;
    int phase_K = 0;
    int phase_V = 0;
    int phase_umma = 0;
    
    mbarrier_wait_fn(mbar_Q, phase_Q); // Await block-invariant Q data
    
    uint32_t idesc_QK = (1u << 4) | (1u << 7) | (1u << 10) | (0u << 15) | (0u << 16) | (16u << 17) | (8u << 24);
    uint32_t idesc_PV = (1u << 4) | (1u << 7) | (1u << 10) | (0u << 15) | (1u << 16) | (8u << 17)  | (8u << 24);

    for (int k_idx = 0; k_idx <= q_idx; k_idx += 128) {
        int buf_idx = (k_idx / 128) % 2;
        int next_k_idx = k_idx + 128;
        int next_buf_idx = (buf_idx + 1) % 2;
        
        // --- 1. Compute Q @ K^T ---
        mbarrier_wait_fn(mbar_K, phase_K);
        
        // Double-buffer prefetch K
        if (next_k_idx <= q_idx) {
            if (tid == 0) {
                mbarrier_arrive_and_expect_tx_fn(mbar_K, 32768);
                tma_load_2d_fn(&tma_K, mbar_K, smem_K[next_buf_idx], 0, start_row + next_k_idx);
                tma_load_2d_fn(&tma_K, mbar_K, smem_K[next_buf_idx] + 8192, 64, start_row + next_k_idx);
            }
        }
        
        if (tid == 0) {
            // First 64 K-dimension half
            for(int k = 0; k < 64; k += 16) {
                uint64_t desc_Q = my_make_smem_desc(smem_Q + k, 1, 1024, 3);
                uint64_t desc_K = my_make_smem_desc(smem_K[buf_idx] + k, 1, 1024, 3);
                uint32_t accum = (k > 0) ? 1 : 0;
                umma_f16_cg1_fn(tmem_base, desc_Q, desc_K, idesc_QK, accum);
            }
            // Second 64 K-dimension half
            for(int k = 0; k < 64; k += 16) {
                uint64_t desc_Q = my_make_smem_desc(smem_Q + 8192 + k, 1, 1024, 3);
                uint64_t desc_K = my_make_smem_desc(smem_K[buf_idx] + 8192 + k, 1, 1024, 3);
                umma_f16_cg1_fn(tmem_base, desc_Q, desc_K, idesc_QK, 1);
            }
            tcgen05_commit_cg1_fn(mbar_umma);
        }
        mbarrier_wait_fn(mbar_umma, phase_umma);
        phase_umma ^= 1;
        
        // --- 2. Scaling & Softmax ---
        float S_row[128];
        for(int c = 0; c < 128; c += 8) {
            tmem_load_8x_fn(tmem_base + c, 
                (uint32_t*)&S_row[c+0], (uint32_t*)&S_row[c+1], (uint32_t*)&S_row[c+2], (uint32_t*)&S_row[c+3],
                (uint32_t*)&S_row[c+4], (uint32_t*)&S_row[c+5], (uint32_t*)&S_row[c+6], (uint32_t*)&S_row[c+7]);
        }
        tmem_load_fence_fn(); // Barrier preceding math
        
        float row_max = -1e20f;
        int global_q = q_idx + tid;
        for(int c = 0; c < 128; c++) {
            int global_k = k_idx + c;
            if (global_q < S && global_k < S) {
                if (global_k > global_q) { // Causal masking
                    S_row[c] = -1e20f;
                } else {
                    S_row[c] *= 0.08838834764f; // sqrt(128) scaling factor
                    if (S_row[c] > row_max) row_max = S_row[c];
                }
            } else {
                S_row[c] = -1e20f;
            }
        }
        
        float m_new = max(m_val, row_max);
        float scale = 0.0f;
        if (m_val != -1e20f) {
            scale = fast_exp2f_fn((m_val - m_new) * 1.44269504089f);
        }
        for(int i = 0; i < 128; i++) O_acc[i] *= scale; // Fix current tracked accumulator
        
        float row_sum = 0.0f;
        for(int c = 0; c < 128; c++) {
            float p = 0.0f;
            if (S_row[c] != -1e20f) {
                p = fast_exp2f_fn((S_row[c] - m_new) * 1.44269504089f);
            }
            S_row[c] = p;
            row_sum += p;
        }
        l_val = l_val * scale + row_sum;
        m_val = m_new;
        
        // Vectorized packed P layout mapping for SMEM writing
        float4* smem_P0_f4 = (float4*)smem_P;
        float4* smem_P1_f4 = (float4*)(smem_P + 8192);
        for (int c_blk = 0; c_blk < 64; c_blk += 8) {
            int x = c_blk / 8;
            int sx = (tid % 8) ^ x;
            
            uint32_t p0 = pack_bf16_fn(*(uint32_t*)&S_row[c_blk+0], *(uint32_t*)&S_row[c_blk+1]);
            uint32_t p1 = pack_bf16_fn(*(uint32_t*)&S_row[c_blk+2], *(uint32_t*)&S_row[c_blk+3]);
            uint32_t p2 = pack_bf16_fn(*(uint32_t*)&S_row[c_blk+4], *(uint32_t*)&S_row[c_blk+5]);
            uint32_t p3 = pack_bf16_fn(*(uint32_t*)&S_row[c_blk+6], *(uint32_t*)&S_row[c_blk+7]);
            smem_P0_f4[tid * 8 + sx] = make_float4(*(float*)&p0, *(float*)&p1, *(float*)&p2, *(float*)&p3);
            
            int c1 = c_blk + 64;
            uint32_t p1_0 = pack_bf16_fn(*(uint32_t*)&S_row[c1+0], *(uint32_t*)&S_row[c1+1]);
            uint32_t p1_1 = pack_bf16_fn(*(uint32_t*)&S_row[c1+2], *(uint32_t*)&S_row[c1+3]);
            uint32_t p1_2 = pack_bf16_fn(*(uint32_t*)&S_row[c1+4], *(uint32_t*)&S_row[c1+5]);
            uint32_t p1_3 = pack_bf16_fn(*(uint32_t*)&S_row[c1+6], *(uint32_t*)&S_row[c1+7]);
            smem_P1_f4[tid * 8 + sx] = make_float4(*(float*)&p1_0, *(float*)&p1_1, *(float*)&p1_2, *(float*)&p1_3);
        }
        __syncthreads();
        
        // --- 3. Compute P @ V ---
        mbarrier_wait_fn(mbar_V, phase_V);
        
        // Double-buffer prefetch V
        if (next_k_idx <= q_idx) {
            if (tid == 0) {
                mbarrier_arrive_and_expect_tx_fn(mbar_V, 32768);
                tma_load_2d_fn(&tma_V, mbar_V, smem_V[next_buf_idx], 0, start_row + next_k_idx);
                tma_load_2d_fn(&tma_V, mbar_V, smem_V[next_buf_idx] + 8192, 64, start_row + next_k_idx);
            }
        }
        
        if (tid == 0) {
            // First 64 V dimension half (O_left)
            for(int k = 0; k < 128; k += 16) {
                uint64_t desc_P;
                if (k < 64) {
                    desc_P = my_make_smem_desc(smem_P + k, 1, 1024, 3);
                } else {
                    desc_P = my_make_smem_desc(smem_P + 8192 + (k - 64), 1, 1024, 3);
                }
                uint64_t desc_V_left = my_make_smem_desc(smem_V[buf_idx] + k * 64, 16384, 1024, 3);
                uint32_t accum = (k > 0) ? 1 : 0;
                umma_f16_cg1_fn(tmem_base, desc_P, desc_V_left, idesc_PV, accum);
            }
            // Second 64 V dimension half (O_right)  -> Stored starting col 64 in TMEM
            for(int k = 0; k < 128; k += 16) {
                uint64_t desc_P;
                if (k < 64) {
                    desc_P = my_make_smem_desc(smem_P + k, 1, 1024, 3);
                } else {
                    desc_P = my_make_smem_desc(smem_P + 8192 + (k - 64), 1, 1024, 3);
                }
                uint64_t desc_V_right = my_make_smem_desc(smem_V[buf_idx] + 8192 + k * 64, 16384, 1024, 3);
                uint32_t accum = (k > 0) ? 1 : 0;
                umma_f16_cg1_fn(tmem_base + 64, desc_P, desc_V_right, idesc_PV, accum);
            }
            tcgen05_commit_cg1_fn(mbar_umma);
        }
        mbarrier_wait_fn(mbar_umma, phase_umma);
        phase_umma ^= 1;
        
        // Transfer and accumulate full 128 result out of TMEM
        float pv_vals[128];
        for(int c = 0; c < 128; c += 8) {
            tmem_load_8x_fn(tmem_base + c, 
                (uint32_t*)&pv_vals[c+0], (uint32_t*)&pv_vals[c+1], (uint32_t*)&pv_vals[c+2], (uint32_t*)&pv_vals[c+3],
                (uint32_t*)&pv_vals[c+4], (uint32_t*)&pv_vals[c+5], (uint32_t*)&pv_vals[c+6], (uint32_t*)&pv_vals[c+7]);
        }
        tmem_load_fence_fn();
        
        for(int c = 0; c < 128; c++) O_acc[c] += pv_vals[c];
        
        phase_K ^= 1;
        phase_V ^= 1;
    } // End sequential Block matching accumulation
    
    // Finalize LSE divisions
    for(int c = 0; c < 128; c++) O_acc[c] /= l_val;
    
    // Pack optimized O writeback formatting referencing double buffer free zones
    uint16_t* smem_O_left = smem_K[0];
    uint16_t* smem_O_right = smem_V[0];
    float4* smem_O_left_f4 = (float4*)smem_O_left;
    float4* smem_O_right_f4 = (float4*)smem_O_right;
    
    for (int c_blk = 0; c_blk < 64; c_blk += 8) {
        int x = c_blk / 8;
        int sx = (tid % 8) ^ x;
        
        uint32_t o0 = pack_bf16_fn(*(uint32_t*)&O_acc[c_blk+0], *(uint32_t*)&O_acc[c_blk+1]);
        uint32_t o1 = pack_bf16_fn(*(uint32_t*)&O_acc[c_blk+2], *(uint32_t*)&O_acc[c_blk+3]);
        uint32_t o2 = pack_bf16_fn(*(uint32_t*)&O_acc[c_blk+4], *(uint32_t*)&O_acc[c_blk+5]);
        uint32_t o3 = pack_bf16_fn(*(uint32_t*)&O_acc[c_blk+6], *(uint32_t*)&O_acc[c_blk+7]);
        smem_O_left_f4[tid * 8 + sx] = make_float4(*(float*)&o0, *(float*)&o1, *(float*)&o2, *(float*)&o3);
        
        int c1 = c_blk + 64;
        uint32_t o1_0 = pack_bf16_fn(*(uint32_t*)&O_acc[c1+0], *(uint32_t*)&O_acc[c1+1]);
        uint32_t o1_1 = pack_bf16_fn(*(uint32_t*)&O_acc[c1+2], *(uint32_t*)&O_acc[c1+3]);
        uint32_t o1_2 = pack_bf16_fn(*(uint32_t*)&O_acc[c1+4], *(uint32_t*)&O_acc[c1+5]);
        uint32_t o1_3 = pack_bf16_fn(*(uint32_t*)&O_acc[c1+6], *(uint32_t*)&O_acc[c1+7]);
        smem_O_right_f4[tid * 8 + sx] = make_float4(*(float*)&o1_0, *(float*)&o1_1, *(float*)&o1_2, *(float*)&o1_3);
    }
    
    tma_store_fence_fn(); // Ensure async access respects prior generics writes
    __syncthreads();
    
    if (tid == 0) {
        tma_store_2d_fn(&tma_O, smem_O_left, 0, start_row + q_idx);
        tma_store_2d_fn(&tma_O, smem_O_right, 64, start_row + q_idx);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }
    __syncthreads();
    
    // Housekeeping TMEM
    if (tid < 32) tmem_dealloc_cg1_fn(tmem_base, 128);
    
    int global_q = q_idx + tid;
    if (global_q < S) {
        float lse = m_val + logf(l_val);
        LSE[b * H * S + h * S + global_q] = lse;
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);

    const __nv_bfloat16* q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* k_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* v_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());

    CUtensorMap tma_Q, tma_K, tma_V, tma_O;
    CU_CHECK(create_tma_desc(&tma_Q, (void*)q_ptr, B * H * S));
    CU_CHECK(create_tma_desc(&tma_K, (void*)k_ptr, B * H * S));
    CU_CHECK(create_tma_desc(&tma_V, (void*)v_ptr, B * H * S));
    CU_CHECK(create_tma_desc(&tma_O, (void*)o_ptr, B * H * S));

    int64_t threads = 128;
    dim3 blocks((S + 127) / 128, B * H);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    cudaFuncSetAttribute(mha_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 196640);

    mha_fwd_kernel<<<blocks, threads, 196640, stream>>>(tma_Q, tma_K, tma_V, tma_O, lse_ptr, S, H);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}  // namespace tvm_ffi_mha