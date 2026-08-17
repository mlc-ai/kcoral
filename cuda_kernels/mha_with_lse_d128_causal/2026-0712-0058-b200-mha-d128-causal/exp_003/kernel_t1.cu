#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdio.h>
#include <stdlib.h>
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
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace tvm_ffi_kernel {

// -------------------------------------------------------------------------
// Device Helper Functions
// -------------------------------------------------------------------------

__device__ __forceinline__ uint32_t swizzle_128B_byte(uint32_t row, uint32_t col_bytes) {
    uint32_t x = col_bytes / 16;
    uint32_t rem = col_bytes % 16;
    uint32_t swizzled_x = (row % 8) ^ x;
    return row * 128 + swizzled_x * 16 + rem;
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
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)), "l"((uint64_t)d),
           "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_store_2d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];"
        :: "l"((uint64_t)d), "r"((uint32_t)__cvta_generic_to_shared(smem)),
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

__device__ __forceinline__ void tmem_store_4x_fn(uint32_t col, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
        :: "r"(col), "r"(r0), "r"(r1), "r"(r2), "r"(r3));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_store_fence_fn() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
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

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

// -------------------------------------------------------------------------
// Causal Attention Kernel
// -------------------------------------------------------------------------

__global__ void __launch_bounds__(128, 2) causal_attention_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    float* LSE, int S_len)
{
    extern __shared__ __align__(1024) uint8_t smem_pool[];
    uint8_t* smem_Q_0 = smem_pool;               // 16384 bytes
    uint8_t* smem_Q_1 = smem_pool + 16384;       // 16384 bytes
    uint8_t* smem_K_0 = smem_pool + 32768;       // 16384 bytes
    uint8_t* smem_K_1 = smem_pool + 49152;       // 16384 bytes
    uint8_t* smem_V_0 = smem_pool + 65536;       // 16384 bytes
    uint8_t* smem_V_1 = smem_pool + 81920;       // 16384 bytes
    uint8_t* smem_P   = smem_pool + 98304;       // 16384 bytes
    uint8_t* smem_D_0 = smem_pool + 114688;      // 16384 bytes
    uint8_t* smem_D_1 = smem_pool + 131072;      // 16384 bytes
    uint64_t* mbar_Q  = (uint64_t*)(smem_pool + 147456);
    uint64_t* mbar_KV = (uint64_t*)(smem_pool + 147464);

    int batch_head_offset = blockIdx.y * S_len;
    int seq_idx = blockIdx.x * 128;
    if (seq_idx >= S_len) return;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_KV, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        uint32_t* tmem_S_p = (uint32_t*)(smem_pool + 147472);
        uint32_t* tmem_O_0_p = (uint32_t*)(smem_pool + 147480);
        uint32_t* tmem_O_1_p = (uint32_t*)(smem_pool + 147488);
        tmem_alloc_fn(tmem_S_p, 128);
        tmem_alloc_fn(tmem_O_0_p, 128);
        tmem_alloc_fn(tmem_O_1_p, 128);
    }
    __syncthreads();
    
    uint32_t* tmem_S_p = (uint32_t*)(smem_pool + 147472);
    uint32_t* tmem_O_0_p = (uint32_t*)(smem_pool + 147480);
    uint32_t* tmem_O_1_p = (uint32_t*)(smem_pool + 147488);
    uint32_t tmem_S = tmem_S_p[0];
    uint32_t tmem_O_0 = tmem_O_0_p[0];
    uint32_t tmem_O_1 = tmem_O_1_p[0];

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 32768);
        tma_load_2d_fn(&tma_Q, mbar_Q, smem_Q_0, 0, batch_head_offset + seq_idx);
        tma_load_2d_fn(&tma_Q, mbar_Q, smem_Q_1, 64, batch_head_offset + seq_idx);
    }
    mbarrier_wait_fn(mbar_Q, 0);

    float m_val_top = -INFINITY;
    float l_val_top = 0.0f;

    float m_val_bot = -INFINITY;
    float l_val_bot = 0.0f;

    int phase = 0;

    for (int j_blk = 0; j_blk <= blockIdx.x; j_blk++) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_KV, 65536);
            tma_load_2d_fn(&tma_K, mbar_KV, smem_K_0, 0, batch_head_offset + j_blk * 128);
            tma_load_2d_fn(&tma_K, mbar_KV, smem_K_1, 64, batch_head_offset + j_blk * 128);
            tma_load_2d_fn(&tma_V, mbar_KV, smem_V_0, 0, batch_head_offset + j_blk * 128);
            tma_load_2d_fn(&tma_V, mbar_KV, smem_V_1, 64, batch_head_offset + j_blk * 128);
        }
        mbarrier_wait_fn(mbar_KV, phase);

        if (threadIdx.x == 0) {
            uint32_t idesc_Q = make_instr_desc_fn(128, 128);
            for (int k = 0; k < 4; k++) {
                uint64_t desc_Q = make_smem_desc_sm100_fn(smem_Q_0 + k * 32, 1, 1024);
                uint64_t desc_K = make_smem_desc_sm100_fn(smem_K_0 + k * 32, 1, 1024);
                umma_f16_cg2_fn(tmem_S, desc_Q, desc_K, idesc_Q, k == 0 ? 0 : 1);
            }
            for (int k = 0; k < 4; k++) {
                uint64_t desc_Q = make_smem_desc_sm100_fn(smem_Q_1 + k * 32, 1, 1024);
                uint64_t desc_K = make_smem_desc_sm100_fn(smem_K_1 + k * 32, 1, 1024);
                umma_f16_cg2_fn(tmem_S, desc_Q, desc_K, idesc_Q, 1);
            }
        }
        umma_commit_2sm_fn(mbar_KV);
        mbarrier_wait_fn(mbar_KV, phase ^ 1);

        float rowmax_top = -INFINITY;
        float rowmax_bot = -INFINITY;
        
        // Split S into two distinct chunks along the N dimension to effectively double TMEM bandwidth utilization
        if (threadIdx.x < 64) {
            int row = threadIdx.x;
            for (int col = 0; col < 64; col += 4) {
                uint32_t r0, r1, r2, r3;
                tmem_load_4x_fn(tmem_S + col, &r0, &r1, &r2, &r3);
                tmem_load_fence_fn();
                
                float f0 = __uint_as_float(r0);
                float f1 = __uint_as_float(r1);
                float f2 = __uint_as_float(r2);
                float f3 = __uint_as_float(r3);
                
                int q_pos = seq_idx + row;
                int col_idx = col;
                int k_pos = j_blk * 128 + col_idx;
                if (k_pos <= q_pos && k_pos < S_len) {
                    float val = f0 * 0.08838834764831845f;
                    if (val > rowmax_top) rowmax_top = val;
                }
                k_pos = j_blk * 128 + col_idx + 1;
                if (k_pos <= q_pos && k_pos < S_len) {
                    float val = f1 * 0.08838834764831845f;
                    if (val > rowmax_top) rowmax_top = val;
                }
                k_pos = j_blk * 128 + col_idx + 2;
                if (k_pos <= q_pos && k_pos < S_len) {
                    float val = f2 * 0.08838834764831845f;
                    if (val > rowmax_top) rowmax_top = val;
                }
                k_pos = j_blk * 128 + col_idx + 3;
                if (k_pos <= q_pos && k_pos < S_len) {
                    float val = f3 * 0.08838834764831845f;
                    if (val > rowmax_top) rowmax_top = val;
                }
            }
            
            // Perform partial warp-level reduction to coalesce maximum values effectively
            int warp_id = threadIdx.x / 32;
            int lane_id = threadIdx.x % 32;
            for (int offset = 16; offset > 0; offset /= 2) {
                rowmax_top = fmaxf(rowmax_top, __shfl_down_sync(0xffffffff, rowmax_top, offset));
            }
        } else {
            int row = threadIdx.x - 64;
            for (int col = 0; col < 64; col += 4) {
                uint32_t r0, r1, r2, r3;
                tmem_load_4x_fn(tmem_S + 64 + col, &r0, &r1, &r2, &r3);
                tmem_load_fence_fn();
                
                float f0 = __uint_as_float(r0);
                float f1 = __uint_as_float(r1);
                float f2 = __uint_as_float(r2);
                float f3 = __uint_as_float(r3);
                
                int q_pos = seq_idx + row;
                int col_idx = 64 + col;
                int k_pos = j_blk * 128 + col_idx;
                if (k_pos <= q_pos && k_pos < S_len) {
                    float val = f0 * 0.08838834764831845f;
                    if (val > rowmax_bot) rowmax_bot = val;
                }
                k_pos = j_blk * 128 + col_idx + 1;
                if (k_pos <= q_pos && k_pos < S_len) {
                    float val = f1 * 0.08838834764831845f;
                    if (val > rowmax_bot) rowmax_bot = val;
                }
                k_pos = j_blk * 128 + col_idx + 2;
                if (k_pos <= q_pos && k_pos < S_len) {
                    float val = f2 * 0.08838834764831845f;
                    if (val > rowmax_bot) rowmax_bot = val;
                }
                k_pos = j_blk * 128 + col_idx + 3;
                if (k_pos <= q_pos && k_pos < S_len) {
                    float val = f3 * 0.08838834764831845f;
                    if (val > rowmax_bot) rowmax_bot = val;
                }
            }
            
            int warp_id = threadIdx.x / 32;
            int lane_id = threadIdx.x % 32;
            for (int offset = 16; offset > 0; offset /= 2) {
                rowmax_bot = fmaxf(rowmax_bot, __shfl_down_sync(0xffffffff, rowmax_bot, offset));
            }
        }

        __syncwarp(); // ensure reductions within warp are visible to all threads executing subsequent conditional blocks
        
        float rowsum_top = 0.0f;
        float rowsum_bot = 0.0f;
        int row = (threadIdx.x < 64) ? threadIdx.x : (threadIdx.x - 64);
        int col_base = (threadIdx.x < 64) ? 0 : 64;
        
        if (threadIdx.x < 64) {
            for (int col = 0; col < 64; col += 4) {
                uint32_t r0, r1, r2, r3;
                tmem_load_4x_fn(tmem_S + col, &r0, &r1, &r2, &r3);
                tmem_load_fence_fn();
                
                float f0 = __uint_as_float(r0);
                float f1 = __uint_as_float(r1);
                float f2 = __uint_as_float(r2);
                float f3 = __uint_as_float(r3);
                
                int q_pos = seq_idx + row;
                int col_idx = col;
                int k_pos = j_blk * 128 + col_idx;
                float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
                if (k_pos <= q_pos && k_pos < S_len) {
                    float val = f0 * 0.08838834764831845f;
                    p0 = fast_exp2f_fn((val - rowmax_top) * 1.44269504f);
                    rowsum_top += p0;
                }
                k_pos = j_blk * 128 + col_idx + 1;
                if (k_pos <= q_pos && k_pos < S_len) {
                    float val = f1 * 0.08838834764831845f;
                    p1 = fast_exp2f_fn((val - rowmax_top) * 1.44269504f);
                    rowsum_top += p1;
                }
                k_pos = j_blk * 128 + col_idx + 2;
                if (k_pos <= q_pos && k_pos < S_len) {
                    float val = f2 * 0.08838834764831845f;
                    p2 = fast_exp2f_fn((val - rowmax_top) * 1.44269504f);
                    rowsum_top += p2;
                }
                k_pos = j_blk * 128 + col_idx + 3;
                if (k_pos <= q_pos && k_pos < S_len) {
                    float val = f3 * 0.08838834764831845f;
                    p3 = fast_exp2f_fn((val - rowmax_top) * 1.44269504f);
                    rowsum_top += p3;
                }
                
                uint32_t p_off0 = swizzle_128B_byte(row, col_base + col) * 2;
                uint32_t p_off1 = swizzle_128B_byte(row, col_base + col + 1) * 2;
                uint32_t p_off2 = swizzle_128B_byte(row, col_base + col + 2) * 2;
                uint32_t p_off3 = swizzle_128B_byte(row, col_base + col + 3) * 2;
                
                *(uint16_t*)&smem_P[p_off0] = *(uint16_t*)&__float2bfloat16(p0);
                *(uint16_t*)&smem_P[p_off1] = *(uint16_t*)&__float2bfloat16(p1);
                *(uint16_t*)&smem_P[p_off2] = *(uint16_t*)&__float2bfloat16(p2);
                *(uint16_t*)&smem_P[p_off3] = *(uint16_t*)&__float2bfloat16(p3);
            }
        } else {
            for (int col = 0; col < 64; col += 4) {
                uint32_t r0, r1, r2, r3;
                tmem_load_4x_fn(tmem_S + 64 + col, &r0, &r1, &r2, &r3);
                tmem_load_fence_fn();
                
                float f0 = __uint_as_float(r0);
                float f1 = __uint_as_float(r1);
                float f2 = __uint_as_float(r2);
                float f3 = __uint_as_float(r3);
                
                int q_pos = seq_idx + row;
                int col_idx = 64 + col;
                int k_pos = j_blk * 128 + col_idx;
                float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
                if (k_pos <= q_pos && k_pos < S_len) {
                    float val = f0 * 0.08838834764831845f;
                    p0 = fast_exp2f_fn((val - rowmax_bot) * 1.44269504f);
                    rowsum_bot += p0;
                }
                k_pos = j_blk * 128 + col_idx + 1;
                if (k_pos <= q_pos && k_pos < S_len) {
                    float val = f1 * 0.08838834764831845f;
                    p1 = fast_exp2f_fn((val - rowmax_bot) * 1.44269504f);
                    rowsum_bot += p1;
                }
                k_pos = j_blk * 128 + col_idx + 2;
                if (k_pos <= q_pos && k_pos < S_len) {
                    float val = f2 * 0.08838834764831845f;
                    p2 = fast_exp2f_fn((val - rowmax_bot) * 1.44269504f);
                    rowsum_bot += p2;
                }
                k_pos = j_blk * 128 + col_idx + 3;
                if (k_pos <= q_pos && k_pos < S_len) {
                    float val = f3 * 0.08838834764831845f;
                    p3 = fast_exp2f_fn((val - rowmax_bot) * 1.44269504f);
                    rowsum_bot += p3;
                }
                
                uint32_t p_off0 = swizzle_128B_byte(row, col_base + col) * 2;
                uint32_t p_off1 = swizzle_128B_byte(row, col_base + col + 1) * 2;
                uint32_t p_off2 = swizzle_128B_byte(row, col_base + col + 2) * 2;
                uint32_t p_off3 = swizzle_128B_byte(row, col_base + col + 3) * 2;
                
                *(uint16_t*)&smem_P[p_off0] = *(uint16_t*)&__float2bfloat16(p0);
                *(uint16_t*)&smem_P[p_off1] = *(uint16_t*)&__float2bfloat16(p1);
                *(uint16_t*)&smem_P[p_off2] = *(uint16_t*)&__float2bfloat16(p2);
                *(uint16_t*)&smem_P[p_off3] = *(uint16_t*)&__float2bfloat16(p3);
            }
        }

        // Accumulate Local Sum Reduction
        int warp_id = threadIdx.x / 32;
        int lane_id = threadIdx.x % 32;
        if (threadIdx.x < 64) {
            for (int offset = 16; offset > 0; offset /= 2) {
                rowsum_top += __shfl_down_sync(0xffffffff, rowsum_top, offset);
            }
            float old_m = m_val_top;
            float m_new = fmaxf(old_m, rowmax_top);
            float alpha = fast_exp2f_fn((old_m - m_new) * 1.44269504f);
            l_val_top = l_val_top * alpha + rowsum_top * fast_exp2f_fn((rowmax_top - m_new) * 1.44269504f);
            m_val_top = m_new;
        } else {
            for (int offset = 16; offset > 0; offset /= 2) {
                rowsum_bot += __shfl_down_sync(0xffffffff, rowsum_bot, offset);
            }
            float old_m = m_val_bot;
            float m_new = fmaxf(old_m, rowmax_bot);
            float alpha = fast_exp2f_fn((old_m - m_new) * 1.44269504f);
            l_val_bot = l_val_bot * alpha + rowsum_bot * fast_exp2f_fn((rowmax_bot - m_new) * 1.44269504f);
            m_val_bot = m_new;
        }

        // Vectorized 32-bit store of P to leverage fully aligned SWIZZLE_128B architectures boundaries (Critical for avoiding WGMMA bank conflicts)
        for (int i = 0; i < 64; i += 4) {
            uint32_t v0 = *(uint32_t*)&smem_P[swizzle_128B_byte(threadIdx.x, col_base + i) * 2];
            uint32_t v1 = *(uint32_t*)&smem_P[swizzle_128B_byte(threadIdx.x, col_base + i + 1) * 2];
            uint32_t v2 = *(uint32_t*)&smem_P[swizzle_128B_byte(threadIdx.x, col_base + i + 2) * 2];
            uint32_t v3 = *(uint32_t*)&smem_P[swizzle_128B_byte(threadIdx.x, col_base + i + 3) * 2];
            uint32_t packed0 = (v1 << 16) | v0;
            uint32_t packed1 = (v3 << 16) | v2;
            asm volatile("st.shared.v2.b32 [%0], {%1, %2};" :: "r"(swizzle_128B_byte(threadIdx.x, (col_base + i) * 2)), "r"(packed0), "r"(packed1));
        }

        __syncthreads();

        if (threadIdx.x == 0) {
            uint32_t idesc_PV_0 = make_instr_desc_fn(128, 64);
            uint32_t idesc_PV_1 = make_instr_desc_fn(128, 64);
            
            for (int k = 0; k < 4; k++) {
                uint64_t desc_P0 = make_smem_desc_sm100_fn(smem_P + k * 32, 1, 1024);
                uint64_t desc_V0 = make_smem_desc_sm100_fn(smem_V_0 + k * 2048, 1024, 1024);
                
                umma_f16_cg2_fn(tmem_O_0, desc_P0, desc_V0, idesc_PV_0, k == 0 ? 0 : 1);
            }
            
            for (int k = 0; k < 4; k++) {
                uint64_t desc_P1 = make_smem_desc_sm100_fn(smem_P + 8192 + k * 32, 1, 1024);
                uint64_t desc_V1 = make_smem_desc_sm100_fn(smem_V_1 + k * 2048, 1024, 1024);
                
                umma_f16_cg2_fn(tmem_O_0, desc_P1, desc_V1, idesc_PV_0, 1);
                umma_f16_cg2_fn(tmem_O_1, desc_P1, desc_V1, idesc_PV_1, k == 0 ? 0 : 1);
            }
        }
        umma_commit_2sm_fn(mbar_KV);
        mbarrier_wait_fn(mbar_KV, phase);
        phase ^= 1;
    }

    for (int col = 0; col < 64; col += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(tmem_O_0 + col, &r0, &r1, &r2, &r3);
        tmem_load_fence_fn();
        
        // Emulate vectorized packing using intrinsic logic which guarantees strictly ordered memory layout mappings
        uint32_t packed0 = pack_bf16_fn(r0, r1);
        uint32_t packed1 = pack_bf16_fn(r2, r3);
        uint32_t off0 = swizzle_128B_byte(threadIdx.x, col * 2);
        uint32_t off1 = swizzle_128B_byte(threadIdx.x, (col + 2) * 2);
        
        *(uint32_t*)&smem_D_0[off0] = packed0;
        *(uint32_t*)&smem_D_0[off1] = packed1;
    }
    
    for (int col = 0; col < 64; col += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(tmem_O_1 + col, &r0, &r1, &r2, &r3);
        tmem_load_fence_fn();
        
        uint32_t packed0 = pack_bf16_fn(r0, r1);
        uint32_t packed1 = pack_bf16_fn(r2, r3);
        uint32_t off0 = swizzle_128B_byte(threadIdx.x, col * 2);
        uint32_t off1 = swizzle_128B_byte(threadIdx.x, (col + 2) * 2);
        
        *(uint32_t*)&smem_D_1[off0] = packed0;
        *(uint32_t*)&smem_D_1[off1] = packed1;
    }
    
    __syncthreads();

    if (threadIdx.x == 0) {
        tma_store_fence_fn();
        tma_store_2d_fn(&tma_O, smem_D_0, 0, batch_head_offset + seq_idx);
        tma_store_2d_fn(&tma_O, smem_D_1, 64, batch_head_offset + seq_idx);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }
    __syncthreads();

    if (threadIdx.x < 128) {
        float lse = (threadIdx.x < 64) ? (m_val_top + logf(l_val_top)) : (m_val_bot + logf(l_val_bot));
        LSE[batch_head_offset + seq_idx + threadIdx.x] = lse;
    }

    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_S, 128);
        tmem_dealloc_fn(tmem_O_0, 128);
        tmem_dealloc_fn(tmem_O_1, 128);
    }
}

// -------------------------------------------------------------------------
// TMA Descriptor Creation
// -------------------------------------------------------------------------

CUresult create_tma_2d_descriptor_2B(
    CUtensorMap* d, void* globalAddress, 
    uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, 
    uint32_t smem_inner_dim, uint32_t smem_outer_dim, 
    CUtensorMapDataType dataType,
    CUtensorMapSwizzle swizzle, 
    CUtensorMapL2promotion l2Promotion, 
    CUtensorMapFloatOOBfill oobFill) 
{
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, dataType, 2, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle, l2Promotion, oobFill
    );
}

// -------------------------------------------------------------------------
// TVM-FFI Binding
// -------------------------------------------------------------------------

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, 
         tvm::ffi::TensorView V, tvm::ffi::TensorView O, 
         tvm::ffi::TensorView LSE) 
{
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3); 
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O;
    void* q_ptr = Q.data_ptr();
    void* k_ptr = K.data_ptr();
    void* v_ptr = V.data_ptr();
    void* o_ptr = O.data_ptr();

    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, q_ptr, D, B*H*S, 64, 128, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, k_ptr, D, B*H*S, 64, 128, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, v_ptr, D, B*H*S, 64, 128, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_O, o_ptr, D, B*H*S, 64, 128, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    float* lse_ptr = static_cast<float*>(LSE.data_ptr());

    dim3 grid((S + 127) / 128, B * H);
    dim3 block(128);
    
    int smem_size = 164 * 1024;
    CUDA_CHECK(cudaFuncSetAttribute(causal_attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    // Use explicit cluster launch config to guarantee proper underlying hardware topology mapping execution
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
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, causal_attention_kernel, tma_Q, tma_K, tma_V, tma_O, lse_ptr, S));
        
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_kernel