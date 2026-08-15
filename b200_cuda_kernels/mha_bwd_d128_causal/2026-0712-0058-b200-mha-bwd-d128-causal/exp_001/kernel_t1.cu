#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <math.h>
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

#define CU_CHECK(call) do {                                      \
    CUresult _e = (call);                                        \
    if (_e != CUDA_SUCCESS) {                                    \
        fprintf(stderr, "CU error %d at %s:%d\n",                \
                _e, __FILE__, __LINE__);                         \
        exit(1);                                                 \
    }                                                            \
} while(0)

namespace tvm_ffi_mha_bwd {

constexpr uint32_t BM_Q = 128; 
constexpr uint32_t BN_d = 128;

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
        "{\n.reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase));
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

__device__ __forceinline__ uint64_t make_smem_desc_sm100(void* smem_ptr, uint32_t LBO, uint32_t SBO) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((LBO & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((SBO & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_mn_major(void* smem_ptr, uint32_t BM, uint32_t BN, uint32_t BK) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    uint32_t SBO = 8 * 128;
    uint32_t LBO = (BK / 8) * SBO;
    d |= (uint64_t)((LBO & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((SBO & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
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

__device__ __forceinline__ void gemm_128x128_k_major(
    uint32_t tmem_C, uint8_t* smem_A, uint8_t* smem_B,
    uint32_t M, uint32_t N, uint32_t K_steps, uint64_t* barrier, uint32_t& phase) 
{
    uint32_t SBO_A = 1024, LBO_A = 1;
    uint32_t SBO_B = 1024, LBO_B = 1;

    mbarrier_arrive_and_expect_tx_fn(barrier, 4096); 
    
    for (uint32_t k = 0; k < K_steps; ++k) {
        uint64_t desc_a = make_smem_desc_sm100(smem_A + k * 16, LBO_A, SBO_A);
        uint64_t desc_b = make_smem_desc_sm100(smem_B + k * 16, LBO_B, SBO_B);
        uint32_t idesc = make_instr_desc_fn(M, N);
        bool accum = (k == 0) ? false : true;
        umma_f16_cg2_fn(tmem_C, desc_a, desc_b, idesc, accum ? 1 : 0);
    }
    umma_commit_2sm_fn(barrier);
    mbarrier_wait_fn(barrier, phase);
    phase ^= 1;
}

__device__ __forceinline__ void gemm_128x128_mn_major(
    uint32_t tmem_C, uint8_t* smem_A, uint8_t* smem_B,
    uint32_t M, uint32_t N, uint32_t K_steps, uint64_t* barrier, uint32_t& phase) 
{
    mbarrier_arrive_and_expect_tx_fn(barrier, 4096); 
    
    for (uint32_t k = 0; k < K_steps; ++k) {
        uint64_t desc_a = make_smem_desc_sm100_mn_major(smem_A + k * 1024, 128, 128, 128);
        uint64_t desc_b = make_smem_desc_sm100_mn_major(smem_B + k * 1024, 128, 128, 128);
        uint32_t idesc = make_instr_desc_fn(M, N);
        idesc |= (1u << 15);
        idesc |= (1u << 16);
        bool accum = (k == 0) ? false : true;
        umma_f16_cg2_fn(tmem_C, desc_a, desc_b, idesc, accum ? 1 : 0);
    }
    umma_commit_2sm_fn(barrier);
    mbarrier_wait_fn(barrier, phase);
    phase ^= 1;
}

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

__device__ __forceinline__ void load_tile_128x128(
    const CUtensorMap* desc, uint64_t* bar, uint8_t* smem_A, 
    uint32_t base_S, uint32_t head, uint32_t batch) 
{
    mbarrier_arrive_and_expect_tx_fn(bar, 32768);
    tma_load_4d_fn(desc, bar, smem_A, 0, base_S, head, batch);
    tma_load_4d_fn(desc, bar, smem_A+8192, 64, base_S, head, batch);
    tma_load_4d_fn(desc, bar, smem_A+4096, 0, base_S+64, head, batch);
    tma_load_4d_fn(desc, bar, smem_A+12288, 64, base_S+64, head, batch);
    mbarrier_wait_fn(bar, 0); 
}

__device__ __forceinline__ __nv_bfloat16 read_swizzled(__nv_bfloat16* ptr, int row, int col) {
    uint32_t x_idx = col / 8;
    uint32_t rem_x = col % 8;
    uint32_t xor_idx = (row % 8) ^ x_idx;
    uint32_t swizzled_col = xor_idx * 8 + rem_x;
    return ptr[row * 64 + swizzled_col];
}

__device__ __forceinline__ __nv_bfloat16 read_swizzled_128x128(uint8_t* smem_P, int row, int col) {
    if (row < 64 && col < 64) return read_swizzled((__nv_bfloat16*)smem_P, row, col);
    if (row < 64 && col >= 64) return read_swizzled((__nv_bfloat16*)(smem_P + 16384), row, col - 64);
    if (row >= 64 && col < 64) return read_swizzled((__nv_bfloat16*)(smem_P + 8192), row - 64, col);
    if (row >= 64 && col >= 64) return read_swizzled((__nv_bfloat16*)(smem_P + 24576), row - 64, col - 64);
    return 0;
}

__device__ __forceinline__ void write_swizzled(__nv_bfloat16* ptr, int row, int col, float val) {
    __nv_bfloat16 v = __float2bfloat16(val);
    uint32_t x_idx = col / 8;
    uint32_t rem_x = col % 8;
    uint32_t xor_idx = (row % 8) ^ x_idx;
    uint32_t swizzled_col = xor_idx * 8 + rem_x;
    ptr[row * 64 + swizzled_col] = v;
}

__device__ __forceinline__ void write_to_f_swizzled(uint8_t* smem, int row, int col, float val) {
    if (row < 64 && col < 64) write_swizzled((__nv_bfloat16*)smem, row, col, val);
    else if (row < 64 && col >= 64) write_swizzled((__nv_bfloat16*)(smem + 16384), row, col - 64, val);
    else if (row >= 64 && col < 64) write_swizzled((__nv_bfloat16*)(smem + 8192), row - 64, col, val);
    else if (row >= 64 && col >= 64) write_swizzled((__nv_bfloat16*)(smem + 24576), row - 64, col - 64, val);
}

// Optimization: Utilize warp-level reduction (shfl_down) to accelerate row-sum calculations over the 128-element row width
__device__ __forceinline__ float compute_D_pass(
    uint8_t* smem_P, uint8_t* smem_dP, int row_base, int col_base, int S) {
    float sum = 0.0f;
    for (int c = threadIdx.x; c < 128; c += 128) {
        int s_j = col_base + c;
        if (s_j < S) {
            float p = __bfloat162float(read_swizzled_128x128(smem_P, row_base + threadIdx.x, c));
            float dp = __bfloat162float(read_swizzled_128x128(smem_dP, row_base + threadIdx.x, c));
            sum += p * dp;
        }
    }
    // Reduce across the 32 threads within a single warp
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    return sum;
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void named_barrier_sync_fn(int bar_id, int count) {
    asm volatile("barrier.sync.aligned %0, %1;" :: "r"(bar_id), "r"(count));
}

__device__ __forceinline__ void tmem_epilogue_coalesced_4w_fn(
    __nv_bfloat16* D, __nv_bfloat16* smem_out,
    uint32_t M, uint32_t N, uint32_t m_block, uint32_t n_block,
    uint32_t BM, uint32_t BN) {
    for (uint32_t col = 0; col < BN; col += 4) {
 uint32_t r0, r1, r2, r3;
 asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
: "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col));
 uint32_t base = threadIdx.x * BN + col;
 smem_out[base + 0] = __float2bfloat16(__uint_as_float(r0));
 smem_out[base + 1] = __float2bfloat16(__uint_as_float(r1));
 smem_out[base + 2] = __float2bfloat16(__uint_as_float(r2));
 smem_out[base + 3] = __float2bfloat16(__uint_as_float(r3));
    }
    tmem_load_fence_fn();
    named_barrier_sync_fn(1, 128);
    
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = (BM + 3) / 4;
    for (uint32_t step = 0; step < num_steps; ++step) {
 uint32_t row = step * 4 + warp_id;
 if (row >= BM) continue;
 uint32_t global_row = m_block * BM + row;
 uint32_t col_start = lane_id * 4;
 uint32_t global_col = n_block * BN + col_start;
 if (global_row < M && global_col + 3 < N) {
     uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * BN + col_start]);
     *reinterpret_cast<uint2*>(D + (uint64_t)global_row * N + global_col) = data;
 }
    }
}

__global__ void mha_bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* L_ptr,
    float* L_ptr_D,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    uint32_t S, uint32_t H, uint32_t B, bool is_forward)
{
    extern __shared__ __align__(128) uint64_t barriers[];
    uint64_t* bar_q = &barriers[0];
    uint64_t* bar_k = &barriers[1];
    uint64_t* bar_v = &barriers[2];
    uint64_t* bar_o = &barriers[3];
    uint64_t* bar_do = &barriers[4];

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar_q, 1);
        init_smem_barrier_fn(bar_k, 1);
        init_smem_barrier_fn(bar_v, 1);
        init_smem_barrier_fn(bar_o, 1);
        init_smem_barrier_fn(bar_do, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    extern __shared__ __align__(16) uint32_t tmem_buf[];
    uint32_t* tmem_A = tmem_buf;
    uint32_t* tmem_B = tmem_buf + 1;
    uint32_t* tmem_C = tmem_buf + 2;

    int ncols_A = 256;
    int ncols_B = 256;
    int ncols_C = 256;

    if (threadIdx.x == 0) {
        tmem_alloc_fn(tmem_A, ncols_A);
        tmem_alloc_fn(tmem_B, ncols_B);
        tmem_alloc_fn(tmem_C, ncols_C);
    }
    __syncthreads(); 

    uint32_t num_blocks = (S + 127) / 128;
    uint32_t i_block = blockIdx.x;
    uint32_t head_idx = blockIdx.y;
    uint32_t head = head_idx % H;
    uint32_t batch = head_idx / H;
    uint32_t s_i = i_block * 128;

    if (is_forward) {
        uint32_t phase_q = 0, phase_k = 0, phase_v = 0, phase_o = 0, phase_do = 0;

        extern __shared__ char smem_buf[];
        uintptr_t smem_addr = (uintptr_t)smem_buf;
        uint32_t align_offset = (1024 - (smem_addr % 1024)) % 1024;
        char* smem_aligned = smem_buf + align_offset;

        uint8_t* smem_Q = (uint8_t*)smem_aligned;
        uint8_t* smem_K = (uint8_t*)smem_aligned + 32768;
        uint8_t* smem_V = (uint8_t*)smem_aligned + 65536;
        uint8_t* smem_O = (uint8_t*)smem_aligned + 98304;
        uint8_t* smem_dO = (uint8_t*)smem_aligned + 131072;
        uint8_t* smem_P = (uint8_t*)smem_aligned + 163840;
        uint8_t* smem_dP = (uint8_t*)smem_aligned + 196608; 
        uint8_t* smem_dP_corrected = (uint8_t*)smem_aligned + 229376;
        uint8_t* smem_F = smem_dP;
        uint8_t* smem_DQ = smem_Q;
        uint8_t* smem_DK = smem_K;
        uint8_t* smem_dV = smem_V;

        uint32_t phase_F = 0, phase_dP = 0, phase_sD = 0, phase_DQ = 0;
        
        load_tile_128x128(&tma_Q, bar_q, smem_Q, s_i, head, batch);
        load_tile_128x128(&tma_O, bar_o, smem_O, s_i, head, batch);
        load_tile_128x128(&tma_dO, bar_do, smem_dO, s_i, head, batch);

        uint32_t old_phase_k = phase_k;
        uint32_t old_phase_v = phase_v;
        
        float sD = 0;
        // Pass 1: calculate P, Dp, and accumulate correction factor sum(Dp * P)
        for (int j_block = 0; j_block <= i_block; ++j_block) {
            uint32_t s_j = j_block * 128;
            load_tile_128x128(&tma_K, bar_k, smem_K, s_j, head, batch);
            load_tile_128x128(&tma_V, bar_v, smem_V, s_j, head, batch);
            
            phase_k = old_phase_k;
            phase_v = old_phase_v;
            gemm_128x128_k_major((uint32_t)__cvta_generic_to_shared(tmem_C), smem_Q, smem_K, 128, 128, 8, bar_k, phase_k);
            gemm_128x128_k_major((uint32_t)__cvta_generic_to_shared(tmem_B), smem_dO, smem_V, 128, 128, 8, bar_v, phase_v);
            
            if (threadIdx.x < 128) {
                int row = threadIdx.x;
                int col = threadIdx.x;
                float F_val = __uint_as_float(*((uint32_t*)&tmem_C[(row << 8) | col]));
                F_val *= 0.08838834764831845; // scale 1/sqrt(128)
                
                int s_j_idx = s_j + col;
                int s_i_idx = s_i + row;
                if (s_j_idx > s_i_idx || j_block > i_block) {
                    F_val = -1e20;
                }
                write_to_f_swizzled(smem_F, row, col, F_val);
                
                float P_val = expf(F_val - L_ptr[head_idx * S + s_i_idx]);
                if (s_i_idx >= S || s_j_idx >= S) P_val = 0;
                write_to_f_swizzled(smem_P, row, col, P_val);
                
                float dP_val = __uint_as_float(*((uint32_t*)&tmem_B[(row << 8) | col]));
                write_to_f_swizzled(smem_dP, row, col, dP_val);
            }
            __syncthreads();
            fence_proxy_async_fn();
            
            if (threadIdx.x < 128) {
                float D_val = compute_D_pass(smem_P, smem_dP, threadIdx.x, 0, S);
                __shared__ float partial_D[128];
                partial_D[threadIdx.x] = D_val;
                __syncthreads();
                if (threadIdx.x == 0) {
                    for (int i=0; i<128; ++i) sD += partial_D[i];
                }
            }
            __syncthreads();
        }
        
        // Correct dp and accumulate DQ
        for (int j_block = 0; j_block <= i_block; ++j_block) {
            uint32_t s_j = j_block * 128;
            load_tile_128x128(&tma_K, bar_k, smem_K, s_j, head, batch);
            load_tile_128x128(&tma_V, bar_v, smem_V, s_j, head, batch);
            
            phase_k = old_phase_k;
            phase_v = old_phase_v;
            gemm_128x128_k_major((uint32_t)__cvta_generic_to_shared(tmem_C), smem_Q, smem_K, 128, 128, 8, bar_k, phase_k);
            gemm_128x128_k_major((uint32_t)__cvta_generic_to_shared(tmem_B), smem_dO, smem_V, 128, 128, 8, bar_v, phase_v);
            
            if (threadIdx.x < 128) {
                int row = threadIdx.x;
                int col = threadIdx.x;
                float F_val = __uint_as_float(*((uint32_t*)&tmem_C[(row << 8) | col]));
                F_val *= 0.08838834764831845; 
                
                int s_j_idx = s_j + col;
                int s_i_idx = s_i + row;
                if (s_j_idx > s_i_idx || j_block > i_block) {
                    F_val = -1e20;
                }
                
                float P_val = expf(F_val - L_ptr[head_idx * S + s_i_idx]);
                if (s_i_idx >= S || s_j_idx >= S) P_val = 0;
                
                float dP_val = __uint_as_float(*((uint32_t*)&tmem_B[(row << 8) | col]));
                float dP_corr = P_val * (dP_val - sD);
                if (s_j_idx > s_i_idx || j_block > i_block) dP_corr = 0;
                write_to_f_swizzled(smem_dP_corrected, row, col, dP_corr);
            }
            __syncthreads();
            fence_proxy_async_fn();
            
            gemm_128x128_k_major((uint32_t)__cvta_generic_to_shared(tmem_A), smem_dP_corrected, smem_K, 128, 128, 8, bar_k, phase_DQ);
        }
        
        fence_proxy_async_fn();
        tmem_epilogue_coalesced_4w_fn(dQ + head_idx * S * 128, (__nv_bfloat16*)smem_DQ, 
            S, 128, i_block, 0, 128, 128); 
            
        if (threadIdx.x == 0) {
            *(float*)(L_ptr_D + head_idx * num_blocks + i_block) = sD;
        }
    } else {
        uint32_t phase_q = 0, phase_k = 0, phase_v = 0, phase_o = 0, phase_do = 0;

        extern __shared__ char smem_buf[];
        uintptr_t smem_addr = (uintptr_t)smem_buf;
        uint32_t align_offset = (1024 - (smem_addr % 1024)) % 1024;
        char* smem_aligned = smem_buf + align_offset;

        uint8_t* smem_Q = (uint8_t*)smem_aligned;
        uint8_t* smem_K = (uint8_t*)smem_aligned + 32768;
        uint8_t* smem_V = (uint8_t*)smem_aligned + 65536;
        uint8_t* smem_O = (uint8_t*)smem_aligned + 98304;
        uint8_t* smem_dO = (uint8_t*)smem_aligned + 131072;
        uint8_t* smem_P = (uint8_t*)smem_aligned + 163840;
        uint8_t* smem_dP = (uint8_t*)smem_aligned + 196608; 
        uint8_t* smem_dP_corrected = (uint8_t*)smem_aligned + 229376;
        uint8_t* smem_F = smem_dP;
        uint8_t* smem_DQ = smem_Q;
        uint8_t* smem_DK = smem_K;
        uint8_t* smem_dV = smem_V;

        uint32_t j_block = i_block;
        uint32_t s_j = j_block * 128;
        
        load_tile_128x128(&tma_K, bar_k, smem_K, s_j, head, batch);
        load_tile_128x128(&tma_V, bar_v, smem_V, s_j, head, batch);
        
        uint32_t old_phase_q = phase_q;
        uint32_t old_phase_o = phase_o;
        uint32_t old_phase_do = phase_do;
        
        uint32_t phase_DK = 0, phase_dV = 0;
        
        // Compute dk and dv symmetric to assure correct causal masking contributions
        for (int i_block_b = num_blocks - 1; i_block_b >= j_block; --i_block_b) {
            uint32_t s_i_b = i_block_b * 128;
            
            load_tile_128x128(&tma_Q, bar_q, smem_Q, s_i_b, head, batch);
            load_tile_128x128(&tma_O, bar_o, smem_O, s_i_b, head, batch);
            load_tile_128x128(&tma_dO, bar_do, smem_dO, s_i_b, head, batch);
            
            phase_q = old_phase_q;
            phase_o = old_phase_o;
            phase_do = old_phase_do;
            
            gemm_128x128_k_major((uint32_t)__cvta_generic_to_shared(tmem_C), smem_Q, smem_K, 128, 128, 8, bar_k, phase_k);
            gemm_128x128_k_major((uint32_t)__cvta_generic_to_shared(tmem_B), smem_dO, smem_V, 128, 128, 8, bar_v, phase_v);
            
            if (threadIdx.x < 128) {
                int row = threadIdx.x;
                int col = threadIdx.x;
                float F_val = __uint_as_float(*((uint32_t*)&tmem_C[(row << 8) | col]));
                F_val *= 0.08838834764831845; 
                
                int s_i_idx = s_i_b + row;
                int s_j_idx = s_j + col;
                if (s_j_idx > s_i_idx || j_block > i_block_b) {
                    F_val = -1e20;
                }
                write_to_f_swizzled(smem_F, row, col, F_val);
                
                float P_val = expf(F_val - L_ptr[head_idx * S + s_i_idx]);
                if (s_i_idx >= S || s_j_idx >= S) P_val = 0;
                write_to_f_swizzled(smem_P, row, col, P_val);
                
                float dP_val = __uint_as_float(*((uint32_t*)&tmem_B[(row << 8) | col]));
                write_to_f_swizzled(smem_dP, row, col, dP_val);
            }
            __syncthreads();
            fence_proxy_async_fn();
            
            float sD = 0;
            if (threadIdx.x < 128) {
                float D_val = compute_D_pass(smem_P, smem_dP, threadIdx.x, 0, S);
                __shared__ float partial_D[128];
                partial_D[threadIdx.x] = D_val;
                __syncthreads();
                if (threadIdx.x == 0) {
                    for (int i=0; i<128; ++i) sD += partial_D[i];
                }
            }
            __syncthreads();
            
            // Override sD with the dynamic programming state preserved directly via global memory in the forward pass
            sD = *(float*)(L_ptr_D + head_idx * num_blocks + i_block_b);
            
            if (threadIdx.x < 128) {
                int row = threadIdx.x;
                int col = threadIdx.x;
                
                float P_val = __bfloat162float(read_swizzled_128x128(smem_P, row, col));
                float dP_val = __bfloat162float(read_swizzled_128x128(smem_dP, row, col));
                float dP_corr = P_val * (dP_val - sD);
                
                int s_i_idx = s_i_b + row;
                int s_j_idx = s_j + col;
                if (s_j_idx > s_i_idx || j_block > i_block_b) {
                    dP_corr = 0;
                }
                write_to_f_swizzled(smem_dP_corrected, row, col, dP_corr);
            }
            __syncthreads();
            fence_proxy_async_fn();
            
            gemm_128x128_mn_major((uint32_t)__cvta_generic_to_shared(tmem_C), smem_P, smem_dO, 128, 128, 4, bar_p, phase_dV);
            gemm_128x128_mn_major((uint32_t)__cvta_generic_to_shared(tmem_B), smem_dP_corrected, smem_Q, 128, 128, 4, bar_dpc, phase_DK);
        }
        
        fence_proxy_async_fn();
        tmem_epilogue_coalesced_4w_fn(dK + head_idx * S * 128, (__nv_bfloat16*)smem_DK, 
            S, 128, j_block, 0, 128, 128);
        tmem_epilogue_coalesced_4w_fn(dV + head_idx * S * 128, (__nv_bfloat16*)smem_dV, 
            S, 128, j_block, 0, 128, 128);
    }

    if (threadIdx.x == 0) {
        tmem_dealloc_fn((uint32_t)__cvta_generic_to_shared(tmem_A), ncols_A);
        tmem_dealloc_fn((uint32_t)__cvta_generic_to_shared(tmem_B), ncols_B);
        tmem_dealloc_fn((uint32_t)__cvta_generic_to_shared(tmem_C), ncols_C);
    }
    __syncthreads();
}

CUresult create_tma_4d_descriptor(CUtensorMap* d, void* globalAddress, 
                                  uint64_t dim0, uint64_t dim1, uint64_t dim2, uint64_t dim3,
                                  uint32_t box0, uint32_t box1) {
    cuuint64_t globalDim[4] = {dim0, dim1, dim2, dim3};
    cuuint64_t globalStrides[3] = {dim0 * 2, dim0 * dim1 * 2, dim0 * dim1 * dim2 * 2};
    cuuint32_t boxDim[4] = {box0, box1, 1, 1};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        4, 
        globalAddress,
        globalDim,
        globalStrides,
        boxDim, 
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    uint32_t B = Q.size(0);
    uint32_t H = Q.size(1);
    uint32_t S = Q.size(2);
    uint32_t d = Q.size(3); 
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O, tma_dO;
    CU_CHECK(create_tma_4d_descriptor(&tma_Q, Q.data_ptr(), d, S, H, B, 64, 64));
    CU_CHECK(create_tma_4d_descriptor(&tma_K, K.data_ptr(), d, S, H, B, 64, 64));
    CU_CHECK(create_tma_4d_descriptor(&tma_V, V.data_ptr(), d, S, H, B, 64, 64));
    CU_CHECK(create_tma_4d_descriptor(&tma_O, O.data_ptr(), d, S, H, B, 64, 64));
    CU_CHECK(create_tma_4d_descriptor(&tma_dO, dO.data_ptr(), d, S, H, B, 64, 64));
    
    float* L_ptr;
    uint32_t num_blocks = (S + 127) / 128;
    CUDA_CHECK(cudaMallocAsync(&L_ptr, num_blocks * H * B * sizeof(float), stream));
    
    dim3 grid(num_blocks, H * B);
    dim3 block(128);
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 262144;
    config.stream = stream;
    
    cudaLaunchAttribute attrs[2];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    
    attrs[1].id = cudaLaunchAttributeMaxDynamicSharedMemorySize;
    attrs[1].val.maxDynamicSharedMemorySize = 262144;
    
    config.attrs = attrs;
    config.numAttrs = 2;
    
    CUDA_CHECK(cudaFuncSetAttribute(tvm_ffi_mha_bwd::mha_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 262144));
    CUDA_CHECK(cudaFuncSetAttribute(tvm_ffi_mha_bwd::mha_bwd_kernel, cudaFuncAttributeClusterSharedMemory, 1));
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, tvm_ffi_mha_bwd::mha_bwd_kernel,
        tma_Q, tma_K, tma_V, tma_O, tma_dO, 
        static_cast<const float*>(L.data_ptr()), L_ptr,
        static_cast<__nv_bfloat16*>(dQ.data_ptr()), 
        static_cast<__nv_bfloat16*>(dK.data_ptr()), 
        static_cast<__nv_bfloat16*>(dV.data_ptr()), 
        S, H, B, true)); 
        
    CUDA_CHECK(cudaLaunchKernelEx(&config, tvm_ffi_mha_bwd::mha_bwd_kernel,
        tma_Q, tma_K, tma_V, tma_O, tma_dO, 
        static_cast<const float*>(L.data_ptr()), L_ptr,
        static_cast<__nv_bfloat16*>(dQ.data_ptr()), 
        static_cast<__nv_bfloat16*>(dK.data_ptr()), 
        static_cast<__nv_bfloat16*>(dV.data_ptr()), 
        S, H, B, false));
        
    CUDA_CHECK(cudaFreeAsync(L_ptr, stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

} // namespace tvm_ffi_mha_bwd