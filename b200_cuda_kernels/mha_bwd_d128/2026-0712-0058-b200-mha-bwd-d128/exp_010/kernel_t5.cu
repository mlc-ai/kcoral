#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_mha_bwd {

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(phase));
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(bar)),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_store_2d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1) : "memory");
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

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= ((uint64_t)((addr >> 7) & 0x7)) << 49;
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

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, bool a_major, bool b_major) {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= ((a_major ? 1 : 0) << 15);
    d |= ((b_major ? 1 : 0) << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ uint32_t pack_bf16_to_uint32(__nv_bfloat16 a, __nv_bfloat16 b) {
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&a)),
          "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}

__device__ __forceinline__ void read_float4_from_tmem(uint32_t col_addr, uint32_t* r) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
    : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(col_addr));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void write_to_smem_64x64_swizzled(char* smem, int row, int col, uint32_t val) {
    int x = col / 8;
    int rem = col % 8;
    int chunk = (row % 8) ^ x;
    int swizzled_byte_idx = (chunk * 8 + rem) * 2;
    *reinterpret_cast<uint32_t*>(&smem[row * 128 + swizzled_byte_idx]) = val;
}

__device__ __forceinline__ void write_to_smem_64x64_swizzled_transposed(char* smem, int r, int c, uint32_t val) {
    int x = r / 8;
    int rem = r % 8;
    int chunk = (c % 8) ^ x;
    int swizzled_byte_idx = (chunk * 8 + rem) * 2;
    *reinterpret_cast<uint32_t*>(&smem[c * 128 + swizzled_byte_idx]) = val;
}

__device__ __forceinline__ __nv_bfloat16 get_swizzled(const char* smem, int row, int col) {
    int x = col / 8;
    int rem = col % 8;
    int chunk = (row % 8) ^ x;
    int swizzled_byte_idx = (chunk * 8 + rem) * 2;
    return *reinterpret_cast<const __nv_bfloat16*>(&smem[row * 128 + swizzled_byte_idx]);
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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

__device__ __forceinline__ void element_wise_softmax_64x64_transposed(
    float sq_scale, const float* L, int row_offset, int col_offset, int S,
    uint32_t* S_T_ptr, uint32_t* dP_T_ptr, char* smem_P_T, char* smem_dS_T, const float* smem_D) 
{
    for (int k = 0; k < 4; ++k) {
        uint32_t col_addr0 = *S_T_ptr + ((threadIdx.x / 64) * 64) + (threadIdx.x % 64) * 16 + k * 16;
        uint32_t col_addr1 = *dP_T_ptr + ((threadIdx.x / 64) * 64) + (threadIdx.x % 64) * 16 + k * 16;
        
        uint32_t r_S0[4]; read_float4_from_tmem(col_addr0, r_S0);
        uint32_t r_dP0[4]; read_float4_from_tmem(col_addr1, r_dP0);
        
        int r_idx = (threadIdx.x % 64) + (threadIdx.x / 64) * 64;
        int c_base = k * 16 + (threadIdx.x % 64) * 16;
        
        for(int i = 0; i < 4; ++i) {
            int c_idx = c_base + i;
            float s_val = __uint_as_float(r_S0[i]);
            float dp_val = __uint_as_float(r_dP0[i]);
            
            int global_row = row_offset + r_idx;
            float lse_val = (global_row < S) ? L[global_row] : 0.0f;
            float p_val = (c_idx < 64 && global_row < S) ? expf(s_val * sq_scale - lse_val) : 0.0f;
            
            float ds_val = p_val * (dp_val - smem_D[r_idx]);
            
            write_to_smem_64x64_swizzled_transposed(smem_P_T, r_idx, c_idx, pack_bf16_to_uint32(__float2bfloat16(p_val), __float2bfloat16(p_val)));
            write_to_smem_64x64_swizzled_transposed(smem_dS_T, r_idx, c_idx, pack_bf16_to_uint32(__float2bfloat16(ds_val), __float2bfloat16(ds_val)));
        }
    }
    __syncthreads();
    fence_proxy_async_fn();
}

struct SharedStorage {
    __align__(128) char smem_Q0[8192];
    char smem_Q1[8192];
    char smem_K0[8192];
    char smem_K1[8192];
    char smem_V0[8192];
    char smem_V1[8192];
    char smem_dO0[8192];
    char smem_dO1[8192];
    char smem_O0[8192];
    char smem_O1[8192];
    
    char smem_P_T[8192];
    char smem_dS_T[8192];
    
    __align__(128) char dQ0[8192];
    char dQ1[8192];
    char dK0[8192];
    char dK1[8192];
    char dV0[8192];
    char dV1[8192];
    
    float smem_D[64];
    
    uint64_t mbar_q;
    uint64_t mbar_k;
    
    uint32_t tmem_S_T;
    uint32_t tmem_P_T;
    uint32_t tmem_dP_T;
    uint32_t tmem_dS_T;
    
    uint32_t tmem_dQ0_T;
    uint32_t tmem_dQ1_T;
    uint32_t tmem_dK0_T;
    uint32_t tmem_dK1_T;
    uint32_t tmem_dV0_T;
    uint32_t tmem_dV1_T;
};

__device__ void load_tile(const CUtensorMap* tma, uint64_t* mbar, char* smem0, char* smem1, int head_offset, int seq_offset, uint32_t& phase) {
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar, 16384);
        tma_load_2d_fn(tma, mbar, smem0, head_offset, seq_offset);
        tma_load_2d_fn(tma, mbar, smem1, head_offset + 64, seq_offset);
    }
    mbarrier_wait_fn(mbar, phase);
    phase ^= 1;
    fence_proxy_async_fn();
    __syncthreads();
}

extern __shared__ __align__(128) uint8_t smem_pool[];

__global__ void __launch_bounds__(128) mha_bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dQ,
    const __grid_constant__ CUtensorMap tma_dK,
    const __grid_constant__ CUtensorMap tma_dV,
    float sq_scale,
    int S,
    int H,
    const float* L_global,
    float* dQ_fp32) 
{
    extern __shared__ char smem_buf[];
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_buf);

    int batch_head_idx = blockIdx.z;
    int b_idx = batch_head_idx / H;
    int h_idx = batch_head_idx % H;
    const float* L = L_global + b_idx * (H * S) + h_idx * S;

    int cluster_rank = (blockIdx.x % 2 == 0) ? 0 : 1; // Deterministic election leader role mapping over even/odd blocks
    int my_offset = cluster_rank * 128;
    int kv_idx = blockIdx.x * 128;
    if (kv_idx >= S) return;
    
    uint32_t seq_offset_k = b_idx * (H * S) + h_idx * S + kv_idx;

    if (threadIdx.x == 0) {
        tmem_alloc_fn(&smem.tmem_S_T, 128);
        tmem_alloc_fn(&smem.tmem_P_T, 128);
        tmem_alloc_fn(&smem.tmem_dP_T, 128);
        tmem_alloc_fn(&smem.tmem_dS_T, 128);
        
        tmem_alloc_fn(&smem.tmem_dQ0_T, 64);
        tmem_alloc_fn(&smem.tmem_dQ1_T, 64);
        
        tmem_alloc_fn(&smem.tmem_dK0_T, 64);
        tmem_alloc_fn(&smem.tmem_dK1_T, 64);
        
        tmem_alloc_fn(&smem.tmem_dV0_T, 64);
        tmem_alloc_fn(&smem.tmem_dV1_T, 64);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&smem.mbar_q, 1);
        init_smem_barrier_fn(&smem.mbar_k, 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();

    uint32_t* S_T_ptr = &smem.tmem_S_T;
    uint32_t* P_T_ptr = &smem.tmem_P_T;
    uint32_t* dP_T_ptr = &smem.tmem_dP_T;
    uint32_t* dS_T_ptr = &smem.tmem_dS_T;
    uint32_t* dK0_T_ptr = &smem.tmem_dK0_T;
    uint32_t* dK1_T_ptr = &smem.tmem_dK1_T;
    uint32_t* dV0_T_ptr = &smem.tmem_dV0_T;
    uint32_t* dV1_T_ptr = &smem.tmem_dV1_T;
    
    uint32_t phase0 = 0, phase1 = 0;
    
    load_tile(&tma_K, &smem.mbar_k, smem.smem_K0, smem.smem_K1, 0, seq_offset_k, phase1);
    load_tile(&tma_V, &smem.mbar_k, smem.smem_V0, smem.smem_V1, 0, seq_offset_k, phase1);

    uint32_t idesc = make_instr_desc_fn(64, 64, false, true);
    uint32_t idesc_S_half0 = make_instr_desc_fn(64, 64, false, true);
    uint32_t idesc_S_half1 = make_instr_desc_fn(64, 64, false, true);
    
    uint32_t idesc_dQ_half0 = make_instr_desc_fn(64, 64, true, false);
    uint32_t idesc_dQ_half1 = make_instr_desc_fn(64, 64, true, false);
    
    uint32_t idesc_dK_half0 = make_instr_desc_fn(64, 64, true, false);
    uint32_t idesc_dK_half1 = make_instr_desc_fn(64, 64, true, false);
    
    uint32_t idesc_dV_half0 = make_instr_desc_fn(64, 64, true, false);
    uint32_t idesc_dV_half1 = make_instr_desc_fn(64, 64, true, false);

    int num_steps = S / 64;
    
    for (int idx_step = 0; idx_step < num_steps; ++idx_step) {
        int q_idx = (blockIdx.y * 128 + idx_step * 64 + my_offset) % S;
        uint32_t seq_offset_q = b_idx * (H * S) + h_idx * S + q_idx;
        
        load_tile(&tma_Q, &smem.mbar_q, smem.smem_Q0, smem.smem_Q1, 0, seq_offset_q, phase0);
        load_tile(&tma_O, &smem.mbar_q, smem.smem_O0, smem.smem_O1, 0, seq_offset_q, phase0);
        load_tile(&tma_dO, &smem.mbar_q, smem.smem_dO0, smem.smem_dO1, 0, seq_offset_q, phase0);
        
        if (threadIdx.x < 64) {
            float d_val = 0;
            for (int c = 0; c < 64; ++c) {
                d_val += __bfloat162float(get_swizzled(smem.smem_O0, threadIdx.x, c)) * 
                         __bfloat162float(get_swizzled(smem.smem_dO0, threadIdx.x, c));
                d_val += __bfloat162float(get_swizzled(smem.smem_O1, threadIdx.x, c)) * 
                         __bfloat162float(get_swizzled(smem.smem_dO1, threadIdx.x, c));
            }
            smem.smem_D[threadIdx.x] = d_val;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            for (int k = 0; k < 4; ++k) {
                uint32_t tmem_s0 = *S_T_ptr + k * 16;
                uint64_t desc_Q0_S = make_smem_desc_sm100_fn((char*)(smem.smem_Q0 + k * 32), 0, 1024);
                uint64_t desc_K0_S = make_smem_desc_sm100_fn((char*)(smem.smem_K0 + k * 32), 0, 1024);
                umma_f16_cg2_fn(tmem_s0, desc_Q0_S, desc_K0_S, idesc_S_half0, (k == 0) ? 0 : 1);
                
                uint32_t tmem_s1 = *S_T_ptr + 64 + k * 16;
                uint64_t desc_Q1_S = make_smem_desc_sm100_fn((char*)(smem.smem_Q1 + k * 32), 0, 1024);
                uint64_t desc_K1_S = make_smem_desc_sm100_fn((char*)(smem.smem_K1 + k * 32), 0, 1024);
                umma_f16_cg2_fn(tmem_s1, desc_Q1_S, desc_K1_S, idesc_S_half1, 1);
                
                uint32_t tmem_dp0 = *dP_T_ptr + k * 16;
                uint64_t desc_dO0_S = make_smem_desc_sm100_fn((char*)(smem.smem_dO0 + k * 32), 0, 1024);
                uint64_t desc_V0_S = make_smem_desc_sm100_fn((char*)(smem.smem_V0 + k * 32), 0, 1024);
                umma_f16_cg2_fn(tmem_dp0, desc_dO0_S, desc_V0_S, idesc, (k == 0) ? 0 : 1);
                
                uint32_t tmem_dp1 = *dP_T_ptr + 64 + k * 16;
                uint64_t desc_dO1_S = make_smem_desc_sm100_fn((char*)(smem.smem_dO1 + k * 32), 0, 1024);
                uint64_t desc_V1_S = make_smem_desc_sm100_fn((char*)(smem.smem_V1 + k * 32), 0, 1024);
                umma_f16_cg2_fn(tmem_dp1, desc_dO1_S, desc_V1_S, idesc, 1);
            }
        }
        __syncthreads();
        
        element_wise_softmax_64x64_transposed(sq_scale, L, q_idx, kv_idx, S, S_T_ptr, dP_T_ptr, smem.smem_P_T, smem.smem_dS_T, smem.smem_D);
        
        if (threadIdx.x == 0) {
            for (int k = 0; k < 4; ++k) {
                uint32_t tmem_dQ0 = *dQ0_T_ptr + k * 16;
                uint64_t desc_dS = make_smem_desc_sm100_fn((char*)(smem.smem_dS_T + k * 32 * 64), 8192, 1024);
                uint64_t desc_K0_N = make_smem_desc_sm100_fn((char*)(smem.smem_K0 + k * 32 * 64), 8192, 1024);
                umma_f16_cg2_fn(tmem_dQ0, desc_dS, desc_K0_N, idesc_dQ_half0, (k == 0) ? 0 : 1);
                
                uint32_t tmem_dQ1 = *dQ1_T_ptr + k * 16;
                uint64_t desc_K1_N = make_smem_desc_sm100_fn((char*)(smem.smem_K1 + k * 32 * 64), 8192, 1024);
                umma_f16_cg2_fn(tmem_dQ1, desc_dS, desc_K1_N, idesc_dQ_half1, (k == 0) ? 0 : 1);
            }
            
            for (int k = 0; k < 4; ++k) {
                uint32_t tmem_dK0 = *dK0_T_ptr + k * 16;
                uint64_t desc_dST = make_smem_desc_sm100_fn((char*)(smem.smem_dS_T + k * 32 * 64), 8192, 1024);
                uint64_t desc_Q0_N = make_smem_desc_sm100_fn((char*)(smem.smem_Q0 + k * 32 * 64), 8192, 1024);
                umma_f16_cg2_fn(tmem_dK0, desc_dST, desc_Q0_N, idesc_dK_half0, 1);
                
                uint32_t tmem_dK1 = *dK1_T_ptr + k * 16;
                uint64_t desc_Q1_N = make_smem_desc_sm100_fn((char*)(smem.smem_Q1 + k * 32 * 64), 8192, 1024);
                umma_f16_cg2_fn(tmem_dK1, desc_dST, desc_Q1_N, idesc_dK_half1, 1);
            }
            
            for (int k = 0; k < 4; ++k) {
                uint32_t tmem_dV0 = *dV0_T_ptr + k * 16;
                uint64_t desc_P_T = make_smem_desc_sm100_fn((char*)(smem.smem_P_T + k * 32 * 64), 8192, 1024);
                uint64_t desc_dO0_N = make_smem_desc_sm100_fn((char*)(smem.smem_dO0 + k * 32 * 64), 8192, 1024);
                umma_f16_cg2_fn(tmem_dV0, desc_P_T, desc_dO0_N, idesc_dV_half0, 1);
                
                uint32_t tmem_dV1 = *dV1_T_ptr + k * 16;
                uint64_t desc_dO1_N = make_smem_desc_sm100_fn((char*)(smem.smem_dO1 + k * 32 * 64), 8192, 1024);
                umma_f16_cg2_fn(tmem_dV1, desc_P_T, desc_dO1_N, idesc_dV_half1, 1);
            }
        }
        __syncthreads();

        // Perform highly concurrent atomic adds directly scaling out global pressure bottlenecks natively resolving accumulation races.
        for (uint32_t col = 0; col < 64; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(*dQ0_T_ptr + col));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float4 vals = make_float4(__uint_as_float(r0), __uint_as_float(r1), __uint_as_float(r2), __uint_as_float(r3));
            
            int row = threadIdx.x % 64;
            int global_row = q_idx * 64 + row;
            
            if (global_row < S * 64) {
                 atomicAdd(&dQ_fp32[global_row * 128 + col], vals.x);
                 atomicAdd(&dQ_fp32[global_row * 128 + col + 1], vals.y);
                 atomicAdd(&dQ_fp32[global_row * 128 + col + 2], vals.z);
                 atomicAdd(&dQ_fp32[global_row * 128 + col + 3], vals.w);
            }
        }
        
        for (uint32_t col = 0; col < 64; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(*dQ1_T_ptr + col));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float4 vals = make_float4(__uint_as_float(r0), __uint_as_float(r1), __uint_as_float(r2), __uint_as_float(r3));
            
            int row = threadIdx.x % 64;
            int global_row = q_idx * 64 + row;
            
            if (global_row < S * 64) {
                 atomicAdd(&dQ_fp32[global_row * 128 + col + 64], vals.x);
                 atomicAdd(&dQ_fp32[global_row * 128 + col + 65], vals.y);
                 atomicAdd(&dQ_fp32[global_row * 128 + col + 66], vals.z);
                 atomicAdd(&dQ_fp32[global_row * 128 + col + 67], vals.w);
            }
        }
        __syncthreads();
    }

    for (uint32_t col = 0; col < 64; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(*dK0_T_ptr + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        uint32_t packed0 = pack_bf16_to_uint32(__float2bfloat16(__uint_as_float(r0)), __float2bfloat16(__uint_as_float(r1)));
        uint32_t packed1 = pack_bf16_to_uint32(__float2bfloat16(__uint_as_float(r2)), __float2bfloat16(__uint_as_float(r3)));
        
        int row = threadIdx.x % 64;
        int x = col / 8;
        int rem = col % 8;
        int chunk = (row % 8) ^ x;
        int swizzled_byte_idx = (chunk * 8 + rem) * 2;
        *reinterpret_cast<uint32_t*>(&smem.dK0[row * 128 + swizzled_byte_idx]) = packed0;
        *reinterpret_cast<uint32_t*>(&smem.dK1[row * 128 + swizzled_byte_idx]) = packed1;
    }
    
    for (uint32_t col = 0; col < 64; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(*dV0_T_ptr + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        uint32_t packed0 = pack_bf16_to_uint32(__float2bfloat16(__uint_as_float(r0)), __float2bfloat16(__uint_as_float(r1)));
        uint32_t packed1 = pack_bf16_to_uint32(__float2bfloat16(__uint_as_float(r2)), __float2bfloat16(__uint_as_float(r3)));
        
        int row = threadIdx.x % 64;
        int x = col / 8;
        int rem = col % 8;
        int chunk = (row % 8) ^ x;
        int swizzled_byte_idx = (chunk * 8 + rem) * 2;
        *reinterpret_cast<uint32_t*>(&smem.dV0[row * 128 + swizzled_byte_idx]) = packed0;
        *reinterpret_cast<uint32_t*>(&smem.dV1[row * 128 + swizzled_byte_idx]) = packed1;
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        tma_store_2d_fn(&tma_dK, smem.dK0, 0, seq_offset_k);
        tma_store_2d_fn(&tma_dK, smem.dK1, 64, seq_offset_k);
        tma_store_2d_fn(&tma_dV, smem.dV0, 0, seq_offset_k);
        tma_store_2d_fn(&tma_dV, smem.dV1, 64, seq_offset_k);
        tma_store_commit_fn();
    }
    tma_store_wait_fn<0>();
    __syncthreads();
}

__global__ void cast_fp32_to_bf16(__nv_bfloat16* out, const float* in, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
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

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L, tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3);

    if (d != 128) {
        fprintf(stderr, "Error: head dimension must be 128\n");
        exit(1);
    }

    uint64_t gmem_inner_dim = 64;
    uint64_t gmem_outer_dim = B * H * S;
    uint32_t smem_inner_dim = 64; 
    uint32_t smem_outer_dim = 64;

    CUtensorMap tma_Q, tma_K, tma_V, tma_dO, tma_O, tma_dQ, tma_dK, tma_dV;

    #define CREATE_TMA(tma, ptr) do { \
        CUresult res = create_tma_2d_descriptor_2B(&(tma), (ptr), gmem_inner_dim, gmem_outer_dim, smem_inner_dim, smem_outer_dim, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE); \
        if (res != CUDA_SUCCESS) { \
            fprintf(stderr, "Error creating TMA descriptor\n"); \
            exit(1); \
        } \
    } while(0)

    CREATE_TMA(tma_Q, Q.data_ptr());
    CREATE_TMA(tma_K, K.data_ptr());
    CREATE_TMA(tma_V, V.data_ptr());
    CREATE_TMA(tma_dO, dO.data_ptr());
    CREATE_TMA(tma_O, O.data_ptr());
    CREATE_TMA(tma_dQ, dQ.data_ptr());
    CREATE_TMA(tma_dK, dK.data_ptr());
    CREATE_TMA(tma_dV, dV.data_ptr());

    dim3 grid(ceil(S / 128.0), ceil(S / 128.0), B * H);
    dim3 block(128);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    float* dQ_fp32;
    CUDA_CHECK(cudaMalloc(&dQ_fp32, B * H * S * d * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(dQ_fp32, 0, B * H * S * d * sizeof(float), stream));

    // Allocate additional dynamic smem space padding to resolve alignment limits 
    int smem_size = sizeof(SharedStorage) + 1024;
    CUDA_CHECK(cudaFuncSetAttribute(mha_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
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
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, mha_bwd_kernel,
        tma_Q, tma_K, tma_V, tma_dO, tma_O, tma_dQ, tma_dK, tma_dV,
        1.0f / sqrt(128.0f), S, H, static_cast<const float*>(L.data_ptr()), dQ_fp32
    ));
    
    CUDA_CHECK(cudaGetLastError());
    
    int64_t num_elements = B * H * S * d;
    int cast_threads = 256;
    int cast_blocks = (num_elements + cast_threads - 1) / cast_threads;
    cast_fp32_to_bf16<<<cast_blocks, cast_threads, 0, stream>>>(static_cast<__nv_bfloat16*>(dQ.data_ptr()), dQ_fp32, num_elements);
    
    CUDA_CHECK(cudaFree(dQ_fp32));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

} // namespace tvm_ffi_mha_bwd