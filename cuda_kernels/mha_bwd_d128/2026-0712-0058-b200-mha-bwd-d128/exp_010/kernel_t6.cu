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
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
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

__device__ __forceinline__ __nv_bfloat16 get_swizzled(const char* smem, int row, int col) {
    int x = col / 8;
    int rem = col % 8;
    int chunk = (row % 8) ^ x;
    int swizzled_byte_idx = (chunk * 8 + rem) * 2;
    return *reinterpret_cast<const __nv_bfloat16*>(&smem[row * 128 + swizzled_byte_idx]);
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

__device__ __forceinline__ void element_wise_softmax_64x64_transposed(
    float sq_scale, const float* L, int row_offset, int col_offset, int S,
    uint32_t* S_T_ptr, uint32_t* dP_T_ptr, char* smem_P_T, char* smem_dS_T, const float* smem_D) 
{
    int my_row = threadIdx.x % 64;
    int half_row = threadIdx.x / 64;
    int global_row = row_offset + my_row;
    float lse_val = (global_row < S) ? L[global_row] : 0.0f;
    float d_val = smem_D[my_row];
    
    for (uint32_t col = 0; col < 64; col += 4) {
        uint32_t addr_S = *S_T_ptr + half_row * 64 + col;
        uint32_t addr_dP = *dP_T_ptr + half_row * 64 + col;
        
        uint32_t r_S[4], r_dP[4];
        read_float4_from_tmem(addr_S, r_S);
        read_float4_from_tmem(addr_dP, r_dP);

        uint32_t packed_p[4];
        uint32_t packed_ds[4];
        
        for(int k = 0; k < 4; ++k) {
            int col_idx = col + k;
            int global_col = col_offset + col_idx;
            float col_val = __uint_as_float(r_S[k]);
            float dp_val = __uint_as_float(r_dP[k]);
            
            float p_val = (global_col < S && global_row < S) ? expf(col_val * sq_scale - lse_val) : 0.0f;
            float ds_val = p_val * (dp_val - d_val);
            
            packed_p[k] = pack_bf16_to_uint32(__float2bfloat16(p_val), __float2bfloat16(p_val));
            packed_ds[k] = pack_bf16_to_uint32(__float2bfloat16(ds_val), __float2bfloat16(ds_val));
        }
        
        uint32_t smem_addr_p = (uint32_t)__cvta_generic_to_shared(smem_P_T) + my_row * 128 + (col / 8) * 16;
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" :: "r"(smem_addr_p), "r"(packed_p[0]), "r"(packed_p[1]), "r"(packed_p[2]), "r"(packed_p[3]) : "memory");
        uint32_t smem_addr_ds = (uint32_t)__cvta_generic_to_shared(smem_dS_T) + my_row * 128 + (col / 8) * 16;
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" :: "r"(smem_addr_ds), "r"(packed_ds[0]), "r"(packed_ds[1]), "r"(packed_ds[2]), "r"(packed_ds[3]) : "memory");
    }
}

struct SharedStorage {
    __align__(128) char Q0[8192];
    char Q1[8192];
    char K0[8192];
    char K1[8192];
    char V0[8192];
    char V1[8192];
    char dO0[8192];
    char dO1[8192];
    char O0[8192];
    char O1[8192];
    
    __align__(128) char P_T[8192];
    __align__(128) char dS_T[8192];
    
    __align__(128) char dQ0[8192];
    char dQ1[8192];
    char dK0[8192];
    char dK1[8192];
    char dV0[8192];
    char dV1[8192];
    
    float D[64];
    
    uint64_t mbar_q;
    uint64_t mbar_k;
    
    uint32_t S_T;
    uint32_t P_T;
    uint32_t dP_T;
    uint32_t dS_T;
    
    uint32_t dQ0_T;
    uint32_t dQ1_T;
    uint32_t dK0_T;
    uint32_t dK1_T;
    uint32_t dV0_T;
    uint32_t dV1_T;
};

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
    const float* L_global) 
{
    extern __shared__ char smem_buf[];
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_buf);

    int batch_head_idx = blockIdx.y;
    int b_idx = batch_head_idx / H;
    int h_idx = batch_head_idx % H;
    const float* L = L_global + b_idx * (H * S) + h_idx * S;

    if (threadIdx.x == 0) {
        tmem_alloc_fn(&smem.S_T, 128);
        tmem_alloc_fn(&smem.P_T, 128);
        tmem_alloc_fn(&smem.dP_T, 128);
        tmem_alloc_fn(&smem.dS_T, 128);
        
        tmem_alloc_fn(&smem.dQ0_T, 64);
        tmem_alloc_fn(&smem.dQ1_T, 64);
        
        tmem_alloc_fn(&smem.dK0_T, 64);
        tmem_alloc_fn(&smem.dK1_T, 64);
        
        tmem_alloc_fn(&smem.dV0_T, 64);
        tmem_alloc_fn(&smem.dV1_T, 64);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&smem.mbar_q, 1);
        init_smem_barrier_fn(&smem.mbar_k, 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();

    uint32_t* S_T_ptr = &smem.S_T;
    uint32_t* P_T_ptr = &smem.P_T;
    uint32_t* dP_T_ptr = &smem.dP_T;
    uint32_t* dS_T_ptr = &smem.dS_T;
    uint32_t* dK0_T_ptr = &smem.dK0_T;
    uint32_t* dK1_T_ptr = &smem.dK1_T;
    uint32_t* dV0_T_ptr = &smem.dV0_T;
    uint32_t* dV1_T_ptr = &smem.dV1_T;
    uint32_t* dQ0_T_ptr = &smem.dQ0_T;
    uint32_t* dQ1_T_ptr = &smem.dQ1_T;
    
    // ==========================================
    // Pass 1: Compute dQ
    // ==========================================
    int idx_q = blockIdx.x;
    if (idx_q < S / 64) {
        uint32_t seq_offset_q = b_idx * (H * S) + h_idx * S + idx_q * 64;
        uint32_t phase0 = 0, phase1 = 0;
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&smem.mbar_q, 16384);
            tma_load_2d_fn(&tma_Q, &smem.mbar_q, smem.Q0, 0, seq_offset_q);
            tma_load_2d_fn(&tma_Q, &smem.mbar_q, smem.Q1, 64, seq_offset_q);
            
            mbarrier_arrive_and_expect_tx_fn(&smem.mbar_q, 16384);
            tma_load_2d_fn(&tma_O, &smem.mbar_q, smem.O0, 0, seq_offset_q);
            tma_load_2d_fn(&tma_O, &smem.mbar_q, smem.O1, 64, seq_offset_q);
            
            mbarrier_arrive_and_expect_tx_fn(&smem.mbar_q, 16384);
            tma_load_2d_fn(&tma_dO, &smem.mbar_q, smem.dO0, 0, seq_offset_q);
            tma_load_2d_fn(&tma_dO, &smem.mbar_q, smem.dO1, 64, seq_offset_q);
        }
        mbarrier_wait_fn(&smem.mbar_q, phase0);
        phase0 ^= 1;
        fence_proxy_async_fn();
        __syncthreads();
        
        if (threadIdx.x < 64) {
            float d_val = 0;
            for (int c = 0; c < 64; ++c) {
                d_val += __bfloat162float(get_swizzled(smem.O0, threadIdx.x, c)) * 
                         __bfloat162float(get_swizzled(smem.dO0, threadIdx.x, c));
                d_val += __bfloat162float(get_swizzled(smem.O1, threadIdx.x, c)) * 
                         __bfloat162float(get_swizzled(smem.dO1, threadIdx.x, c));
            }
            smem.D[threadIdx.x] = d_val;
        }
        __syncthreads();

        uint32_t idesc = make_instr_desc_fn(64, 64, false, true);
        uint32_t idesc_dQ_half0 = make_instr_desc_fn(64, 64, true, true);
        uint32_t idesc_dQ_half1 = make_instr_desc_fn(64, 64, true, true);
        
        for (int idx_step = 0; idx_step < S / 64; ++idx_step) {
            uint32_t seq_offset_k = b_idx * (H * S) + h_idx * S + idx_step * 64;
            
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&smem.mbar_k, 16384);
                tma_load_2d_fn(&tma_K, &smem.mbar_k, smem.K0, 0, seq_offset_k);
                tma_load_2d_fn(&tma_K, &smem.mbar_k, smem.K1, 64, seq_offset_k);

                mbarrier_arrive_and_expect_tx_fn(&smem.mbar_k, 16384);
                tma_load_2d_fn(&tma_V, &smem.mbar_k, smem.V0, 0, seq_offset_k);
                tma_load_2d_fn(&tma_V, &smem.mbar_k, smem.V1, 64, seq_offset_k);
            }
            mbarrier_wait_fn(&smem.mbar_k, phase1);
            phase1 ^= 1;
            fence_proxy_async_fn();
            __syncthreads();

            bool first_dq = (idx_step == 0);
            if (threadIdx.x == 0) {
                for (int k = 0; k < 4; ++k) {
                    uint32_t tmem_s0 = *S_T_ptr + k * 16;
                    uint64_t desc_Q0_S = make_smem_desc_sm100_fn((char*)(smem.Q0 + k * 32), 1, 1024);
                    uint64_t desc_K0_S = make_smem_desc_sm100_fn((char*)(smem.K0 + k * 32), 1, 1024);
                    umma_f16_cg1_fn(tmem_s0, desc_Q0_S, desc_K0_S, idesc, first_dq ? 0 : 1);
                    
                    uint32_t tmem_s1 = *S_T_ptr + 64 + k * 16;
                    uint64_t desc_Q1_S = make_smem_desc_sm100_fn((char*)(smem.Q1 + k * 32), 1, 1024);
                    uint64_t desc_K1_S = make_smem_desc_sm100_fn((char*)(smem.K1 + k * 32), 1, 1024);
                    umma_f16_cg1_fn(tmem_s1, desc_Q1_S, desc_K1_S, idesc, 1);
                    
                    uint32_t tmem_dp0 = *dP_T_ptr + k * 16;
                    uint64_t desc_dO0_S = make_smem_desc_sm100_fn((char*)(smem.dO0 + k * 32), 1, 1024);
                    uint64_t desc_V0_S = make_smem_desc_sm100_fn((char*)(smem.V0 + k * 32), 1, 1024);
                    umma_f16_cg1_fn(tmem_dp0, desc_dO0_S, desc_V0_S, idesc, first_dq ? 0 : 1);
                    
                    uint32_t tmem_dp1 = *dP_T_ptr + 64 + k * 16;
                    uint64_t desc_dO1_S = make_smem_desc_sm100_fn((char*)(smem.dO1 + k * 32), 1, 1024);
                    uint64_t desc_V1_S = make_smem_desc_sm100_fn((char*)(smem.V1 + k * 32), 1, 1024);
                    umma_f16_cg1_fn(tmem_dp1, desc_dO1_S, desc_V1_S, idesc, 1);
                }
                
                for (int k = 0; k < 4; ++k) {
                    uint32_t tmem_dQ0 = *dQ0_T_ptr + k * 16;
                    uint64_t desc_dS = make_smem_desc_sm100_fn((char*)(smem.dS_T + k * 32), 128, 1024);
                    uint64_t desc_K0_N = make_smem_desc_sm100_fn((char*)(smem.K0 + k * 32), 128, 1024);
                    umma_f16_cg1_fn(tmem_dQ0, desc_dS, desc_K0_N, idesc_dQ_half0, (k == 0 && first_dq) ? 0 : 1);
                    
                    uint32_t tmem_dQ1 = *dQ1_T_ptr + k * 16;
                    uint64_t desc_K1_N = make_smem_desc_sm100_fn((char*)(smem.K1 + k * 32), 128, 1024);
                    umma_f16_cg1_fn(tmem_dQ1, desc_dS, desc_K1_N, idesc_dQ_half1, (k == 0 && first_dq) ? 0 : 1);
                }
            }
            __syncthreads();
            
            element_wise_softmax_64x64_transposed(sq_scale, L, idx_q * 64, idx_step * 64, S, S_T_ptr, dP_T_ptr, smem.P_T, smem.dS_T, smem.D);
            
            fence_proxy_async_fn();
            __syncthreads();
        }

        for (uint32_t col = 0; col < 64; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(*dQ0_T_ptr + col));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            uint32_t packed0 = pack_bf16_to_uint32(__float2bfloat16(__uint_as_float(r0)), __float2bfloat16(__uint_as_float(r1)));
            uint32_t packed1 = pack_bf16_to_uint32(__float2bfloat16(__uint_as_float(r2)), __float2bfloat16(__uint_as_float(r3)));
            
            int row = threadIdx.x % 64;
            int half = threadIdx.x / 64;
            int x = col / 8;
            int rem = col % 8;
            int chunk = (row % 8) ^ x;
            int swizzled_byte_idx = (chunk * 8 + rem) * 2;
            
            *reinterpret_cast<uint32_t*>(&smem.dQ0[row * 128 + swizzled_byte_idx]) = packed0;
            *reinterpret_cast<uint32_t*>(&smem.dQ1[row * 128 + swizzled_byte_idx]) = packed1;
        }
        __syncthreads();
        
        if (threadIdx.x == 0) {
            tma_store_2d_fn(&tma_dQ, smem.dQ0, 0, seq_offset_q);
            tma_store_2d_fn(&tma_dQ, smem.dQ1, 64, seq_offset_q);
            tma_store_commit_fn();
        }
        tma_store_wait_fn<0>();
        __syncthreads();
    }

    // ==========================================
    // Pass 2: Compute dK and dV
    // ==========================================
    int idx_kv = blockIdx.x;
    if (idx_kv < S / 64) {
        uint32_t seq_offset_k = b_idx * (H * S) + h_idx * S + idx_kv * 64;
        uint32_t phase0 = 0, phase1 = 0;
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&smem.mbar_k, 16384);
            tma_load_2d_fn(&tma_K, &smem.mbar_k, smem.K0, 0, seq_offset_k);
            tma_load_2d_fn(&tma_K, &smem.mbar_k, smem.K1, 64, seq_offset_k);

            mbarrier_arrive_and_expect_tx_fn(&smem.mbar_k, 16384);
            tma_load_2d_fn(&tma_V, &smem.mbar_k, smem.V0, 0, seq_offset_k);
            tma_load_2d_fn(&tma_V, &smem.mbar_k, smem.V1, 64, seq_offset_k);
        }
        mbarrier_wait_fn(&smem.mbar_k, phase1);
        phase1 ^= 1;
        fence_proxy_async_fn();
        __syncthreads();

        uint32_t idesc = make_instr_desc_fn(64, 64, false, true);
        uint32_t idesc_dV_half0 = make_instr_desc_fn(64, 64, true, true);
        uint32_t idesc_dV_half1 = make_instr_desc_fn(64, 64, true, true);
        
        uint32_t idesc_dK_half0 = make_instr_desc_fn(64, 64, true, true);
        uint32_t idesc_dK_half1 = make_instr_desc_fn(64, 64, true, true);
        
        for (int idx_step = 0; idx_step < S / 64; ++idx_step) {
            uint32_t seq_offset_q = b_idx * (H * S) + h_idx * S + idx_step * 64;
            
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&smem.mbar_q, 16384);
                tma_load_2d_fn(&tma_Q, &smem.mbar_q, smem.Q0, 0, seq_offset_q);
                tma_load_2d_fn(&tma_Q, &smem.mbar_q, smem.Q1, 64, seq_offset_q);
                
                mbarrier_arrive_and_expect_tx_fn(&smem.mbar_q, 16384);
                tma_load_2d_fn(&tma_O, &smem.mbar_q, smem.O0, 0, seq_offset_q);
                tma_load_2d_fn(&tma_O, &smem.mbar_q, smem.O1, 64, seq_offset_q);
                
                mbarrier_arrive_and_expect_tx_fn(&smem.mbar_q, 16384);
                tma_load_2d_fn(&tma_dO, &smem.mbar_q, smem.dO0, 0, seq_offset_q);
                tma_load_2d_fn(&tma_dO, &smem.mbar_q, smem.dO1, 64, seq_offset_q);
            }
            mbarrier_wait_fn(&smem.mbar_q, phase0);
            phase0 ^= 1;
            fence_proxy_async_fn();
            __syncthreads();
            
            if (threadIdx.x < 64) {
                float d_val = 0;
                for (int c = 0; c < 64; ++c) {
                    d_val += __bfloat162float(get_swizzled(smem.O0, threadIdx.x, c)) * 
                             __bfloat162float(get_swizzled(smem.dO0, threadIdx.x, c));
                    d_val += __bfloat162float(get_swizzled(smem.O1, threadIdx.x, c)) * 
                             __bfloat162float(get_swizzled(smem.dO1, threadIdx.x, c));
                }
                smem.D[threadIdx.x] = d_val;
            }
            __syncthreads();

            bool first_dk = (idx_step == 0);
            bool first_dv = (idx_step == 0);
            
            if (threadIdx.x == 0) {
                for (int k = 0; k < 4; ++k) {
                    uint32_t tmem_s0 = *S_T_ptr + k * 16;
                    uint64_t desc_Q0_S = make_smem_desc_sm100_fn((char*)(smem.Q0 + k * 32), 1, 1024);
                    uint64_t desc_K0_S = make_smem_desc_sm100_fn((char*)(smem.K0 + k * 32), 1, 1024);
                    umma_f16_cg1_fn(tmem_s0, desc_Q0_S, desc_K0_S, idesc, first_dk ? 0 : 1);
                    
                    uint32_t tmem_s1 = *S_T_ptr + 64 + k * 16;
                    uint64_t desc_Q1_S = make_smem_desc_sm100_fn((char*)(smem.Q1 + k * 32), 1, 1024);
                    uint64_t desc_K1_S = make_smem_desc_sm100_fn((char*)(smem.K1 + k * 32), 1, 1024);
                    umma_f16_cg1_fn(tmem_s1, desc_Q1_S, desc_K1_S, idesc, 1);
                    
                    uint32_t tmem_dp0 = *dP_T_ptr + k * 16;
                    uint64_t desc_dO0_S = make_smem_desc_sm100_fn((char*)(smem.dO0 + k * 32), 1, 1024);
                    uint64_t desc_V0_S = make_smem_desc_sm100_fn((char*)(smem.V0 + k * 32), 1, 1024);
                    umma_f16_cg1_fn(tmem_dp0, desc_dO0_S, desc_V0_S, idesc, first_dv ? 0 : 1);
                    
                    uint32_t tmem_dp1 = *dP_T_ptr + 64 + k * 16;
                    uint64_t desc_dO1_S = make_smem_desc_sm100_fn((char*)(smem.dO1 + k * 32), 1, 1024);
                    uint64_t desc_V1_S = make_smem_desc_sm100_fn((char*)(smem.V1 + k * 32), 1, 1024);
                    umma_f16_cg1_fn(tmem_dp1, desc_dO1_S, desc_V1_S, idesc, 1);
                }
                
                for (int k = 0; k < 4; ++k) {
                    uint32_t tmem_dK0 = *dK0_T_ptr + k * 16;
                    uint64_t desc_dST = make_smem_desc_sm100_fn((char*)(smem.dS_T + k * 32), 128, 1024);
                    uint64_t desc_Q0_N = make_smem_desc_sm100_fn((char*)(smem.Q0 + k * 32), 128, 1024);
                    umma_f16_cg1_fn(tmem_dK0, desc_dST, desc_Q0_N, idesc_dK_half0, (k == 0 && first_dk) ? 0 : 1);
                    
                    uint32_t tmem_dK1 = *dK1_T_ptr + k * 16;
                    uint64_t desc_Q1_N = make_smem_desc_sm100_fn((char*)(smem.Q1 + k * 32), 128, 1024);
                    umma_f16_cg1_fn(tmem_dK1, desc_dST, desc_Q1_N, idesc_dK_half1, (k == 0 && first_dk) ? 0 : 1);
                }
                
                for (int k = 0; k < 4; ++k) {
                    uint32_t tmem_dV0 = *dV0_T_ptr + k * 16;
                    uint64_t desc_PT = make_smem_desc_sm100_fn((char*)(smem.P_T + k * 32), 128, 1024);
                    uint64_t desc_dO0_N = make_smem_desc_sm100_fn((char*)(smem.dO0 + k * 32), 128, 1024);
                    umma_f16_cg1_fn(tmem_dV0, desc_PT, desc_dO0_N, idesc_dV_half0, (k == 0 && first_dv) ? 0 : 1);
                    
                    uint32_t tmem_dV1 = *dV1_T_ptr + k * 16;
                    uint64_t desc_dO1_N = make_smem_desc_sm100_fn((char*)(smem.dO1 + k * 32), 128, 1024);
                    umma_f16_cg1_fn(tmem_dV1, desc_PT, desc_dO1_N, idesc_dV_half1, (k == 0 && first_dv) ? 0 : 1);
                }
            }
            __syncthreads();
            
            element_wise_softmax_64x64_transposed(sq_scale, L, idx_step * 64, idx_kv * 64, S, S_T_ptr, dP_T_ptr, smem.P_T, smem.dS_T, smem.D);
            
            fence_proxy_async_fn();
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
            int half = threadIdx.x / 64;
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
            int half = threadIdx.x / 64;
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

    uint64_t gmem_inner_dim = 128;
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

    dim3 grid(ceil(S / 64.0), B * H);
    dim3 block(128);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    // Allocate additional dynamic smem space padding to resolve alignment limits 
    int smem_size = sizeof(SharedStorage) + 1024;
    CUDA_CHECK(cudaFuncSetAttribute(mha_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    mha_bwd_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, tma_dO, tma_O, tma_dQ, tma_dK, tma_dV,
        1.0f / sqrt(128.0f), S, H, static_cast<const float*>(L.data_ptr())
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

} // namespace tvm_ffi_mha_bwd