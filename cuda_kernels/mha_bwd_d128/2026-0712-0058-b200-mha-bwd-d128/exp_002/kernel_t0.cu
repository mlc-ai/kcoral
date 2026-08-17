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

#define CUDA_DRIVER_CHECK(call) do {                               \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        fprintf(stderr, "CUDA Driver error %d at %s:%d\n",         \
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_mha_bwd {

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
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

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
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

__device__ __forceinline__ void tmemp_store_bf32(uint32_t tmem_addr, uint32_t lane, float val) {
    uint32_t addr = (tmem_addr >> 6) + (lane << 10);
    uint32_t val_u32 = __float_as_uint(val);
    asm volatile("cp.register.to.shared.b32 %0, %1;" :: "r"(addr), "r"(val_u32));
}

__device__ __forceinline__ void tmem_epilogue_coalesced_4w_fn(
    __nv_bfloat16* D, __nv_bfloat16* smem_out,
    uint32_t M, uint32_t N, uint32_t m_block, uint32_t n_block,
    uint32_t BM, uint32_t BN) {
    for (uint32_t col = 0; col < BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t base = threadIdx.x * BN + col;
        smem_out[base + 0] = __float2bfloat16(__uint_as_float(r0));
        smem_out[base + 1] = __float2bfloat16(__uint_as_float(r1));
        smem_out[base + 2] = __float2bfloat16(__uint_as_float(r2));
        smem_out[base + 3] = __float2bfloat16(__uint_as_float(r3));
    }
    __syncthreads();
    
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
    const __grid_constant__ CUtensorMap tma_q,
    const __grid_constant__ CUtensorMap tma_k,
    const __grid_constant__ CUtensorMap tma_v,
    const __grid_constant__ CUtensorMap tma_o,
    const __grid_constant__ CUtensorMap tma_do,
    __nv_bfloat16* dQ,
    __nv_bfloat16* dK,
    __nv_bfloat16* dV,
    const float* L,
    int32_t S,
    int32_t part)
{
    setmaxnreg_inc_sync_fn<248>();

    extern __shared__ __align__(128) uint8_t smem_raw[];
    uint32_t raw_ptr = (uint32_t)__cvta_generic_to_shared(smem_raw);
    uint32_t ptr = (raw_ptr + 1023) & ~1023; // Ensure 1024-byte alignment for SWIZZLE_128B patterns
    
    __nv_bfloat16* smem_do = (__nv_bfloat16*)ptr;
    __nv_bfloat16* smem_o = smem_do + 8192;
    __nv_bfloat16* smem_q = smem_o + 8192;
    __nv_bfloat16* smem_p = smem_q + 8192;
    __nv_bfloat16* smem_v = smem_p + 4096;
    __nv_bfloat16* smem_k = smem_v + 8192;
    __nv_bfloat16* smem_dp = smem_k + 8192;
    float* smem_l = (float*)(smem_dp + 4096);
    float* smem_rowsum_P_D = smem_l + 64;
    __nv_bfloat16* smem_out = (__nv_bfloat16*)(smem_rowsum_P_D + 64);
    
    uint64_t* mbar = (uint64_t*)(smem_out + 8192);

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&mbar[0], 1);
        init_smem_barrier_fn(&mbar[1], 1);
        init_smem_barrier_fn(&mbar[2], 1);
        init_smem_barrier_fn(&mbar[3], 1);
        init_smem_barrier_fn(&mbar[4], 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    uint32_t tmem_d, tmem_s, tmem_dq, tmem_dk, tmem_dv, tmem_dp;
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_d, 64);
        tmem_alloc_fn(&tmem_s, 64);
        tmem_alloc_fn(&tmem_dq, 128);
        tmem_alloc_fn(&tmem_dk, 128);
        tmem_alloc_fn(&tmem_dv, 128);
        tmem_alloc_fn(&tmem_dp, 64);
    }
    __syncthreads();

    uint32_t phase[5] = {0};
    
    if (part == 1) {
        // ========== Compute dQ = P * (dS - dS@P^T) @ K ==========
        uint32_t bhs_base = blockIdx.x * 64;
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar[0], 16384 * 3);
            tma_load_2d_fn(&tma_do, &mbar[0], smem_do, 0, bhs_base);
            tma_load_2d_fn(&tma_do, &mbar[0], smem_do + 4096, 64, bhs_base);
            tma_load_2d_fn(&tma_o, &mbar[0], smem_o, 0, bhs_base);
            tma_load_2d_fn(&tma_o, &mbar[0], smem_o + 4096, 64, bhs_base);
            tma_load_2d_fn(&tma_q, &mbar[0], smem_q, 0, bhs_base);
            tma_load_2d_fn(&tma_q, &mbar[0], smem_q + 4096, 64, bhs_base);
        }
        if (threadIdx.x < 64) {
            smem_l[threadIdx.x] = L[bhs_base + threadIdx.x];
        }
        mbarrier_wait_fn(&mbar[0], phase[0]);
        phase[0] ^= 1;
        
        if (threadIdx.x < 64) {
            float sum = 0.0f;
            for (uint32_t col = 0; col < 64; col++) {
                sum += __bfloat162float(smem_do[threadIdx.x * 64 + col]) * __bfloat162float(smem_o[threadIdx.x * 64 + col]);
                sum += __bfloat162float(smem_do[threadIdx.x * 64 + col + 4096]) * __bfloat162float(smem_o[threadIdx.x * 64 + col + 4096]);
            }
            smem_rowsum_P_D[threadIdx.x] = sum;
        }
        __syncthreads();
        
        uint32_t phase_d = 0, phase_k = 0, phase_v = 0, phase_dp = 0, phase_dq = 0;
        uint32_t idesc = make_instr_desc_fn(64, 64);
        uint32_t idesc_dq = make_instr_desc_fn(64, 64);
        idesc_dq |= (1 << 16);
        
        for (uint32_t s_base = 0; s_base < S; s_base += 64) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&mbar[2], 16384);
                tma_load_2d_fn(&tma_v, &mbar[2], smem_v, 0, s_base);
                tma_load_2d_fn(&tma_v, &mbar[2], smem_v + 4096, 64, s_base);
                
                mbarrier_arrive_and_expect_tx_fn(&mbar[3], 16384);
                tma_load_2d_fn(&tma_k, &mbar[3], smem_k, 0, s_base);
                tma_load_2d_fn(&tma_k, &mbar[3], smem_k + 4096, 64, s_base);
            }
            mbarrier_wait_fn(&mbar[2], phase_v);
            phase_v ^= 1;
            mbarrier_wait_fn(&mbar[3], phase_k);
            phase_k ^= 1;
            
            // D = dO @ V^T
            for (uint32_t k = 0; k < 128; k += 16) {
                uint64_t desc_a = make_smem_desc_sm100_fn(smem_do + k, 0, 1024);
                uint64_t desc_b = make_smem_desc_sm100_fn(smem_v + k, 0, 1024);
                umma_f16_cg2_fn(tmem_d, desc_a, desc_b, idesc, (phase_d == 0) ? 0 : 1);
                phase_d++;
            }
            umma_commit_2sm_fn(&mbar[1]);
            mbarrier_wait_fn(&mbar[1], phase[1]);
            phase[1] ^= 1;
            
            // dP = P * (D - rowsum_P_D)
            for (uint32_t i = 0; i < 64; i += 4) {
                uint32_t r_d0, r_d1, r_d2, r_d3;
                tmem_load_4x_fn(tmem_d + threadIdx.x * 64 + i, &r_d0, &r_d1, &r_d2, &r_d3);
                
                float f_d0 = __uint_as_float(r_d0);
                float f_d1 = __uint_as_float(r_d1);
                float f_d2 = __uint_as_float(r_d2);
                float f_d3 = __uint_as_float(r_d3);
                
                float dp0 = __bfloat162float(smem_p[(threadIdx.x * 64 + i)]) * (f_d0 - smem_rowsum_P_D[threadIdx.x]);
                float dp1 = __bfloat162float(smem_p[(threadIdx.x * 64 + i + 1)]) * (f_d1 - smem_rowsum_P_D[threadIdx.x]);
                float dp2 = __bfloat162float(smem_p[(threadIdx.x * 64 + i + 2)]) * (f_d2 - smem_rowsum_P_D[threadIdx.x]);
                float dp3 = __bfloat162float(smem_p[(threadIdx.x * 64 + i + 3)]) * (f_d3 - smem_rowsum_P_D[threadIdx.x]);
                
                uint32_t p0 = pack_bf16_fn(__float_as_uint(dp0), __float_as_uint(dp1));
                uint32_t p1 = pack_bf16_fn(__float_as_uint(dp2), __float_as_uint(dp3));
                *(uint32_t*)&smem_dp[(threadIdx.x * 64 + i) * 2 + 0] = p0;
                *(uint32_t*)&smem_dp[(threadIdx.x * 64 + i) * 2 + 2] = p1;
            }
            __syncthreads();
            fence_async_shared_fn();
            
            // dQ += dP @ K
            for (uint32_t k = 0; k < 64; k += 16) {
                uint64_t desc_a = make_smem_desc_sm100_fn(smem_dp + k, 0, 1024);
                uint64_t desc_b0 = make_smem_desc_sm100_fn(smem_k + k*64, 8192, 1024);
                umma_f16_cg2_fn(tmem_dq, desc_a, desc_b0, idesc_dq, (phase_dq == 0) ? 0 : 1);
                phase_dq++;
                
                uint64_t desc_b1 = make_smem_desc_sm100_fn(smem_k + 4096 + k*64, 8192, 1024);
                umma_f16_cg2_fn(tmem_dq + 64, desc_a, desc_b1, idesc_dq, (phase_dq == 0) ? 0 : 1);
                phase_dq++;
            }
            umma_commit_2sm_fn(&mbar[1]);
            mbarrier_wait_fn(&mbar[1], phase[1]);
            phase[1] ^= 1;
        }
        
        // Epilogue for dQ
        tmem_epilogue_coalesced_4w_fn(dQ, smem_out, B_total * H_total * S, 128, bhs_base, 0, 64, 64);
        tmem_epilogue_coalesced_4w_fn(dQ, smem_out, B_total * H_total * S, 128, bhs_base, 64, 64, 64);

    } else {
        // ========== Compute dK = Q^T @ dP and dV = P^T @ dO ==========
        uint32_t s_base = blockIdx.x * 64;
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar[3], 16384);
            tma_load_2d_fn(&tma_k, &mbar[3], smem_k, 0, s_base);
            tma_load_2d_fn(&tma_k, &mbar[3], smem_k + 4096, 64, s_base);
            
            mbarrier_arrive_and_expect_tx_fn(&mbar[2], 16384);
            tma_load_2d_fn(&tma_v, &mbar[2], smem_v, 0, s_base);
            tma_load_2d_fn(&tma_v, &mbar[2], smem_v + 4096, 64, s_base);
        }
        mbarrier_wait_fn(&mbar[3], phase_k);
        phase_k ^= 1;
        mbarrier_wait_fn(&mbar[2], phase_v);
        phase_v ^= 1;

        uint32_t phase_dp = 0, phase_dk = 0, phase_dv = 0;
        uint32_t idesc_dp = make_instr_desc_fn(64, 64);
        uint32_t idesc_dk = make_instr_desc_fn(64, 64);
        idesc_dk |= (1 << 15);
        idesc_dk |= (1 << 16);
        uint32_t idesc_dv = make_instr_desc_fn(64, 64);
        idesc_dv |= (1 << 15);
        idesc_dv |= (1 << 16);
        
        for (uint32_t bhs_base = 0; bhs_base < B_total * H_total * S; bhs_base += 64) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&mbar[1], 16384 * 3);
                tma_load_2d_fn(&tma_q, &mbar[1], smem_q, 0, bhs_base);
                tma_load_2d_fn(&tma_q, &mbar[1], smem_q + 4096, 64, bhs_base);
                tma_load_2d_fn(&tma_do, &mbar[1], smem_do, 0, bhs_base);
                tma_load_2d_fn(&tma_do, &mbar[1], smem_do + 4096, 64, bhs_base);
                tma_load_2d_fn(&tma_o, &mbar[1], smem_o, 0, bhs_base);
                tma_load_2d_fn(&tma_o, &mbar[1], smem_o + 4096, 64, bhs_base);
                
                mbarrier_arrive_and_expect_tx_fn(&mbar[4], 8192);
                tma_load_2d_fn(&tma_p, &mbar[4], smem_p, s_base, bhs_base);
            }
            mbarrier_wait_fn(&mbar[1], phase_do);
            phase_do ^= 1;
            mbarrier_wait_fn(&mbar[4], phase_p);
            phase_p ^= 1;
            
            if (threadIdx.x < 64) {
                float sum = 0.0f;
                for (uint32_t col = 0; col < 64; col++) {
                    sum += __bfloat162float(smem_do[threadIdx.x * 64 + col]) * __bfloat162float(smem_o[threadIdx.x * 64 + col]);
                    sum += __bfloat162float(smem_do[threadIdx.x * 64 + col + 4096]) * __bfloat162float(smem_o[threadIdx.x * 64 + col + 4096]);
                }
                smem_rowsum_P_D[threadIdx.x] = sum;
            }
            __syncthreads();
            
            // D = dO @ V^T
            for (uint32_t k = 0; k < 128; k += 16) {
                uint64_t desc_a = make_smem_desc_sm100_fn(smem_do + k, 0, 1024);
                uint64_t desc_b = make_smem_desc_sm100_fn(smem_v + k, 0, 1024);
                umma_f16_cg2_fn(tmem_d, desc_a, desc_b, idesc, (phase_d == 0) ? 0 : 1);
                phase_d++;
            }
            umma_commit_2sm_fn(&mbar[1]);
            mbarrier_wait_fn(&mbar[1], phase[1]);
            phase[1] ^= 1;
            
            // dP = P * (D - rowsum_P_D)
            for (uint32_t i = 0; i < 64; i += 4) {
                uint32_t r_d0, r_d1, r_d2, r_d3;
                tmem_load_4x_fn(tmem_d + threadIdx.x * 64 + i, &r_d0, &r_d1, &r_d2, &r_d3);
                
                float f_d0 = __uint_as_float(r_d0);
                float f_d1 = __uint_as_float(r_d1);
                float f_d2 = __uint_as_float(r_d2);
                float f_d3 = __uint_as_float(r_d3);
                
                float dp0 = __bfloat162float(smem_p[(threadIdx.x * 64 + i)]) * (f_d0 - smem_rowsum_P_D[threadIdx.x]);
                float dp1 = __bfloat162float(smem_p[(threadIdx.x * 64 + i + 1)]) * (f_d1 - smem_rowsum_P_D[threadIdx.x]);
                float dp2 = __bfloat162float(smem_p[(threadIdx.x * 64 + i + 2)]) * (f_d2 - smem_rowsum_P_D[threadIdx.x]);
                float dp3 = __bfloat162float(smem_p[(threadIdx.x * 64 + i + 3)]) * (f_d3 - smem_rowsum_P_D[threadIdx.x]);
                
                uint32_t p0 = pack_bf16_fn(__float_as_uint(dp0), __float_as_uint(dp1));
                uint32_t p1 = pack_bf16_fn(__float_as_uint(dp2), __float_as_uint(dp3));
                *(uint32_t*)&smem_dp[(threadIdx.x * 64 + i) * 2 + 0] = p0;
                *(uint32_t*)&smem_dp[(threadIdx.x * 64 + i) * 2 + 2] = p1;
            }
            __syncthreads();
            fence_async_shared_fn();
            
            // dK += Q^T @ dP
            for (uint32_t k = 0; k < 64; k += 16) {
                uint64_t desc_q = make_smem_desc_sm100_fn(smem_q + k*64, 8192, 1024);
                uint64_t desc_dp = make_smem_desc_sm100_fn(smem_dp + k*64, 8192, 1024);
                umma_f16_cg2_fn(tmem_dk, desc_q, desc_dp, idesc_dk, (phase_dk == 0) ? 0 : 1);
                phase_dk++;
                
                uint64_t desc_q1 = make_smem_desc_sm100_fn(smem_q + 4096 + k*64, 8192, 1024);
                umma_f16_cg2_fn(tmem_dk + 64, desc_q1, desc_dp, idesc_dk, (phase_dk == 0) ? 0 : 1);
                phase_dk++;
            }
            
            // dV += P^T @ dO
            for (uint32_t k = 0; k < 64; k += 16) {
                uint64_t desc_p = make_smem_desc_sm100_fn(smem_p + k*64, 8192, 1024);
                uint64_t desc_do = make_smem_desc_sm100_fn(smem_do + k*64, 8192, 1024);
                umma_f16_cg2_fn(tmem_dv, desc_p, desc_do, idesc_dv, (phase_dv == 0) ? 0 : 1);
                phase_dv++;
                
                uint64_t desc_do1 = make_smem_desc_sm100_fn(smem_do + 4096 + k*64, 8192, 1024);
                umma_f16_cg2_fn(tmem_dv + 64, desc_p, desc_do1, idesc_dv, (phase_dv == 0) ? 0 : 1);
                phase_dv++;
            }
            umma_commit_2sm_fn(&mbar[1]);
            mbarrier_wait_fn(&mbar[1], phase[1]);
            phase[1] ^= 1;
        }
        
        // Epilogue for dK and dV
        tmem_epilogue_coalesced_4w_fn(dK, smem_out, S, 128, s_base, 0, 64, 64);
        tmem_epilogue_coalesced_4w_fn(dK, smem_out, S, 128, s_base, 64, 64, 64);
        tmem_epilogue_coalesced_4w_fn(dV, smem_out, S, 128, s_base, 0, 64, 64);
        tmem_epilogue_coalesced_4w_fn(dV, smem_out, S, 128, s_base, 64, 64, 64);
    }
    
    __syncthreads();
    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_d, 64);
        tmem_dealloc_fn(tmem_s, 64);
        tmem_dealloc_fn(tmem_dq, 128);
        tmem_dealloc_fn(tmem_dk, 128);
        tmem_dealloc_fn(tmem_dv, 128);
        tmem_dealloc_fn(tmem_dp, 64);
    }
}

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, CUtensorMapDataType dataType, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, dataType, 2, globalAddress, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3); 
    
    CUtensorMap tma_q, tma_k, tma_v, tma_o, tma_do, tma_p;
    CUtensorMapDataType dataType = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    
    create_tma_2d_descriptor_2B(&tma_q, dataType, Q.data_ptr(), d, B*H*S, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_k, dataType, K.data_ptr(), d, B*H*S, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_v, dataType, V.data_ptr(), d, B*H*S, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_o, dataType, O.data_ptr(), d, B*H*S, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_do, dataType, dO.data_ptr(), d, B*H*S, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_p, dataType, P.data_ptr(), S, B*H*S, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    cudaLaunchConfig_t config = {};
    config.blockDim.x = 128;
    config.dynamicSmemBytes = 1024 * 1024; 
    config.stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 2;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    // 1. Compute dQ
    uint32_t gx1 = (B * H * S + 63) / 64;
    config.gridDim = dim3(gx1, 2);
    cudaLaunchKernelEx(&config, mha_bwd_kernel, tma_q, tma_k, tma_v, tma_o, tma_do, tma_p, 
                        static_cast<__nv_bfloat16*>(dQ.data_ptr()), nullptr, nullptr, 
                        static_cast<const float*>(L.data_ptr()), S, 1);
    CUDA_CHECK(cudaGetLastError());
    
    // 2. Compute dK and dV
    uint32_t gx2 = (S + 63) / 64;
    config.gridDim = dim3(gx2, 2);
    cudaLaunchKernelEx(&config, mha_bwd_kernel, tma_q, tma_k, tma_v, tma_o, tma_do, tma_p, 
                        nullptr, static_cast<__nv_bfloat16*>(dK.data_ptr()), static_cast<__nv_bfloat16*>(dV.data_ptr()), 
                        static_cast<const float*>(L.data_ptr()), S, 2);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaStreamSynchronize(config.stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_mha_bwd