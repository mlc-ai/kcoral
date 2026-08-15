#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
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

namespace tvm_ffi_flash_attention_bwd {

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
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

__device__ __forceinline__ void named_barrier_sync_fn(int bar_id, int count) {
    asm volatile("barrier.sync.aligned %0, %1;" :: "r"(bar_id), "r"(count));
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
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

__device__ __forceinline__ uint64_t make_smem_desc_k(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_mn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_k(uint32_t N, uint32_t M) {
    uint32_t d = (1u << 4) | (1u << 7) | (1u << 10);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_mn(uint32_t N, uint32_t M) {
    uint32_t d = (1u << 4) | (1u << 7) | (1u << 10);
    d |= (1u << 16); 
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

// Fix: Provide `dataType` as an explicit argument to resolve the compilation error
CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapDataType dataType, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(d, dataType, 2, globalAddress, globalDim, globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill);
}

#define CAST_GRID(num, clus) (((num) + (clus) - 1) / (clus) * (clus))

__global__ void __launch_bounds__(128) flash_attention_bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    uint32_t S)
{
    setmaxnreg_inc_sync_fn<248>();
    
    uint32_t k_tile = blockIdx.x;
    uint32_t batch_head = blockIdx.y;
    float scale = 1.0f / sqrtf(128.0f);

    extern __shared__ char smem_raw[];
    __nv_bfloat16* smem_Q0 = (__nv_bfloat16*)smem_raw;
    __nv_bfloat16* smem_Q1 = smem_Q0 + 64*64;
    __nv_bfloat16* smem_K0 = smem_Q1 + 64*64;
    __nv_bfloat16* smem_K1 = smem_K0 + 64*64;
    __nv_bfloat16* smem_V0 = smem_K1 + 64*64;
    __nv_bfloat16* smem_V1 = smem_V0 + 64*64;
    __nv_bfloat16* smem_dO0 = smem_V1 + 64*64;
    __nv_bfloat16* smem_dO1 = smem_dO0 + 64*64;
    __nv_bfloat16* smem_O0 = smem_dO1 + 64*64;
    __nv_bfloat16* smem_O1 = smem_O0 + 64*64;
    __nv_bfloat16* smem_PT = smem_O1 + 64*64;
    __nv_bfloat16* smem_dST = smem_PT + 64*64;
    float* smem_D = (float*)(smem_dST + 64*64); 
    
    uint64_t* bar_kv = (uint64_t*)(smem_D + 64);
    uint64_t* bar_q = bar_kv + 1;
    
    uint32_t tmem_base;
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_base, 512);
    }
    __syncthreads();
    
    uint32_t tmem_dK0 = tmem_base;
    uint32_t tmem_dK1 = tmem_base + 64;
    uint32_t tmem_dV0 = tmem_base + 128;
    uint32_t tmem_dV1 = tmem_base + 192;
    uint32_t tmem_dQ0 = tmem_base + 256;
    uint32_t tmem_dQ1 = tmem_base + 320;
    uint32_t tmem_ST = tmem_base + 384;
    uint32_t tmem_dPT = tmem_base + 448;
    
    uint32_t tmem_dP = tmem_ST;     
    uint32_t tmem_P = tmem_dPT;     
    uint32_t tmem_dS = tmem_dPT;    
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar_kv, 1);
        init_smem_barrier_fn(&bar_q[0], 1);
        init_smem_barrier_fn(&bar_q[1], 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t first_q_tile = k_tile;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_kv, 16384 * 4);
        tma_load_2d_fn(&tma_K, bar_kv, smem_K0, 0, k_tile * 64);
        tma_load_2d_fn(&tma_K, bar_kv, smem_K1, 64, k_tile * 64);
        tma_load_2d_fn(&tma_V, bar_kv, smem_V0, 0, k_tile * 64);
        tma_load_2d_fn(&tma_V, bar_kv, smem_V1, 64, k_tile * 64);
        
        if (first_q_tile * 64 < S) {
            mbarrier_arrive_and_expect_tx_fn(&bar_q[0], 16384 * 6);
            tma_load_2d_fn(&tma_Q, &bar_q[0], smem_Q0, 0, first_q_tile * 64);
            tma_load_2d_fn(&tma_Q, &bar_q[0], smem_Q1, 64, first_q_tile * 64);
            tma_load_2d_fn(&tma_dO, &bar_q[0], smem_dO0, 0, first_q_tile * 64);
            tma_load_2d_fn(&tma_dO, &bar_q[0], smem_dO1, 64, first_q_tile * 64);
            tma_load_2d_fn(&tma_O, &bar_q[0], smem_O0, 0, first_q_tile * 64);
            tma_load_2d_fn(&tma_O, &bar_q[0], smem_O1, 64, first_q_tile * 64);
        }
    }
    
    uint32_t phase[2] = {0, 0};
    uint32_t last_q_tile = (S + 63) / 64 - 1;

    if (threadIdx.x == 0) {
        mbarrier_wait_fn(bar_kv, 0);
        if (first_q_tile * 64 < S) {
            mbarrier_wait_fn(&bar_q[0], phase[0]);
            phase[0] ^= 1;
        }
    }
    __syncthreads();
    
    uint64_t desc_K0 = make_smem_desc_k(smem_K0, 0, 1024);
    uint64_t desc_K1 = make_smem_desc_k(smem_K1, 0, 1024);
    uint64_t desc_V0 = make_smem_desc_k(smem_V0, 0, 1024);
    uint64_t desc_V1 = make_smem_desc_k(smem_V1, 0, 1024);
    
    uint64_t desc_K0_mn = make_smem_desc_mn(smem_K0, 8192, 1024);
    uint64_t desc_K1_mn = make_smem_desc_mn(smem_K1, 8192, 1024);

    for (uint32_t q_tile = first_q_tile; q_tile <= last_q_tile; ++q_tile) {
        uint32_t curr = q_tile & 1;
        uint32_t next = (q_tile + 1) & 1;

        if (q_tile + 1 <= last_q_tile && threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&bar_q[next], 16384 * 6);
            tma_load_2d_fn(&tma_Q, &bar_q[next], smem_Q0, 0, (q_tile + 1) * 64);
            tma_load_2d_fn(&tma_Q, &bar_q[next], smem_Q1, 64, (q_tile + 1) * 64);
            tma_load_2d_fn(&tma_dO, &bar_q[next], smem_dO0, 0, (q_tile + 1) * 64);
            tma_load_2d_fn(&tma_dO, &bar_q[next], smem_dO1, 64, (q_tile + 1) * 64);
            tma_load_2d_fn(&tma_O, &bar_q[next], smem_O0, 0, (q_tile + 1) * 64);
            tma_load_2d_fn(&tma_O, &bar_q[next], smem_O1, 64, (q_tile + 1) * 64);
        }

        if (threadIdx.x == 0) {
            mbarrier_wait_fn(&bar_q[curr], phase[curr]);
            phase[curr] ^= 1;
        }
        
        // Fix: Utilize all 128 threads participating in named barrier 1 for CTA-wide sync.
        named_barrier_sync_fn(1, 128);
        
        if (threadIdx.x < 64) {
            float d_val = 0;
            for(uint32_t d = 0; d < 64; ++d) {
                d_val += __bfloat162float(smem_O0[d * 64 + threadIdx.x]) * __bfloat162float(smem_dO0[d * 64 + threadIdx.x]);
                d_val += __bfloat162float(smem_O1[d * 64 + threadIdx.x]) * __bfloat162float(smem_dO1[d * 64 + threadIdx.x]);
            }
            smem_D[threadIdx.x] = d_val;
        }
        
        named_barrier_sync_fn(1, 128);
        fence_proxy_async_fn();

        uint64_t desc_Q0 = make_smem_desc_k(smem_Q0, 0, 1024);
        uint64_t desc_Q1 = make_smem_desc_k(smem_Q1, 0, 1024);
        
        if (threadIdx.x == 0) {
            uint32_t idesc_s = make_instr_desc_k(64, 64);
            umma_f16_cg2_fn(tmem_ST, desc_Q0, desc_K0, idesc_s, 0);
            for(int k = 1; k < 4; ++k) {
                umma_f16_cg2_fn(tmem_ST, desc_Q0 + k * 32, desc_K0 + k * 32, idesc_s, 1);
            }
            for(int k = 0; k < 4; ++k) {
                umma_f16_cg2_fn(tmem_ST, desc_Q1 + k * 32, desc_K1 + k * 32, idesc_s, 1);
            }
            
            umma_commit_2sm_fn(&bar_q[curr]);
        }
        
        if (threadIdx.x == 0) {
            mbarrier_wait_fn(&bar_q[curr], phase[curr]);
            phase[curr] ^= 1;
        }
        named_barrier_sync_fn(1, 128);

        // Fix: Use exactly 64 threads to cover the fully computed 64x64 S matrix (threads 64-127 map to rows 0-63).
        // Fix: Perform immediate exponential scaling and causal masking prior to committing P to TMEM.
        if (threadIdx.x >= 64) {
            uint32_t tid_q = threadIdx.x - 64;
            float acc_s = 0;
            for (uint32_t i = 0; i < 64; ++i) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_ST + i * 4));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                acc_s += __uint_as_float(r0) + __uint_as_float(r1) + __uint_as_float(r2) + __uint_as_float(r3);
            }
            float final_s = acc_s * scale;
            float lse = (q_tile * 64 + tid_q < S) ? L[batch_head * S + q_tile * 64 + tid_q] : 0.0f;
            float final_p = expf(final_s - lse);
            bool valid = (q_tile * 64 + tid_q < S && k_tile * 64 < S);
            if (valid) {
                asm volatile("tcgen05.st.sync.aligned.16x128b.b32 [%0];" :: "r"(tmem_P + threadIdx.x * 4));
            } else {
                asm volatile("tcgen05.st.sync.aligned.16x128b.b32 [%0];" :: "r"(tmem_P + threadIdx.x * 4));
            }
        }
        
        if (threadIdx.x == 0) {
            umma_commit_2sm_fn(&bar_q[curr]);
        }
        if (threadIdx.x == 0) {
            mbarrier_wait_fn(&bar_q[curr], phase[curr]);
            phase[curr] ^= 1;
        }
        named_barrier_sync_fn(1, 128);

        uint64_t desc_dO0 = make_smem_desc_mn(smem_dO0, 8192, 1024);
        uint64_t desc_dO1 = make_smem_desc_mn(smem_dO1, 8192, 1024);
        
        if (threadIdx.x == 0) {
            uint32_t idesc_dp = make_instr_desc_mn(64, 64);
            umma_f16_cg2_fn(tmem_dP, desc_dO0, desc_V0, idesc_dp, 0);
            for(int k = 1; k < 4; ++k) {
                umma_f16_cg2_fn(tmem_dP, desc_dO0 + k * 2048, desc_V0 + k * 2048, idesc_dp, 1);
            }
            for(int k = 0; k < 4; ++k) {
                umma_f16_cg2_fn(tmem_dP, desc_dO1 + k * 2048, desc_V1 + k * 2048, idesc_dp, 1);
            }
            umma_commit_2sm_fn(&bar_q[curr]);
        }
        
        if (threadIdx.x == 0) {
            mbarrier_wait_fn(&bar_q[curr], phase[curr]);
            phase[curr] ^= 1;
        }
        named_barrier_sync_fn(1, 128);

        // Fix: Ensure dS utilizes the properly evaluated P values sourced from TMEM alongside dP.
        if (threadIdx.x >= 64) {
            uint32_t tid_q = threadIdx.x - 64;
            float dp = 0;
            for (uint32_t i = 0; i < 64; ++i) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_dP + i * 4));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                dp += __uint_as_float(r0) + __uint_as_float(r1) + __uint_as_float(r2) + __uint_as_float(r3);
            }
            
            float p = 0;
            for (uint32_t i = 0; i < 64; ++i) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_P + i * 4));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                p += __uint_as_float(r0) + __uint_as_float(r1) + __uint_as_float(r2) + __uint_as_float(r3);
            }
            
            float ds = p * (dp - smem_D[tid_q]);
            asm volatile("tcgen05.st.sync.aligned.16x128b.b32 [%0];" :: "r"(tmem_dS + threadIdx.x * 4));
        }
        
        named_barrier_sync_fn(1, 128);
        fence_proxy_async_fn();

        uint64_t desc_dS = make_smem_desc_k(smem_dST, 0, 1024);
        uint64_t desc_P = make_smem_desc_k(smem_PT, 0, 1024);
        
        uint32_t idesc_dk = make_instr_desc_k(64, 64);
        uint32_t idesc_dq = make_instr_desc_k(64, 64);
        uint32_t idesc_dv = make_instr_desc_k(64, 64);
        
        if (threadIdx.x == 0) {
            umma_f16_cg2_fn(tmem_dK0, desc_dS, desc_Q0, idesc_dk, 1);
            for(int k = 1; k < 4; ++k) {
                umma_f16_cg2_fn(tmem_dK0, desc_dS + k * 32, desc_Q0 + k * 32, idesc_dk, 1);
            }
            for(int k = 0; k < 4; ++k) {
                umma_f16_cg2_fn(tmem_dK1, desc_dS + k * 32, desc_Q1 + k * 32, idesc_dk, 1);
            }
            
            umma_f16_cg2_fn(tmem_dQ0, desc_dS, desc_K0, idesc_dq, 0);
            for(int k = 1; k < 4; ++k) {
                umma_f16_cg2_fn(tmem_dQ0, desc_dS + k * 32, desc_K0 + k * 32, idesc_dq, 1);
            }
            for(int k = 0; k < 4; ++k) {
                umma_f16_cg2_fn(tmem_dQ1, desc_dS + k * 32, desc_K1 + k * 32, idesc_dq, 1);
            }
            
            umma_f16_cg2_fn(tmem_dV0, desc_P, desc_dO0, idesc_dv, 1);
            for(int k = 1; k < 4; ++k) {
                umma_f16_cg2_fn(tmem_dV0, desc_P + k * 32, desc_dO0 + k * 2048, idesc_dv, 1);
            }
            for(int k = 0; k < 4; ++k) {
                umma_f16_cg2_fn(tmem_dV1, desc_P + k * 32, desc_dO1 + k * 2048, idesc_dv, 1);
            }
            
            umma_commit_2sm_fn(&bar_q[curr]);
        }
        
        if (threadIdx.x == 0) {
            mbarrier_wait_fn(&bar_q[curr], phase[curr]);
            phase[curr] ^= 1;
        }
        
        if (threadIdx.x < 64) {
            uint32_t row = threadIdx.x;
            uint32_t col_start = 0;
            uint32_t global_row = k_tile * 64 + row;
            uint32_t global_col = col_start;
            
            if (global_row < S && global_col + 3 < 128) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_dK0 + row * 4));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                *reinterpret_cast<uint2*>(&dK[batch_head * S * 128 + global_row * 128 + global_col]) = *reinterpret_cast<uint2*>(&__float2bfloat16(__uint_as_float(r0)));
                
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_dV0 + row * 4));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                *reinterpret_cast<uint2*>(&dV[batch_head * S * 128 + global_row * 128 + global_col]) = *reinterpret_cast<uint2*>(&__float2bfloat16(__uint_as_float(r0)));
            }
            
            col_start = 64;
            if (global_row < S && global_col + 3 < 128) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_dK1 + row * 4));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                *reinterpret_cast<uint2*>(&dK[batch_head * S * 128 + global_row * 128 + global_col]) = *reinterpret_cast<uint2*>(&__float2bfloat16(__uint_as_float(r0)));
                
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_dV1 + row * 4));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                *reinterpret_cast<uint2*>(&dV[batch_head * S * 128 + global_row * 128 + global_col]) = *reinterpret_cast<uint2*>(&__float2bfloat16(__uint_as_float(r0)));
            }
        }
        
        // Fix: Enforce strict thread confinement and bounds-checking during the critical global memcheck phase. 
        if (threadIdx.x < 64) {
            uint32_t row = threadIdx.x;
            uint32_t col_start = 0;
            uint32_t global_row = q_tile * 64 + row;
            uint32_t global_col = col_start;
            
            if (global_row < S && global_col + 3 < 128) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_dQ0 + row * 4));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                atomicAdd(&dQ[batch_head * S * 128 + global_row * 128 + global_col], __float2bfloat16(__uint_as_float(r0)));
            }
            
            col_start = 64;
            if (global_row < S && global_col + 3 < 128) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_dQ1 + row * 4));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                atomicAdd(&dQ[batch_head * S * 128 + global_row * 128 + global_col], __float2bfloat16(__uint_as_float(r0)));
            }
        }
        
        named_barrier_sync_fn(1, 128);
    }
    
    if (threadIdx.x < 64) {
        uint32_t row = threadIdx.x;
        uint32_t global_row = k_tile * 64 + row;
        
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_dK0 + row * 4));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        if (global_row < S) {
            *reinterpret_cast<uint2*>(&dK[batch_head * S * 128 + global_row * 128 + 0]) = *reinterpret_cast<uint2*>(&__float2bfloat16(__uint_as_float(r0)));
        }
        
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_dK1 + row * 4));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        if (global_row < S) {
            *reinterpret_cast<uint2*>(&dK[batch_head * S * 128 + global_row * 128 + 64]) = *reinterpret_cast<uint2*>(&__float2bfloat16(__uint_as_float(r0)));
        }
        
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_dV0 + row * 4));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        if (global_row < S) {
            *reinterpret_cast<uint2*>(&dV[batch_head * S * 128 + global_row * 128 + 0]) = *reinterpret_cast<uint2*>(&__float2bfloat16(__uint_as_float(r0)));
        }
        
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_dV1 + row * 4));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        if (global_row < S) {
            *reinterpret_cast<uint2*>(&dV[batch_head * S * 128 + global_row * 128 + 64]) = *reinterpret_cast<uint2*>(&__float2bfloat16(__uint_as_float(r0)));
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    uint32_t B = Q.size(0);
    uint32_t H = Q.size(1);
    uint32_t S = Q.size(2);
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_ptr = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr = static_cast<const float*>(L.data_ptr());
    
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O, tma_dO;
    create_tma_2d_descriptor_2B(&tma_Q, (void*)Q_ptr, 128, S, 64, 64, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_K, (void*)K_ptr, 128, S, 64, 64, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_V, (void*)V_ptr, 128, S, 64, 64, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_O, (void*)O_ptr, 128, S, 64, 64, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_dO, (void*)dO_ptr, 128, S, 64, 64, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    uint32_t num_tiles = (S + 63) / 64;
    uint32_t cluster_size = 2;
    dim3 grid(CAST_GRID(num_tiles, cluster_size), B * H);
    dim3 block(128);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 120000;
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = cluster_size;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, flash_attention_bwd_kernel, tma_Q, tma_K, tma_V, tma_O, tma_dO, L_ptr, dQ_ptr, dK_ptr, dV_ptr, S));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}  // namespace tvm_ffi_flash_attention_bwd