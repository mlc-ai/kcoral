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

namespace mha_bwd_d128_causal {

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

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_fn(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])) : "memory");
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

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, uint32_t smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"(smem), "l"((uint64_t)d), "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void umma_commit_cg1_fn(uint64_t* bar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "l"(bar) : "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(uint32_t addr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
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

__device__ __forceinline__ uint64_t advance_desc(uint64_t desc, uint32_t offset_bytes) {
    uint32_t addr = (desc & 0x3FFFF) << 4;
    addr += offset_bytes;
    desc &= ~0x3FFFFull; 
    desc |= (addr >> 4);
    uint32_t base_offset = (addr >> 7) & 7;
    desc &= ~(7ull << 49);
    desc |= ((uint64_t)base_offset << 49);
    return desc;
}

__device__ __forceinline__ void umma_64x64(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, bool accumulate) {
    for (int k = 0; k < 4; ++k) {
        if (threadIdx.x == 0) {
            uint32_t acc = (k == 0 && !accumulate) ? 0 : 1;
            asm volatile(
                "{\n.reg .pred p;\n"
                "setp.ne.b32 p, %4, 0;\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(acc));
        }
        desc_a = advance_desc(desc_a, (idesc & (1<<15)) ? 2048 : 32);
        desc_b = advance_desc(desc_b, (idesc & (1<<16)) ? 2048 : 32);
    }
}

__device__ __forceinline__ uint32_t swizzle_128B(uint32_t row, uint32_t col_bytes) {
    return (col_bytes & ~127) | ((col_bytes & 127) ^ ((row & 7) * 16));
}

__device__ __forceinline__ void tmem_store_bf16_64x64(uint32_t tmem_base, uint8_t* smem_base) {
    for (uint32_t col = 0; col < 64; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_base + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        uint32_t warp_id = threadIdx.x / 32;
        if (warp_id < 2) {
            uint32_t row = warp_id * 32 + (threadIdx.x % 32);
            uint32_t p0 = pack_bf16_fn(r0, r1);
            uint32_t p1 = pack_bf16_fn(r2, r3);
            uint32_t offset = swizzle_128B(row, col * 2);
            uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_base + row * 128 + offset);
            asm volatile("st.shared.v2.b32 [%0], {%1, %2};" :: "r"(addr), "r"(p0), "r"(p1) : "memory");
        }
    }
}

__device__ __forceinline__ void atomic_add_global(uint8_t* smem_base1, uint8_t* smem_base2, __nv_bfloat162* global_ptr, uint32_t row_start, uint32_t S, uint32_t head, uint32_t batch, uint32_t H) {
    for (int idx = threadIdx.x; idx < 64 * 32; idx += 128) {
        uint32_t row = idx / 32;
        uint32_t col_u32 = idx % 32;
        uint32_t offset = row * 128 + swizzle_128B(row, col_u32 * 4);
        uint32_t val = *reinterpret_cast<uint32_t*>(smem_base1 + offset);
        if (row_start + row < S) {
            uint64_t global_idx = ((uint64_t)(batch * H + head) * S + row_start + row) * 64 + col_u32;
            atomicAdd(&global_ptr[global_idx], *reinterpret_cast<__nv_bfloat162*>(&val));
        }
    }
    for (int idx = threadIdx.x; idx < 64 * 32; idx += 128) {
        uint32_t row = idx / 32;
        uint32_t col_u32 = idx % 32;
        uint32_t offset = row * 128 + swizzle_128B(row, col_u32 * 4);
        uint32_t val = *reinterpret_cast<uint32_t*>(smem_base2 + offset);
        if (row_start + row < S) {
            uint64_t global_idx = ((uint64_t)(batch * H + head) * S + row_start + row) * 64 + 32 + col_u32;
            atomicAdd(&global_ptr[global_idx], *reinterpret_cast<__nv_bfloat162*>(&val));
        }
    }
}

__device__ __forceinline__ void store_global(uint8_t* smem_base1, uint8_t* smem_base2, __nv_bfloat162* global_ptr, uint32_t row_start, uint32_t S, uint32_t head, uint32_t batch, uint32_t H) {
    for (int idx = threadIdx.x; idx < 64 * 32; idx += 128) {
        uint32_t row = idx / 32;
        uint32_t col_u32 = idx % 32;
        uint32_t offset = row * 128 + swizzle_128B(row, col_u32 * 4);
        uint32_t val = *reinterpret_cast<uint32_t*>(smem_base1 + offset);
        if (row_start + row < S) {
            uint64_t global_idx = ((uint64_t)(batch * H + head) * S + row_start + row) * 64 + col_u32;
            global_ptr[global_idx] = *reinterpret_cast<__nv_bfloat162*>(&val);
        }
    }
    for (int idx = threadIdx.x; idx < 64 * 32; idx += 128) {
        uint32_t row = idx / 32;
        uint32_t col_u32 = idx % 32;
        uint32_t offset = row * 128 + swizzle_128B(row, col_u32 * 4);
        uint32_t val = *reinterpret_cast<uint32_t*>(smem_base2 + offset);
        if (row_start + row < S) {
            uint64_t global_idx = ((uint64_t)(batch * H + head) * S + row_start + row) * 64 + 32 + col_u32;
            global_ptr[global_idx] = *reinterpret_cast<__nv_bfloat162*>(&val);
        }
    }
}

extern __shared__ __align__(128) uint8_t smem_pool[];
#define SMEM_PTR(off) (smem_pool + (off))
#define SMEM_ADDR(off) ((uint32_t)__cvta_generic_to_shared(SMEM_PTR(off)))

__global__ void mha_bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* __restrict__ L_ptr,
    __nv_bfloat16* __restrict__ dQ_ptr,
    __nv_bfloat16* __restrict__ dK_ptr,
    __nv_bfloat16* __restrict__ dV_ptr,
    uint32_t S, uint32_t H
) {
    uint32_t j = blockIdx.x;
    uint32_t head = blockIdx.y;
    uint32_t batch = blockIdx.z;

    if (j * 64 >= S) return;
    
    uint32_t off_K1 = 0;
    uint32_t off_K2 = off_K1 + 8192;
    uint32_t off_V1 = off_K2 + 8192;
    uint32_t off_V2 = off_V1 + 8192;
    uint32_t off_Q1 = off_V2 + 8192;
    uint32_t off_Q2 = off_Q1 + 8192;
    uint32_t off_dO1 = off_Q2 + 8192;
    uint32_t off_dO2 = off_dO1 + 8192;
    uint32_t off_O1 = off_dO2 + 8192;
    uint32_t off_O2 = off_O1 + 8192;
    uint32_t off_P  = off_O2 + 8192;
    uint32_t off_dS = off_P + 8192;
    uint32_t off_dQ1 = off_dS + 8192;
    uint32_t off_dQ2 = off_dQ1 + 8192;
    uint32_t off_D  = off_dQ2 + 8192;
    uint32_t off_L  = off_D + 256;
    uint32_t off_mbar = off_L + 256;
    uint32_t off_mbar_umma = off_mbar + 8;

    uint64_t* mbar = (uint64_t*)SMEM_PTR(off_mbar);
    uint64_t* mbar_umma = (uint64_t*)SMEM_PTR(off_mbar_umma);

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar, 128);
        init_smem_barrier_fn(mbar_umma, 1);
    }
    __syncthreads();

    uint32_t phase = 0;
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar, 4 * 8192);
        tma_load_4d_fn(&tma_K, mbar, SMEM_ADDR(off_K1), 0, j * 64, head, batch);
        tma_load_4d_fn(&tma_K, mbar, SMEM_ADDR(off_K2), 64, j * 64, head, batch);
        tma_load_4d_fn(&tma_V, mbar, SMEM_ADDR(off_V1), 0, j * 64, head, batch);
        tma_load_4d_fn(&tma_V, mbar, SMEM_ADDR(off_V2), 64, j * 64, head, batch);
    } else {
        mbarrier_arrive_fn(mbar);
    }
    mbarrier_wait_fn(mbar, phase);
    phase ^= 1;

    __shared__ uint32_t tmem_addr_smem;
    if (threadIdx.x / 32 == 0) {
        tmem_alloc_cg1_fn(&tmem_addr_smem, 512);
    }
    __syncthreads();
    
    uint32_t tmem_addr = tmem_addr_smem;
    uint32_t tmem_dK1 = tmem_addr;
    uint32_t tmem_dK2 = tmem_addr + 64;
    uint32_t tmem_dV1 = tmem_addr + 128;
    uint32_t tmem_dV2 = tmem_addr + 192;
    uint32_t tmem_dQ1 = tmem_addr + 256;
    uint32_t tmem_dQ2 = tmem_addr + 320;
    uint32_t tmem_S  = tmem_addr + 384;
    uint32_t tmem_dP = tmem_addr + 448;

    uint64_t desc_K1 = make_smem_desc_sm100_fn(SMEM_ADDR(off_K1), 1, 1024);
    uint64_t desc_K2 = make_smem_desc_sm100_fn(SMEM_ADDR(off_K2), 1, 1024);
    uint64_t desc_V1 = make_smem_desc_sm100_fn(SMEM_ADDR(off_V1), 1, 1024);
    uint64_t desc_V2 = make_smem_desc_sm100_fn(SMEM_ADDR(off_V2), 1, 1024);
    uint64_t desc_Q1 = make_smem_desc_sm100_fn(SMEM_ADDR(off_Q1), 1, 1024);
    uint64_t desc_Q2 = make_smem_desc_sm100_fn(SMEM_ADDR(off_Q2), 1, 1024);
    uint64_t desc_dO1 = make_smem_desc_sm100_fn(SMEM_ADDR(off_dO1), 1, 1024);
    uint64_t desc_dO2 = make_smem_desc_sm100_fn(SMEM_ADDR(off_dO2), 1, 1024);
    
    uint64_t desc_P = make_smem_desc_sm100_fn(SMEM_ADDR(off_P), 1, 1024);
    uint64_t desc_dS = make_smem_desc_sm100_fn(SMEM_ADDR(off_dS), 1, 1024);
    
    uint64_t desc_K1_MN = make_smem_desc_sm100_fn(SMEM_ADDR(off_K1), 8192, 1024);
    uint64_t desc_K2_MN = make_smem_desc_sm100_fn(SMEM_ADDR(off_K2), 8192, 1024);
    uint64_t desc_Q1_MN = make_smem_desc_sm100_fn(SMEM_ADDR(off_Q1), 8192, 1024);
    uint64_t desc_Q2_MN = make_smem_desc_sm100_fn(SMEM_ADDR(off_Q2), 8192, 1024);
    uint64_t desc_dO1_MN = make_smem_desc_sm100_fn(SMEM_ADDR(off_dO1), 8192, 1024);
    uint64_t desc_dO2_MN = make_smem_desc_sm100_fn(SMEM_ADDR(off_dO2), 8192, 1024);
    uint64_t desc_dS_MN = make_smem_desc_sm100_fn(SMEM_ADDR(off_dS), 8192, 1024);

    uint32_t idesc_KK = make_instr_desc_fn(64, 64);
    uint32_t idesc_KN = make_instr_desc_fn(64, 64) | (1<<16);
    uint32_t idesc_MN = make_instr_desc_fn(64, 64) | (1<<15) | (1<<16);
    uint32_t phase_umma = 0;
    
    uint32_t S_blocks = (S + 63) / 64;

    for (uint32_t i = j; i < S_blocks; ++i) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar, 6 * 8192);
            tma_load_4d_fn(&tma_Q, mbar, SMEM_ADDR(off_Q1), 0, i * 64, head, batch);
            tma_load_4d_fn(&tma_Q, mbar, SMEM_ADDR(off_Q2), 64, i * 64, head, batch);
            tma_load_4d_fn(&tma_dO, mbar, SMEM_ADDR(off_dO1), 0, i * 64, head, batch);
            tma_load_4d_fn(&tma_dO, mbar, SMEM_ADDR(off_dO2), 64, i * 64, head, batch);
            tma_load_4d_fn(&tma_O, mbar, SMEM_ADDR(off_O1), 0, i * 64, head, batch);
            tma_load_4d_fn(&tma_O, mbar, SMEM_ADDR(off_O2), 64, i * 64, head, batch);
        } else {
            mbarrier_arrive_fn(mbar);
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;

        if (threadIdx.x < 64) {
            uint32_t g_row = i * 64 + threadIdx.x;
            float l_val = (g_row < S) ? L_ptr[(uint64_t)(batch * H + head) * S + g_row] : 0.0f;
            *reinterpret_cast<float*>(SMEM_PTR(off_L) + threadIdx.x * 4) = l_val;
            
            float sum = 0.0f;
            if (g_row < S) {
                for (uint32_t c = 0; c < 64; c += 8) {
                    uint32_t offset = swizzle_128B(threadIdx.x, c * 2);
                    uint4 do_v = *reinterpret_cast<uint4*>(SMEM_PTR(off_dO1) + threadIdx.x * 128 + offset);
                    uint4 o_v  = *reinterpret_cast<uint4*>(SMEM_PTR(off_O1) + threadIdx.x * 128 + offset);
                    uint32_t* do_arr = (uint32_t*)&do_v;
                    uint32_t* o_arr = (uint32_t*)&o_v;
                    for (int k = 0; k < 4; ++k) {
                        float2 a = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&do_arr[k]));
                        float2 b = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&o_arr[k]));
                        sum += a.x * b.x + a.y * b.y;
                    }
                }
                for (uint32_t c = 0; c < 64; c += 8) {
                    uint32_t offset = swizzle_128B(threadIdx.x, c * 2);
                    uint4 do_v = *reinterpret_cast<uint4*>(SMEM_PTR(off_dO2) + threadIdx.x * 128 + offset);
                    uint4 o_v  = *reinterpret_cast<uint4*>(SMEM_PTR(off_O2) + threadIdx.x * 128 + offset);
                    uint32_t* do_arr = (uint32_t*)&do_v;
                    uint32_t* o_arr = (uint32_t*)&o_v;
                    for (int k = 0; k < 4; ++k) {
                        float2 a = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&do_arr[k]));
                        float2 b = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&o_arr[k]));
                        sum += a.x * b.x + a.y * b.y;
                    }
                }
            }
            *reinterpret_cast<float*>(SMEM_PTR(off_D) + threadIdx.x * 4) = sum;
        }
        __syncthreads();

        umma_64x64(tmem_S, desc_K1, desc_Q1, idesc_KK, false);
        umma_64x64(tmem_S, desc_K2, desc_Q2, idesc_KK, true);
        
        umma_64x64(tmem_dP, desc_V1, desc_dO1, idesc_KK, false);
        umma_64x64(tmem_dP, desc_V2, desc_dO2, idesc_KK, true);
        
        if (threadIdx.x == 0) { umma_commit_cg1_fn(mbar_umma); }
        mbarrier_wait_fn(mbar_umma, phase_umma);
        phase_umma ^= 1;

        for (uint32_t col = 0; col < 64; col += 8) {
            uint32_t S_r[8], dP_r[8];
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                         : "=r"(S_r[0]),"=r"(S_r[1]),"=r"(S_r[2]),"=r"(S_r[3]),"=r"(S_r[4]),"=r"(S_r[5]),"=r"(S_r[6]),"=r"(S_r[7]) : "r"(tmem_S + col));
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                         : "=r"(dP_r[0]),"=r"(dP_r[1]),"=r"(dP_r[2]),"=r"(dP_r[3]),"=r"(dP_r[4]),"=r"(dP_r[5]),"=r"(dP_r[6]),"=r"(dP_r[7]) : "r"(tmem_dP + col));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            uint32_t warp_id = threadIdx.x / 32;
            if (warp_id < 2) {
                uint32_t row = warp_id * 32 + (threadIdx.x % 32);
                uint32_t P_p[4], dS_p[4];
                for (int k = 0; k < 4; ++k) {
                    float L0 = *reinterpret_cast<float*>(SMEM_PTR(off_L) + (col + k*2) * 4);
                    float D0 = *reinterpret_cast<float*>(SMEM_PTR(off_D) + (col + k*2) * 4);
                    float L1 = *reinterpret_cast<float*>(SMEM_PTR(off_L) + (col + k*2 + 1) * 4);
                    float D1 = *reinterpret_cast<float*>(SMEM_PTR(off_D) + (col + k*2 + 1) * 4);
                    
                    uint32_t pos_Q0 = i * 64 + col + k*2;
                    uint32_t pos_Q1 = i * 64 + col + k*2 + 1;
                    uint32_t pos_K  = j * 64 + row;
                    
                    float p0 = 0, ds0 = 0, p1 = 0, ds1 = 0;
                    if (pos_Q0 >= pos_K && pos_Q0 < S && pos_K < S) {
                        p0 = expf(__uint_as_float(S_r[k*2]) * 0.0883883476f - L0);
                        ds0 = p0 * (__uint_as_float(dP_r[k*2]) - D0) * 0.0883883476f;
                    }
                    if (pos_Q1 >= pos_K && pos_Q1 < S && pos_K < S) {
                        p1 = expf(__uint_as_float(S_r[k*2+1]) * 0.0883883476f - L1);
                        ds1 = p1 * (__uint_as_float(dP_r[k*2+1]) - D1) * 0.0883883476f;
                    }
                    P_p[k] = pack_bf16_fn(__float_as_uint(p0), __float_as_uint(p1));
                    dS_p[k] = pack_bf16_fn(__float_as_uint(ds0), __float_as_uint(ds1));
                }
                uint32_t offset = swizzle_128B(row, col * 2);
                uint32_t addr_P = SMEM_ADDR(off_P + row * 128 + offset);
                uint32_t addr_dS = SMEM_ADDR(off_dS + row * 128 + offset);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr_P), "r"(P_p[0]), "r"(P_p[1]), "r"(P_p[2]), "r"(P_p[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(addr_dS), "r"(dS_p[0]), "r"(dS_p[1]), "r"(dS_p[2]), "r"(dS_p[3]) : "memory");
            }
        }
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

        bool accum_kV = (i > j);
        
        umma_64x64(tmem_dV1, desc_P, desc_dO1_MN, idesc_KN, accum_kV);
        umma_64x64(tmem_dV2, desc_P, desc_dO2_MN, idesc_KN, accum_kV);
        
        umma_64x64(tmem_dK1, desc_dS, desc_Q1_MN, idesc_KN, accum_kV);
        umma_64x64(tmem_dK2, desc_dS, desc_Q2_MN, idesc_KN, accum_kV);
        
        umma_64x64(tmem_dQ1, desc_dS_MN, desc_K1_MN, idesc_MN, false);
        umma_64x64(tmem_dQ2, desc_dS_MN, desc_K2_MN, idesc_MN, false);
        
        if (threadIdx.x == 0) { umma_commit_cg1_fn(mbar_umma); }
        mbarrier_wait_fn(mbar_umma, phase_umma);
        phase_umma ^= 1;
        
        tmem_store_bf16_64x64(tmem_dQ1, SMEM_PTR(off_dQ1));
        tmem_store_bf16_64x64(tmem_dQ2, SMEM_PTR(off_dQ2));
        __syncthreads();
        atomic_add_global(SMEM_PTR(off_dQ1), SMEM_PTR(off_dQ2), (__nv_bfloat162*)dQ_ptr, i * 64, S, head, batch, H);
        __syncthreads();
    }
    
    tmem_store_bf16_64x64(tmem_dK1, SMEM_PTR(off_dQ1));
    tmem_store_bf16_64x64(tmem_dK2, SMEM_PTR(off_dQ2));
    __syncthreads();
    store_global(SMEM_PTR(off_dQ1), SMEM_PTR(off_dQ2), (__nv_bfloat162*)dK_ptr, j * 64, S, head, batch, H);
    __syncthreads();
    
    tmem_store_bf16_64x64(tmem_dV1, SMEM_PTR(off_dQ1));
    tmem_store_bf16_64x64(tmem_dV2, SMEM_PTR(off_dQ2));
    __syncthreads();
    store_global(SMEM_PTR(off_dQ1), SMEM_PTR(off_dQ2), (__nv_bfloat162*)dV_ptr, j * 64, S, head, batch, H);
    
    __syncthreads();
    if (threadIdx.x / 32 == 0) {
        tmem_dealloc_cg1_fn(tmem_addr, 512);
    }
}

CUresult create_tma_4d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t d_dim, uint64_t s_dim, uint64_t h_dim, uint64_t b_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim) {
    cuuint64_t globalDim[4] = {d_dim, s_dim, h_dim, b_dim};
    cuuint64_t globalStrides[3] = {d_dim * 2, d_dim * s_dim * 2, d_dim * s_dim * h_dim * 2};
    cuuint32_t boxDim[4] = {smem_inner_dim, smem_outer_dim, 1, 1};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, globalAddress, globalDim, globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L, tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    uint32_t B = Q.size(0);
    uint32_t H = Q.size(1);
    uint32_t S = Q.size(2);
    uint32_t d = Q.size(3);
    
    CUDA_CHECK(cudaMemset(dQ.data_ptr(), 0, B * H * S * d * 2));
    CUDA_CHECK(cudaMemset(dK.data_ptr(), 0, B * H * S * d * 2));
    CUDA_CHECK(cudaMemset(dV.data_ptr(), 0, B * H * S * d * 2));
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O, tma_dO;
    create_tma_4d_descriptor_2B(&tma_Q, Q.data_ptr(), d, S, H, B, 64, 64);
    create_tma_4d_descriptor_2B(&tma_K, K.data_ptr(), d, S, H, B, 64, 64);
    create_tma_4d_descriptor_2B(&tma_V, V.data_ptr(), d, S, H, B, 64, 64);
    create_tma_4d_descriptor_2B(&tma_O, O.data_ptr(), d, S, H, B, 64, 64);
    create_tma_4d_descriptor_2B(&tma_dO, dO.data_ptr(), d, S, H, B, 64, 64);
    
    dim3 grid((S + 63) / 64, H, B);
    dim3 block(128);
    int smem_size = 118 * 1024;
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUDA_CHECK(cudaFuncSetAttribute((const void*)mha_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    mha_bwd_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, tma_O, tma_dO,
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        S, H
    );
    CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128_causal::run);

}