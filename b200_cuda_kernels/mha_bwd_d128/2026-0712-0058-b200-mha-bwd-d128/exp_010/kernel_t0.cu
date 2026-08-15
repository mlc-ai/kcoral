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

namespace tvm_ffi_mha_bwd {

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

__device__ __forceinline__ void umma_f16_cg1_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
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

__device__ __forceinline__ uint32_t pack_bf16_to_uint32(__nv_bfloat16 a, __nv_bfloat16 b) {
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&a)),
          "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}

__device__ __forceinline__ void st_shared_128_fn(uint32_t addr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
}

__device__ __forceinline__ void store_to_smem_64x64_swizzled(char* smem, __nv_bfloat16 arr[64 * 64]) {
    for (int r = 0; r < 64; ++r) {
        for (int c = 0; c < 64; ++c) {
            int x = c / 8;
            int x_rem = c % 8;
            int chunk_x = (r % 8) ^ x;
            int swizzled_c = chunk_x * 8 + x_rem;
            smem[r * 64 + swizzled_c] = arr[r * 64 + c];
        }
    }
}

__device__ __forceinline__ void store_to_smem_64x64_swizzled_transposed(char* smem, __nv_bfloat16 arr[64 * 64]) {
    for (int r = 0; r < 64; ++r) {
        for (int c = 0; c < 64; ++c) {
            int x = c / 8;
            int x_rem = c % 8;
            int chunk_x = (r % 8) ^ x;
            int swizzled_c = chunk_x * 8 + x_rem;
            // Store element c,r of the original array at location r,c of the swizzled target
            smem[r * 64 + swizzled_c] = arr[c * 64 + r];
        }
    }
}

__device__ __forceinline__ int swizzle_128B(int r, int c) {
    int x = c / 8;
    int x_rem = c % 8;
    int chunk_x = (r % 8) ^ x;
    return chunk_x * 8 + x_rem;
}

struct SharedStorage {
    char smem_Q0[8192];
    char smem_K0[8192];
    char smem_V0[8192];
    char smem_dO0[8192];
    char smem_O0[8192];
    char smem_Q1[8192];
    char smem_K1[8192];
    char smem_V1[8192];
    char smem_dO1[8192];
    char smem_O1[8192];
    char smem_P[8192];
    char smem_S[8192];
    char smem_dS_bf16[8192];
    float smem_D[64];
    uint64_t mbar_part0;
    uint64_t mbar_part1;
};

__global__ void __launch_bounds__(128) mha_bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q0,
    const __grid_constant__ CUtensorMap tma_Q1,
    const __grid_constant__ CUtensorMap tma_K0,
    const __grid_constant__ CUtensorMap tma_K1,
    const __grid_constant__ CUtensorMap tma_V0,
    const __grid_constant__ CUtensorMap tma_V1,
    const __grid_constant__ CUtensorMap tma_dO0,
    const __grid_constant__ CUtensorMap tma_dO1,
    const __grid_constant__ CUtensorMap tma_O0,
    const __grid_constant__ CUtensorMap tma_O1,
    const __grid_constant__ CUtensorMap tma_dQ0,
    const __grid_constant__ CUtensorMap tma_dQ1,
    const __grid_constant__ CUtensorMap tma_dK0,
    const __grid_constant__ CUtensorMap tma_dK1,
    const __grid_constant__ CUtensorMap tma_dV0,
    const __grid_constant__ CUtensorMap tma_dV1,
    float sq_scale,
    int S,
    int H) 
{
    extern __shared__ char smem_buf[];
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_buf);

    uint32_t* tmem_S_T = new uint32_t();
    uint32_t* tmem_P_T = new uint32_t();
    uint32_t* tmem_dP_T = new uint32_t();
    uint32_t* tmem_D_T = new uint32_t();
    uint32_t* tmem_dS_T = new uint32_t();
    uint32_t* tmem_dV0 = new uint32_t();
    uint32_t* tmem_dV1 = new uint32_t();
    uint32_t* tmem_dK0 = new uint32_t();
    uint32_t* tmem_dK1 = new uint32_t();
    uint32_t* tmem_dQ0 = new uint32_t();
    uint32_t* tmem_dQ1 = new uint32_t();

    if (threadIdx.x == 0) {
        tmem_alloc_fn(tmem_S_T, 128);
        tmem_alloc_fn(tmem_P_T, 128);
        tmem_alloc_fn(tmem_dP_T, 128);
        tmem_alloc_fn(tmem_D_T, 128);
        tmem_alloc_fn(tmem_dS_T, 128);
        tmem_alloc_fn(tmem_dV0, 128);
        tmem_alloc_fn(tmem_dV1, 128);
        tmem_alloc_fn(tmem_dK0, 128);
        tmem_alloc_fn(tmem_dK1, 128);
        tmem_alloc_fn(tmem_dQ0, 128);
        tmem_alloc_fn(tmem_dQ1, 128);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&smem.mbar_part0, 1);
        init_smem_barrier_fn(&smem.mbar_part1, 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();

    int num_steps = S / 64;
    int batch_head_idx = blockIdx.y;
    int b_idx = batch_head_idx / H;
    int h_idx = batch_head_idx % H;
    const float* L = L_global + b_idx * (H * S) + h_idx * S;

    uint32_t phase0 = 0, phase1 = 0;

    // ==========================================
    // Pass 1: compute dQ
    // ==========================================
    int idx_q = blockIdx.x;
    if (idx_q < num_steps) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&smem.mbar_part0, 16384);
            tma_load_2d_fn(&tma_Q0, &smem.mbar_part0, smem.smem_Q0, 0, b_idx * H * S + h_idx * S + idx_q * 64);
            tma_load_2d_fn(&tma_dO0, &smem.mbar_part0, smem.smem_dO0, 0, b_idx * H * S + h_idx * S + idx_q * 64);
            tma_load_2d_fn(&tma_O0, &smem.mbar_part0, smem.smem_O0, 0, b_idx * H * S + h_idx * S + idx_q * 64);
            
            mbarrier_arrive_and_expect_tx_fn(&smem.mbar_part1, 16384);
            tma_load_2d_fn(&tma_Q1, &smem.mbar_part1, smem.smem_Q1, 0, b_idx * H * S + h_idx * S + idx_q * 64);
            tma_load_2d_fn(&tma_dO1, &smem.mbar_part1, smem.smem_dO1, 0, b_idx * H * S + h_idx * S + idx_q * 64);
            tma_load_2d_fn(&tma_O1, &smem.mbar_part1, smem.smem_O1, 0, b_idx * H * S + h_idx * S + idx_q * 64);
        }
        mbarrier_wait_fn(&smem.mbar_part0, phase0);
        mbarrier_wait_fn(&smem.mbar_part1, phase1);
        phase0 ^= 1; phase1 ^= 1;
        fence_proxy_async_fn();
        __syncthreads();

        float dQ0[32] = {0};
        float dQ1[32] = {0};

        float d_val = 0;
        if (threadIdx.x < 64) {
            for (int c = 0; c < 64; ++c) {
                int swizzled_idx = swizzle_128B(threadIdx.x, c);
                d_val += __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&smem.smem_O0[swizzled_idx])) * 
                         __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&smem.smem_dO0[swizzled_idx]));
                d_val += __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&smem.smem_O1[swizzled_idx])) * 
                         __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&smem.smem_dO1[swizzled_idx]));
            }
            smem.smem_D[threadIdx.x] = d_val;
        }
        __syncthreads();

        uint64_t desc_Q0 = make_smem_desc_sm100_fn(smem.smem_Q0, 0, 128);
        uint64_t desc_Q1 = make_smem_desc_sm100_fn(smem.smem_Q1, 0, 128);
        uint64_t desc_K0 = make_smem_desc_sm100_fn(smem.smem_K0, 0, 128);
        uint64_t desc_K1 = make_smem_desc_sm100_fn(smem.smem_K1, 0, 128);
        uint64_t desc_V0 = make_smem_desc_sm100_fn(smem.smem_V0, 0, 128);
        uint64_t desc_V1 = make_smem_desc_sm100_fn(smem.smem_V1, 0, 128);
        uint64_t desc_dO0 = make_smem_desc_sm100_fn(smem.smem_dO0, 0, 128);
        uint64_t desc_dO1 = make_smem_desc_sm100_fn(smem.smem_dO1, 0, 128);
        uint64_t desc_P = make_smem_desc_sm100_fn(smem.smem_P, 0, 128);
        uint64_t desc_dS = make_smem_desc_sm100_fn(smem.smem_dS_bf16, 0, 128);
        
        uint32_t idesc = make_instr_desc_fn(64, 64);

        for (int idx_step = 0; idx_step < num_steps; ++idx_step) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&smem.mbar_part0, 16384);
                tma_load_2d_fn(&tma_K0, &smem.mbar_part0, smem.smem_K0, 0, b_idx * H * S + h_idx * S + idx_step * 64);
                tma_load_2d_fn(&tma_V0, &smem.mbar_part0, smem.smem_V0, 0, b_idx * H * S + h_idx * S + idx_step * 64);

                mbarrier_arrive_and_expect_tx_fn(&smem.mbar_part1, 16384);
                tma_load_2d_fn(&tma_K1, &smem.mbar_part1, smem.smem_K1, 0, b_idx * H * S + h_idx * S + idx_step * 64);
                tma_load_2d_fn(&tma_V1, &smem.mbar_part1, smem.smem_V1, 0, b_idx * H * S + h_idx * S + idx_step * 64);
            }
            mbarrier_wait_fn(&smem.mbar_part0, phase0);
            mbarrier_wait_fn(&smem.mbar_part1, phase1);
            phase0 ^= 1; phase1 ^= 1;
            fence_proxy_async_fn();
            __syncthreads();

            if (threadIdx.x == 0) {
                umma_f16_cg1_fn((char*)tmem_S_T, desc_Q0, desc_K0, idesc, 0);
                umma_f16_cg1_fn((char*)tmem_S_T + 64, desc_Q1, desc_K1, idesc, 1);
                
                umma_f16_cg1_fn((char*)tmem_dP_T, desc_V0, desc_dO0, idesc, 0);
                umma_f16_cg1_fn((char*)tmem_dP_T + 64, desc_V1, desc_dO1, idesc, 1);
            }
            __syncthreads();

            int tid = threadIdx.x % 64;
            int warp_id = threadIdx.x / 32;
            int row_base = warp_id * 32 + (tid % 32);
            
            __nv_bfloat16 dS_local[64 * 64];
            
            for (uint32_t col = 0; col < 128; col += 4) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_S_T));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                
                uint32_t dp0, dp1, dp2, dp3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(dp0),"=r"(dp1),"=r"(dp2),"=r"(dp3) : "r"(tmem_dP_T));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

                for (uint32_t k = 0; k < 4; ++k) {
                    uint32_t s_idx_c = col + k;
                    uint32_t s_idx_r = row_base;
                    float s_val = (k == 0) ? __uint_as_float(r0) : (k == 1) ? __uint_as_float(r1) : (k == 2) ? __uint_as_float(r2) : __uint_as_float(r3);
                    float dp_val = (k == 0) ? __uint_as_float(dp0) : (k == 1) ? __uint_as_float(dp1) : (k == 2) ? __uint_as_float(dp2) : __uint_as_float(dp3);
                    
                    float lse_val = (row_base < L_rows) ? L[row_base] : 0.0f;
                    float p_val = (s_idx_c < S_cols && row_base < S_rows) ? expf(s_val * sq_scale - lse_val) : 0.0f;
                    
                    float ds_val = p_val * (dp_val - *(float*)(smem_D + tid * sizeof(float)));
                    
                    uint32_t reg_idx = (s_idx_r * 64 + s_idx_c) / 4;
                    P_bf16[reg_idx] = __float2bfloat16(p_val);
                    dS_local[reg_idx * 4 + k] = __float2bfloat16(ds_val);
                }
            }
            
            __syncwarp(); 
            if (threadIdx.x % 32 < 16) { 
                uint32_t reg_idx = (threadIdx.x % 32) + (((threadIdx.x / 32) % 2) * 1024);
                uint32_t packed0 = pack_bf16_to_uint32(P_bf16[reg_idx * 2], P_bf16[reg_idx * 2 + 1]);
                uint32_t packed1 = pack_bf16_to_uint32(P_bf16[reg_idx * 2 + 2], P_bf16[reg_idx * 2 + 3]);
                
                uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem_P) + (reg_idx * 2 * 2);
                st_shared_128_fn(smem_addr, packed0, packed1, 0, 0);
            }
            __syncthreads();
            
            store_to_smem_64x64_swizzled(smem_dS_bf16, dS_local);
            store_to_smem_64x64_swizzled_transposed(smem_ST, dS_local);
            __syncthreads();
            
            if (threadIdx.x == 0) {
                umma_f16_cg1_fn((char*)tmem_dQ0, desc_dS, desc_K0, idesc, 1);
                umma_f16_cg1_fn((char*)tmem_dQ1, desc_dS, desc_K1, idesc, 1);
                umma_f16_cg1_fn((char*)tmem_dV0, desc_P, desc_dO0, idesc, 1);
                umma_f16_cg1_fn((char*)tmem_dV1, desc_P, desc_dO1, idesc, 1);
            }
            __syncthreads();
        }

        for (uint32_t col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_dQ0));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            __nv_bfloat16 bf0 = __float2bfloat16(__uint_as_float(r0));
            __nv_bfloat16 bf1 = __float2bfloat16(__uint_as_float(r1));
            __nv_bfloat16 bf2 = __float2bfloat16(__uint_as_float(r2));
            __nv_bfloat16 bf3 = __float2bfloat16(__uint_as_float(r3));
            uint32_t packed0 = pack_bf16_to_uint32(bf0, bf1);
            uint32_t packed1 = pack_bf16_to_uint32(bf2, bf3);
            uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem_dQ0) + (col * 2);
            st_shared_128_fn(smem_addr, packed0, packed1, 0, 0);
        }
        __syncthreads();
        
        if (threadIdx.x == 0) {
            tma_store_2d_fn(&tma_dQ0, smem_dQ0, 0, b_idx * H * S + h_idx * S + idx_q * 64);
            tma_store_2d_fn(&tma_dQ1, smem_dQ1, 0, b_idx * H * S + h_idx * S + idx_q * 64);
            tma_store_commit_fn();
        }
        tma_store_wait_fn<0>();
        __syncthreads();
    }

    // ==========================================
    // Pass 2: compute dK, dV
    // ==========================================
    int idx_kv = blockIdx.x;
    if (idx_kv < num_steps) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&smem.mbar_part0, 16384);
            tma_load_2d_fn(&tma_K0, &smem.mbar_part0, smem.smem_K0, 0, b_idx * H * S + h_idx * S + idx_kv * 64);
            tma_load_2d_fn(&tma_V0, &smem.mbar_part0, smem.smem_V0, 0, b_idx * H * S + h_idx * S + idx_kv * 64);

            mbarrier_arrive_and_expect_tx_fn(&smem.mbar_part1, 16384);
            tma_load_2d_fn(&tma_K1, &smem.mbar_part1, smem.smem_K1, 0, b_idx * H * S + h_idx * S + idx_kv * 64);
            tma_load_2d_fn(&tma_V1, &smem.mbar_part1, smem.smem_V1, 0, b_idx * H * S + h_idx * S + idx_kv * 64);
        }
        mbarrier_wait_fn(&smem.mbar_part0, phase0);
        mbarrier_wait_fn(&smem.mbar_part1, phase1);
        phase0 ^= 1; phase1 ^= 1;
        fence_proxy_async_fn();
        __syncthreads();

        for (int idx_step = 0; idx_step < num_steps; ++idx_step) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&smem.mbar_part0, 16384);
                tma_load_2d_fn(&tma_Q0, &smem.mbar_part0, smem.smem_Q0, 0, b_idx * H * S + h_idx * S + idx_step * 64);
                tma_load_2d_fn(&tma_dO0, &smem.mbar_part0, smem.smem_dO0, 0, b_idx * H * S + h_idx * S + idx_step * 64);
                tma_load_2d_fn(&tma_O0, &smem.mbar_part0, smem.smem_O0, 0, b_idx * H * S + h_idx * S + idx_step * 64);

                mbarrier_arrive_and_expect_tx_fn(&smem.mbar_part1, 16384);
                tma_load_2d_fn(&tma_Q1, &smem.mbar_part1, smem.smem_Q1, 0, b_idx * H * S + h_idx * S + idx_step * 64);
                tma_load_2d_fn(&tma_dO1, &smem.mbar_part1, smem.smem_dO1, 0, b_idx * H * S + h_idx * S + idx_step * 64);
                tma_load_2d_fn(&tma_O1, &smem.mbar_part1, smem.smem_O1, 0, b_idx * H * S + h_idx * S + idx_step * 64);
            }
            mbarrier_wait_fn(&smem.mbar_part0, phase0);
            mbarrier_wait_fn(&smem.mbar_part1, phase1);
            phase0 ^= 1; phase1 ^= 1;
            fence_proxy_async_fn();
            __syncthreads();

            float d_val = 0;
            if (threadIdx.x < 64) {
                for (int c = 0; c < 64; ++c) {
                    int swizzled_idx = swizzle_128B(threadIdx.x, c);
                    d_val += __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&smem.smem_O0[swizzled_idx])) * 
                             __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&smem.smem_dO0[swizzled_idx]));
                    d_val += __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&smem.smem_O1[swizzled_idx])) * 
                             __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&smem.smem_dO1[swizzled_idx]));
                }
                smem.smem_D[threadIdx.x] = d_val;
            }
            __syncthreads();

            if (threadIdx.x == 0) {
                umma_f16_cg1_fn((char*)tmem_S_T, desc_Q0, desc_K0, idesc, 0);
                umma_f16_cg1_fn((char*)tmem_S_T + 64, desc_Q1, desc_K1, idesc, 1);
                
                umma_f16_cg1_fn((char*)tmem_dP_T, desc_V0, desc_dO0, idesc, 0);
                umma_f16_cg1_fn((char*)tmem_dP_T + 64, desc_V1, desc_dO1, idesc, 1);
            }
            __syncthreads();

            int tid = threadIdx.x % 64;
            int warp_id = threadIdx.x / 32;
            int row_base = warp_id * 32 + (tid % 32);
            
            __nv_bfloat16 dS_local[64 * 64];
            
            for (uint32_t col = 0; col < 128; col += 4) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_S_T));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                
                uint32_t dp0, dp1, dp2, dp3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(dp0),"=r"(dp1),"=r"(dp2),"=r"(dp3) : "r"(tmem_dP_T));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

                for (uint32_t k = 0; k < 4; ++k) {
                    uint32_t s_idx_c = col + k;
                    uint32_t s_idx_r = row_base;
                    float s_val = (k == 0) ? __uint_as_float(r0) : (k == 1) ? __uint_as_float(r1) : (k == 2) ? __uint_as_float(r2) : __uint_as_float(r3);
                    float dp_val = (k == 0) ? __uint_as_float(dp0) : (k == 1) ? __uint_as_float(dp1) : (k == 2) ? __uint_as_float(dp2) : __uint_as_float(dp3);
                    
                    float lse_val = (row_base < L_rows) ? L[row_base] : 0.0f;
                    float p_val = (s_idx_c < S_cols && row_base < S_rows) ? expf(s_val * sq_scale - lse_val) : 0.0f;
                    
                    float ds_val = p_val * (dp_val - *(float*)(smem_D + tid * sizeof(float)));
                    
                    uint32_t reg_idx = (s_idx_r * 64 + s_idx_c) / 4;
                    P_bf16[reg_idx] = __float2bfloat16(p_val);
                    dS_local[reg_idx * 4 + k] = __float2bfloat16(ds_val);
                }
            }
            
            __syncwarp(); 
            if ((threadIdx.x % 32) < 16) { 
                uint32_t reg_idx = (threadIdx.x % 32) + (((threadIdx.x / 32) % 2) * 1024);
                uint32_t packed0 = pack_bf16_to_uint32(P_bf16[reg_idx * 2], P_bf16[reg_idx * 2 + 1]);
                uint32_t packed1 = pack_bf16_to_uint32(P_bf16[reg_idx * 2 + 2], P_bf16[reg_idx * 2 + 3]);
                
                uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem_P) + (reg_idx * 2 * 2);
                st_shared_128_fn(smem_addr, packed0, packed1, 0, 0);
            }
            __syncthreads();
            
            store_to_smem_64x64_swizzled(smem_dS_bf16, dS_local);
            store_to_smem_64x64_swizzled_transposed(smem_ST, dS_local);
            __syncthreads();
            
            if (threadIdx.x == 0) {
                umma_f16_cg1_fn((char*)tmem_dK0, desc_dST, desc_Q0, idesc, 1);
                umma_f16_cg1_fn((char*)tmem_dK1, desc_dST, desc_Q1, idesc, 1);
                umma_f16_cg1_fn((char*)tmem_dV0, desc_PT, desc_dO0, idesc, 1);
                umma_f16_cg1_fn((char*)tmem_dV1, desc_PT, desc_dO1, idesc, 1);
            }
            __syncthreads();
        }

        for (uint32_t col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_dK0));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            __nv_bfloat16 bf0 = __float2bfloat16(__uint_as_float(r0));
            __nv_bfloat16 bf1 = __float2bfloat16(__uint_as_float(r1));
            __nv_bfloat16 bf2 = __float2bfloat16(__uint_as_float(r2));
            __nv_bfloat16 bf3 = __float2bfloat16(__uint_as_float(r3));
            uint32_t packed0 = pack_bf16_to_uint32(bf0, bf1);
            uint32_t packed1 = pack_bf16_to_uint32(bf2, bf3);
            uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem_dK0) + (col * 2);
            st_shared_128_fn(smem_addr, packed0, packed1, 0, 0);
        }
        __syncthreads();
        
        if (threadIdx.x == 0) {
            tma_store_2d_fn(&tma_dK0, smem_dK0, 0, b_idx * H * S + h_idx * S + idx_kv * 64);
            tma_store_2d_fn(&tma_dK1, smem_dK1, 0, b_idx * H * S + h_idx * S + idx_kv * 64);
            tma_store_commit_fn();
        }
        tma_store_wait_fn<0>();
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        tmem_dealloc_fn(*tmem_S_T, 128);
        tmem_dealloc_fn(*tmem_P_T, 128);
        tmem_dealloc_fn(*tmem_dP_T, 128);
        tmem_dealloc_fn(*tmem_D_T, 128);
        tmem_dealloc_fn(*tmem_dS_T, 128);
        tmem_dealloc_fn(*tmem_dV0, 128);
        tmem_dealloc_fn(*tmem_dV1, 128);
        tmem_dealloc_fn(*tmem_dK0, 128);
        tmem_dealloc_fn(*tmem_dK1, 128);
        tmem_dealloc_fn(*tmem_dQ0, 128);
        tmem_dealloc_fn(*tmem_dQ1, 128);
    }
    
    delete tmem_S_T;
    delete tmem_P_T;
    delete tmem_dP_T;
    delete tmem_D_T;
    delete tmem_dS_T;
    delete tmem_dV0;
    delete tmem_dV1;
    delete tmem_dK0;
    delete tmem_dK1;
    delete tmem_dQ0;
    delete tmem_dQ1;
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

    CUtensorMap tma_Q0, tma_Q1, tma_K0, tma_K1, tma_V0, tma_V1, tma_dO0, tma_dO1, tma_O0, tma_O1, tma_dQ0, tma_dQ1, tma_dK0, tma_dK1, tma_dV0, tma_dV1;

    #define CREATE_TMA(tma, ptr) do { \
        CUresult res = create_tma_2d_descriptor_2B(&(tma), (ptr), gmem_inner_dim, gmem_outer_dim, smem_inner_dim, smem_outer_dim, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE); \
        if (res != CUDA_SUCCESS) { \
            fprintf(stderr, "Error creating TMA descriptor\n"); \
            exit(1); \
        } \
    } while(0)

    CREATE_TMA(tma_Q0, Q.data_ptr());
    CREATE_TMA(tma_Q1, Q.data_ptr());
    CREATE_TMA(tma_K0, K.data_ptr());
    CREATE_TMA(tma_K1, K.data_ptr());
    CREATE_TMA(tma_V0, V.data_ptr());
    CREATE_TMA(tma_V1, V.data_ptr());
    CREATE_TMA(tma_dO0, dO.data_ptr());
    CREATE_TMA(tma_dO1, dO.data_ptr());
    CREATE_TMA(tma_O0, O.data_ptr());
    CREATE_TMA(tma_O1, O.data_ptr());
    CREATE_TMA(tma_dQ0, dQ.data_ptr());
    CREATE_TMA(tma_dQ1, dQ.data_ptr());
    CREATE_TMA(tma_dK0, dK.data_ptr());
    CREATE_TMA(tma_dK1, dK.data_ptr());
    CREATE_TMA(tma_dV0, dV.data_ptr());
    CREATE_TMA(tma_dV1, dV.data_ptr());

    dim3 grid(ceil(S / 64.0), B * H);
    dim3 block(128);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(mha_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedStorage)));

    mha_bwd_kernel<<<grid, block, sizeof(SharedStorage), stream>>>(
        tma_Q0, tma_Q1, tma_K0, tma_K1, tma_V0, tma_V1, tma_dO0, tma_dO1, tma_O0, tma_O1,
        tma_dQ0, tma_dQ1, tma_dK0, tma_dK1, tma_dV0, tma_dV1,
        1.0f / sqrt(128.0f), S, H
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

} // namespace tvm_ffi_mha_bwd