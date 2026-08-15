#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>
#include <stdio.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

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

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61; // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint32_t make_idesc(uint32_t M, uint32_t N, int a_maj, int b_maj) {
    uint32_t d = 0;
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= (a_maj << 15);
    d |= (b_maj << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ uint64_t advance_desc_col(uint64_t desc, int steps) {
    uint32_t addr = (desc & 0x3FFF) << 4;
    addr += steps * 32; // 16 elements = 32 bytes
    desc &= ~0x3FFFULL;
    desc |= (addr >> 4);
    return desc;
}

__device__ __forceinline__ uint64_t advance_desc_row(uint64_t desc, int steps, int row_bytes) {
    uint32_t addr = (desc & 0x3FFF) << 4;
    addr += steps * 16 * row_bytes;
    desc &= ~0x3FFFULL;
    desc |= (addr >> 4);
    return desc;
}

__device__ __forceinline__ void umma_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(a));
}

__device__ __forceinline__ void tmem_epilogue_coalesced_4w_fn(
    __nv_bfloat16* D, __nv_bfloat16* smem_out, uint32_t tmem_col,
    uint32_t S, uint32_t N, uint32_t m_block, uint32_t n_block,
    uint32_t BM, uint32_t BN) {
    for (uint32_t col = 0; col < BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_col + col));
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
        if (global_row >= S) continue;
        
        if (lane_id < BN / 4) {
            uint32_t col_start = lane_id * 4;
            uint32_t global_col = n_block * BN + col_start;
            if (global_col + 3 < N) {
                uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * BN + col_start]);
                *reinterpret_cast<uint2*>(D + (uint64_t)global_row * N + global_col) = data;
            }
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void tmem_epilogue_atomic_4w_fn(
    __nv_bfloat16* D, __nv_bfloat16* smem_out, uint32_t tmem_col,
    uint32_t S, uint32_t N, uint32_t m_block, uint32_t n_block,
    uint32_t BM, uint32_t BN) {
    for (uint32_t col = 0; col < BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_col + col));
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
        if (global_row >= S) continue;
        
        uint32_t col_start = lane_id * 2;
        if (col_start < BN) {
            uint32_t global_col = n_block * BN + col_start;
            if (global_col + 1 < N) {
                __nv_bfloat162 v01 = *reinterpret_cast<__nv_bfloat162*>(&smem_out[row * BN + col_start]);
                atomicAdd(reinterpret_cast<__nv_bfloat162*>(D + (uint64_t)global_row * N + global_col), v01);
            }
        }
    }
    __syncthreads();
}

struct SharedStorage {
    __nv_bfloat16 Q0[128 * 64]; 
    __nv_bfloat16 Q1[128 * 64]; 
    __nv_bfloat16 K0[128 * 64]; 
    __nv_bfloat16 K1[128 * 64]; 
    __nv_bfloat16 V0[128 * 64]; 
    __nv_bfloat16 V1[128 * 64]; 
    __nv_bfloat16 dO0[128 * 64]; 
    __nv_bfloat16 dO1[128 * 64];
    __nv_bfloat16 P0[128 * 64]; 
    __nv_bfloat16 P1[128 * 64]; 
    __nv_bfloat16 dS0[128 * 64]; 
    __nv_bfloat16 dS1[128 * 64]; 
    __nv_bfloat16 O0[128 * 64]; 
    __nv_bfloat16 O1[128 * 64];
    uint64_t mbar[2];
    float D[128];
    float L[128];
    uint32_t tmem_base;
};

__global__ void __launch_bounds__(128, 1) mha_bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* L_ptr, __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    int S, float scale) 
{
    setmaxnreg_inc_sync_fn<256>();
    __syncthreads();
    
    extern __shared__ SharedStorage smem[];
    
    int i_block = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    int S_blocks = (S + 127) / 128;
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&smem->mbar[0], 1);
        init_smem_barrier_fn(&smem->mbar[1], 1);
        fence_smem_barrier_init_fn();
    }
    
    if (threadIdx.x < 32) {
        uint32_t a = (uint32_t)__cvta_generic_to_shared(&smem->tmem_base);
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(512));
    }
    __syncthreads();
    
    uint32_t tmem_base = smem->tmem_base;
    uint32_t dK0_tmem = tmem_base + 0;
    uint32_t dK1_tmem = tmem_base + 64;
    uint32_t dV0_tmem = tmem_base + 128;
    uint32_t dV1_tmem = tmem_base + 192;
    uint32_t S_tmem  = tmem_base + 256;
    uint32_t dP_tmem = tmem_base + 320;
    uint32_t dQ0_tmem = tmem_base + 256; 
    uint32_t dQ1_tmem = tmem_base + 320;
    
    uint64_t b_h_offset = (uint64_t)b * gridDim.y * S + (uint64_t)h * S;
    uint32_t phase_tma = 0;
    uint32_t phase_mma = 0;
    
    uint64_t desc_Q0_Kmaj = make_smem_desc_sm100_fn(smem->Q0, 1, 1024);
    uint64_t desc_Q1_Kmaj = make_smem_desc_sm100_fn(smem->Q1, 1, 1024);
    uint64_t desc_K0_Kmaj = make_smem_desc_sm100_fn(smem->K0, 1, 1024);
    uint64_t desc_K1_Kmaj = make_smem_desc_sm100_fn(smem->K1, 1, 1024);
    uint64_t desc_V0_Kmaj = make_smem_desc_sm100_fn(smem->V0, 1, 1024);
    uint64_t desc_V1_Kmaj = make_smem_desc_sm100_fn(smem->V1, 1, 1024);
    uint64_t desc_dO0_Kmaj = make_smem_desc_sm100_fn(smem->dO0, 1, 1024);
    uint64_t desc_dO1_Kmaj = make_smem_desc_sm100_fn(smem->dO1, 1, 1024);
    uint64_t desc_dS0_Kmaj = make_smem_desc_sm100_fn(smem->dS0, 1, 1024);
    uint64_t desc_dS1_Kmaj = make_smem_desc_sm100_fn(smem->dS1, 1, 1024);
    
    uint64_t desc_K0_Nmaj = make_smem_desc_sm100_fn(smem->K0, 16384, 1024);
    uint64_t desc_K1_Nmaj = make_smem_desc_sm100_fn(smem->K1, 16384, 1024);
    uint64_t desc_V0_Nmaj = make_smem_desc_sm100_fn(smem->V0, 16384, 1024);
    uint64_t desc_V1_Nmaj = make_smem_desc_sm100_fn(smem->V1, 16384, 1024);
    uint64_t desc_dS0_Nmaj = make_smem_desc_sm100_fn(smem->dS0, 16384, 1024);
    uint64_t desc_dS1_Nmaj = make_smem_desc_sm100_fn(smem->dS1, 16384, 1024);
    uint64_t desc_P0_Nmaj = make_smem_desc_sm100_fn(smem->P0, 16384, 1024);
    uint64_t desc_P1_Nmaj = make_smem_desc_sm100_fn(smem->P1, 16384, 1024);
    
    uint32_t smem_O0_addr = (uint32_t)__cvta_generic_to_shared(smem->O0);
    uint32_t smem_O1_addr = (uint32_t)__cvta_generic_to_shared(smem->O1);
    uint32_t smem_dO0_addr = (uint32_t)__cvta_generic_to_shared(smem->dO0);
    uint32_t smem_dO1_addr = (uint32_t)__cvta_generic_to_shared(smem->dO1);
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&smem->mbar[0], 64 * 128 * 2 * 4);
        tma_load_2d_fn(&tma_K, &smem->mbar[0], smem->K0, 0, b_h_offset + i_block * 128);
        tma_load_2d_fn(&tma_K, &smem->mbar[0], smem->K1, 64, b_h_offset + i_block * 128);
        tma_load_2d_fn(&tma_V, &smem->mbar[0], smem->V0, 0, b_h_offset + i_block * 128);
        tma_load_2d_fn(&tma_V, &smem->mbar[0], smem->V1, 64, b_h_offset + i_block * 128);
    }
    
    for (int c = 0; c < 64; c += 4) {
        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dK0_tmem + c), "r"(0), "r"(0), "r"(0), "r"(0));
        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dK1_tmem + c), "r"(0), "r"(0), "r"(0), "r"(0));
        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dV0_tmem + c), "r"(0), "r"(0), "r"(0), "r"(0));
        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dV1_tmem + c), "r"(0), "r"(0), "r"(0), "r"(0));
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    mbarrier_wait_fn(&smem->mbar[0], phase_tma); phase_tma ^= 1;
    
    for (int j_block = i_block; j_block < S_blocks; ++j_block) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&smem->mbar[0], 64 * 128 * 2 * 6);
            tma_load_2d_fn(&tma_Q, &smem->mbar[0], smem->Q0, 0, b_h_offset + j_block * 128);
            tma_load_2d_fn(&tma_Q, &smem->mbar[0], smem->Q1, 64, b_h_offset + j_block * 128);
            tma_load_2d_fn(&tma_O, &smem->mbar[0], smem->O0, 0, b_h_offset + j_block * 128);
            tma_load_2d_fn(&tma_O, &smem->mbar[0], smem->O1, 64, b_h_offset + j_block * 128);
            tma_load_2d_fn(&tma_dO, &smem->mbar[0], smem->dO0, 0, b_h_offset + j_block * 128);
            tma_load_2d_fn(&tma_dO, &smem->mbar[0], smem->dO1, 64, b_h_offset + j_block * 128);
        }
        if (threadIdx.x < 128) {
            uint64_t idx = b_h_offset + j_block * 128 + threadIdx.x;
            smem->L[threadIdx.x] = (j_block * 128 + threadIdx.x < S) ? L_ptr[idx] : 0.0f;
        }
        mbarrier_wait_fn(&smem->mbar[0], phase_tma); phase_tma ^= 1;
        __syncthreads();
        
        int row = threadIdx.x;
        float d_val = 0;
        for(int k = 0; k < 2; ++k) {
            uint32_t O_addr = (k == 0) ? smem_O0_addr : smem_O1_addr;
            uint32_t dO_addr = (k == 0) ? smem_dO0_addr : smem_dO1_addr;
            for(int chunk = 0; chunk < 8; ++chunk) {
                int swizzled_chunk = (row % 8) ^ chunk;
                uint32_t offset = row * 128 + swizzled_chunk * 16;
                uint4 vO, vdO;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(vO.x),"=r"(vO.y),"=r"(vO.z),"=r"(vO.w) : "r"(O_addr + offset));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(vdO.x),"=r"(vdO.y),"=r"(vdO.z),"=r"(vdO.w) : "r"(dO_addr + offset));
                
                float2 o01 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&vO.x));
                float2 o23 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&vO.y));
                float2 o45 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&vO.z));
                float2 o67 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&vO.w));
                float2 do01 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&vdO.x));
                float2 do23 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&vdO.y));
                float2 do45 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&vdO.z));
                float2 do67 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&vdO.w));
                
                d_val += o01.x * do01.x + o01.y * do01.y + o23.x * do23.x + o23.y * do23.y;
                d_val += o45.x * do45.x + o45.y * do45.y + o67.x * do67.x + o67.y * do67.y;
            }
        }
        smem->D[row] = d_val;
        __syncthreads();
        
        tcgen05_fence_after_fn();
        uint32_t idesc_S = make_idesc(128, 64, 0, 1);
        
        for (int part = 0; part < 2; ++part) {
            for (int c = 0; c < 64; c += 4) {
                asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};" :: "r"(S_tmem + c), "r"(0), "r"(0), "r"(0), "r"(0));
                asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dP_tmem + c), "r"(0), "r"(0), "r"(0), "r"(0));
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            
            if (threadIdx.x == 0) {
                uint64_t k_desc = (part == 0) ? desc_K0_Nmaj : desc_K1_Nmaj;
                for (int step = 0; step < 4; ++step) {
                    uint64_t q_addr = advance_desc_col(desc_Q0_Kmaj, step);
                    uint64_t k_addr = advance_desc_row(k_desc, step, 128);
                    uint32_t accum = (step == 0) ? 0 : 1;
                    asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, %4;\n" 
                                 :: "r"(S_tmem), "l"(q_addr), "l"(k_addr), "r"(idesc_S), "r"(accum));
                }
                for (int step = 0; step < 4; ++step) {
                    uint64_t q_addr = advance_desc_col(desc_Q1_Kmaj, step);
                    uint64_t k_addr = advance_desc_row(k_desc, 4 + step, 128);
                    asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 1;\n" 
                                 :: "r"(S_tmem), "l"(q_addr), "l"(k_addr), "r"(idesc_S));
                }
                
                uint64_t v_desc = (part == 0) ? desc_V0_Nmaj : desc_V1_Nmaj;
                for (int step = 0; step < 4; ++step) {
                    uint64_t do_addr = advance_desc_col(desc_dO0_Kmaj, step);
                    uint64_t v_addr = advance_desc_row(v_desc, step, 128);
                    uint32_t accum = (step == 0) ? 0 : 1;
                    asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, %4;\n" 
                                 :: "r"(dP_tmem), "l"(do_addr), "l"(v_addr), "r"(idesc_S), "r"(accum));
                }
                for (int step = 0; step < 4; ++step) {
                    uint64_t do_addr = advance_desc_col(desc_dO1_Kmaj, step);
                    uint64_t v_addr = advance_desc_row(v_desc, 4 + step, 128);
                    asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 1;\n" 
                                 :: "r"(dP_tmem), "l"(do_addr), "l"(v_addr), "r"(idesc_S));
                }
                umma_commit_cg1_fn(&smem->mbar[1]);
                mbarrier_wait_fn(&smem->mbar[1], phase_mma); phase_mma ^= 1;
            }
            __syncthreads();
            
            uint32_t P_smem_addr = (part == 0) ? (uint32_t)__cvta_generic_to_shared(smem->P0) : (uint32_t)__cvta_generic_to_shared(smem->P1);
            uint32_t dS_smem_addr = (part == 0) ? (uint32_t)__cvta_generic_to_shared(smem->dS0) : (uint32_t)__cvta_generic_to_shared(smem->dS1);
            
            for (int c = 0; c < 64; c += 8) {
                uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(S_tmem + c));
                uint32_t dp0, dp1, dp2, dp3, dp4, dp5, dp6, dp7;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                    : "=r"(dp0),"=r"(dp1),"=r"(dp2),"=r"(dp3),"=r"(dp4),"=r"(dp5),"=r"(dp6),"=r"(dp7) : "r"(dP_tmem + c));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                
                int Q_pos = j_block * 128 + threadIdx.x;
                float lse = smem->L[threadIdx.x];
                float d_v = smem->D[threadIdx.x];
                
                float f[8], ds[8];
                f[0] = __uint_as_float(r0) * scale; f[1] = __uint_as_float(r1) * scale;
                f[2] = __uint_as_float(r2) * scale; f[3] = __uint_as_float(r3) * scale;
                f[4] = __uint_as_float(r4) * scale; f[5] = __uint_as_float(r5) * scale;
                f[6] = __uint_as_float(r6) * scale; f[7] = __uint_as_float(r7) * scale;
                
                for(int w=0; w<8; ++w) {
                    int K_pos = i_block * 128 + part * 64 + c + w;
                    if (K_pos > Q_pos || Q_pos >= S || K_pos >= S) f[w] = -INFINITY;
                    f[w] = fast_exp2f_fn((f[w] - lse) * 1.44269504f);
                    if (K_pos > Q_pos || Q_pos >= S || K_pos >= S) f[w] = 0.0f;
                }
                
                ds[0] = f[0] * (__uint_as_float(dp0) - d_v); ds[1] = f[1] * (__uint_as_float(dp1) - d_v);
                ds[2] = f[2] * (__uint_as_float(dp2) - d_v); ds[3] = f[3] * (__uint_as_float(dp3) - d_v);
                ds[4] = f[4] * (__uint_as_float(dp4) - d_v); ds[5] = f[5] * (__uint_as_float(dp5) - d_v);
                ds[6] = f[6] * (__uint_as_float(dp6) - d_v); ds[7] = f[7] * (__uint_as_float(dp7) - d_v);
                
                uint32_t p01 = pack_bf16_fn(__float_as_uint(f[0]), __float_as_uint(f[1]));
                uint32_t p23 = pack_bf16_fn(__float_as_uint(f[2]), __float_as_uint(f[3]));
                uint32_t p45 = pack_bf16_fn(__float_as_uint(f[4]), __float_as_uint(f[5]));
                uint32_t p67 = pack_bf16_fn(__float_as_uint(f[6]), __float_as_uint(f[7]));
                
                uint32_t s01 = pack_bf16_fn(__float_as_uint(ds[0]), __float_as_uint(ds[1]));
                uint32_t s23 = pack_bf16_fn(__float_as_uint(ds[2]), __float_as_uint(ds[3]));
                uint32_t s45 = pack_bf16_fn(__float_as_uint(ds[4]), __float_as_uint(ds[5]));
                uint32_t s67 = pack_bf16_fn(__float_as_uint(ds[6]), __float_as_uint(ds[7]));
                
                int swizzled_chunk = (threadIdx.x % 8) ^ (c / 8);
                uint32_t p_addr = P_smem_addr + threadIdx.x * 128 + swizzled_chunk * 16;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(p_addr), "r"(p01), "r"(p23), "r"(p45), "r"(p67));
                
                uint32_t ds_addr = dS_smem_addr + threadIdx.x * 128 + swizzled_chunk * 16;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ds_addr), "r"(s01), "r"(s23), "r"(s45), "r"(s67));
            }
        }
        fence_proxy_async_fn();
        __syncthreads();
        
        for (int c = 0; c < 64; c += 4) {
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dQ0_tmem + c), "r"(0), "r"(0), "r"(0), "r"(0));
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};" :: "r"(dQ1_tmem + c), "r"(0), "r"(0), "r"(0), "r"(0));
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        tcgen05_fence_after_fn();
        
        if (threadIdx.x == 0) {
            uint32_t idesc_dQ = make_idesc(128, 64, 0, 0); 
            for (int part = 0; part < 2; ++part) {
                uint64_t ds_desc = (part == 0) ? desc_dS0_Kmaj : desc_dS1_Kmaj;
                for (int step = 0; step < 4; ++step) {
                    uint64_t ds_addr = advance_desc_col(ds_desc, step);
                    uint64_t k0_addr = advance_desc_row(desc_K0_Kmaj, part * 4 + step, 128);
                    uint32_t accum = (part == 0 && step == 0) ? 0 : 1;
                    asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, %4;\n" 
                                 :: "r"(dQ0_tmem), "l"(ds_addr), "l"(k0_addr), "r"(idesc_dQ), "r"(accum));
                }
            }
            for (int part = 0; part < 2; ++part) {
                uint64_t ds_desc = (part == 0) ? desc_dS0_Kmaj : desc_dS1_Kmaj;
                for (int step = 0; step < 4; ++step) {
                    uint64_t ds_addr = advance_desc_col(ds_desc, step);
                    uint64_t k1_addr = advance_desc_row(desc_K1_Kmaj, part * 4 + step, 128);
                    uint32_t accum = (part == 0 && step == 0) ? 0 : 1;
                    asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, %4;\n" 
                                 :: "r"(dQ1_tmem), "l"(ds_addr), "l"(k1_addr), "r"(idesc_dQ), "r"(accum));
                }
            }
            
            uint32_t idesc_dK = make_idesc(64, 64, 1, 0); 
            for (int step = 0; step < 8; ++step) {
                uint64_t ds0_addr = advance_desc_row(desc_dS0_Nmaj, step, 128);
                uint64_t q0_addr = advance_desc_row(desc_Q0_Kmaj, step, 128);
                asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 1;\n" 
                             :: "r"(dK0_tmem), "l"(ds0_addr), "l"(q0_addr), "r"(idesc_dK));
            }
            uint32_t dK0_bot = dK0_tmem | (64 << 16);
            for (int step = 0; step < 8; ++step) {
                uint64_t ds1_addr = advance_desc_row(desc_dS1_Nmaj, step, 128);
                uint64_t q0_addr = advance_desc_row(desc_Q0_Kmaj, step, 128);
                asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 1;\n" 
                             :: "r"(dK0_bot), "l"(ds1_addr), "l"(q0_addr), "r"(idesc_dK));
            }
            
            for (int step = 0; step < 8; ++step) {
                uint64_t ds0_addr = advance_desc_row(desc_dS0_Nmaj, step, 128);
                uint64_t q1_addr = advance_desc_row(desc_Q1_Kmaj, step, 128);
                asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 1;\n" 
                             :: "r"(dK1_tmem), "l"(ds0_addr), "l"(q1_addr), "r"(idesc_dK));
            }
            uint32_t dK1_bot = dK1_tmem | (64 << 16);
            for (int step = 0; step < 8; ++step) {
                uint64_t ds1_addr = advance_desc_row(desc_dS1_Nmaj, step, 128);
                uint64_t q1_addr = advance_desc_row(desc_Q1_Kmaj, step, 128);
                asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 1;\n" 
                             :: "r"(dK1_bot), "l"(ds1_addr), "l"(q1_addr), "r"(idesc_dK));
            }
            
            uint32_t idesc_dV = make_idesc(64, 64, 1, 0); 
            for (int step = 0; step < 8; ++step) {
                uint64_t p0_addr = advance_desc_row(desc_P0_Nmaj, step, 128);
                uint64_t do0_addr = advance_desc_row(desc_dO0_Kmaj, step, 128);
                asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 1;\n" 
                             :: "r"(dV0_tmem), "l"(p0_addr), "l"(do0_addr), "r"(idesc_dV));
            }
            uint32_t dV0_bot = dV0_tmem | (64 << 16);
            for (int step = 0; step < 8; ++step) {
                uint64_t p1_addr = advance_desc_row(desc_P1_Nmaj, step, 128);
                uint64_t do0_addr = advance_desc_row(desc_dO0_Kmaj, step, 128);
                asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 1;\n" 
                             :: "r"(dV0_bot), "l"(p1_addr), "l"(do0_addr), "r"(idesc_dV));
            }
            
            for (int step = 0; step < 8; ++step) {
                uint64_t p0_addr = advance_desc_row(desc_P0_Nmaj, step, 128);
                uint64_t do1_addr = advance_desc_row(desc_dO1_Kmaj, step, 128);
                asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 1;\n" 
                             :: "r"(dV1_tmem), "l"(p0_addr), "l"(do1_addr), "r"(idesc_dV));
            }
            uint32_t dV1_bot = dV1_tmem | (64 << 16);
            for (int step = 0; step < 8; ++step) {
                uint64_t p1_addr = advance_desc_row(desc_P1_Nmaj, step, 128);
                uint64_t do1_addr = advance_desc_row(desc_dO1_Kmaj, step, 128);
                asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 1;\n" 
                             :: "r"(dV1_bot), "l"(p1_addr), "l"(do1_addr), "r"(idesc_dV));
            }
            
            umma_commit_cg1_fn(&smem->mbar[1]);
            mbarrier_wait_fn(&smem->mbar[1], phase_mma); phase_mma ^= 1;
        }
        __syncthreads();
        
        tmem_epilogue_atomic_4w_fn(dQ, smem->Q0, dQ0_tmem, S, 128, j_block, 0, 128, 64);
        tmem_epilogue_atomic_4w_fn(dQ, smem->Q1, dQ1_tmem, S, 128, j_block, 1, 128, 64);
    }
    
    tmem_epilogue_coalesced_4w_fn(dK, smem->K0, dK0_tmem, S, 128, i_block, 0, 128, 64);
    tmem_epilogue_coalesced_4w_fn(dK, smem->K1, dK1_tmem, S, 128, i_block, 1, 128, 64);
    tmem_epilogue_coalesced_4w_fn(dV, smem->V0, dV0_tmem, S, 128, i_block, 0, 128, 64);
    tmem_epilogue_coalesced_4w_fn(dV, smem->V1, dV1_tmem, S, 128, i_block, 1, 128, 64);
    
    __syncthreads();
    
    if (threadIdx.x < 32) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_base), "r"(512));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
}

namespace tvm_ffi_example_cuda {

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, globalAddress, globalDim, globalStrides, boxDim,
        elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L, 
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
         
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    
    float scale = 1.0f / sqrtf(128.0f);
    
    CUDA_CHECK(cudaMemsetAsync(dQ.data_ptr(), 0, B * H * S * 128 * sizeof(uint16_t), stream));
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O, tma_dO;
    if (create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), 128, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B) != CUDA_SUCCESS) exit(1);
    if (create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), 128, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B) != CUDA_SUCCESS) exit(1);
    if (create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), 128, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B) != CUDA_SUCCESS) exit(1);
    if (create_tma_2d_descriptor_2B(&tma_O, O.data_ptr(), 128, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B) != CUDA_SUCCESS) exit(1);
    if (create_tma_2d_descriptor_2B(&tma_dO, dO.data_ptr(), 128, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B) != CUDA_SUCCESS) exit(1);
    
    int S_blocks = (S + 127) / 128;
    dim3 grid(S_blocks, H, B);
    dim3 block(128);
    
    int max_smem = sizeof(SharedStorage);
    CUDA_CHECK(cudaFuncSetAttribute(
        mha_bwd_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        max_smem
    ));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = max_smem;
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 1;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, mha_bwd_kernel, tma_Q, tma_K, tma_V, tma_O, tma_dO, 
        static_cast<const float*>(L.data_ptr()), 
        static_cast<__nv_bfloat16*>(dQ.data_ptr()), 
        static_cast<__nv_bfloat16*>(dK.data_ptr()), 
        static_cast<__nv_bfloat16*>(dV.data_ptr()), 
        S, scale));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

} // namespace tvm_ffi_example_cuda