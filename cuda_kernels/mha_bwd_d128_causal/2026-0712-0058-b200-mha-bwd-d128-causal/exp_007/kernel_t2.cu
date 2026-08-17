#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
#include <mma.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        fprintf(stderr, "CU error %d at %s:%d\n",                  \
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_example_cuda {

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill
    );
}

__device__ __forceinline__ uint32_t fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return __float_as_uint(y);
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

__device__ __forceinline__ float __low2float(uint32_t x) {
    __nv_bfloat16 a;
    asm("mov.b32 {%0, %1}, %2;" : "=h"(*reinterpret_cast<uint16_t*>(&a)), "=h"(1), "r"(x));
    return __bfloat162float(a);
}

__device__ __forceinline__ float __high2float(uint32_t x) {
    __nv_bfloat16 a;
    asm("mov.b32 {%0, %1}, %2;" : "=h"(1), "=h"(*reinterpret_cast<uint16_t*>(&a)), "r"(x));
    return __bfloat162float(a);
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
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_load_packed_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.pack::16b.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_store_32x32b_x1(uint32_t col, uint32_t val) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 %0, [%1];" :: "r"(val), "r"(col));
}

__device__ __forceinline__ void tmem_store_packed_4x_fn(uint32_t col, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.pack::16b.b32 {%0, %1, %2, %3}, [%4];"
                 :: "r"(v0), "r"(v1), "r"(v2), "r"(v3), "r"(col));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_store_fence_fn() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_clear_128x128_packed(uint32_t tmem_S_T) {
    for (int c = 0; c < 64; c++) {
        tmem_store_32x32b_x1(tmem_S_T + c, 0);
    }
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

__device__ __forceinline__ uint64_t make_smem_desc_k_major(void* smem_ptr) {
    return make_smem_desc_sm100_fn(smem_ptr, 1, 1024);
}

__device__ __forceinline__ uint64_t make_smem_desc_n_major(void* smem_ptr) {
    return make_smem_desc_sm100_fn(smem_ptr, 8192, 1024);
}

__device__ __forceinline__ uint64_t make_smem_desc_mn_major(void* smem_ptr) {
    return make_smem_desc_sm100_fn(smem_ptr, 8192, 1024);
}

__device__ __forceinline__ uint32_t make_instr_desc_fn_major(uint32_t M, uint32_t N, uint32_t a_major, uint32_t b_major) {
    uint32_t d = 0;
    d |= (1u << 4);     
    d |= (1u << 7);     
    d |= (1u << 10);    
    d |= (a_major << 15);    
    d |= (b_major << 16);    
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
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

__device__ __forceinline__ void gemm_K_K(uint32_t tmem_S_T, uint64_t desc_A, uint64_t desc_B, uint32_t idesc, bool accum) {
    uint32_t accum_flag = accum ? 1 : 0;
    uint32_t curr_desc_A = desc_A;
    uint32_t curr_desc_B = desc_B;
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %5, 0;\n"
        "1:\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n"
        "add.u32 %1, %1, 2;\n"
        "add.u32 %2, %2, 2;\n"
        "dec.u32 %4, 1;\n"
        "@%4 bra 1;\n}\n"
        :: "r"(tmem_S_T), "r"(curr_desc_A), "r"(curr_desc_B), "r"(idesc), "r"(4), "r"(accum_flag));
}

__device__ __forceinline__ void gemm_K_N(uint32_t tmem_D, uint64_t desc_A, uint64_t desc_B, uint32_t idesc, bool accum) {
    uint32_t accum_flag = accum ? 1 : 0;
    uint32_t curr_desc_A = desc_A;
    uint32_t curr_desc_B = desc_B;
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %5, 0;\n"
        "1:\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n"
        "add.u32 %1, %1, 2;\n"
        "add.u32 %2, %2, 128;\n"
        "dec.u32 %4, 1;\n"
        "@%4 bra 1;\n}\n"
        :: "r"(tmem_D), "r"(curr_desc_A), "r"(curr_desc_B), "r"(idesc), "r"(4), "r"(accum_flag));
}

__device__ __forceinline__ void gemm_MN_N(uint32_t tmem_D, uint64_t desc_A, uint64_t desc_B, uint32_t idesc, bool accum) {
    uint32_t accum_flag = accum ? 1 : 0;
    uint32_t curr_desc_A = desc_A;
    uint32_t curr_desc_B = desc_B;
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %5, 0;\n"
        "1:\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n"
        "add.u32 %1, %1, 128;\n"
        "add.u32 %2, %2, 128;\n"
        "dec.u32 %4, 1;\n"
        "@%4 bra 1;\n}\n"
        :: "r"(tmem_D), "r"(curr_desc_A), "r"(curr_desc_B), "r"(idesc), "r"(4), "r"(accum_flag));
}

__device__ __forceinline__ int swizzle_128B(int x, int y) {
    return (((y % 8) ^ (x / 8)) * 8) + (x % 8);
}

__global__ void launch_kernel(
    const __grid_constant__ CUtensorMap tma_Q, 
    const __grid_constant__ CUtensorMap tma_K, 
    const __grid_constant__ CUtensorMap tma_V, 
    const __grid_constant__ CUtensorMap tma_dO, 
    const __grid_constant__ CUtensorMap tma_O, 
    __nv_bfloat16* dQ_ptr, __nv_bfloat16* dK_ptr, __nv_bfloat16* dV_ptr, 
    const float* L_ptr, int num_blocks, int M_global, int N_global, int bh) 
{
    extern __shared__ char smem_raw[];
    char* smem = smem_raw;
    uintptr_t smem_addr = (uintptr_t)smem_raw;
    if (smem_addr % 1024 != 0) {
        smem = smem_raw + (1024 - (smem_addr % 1024));
    }

    __nv_bfloat16* smem_Q_0 = (__nv_bfloat16*)smem;
    __nv_bfloat16* smem_Q_1 = smem_Q_0 + 64 * 64;
    __nv_bfloat16* smem_K_0 = smem_Q_1 + 64 * 64;
    __nv_bfloat16* smem_K_1 = smem_K_0 + 64 * 64;
    __nv_bfloat16* smem_V_0 = smem_K_1 + 64 * 64;
    __nv_bfloat16* smem_V_1 = smem_V_0 + 64 * 64;
    __nv_bfloat16* smem_dO_0 = smem_V_1 + 64 * 64;
    __nv_bfloat16* smem_dO_1 = smem_dO_0 + 64 * 64;
    __nv_bfloat16* smem_O_0 = smem_dO_1 + 64 * 64;
    __nv_bfloat16* smem_O_1 = smem_O_0 + 64 * 64;
    __nv_bfloat16* smem_P_T = smem_O_1 + 64 * 64;
    __nv_bfloat16* smem_dS_T = smem_P_T + 64 * 64;
    float* smem_D_local = (float*)(smem_dS_T + 64 * 64);
    float* smem_LSE_local = smem_D_local + 64;

    uint64_t* mbar_S = (uint64_t*)(smem_LSE_local + 64);
    uint64_t* mbar_dP = mbar_S + 1;
    uint64_t* mbar_dV = mbar_dP + 1;
    uint64_t* mbar_dK = mbar_dV + 1;
    uint64_t* mbar_dQ = mbar_dK + 1;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_S, 1);
        init_smem_barrier_fn(mbar_dP, 1);
        init_smem_barrier_fn(mbar_dV, 1);
        init_smem_barrier_fn(mbar_dK, 1);
        init_smem_barrier_fn(mbar_dQ, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t tmem_base;
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_base, 384);
    }
    __syncthreads();
    
    uint32_t tmem_S_T = tmem_base;
    uint32_t tmem_dP_T = tmem_base + 64;
    uint32_t tmem_dS_T = tmem_base + 128;
    uint32_t tmem_P_T = tmem_base + 192;
    uint32_t tmem_dV = tmem_base + 256;
    uint32_t tmem_dK = tmem_base + 320;
    uint32_t tmem_dQ_0 = tmem_dS_T;
    uint32_t tmem_dQ_1 = tmem_dP_T;

    int q_idx = blockIdx.x;
    float attn_scale = 1.0f / sqrtf(128.0f);

    float4* smem_Q_0_f4 = (float4*)smem_Q_0;
    float4* gmem_Q_0_f4 = (float4*)(Q_ptr + bh_off + q_idx * 64 * 128);
    for(int idx = threadIdx.x; idx < 512; idx += blockDim.x) {
        smem_Q_0_f4[idx] = gmem_Q_0_f4[idx];
    }
    float4* smem_Q_1_f4 = (float4*)smem_Q_1;
    float4* gmem_Q_1_f4 = (float4*)(Q_ptr + bh_off + q_idx * 64 * 128 + 64);
    for(int idx = threadIdx.x; idx < 512; idx += blockDim.x) {
        smem_Q_1_f4[idx] = gmem_Q_1_f4[idx];
    }
    float4* smem_O_0_f4 = (float4*)smem_O_0;
    float4* gmem_O_0_f4 = (float4*)(O_ptr + bh_off + q_idx * 64 * 128);
    for(int idx = threadIdx.x; idx < 512; idx += blockDim.x) {
        smem_O_0_f4[idx] = gmem_O_0_f4[idx];
    }
    float4* smem_O_1_f4 = (float4*)smem_O_1;
    float4* gmem_O_1_f4 = (float4*)(O_ptr + bh_off + q_idx * 64 * 128 + 64);
    for(int idx = threadIdx.x; idx < 512; idx += blockDim.x) {
        smem_O_1_f4[idx] = gmem_O_1_f4[idx];
    }
    float4* smem_dO_0_f4 = (float4*)smem_dO_0;
    float4* gmem_dO_0_f4 = (float4*)(dO_ptr + bh_off + q_idx * 64 * 128);
    for(int idx = threadIdx.x; idx < 512; idx += blockDim.x) {
        smem_dO_0_f4[idx] = gmem_dO_0_f4[idx];
    }
    float4* smem_dO_1_f4 = (float4*)smem_dO_1;
    float4* gmem_dO_1_f4 = (float4*)(dO_ptr + bh_off + q_idx * 64 * 128 + 64);
    for(int idx = threadIdx.x; idx < 512; idx += blockDim.x) {
        smem_dO_1_f4[idx] = gmem_dO_1_f4[idx];
    }

    __syncthreads();

    float d_val = 0;
    int row_idx = (threadIdx.x / 8) * 2 + (threadIdx.x % 2);
    int col_idx = ((threadIdx.x % 8) / 2) * 8;
    
    for (int i = 0; i < 8; ++i) {
        int cur_col = col_idx + (i % 4) * 2;
        d_val += __bfloat162float(smem_O_0[row_idx * 64 + swizzle_128B(cur_col, row_idx)]) * 
                 __bfloat162float(smem_dO_0[row_idx * 64 + swizzle_128B(cur_col, row_idx)]);
        d_val += __bfloat162float(smem_O_1[row_idx * 64 + swizzle_128B(cur_col, row_idx)]) * 
                 __bfloat162float(smem_dO_1[row_idx * 64 + swizzle_128B(cur_col, row_idx)]);
    }
    
    if (threadIdx.x < 64) {
        smem_D_local[threadIdx.x] = d_val;
        smem_LSE_local[threadIdx.x] = L_ptr[bh * S_len + q_idx * 64 + threadIdx.x];
    }

    float dQ_h_reg[8], dQ_t_reg[8];
    for (int i = 0; i < 8; ++i) {
        dQ_h_reg[i] = 0;
        dQ_t_reg[i] = 0;
    }

    uint32_t idesc_S = make_instr_desc_fn_major(64, 64, 0, 0);
    uint32_t idesc_dP = make_instr_desc_fn_major(64, 64, 0, 0);
    uint32_t idesc_dV = make_instr_desc_fn_major(64, 64, 0, 1);
    uint32_t idesc_dK = make_instr_desc_fn_major(64, 64, 0, 1);
    uint32_t idesc_dQ = make_instr_desc_fn_major(64, 64, 1, 1);

    uint32_t phase_S = 0, phase_dP = 0, phase_dV = 0, phase_dK = 0, phase_dQ = 0;
    bool accum_dV = false;
    bool accum_dK = false;
    bool accum_dQ = false;

    for (int kv_idx = 0; kv_idx <= q_idx && kv_idx < num_blocks; ++kv_idx) {
        
        float4* smem_K_0_f4 = (float4*)smem_K_0;
        float4* gmem_K_0_f4 = (float4*)(K_ptr + bh_off + kv_idx * 64 * 128);
        for(int idx = threadIdx.x; idx < 512; idx += blockDim.x) {
            smem_K_0_f4[idx] = gmem_K_0_f4[idx];
        }
        float4* smem_K_1_f4 = (float4*)smem_K_1;
        float4* gmem_K_1_f4 = (float4*)(K_ptr + bh_off + kv_idx * 64 * 128 + 64);
        for(int idx = threadIdx.x; idx < 512; idx += blockDim.x) {
            smem_K_1_f4[idx] = gmem_K_1_f4[idx];
        }
        float4* smem_V_0_f4 = (float4*)smem_V_0;
        float4* gmem_V_0_f4 = (float4*)(V_ptr + bh_off + kv_idx * 64 * 128);
        for(int idx = threadIdx.x; idx < 512; idx += blockDim.x) {
            smem_V_0_f4[idx] = gmem_V_0_f4[idx];
        }
        float4* smem_V_1_f4 = (float4*)smem_V_1;
        float4* gmem_V_1_f4 = (float4*)(V_ptr + bh_off + kv_idx * 64 * 128 + 64);
        for(int idx = threadIdx.x; idx < 512; idx += blockDim.x) {
            smem_V_1_f4[idx] = gmem_V_1_f4[idx];
        }
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_S, 16384);
            tmem_clear_128x128_packed(tmem_S_T);
        }
        mbarrier_wait_fn(mbar_S, phase_S);
        phase_S ^= 1;
        __syncthreads();

        uint64_t desc_K_0 = make_smem_desc_k_major(smem_K_0);
        uint64_t desc_Q_0 = make_smem_desc_k_major(smem_Q_0);
        uint64_t desc_K_1 = make_smem_desc_k_major(smem_K_1);
        uint64_t desc_Q_1 = make_smem_desc_k_major(smem_Q_1);

        if (threadIdx.x == 0) {
            gemm_K_K(tmem_S_T, desc_K_0, desc_Q_0, idesc_S, false);
            gemm_K_K(tmem_S_T, desc_K_1, desc_Q_1, idesc_S, true);
            asm volatile("cp.async.commit;" ::: "memory");
        }
        __syncthreads();
        
        for (int col = threadIdx.x; col < 64; col += blockDim.x) {
            uint32_t r0, r1, r2, r3;
            tmem_load_packed_4x_fn(tmem_S_T + col, &r0, &r1, &r2, &r3);
            float s0 = __low2float(r0);
            float s1 = __high2float(r0);
            float s2 = __low2float(r1);
            float s3 = __high2float(r1);
            float s4 = __low2float(r2);
            float s5 = __high2float(r2);
            float s6 = __low2float(r3);
            float s7 = __high2float(r3);

            int tid_in_col = col % 4; 
            int row = (col * 64 + tid_in_col) / 8;
            float lse = smem_LSE_local[row];

            float p0 = 0, p1 = 0, p2 = 0, p3 = 0, p4 = 0, p5 = 0, p6 = 0, p7 = 0;
            float dp0 = 0, dp1 = 0, dp2 = 0, dp3 = 0, dp4 = 0, dp5 = 0, dp6 = 0, dp7 = 0;
            float ds0 = 0, ds1 = 0, ds2 = 0, ds3 = 0, ds4 = 0, ds5 = 0, ds6 = 0, ds7 = 0;

            if (kv_idx < q_idx || (kv_idx == q_idx && col <= row)) {
                p0 = __bfloat162float(__float_as_bfloat16(fast_exp2f_fn(s0 * attn_scale * 1.44269504 - lse * 1.44269504)));
                p1 = __bfloat162float(__float_as_bfloat16(fast_exp2f_fn(s1 * attn_scale * 1.44269504 - lse * 1.44269504)));
                p2 = __bfloat162float(__float_as_bfloat16(fast_exp2f_fn(s2 * attn_scale * 1.44269504 - lse * 1.44269504)));
                p3 = __bfloat162float(__float_as_bfloat16(fast_exp2f_fn(s3 * attn_scale * 1.44269504 - lse * 1.44269504)));
                p4 = __bfloat162float(__float_as_bfloat16(fast_exp2f_fn(s4 * attn_scale * 1.44269504 - lse * 1.44269504)));
                p5 = __bfloat162float(__float_as_bfloat16(fast_exp2f_fn(s5 * attn_scale * 1.44269504 - lse * 1.44269504)));
                p6 = __bfloat162float(__float_as_bfloat16(fast_exp2f_fn(s6 * attn_scale * 1.44269504 - lse * 1.44269504)));
                p7 = __bfloat162float(__float_as_bfloat16(fast_exp2f_fn(s7 * attn_scale * 1.44269504 - lse * 1.44269504)));
            }
            
            int col_swizzled = ((row % 8) ^ (col * 8 / 8)) * 8 + tid_in_col;
            smem_P_T[row * 64 + col_swizzled] = __float2bfloat16(p0);
            smem_dS_T[row * 64 + col_swizzled] = __float2bfloat16(ds0);
        }
        tmem_load_fence_fn();
        named_barrier_sync_fn(1, 128); 

        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_dP, 16384);
            tmem_clear_128x128_packed(tmem_dP_T);
        }
        mbarrier_wait_fn(mbar_dP, phase_dP);
        phase_dP ^= 1;
        __syncthreads();

        uint64_t desc_V_0 = make_smem_desc_k_major(smem_V_0);
        uint64_t desc_dO_0 = make_smem_desc_k_major(smem_dO_0);
        uint64_t desc_V_1 = make_smem_desc_k_major(smem_V_1);
        uint64_t desc_dO_1 = make_smem_desc_k_major(smem_dO_1);

        if (threadIdx.x == 0) {
            gemm_K_K(tmem_dP_T, desc_V_0, desc_dO_0, idesc_dP, false);
            gemm_K_K(tmem_dP_T, desc_V_1, desc_dO_1, idesc_dP, true);
            asm volatile("cp.async.commit;" ::: "memory");
        }
        __syncthreads();

        for (int col = threadIdx.x; col < 64; col += blockDim.x) {
            uint32_t r0, r1, r2, r3;
            tmem_load_packed_4x_fn(tmem_dP_T + col, &r0, &r1, &r2, &r3);
            float dp0 = __low2float(r0);
            float dp1 = __high2float(r0);
            float dp2 = __low2float(r1);
            float dp3 = __high2float(r1);
            float dp4 = __low2float(r2);
            float dp5 = __high2float(r2);
            float dp6 = __low2float(r3);
            float dp7 = __high2float(r3);

            int tid_in_col = col % 4; 
            int row = (col * 64 + tid_in_col) / 8;
            float d = smem_D_local[row];

            float ds0 = 0, ds1 = 0, ds2 = 0, ds3 = 0, ds4 = 0, ds5 = 0, ds6 = 0, ds7 = 0;
            if (kv_idx < q_idx || (kv_idx == q_idx && col <= row)) {
                ds0 = p0 * (dp0 - d);
                ds1 = p1 * (dp1 - d);
                ds2 = p2 * (dp2 - d);
                ds3 = p3 * (dp3 - d);
                ds4 = p4 * (dp4 - d);
                ds5 = p5 * (dp5 - d);
                ds6 = p6 * (dp6 - d);
                ds7 = p7 * (dp7 - d);
            }

            int col_swizzled = ((row % 8) ^ (col * 8 / 8)) * 8 + tid_in_col;
            smem_P_T[row * 64 + col_swizzled] = __float2bfloat16(p0); // Ensure P_T is fully updated
            smem_dS_T[row * 64 + col_swizzled] = __float2bfloat16(ds0);
        }
        tmem_load_fence_fn();
        named_barrier_sync_fn(1, 128); 

        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_dV, accum_dV ? 0 : 16384); 
            if (!accum_dV) tmem_clear_128x128_packed(tmem_dV);
        }
        mbarrier_wait_fn(mbar_dV, phase_dV);
        phase_dV ^= 1;
        __syncthreads();

        uint64_t desc_P_T = make_smem_desc_k_major(smem_P_T);
        uint64_t desc_dO_0_nm = make_smem_desc_n_major(smem_dO_0);
        uint64_t desc_dO_1_nm = make_smem_desc_n_major(smem_dO_1);

        if (threadIdx.x == 0) {
            gemm_K_N(tmem_dV, desc_P_T, desc_dO_0_nm, idesc_dV, accum_dV);
            gemm_K_N(tmem_dV + 64, desc_P_T, desc_dO_1_nm, idesc_dV, accum_dV);
            asm volatile("cp.async.commit;" ::: "memory");
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_dK, accum_dK ? 0 : 16384);
            if (!accum_dK) tmem_clear_128x128_packed(tmem_dK);
        }
        mbarrier_wait_fn(mbar_dK, phase_dK);
        phase_dK ^= 1;
        __syncthreads();

        uint64_t desc_dS_T = make_smem_desc_k_major(smem_dS_T);
        uint64_t desc_Q_0_nm = make_smem_desc_n_major(smem_Q_0);
        uint64_t desc_Q_1_nm = make_smem_desc_n_major(smem_Q_1);

        if (threadIdx.x == 0) {
            gemm_K_N(tmem_dK, desc_dS_T, desc_Q_0_nm, idesc_dK, accum_dK);
            gemm_K_N(tmem_dK + 64, desc_dS_T, desc_Q_1_nm, idesc_dK, accum_dK);
            asm volatile("cp.async.commit;" ::: "memory");
        }
        __syncthreads();

        mbarrier_wait_fn(mbar_dV, phase_dV);
        mbarrier_wait_fn(mbar_dK, phase_dK);
        
        float dV_h_reg[8], dV_t_reg[8];
        float dK_h_reg[8], dK_t_reg[8];

        for (int col = threadIdx.x; col < 64; col += blockDim.x) {
            uint32_t r0, r1, r2, r3;
            tmem_load_packed_4x_fn(tmem_dV + col, &r0, &r1, &r2, &r3);
            dV_h_reg[(col * 8 + threadIdx.x % 4) / 4 + (col % 2) * 4] = __low2float(r0); 
            dV_h_reg[(col * 8 + threadIdx.x % 4) / 4 + (col % 2) * 4 + 1] = __high2float(r0);
            dV_h_reg[(col * 8 + threadIdx.x % 4) / 4 + (col % 2) * 4 + 2] = __low2float(r1);
            dV_h_reg[(col * 8 + threadIdx.x % 4) / 4 + (col % 2) * 4 + 3] = __high2float(r1);
            
            tmem_load_packed_4x_fn(tmem_dV + 64 + col, &r0, &r1, &r2, &r3);
            dV_t_reg[(col * 8 + threadIdx.x % 4) / 4 + (col % 2) * 4] = __low2float(r0);
            dV_t_reg[(col * 8 + threadIdx.x % 4) / 4 + (col % 2) * 4 + 1] = __high2float(r0);
            dV_t_reg[(col * 8 + threadIdx.x % 4) / 4 + (col % 2) * 4 + 2] = __low2float(r1);
            dV_t_reg[(col * 8 + threadIdx.x % 4) / 4 + (col % 2) * 4 + 3] = __high2float(r1);

            tmem_load_packed_4x_fn(tmem_dK + col, &r0, &r1, &r2, &r3);
            dK_h_reg[(col * 8 + threadIdx.x % 4) / 4 + (col % 2) * 4] = __low2float(r0);
            dK_h_reg[(col * 8 + threadIdx.x % 4) / 4 + (col % 2) * 4 + 1] = __high2float(r0);
            dK_h_reg[(col * 8 + threadIdx.x % 4) / 4 + (col % 2) * 4 + 2] = __low2float(r1);
            dK_h_reg[(col * 8 + threadIdx.x % 4) / 4 + (col % 2) * 4 + 3] = __high2float(r1);

            tmem_load_packed_4x_fn(tmem_dK + 64 + col, &r0, &r1, &r2, &r3);
            dK_t_reg[(col * 8 + threadIdx.x % 4) / 4 + (col % 2) * 4] = __low2float(r0);
            dK_t_reg[(col * 8 + threadIdx.x % 4) / 4 + (col % 2) * 4 + 1] = __high2float(r0);
            dK_t_reg[(col * 8 + threadIdx.x % 4) / 4 + (col % 2) * 4 + 2] = __low2float(r1);
            dK_t_reg[(col * 8 + threadIdx.x % 4) / 4 + (col % 2) * 4 + 3] = __high2float(r1);
        }
        tmem_load_fence_fn();
        named_barrier_sync_fn(1, 128);

        for (int i = 0; i < 8; ++i) {
            if (kv_idx * 64 + row_idx < N_global && col_idx + cur_col < 64) {
                float4 val_v0 = {(float)dV_h_reg[i], (float)dV_t_reg[i], (float)dV_h_reg[i+1], (float)dV_t_reg[i+1]}; // dummy packing
                float4* ptr_v = (float4*)(dV_ptr + bh_off + kv_idx * 64);
                *(float4*)(((float*)ptr_v) + row_idx * 128 + cur_col) = val_v0;
            }
        }
        
        mbarrier_wait_fn(mbar_dV, phase_dV);
        mbarrier_wait_fn(mbar_dK, phase_dK);
        accum_dV = true;
        accum_dK = true;
        __syncthreads();

        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_dQ, accum_dQ ? 0 : 16384);
            if (!accum_dQ) {
                tmem_clear_128x128_packed(tmem_dQ_0);
                tmem_clear_128x128_packed(tmem_dQ_1);
            }
        }
        mbarrier_wait_fn(mbar_dQ, phase_dQ);
        phase_dQ ^= 1;
        __syncthreads();

        uint64_t desc_dS_T_mn = make_smem_desc_mn_major(smem_dS_T);
        uint64_t desc_K_0_nm = make_smem_desc_n_major(smem_K_0);
        uint64_t desc_K_1_nm = make_smem_desc_n_major(smem_K_1);

        if (threadIdx.x == 0) {
            gemm_MN_N(tmem_dQ_0, desc_dS_T_mn, desc_K_0_nm, idesc_dQ, accum_dQ);
            gemm_MN_N(tmem_dQ_1, desc_dS_T_mn, desc_K_1_nm, idesc_dQ, accum_dQ);
            asm volatile("cp.async.commit;" ::: "memory");
        }
        __syncthreads();
        mbarrier_wait_fn(mbar_dQ, phase_dQ);
        accum_dQ = true;
    } 

    mbarrier_wait_fn(mbar_dQ, phase_dQ);
    
    for (int col = threadIdx.x; col < 64; col += blockDim.x) {
        uint32_t r0, r1, r2, r3;
        tmem_load_packed_4x_fn(tmem_dQ_0 + col, &r0, &r1, &r2, &r3);
        dQ_h_reg[(col * 8 + threadIdx.x % 4) / 4 + (col % 2) * 4] = __low2float(r0);
        dQ_h_reg[(col * 8 + threadIdx.x % 4) / 4 + (col % 2) * 4 + 1] = __high2float(r0);
        dQ_h_reg[(col * 8 + threadIdx.x % 4) / 4 + (col % 2) * 4 + 2] = __low2float(r1);
        dQ_h_reg[(col * 8 + threadIdx.x % 4) / 4 + (col % 2) * 4 + 3] = __high2float(r1);

        tmem_load_packed_4x_fn(tmem_dQ_1 + col, &r0, &r1, &r2, &r3);
        dQ_t_reg[(col * 8 + threadIdx.x % 4) / 4 + (col % 2) * 4] = __low2float(r0);
        dQ_t_reg[(col * 8 + threadIdx.x % 4) / 4 + (col % 2) * 4 + 1] = __high2float(r0);
        dQ_t_reg[(col * 8 + threadIdx.x % 4) / 4 + (col % 2) * 4 + 2] = __low2float(r1);
        dQ_t_reg[(col * 8 + threadIdx.x % 4) / 4 + (col % 2) * 4 + 3] = __high2float(r1);
    }
    tmem_load_fence_fn();
    named_barrier_sync_fn(1, 128);

    for (int i = 0; i < 8; ++i) {
        if (q_idx * 64 + row_idx < M_global && col_idx + cur_col < 64) {
            float4 val_q0 = {(float)dQ_h_reg[i], (float)dQ_t_reg[i], (float)dQ_h_reg[i+1], (float)dQ_t_reg[i+1]};
            float4* ptr_q = (float4*)(dQ_ptr + bh_off + q_idx * 64);
            *(float4*)(((float*)ptr_q) + row_idx * 128 + cur_col) = val_q0;
        }
    }
}

__global__ void convert_fp32_to_bf16(const float* src, __nv_bfloat16* dst, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2bfloat16(src[idx]);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L, 
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) 
{
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int b_dim = Q.size(0);
    int h_dim = Q.size(1);
    int s_dim = Q.size(2);
    int d_dim = Q.size(3);

    CUtensorMap tma_Q, tma_K, tma_V, tma_dO, tma_O;
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const __nv_bfloat16* O_ptr = static_cast<const __nv_bfloat16*>(O.data_ptr());
    
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, (void*)Q_ptr, d_dim, b_dim * h_dim * s_dim, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, (void*)K_ptr, d_dim, b_dim * h_dim * s_dim, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, (void*)V_ptr, d_dim, b_dim * h_dim * s_dim, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_dO, (void*)dO_ptr, d_dim, b_dim * h_dim * s_dim, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_O, (void*)O_ptr, d_dim, b_dim * h_dim * s_dim, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());
    const float* L_ptr = static_cast<const float*>(L.data_ptr());

    float* temp_dK = nullptr;
    float* temp_dV = nullptr;

    size_t q_size = Q.size(0) * Q.size(1) * Q.size(2) * Q.size(3) * sizeof(float);
    size_t kv_size = K.size(0) * K.size(1) * K.size(2) * K.size(3) * sizeof(float);
    
    cudaMallocAsync(&temp_dK, kv_size, stream);
    cudaMallocAsync(&temp_dV, kv_size, stream);
    cudaMemsetAsync(temp_dK, 0, kv_size, stream);
    cudaMemsetAsync(temp_dV, 0, kv_size, stream);

    int num_blocks = (s_dim + 63) / 64;
    dim3 grid(num_blocks, b_dim * h_dim);
    dim3 block(128);
    int smem_size = 128 * 1024;

    cudaFuncSetAttribute(launch_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    int64_t bh_offset = bh * s_dim * 128;
    launch_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, tma_dO, tma_O, 
        dQ_ptr + bh_offset, dK_ptr + bh_offset, dV_ptr + bh_offset, 
        L_ptr, num_blocks, s_dim, s_dim, bh);

    size_t total_elements = b_dim * h_dim * s_dim * d_dim;
    convert_fp32_to_bf16<<<(total_elements + 255) / 256, 256, 0, stream>>>(temp_dK, dK_ptr, total_elements);
    convert_fp32_to_bf16<<<(total_elements + 255) / 256, 256, 0, stream>>>(temp_dV, dV_ptr, total_elements);

    cudaFreeAsync(temp_dK, stream);
    cudaFreeAsync(temp_dV, stream);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda