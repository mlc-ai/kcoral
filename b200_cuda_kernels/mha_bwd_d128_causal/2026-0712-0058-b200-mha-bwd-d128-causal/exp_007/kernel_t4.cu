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
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, globalDim, globalStrides,
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

__device__ __forceinline__ uint32_t pack_f32_to_bf16(float f) {
    __nv_bfloat16 b = __float2bfloat16(f);
    return *reinterpret_cast<uint32_t*>(&b);
}

__device__ __forceinline__ float __low2float(uint32_t x) {
    uint16_t low = x & 0xFFFF;
    __nv_bfloat16 a;
    *reinterpret_cast<uint16_t*>(&a) = low;
    return __bfloat162float(a);
}

__device__ __forceinline__ float __high2float(uint32_t x) {
    uint16_t high = (x >> 16) & 0xFFFF;
    __nv_bfloat16 a;
    *reinterpret_cast<uint16_t*>(&a) = high;
    return __bfloat162float(a);
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_expect_tx(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.expect_tx.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
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

__device__ __forceinline__ void tmem_store_packed_4x_fn(uint32_t col, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.pack::16b.b32 {%0, %1, %2, %3}, [%4];"
                 :: "r"(v0), "r"(v1), "r"(v2), "r"(v3), "r"(col));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_clear_128x128_packed(uint32_t tmem_addr) {
    for (int c = threadIdx.x; c < 64; c += blockDim.x) {
        asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 %0, [%1];" :: "r"(0), "r"(tmem_addr + c));
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

__device__ __forceinline__ void gemm_K_K(uint32_t tmem_D, uint64_t desc_A, uint64_t desc_B, uint32_t idesc, bool accum) {
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
        :: "r"(tmem_D), "r"(curr_desc_A), "r"(curr_desc_B), "r"(idesc), "r"(4), "r"(accum_flag));
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
    __nv_bfloat16* dQ_ptr, float* temp_dK, float* temp_dV, 
    const float* L_ptr, int num_blocks, int M_global, int N_global,
    const __nv_bfloat16* Q_ptr, const __nv_bfloat16* K_ptr, 
    const __nv_bfloat16* V_ptr, const __nv_bfloat16* dO_ptr, 
    const __nv_bfloat16* O_ptr, int S_len) 
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
        tmem_alloc_fn(&tmem_base, 512);
    }
    __syncthreads();
    
    uint32_t tmem_S_T = tmem_base;
    uint32_t tmem_dP_T = tmem_base + 64;
    uint32_t tmem_dS_T = tmem_base + 128;
    uint32_t tmem_P_T = tmem_base + 192;
    uint32_t tmem_dV = tmem_base + 256;
    uint32_t tmem_dK = tmem_base + 320;
    uint32_t tmem_dQ_0 = tmem_base + 384;
    uint32_t tmem_dQ_1 = tmem_base + 448;

    int q_idx = blockIdx.x;
    int bh = blockIdx.y;
    float attn_scale = 1.0f / sqrtf(128.0f);

    const __nv_bfloat16* Q_flat = Q_ptr + bh * S_len * 128 + q_idx * 64 * 128;
    const __nv_bfloat16* K_flat = K_ptr + bh * S_len * 128;
    const __nv_bfloat16* V_flat = V_ptr + bh * S_len * 128;
    const __nv_bfloat16* dO_flat = dO_ptr + bh * S_len * 128;
    const __nv_bfloat16* O_flat = O_ptr + bh * S_len * 128;

    // Load Q_0, Q_1
    for(int i = threadIdx.x; i < 512; i += blockDim.x) {
        int r = (i / 8) + ((i % 8) / 4) * 8;
        int c_swizzled = swizzle_128B((i % 4) * 8 + ((i % 8) % 4), r);
        
        *(uint4*)&smem_Q_0[r * 64 + c_swizzled] = *(uint4*)&Q_flat[i * 8];
        *(uint4*)&smem_Q_1[r * 64 + c_swizzled] = *(uint4*)&Q_flat[64 * 128 + i * 8];
    }
    
    // Load O_0, O_1
    for(int i = threadIdx.x; i < 512; i += blockDim.x) {
        int r = (i / 8) + ((i % 8) / 4) * 8;
        int c_swizzled = swizzle_128B((i % 4) * 8 + ((i % 8) % 4), r);
        
        *(uint4*)&smem_O_0[r * 64 + c_swizzled] = *(uint4*)&O_flat[i * 8];
        *(uint4*)&smem_O_1[r * 64 + c_swizzled] = *(uint4*)&O_flat[64 * 128 + i * 8];
    }
    
    // Load dO_0, dO_1
    for(int i = threadIdx.x; i < 512; i += blockDim.x) {
        int r = (i / 8) + ((i % 8) / 4) * 8;
        int c_swizzled = swizzle_128B((i % 4) * 8 + ((i % 8) % 4), r);
        
        *(uint4*)&smem_dO_0[r * 64 + c_swizzled] = *(uint4*)&dO_flat[i * 8];
        *(uint4*)&smem_dO_1[r * 64 + c_swizzled] = *(uint4*)&dO_flat[64 * 128 + i * 8];
    }

    __syncthreads();

    if (threadIdx.x < 64) {
        float d0 = 0, d1 = 0;
        for (int col = 0; col < 64; ++col) {
            int swizzled_idx = swizzle_128B(col, threadIdx.x);
            d0 += __bfloat162float(smem_O_0[threadIdx.x * 64 + swizzled_idx]) * 
                  __bfloat162float(smem_dO_0[threadIdx.x * 64 + swizzled_idx]);
            d1 += __bfloat162float(smem_O_1[threadIdx.x * 64 + swizzled_idx]) * 
                  __bfloat162float(smem_dO_1[threadIdx.x * 64 + swizzled_idx]);
        }
        smem_D_local[threadIdx.x] = d0 + d1;
        smem_LSE_local[threadIdx.x] = L_ptr[bh * S_len + q_idx * 64 + threadIdx.x];
    }
    __syncthreads();

    float dQ_h_regs[8] = {0}, dQ_t_regs[8] = {0};

    uint32_t idesc_S = make_instr_desc_fn_major(64, 64, 0, 0);
    uint32_t idesc_dP = make_instr_desc_fn_major(64, 64, 0, 0);
    uint32_t idesc_dV = make_instr_desc_fn_major(64, 64, 0, 1);
    uint32_t idesc_dK = make_instr_desc_fn_major(64, 64, 1, 1);
    uint32_t idesc_dQ = make_instr_desc_fn_major(64, 64, 1, 1);

    uint32_t phase_S = 0, phase_dP = 0, phase_dV = 0, phase_dK = 0, phase_dQ = 0;
    bool accum_dV = false;
    bool accum_dK = false;
    bool accum_dQ = false;

    for (int kv_idx = 0; kv_idx <= q_idx && kv_idx < num_blocks; ++kv_idx) {
        
        const __nv_bfloat16* K_curr_flat = K_flat + kv_idx * 64 * 128;
        const __nv_bfloat16* V_curr_flat = V_flat + kv_idx * 64 * 128;

        for(int i = threadIdx.x; i < 512; i += blockDim.x) {
            int r = (i / 8) + ((i % 8) / 4) * 8;
            int c_swizzled = swizzle_128B((i % 4) * 8 + ((i % 8) % 4), r);
            
            *(uint4*)&smem_K_0[r * 64 + c_swizzled] = *(uint4*)&K_curr_flat[i * 8];
            *(uint4*)&smem_K_1[r * 64 + c_swizzled] = *(uint4*)&K_curr_flat[64 * 128 + i * 8];
            
            *(uint4*)&smem_V_0[r * 64 + c_swizzled] = *(uint4*)&V_curr_flat[i * 8];
            *(uint4*)&smem_V_1[r * 64 + c_swizzled] = *(uint4*)&V_curr_flat[64 * 128 + i * 8];
        }
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_S, 16384);
        }
        named_barrier_sync_fn(1, 128);

        uint64_t desc_K_0 = make_smem_desc_sm100_fn(smem_K_0, 1, 1024);
        uint64_t desc_Q_0 = make_smem_desc_sm100_fn(smem_Q_0, 1, 1024);
        uint64_t desc_K_1 = make_smem_desc_sm100_fn(smem_K_1, 1, 1024);
        uint64_t desc_Q_1 = make_smem_desc_sm100_fn(smem_Q_1, 1, 1024);

        if (threadIdx.x == 0) {
            tmem_clear_128x128_packed(tmem_S_T);
            gemm_K_K(tmem_S_T, desc_K_0, desc_Q_0, idesc_S, false);
            gemm_K_K(tmem_S_T, desc_K_1, desc_Q_1, idesc_S, true);
        }
        mbarrier_wait_fn(mbar_S, phase_S);
        phase_S ^= 1;
        __syncthreads();
        
        for (int col = threadIdx.x; col < 64; col += blockDim.x) {
            uint32_t r0, r1, r2, r3;
            tmem_load_packed_4x_fn(tmem_S_T + col, &r0, &r1, &r2, &r3);
            
            int tid_in_col = col % 4; 
            int row = (col * 64 + tid_in_col) / 8;
            float lse = (row < 64) ? smem_LSE_local[row] : 0.0f;

            float s0_f = __low2float(r0);
            float s1_f = __high2float(r0);
            float s2_f = __low2float(r1);
            float s3_f = __high2float(r1);

            // Causal Masking Logic
            float p0_f = 0, p1_f = 0, p2_f = 0, p3_f = 0;
            int q_idx_local = (col / 8) * 8 + (tid_in_col / 2);
            int k_idx_local = kv_idx * 64 + ((col % 8) * 8 + (tid_in_col % 2));
            bool masked = (k_idx_local <= q_idx_local);

            if (masked) {
                p0_f = expf(s0_f * attn_scale - lse);
                p1_f = expf(s1_f * attn_scale - lse);
                p2_f = expf(s2_f * attn_scale - lse);
                p3_f = expf(s3_f * attn_scale - lse);
            }

            float dp0_f = __low2float(r2);
            float dp1_f = __high2float(r2);
            float dp2_f = __low2float(r3);
            float dp3_f = __high2float(r3);
            
            float d = (row < 64) ? smem_D_local[row] : 0.0f;

            float ds0_f = (masked) ? p0_f * (dp0_f - d) : 0.0f;
            float ds1_f = (masked) ? p1_f * (dp1_f - d) : 0.0f;
            float ds2_f = (masked) ? p2_f * (dp2_f - d) : 0.0f;
            float ds3_f = (masked) ? p3_f * (dp3_f - d) : 0.0f;

            int col_swizzled = (((row % 8) ^ (col / 8)) * 8) + (col % 8);
            
            uint32_t bp0 = pack_f32_to_bf16(p0_f);
            uint32_t bp1 = pack_f32_to_bf16(p1_f);
            uint32_t bp2 = pack_f32_to_bf16(p2_f);
            uint32_t bp3 = pack_f32_to_bf16(p3_f);
            *(uint32_t*)&smem_P_T[row * 64 + col_swizzled] = *(uint32_t*)&bp0;
            *(uint32_t*)&smem_P_T[row * 64 + col_swizzled + 1] = *(uint32_t*)&bp1;
            *(uint32_t*)&smem_P_T[row * 64 + col_swizzled + 2] = *(uint32_t*)&bp2;
            *(uint32_t*)&smem_P_T[row * 64 + col_swizzled + 3] = *(uint32_t*)&bp3;
            
            uint32_t bds0 = pack_f32_to_bf16(ds0_f);
            uint32_t bds1 = pack_f32_to_bf16(ds1_f);
            uint32_t bds2 = pack_f32_to_bf16(ds2_f);
            uint32_t bds3 = pack_f32_to_bf16(ds3_f);
            *(uint32_t*)&smem_dS_T[row * 64 + col_swizzled] = *(uint32_t*)&bds0;
            *(uint32_t*)&smem_dS_T[row * 64 + col_swizzled + 1] = *(uint32_t*)&bds1;
            *(uint32_t*)&smem_dS_T[row * 64 + col_swizzled + 2] = *(uint32_t*)&bds2;
            *(uint32_t*)&smem_dS_T[row * 64 + col_swizzled + 3] = *(uint32_t*)&bds3;
        }
        tmem_load_fence_fn();
        fence_proxy_async_fn();
        named_barrier_sync_fn(1, 128);

        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_dP, 16384);
        }
        named_barrier_sync_fn(1, 128);

        uint64_t desc_V_0 = make_smem_desc_sm100_fn(smem_V_0, 1, 1024);
        uint64_t desc_dO_0 = make_smem_desc_sm100_fn(smem_dO_0, 1, 1024);
        uint64_t desc_V_1 = make_smem_desc_sm100_fn(smem_V_1, 1, 1024);
        uint64_t desc_dO_1 = make_smem_desc_sm100_fn(smem_dO_1, 1, 1024);

        if (threadIdx.x == 0) {
            tmem_clear_128x128_packed(tmem_dP_T);
            gemm_K_K(tmem_dP_T, desc_V_0, desc_dO_0, idesc_dP, false);
            gemm_K_K(tmem_dP_T, desc_V_1, desc_dO_1, idesc_dP, true);
        }
        mbarrier_wait_fn(mbar_dP, phase_dP);
        phase_dP ^= 1;
        __syncthreads();
        
        for (int col = threadIdx.x; col < 64; col += blockDim.x) {
            uint32_t v0, v1, v2, v3;
            *(uint32_t*)&v0 = *(uint32_t*)&smem_P_T[col * 64 + 0];
            *(uint32_t*)&v1 = *(uint32_t*)&smem_P_T[col * 64 + 1];
            *(uint32_t*)&v2 = *(uint32_t*)&smem_P_T[col * 64 + 2];
            *(uint32_t*)&v3 = *(uint32_t*)&smem_P_T[col * 64 + 3];
            tmem_store_packed_4x_fn(tmem_P_T + col, v0, v1, v2, v3);
            
            *(uint32_t*)&v0 = *(uint32_t*)&smem_dS_T[col * 64 + 0];
            *(uint32_t*)&v1 = *(uint32_t*)&smem_dS_T[col * 64 + 1];
            *(uint32_t*)&v2 = *(uint32_t*)&smem_dS_T[col * 64 + 2];
            *(uint32_t*)&v3 = *(uint32_t*)&smem_dS_T[col * 64 + 3];
            tmem_store_packed_4x_fn(tmem_dS_T + col, v0, v1, v2, v3);
        }
        tmem_load_fence_fn();
        __syncthreads();

        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_dV, 16384 * 2);
        }
        named_barrier_sync_fn(1, 128);

        uint64_t desc_P_T = make_smem_desc_sm100_fn(smem_P_T, 1, 1024);
        uint64_t desc_dO_0_nm = make_smem_desc_sm100_fn(smem_dO_0, 8192, 1024);
        uint64_t desc_dO_1_nm = make_smem_desc_sm100_fn(smem_dO_1, 8192, 1024);

        if (threadIdx.x == 0) {
            if (!accum_dV) tmem_clear_128x128_packed(tmem_dV);
            gemm_K_N(tmem_dV, desc_P_T, desc_dO_0_nm, idesc_dV, accum_dV);
            gemm_K_N(tmem_dV + 64, desc_P_T, desc_dO_1_nm, idesc_dV, accum_dV);
        }
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_dK, 16384 * 2);
        }
        named_barrier_sync_fn(1, 128);

        uint64_t desc_dS_T_mn = make_smem_desc_sm100_fn(smem_dS_T, 8192, 1024);
        uint64_t desc_Q_0_nm = make_smem_desc_sm100_fn(smem_Q_0, 8192, 1024);
        uint64_t desc_Q_1_nm = make_smem_desc_sm100_fn(smem_Q_1, 8192, 1024);

        if (threadIdx.x == 0) {
            if (!accum_dK) tmem_clear_128x128_packed(tmem_dK);
            gemm_MN_N(tmem_dK, desc_dS_T_mn, desc_Q_0_nm, idesc_dK, accum_dK);
            gemm_MN_N(tmem_dK + 64, desc_dS_T_mn, desc_Q_1_nm, idesc_dK, accum_dK);
        }
        
        mbarrier_wait_fn(mbar_dV, phase_dV);
        mbarrier_wait_fn(mbar_dK, phase_dK);
        phase_dV ^= 1;
        phase_dK ^= 1;
        
        for (int col = threadIdx.x; col < 64; col += blockDim.x) {
            uint32_t r0, r1, r2, r3;
            tmem_load_packed_4x_fn(tmem_dV + col, &r0, &r1, &r2, &r3);
            float dv0 = __low2float(r0);
            float dv1 = __high2float(r0);
            float dv2 = __low2float(r1);
            float dv3 = __high2float(r1);
            
            int tid_in_col = col % 4;
            int row = (col * 64 + tid_in_col) / 8;
            int global_row = kv_idx * 64 + row;
            
            if (global_row < N_global) {
                if (col < 64) atomicAdd(&temp_dV[bh * S_len * 128 + global_row * 128 + col], dv0);
                if (col + 1 < 64) atomicAdd(&temp_dV[bh * S_len * 128 + global_row * 128 + col + 1], dv1);
                if (col + 2 < 64) atomicAdd(&temp_dV[bh * S_len * 128 + global_row * 128 + col + 2], dv2);
                if (col + 3 < 64) atomicAdd(&temp_dV[bh * S_len * 128 + global_row * 128 + col + 3], dv3);
            }

            tmem_load_packed_4x_fn(tmem_dV + 64 + col, &r0, &r1, &r2, &r3);
            dv0 = __low2float(r0); dv1 = __high2float(r0); dv2 = __low2float(r1); dv3 = __high2float(r1);
            if (global_row < N_global) {
                if (64 + col < 128) atomicAdd(&temp_dV[bh * S_len * 128 + global_row * 128 + 64 + col], dv0);
                if (64 + col + 1 < 128) atomicAdd(&temp_dV[bh * S_len * 128 + global_row * 128 + 64 + col + 1], dv1);
                if (64 + col + 2 < 128) atomicAdd(&temp_dV[bh * S_len * 128 + global_row * 128 + 64 + col + 2], dv2);
                if (64 + col + 3 < 128) atomicAdd(&temp_dV[bh * S_len * 128 + global_row * 128 + 64 + col + 3], dv3);
            }

            tmem_load_packed_4x_fn(tmem_dK + col, &r0, &r1, &r2, &r3);
            float dk0 = __low2float(r0);
            float dk1 = __high2float(r0);
            float dk2 = __low2float(r1);
            float dk3 = __high2float(r1);
            if (global_row < N_global) {
                if (col < 64) atomicAdd(&temp_dK[bh * S_len * 128 + global_row * 128 + col], dk0);
                if (col + 1 < 64) atomicAdd(&temp_dK[bh * S_len * 128 + global_row * 128 + col + 1], dk1);
                if (col + 2 < 64) atomicAdd(&temp_dK[bh * S_len * 128 + global_row * 128 + col + 2], dk2);
                if (col + 3 < 64) atomicAdd(&temp_dK[bh * S_len * 128 + global_row * 128 + col + 3], dk3);
            }

            tmem_load_packed_4x_fn(tmem_dK + 64 + col, &r0, &r1, &r2, &r3);
            dk0 = __low2float(r0); dk1 = __high2float(r0); dk2 = __low2float(r1); dk3 = __high2float(r1);
            if (global_row < N_global) {
                if (64 + col < 128) atomicAdd(&temp_dK[bh * S_len * 128 + global_row * 128 + 64 + col], dk0);
                if (64 + col + 1 < 128) atomicAdd(&temp_dK[bh * S_len * 128 + global_row * 128 + 64 + col + 1], dk1);
                if (64 + col + 2 < 128) atomicAdd(&temp_dK[bh * S_len * 128 + global_row * 128 + 64 + col + 2], dk2);
                if (64 + col + 3 < 128) atomicAdd(&temp_dK[bh * S_len * 128 + global_row * 128 + 64 + col + 3], dk3);
            }
        }
        tmem_load_fence_fn();
        
        accum_dV = true;
        accum_dK = true;
        __syncthreads();

        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_dQ, 16384 * 2);
        }
        named_barrier_sync_fn(1, 128);

        uint64_t desc_dS_T_mn_dQ = make_smem_desc_sm100_fn(smem_dS_T, 8192, 1024);
        uint64_t desc_K_0_nm = make_smem_desc_sm100_fn(smem_K_0, 8192, 1024);
        uint64_t desc_K_1_nm = make_smem_desc_sm100_fn(smem_K_1, 8192, 1024);

        if (threadIdx.x == 0) {
            if (!accum_dQ) {
                tmem_clear_128x128_packed(tmem_dQ_0);
                tmem_clear_128x128_packed(tmem_dQ_1);
            }
            gemm_MN_N(tmem_dQ_0, desc_dS_T_mn_dQ, desc_K_0_nm, idesc_dQ, accum_dQ);
            gemm_MN_N(tmem_dQ_1, desc_dS_T_mn_dQ, desc_K_1_nm, idesc_dQ, accum_dQ);
        }
        mbarrier_wait_fn(mbar_dQ, phase_dQ);
        phase_dQ ^= 1;
        accum_dQ = true;
    } 

    mbarrier_wait_fn(mbar_dQ, phase_dQ);
    __syncthreads();
    
    for (int col = threadIdx.x; col < 64; col += blockDim.x) {
        uint32_t r0, r1, r2, r3;
        tmem_load_packed_4x_fn(tmem_dQ_0 + col, &r0, &r1, &r2, &r3);
        float dq0 = __low2float(r0);
        float dq1 = __high2float(r0);
        float dq2 = __low2float(r1);
        float dq3 = __high2float(r1);
        
        int tid_in_col = col % 4;
        int row = (col * 64 + tid_in_col) / 8;
        int global_row = q_idx * 64 + row;
        
        if (global_row < M_global) {
             dQ_ptr[(uint64_t)bh * S_len * 128 + (uint64_t)global_row * 128 + col] = __float2bfloat16(dq0);
             dQ_ptr[(uint64_t)bh * S_len * 128 + (uint64_t)global_row * 128 + col + 1] = __float2bfloat16(dq1);
             dQ_ptr[(uint64_t)bh * S_len * 128 + (uint64_t)global_row * 128 + col + 2] = __float2bfloat16(dq2);
             dQ_ptr[(uint64_t)bh * S_len * 128 + (uint64_t)global_row * 128 + col + 3] = __float2bfloat16(dq3);
        }

        tmem_load_packed_4x_fn(tmem_dQ_1 + col, &r0, &r1, &r2, &r3);
        dq0 = __low2float(r0); dq1 = __high2float(r0); dq2 = __low2float(r1); dq3 = __high2float(r1);
        
        if (global_row < M_global) {
             dQ_ptr[(uint64_t)bh * S_len * 128 + (uint64_t)global_row * 128 + 64 + col] = __float2bfloat16(dq0);
             dQ_ptr[(uint64_t)bh * S_len * 128 + (uint64_t)global_row * 128 + 64 + col + 1] = __float2bfloat16(dq1);
             dQ_ptr[(uint64_t)bh * S_len * 128 + (uint64_t)global_row * 128 + 64 + col + 2] = __float2bfloat16(dq2);
             dQ_ptr[(uint64_t)bh * S_len * 128 + (uint64_t)global_row * 128 + 64 + col + 3] = __float2bfloat16(dq3);
        }
    }
    tmem_load_fence_fn();
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

    float* temp_dV;
    float* temp_dK;
    size_t kv_size = b_dim * h_dim * s_dim * d_dim * sizeof(float);
    cudaMallocAsync(&temp_dV, kv_size, stream);
    cudaMallocAsync(&temp_dK, kv_size, stream);
    cudaMemsetAsync(temp_dV, 0, kv_size, stream);
    cudaMemsetAsync(temp_dK, 0, kv_size, stream);

    int num_blocks = (s_dim + 63) / 64;
    dim3 grid(num_blocks, b_dim * h_dim);
    dim3 block(128);
    int smem_size = 128 * 1024;

    cudaFuncSetAttribute(launch_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    launch_kernel<<<grid, block, smem_size, stream>>>(
        dQ_ptr, temp_dK, temp_dV, 
        L_ptr, num_blocks, s_dim, s_dim,
        Q_ptr, K_ptr, V_ptr, dO_ptr, O_ptr, s_dim);

    size_t total_elements = b_dim * h_dim * s_dim * d_dim;
    convert_fp32_to_bf16<<<(total_elements + 255) / 256, 256, 0, stream>>>(temp_dV, dV_ptr, total_elements);
    convert_fp32_to_bf16<<<(total_elements + 255) / 256, 256, 0, stream>>>(temp_dK, dK_ptr, total_elements);

    cudaFreeAsync(temp_dV, stream);
    cudaFreeAsync(temp_dK, stream);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda