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
#include <nvcuda.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

#define CU_CHECK_DRIVER(call) do {                                 \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        fprintf(stderr, "CUDA Driver error %d at %s:%d\n",         \
                (int)_e, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

#define MIN(A,B) ((A)<(B) ? (A) : (B))

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
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

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
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

__device__ __forceinline__ void umma_commit_1sm_fn(uint64_t* bar) {
    nvcuda::tcgen05::commit<1>(bar);
}

__device__ __forceinline__ uint64_t make_smem_desc_swizzled(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    uint32_t base_offset = (addr >> 7) & 0x7;
    d |= ((uint64_t)base_offset << 49);
    d |= (uint64_t)2 << 61;   // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, uint32_t a_major, uint32_t b_major) {
    uint32_t d = 0;
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= (a_major << 15);   
    d |= (b_major << 16);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapDataType dataType, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d,
        dataType,
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

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

namespace tvm_ffi_mha {

__global__ void __launch_bounds__(128, 1) mha_opt_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* D, float* lse_data, int S, int D_dim, int H)
{
    setmaxnreg_inc_sync_fn<256>();

    int row_start = blockIdx.x * 128;
    int head_id = blockIdx.y;

    extern __shared__ char smem_pool[];
    uintptr_t smem_addr = (uintptr_t)smem_pool;
    uintptr_t smem_aligned = (smem_addr + 4095) & ~4095; 
    
    char* smem_Q0 = (char*)smem_aligned;
    char* smem_Q1 = smem_Q0 + 16384;
    char* smem_K0 = smem_Q1 + 16384;
    char* smem_K1 = smem_K0 + 16384;
    char* smem_S  = smem_K1 + 16384;

    __shared__ __align__(128) float smem_scale[128];
    __shared__ __align__(128) float smem_sum[128];

    int tid = threadIdx.x;
    if (tid < 128) {
        smem_scale[tid] = -INFINITY;
        smem_sum[tid] = 0.0f;
    }

    __shared__ alignas(16) uint32_t tmem_P[1]; 
    __shared__ alignas(16) uint32_t tmem_O0[1]; 
    __shared__ alignas(16) uint32_t tmem_O1[1]; 
    
    if (tid == 0) {
        tmem_alloc_fn(tmem_P, 128);
        tmem_alloc_fn(tmem_O0, 64);
        tmem_alloc_fn(tmem_O1, 64);
    }
    __syncthreads();

    __shared__ alignas(8) uint64_t mbar_Q[1];
    __shared__ alignas(8) uint64_t mbar_KV[1];
    
    if (tid == 0) {
        init_smem_barrier_fn(&mbar_Q[0], 1);
        init_smem_barrier_fn(&mbar_KV[0], 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();
    __syncthreads();

    if (tid == 0) {
        int32_t base_coord = head_id * S;
        mbarrier_arrive_and_expect_tx_fn(&mbar_Q[0], 4 * 8192);
        tma_load_2d_fn(&tma_Q, &mbar_Q[0], smem_Q0, 0, base_coord + row_start);
        tma_load_2d_fn(&tma_Q, &mbar_Q[0], smem_Q0 + 8192, 64, base_coord + row_start);
        
        tma_load_2d_fn(&tma_Q, &mbar_Q[0], smem_Q1, 0, base_coord + row_start + 64);
        tma_load_2d_fn(&tma_Q, &mbar_Q[0], smem_Q1 + 8192, 64, base_coord + row_start + 64);
    }
    mbarrier_wait_fn(&mbar_Q[0], 0);

    uint64_t desc_q0_0 = make_smem_desc_swizzled(smem_Q0 + 0, 1, 1024);
    uint64_t desc_q0_1 = make_smem_desc_swizzled(smem_Q0 + 1024, 1, 1024);
    uint64_t desc_q0_2 = make_smem_desc_swizzled(smem_Q0 + 2048, 1, 1024);
    uint64_t desc_q0_3 = make_smem_desc_swizzled(smem_Q0 + 3072, 1, 1024);

    uint64_t desc_q1_0 = make_smem_desc_swizzled(smem_Q1 + 0, 1, 1024);
    uint64_t desc_q1_1 = make_smem_desc_swizzled(smem_Q1 + 1024, 1, 1024);
    uint64_t desc_q1_2 = make_smem_desc_swizzled(smem_Q1 + 2048, 1, 1024);
    uint64_t desc_q1_3 = make_smem_desc_swizzled(smem_Q1 + 3072, 1, 1024);

    uint64_t desc_k0_0 = make_smem_desc_swizzled(smem_K0 + 0, 1, 1024);
    uint64_t desc_k0_1 = make_smem_desc_swizzled(smem_K0 + 1024, 1, 1024);
    uint64_t desc_k0_2 = make_smem_desc_swizzled(smem_K0 + 2048, 1, 1024);
    uint64_t desc_k0_3 = make_smem_desc_swizzled(smem_K0 + 3072, 1, 1024);

    uint64_t desc_k1_0 = make_smem_desc_swizzled(smem_K1 + 0, 1, 1024);
    uint64_t desc_k1_1 = make_smem_desc_swizzled(smem_K1 + 1024, 1, 1024);
    uint64_t desc_k1_2 = make_smem_desc_swizzled(smem_K1 + 2048, 1, 1024);
    uint64_t desc_k1_3 = make_smem_desc_swizzled(smem_K1 + 3072, 1, 1024);

    uint64_t desc_q0_array[4] = {desc_q0_0, desc_q0_1, desc_q0_2, desc_q0_3};
    uint64_t desc_q1_array[4] = {desc_q1_0, desc_q1_1, desc_q1_2, desc_q1_3};
    uint64_t desc_k0_array[4] = {desc_k0_0, desc_k0_1, desc_k0_2, desc_k0_3};
    uint64_t desc_k1_array[4] = {desc_k1_0, desc_k1_1, desc_k1_2, desc_k1_3};

    uint64_t desc_s_0 = make_smem_desc_swizzled(smem_S + 0, 1, 1024);
    uint64_t desc_s_1 = make_smem_desc_swizzled(smem_S + 1024, 1, 1024);
    uint64_t desc_s_2 = make_smem_desc_swizzled(smem_S + 2048, 1, 1024);
    uint64_t desc_s_3 = make_smem_desc_swizzled(smem_S + 3072, 1, 1024);
    uint64_t desc_s_array[4] = {desc_s_0, desc_s_1, desc_s_2, desc_s_3};

    uint64_t desc_s1_0 = make_smem_desc_swizzled(smem_S + 8192 + 0, 1, 1024);
    uint64_t desc_s1_1 = make_smem_desc_swizzled(smem_S + 8192 + 1024, 1, 1024);
    uint64_t desc_s1_2 = make_smem_desc_swizzled(smem_S + 8192 + 2048, 1, 1024);
    uint64_t desc_s1_3 = make_smem_desc_swizzled(smem_S + 8192 + 3072, 1, 1024);
    uint64_t desc_s1_array[4] = {desc_s1_0, desc_s1_1, desc_s1_2, desc_s1_3};

    uint64_t desc_v0_0 = make_smem_desc_swizzled(smem_K0 + 0, 8192, 1024);
    uint64_t desc_v0_1 = make_smem_desc_swizzled(smem_K0 + 2048, 8192, 1024);
    uint64_t desc_v0_2 = make_smem_desc_swizzled(smem_K0 + 4096, 8192, 1024);
    uint64_t desc_v0_3 = make_smem_desc_swizzled(smem_K0 + 6144, 8192, 1024);
    uint64_t desc_v0_array[4] = {desc_v0_0, desc_v0_1, desc_v0_2, desc_v0_3};

    uint64_t desc_v1_0 = make_smem_desc_swizzled(smem_K1 + 0, 8192, 1024);
    uint64_t desc_v1_1 = make_smem_desc_swizzled(smem_K1 + 2048, 8192, 1024);
    uint64_t desc_v1_2 = make_smem_desc_swizzled(smem_K1 + 4096, 8192, 1024);
    uint64_t desc_v1_3 = make_smem_desc_swizzled(smem_K1 + 6144, 8192, 1024);
    uint64_t desc_v1_array[4] = {desc_v1_0, desc_v1_1, desc_v1_2, desc_v1_3};

    uint32_t idesc_A = make_instr_desc_fn(128, 128, 0, 0);
    uint32_t idesc_SV0 = make_instr_desc_fn(128, 64, 0, 1);
    uint32_t idesc_SV1 = make_instr_desc_fn(128, 64, 0, 1);

    float scale_factor = 1.0f / sqrtf(128.0f);

    int phase_K = 0;
    int max_j = MIN(S - 1, row_start + 127);
    
    uint32_t tmp_O0[64], tmp_O1[64];
    for(int i = 0; i < 64; i++) { tmp_O0[i] = 0; tmp_O1[i] = 0; }

    for (int j = 0; j <= max_j; j += 128) {
        if (tid == 0) {
            int32_t base_coord = head_id * S;
            mbarrier_arrive_and_expect_tx_fn(&mbar_KV[0], 4 * 8192);
            tma_load_2d_fn(&tma_K, &mbar_KV[0], smem_K0, 0, base_coord + j);
            tma_load_2d_fn(&tma_K, &mbar_KV[0], smem_K0 + 8192, 64, base_coord + j);
            
            tma_load_2d_fn(&tma_K, &mbar_KV[0], smem_K1, 0, base_coord + j + 64);
            tma_load_2d_fn(&tma_K, &mbar_KV[0], smem_K1 + 8192, 64, base_coord + j + 64);
        }
        mbarrier_wait_fn(&mbar_KV[0], phase_K);
        phase_K ^= 1;
        
        __syncthreads();

        uint32_t accum = 0;
        for (int i = 0; i < 4; i++) {
            if (i == 0) accum = 0; else accum = 1;
            umma_f16_cg1_fn(tmem_P[0] + i * 64, desc_q0_array[i], desc_k0_array[i], idesc_A, accum);
        }
        for (int i = 0; i < 4; i++) {
            umma_f16_cg1_fn(tmem_P[0] + 4096 + i * 64, desc_q1_array[i], desc_k1_array[i], idesc_A, 1);
        }
        
        umma_commit_1sm_fn(&mbar_KV[0]);
        mbarrier_wait_fn(&mbar_KV[0], phase_K);
        phase_K ^= 1;
        __syncthreads();

        float row_max = -INFINITY;
        if (tid < 128) {
            for (int c = 0; c < 128; c += 4) {
                uint32_t r0, r1, r2, r3;
                tmem_load_4x_fn(tmem_P[0] + c, &r0, &r1, &r2, &r3);
                tmem_load_fence_fn();
                
                float f0 = __uint_as_float(r0);
                float f1 = __uint_as_float(r1);
                float f2 = __uint_as_float(r2);
                float f3 = __uint_as_float(r3);
                
                int g_r = row_start + tid;
                int g_c0 = j + c;
                if (g_c0 > g_r || g_c0 >= S) f0 = -INFINITY;
                if (g_c0 + 1 > g_r || g_c0 + 1 >= S) f1 = -INFINITY;
                if (g_c0 + 2 > g_r || g_c0 + 2 >= S) f2 = -INFINITY;
                if (g_c0 + 3 > g_r || g_c0 + 3 >= S) f3 = -INFINITY;
                
                if (g_r < S && g_c0 < S) f0 *= scale_factor;
                if (g_r < S && g_c0 + 1 < S) f1 *= scale_factor;
                if (g_r < S && g_c0 + 2 < S) f2 *= scale_factor;
                if (g_r < S && g_c0 + 3 < S) f3 *= scale_factor;
                
                if (f0 > row_max) row_max = f0;
                if (f1 > row_max) row_max = f1;
                if (f2 > row_max) row_max = f2;
                if (f3 > row_max) row_max = f3;
            }
        }

        __syncthreads();
        
        int r = tid;
        float old_ceil = smem_scale[r];
        float new_ceil = fmaxf(old_ceil, row_max);
        
        float factor = expf(old_ceil - new_ceil);
        smem_sum[r] *= factor;
        smem_scale[r] = new_ceil;

        if (tid < 128) {
            for(int c=0; c<64; c+=2) {
                tmp_O0[c] *= factor; tmp_O0[c+1] *= factor;
                tmp_O1[c] *= factor; tmp_O1[c+1] *= factor;
            }
        }

        if (tid < 128) {
            for (int c = 0; c < 128; c += 4) {
                uint32_t r0, r1, r2, r3;
                tmem_load_4x_fn(tmem_P[0] + c, &r0, &r1, &r2, &r3);
                tmem_load_fence_fn();
                
                float f0 = __uint_as_float(r0);
                float f1 = __uint_as_float(r1);
                float f2 = __uint_as_float(r2);
                float f3 = __uint_as_float(r3);
                
                int g_r = row_start + tid;
                int g_c0 = j + c;
                if (g_c0 > g_r || g_c0 >= S) f0 = -INFINITY;
                if (g_c0 + 1 > g_r || g_c0 + 1 >= S) f1 = -INFINITY;
                if (g_c0 + 2 > g_r || g_c0 + 2 >= S) f2 = -INFINITY;
                if (g_c0 + 3 > g_r || g_c0 + 3 >= S) f3 = -INFINITY;
                
                if (g_r < S && g_c0 < S) f0 *= scale_factor;
                if (g_r < S && g_c0 + 1 < S) f1 *= scale_factor;
                if (g_r < S && g_c0 + 2 < S) f2 *= scale_factor;
                if (g_r < S && g_c0 + 3 < S) f3 *= scale_factor;
                
                float exp0 = expf(f0 - new_ceil);
                float exp1 = expf(f1 - new_ceil);
                float exp2 = expf(f2 - new_ceil);
                float exp3 = expf(f3 - new_ceil);
                
                smem_sum[r] += exp0 + exp1 + exp2 + exp3;
                
                float out0 = exp0 / smem_sum[r];
                float out1 = exp1 / smem_sum[r];
                float out2 = exp2 / smem_sum[r];
                float out3 = exp3 / smem_sum[r];
                
                __nv_bfloat16 prob_bf16[4];
                prob_bf16[0] = __float2bfloat16(out0);
                prob_bf16[1] = __float2bfloat16(out1);
                prob_bf16[2] = __float2bfloat16(out2);
                prob_bf16[3] = __float2bfloat16(out3);

                int c_swizzled = ((tid % 8) ^ (c / 8)) * 8 + (c % 8);
                uint32_t packed0 = ((uint32_t)prob_bf16[1] << 16) | (uint32_t)prob_bf16[0];
                uint32_t packed1 = ((uint32_t)prob_bf16[3] << 16) | (uint32_t)prob_bf16[2];
                
                *(uint32_t*)&((__nv_bfloat16*)smem_S)[tid * 128 + c_swizzled] = packed0;
                *(uint32_t*)&((__nv_bfloat16*)smem_S)[tid * 128 + c_swizzled + 2] = packed1;
            }
        }

        __syncthreads();
        fence_async_shared_fn();

        if (tid == 0) {
            int32_t base_coord = head_id * S;
            mbarrier_arrive_and_expect_tx_fn(&mbar_KV[0], 4 * 8192);
            tma_load_2d_fn(&tma_V, &mbar_KV[0], smem_K0, 0, base_coord + j);
            tma_load_2d_fn(&tma_V, &mbar_KV[0], smem_K0 + 8192, 64, base_coord + j);
            
            tma_load_2d_fn(&tma_V, &mbar_KV[0], smem_K1, 0, base_coord + j + 64);
            tma_load_2d_fn(&tma_V, &mbar_KV[0], smem_K1 + 8192, 64, base_coord + j + 64);
        }
        mbarrier_wait_fn(&mbar_KV[0], phase_K);
        phase_K ^= 1;

        for (int step = 0; step < 4; step++) {
            uint64_t desc_a_S_step = desc_s_array[step];
            uint64_t desc_b_V0_step = desc_v0_array[step];

            uint64_t desc_a_S1_step = desc_s1_array[step];
            uint64_t desc_b_V1_step = desc_v1_array[step]; 

            if (step == 0) accum = 0; else accum = 1;
            umma_f16_cg1_fn(tmem_O0[0] + step * 8192, desc_a_S_step, desc_b_V0_step, idesc_SV0, accum);
            umma_f16_cg1_fn(tmem_O1[0] + step * 8192, desc_a_S1_step, desc_b_V1_step, idesc_SV1, accum);
        }
        
        umma_commit_1sm_fn(&mbar_KV[0]);
        mbarrier_wait_fn(&mbar_KV[0], phase_K);
        phase_K ^= 1;
        __syncthreads();
        
        if (tid < 128) {
            for (int c = 0; c < 64; c += 4) {
                uint32_t r0, r1, r2, r3;
                tmem_load_4x_fn(tmem_O0[0] + c, &r0, &r1, &r2, &r3);
                tmem_load_fence_fn();
                tmp_O0[c] += __uint_as_float(r0);
                tmp_O0[c+1] += __uint_as_float(r1);
                tmp_O0[c+2] += __uint_as_float(r2);
                tmp_O0[c+3] += __uint_as_float(r3);

                tmem_load_4x_fn(tmem_O1[0] + c, &r0, &r1, &r2, &r3);
                tmem_load_fence_fn();
                tmp_O1[c] += __uint_as_float(r0);
                tmp_O1[c+1] += __uint_as_float(r1);
                tmp_O1[c+2] += __uint_as_float(r2);
                tmp_O1[c+3] += __uint_as_float(r3);
            }
        }
        __syncthreads();
    }

    if (tid < 128) {
        int m_idx = row_start + tid;
        if (m_idx < S) {
            __nv_bfloat16* my_D = D + (uint64_t)(head_id / H) * H * S * D_dim + (uint64_t)(head_id % H) * S * D_dim;
            
            for (int col = 0; col < 64; col += 4) {
                float f0 = tmp_O0[col];
                float f1 = tmp_O0[col+1];
                float f2 = tmp_O0[col+2];
                float f3 = tmp_O0[col+3];
                
                __nv_bfloat16 bf0 = __float2bfloat16(f0);
                __nv_bfloat16 bf1 = __float2bfloat16(f1);
                __nv_bfloat16 bf2 = __float2bfloat16(f2);
                __nv_bfloat16 bf3 = __float2bfloat16(f3);
                
                uint32_t packed0 = ((uint32_t)bf1 << 16) | (uint32_t)bf0;
                uint32_t packed1 = ((uint32_t)bf3 << 16) | (uint32_t)bf2;
                
                *(uint32_t*)&my_D[(uint64_t)m_idx * D_dim + col] = packed0;
                *(uint32_t*)&my_D[(uint64_t)m_idx * D_dim + col + 2] = packed1;
            }
            
            for (int col = 0; col < 64; col += 4) {
                float f0 = tmp_O1[col];
                float f1 = tmp_O1[col+1];
                float f2 = tmp_O1[col+2];
                float f3 = tmp_O1[col+3];
                
                __nv_bfloat16 bf0 = __float2bfloat16(f0);
                __nv_bfloat16 bf1 = __float2bfloat16(f1);
                __nv_bfloat16 bf2 = __float2bfloat16(f2);
                __nv_bfloat16 bf3 = __float2bfloat16(f3);
                
                uint32_t packed0 = ((uint32_t)bf1 << 16) | (uint32_t)bf0;
                uint32_t packed1 = ((uint32_t)bf3 << 16) | (uint32_t)bf2;
                
                *(uint32_t*)&my_D[(uint64_t)m_idx * D_dim + 64 + col] = packed0;
                *(uint32_t*)&my_D[(uint64_t)m_idx * D_dim + 64 + col + 2] = packed1;
            }
        }
    }

    if (tid < 128) {
        int m_idx = row_start + tid;
        if (m_idx < S) {
            lse_data[(uint64_t)(head_id / H) * H * S + (uint64_t)(head_id % H) * S + m_idx] = smem_scale[tid] + logf(smem_sum[tid]);
        }
    }

    __syncthreads();
    if (tid == 0) {
        tmem_dealloc_fn(tmem_P[0], 128);
        tmem_dealloc_fn(tmem_O0[0], 64);
        tmem_dealloc_fn(tmem_O1[0], 64);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  
  int64_t B = Q.size(0);
  int64_t H = Q.size(1);
  int64_t S = Q.size(2);
  int64_t D_dim = Q.size(3);
  
  CUtensorMap tma_Q, tma_K, tma_V;
  CU_CHECK_DRIVER(create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), D_dim, B * H * S, 64, 64, 
    CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    CU_TENSOR_MAP_SWIZZLE_128B, 
    CU_TENSOR_MAP_L2_PROMOTION_L2_128B, 
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  
  CU_CHECK_DRIVER(create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), D_dim, B * H * S, 64, 64, 
    CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    CU_TENSOR_MAP_SWIZZLE_128B, 
    CU_TENSOR_MAP_L2_PROMOTION_L2_128B, 
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
  CU_CHECK_DRIVER(create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), D_dim, B * H * S, 64, 64, 
    CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    CU_TENSOR_MAP_SWIZZLE_128B, 
    CU_TENSOR_MAP_L2_PROMOTION_L2_128B, 
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

  __nv_bfloat16* D_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
  float* lse_ptr = static_cast<float*>(LSE.data_ptr());

  dim3 grid((S + 127) / 128, B * H);
  dim3 block(128);

  int smem_size = 7 * 8192 + 4096;
  CUDA_CHECK(cudaFuncSetAttribute(tvm_ffi_mha::mha_opt_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
  tvm_ffi_mha::mha_opt_kernel<<<grid, block, smem_size, stream>>>(
      tma_Q, tma_K, tma_V, D_ptr, lse_ptr, S, D_dim, H
  );
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha::run);

} // namespace tvm_ffi_mha