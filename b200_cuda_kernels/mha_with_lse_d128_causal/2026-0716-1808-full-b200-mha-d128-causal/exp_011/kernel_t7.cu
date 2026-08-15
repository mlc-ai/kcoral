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
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];"
        :: "r"(a) : "memory");
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

__global__ void __launch_bounds__(64, 4) mha_opt_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* D, float* lse_data, int S, int D_dim, int H)
{
    setmaxnreg_inc_sync_fn<256>();

    int row_start = blockIdx.x * 64;
    int head_id = blockIdx.y;
    int batch_id = blockIdx.z;

    extern __shared__ char smem_pool[];
    uintptr_t smem_addr = (uintptr_t)smem_pool;
    // Align to 4096 to guarantee perfectly aligned TMEM operations and swizzling boundaries.
    uintptr_t smem_aligned = (smem_addr + 4095) & ~4095; 
    
    char* smem_Q0 = (char*)smem_aligned;
    char* smem_Q1 = smem_Q0 + 8192;
    char* smem_K0_0 = smem_Q1 + 8192;
    char* smem_K1_0 = smem_K0_0 + 8192;
    char* smem_V0_0 = smem_K1_0 + 8192;
    char* smem_V1_0 = smem_V0_0 + 8192;
    char* smem_K0_1 = smem_V1_0 + 8192;
    char* smem_K1_1 = smem_K0_1 + 8192;
    char* smem_V0_1 = smem_K1_1 + 8192;
    char* smem_V1_1 = smem_V0_1 + 8192;
    char* smem_S  = smem_V1_1 + 8192;

    __shared__ __align__(128) float smem_scale[64];
    __shared__ __align__(128) float smem_sum[64];

    int tid = threadIdx.x;
    if (tid < 64) {
        smem_scale[tid] = -INFINITY;
        smem_sum[tid] = 0.0f;
    }

    __shared__ alignas(16) uint32_t tmem_P[1]; 
    __shared__ alignas(16) uint32_t tmem_O0[1]; 
    __shared__ alignas(16) uint32_t tmem_O1[1]; 
    
    if (tid == 0) {
        tmem_alloc_fn(tmem_P, 64);
        tmem_alloc_fn(tmem_O0, 64);
        tmem_alloc_fn(tmem_O1, 64);
    }
    __syncthreads();

    __shared__ alignas(8) uint64_t mbar_Q[1];
    __shared__ alignas(8) uint64_t mbar_KV[2];
    __shared__ alignas(8) uint64_t mbar_P[1];
    __shared__ alignas(8) uint64_t mbar_S[1];
    
    if (tid == 0) {
        init_smem_barrier_fn(&mbar_Q[0], 1);
        init_smem_barrier_fn(&mbar_KV[0], 1);
        init_smem_barrier_fn(&mbar_KV[1], 1);
        init_smem_barrier_fn(&mbar_P[0], 1);
        init_smem_barrier_fn(&mbar_S[0], 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();
    __syncthreads();

    if (tid == 0) {
        int32_t base_coord = batch_id * H * S + head_id * S;
        mbarrier_arrive_and_expect_tx_fn(&mbar_Q[0], 2 * 8192);
        tma_load_2d_fn(&tma_Q, &mbar_Q[0], smem_Q0, 0, base_coord + row_start);
        tma_load_2d_fn(&tma_Q, &mbar_Q[0], smem_Q1, 64, base_coord + row_start);
    }
    mbarrier_wait_fn(&mbar_Q[0], 0);

    uint32_t idesc_A = make_instr_desc_fn(64, 64, 0, 0);
    uint32_t idesc_SV0 = make_instr_desc_fn(64, 64, 0, 1);
    uint32_t idesc_SV1 = make_instr_desc_fn(64, 64, 0, 1);

    float scale_factor = 1.0f / sqrtf(128.0f);

    int phase_KV[2] = {0, 0};
    int phase_P = 0;
    int phase_S = 0;
    int max_j = MIN(S - 1, row_start + 63);
    
    uint32_t tmp_O0[64], tmp_O1[64];
    for(int i = 0; i < 64; i++) { tmp_O0[i] = 0; tmp_O1[i] = 0; }

    int buf_idx = 0;
    
    // Initial load pump
    if (tid == 0) {
        int32_t base_coord = batch_id * H * S + head_id * S;
        mbarrier_arrive_and_expect_tx_fn(&mbar_KV[0], 4 * 8192);
        tma_load_2d_fn(&tma_K, &mbar_KV[0], smem_K0_0, 0, base_coord + 0);
        tma_load_2d_fn(&tma_K, &mbar_KV[0], smem_K1_0, 64, base_coord + 0);
        tma_load_2d_fn(&tma_V, &mbar_KV[0], smem_V0_0, 0, base_coord + 0);
        tma_load_2d_fn(&tma_V, &mbar_KV[0], smem_V1_0, 64, base_coord + 0);
    }

    for (int j = 0; j <= max_j; j += 64) {
        if (j + 64 <= max_j) {
            int next_buf = buf_idx ^ 1;
            if (tid == 0) {
                int32_t base_coord = batch_id * H * S + head_id * S;
                mbarrier_arrive_and_expect_tx_fn(&mbar_KV[next_buf], 4 * 8192);
                char* nK0 = (next_buf == 0) ? smem_K0_0 : smem_K0_1;
                char* nK1 = (next_buf == 0) ? smem_K1_0 : smem_K1_1;
                char* nV0 = (next_buf == 0) ? smem_V0_0 : smem_V0_1;
                char* nV1 = (next_buf == 0) ? smem_V1_0 : smem_V1_1;
                
                tma_load_2d_fn(&tma_K, &mbar_KV[next_buf], nK0, 0, base_coord + j + 64);
                tma_load_2d_fn(&tma_K, &mbar_KV[next_buf], nK1, 64, base_coord + j + 64);
                tma_load_2d_fn(&tma_V, &mbar_KV[next_buf], nV0, 0, base_coord + j + 64);
                tma_load_2d_fn(&tma_V, &mbar_KV[next_buf], nV1, 64, base_coord + j + 64);
            }
        }

        mbarrier_wait_fn(&mbar_KV[buf_idx], phase_KV[buf_idx]);
        phase_KV[buf_idx] ^= 1;
        __syncthreads();

        char* cK0 = (buf_idx == 0) ? smem_K0_0 : smem_K0_1;
        char* cK1 = (buf_idx == 0) ? smem_K1_0 : smem_K1_1;
        char* cV0 = (buf_idx == 0) ? smem_V0_0 : smem_V0_1;
        char* cV1 = (buf_idx == 0) ? smem_V1_0 : smem_V1_1;

        uint32_t accum = 0;
        for (int i = 0; i < 4; i++) {
            uint64_t dq = make_smem_desc_swizzled((char*)smem_Q0 + i * 32, 1, 1024);
            uint64_t dk = make_smem_desc_swizzled((char*)cK0 + i * 32, 1, 1024);
            if (i == 0) accum = 0; else accum = 1;
            umma_f16_cg1_fn(tmem_P[0] + i * 4, dq, dk, idesc_A, accum);
        }
        for (int i = 0; i < 4; i++) {
            uint64_t dq = make_smem_desc_swizzled((char*)smem_Q1 + i * 32, 1, 1024);
            uint64_t dk = make_smem_desc_swizzled((char*)cK1 + i * 32, 1, 1024);
            umma_f16_cg1_fn(tmem_P[0] + i * 4, dq, dk, idesc_A, 1);
        }
        
        umma_commit_1sm_fn(&mbar_P[0]);
        mbarrier_wait_fn(&mbar_P[0], phase_P);
        phase_P ^= 1;
        __syncthreads();

        float row_max = -INFINITY;
        if (tid < 64) {
            for (int c = 0; c < 64; c += 4) {
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
        float old_ceil = (tid < 64) ? smem_scale[r] : -INFINITY;
        float new_ceil = (tid < 64) ? fmaxf(old_ceil, row_max) : -INFINITY;
        
        if (tid < 64) {
            float factor = expf(old_ceil - new_ceil);
            smem_sum[r] *= factor;
            smem_scale[r] = new_ceil;
        }

        if (tid < 64) {
            for(int c=0; c<64; c+=2) {
                tmp_O0[c] *= expf(old_ceil - new_ceil); tmp_O0[c+1] *= expf(old_ceil - new_ceil);
                tmp_O1[c] *= expf(old_ceil - new_ceil); tmp_O1[c+1] *= expf(old_ceil - new_ceil);
            }
        }

        if (tid < 64) {
            for (int c = 0; c < 64; c += 4) {
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
                
                if (g_c0 > g_r || g_c0 >= S) exp0 = 0.0f;
                if (g_c0 + 1 > g_r || g_c0 + 1 >= S) exp1 = 0.0f;
                if (g_c0 + 2 > g_r || g_c0 + 2 >= S) exp2 = 0.0f;
                if (g_c0 + 3 > g_r || g_c0 + 3 >= S) exp3 = 0.0f;
                
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
                
                *(uint32_t*)&((__nv_bfloat16*)smem_S)[tid * 64 + c_swizzled] = packed0;
                *(uint32_t*)&((__nv_bfloat16*)smem_S)[tid * 64 + c_swizzled + 2] = packed1;
            }
        }

        __syncthreads();
        fence_async_shared_fn();

        uint32_t accum_S = 0;
        for (int k = 0; k < 4; k++) {
            uint64_t desc_a_S = make_smem_desc_swizzled((char*)smem_S + k * 32, 1, 1024); 
            uint64_t desc_b_V0 = make_smem_desc_swizzled((char*)cV0 + k * 2048, 8192, 1024); 
            uint64_t desc_b_V1 = make_smem_desc_swizzled((char*)cV1 + k * 2048, 8192, 1024); 

            if (k == 0) accum_S = 0; else accum_S = 1;
            umma_f16_cg1_fn(tmem_O0[0] + k * 4, desc_a_S, desc_b_V0, idesc_SV0, accum_S);
            umma_f16_cg1_fn(tmem_O1[0] + k * 4, desc_a_S, desc_b_V1, idesc_SV1, accum_S);
        }
        
        umma_commit_1sm_fn(&mbar_S[0]);
        mbarrier_wait_fn(&mbar_S[0], phase_S);
        phase_S ^= 1;
        __syncthreads();
        
        if (tid < 64) {
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
        
        buf_idx ^= 1;
    }

    if (tid < 64) {
        int m_idx = row_start + tid;
        if (m_idx < S) {
            __nv_bfloat16* my_D = D + (uint64_t)batch_id * H * S * D_dim + (uint64_t)head_id * S * D_dim;
            
            for (int col = 0; col < 64; col += 2) {
                float f0 = tmp_O0[col];
                float f1 = tmp_O0[col+1];
                __nv_bfloat16 bf0 = __float2bfloat16(f0);
                __nv_bfloat16 bf1 = __float2bfloat16(f1);
                uint32_t packed0 = ((uint32_t)bf1 << 16) | (uint32_t)bf0;
                *(uint32_t*)&my_D[(uint64_t)m_idx * D_dim + col] = packed0;
                
                float f2 = tmp_O1[col];
                float f3 = tmp_O1[col+1];
                __nv_bfloat16 bf2 = __float2bfloat16(f2);
                __nv_bfloat16 bf3 = __float2bfloat16(f3);
                uint32_t packed1 = ((uint32_t)bf3 << 16) | (uint32_t)bf2;
                *(uint32_t*)&my_D[(uint64_t)m_idx * D_dim + 64 + col] = packed1;
            }
        }
    }

    if (tid < 64) {
        int m_idx = row_start + tid;
        if (m_idx < S) {
            lse_data[(uint64_t)batch_id * H * S + (uint64_t)head_id * S + m_idx] = smem_scale[tid] + logf(smem_sum[tid]);
        }
    }

    __syncthreads();
    if (tid == 0) {
        tmem_dealloc_fn(tmem_P[0], 64);
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

  dim3 grid((S + 63) / 64, H, B);
  dim3 block(64);

  // Request additional dynamic shared memory space (approx 96 KB)
  int smem_size = 98304;
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