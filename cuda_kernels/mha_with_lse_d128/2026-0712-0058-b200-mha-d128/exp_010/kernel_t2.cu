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

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
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

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t tmem_base, uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    uint32_t addr = tmem_base | (col & 0xFFFF);
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(addr));
}

__device__ __forceinline__ void tmem_store_4x_fn(uint32_t tmem_base, uint32_t col, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    uint32_t addr = tmem_base | (col & 0xFFFF);
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};"
        :: "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(addr));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_store_fence_fn() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void zero_tmem_64x64_fn(uint32_t tmem_base) {
    for (uint32_t col = 0; col < 64; col += 4) {
        tmem_store_4x_fn(tmem_base, col, 0, 0, 0, 0);
    }
    tmem_store_fence_fn();
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

__device__ __forceinline__ uint32_t make_instr_desc_f16(uint32_t M, uint32_t N, bool a_transpose, bool b_transpose) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= ((uint32_t)a_transpose << 15);  
    d |= ((uint32_t)b_transpose << 16);  
    d |= ((N >> 3) << 17);     
    d |= ((M >> 4) << 24);     
    return d;
}

struct __align__(1024) SharedStorage {
    __align__(128) __nv_bfloat16 s_Q0[4096]; // 64 * 64
    __align__(128) __nv_bfloat16 s_Q1[4096];
    __align__(128) __nv_bfloat16 s_K0[4096];
    __align__(128) __nv_bfloat16 s_K1[4096];
    __align__(128) __nv_bfloat16 s_V0[4096];
    __align__(128) __nv_bfloat16 s_V1[4096];
    __align__(128) __nv_bfloat16 s_P[4096];
    __align__(8) uint64_t mbar_Q[1];
    __align__(8) uint64_t mbar_K[1];
    __align__(8) uint64_t mbar_V[1];
    __align__(8) uint64_t mbar_COMMIT[1];
    __align__(8) uint64_t mbar_epilogue[1];
};

extern __shared__ __align__(128) uint8_t smem_pool[];

__global__ void __launch_bounds__(256) run_kernel(
    const __grid_constant__ CUtensorMap desc_Q,
    const __grid_constant__ CUtensorMap desc_K,
    const __grid_constant__ CUtensorMap desc_V,
    __nv_bfloat16* O, float* LSE,
    int64_t B, int64_t H, int64_t S, int64_t D) 
{
    int cta_offset = blockIdx.x * 1024;
    int head_idx = blockIdx.y;
    int row_start_cta0 = cta_offset + 0;
    int row_start_cta1 = cta_offset + 512;
    int row_start = (cluster_rank_fn() % 2 == 0) ? row_start_cta0 : row_start_cta1;
    
    extern __shared__ char smem_buf[];
    uintptr_t smem_addr = (uintptr_t)smem_buf;
    uintptr_t aligned_addr = (smem_addr + 1023) & ~1023;
    SharedStorage* shared = reinterpret_cast<SharedStorage*>(aligned_addr);

    uint32_t tid = threadIdx.x;
    int quad_idx = tid % 4;

    if (warpId() == 0 && tid == 0) {
        uint32_t tmem_S, tmem_D0, tmem_D1;
        tmem_alloc_fn(&tmem_S, 64);
        tmem_alloc_fn(&tmem_D0, 64);
        tmem_alloc_fn(&tmem_D1, 64);
        
        init_smem_barrier_fn(shared->mbar_Q, 1);
        init_smem_barrier_fn(shared->mbar_K, 1);
        init_smem_barrier_fn(shared->mbar_V, 1);
        init_smem_barrier_fn(shared->mbar_COMMIT, 256);
        init_smem_barrier_fn(shared->mbar_epilogue, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t tmem_S, tmem_D0, tmem_D1;
    // TMEM base addresses are uniformly allocated sequentially across the entire SM, 
    // so CTAs safely agree on the exact physical layout without additional synchronization.
    tmem_S = *(uint32_t*)__cvta_shared_from_generic(&tmem_S);
    tmem_D0 = *(uint32_t*)__cvta_shared_from_generic(&tmem_D0);
    tmem_D1 = *(uint32_t*)__cvta_shared_from_generic(&tmem_D1);

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(shared->mbar_Q, 16384); 
        tma_load_3d_fn(&desc_Q, shared->mbar_Q, shared->s_Q0, 0, row_start, head_idx);
        tma_load_3d_fn(&desc_Q, shared->mbar_Q, shared->s_Q1, 64, row_start, head_idx);
    }
    
    zero_tmem_64x64_fn(tmem_D0);
    zero_tmem_64x64_fn(tmem_D1);

    mbarrier_wait_fn(shared->mbar_Q, 0);
    int phase_Q = 0;

    // Local state for 4 independently tracked rows spanning both CTAs
    float global_max[4] = {-1e20f, -1e20f, -1e20f, -1e20f};
    float global_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    int phase = 0;
    for (int col_start = 0; col_start < S; col_start += 64) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(shared->mbar_K, 16384);
            tma_load_3d_fn(&desc_K, shared->mbar_K, shared->s_K0, 0, col_start, head_idx);
            tma_load_3d_fn(&desc_K, shared->mbar_K, shared->s_K1, 64, col_start, head_idx);
            
            mbarrier_arrive_and_expect_tx_fn(shared->mbar_V, 16384);
            tma_load_3d_fn(&desc_V, shared->mbar_V, shared->s_V0, 0, col_start, head_idx);
            tma_load_3d_fn(&desc_V, shared->mbar_V, shared->s_V1, 64, col_start, head_idx);
        }
        
        __syncthreads();
        mbarrier_wait_fn(shared->mbar_K, phase % 2);
        mbarrier_wait_fn(shared->mbar_V, phase % 2);

        float row_max[4] = {-1e20f, -1e20f, -1e20f, -1e20f};
        for(int i = 0; i < 2; i++) {
            uint32_t my_row = quad_idx + i * 4;
            if (col_start < S && row_start + my_row < B * H * S) {
                if (threadIdx.x == 0) {
                    uint32_t idesc_S = make_instr_desc_f16(128, 64, false, true);
                    
                    // First gemm accumulating local quadrant context bounds
                    for (int k = 0; k < 64; k += 16) {
                        uint64_t desc_q = make_smem_desc_sm100_fn(shared->s_Q0, 1, 1024) + k;
                        uint64_t desc_k = make_smem_desc_sm100_fn(shared->s_K0, 1, 1024) + k * 8;
                        umma_f16_cg2_fn(tmem_S, desc_q, desc_k, idesc_S, (k == 0 && i == 0) ? 0 : 1);
                    }
                    
                    // Second gemm accumulating local quadrant context bounds
                    for (int k = 64; k < 128; k += 16) {
                        uint64_t desc_q = make_smem_desc_sm100_fn(shared->s_Q1, 1, 1024) + k;
                        uint64_t desc_k = make_smem_desc_sm100_fn(shared->s_K1, 1, 1024) + k * 8;
                        umma_f16_cg2_fn(tmem_S, desc_q, desc_k, idesc_S, 1);
                    }
                    umma_commit_2sm_fn(shared->mbar_COMMIT);
                }
                
                mbarrier_wait_fn(shared->mbar_COMMIT, phase % 2);
                __syncthreads(); 
                
                for(int col = 0; col < 64; col += 4) {
                    uint32_t r0, r1, r2, r3;
                    tmem_load_4x_fn(tmem_S, col, &r0, &r1, &r2, &r3);
                    float f0 = __uint_as_float(r0) / sqrtf(128.0f);
                    float f1 = __uint_as_float(r1) / sqrtf(128.0f);
                    float f2 = __uint_as_float(r2) / sqrtf(128.0f);
                    float f3 = __uint_as_float(r3) / sqrtf(128.0f);
                    
                    row_max[i] = fmaxf(row_max[i], fmaxf(fmaxf(f0, f1), fmaxf(f2, f3)));
                }
                tmem_load_fence_fn();
            }
        }

        // Reduction within theQuad bounding limits safely across participating CTA pairs 
        for(int i = 0; i < 2; i++) {
            uint32_t my_row = quad_idx + i * 4;
            float max_val = row_max[i];
            max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, 1));
            max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, 2));
            max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, 4));
            
            // Crucial: explicitly reduce max bounds outside of Quad boundaries natively leveraging cross-CTA shuffles
            max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, 32));
            max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, 64));
            
            float sum_val = 0.0f;
            for(int col = 0; col < 64; col += 4) {
                uint32_t r0, r1, r2, r3;
                tmem_load_4x_fn(tmem_S, col, &r0, &r1, &r2, &r3);
                float f0 = __uint_as_float(r0) / sqrtf(128.0f);
                float f1 = __uint_as_float(r1) / sqrtf(128.0f);
                float f2 = __uint_as_float(r2) / sqrtf(128.0f);
                float f3 = __uint_as_float(r3) / sqrtf(128.0f);
                
                float p0 = (col_start + col < S && row_start + my_row < B * H * S) ? fast_exp2f_fn((f0 - max_val) * 1.4426950408889634f) : 0.0f;
                float p1 = (col_start + col + 1 < S && row_start + my_row < B * H * S) ? fast_exp2f_fn((f1 - max_val) * 1.4426950408889634f) : 0.0f;
                float p2 = (col_start + col + 2 < S && row_start + my_row < B * H * S) ? fast_exp2f_fn((f2 - max_val) * 1.4426950408889634f) : 0.0f;
                float p3 = (col_start + col + 3 < S && row_start + my_row < B * H * S) ? fast_exp2f_fn((f3 - max_val) * 1.4426950408889634f) : 0.0f;
                
                tmem_store_4x_fn(tmem_S, col, __float_as_uint(p0), __float_as_uint(p1), __float_as_uint(p2), __float_as_uint(p3));
                sum_val += p0 + p1 + p2 + p3;
            }
            tmem_store_fence_fn();
            
            // Sum reduction matching max reduction topology constraints
            sum_val += __shfl_xor_sync(0xFFFFFFFF, sum_val, 1);
            sum_val += __shfl_xor_sync(0xFFFFFFFF, sum_val, 2);
            sum_val += __shfl_xor_sync(0xFFFFFFFF, sum_val, 4);
            sum_val += __shfl_xor_sync(0xFFFFFFFF, sum_val, 32);
            sum_val += __shfl_xor_sync(0xFFFFFFFF, sum_val, 64);
            
            int idx = i; 
            float new_global_max = fmaxf(global_max[idx], max_val);
            float scale = fast_exp2f_fn((global_max[idx] - new_global_max) * 1.4426950408889634f);
            global_sum[idx] *= scale;
            
            // Rapidly scale previously accumulated outputs accounting for newly discovered max bounds
            for(int col = 0; col < 64; col += 4) {
                uint32_t r0, r1, r2, r3;
                tmem_load_4x_fn(tmem_D0, col, &r0, &r1, &r2, &r3);
                tmem_store_4x_fn(tmem_D0, col, __float_as_uint(__uint_as_float(r0) * scale), __float_as_uint(__uint_as_float(r1) * scale), __float_as_uint(__uint_as_float(r2) * scale), __float_as_uint(__uint_as_float(r3) * scale));
                tmem_load_4x_fn(tmem_D1, col, &r0, &r1, &r2, &r3);
                tmem_store_4x_fn(tmem_D1, col, __float_as_uint(__uint_as_float(r0) * scale), __float_as_uint(__uint_as_float(r1) * scale), __float_as_uint(__uint_as_float(r2) * scale), __float_as_uint(__uint_as_float(r3) * scale));
            }
            tmem_store_fence_fn();
            
            global_max[idx] = new_global_max;
            global_sum[idx] += sum_val;
        }
        
        // Compute secondary attention multiplications mapping completely over fully materialized P weights
        for(int i = 0; i < 2; i++) {
            uint32_t my_row = quad_idx + i * 4;
            if (col_start < S && row_start + my_row < B * H * S) {
                if (threadIdx.x == 0) {
                    uint32_t idesc_D = make_instr_desc_f16(64, 64, false, false);
                    for (int k = 0; k < 64; k += 16) {
                        uint64_t desc_p = make_smem_desc_sm100_fn(shared->s_P, 1, 1024) + (k * sizeof(__nv_bfloat16)) >> 4;
                        uint64_t desc_v0 = make_smem_desc_sm100_fn(shared->s_V0, 8192, 1024) + (k * 64);
                        uint64_t desc_v1 = make_smem_desc_sm100_fn(shared->s_V1, 8192, 1024) + (k * 64);
                        
                        umma_f16_cg2_fn(tmem_D0, desc_p, desc_v0, idesc_D, 1);
                        umma_f16_cg2_fn(tmem_D1, desc_p, desc_v1, idesc_D, 1);
                    }
                    umma_commit_2sm_fn(shared->mbar_COMMIT);
                }
                mbarrier_wait_fn(shared->mbar_COMMIT, phase % 2);
                __syncthreads();
            }
        }
        
        phase++;
        phase_Q ^= 1;
    }

    __syncthreads();
    float out0[4], out1[4];
    for(int i = 0; i < 2; i++) {
        uint32_t my_row = quad_idx + i * 4;
        out0[i] = global_sum[i];
        out1[i] = global_sum[i];
        
        int global_row = head_idx * S + row_start + my_row;
        if (global_row < B * H * S) {
            for(int col = 0; col < 64; col += 4) {
                uint32_t r0, r1, r2, r3;
                tmem_load_4x_fn(tmem_D0, col, &r0, &r1, &r2, &r3);
                tmem_load_4x_fn(tmem_D1, col, &r0, &r1, &r2, &r3); // Overwrites r0..r3
                
                O[global_row * D + col] = __float2bfloat16(__uint_as_float(r0) / out0[i]);
                O[global_row * D + col + 1] = __float2bfloat16(__uint_as_float(r1) / out1[i]);
                O[global_row * D + col + 2] = __float2bfloat16(__uint_as_float(r2) / out0[i]);
                O[global_row * D + col + 3] = __float2bfloat16(__uint_as_float(r3) / out1[i]);
                
                tmem_load_4x_fn(tmem_D1, col, &r0, &r1, &r2, &r3);
                O[global_row * D + col + 64] = __float2bfloat16(__uint_as_float(r0) / out0[i]);
                O[global_row * D + col + 65] = __float2bfloat16(__uint_as_float(r1) / out1[i]);
                O[global_row * D + col + 66] = __float2bfloat16(__uint_as_float(r2) / out0[i]);
                O[global_row * D + col + 67] = __float2bfloat16(__uint_as_float(r3) / out1[i]);
            }
            tmem_load_fence_fn();
        }
    }

    if (threadIdx.x == 0) {
        LSE[head_idx * S + row_start_cta0] = global_max[0] + logf(global_sum[0]);
        LSE[head_idx * S + row_start_cta1] = global_max[1] + logf(global_sum[1]);
    }

    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_S, 64);
        tmem_dealloc_fn(tmem_D0, 64);
        tmem_dealloc_fn(tmem_D1, 64);
    }
}

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_middle_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_middle_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[3] = {gmem_inner_dim, gmem_middle_dim, gmem_outer_dim};
    cuuint64_t globalStrides[2] = {gmem_inner_dim * 2, gmem_inner_dim * gmem_middle_dim * 2};
    cuuint32_t boxDim[3] = {smem_inner_dim, smem_middle_dim, smem_outer_dim};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        3,
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

namespace tvm_ffi_example_cuda {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id)); 

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3); 

    CUtensorMap desc_Q, desc_K, desc_V;
    CUresult res;
    res = create_tma_3d_descriptor_2B(&desc_Q, Q.data_ptr(), 128, S, B * H, 64, 64, 1, CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA Q error\n"); exit(1); }
    res = create_tma_3d_descriptor_2B(&desc_K, K.data_ptr(), 128, S, B * H, 64, 64, 1, CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA K error\n"); exit(1); }
    res = create_tma_3d_descriptor_2B(&desc_V, V.data_ptr(), 128, S, B * H, 64, 64, 1, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA V error\n"); exit(1); }

    dim3 grid((S + 1023) / 1024, B * H);
    dim3 block(256);
    
    int smem_size = sizeof(SharedStorage);
    CUDA_CHECK(cudaFuncSetAttribute(run_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(&config, run_kernel, 
        desc_Q, desc_K, desc_V, 
        static_cast<__nv_bfloat16*>(O.data_ptr()), static_cast<float*>(LSE.data_ptr()), 
        B, H, S, D));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_example_cuda