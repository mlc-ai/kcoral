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

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// Helper PTX wrappers for SM100
__device__ __forceinline__ void tmem_alloc_1sm_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_1sm_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_parity_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase));
}

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_before_fn() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)), "l"((uint64_t)d), "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo, int swizzle) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    uint32_t base_offset = (swizzle == 0) ? 0 : ((addr >> 7) & 0x7);
    d |= (uint64_t)base_offset << 49;
    d |= (uint64_t)swizzle << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_cg1_fn(uint32_t M, uint32_t N, int a_major, int b_major) {
    uint32_t d = 0;
    d |= (1u << 4) | (1u << 7) | (1u << 10);
    d |= (a_major << 15) | (b_major << 16);
    d |= ((N / 8) << 17) | ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ void umma_f16_cg1_fn(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_1sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(a));
}

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tmem_load_8x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3, uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),"=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(col));
}

__device__ __forceinline__ __nv_bfloat16* get_swizzled_ptr_128B(__nv_bfloat16* base, int row, int col) {
    int chunk_x = col / 8;
    int rem = col % 8;
    int swizzled_chunk_x = (row % 8) ^ chunk_x;
    return &base[row * 64 + swizzled_chunk_x * 8 + rem]; 
}

extern __shared__ __align__(1024) char smem_raw[];

__global__ void attn_fwd_sm100_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O,
    float* LSE,
    int B, int H, int S) 
{
    int b = blockIdx.z;
    int h = blockIdx.y;
    int m_start = blockIdx.x * 64; 
    
    if (m_start >= S) return;

    __nv_bfloat16* smem_Q_L = (__nv_bfloat16*)(smem_raw + 0 * 8192);
    __nv_bfloat16* smem_Q_R = (__nv_bfloat16*)(smem_raw + 1 * 8192);
    __nv_bfloat16* smem_K_L = (__nv_bfloat16*)(smem_raw + 2 * 8192);
    __nv_bfloat16* smem_K_R = (__nv_bfloat16*)(smem_raw + 3 * 8192);
    __nv_bfloat16* smem_V_L = (__nv_bfloat16*)(smem_raw + 4 * 8192);
    __nv_bfloat16* smem_V_R = (__nv_bfloat16*)(smem_raw + 5 * 8192);
    __nv_bfloat16* smem_P   = (__nv_bfloat16*)(smem_raw + 6 * 8192);

    uint64_t* tma_bar = (uint64_t*)(smem_raw + 7 * 8192);
    uint64_t* umma_bar = (uint64_t*)(smem_raw + 7 * 8192 + 8);
    uint32_t* smem_tmem_base = (uint32_t*)(smem_raw + 7 * 8192 + 16);

    int tid = threadIdx.x;

    if (tid == 0) {
        init_smem_barrier_fn(tma_bar, 1);
        init_smem_barrier_fn(umma_bar, 1);
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    if (tid < 32) {
        tmem_alloc_1sm_fn(smem_tmem_base, 256);
    }
    __syncthreads();

    uint32_t tmem_base = *smem_tmem_base;
    uint32_t tmem_S = tmem_base;
    uint32_t tmem_O0 = tmem_base + 64;
    uint32_t tmem_O1 = tmem_base + 128;

    int32_t q_outer = b * H * S + h * S + m_start;
    int phase_tma = 0;
    
    if (tid == 0) {
        mbarrier_arrive_expect_tx_fn(tma_bar, 16384);
        tma_load_2d_fn(&tma_Q, tma_bar, smem_Q_L, 0, q_outer);
        tma_load_2d_fn(&tma_Q, tma_bar, smem_Q_R, 64, q_outer);
    }
    __syncthreads();
    mbarrier_wait_parity_fn(tma_bar, phase_tma);
    phase_tma ^= 1;

    float reg_O0[64] = {0.0f};
    float reg_O1[64] = {0.0f};
    float row_S[64];
    float m_i_val = -INFINITY;
    float l_i_val = 0.0f;
    int phase_umma = 0;
    
    uint32_t idesc_QK = make_instr_desc_cg1_fn(64, 64, 0, 0); 
    uint32_t idesc_PV = make_instr_desc_cg1_fn(64, 64, 0, 1); 

    for (int n = 0; n < S; n += 64) {
        int32_t kv_outer = b * H * S + h * S + n;
        if (tid == 0) {
            mbarrier_arrive_expect_tx_fn(tma_bar, 32768);
            tma_load_2d_fn(&tma_K, tma_bar, smem_K_L, 0, kv_outer);
            tma_load_2d_fn(&tma_K, tma_bar, smem_K_R, 64, kv_outer);
            tma_load_2d_fn(&tma_V, tma_bar, smem_V_L, 0, kv_outer);
            tma_load_2d_fn(&tma_V, tma_bar, smem_V_R, 64, kv_outer);
        }
        __syncthreads();
        mbarrier_wait_parity_fn(tma_bar, phase_tma);
        phase_tma ^= 1;

        if (tid == 0) {
            tcgen05_fence_after_fn();
            
            uint64_t desc_Q_L = make_smem_desc_sm100_fn(smem_Q_L, 1, 1024, 2);
            uint64_t desc_K_L = make_smem_desc_sm100_fn(smem_K_L, 1, 1024, 2);
            umma_f16_cg1_fn(tmem_S, desc_Q_L, desc_K_L, idesc_QK, 0);

            uint64_t desc_Q_R = make_smem_desc_sm100_fn(smem_Q_R, 1, 1024, 2);
            uint64_t desc_K_R = make_smem_desc_sm100_fn(smem_K_R, 1, 1024, 2);
            umma_f16_cg1_fn(tmem_S, desc_Q_R, desc_K_R, idesc_QK, 1);
            
            umma_commit_1sm_fn(umma_bar);
        }
        tcgen05_fence_before_fn();
        __syncthreads();
        mbarrier_wait_parity_fn(umma_bar, phase_umma);
        phase_umma ^= 1;
        tcgen05_fence_after_fn();

        if (tid < 64) {
            uint32_t r_S[64];
            #pragma unroll
            for (int c = 0; c < 64; c += 8) {
                tmem_load_8x_fn(tmem_S + c, &r_S[c], &r_S[c+1], &r_S[c+2], &r_S[c+3], 
                                            &r_S[c+4], &r_S[c+5], &r_S[c+6], &r_S[c+7]);
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); 

            float row_max = -INFINITY;
            #pragma unroll
            for (int c = 0; c < 64; ++c) {
                float f = __uint_as_float(r_S[c]) * 0.088388347648f; 
                if (n + c >= S) f = -INFINITY;
                row_S[c] = f;
                row_max = fmaxf(row_max, f);
            }
            
            float m_new = fmaxf(m_i_val, row_max);
            float exp_diff = (m_i_val == -INFINITY) ? 0.0f : expf(m_i_val - m_new);
            l_i_val *= exp_diff;
            
            #pragma unroll
            for(int i = 0; i < 64; ++i) {
                reg_O0[i] *= exp_diff;
                reg_O1[i] *= exp_diff;
            }
            
            float row_sum = 0;
            #pragma unroll
            for (int c = 0; c < 64; ++c) {
                float p = (row_S[c] == -INFINITY) ? 0.0f : expf(row_S[c] - m_new);
                row_sum += p;
                __nv_bfloat16* p_ptr = get_swizzled_ptr_128B(smem_P, tid, c);
                *p_ptr = __float2bfloat16(p);
            }
            l_i_val += row_sum;
            m_i_val = m_new;
        }
        tcgen05_fence_before_fn();
        __syncthreads();
        fence_async_shared_fn();

        if (tid == 0) {
            tcgen05_fence_after_fn();
            
            uint64_t desc_P = make_smem_desc_sm100_fn(smem_P, 1, 1024, 2);
            uint64_t desc_V_L = make_smem_desc_sm100_fn(smem_V_L, 8192, 1024, 2);
            umma_f16_cg1_fn(tmem_O0, desc_P, desc_V_L, idesc_PV, 0);

            uint64_t desc_V_R = make_smem_desc_sm100_fn(smem_V_R, 8192, 1024, 2);
            umma_f16_cg1_fn(tmem_O1, desc_P, desc_V_R, idesc_PV, 0);
            
            umma_commit_1sm_fn(umma_bar);
        }
        tcgen05_fence_before_fn();
        __syncthreads();
        mbarrier_wait_parity_fn(umma_bar, phase_umma);
        phase_umma ^= 1;
        tcgen05_fence_after_fn();

        if (tid < 64) {
            uint32_t r_O0[64];
            uint32_t r_O1[64];
            #pragma unroll
            for (int c = 0; c < 64; c += 8) {
                tmem_load_8x_fn(tmem_O0 + c, &r_O0[c], &r_O0[c+1], &r_O0[c+2], &r_O0[c+3], 
                                             &r_O0[c+4], &r_O0[c+5], &r_O0[c+6], &r_O0[c+7]);
            }
            #pragma unroll
            for (int c = 0; c < 64; c += 8) {
                tmem_load_8x_fn(tmem_O1 + c, &r_O1[c], &r_O1[c+1], &r_O1[c+2], &r_O1[c+3], 
                                             &r_O1[c+4], &r_O1[c+5], &r_O1[c+6], &r_O1[c+7]);
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

            #pragma unroll
            for (int c = 0; c < 64; ++c) {
                reg_O0[c] += __uint_as_float(r_O0[c]);
                reg_O1[c] += __uint_as_float(r_O1[c]);
            }
        }
        tcgen05_fence_before_fn();
        __syncthreads(); 
    }

    if (tid < 64) {
        int global_m = m_start + tid;
        if (global_m < S) {
            float inv_l = 1.0f / l_i_val;
            int64_t out_base_idx = (int64_t)b * H * S * 128 + (int64_t)h * S * 128 + (int64_t)global_m * 128;
            #pragma unroll
            for (int c = 0; c < 64; ++c) {
                O[out_base_idx + c] = __float2bfloat16(reg_O0[c] * inv_l);
            }
            #pragma unroll
            for (int c = 0; c < 64; ++c) {
                O[out_base_idx + 64 + c] = __float2bfloat16(reg_O1[c] * inv_l);
            }
            int64_t lse_idx = (int64_t)b * H * S + (int64_t)h * S + global_m;
            LSE[lse_idx] = m_i_val + logf(l_i_val);
        }
    }
    
    tcgen05_fence_before_fn();
    __syncthreads();

    if (tid < 32) {
        tmem_dealloc_1sm_fn(*smem_tmem_base, 256);
    }
}

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

namespace tvm_ffi_example_cuda {
void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    CUtensorMap tma_Q, tma_K, tma_V;
    create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), 128, B*H*S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), 128, B*H*S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), 128, B*H*S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    int smem_size = 57600; 
    dim3 grid((S + 63) / 64, H, B);
    dim3 block(128);
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    cudaFuncSetAttribute(attn_fwd_sm100_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    attn_fwd_sm100_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, static_cast<__nv_bfloat16*>(O.data_ptr()), static_cast<float*>(LSE.data_ptr()), B, H, S
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
}