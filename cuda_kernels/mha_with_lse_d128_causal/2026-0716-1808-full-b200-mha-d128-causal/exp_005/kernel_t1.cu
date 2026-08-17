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

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        fprintf(stderr, "CU error %d at %s:%d\n",                 \
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_mha {

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle) {
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
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
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
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
       :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
       :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (addr & 0x3FFFF) >> 4;
    d |= ((uint64_t)1 << 16);
    d |= ((uint64_t)128 << 32);
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)0 << 61;
    return d;
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void named_barrier_sync_fn(int bar_id, int count) {
    asm volatile("barrier.sync.aligned %0, %1;" :: "r"(bar_id), "r"(count));
}

__global__ __launch_bounds__(128) void mha_with_lse_d128_causal(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O, float* LSE, int S, int num_heads)
{
    int b_idx = blockIdx.y / num_heads;
    int h_idx = blockIdx.y % num_heads;
    int i = blockIdx.x * 128;
    int tid = threadIdx.x;

    extern __shared__ __align__(1024) char smem_buf[];
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)smem_buf;
    __nv_bfloat16* smem_K = (__nv_bfloat16*)(smem_buf + 32768);
    __nv_bfloat16* smem_V = (__nv_bfloat16*)(smem_buf + 65536);
    __nv_bfloat16* smem_P = smem_Q;

    extern __shared__ __align__(4) uint32_t tmem_S_buf[];
    uint32_t* tmem_S = (uint32_t*)tmem_S_buf;
    uint32_t* tmem_P = (uint32_t*)(tmem_S_buf + 16384);

    extern __shared__ __align__(8) uint64_t bar_Q[];
    extern __shared__ __align__(8) uint64_t bar_K[];
    extern __shared__ __align__(8) uint64_t bar_V[];

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&bar_Q[0], 1);
        init_smem_barrier_fn(&bar_K[0], 1);
        init_smem_barrier_fn(&bar_V[0], 1);
        
        tmem_alloc_fn(&tmem_S, 128);
        tmem_alloc_fn(&tmem_P, 128);
    }
    __syncthreads();

    float m_val[128];
    float l_val[128];
    for (int r = 0; r < 128; ++r) {
        m_val[r] = -1e20f;
        l_val[r] = 0.0f;
    }

    int s_base = b_idx * num_heads * S + h_idx * S;
    uint32_t zero = 0;
    uint32_t one = 0xFFFFFFFF;

    for (; i < S; i += 128) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&bar_Q, 32768);
            tma_load_2d_fn(&tma_Q, &bar_Q, smem_Q, 0, s_base + i * 128);
        }
        mbarrier_wait_fn(&bar_Q, 0);

        float o_acc[128];
        for (int r = 0; r < 128; ++r) {
            for (int c = 0; c < 128; ++c) {
                o_acc[r][c] = 0.0f;
            }
        }

        for (int j = 0; j <= i; j += 128) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&bar_K, 32768);
                tma_load_2d_fn(&tma_K, &bar_K, smem_K, 0, s_base + j * 128);
                
                mbarrier_arrive_and_expect_tx_fn(&bar_V, 32768);
                tma_load_2d_fn(&tma_V, &bar_V, smem_V, 0, s_base + j * 128);
            }
            mbarrier_wait_fn(&bar_K, 0);
            mbarrier_wait_fn(&bar_V, 0);

            if (threadIdx.x == 0) {
                asm volatile("tcgen05.cp.cta_group::2.128x128b [%0], [%1];" 
                             :: "r"(tmem_S), "r"(tmem_P) : "memory");
            }

            float S_val[4];
            for(int k_idx = 0; k_idx < 4; ++k_idx) S_val[k_idx] = 0.0f;

            for (int k = 0; k < 128; k += 16) {
                for (int n = 0; n < 128; n += 16) {
                    uint64_t desc_A = make_smem_desc_sm100_fn(smem_Q + k);
                    uint64_t desc_B = make_smem_desc_sm100_fn(smem_K + n);
                    
                    float scale = 1.0f / sqrtf(128.0f);
                    uint32_t scale_bits = __float_as_uint(scale);

                    asm volatile("wgmma.mma_sync_acc .16bits .32b .matrix_a_nobroadcast "
                                 "[%0], %1, %2, desc_A, desc_B;\n"
                                 ":: "r"(tmem_S), "r"(scale), "r"(zero) : : "memory");
                }
            }

            for(int idx = 0; idx < 4; ++idx) {
                int row = (idx / 2) * 64 + (tid / 2) % 64;
                int col = (idx % 2) * 64 + (tid % 2) * 32 + (tid / 32) % 2 * 16;
                
                float val = S_val[idx];
                int global_row = i * 128 + row;
                int global_col = j * 128 + col;
                int global_col_max = min(global_row, S - 1);
                
                if (global_col > global_col_max || global_col >= S) {
                    val = -1e20f;
                }
                S_val[idx] = val;
            }

            float temp_m[2] = {-1e20f, -1e20f};
            float temp_l[2] = {0.0f, 0.0f};

            for(int idx = 0; idx < 4; ++idx) {
                int row_part = (idx % 2);
                temp_m[row_part] = max(temp_m[row_part], S_val[idx]);
            }

            int row_0 = (0 / 2) * 64 + (tid / 2) % 64;
            int row_1 = (1 / 2) * 64 + (tid / 2) % 64;

            __shared__ float smem_m[128][2];
            for(int idx = 0; idx < 4; ++idx) {
                int row_part = (idx % 2);
                smem_m[(idx / 2) * 64 + (tid / 2) % 64][row_part] = S_val[idx];
            }
            __syncthreads();
            
            for(int r_local = 0; r_local < 64; ++r_local) {
                temp_m[0] = max(temp_m[0], smem_m[r_local][0]);
                temp_m[1] = max(temp_m[1], smem_m[r_local][1]);
            }
            __syncthreads();

            float m_new_0 = max(m_val[row_0], temp_m[0]);
            float m_new_1 = max(m_val[row_1], temp_m[1]);

            if (m_new_0 > m_val[row_0]) {
                float factor = fast_exp2f_fn((m_val[row_0] - m_new_0) * 1.44269504f);
                for (int c = 0; c < 128; ++c) {
                    o_acc[row_0][c] *= factor;
                }
                l_val[row_0] *= factor;
            }
            if (m_new_1 > m_val[row_1]) {
                float factor = fast_exp2f_fn((m_val[row_1] - m_new_1) * 1.44269504f);
                for (int c = 0; c < 128; ++c) {
                    o_acc[row_1][c] *= factor;
                }
                l_val[row_1] *= factor;
            }
            m_val[row_0] = m_new_0;
            m_val[row_1] = m_new_1;

            for(int idx = 0; idx < 4; ++idx) {
                int row_part = (idx % 2);
                float m_new = (row_part == 0) ? m_new_0 : m_new_1;
                float exp_f = fast_exp2f_fn((S_val[idx] - m_new) * 1.44269504f);
                
                temp_l[row_part] += exp_f;
                S_val[idx] = exp_f;
            }

            __shared__ float smem_l[128][2];
            for(int idx = 0; idx < 4; ++idx) {
                int row_part = (idx % 2);
                smem_l[(idx / 2) * 64 + (tid / 2) % 64][row_part] = S_val[idx];
            }
            __syncthreads();
            
            for(int r_local = 0; r_local < 64; ++r_local) {
                temp_l[0] += smem_l[r_local][0];
                temp_l[1] += smem_l[r_local][1];
            }
            __syncthreads();

            l_val[row_0] += temp_l[0];
            l_val[row_1] += temp_l[1];

            __syncthreads();
            // Ensure all threads have finished reading P from SMEM before modifying it
            for (int j_p = 0; j_p < 128; ++j_p) {
                float p = (j_p <= (min(i * 128 + tid, S - 1) - j * 128)) ? S_val[(j_p - j * 128) / 2] : 0.0f;
                smem_P[tid * 128 + j_p] = __float2bfloat16(p);
            }
            __syncthreads();

            for (int k_pv = 0; k_pv < 128; k_pv += 16) {
                for (int n_pv = 0; n_pv < 128; n_pv += 16) {
                    uint64_t desc_A2 = make_smem_desc_sm100_fn(smem_P + k_pv);
                    uint64_t desc_B2 = make_smem_desc_sm100_fn(smem_V + n_pv);
                    
                    asm volatile("wgmma.mma_sync_acc .16bits .32b .matrix_a_nobroadcast "
                                 "[%0], %1, %2, desc_A2, desc_B2;\n"
                                 ":: "r"(tmem_P), "r"(one), "r"(zero) : : "memory");
                }
            }
            
            named_barrier_sync_fn(1, 128);
        }

        for (int r = 0; r < 128; ++r) {
            if (l_val[r] > 0.0f) {
                for (int c = 0; c < 128; ++c) {
                    out_ptr[row * 128 + c] = __float2bfloat16(o_acc[c] / l_val[r]);
                }
            }
        }

        for (int r = 0; r < 128; ++r) {
            if (global_row < M && l_val[r] > 0.0f) {
                LSE[lse_idx] = m_val[r] + logf(l_val[r]);
            }
        }
        
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        tmem_dealloc_fn(*tmem_S, 128);
        tmem_dealloc_fn(*tmem_P, 128);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    if (S == 0) return;

    int num_heads = B * H;
    int num_blocks = (S + 127) / 128;
    int smem_size = 200000;

    CUDA_CHECK(cudaFuncSetAttribute(
        mha_with_lse_d128_causal,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size));

    CUtensorMap tma_Q, tma_K, tma_V;
    uint64_t total_S = B * H * S;
    
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), 128, total_S, 128, 128, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), 128, total_S, 128, 128, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), 128, total_S, 128, 128, CU_TENSOR_MAP_SWIZZLE_128B));

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    dim3 grid(num_blocks, num_heads);
    
    mha_with_lse_d128_causal<<<dim3(num_blocks, num_heads), 128, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, 
        static_cast<__nv_bfloat16*>(O.data_ptr()), 
        static_cast<float*>(LSE.data_ptr()), 
        S, num_heads);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha::run);

} // namespace tvm_ffi_mha