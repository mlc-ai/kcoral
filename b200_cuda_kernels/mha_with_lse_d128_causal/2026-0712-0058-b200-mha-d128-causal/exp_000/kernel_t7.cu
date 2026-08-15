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

namespace tvm_ffi_mha {

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

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float4* read_swizzled_ptr(__nv_bfloat16* smem, int row, int col) {
    int x_chunk = col / 8;
    int x_swizzled = x_chunk ^ (row % 8);
    int swizzled_c = x_swizzled * 8 + (col % 8);
    return (float4*)&smem[row * 128 + swizzled_c];
}


__global__ __launch_bounds__(128, 1)
void mha_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O, float* LSE, uint32_t S, uint32_t H) 
{
    extern __shared__ __align__(1024) uint8_t smem_pool[];
    
    __nv_bfloat16* q_smem = (__nv_bfloat16*)smem_pool;                
    __nv_bfloat16* k_smem = q_smem + 16384;                           
    __nv_bfloat16* v_smem = k_smem + 16384;                            
    __nv_bfloat16* p_smem = v_smem + 16384;                            
    uint64_t* bar_q = (uint64_t*)(p_smem + 16384);
    uint64_t* bar_k = bar_q + 1;
    uint64_t* bar_v = bar_k + 1;

    uint32_t total_q_blks = (S + 127) / 128;
    uint32_t q_blk_idx = blockIdx.x % total_q_blks;
    uint32_t total_head_idx = blockIdx.x / total_q_blks;
    uint32_t h_idx = total_head_idx % H;
    uint32_t b_idx = total_head_idx / H;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar_q, 1);
        init_smem_barrier_fn(bar_k, 1);
        init_smem_barrier_fn(bar_v, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    int phase[3] = {0, 0, 0};
    float scale_factor = 1.0f / sqrtf(128.0f);

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_q, 32768);
        tma_load_3d_fn(&tma_Q, bar_q, q_smem, 0, q_blk_idx * 128, b_idx * H + h_idx);
        tma_load_3d_fn(&tma_Q, bar_q, q_smem + 8192, 64, q_blk_idx * 128, b_idx * H + h_idx);
    }
    mbarrier_wait_fn(bar_q, phase[0]);
    phase[0] ^= 1;
    __syncthreads();

    float o_frag[2][64]; 
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 64; ++j) {
            o_frag[i][j] = 0.0f;
        }
    }

    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    
    #pragma unroll 1
    for (uint32_t k_blk_idx = 0; k_blk_idx <= q_blk_idx; ++k_blk_idx) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_k, 32768);
            tma_load_3d_fn(&tma_K, bar_k, k_smem, 0, k_blk_idx * 128, b_idx * H + h_idx);
            tma_load_3d_fn(&tma_K, bar_k, k_smem + 8192, 64, k_blk_idx * 128, b_idx * H + h_idx);
        }
        mbarrier_wait_fn(bar_k, phase[1]);
        phase[1] ^= 1;
        __syncthreads();

        float s_vals[128];
        for (int r = 0; r < 128; ++r) {
            float s = 0;
            for (int c = 0; c < 64; c += 4) {
                float4 q = *(float4*)read_swizzled_ptr(q_smem, threadIdx.x, c);
                float4 k = *(float4*)read_swizzled_ptr(k_smem, r, c);
                s += q.x * k.x + q.y * k.y + q.z * k.z + q.w * k.w;
            }
            for (int c = 0; c < 64; c += 4) {
                float4 q = *(float4*)read_swizzled_ptr(q_smem + 8192, threadIdx.x, c);
                float4 k = *(float4*)read_swizzled_ptr(k_smem + 8192, r, c);
                s += q.x * k.x + q.y * k.y + q.z * k.z + q.w * k.w;
            }
            
            uint32_t global_q_idx = q_blk_idx * 128 + threadIdx.x;
            uint32_t global_k_idx = k_blk_idx * 128 + r;
            if (global_q_idx >= S || global_k_idx >= S || global_k_idx > global_q_idx) {
                s = -INFINITY;
            } else {
                s *= scale_factor;
            }
            s_vals[r] = s;
        }

        float m_curr = -INFINITY;
        for (int r = 0; r < 128; ++r) {
            m_curr = fmaxf(m_curr, s_vals[r]);
        }
        
        float m_new = fmaxf(m_prev, m_curr);
        float l_curr = 0;
        
        for (int r = 0; r < 128; ++r) {
            float p = fast_exp2f_fn((s_vals[r] - m_new) * 1.4426950f);
            l_curr += p;
            p_smem[threadIdx.x * 128 + r] = __float2bfloat16(p);
        }
        
        float l_new = l_prev * fast_exp2f_fn((m_prev - m_new) * 1.4426950f) + l_curr;
        
        if (l_prev > 0.0f) {
            float scale = fast_exp2f_fn((m_prev - m_new) * 1.4426950f);
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 64; ++j) {
                    o_frag[i][j] *= scale;
                }
            }
        }

        m_prev = m_new;
        l_prev = l_new;
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_v, 32768);
            tma_load_3d_fn(&tma_V, bar_v, v_smem, 0, k_blk_idx * 128, b_idx * H + h_idx);
            tma_load_3d_fn(&tma_V, bar_v, v_smem + 8192, 64, k_blk_idx * 128, b_idx * H + h_idx);
        }
        mbarrier_wait_fn(bar_v, phase[2]);
        phase[2] ^= 1;
        __syncthreads();

        for (int c = 0; c < 64; c += 4) {
            float4 sum = {0, 0, 0, 0};
            for (int r = 0; r < 128; ++r) {
                float p = p_smem[threadIdx.x * 128 + r];
                float4 v = *(float4*)read_swizzled_ptr(v_smem, r, c);
                sum.x += p * v.x;
                sum.y += p * v.y;
                sum.z += p * v.z;
                sum.w += p * v.w;
            }
            *(float4*)read_swizzled_ptr((__nv_bfloat16*)o_frag, 0, c) = sum;
        }
        
        for (int c = 0; c < 64; c += 4) {
            float4 sum = {0, 0, 0, 0};
            for (int r = 0; r < 128; ++r) {
                float p = p_smem[threadIdx.x * 128 + r];
                float4 v = *(float4*)read_swizzled_ptr(v_smem + 8192, r, c);
                sum.x += p * v.x;
                sum.y += p * v.y;
                sum.z += p * v.z;
                sum.w += p * v.w;
            }
            *(float4*)read_swizzled_ptr((__nv_bfloat16*)o_frag, 1, c) = sum;
        }

        __syncthreads();
    }

    for (int c = 0; c < 64; c += 4) {
        float4 vals = *(float4*)read_swizzled_ptr((__nv_bfloat16*)o_frag, 0, c);
        float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
        vals.x *= inv_l; vals.y *= inv_l; vals.z *= inv_l; vals.w *= inv_l;
        
        uint32_t global_row = q_blk_idx * 128 + threadIdx.x;
        if (global_row < S) {
            uint32_t base_idx = total_head_idx * S * 128 + global_row * 128 + c;
            O[base_idx] = __float2bfloat16(vals.x);
            O[base_idx + 1] = __float2bfloat16(vals.y);
            O[base_idx + 2] = __float2bfloat16(vals.z);
            O[base_idx + 3] = __float2bfloat16(vals.w);
        }
    }
    
    for (int c = 0; c < 64; c += 4) {
        float4 vals = *(float4*)read_swizzled_ptr((__nv_bfloat16*)o_frag, 1, c);
        float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
        vals.x *= inv_l; vals.y *= inv_l; vals.z *= inv_l; vals.w *= inv_l;
        
        uint32_t global_row = q_blk_idx * 128 + threadIdx.x;
        if (global_row < S) {
            uint32_t base_idx = total_head_idx * S * 128 + global_row * 128 + 64 + c;
            O[base_idx] = __float2bfloat16(vals.x);
            O[base_idx + 1] = __float2bfloat16(vals.y);
            O[base_idx + 2] = __float2bfloat16(vals.z);
            O[base_idx + 3] = __float2bfloat16(vals.w);
        }
    }

    uint32_t global_row = q_blk_idx * 128 + threadIdx.x;
    if (threadIdx.x < 128 && global_row < S) {
        uint32_t lse_idx = total_head_idx * S + global_row;
        LSE[lse_idx] = logf(l_prev);
    }
}

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_dim0, uint64_t gmem_dim1, uint64_t gmem_dim2, uint32_t smem_dim0, uint32_t smem_dim1, uint32_t smem_dim2, CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[3] = {gmem_dim0, gmem_dim1, gmem_dim2};
    cuuint64_t globalStrides[2] = {gmem_dim0 * 2, gmem_dim0 * gmem_dim1 * 2};
    cuuint32_t boxDim[3] = {smem_dim0, smem_dim1, smem_dim2};
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
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);

    CUtensorMap tma_Q, tma_K, tma_V;
    create_tma_3d_descriptor_2B(&tma_Q, Q.data_ptr(), D, S, B * H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B);
    create_tma_3d_descriptor_2B(&tma_K, K.data_ptr(), D, S, B * H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B);
    create_tma_3d_descriptor_2B(&tma_V, V.data_ptr(), D, S, B * H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B);

    int64_t total_q_blks = (S + 127) / 128;
    dim3 grid(total_q_blks * B * H, 1, 1);
    dim3 block(128, 1, 1);

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUDA_CHECK(cudaFuncSetAttribute(mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 132096));
    mha_kernel<<<grid, block, 132096, stream>>>(
        tma_Q, tma_K, tma_V, 
        static_cast<__nv_bfloat16*>(O.data_ptr()), 
        static_cast<float*>(LSE.data_ptr()), 
        S, H);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha::run);

}  // namespace tvm_ffi_mha