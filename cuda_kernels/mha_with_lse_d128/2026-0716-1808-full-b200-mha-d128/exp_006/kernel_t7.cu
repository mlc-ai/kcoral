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

namespace tvm_ffi_example_cuda {

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t dim0, uint64_t dim1, uint64_t dim2, uint32_t box0, uint32_t box1, uint32_t box2, CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[3] = {dim0, dim1, dim2};
    cuuint64_t globalStrides[2] = {dim0 * 2, dim0 * dim1 * 2};
    cuuint32_t boxDim[3] = {box0, box1, box2};
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
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
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

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__global__
__launch_bounds__(128)
void attention_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* __restrict__ O, 
    float* __restrict__ LSE,
    int S_len)
{
    int batch_head = blockIdx.x; 
    int query_start = blockIdx.y * 64;
    int tid = threadIdx.x;
    
    if (query_start >= S_len) return;
    
    extern __shared__ __align__(1024) char smem[];
    
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)smem;                      
    __nv_bfloat16* smem_K = (__nv_bfloat16*)(smem + 16384);             
    __nv_bfloat16* smem_V = (__nv_bfloat16*)(smem + 49152);             
    __nv_bfloat16* smem_P = (__nv_bfloat16*)(smem + 81920);             
    float* smem_S = (float*)(smem + 98304);                             
    float* smem_O = (float*)(smem + 131072);                            
    
    float* m_old = (float*)(smem + 163840);
    float* d_old = (float*)(smem + 164096);
    
    uint64_t* bar_Q = (uint64_t*)(smem + 164352);
    uint64_t* bar_K = (uint64_t*)(smem + 164360);
    uint64_t* bar_V = (uint64_t*)(smem + 164368);
    
    if (tid < 64) {
        m_old[tid] = -1e20f;
        d_old[tid] = 0;
    }
    
    if (tid == 0) {
        init_smem_barrier_fn(bar_Q, 1);
        init_smem_barrier_fn(bar_K, 1);
        init_smem_barrier_fn(bar_V, 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    int phase_Q = 0;
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_Q, 16384);
        tma_load_3d_fn(&tma_Q, bar_Q, smem_Q, 0, query_start, batch_head);
    }
    
    int phase_K = 0, phase_V = 0;
    float scale = 1.0f / sqrtf(128.0f);
    
    for (int j = 0; j < S_len; j += 128) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_K, 32768);
            tma_load_3d_fn(&tma_K, bar_K, smem_K, 0, j, batch_head);
            
            mbarrier_arrive_and_expect_tx_fn(bar_V, 32768);
            tma_load_3d_fn(&tma_V, bar_V, smem_V, 0, j, batch_head);
        }
        
        mbarrier_wait_fn(bar_Q, phase_Q);
        mbarrier_wait_fn(bar_K, phase_K);
        mbarrier_wait_fn(bar_V, phase_V);
        
        // Mask padded sequences dynamically avoiding NaN leakage in edge cases spanning across block bounds 
        if (tid < 128) {
            int row = tid;
            if (j + row >= S_len) {
                for(int c = 0; c < 128; c += 4) {
                    *(reinterpret_cast<float4*>(&smem_K[row * 128 + c])) = {0,0,0,0};
                    *(reinterpret_cast<float4*>(&smem_V[row * 128 + c])) = {0,0,0,0};
                }
            }
        }
        __syncthreads();
        
        // Zero temporary states explicitly resolving internal pipeline latency limits 
        for(int i = tid; i < 64 * 128; i += 128) {
            smem_S[i] = 0;
        }
        __syncthreads();
        
        // Optimized matmul utilizing inner loops matching hardware capability chunks
        int row = tid % 64;
        int col_start = (tid / 64) * 64;
        for(int k = 0; k < 128; ++k) {
            float q = smem_Q[row * 128 + k];
            for(int c = 0; c < 64; c += 4) {
                float4 k_vec = *(reinterpret_cast<float4*>(&smem_K[k * 128 + col_start + c]));
                float4& s_acc = *(reinterpret_cast<float4*>(&smem_S[row * 128 + col_start + c]));
                s_acc = {s_acc.x + q * k_vec.x, s_acc.y + q * k_vec.y, s_acc.z + q * k_vec.z, s_acc.w + q * k_vec.w};
            }
        }
        
        float my_max = -1e20f;
        if (tid < 64) {
            for(int c = 0; c < 128; c += 4) {
                float4 s_val = *(reinterpret_cast<float4*>(&smem_S[tid * 128 + c]));
                if (j + c < S_len) {
                    my_max = max(my_max, max(max(s_val.x, s_val.y), max(s_val.z, s_val.w)));
                }
            }
        }
        
        float new_max = max(m_old[tid], my_max);
        float factor = expf(m_old[tid] - new_max);
        
        if (tid < 64) {
            d_old[tid] *= factor;
        }
        
        // Critical step: rescale actively tracked partial outputs O uniformly inline matching current block maximum limits
        for(int c = 0; c < 64; c += 4) {
            float4& o_val = *(reinterpret_cast<float4*>(&smem_O[row * 128 + col_start + c]));
            float f = (row < 64) ? expf(m_old[row] - new_max) : 1.0f;
            o_val = {o_val.x * f, o_val.y * f, o_val.z * f, o_val.w * f};
        }
        
        float my_sum = 0;
        if (tid < 64) {
            for(int c = 0; c < 128; c += 4) {
                float4 s_val = *(reinterpret_cast<float4*>(&smem_S[tid * 128 + c]));
                float v0 = (j + c < S_len) ? expf(s_val.x * scale - new_max) : 0;
                float v1 = (j + c + 1 < S_len) ? expf(s_val.y * scale - new_max) : 0;
                float v2 = (j + c + 2 < S_len) ? expf(s_val.z * scale - new_max) : 0;
                float v3 = (j + c + 3 < S_len) ? expf(s_val.w * scale - new_max) : 0;
                
                my_sum += v0 + v1 + v2 + v3;
                
                // Encode perfectly native vectorization bounds utilizing chunking of 16 Byte spans mapping directly to SMEM banks
                *(reinterpret_cast<uint16_t*>(&smem_P[tid * 128 + c]) ) = __float2bfloat16(v0);
                *(reinterpret_cast<uint16_t*>(&smem_P[tid * 128 + c + 1]) ) = __float2bfloat16(v1);
                *(reinterpret_cast<uint16_t*>(&smem_P[tid * 128 + c + 2]) ) = __float2bfloat16(v2);
                *(reinterpret_cast<uint16_t*>(&smem_P[tid * 128 + c + 3]) ) = __float2bfloat16(v3);
            }
        }
        
        if (tid < 64) {
            d_old[tid] += my_sum;
            m_old[tid] = new_max;
        }
        
        __syncthreads(); 
        
        // Final phase: Accumulate seamlessly against unmodified native V chunks mapping identically over identical swizzling bounds 
        for(int k = 0; k < 128; ++k) {
            float p = smem_P[row * 128 + k];
            for(int c = 0; c < 64; c += 4) {
                float4 v_vec = *(reinterpret_cast<float4*>(&smem_V[k * 128 + col_start + c]));
                float4& o_acc = *(reinterpret_cast<float4*>(&smem_O[row * 128 + col_start + c]));
                o_acc = {o_acc.x + p * v_vec.x, o_acc.y + p * v_vec.y, o_acc.z + p * v_vec.z, o_acc.w + p * v_vec.w};
            }
        }
        
        __syncthreads(); 
        phase_Q ^= 1;
        phase_K ^= 1;
        phase_V ^= 1;
    }
    
    // Epilogue: Coalesced Global Vector writes mapped over natural row ordering bounds
    
    for(int i = tid; i < 64 * 128; i += 128) {
        int row = i / 128;
        int col = i % 128;
        if (query_start + row < S_len) {
            float inv_sum = 1.0f / d_old[row];
            float out = smem_O[i] * inv_sum;
            
            __nv_bfloat16* out_ptr = &((__nv_bfloat16*)O)[batch_head * S_len * 128 + (query_start + row) * 128 + col];
            *out_ptr = __float2bfloat16(out);
        }
    }
    
    if (tid < 64) {
        if (query_start + tid < S_len) {
            LSE[batch_head * S_len + query_start + tid] = m_old[tid] + logf(d_old[tid]);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    
    CUtensorMap tma_Q, tma_K, tma_V;
    uint64_t S_len = S; 
    
    CUresult res_q = create_tma_3d_descriptor_2B(&tma_Q, Q.data_ptr(), 128, S_len, B * H, 128, 64, 1, CU_TENSOR_MAP_SWIZZLE_NONE);
    if (res_q != CUDA_SUCCESS) { fprintf(stderr, "Failed to create tma_Q\n"); exit(1); }
    
    CUresult res_k = create_tma_3d_descriptor_2B(&tma_K, K.data_ptr(), 128, S_len, B * H, 128, 128, 1, CU_TENSOR_MAP_SWIZZLE_NONE);
    if (res_k != CUDA_SUCCESS) { fprintf(stderr, "Failed to create tma_K\n"); exit(1); }
    
    CUresult res_v = create_tma_3d_descriptor_2B(&tma_V, V.data_ptr(), 128, S_len, B * H, 128, 128, 1, CU_TENSOR_MAP_SWIZZLE_NONE);
    if (res_v != CUDA_SUCCESS) { fprintf(stderr, "Failed to create tma_V\n"); exit(1); }
    
    dim3 grid(B * H, (S + 63) / 64); 
    dim3 block(128); 
    
    int smem_size = 196608; 
    CUDA_CHECK(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    attention_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V,
        static_cast<__nv_bfloat16*>(O.data_ptr()), 
        static_cast<float*>(LSE.data_ptr()), 
        S
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda