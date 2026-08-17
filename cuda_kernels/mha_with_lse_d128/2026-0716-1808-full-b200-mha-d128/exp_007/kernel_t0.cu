#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <stdio.h>
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

__global__ __launch_bounds__(128, 1) void mha_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* LSE,
    int S, int B_H)
{
    int bh = blockIdx.z;
    int q_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (q_idx >= S) return;
    
    extern __shared__ char smem[];
    __nv_bfloat16* smem_K = (__nv_bfloat16*)smem;                     // 33216 bytes
    __nv_bfloat16* smem_V = (__nv_bfloat16*)(smem + 33216);           // 33216 bytes
    float* smem_p = (float*)(smem + 66432);                           // 512 bytes
    float* smem_warp_max = (float*)(smem + 66944);                    // 16 bytes
    float* smem_warp_sum = (float*)(smem + 66960);                    // 16 bytes
    float* smem_prev_max = (float*)(smem + 66976);                    // 4 bytes
    float* smem_new_max = (float*)(smem + 66980);                     // 4 bytes
    
    const __nv_bfloat16* g_Q = Q + bh * S * 128;
    const __nv_bfloat16* g_K = K + bh * S * 128;
    const __nv_bfloat16* g_V = V + bh * S * 128;
    __nv_bfloat16* g_O = O + bh * S * 128;
    float* g_LSE = LSE + bh * S;
    
    float2 my_q[64];
    for (int i = 0; i < 64; ++i) {
        my_q[i] = ((const float2*)(g_Q + q_idx * 128))[i];
    }
    
    float acc_o = 0.0f;
    float global_max = -INFINITY;
    float global_sum = 0.0f;
    float scale = 1.0f / __sqrtf(128.0f);
    
    int num_steps = (S + 127) / 128;
    
    for (int step = 0; step < num_steps; ++step) {
        for (int i = 0; i < 128; ++i) {
            int idx = step * 128 + tid;
            __nv_bfloat16 vk = __float2bfloat16(0.0f);
            __nv_bfloat16 vv = __float2bfloat16(0.0f);
            if (idx < S) {
                vk = ((const __nv_bfloat16*)(g_K + idx * 128))[i];
                vv = ((const __nv_bfloat16*)(g_V + idx * 128))[i];
            }
            smem_K[tid * 129 + i] = vk;
            smem_V[tid * 129 + i] = vv;
        }
        __syncthreads();
        
        int key_idx = step * 128 + tid;
        float s = 0.0f;
        if (key_idx < S) {
            for (int d = 0; d < 64; ++d) {
                float2 k_d = ((const float2*)(smem_K + tid * 129))[d];
                s += __low2float(my_q[d]) * __low2float(k_d);
                s += __high2float(my_q[d]) * __high2float(k_d);
            }
        }
        s *= scale;
        if (key_idx >= S) s = -INFINITY;
        
        float m = s;
        m = max(m, __shfl_xor_sync(0xFFFFFFFF, m, 1));
        m = max(m, __shfl_xor_sync(0xFFFFFFFF, m, 2));
        m = max(m, __shfl_xor_sync(0xFFFFFFFF, m, 4));
        m = max(m, __shfl_xor_sync(0xFFFFFFFF, m, 8));
        
        if (tid % 32 == 0) smem_warp_max[tid / 32] = m;
        __syncthreads();
        if (tid < 2) smem_warp_max[tid] = max(smem_warp_max[tid], smem_warp_max[tid+1]);
        __syncthreads();
        if (tid < 1) smem_warp_max[tid] = max(smem_warp_max[tid], smem_warp_max[tid+1]);
        __syncthreads();
        float row_max = smem_warp_max[0];
        
        float prev_max = global_max;
        float new_max = max(prev_max, row_max);
        
        if (tid == 0) smem_prev_max[0] = prev_max;
        if (tid == 0) smem_new_max[0] = new_max;
        __syncthreads();
        prev_max = smem_prev_max[0];
        new_max = smem_new_max[0];
        
        if (new_max > prev_max) {
            float factor = __expf(prev_max - new_max);
            acc_o *= factor;
            global_sum *= factor;
        }
        global_max = new_max;
        
        float p = (row_max == -INFINITY) ? 0.0f : __expf(s - row_max);
        
        float sum = p;
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, 1);
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, 2);
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, 4);
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, 8);
        
        if (tid % 32 == 0) smem_warp_sum[tid / 32] = sum;
        __syncthreads();
        if (tid < 2) smem_warp_sum[tid] += smem_warp_sum[tid+1];
        __syncthreads();
        if (tid < 1) smem_warp_sum[tid] += smem_warp_sum[tid+1];
        __syncthreads();
        float row_sum = smem_warp_sum[0];
        
        global_sum += row_sum * __expf(row_max - global_max);
        
        p *= __expf(row_max - global_max);
        
        smem_p[tid] = p;
        __syncthreads();
        
        for (int k = 0; k < 128; ++k) {
            acc_o += smem_p[k] * __bfloat162float(smem_V[k * 129 + tid]);
        }
        
        __syncthreads(); 
    }
    
    if (global_sum > 0.0f) {
        float out_val = acc_o / global_sum;
        g_O[q_idx * 128 + tid] = __float2bfloat16(out_val);
        
        if (tid == 0) {
            g_LSE[q_idx] = global_max + __logf(global_sum);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, 
         tvm::ffi::TensorView V, tvm::ffi::TensorView O, 
         tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    if (S == 0) return;
    
    int B_H = B * H;
    int smem_size = 67240; 
    
    CUDA_CHECK(cudaFuncSetAttribute(
        mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    dim3 grid(S, 1, B_H);
    dim3 block(128);
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    mha_kernel<<<grid, block, smem_size, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<__nv_bfloat16*>(O.data_ptr()),
        static_cast<float*>(LSE.data_ptr()),
        S, B_H);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha::run);

}  // namespace tvm_ffi_mha