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

__global__ void __launch_bounds__(64) attention_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* LSE, int S)
{
    int bh = blockIdx.x;
    int row_start = blockIdx.y * 64;
    int row = row_start + threadIdx.x;
    
    if (row >= S) return;
    
    float scale = 1.0f / sqrtf(128.0f);
    
    // Dynamic shared memory for Q, K, V tiles with SWIZZLE_128B layout
    extern __shared__ __align__(1024) char smem_pool[];
    __nv_bfloat16 (*smem_Q)[128] = reinterpret_cast<__nv_bfloat16 (*)[128]>(smem_pool);
    __nv_bfloat16 (*smem_K)[128] = reinterpret_cast<__nv_bfloat16 (*)[128]>(smem_pool + 16384);
    __nv_bfloat16 (*smem_V)[128] = reinterpret_cast<__nv_bfloat16 (*)[128]>(smem_pool + 32768);
    
    // Helper lambda for SWIZZLE_128B coordinate mapping
    auto swizzle_128B = [](int r, int c) {
        return (((c >> 3) ^ (r & 7)) << 3) | (c & 7);
    };
    
    // Load Q tile into SMEM
    for (int idx = threadIdx.x; idx < 64 * 128; idx += blockDim.x) {
        int r = idx / 128;
        int c = idx % 128;
        int global_r = row_start + r;
        if (global_r < S) {
            smem_Q[r][swizzle_128B(r, c)] = Q[bh * S * 128 + global_r * 128 + c];
        } else {
            smem_Q[r][swizzle_128B(r, c)] = __float2bfloat16(0.0f);
        }
    }
    __syncthreads();
    
    float m = -1e20f;
    float l = 0.0f;
    float O_reg[128] = {0};
    
    for (int chunk = 0; chunk < S; chunk += 64) {
        // Load K and V tiles into SMEM
        for (int idx = threadIdx.x; idx < 64 * 128; idx += blockDim.x) {
            int r = idx / 128;
            int c = idx % 128;
            int global_r = chunk + r;
            if (global_r < S) {
                smem_K[r][swizzle_128B(r, c)] = K[bh * S * 128 + global_r * 128 + c];
                smem_V[r][swizzle_128B(r, c)] = V[bh * S * 128 + global_r * 128 + c];
            } else {
                smem_K[r][swizzle_128B(r, c)] = __float2bfloat16(0.0f);
                smem_V[r][swizzle_128B(r, c)] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();
        
        // Compute chunk and online softmax
        for (int j = 0; j < 64; j++) {
            int global_j = chunk + j;
            float s = 0;
            if (global_j < S) {
                for (int d = 0; d < 128; d++) {
                    s += __bfloat162float(smem_Q[threadIdx.x][swizzle_128B(threadIdx.x, d)]) * 
                         __bfloat162float(smem_K[j][swizzle_128B(j, d)]);
                }
                s *= scale;
            } else {
                s = -1e20f;
            }
            
            float new_m = fmaxf(m, s);
            if (new_m != m) {
                for (int d = 0; d < 128; d++) {
                    O_reg[d] *= expf(m - new_m);
                }
                l *= expf(m - new_m);
                m = new_m;
            }
            float p = expf(s - m);
            l += p;
            if (global_j < S) {
                for (int d = 0; d < 128; d++) {
                    O_reg[d] += p * __bfloat162float(smem_V[j][swizzle_128B(j, d)]);
                }
            }
        }
    }
    
    // Write output
    for (int d = 0; d < 128; d++) {
        O[bh * S * 128 + row * 128 + d] = __float2bfloat16(O_reg[d] / l);
    }
    LSE[bh * S + row] = m + logf(l);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE.data_ptr());
    
    dim3 grid(B * H, (S + 63) / 64);
    dim3 block(64);
    int smem_size = 49152;
    
    CUDA_CHECK(cudaFuncSetAttribute(attention_kernel, 
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    attention_kernel<<<grid, block, smem_size, stream>>>(Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr, S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda