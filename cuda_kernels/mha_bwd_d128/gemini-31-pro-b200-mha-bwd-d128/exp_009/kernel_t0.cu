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

#define S_Q_STRIDE 130
#define S_DO_STRIDE 130
#define S_DQ_STRIDE 129
#define S_K_STRIDE 130
#define S_V_STRIDE 130
#define S_S_STRIDE 65
#define S_DP_STRIDE 65
#define S_P_STRIDE 65

__device__ void load_Q_dO_O_compute_D(
    const __nv_bfloat16* Q_g, const __nv_bfloat16* dO_g, const __nv_bfloat16* O_g, const float* L_g,
    __nv_bfloat16* s_Q, __nv_bfloat16* s_dO, float* s_D, float* s_L,
    int i, int S, int d, int tx) 
{
    int r = tx / 2;
    int c_start = (tx % 2) * 64;
    
    if (tx < 64) {
        s_D[tx] = 0.0f;
        if (i + tx < S) s_L[tx] = L_g[i + tx];
        else s_L[tx] = 0.0f;
    }
    __syncthreads();
    
    float partial_D = 0.0f;
    for (int c = c_start; c < c_start + 64; ++c) {
        if (i + r < S) {
            __nv_bfloat16 q = Q_g[(size_t)(i + r) * d + c];
            __nv_bfloat16 do_val = dO_g[(size_t)(i + r) * d + c];
            __nv_bfloat16 o_val = O_g[(size_t)(i + r) * d + c];
            
            s_Q[r * S_Q_STRIDE + c] = q;
            s_dO[r * S_DO_STRIDE + c] = do_val;
            
            partial_D += (float)do_val * (float)o_val;
        } else {
            s_Q[r * S_Q_STRIDE + c] = __float2bfloat16(0.0f);
            s_dO[r * S_DO_STRIDE + c] = __float2bfloat16(0.0f);
        }
    }
    
    atomicAdd(&s_D[r], partial_D);
    __syncthreads();
}

__device__ void load_K_V(
    const __nv_bfloat16* K_g, const __nv_bfloat16* V_g,
    __nv_bfloat16* s_K, __nv_bfloat16* s_V,
    int j, int S, int d, int tx) 
{
    int r = tx / 2;
    int c_start = (tx % 2) * 64;
    
    for (int c = c_start; c < c_start + 64; ++c) {
        if (j + r < S) {
            s_K[r * S_K_STRIDE + c] = K_g[(size_t)(j + r) * d + c];
            s_V[r * S_V_STRIDE + c] = V_g[(size_t)(j + r) * d + c];
        } else {
            s_K[r * S_K_STRIDE + c] = __float2bfloat16(0.0f);
            s_V[r * S_V_STRIDE + c] = __float2bfloat16(0.0f);
        }
    }
    __syncthreads();
}

__device__ void init_dQ(float* s_dQ, int tx) {
    int r = tx / 2;
    int c_start = (tx % 2) * 64;
    for (int c = c_start; c < c_start + 64; ++c) {
        s_dQ[r * S_DQ_STRIDE + c] = 0.0f;
    }
    __syncthreads();
}

__device__ void compute_S(
    const __nv_bfloat16* s_Q, const __nv_bfloat16* s_K,
    float* s_S_dS, float scale, int tx) 
{
    int r = tx / 2;
    int c_start = (tx % 2) * 32;
    
    for (int c = c_start; c < c_start + 32; ++c) {
        float sum = 0.0f;
        for (int k = 0; k < 128; ++k) {
            sum += (float)s_Q[r * S_Q_STRIDE + k] * (float)s_K[c * S_K_STRIDE + k];
        }
        s_S_dS[r * S_S_STRIDE + c] = sum * scale;
    }
    __syncthreads();
}

__device__ void compute_dP(
    const __nv_bfloat16* s_dO, const __nv_bfloat16* s_V,
    float* s_dP, int tx) 
{
    int r = tx / 2;
    int c_start = (tx % 2) * 32;
    
    for (int c = c_start; c < c_start + 32; ++c) {
        float sum = 0.0f;
        for (int k = 0; k < 128; ++k) {
            sum += (float)s_dO[r * S_DO_STRIDE + k] * (float)s_V[c * S_V_STRIDE + k];
        }
        s_dP[r * S_DP_STRIDE + c] = sum;
    }
    __syncthreads();
}

__device__ void compute_P_and_dS(
    float* s_S_dS, float* s_dP, float* s_L, float* s_D,
    float* s_P, int i, int j, int S, int tx) 
{
    int r = tx / 2;
    int c_start = (tx % 2) * 32;
    
    float l_val = s_L[r];
    float d_val = s_D[r];
    
    bool valid_i = (i + r < S);
    
    for (int c = c_start; c < c_start + 32; ++c) {
        bool valid_j = (j + c < S);
        if (valid_i && valid_j) {
            float s_val = s_S_dS[r * S_S_STRIDE + c];
            float p_val = expf(s_val - l_val);
            s_P[r * S_P_STRIDE + c] = p_val;
            
            float dp_val = s_dP[r * S_DP_STRIDE + c];
            s_S_dS[r * S_S_STRIDE + c] = p_val * (dp_val - d_val);
        } else {
            s_P[r * S_P_STRIDE + c] = 0.0f;
            s_S_dS[r * S_S_STRIDE + c] = 0.0f;
        }
    }
    __syncthreads();
}

__device__ void compute_dQ_add(
    const float* s_dS, const __nv_bfloat16* s_K,
    float* s_dQ, int tx) 
{
    int r = tx / 2;
    int c_start = (tx % 2) * 64;
    
    for (int c = c_start; c < c_start + 64; ++c) {
        float sum = 0.0f;
        for (int k = 0; k < 64; ++k) {
            sum += s_dS[r * S_S_STRIDE + k] * (float)s_K[k * S_K_STRIDE + c];
        }
        s_dQ[r * S_DQ_STRIDE + c] += sum;
    }
    __syncthreads();
}

__device__ void compute_dK_dV_add_global(
    const float* s_dS, const float* s_P,
    const __nv_bfloat16* s_Q, const __nv_bfloat16* s_dO,
    __nv_bfloat16* dK_g, __nv_bfloat16* dV_g,
    int j, int S, int d, int tx) 
{
    int r = tx / 2;
    int c_start = (tx % 2) * 64;
    
    if (j + r >= S) return; 
    
    for (int c = c_start; c < c_start + 64; ++c) {
        float sum_dk = 0.0f;
        float sum_dv = 0.0f;
        for (int k = 0; k < 64; ++k) {
            float ds_val = s_dS[k * S_S_STRIDE + r]; 
            float p_val  = s_P[k * S_P_STRIDE + r];
            float q_val  = (float)s_Q[k * S_Q_STRIDE + c];
            float do_val = (float)s_dO[k * S_DO_STRIDE + c];
            
            sum_dk += ds_val * q_val;
            sum_dv += p_val * do_val;
        }
        
        size_t idx = (size_t)(j + r) * d + c;
        float curr_dk = (float)dK_g[idx];
        float curr_dv = (float)dV_g[idx];
        
        dK_g[idx] = __float2bfloat16(curr_dk + sum_dk);
        dV_g[idx] = __float2bfloat16(curr_dv + sum_dv);
    }
}

__device__ void write_dQ(
    const float* s_dQ, __nv_bfloat16* dQ_g,
    int i, int S, int d, int tx) 
{
    int r = tx / 2;
    int c_start = (tx % 2) * 64;
    
    if (i + r < S) {
        for (int c = c_start; c < c_start + 64; ++c) {
            dQ_g[(size_t)(i + r) * d + c] = __float2bfloat16(s_dQ[r * S_DQ_STRIDE + c]);
        }
    }
    __syncthreads();
}

__global__ void fa_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int S, int d, float scale)
{
    int b = blockIdx.x;
    int h = blockIdx.y;
    int tx = threadIdx.x;
    
    int64_t base_seq = (int64_t)(b * gridDim.y + h) * S * d;
    int64_t base_L = (int64_t)(b * gridDim.y + h) * S;
    
    const __nv_bfloat16* Q_g = Q + base_seq;
    const __nv_bfloat16* K_g = K + base_seq;
    const __nv_bfloat16* V_g = V + base_seq;
    const __nv_bfloat16* O_g = O + base_seq;
    const __nv_bfloat16* dO_g = dO + base_seq;
    const float* L_g = L + base_L;
    
    __nv_bfloat16* dQ_g = dQ + base_seq;
    __nv_bfloat16* dK_g = dK + base_seq;
    __nv_bfloat16* dV_g = dV + base_seq;
    
    extern __shared__ char smem[];
    
    __nv_bfloat16* s_Q = (__nv_bfloat16*)smem;
    __nv_bfloat16* s_dO = s_Q + 64 * S_Q_STRIDE;
    float* s_dQ = (float*)(s_dO + 64 * S_DO_STRIDE);
    
    __nv_bfloat16* s_K = (__nv_bfloat16*)(s_dQ + 64 * S_DQ_STRIDE);
    __nv_bfloat16* s_V = s_K + 64 * S_K_STRIDE;
    
    float* s_S_dS = (float*)(s_V + 64 * S_V_STRIDE);
    float* s_dP = s_S_dS + 64 * S_S_STRIDE;
    float* s_P = s_dP + 64 * S_DP_STRIDE;
    
    float* s_D = s_P + 64 * S_P_STRIDE;
    float* s_L = s_D + 64;
    
    for (int i = 0; i < S; i += 64) {
        load_Q_dO_O_compute_D(Q_g, dO_g, O_g, L_g, s_Q, s_dO, s_D, s_L, i, S, d, tx);
        init_dQ(s_dQ, tx);
        
        for (int j = 0; j < S; j += 64) {
            load_K_V(K_g, V_g, s_K, s_V, j, S, d, tx);
            
            compute_S(s_Q, s_K, s_S_dS, scale, tx);
            compute_dP(s_dO, s_V, s_dP, tx);
            compute_P_and_dS(s_S_dS, s_dP, s_L, s_D, s_P, i, j, S, tx);
            
            compute_dQ_add(s_S_dS, s_K, s_dQ, tx);
            compute_dK_dV_add_global(s_S_dS, s_P, s_Q, s_dO, dK_g, dV_g, j, S, d, tx);
            
            __syncthreads();
        }
        
        write_dQ(s_dQ, dQ_g, i, S, d, tx);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3);

    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    size_t bytes = (size_t)B * H * S * d * sizeof(__nv_bfloat16);
    CUDA_CHECK(cudaMemsetAsync(dK.data_ptr(), 0, bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(dV.data_ptr(), 0, bytes, stream));

    dim3 grid(B, H, 1);
    dim3 block(128, 1, 1);
    
    float scale = 1.0f / sqrtf((float)d);

    int smem_size = 150016;
    CUDA_CHECK(cudaFuncSetAttribute(fa_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    fa_bwd_kernel<<<grid, block, smem_size, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(O.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        (int)S, (int)d, scale
    );
    CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

} // namespace tvm_ffi_example_cuda