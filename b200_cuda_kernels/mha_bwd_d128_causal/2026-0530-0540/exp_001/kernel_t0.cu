#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <stdio.h>
#include <cmath>
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

namespace tvm_ffi_mha_bwd {

__inline__ __device__ float blockReduceSum(float val, float* shared_mem) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    if (threadIdx.x % 32 == 0) shared_mem[threadIdx.x / 32] = val;
    __syncthreads();
    val = (threadIdx.x < 32) ? shared_mem[threadIdx.x] : 0.0f;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    if (threadIdx.x == 0) shared_mem[0] = val;
    __syncthreads();
    float res = shared_mem[0];
    __syncthreads();
    return res;
}

__global__ void compute_dQ(
    const __nv_bfloat16* __restrict__ Q, 
    const __nv_bfloat16* __restrict__ K, 
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O, 
    const __nv_bfloat16* __restrict__ dO, 
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ, 
    int S, int d, float scale
) {
    int i = blockIdx.x; 
    int bh = blockIdx.y;
    int tid = threadIdx.x; 
    
    int base_idx = bh * S * d;
    int q_offset = base_idx + i * d + tid;
    
    float q_val = tid < d ? __bfloat162float(Q[q_offset]) : 0.0f;
    float o_val = tid < d ? __bfloat162float(O[q_offset]) : 0.0f;
    float do_val = tid < d ? __bfloat162float(dO[q_offset]) : 0.0f;
    
    __shared__ float shared_mem[32];
    
    float D_i = blockReduceSum(o_val * do_val, shared_mem);
    float l_i = L[bh * S + i];
    float dq_val = 0.0f;
    
    for (int j = 0; j <= i; ++j) {
        int k_offset = base_idx + j * d + tid;
        
        float k_val = tid < d ? __bfloat162float(K[k_offset]) : 0.0f;
        float v_val = tid < d ? __bfloat162float(V[k_offset]) : 0.0f;
        
        float qk = blockReduceSum(q_val * k_val, shared_mem);
        float S_ij = qk * scale;
        float P_ij = expf(S_ij - l_i);
        
        float dS_ij = blockReduceSum(do_val * v_val, shared_mem);
        float dP_ij = P_ij * (dS_ij - D_i);
        
        dq_val += dP_ij * k_val;
    }
    
    dq_val *= scale;
    if (tid < d) {
        dQ[q_offset] = __float2bfloat16(dq_val);
    }
}

__global__ void compute_dK_dV(
    const __nv_bfloat16* __restrict__ Q, 
    const __nv_bfloat16* __restrict__ K, 
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O, 
    const __nv_bfloat16* __restrict__ dO, 
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dK, 
    __nv_bfloat16* __restrict__ dV,
    int S, int d, float scale
) {
    int j = blockIdx.x; 
    int bh = blockIdx.y;
    int tid = threadIdx.x; 
    
    int base_idx = bh * S * d;
    int j_offset = base_idx + j * d + tid;
    
    float k_val = tid < d ? __bfloat162float(K[j_offset]) : 0.0f;
    float v_val = tid < d ? __bfloat162float(V[j_offset]) : 0.0f;
    
    __shared__ float shared_mem[32];
    
    float dk_val = 0.0f;
    float dv_val = 0.0f;
    
    for (int i = j; i < S; ++i) {
        int i_offset = base_idx + i * d + tid;
        
        float q_val = tid < d ? __bfloat162float(Q[i_offset]) : 0.0f;
        float o_val = tid < d ? __bfloat162float(O[i_offset]) : 0.0f;
        float do_val = tid < d ? __bfloat162float(dO[i_offset]) : 0.0f;
        
        float D_i = blockReduceSum(o_val * do_val, shared_mem);
        float qk = blockReduceSum(q_val * k_val, shared_mem);
        
        float l_i = L[bh * S + i];
        float S_ij = qk * scale;
        float P_ij = expf(S_ij - l_i);
        
        float dS_ij = blockReduceSum(do_val * v_val, shared_mem);
        float dP_ij = P_ij * (dS_ij - D_i);
        
        dk_val += dP_ij * q_val;
        dv_val += P_ij * do_val;
    }
    
    dk_val *= scale;
    
    if (tid < d) {
        dK[j_offset] = __float2bfloat16(dk_val);
        dV[j_offset] = __float2bfloat16(dv_val);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int d = Q.size(3);
    
    float scale = 1.0f / std::sqrt(static_cast<float>(d));
    
    dim3 grid(S, B * H);
    dim3 block(128);
    
    compute_dQ<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(O.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        S, d, scale
    );
    
    compute_dK_dV<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(O.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        S, d, scale
    );
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

} // namespace tvm_ffi_mha_bwd