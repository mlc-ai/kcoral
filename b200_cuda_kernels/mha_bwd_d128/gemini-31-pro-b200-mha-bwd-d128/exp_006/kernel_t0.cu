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

namespace tvm_ffi_mha_bwd {

__device__ __forceinline__ uint32_t pack_bf16(float a, float b) {
    __nv_bfloat16 ba = __float2bfloat16(a);
    __nv_bfloat16 bb = __float2bfloat16(b);
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&ba)),
          "h"(*reinterpret_cast<uint16_t*>(&bb)));
    return result;
}

__global__ void pass1_dQ(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    int B, int H, int S,
    float scale
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int q_idx = blockIdx.z;
    
    int Br = 32;
    int Bc = 32;
    int d = 128;
    
    int start_q = q_idx * Br;
    if (start_q >= S) return;
    
    int tid = threadIdx.x;
    int num_threads = blockDim.x; 
    
    long long base_idx = ((long long)b * H + h) * S * d;
    long long q_base = base_idx + start_q * d;
    
    __align__(16) __shared__ __nv_bfloat16 s_Q[32 * 128];
    __align__(16) __shared__ __nv_bfloat16 s_O[32 * 128];
    __align__(16) __shared__ __nv_bfloat16 s_dO[32 * 128];
    __align__(16) __shared__ __nv_bfloat16 s_K[32 * 128];
    __align__(16) __shared__ __nv_bfloat16 s_V[32 * 128];
    __align__(16) __shared__ float s_L[32];
    __align__(16) __shared__ float s_D[32];
    
    for (int i = tid; i < (Br * d) / 8; i += num_threads) {
        int row = i / 16;
        if (start_q + row < S) {
            ((uint4*)s_Q)[i] = ((const uint4*)(Q + q_base))[i];
            ((uint4*)s_O)[i] = ((const uint4*)(O + q_base))[i];
            ((uint4*)s_dO)[i] = ((const uint4*)(dO + q_base))[i];
        } else {
            uint4 zero = {0, 0, 0, 0};
            ((uint4*)s_Q)[i] = zero;
            ((uint4*)s_O)[i] = zero;
            ((uint4*)s_dO)[i] = zero;
        }
    }
    
    for (int i = tid; i < Br; i += num_threads) {
        if (start_q + i < S) {
            long long l_idx = ((long long)b * H + h) * S + start_q + i;
            s_L[i] = L[l_idx];
            
            float D_val = 0.0f;
            __nv_bfloat162* o_ptr = (__nv_bfloat162*)&s_O[i * d];
            __nv_bfloat162* do_ptr = (__nv_bfloat162*)&s_dO[i * d];
            for (int j = 0; j < 64; ++j) {
                float2 o2 = __bfloat1622float2(o_ptr[j]);
                float2 do2 = __bfloat1622float2(do_ptr[j]);
                D_val += o2.x * do2.x + o2.y * do2.y;
            }
            s_D[i] = D_val;
        } else {
            s_L[i] = -1e20f;
            s_D[i] = 0.0f;
        }
    }
    
    __syncthreads();
    
    int row = tid / 4; 
    int col_start = (tid % 4) * 32;
    float dQ_acc[32] = {0.0f};
    
    for (int start_k = 0; start_k < S; start_k += Bc) {
        long long k_base = base_idx + start_k * d;
        
        for (int i = tid; i < (Bc * d) / 8; i += num_threads) {
            int r = i / 16;
            if (start_k + r < S) {
                ((uint4*)s_K)[i] = ((const uint4*)(K + k_base))[i];
                ((uint4*)s_V)[i] = ((const uint4*)(V + k_base))[i];
            } else {
                uint4 zero = {0, 0, 0, 0};
                ((uint4*)s_K)[i] = zero;
                ((uint4*)s_V)[i] = zero;
            }
        }
        __syncthreads();
        
        __nv_bfloat162* q_ptr = (__nv_bfloat162*)&s_Q[row * d];
        __nv_bfloat162* do_ptr = (__nv_bfloat162*)&s_dO[row * d];
        
        for (int j = 0; j < Bc; ++j) {
            if (start_k + j >= S) continue;
            
            __nv_bfloat162* k_ptr = (__nv_bfloat162*)&s_K[j * d];
            __nv_bfloat162* v_ptr = (__nv_bfloat162*)&s_V[j * d];
            
            float s_ij = 0.0f;
            float dp_ij = 0.0f;
            
            for (int c = 0; c < 64; ++c) {
                float2 q2 = __bfloat1622float2(q_ptr[c]);
                float2 k2 = __bfloat1622float2(k_ptr[c]);
                s_ij += q2.x * k2.x + q2.y * k2.y;
                
                float2 do2 = __bfloat1622float2(do_ptr[c]);
                float2 v2 = __bfloat1622float2(v_ptr[c]);
                dp_ij += do2.x * v2.x + do2.y * v2.y;
            }
            s_ij *= scale;
            
            float p_ij = expf(s_ij - s_L[row]);
            float ds_ij = p_ij * (dp_ij - s_D[row]);
            
            __nv_bfloat162* k_col_ptr = (__nv_bfloat162*)&s_K[j * d + col_start];
            for (int c = 0; c < 16; ++c) {
                float2 k2 = __bfloat1622float2(k_col_ptr[c]);
                dQ_acc[c * 2 + 0] += ds_ij * k2.x;
                dQ_acc[c * 2 + 1] += ds_ij * k2.y;
            }
        }
        __syncthreads();
    }
    
    if (start_q + row < S) {
        uint4* dQ_ptr = (uint4*)(dQ + q_base + row * d + col_start);
        for (int c = 0; c < 4; ++c) {
            uint4 out;
            out.x = pack_bf16(dQ_acc[c * 8 + 0], dQ_acc[c * 8 + 1]);
            out.y = pack_bf16(dQ_acc[c * 8 + 2], dQ_acc[c * 8 + 3]);
            out.z = pack_bf16(dQ_acc[c * 8 + 4], dQ_acc[c * 8 + 5]);
            out.w = pack_bf16(dQ_acc[c * 8 + 6], dQ_acc[c * 8 + 7]);
            dQ_ptr[c] = out;
        }
    }
}

__global__ void pass2_dK_dV(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int S,
    float scale
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int k_idx = blockIdx.z;
    
    int Br = 32;
    int Bc = 32;
    int d = 128;
    
    int start_k = k_idx * Bc;
    if (start_k >= S) return;
    
    int tid = threadIdx.x;
    int num_threads = blockDim.x; 
    
    long long base_idx = ((long long)b * H + h) * S * d;
    long long k_base = base_idx + start_k * d;
    
    __align__(16) __shared__ __nv_bfloat16 s_K[32 * 128];
    __align__(16) __shared__ __nv_bfloat16 s_V[32 * 128];
    __align__(16) __shared__ __nv_bfloat16 s_Q[32 * 128];
    __align__(16) __shared__ __nv_bfloat16 s_O[32 * 128];
    __align__(16) __shared__ __nv_bfloat16 s_dO[32 * 128];
    __align__(16) __shared__ float s_L[32];
    __align__(16) __shared__ float s_D[32];
    
    for (int i = tid; i < (Bc * d) / 8; i += num_threads) {
        int row = i / 16;
        if (start_k + row < S) {
            ((uint4*)s_K)[i] = ((const uint4*)(K + k_base))[i];
            ((uint4*)s_V)[i] = ((const uint4*)(V + k_base))[i];
        } else {
            uint4 zero = {0, 0, 0, 0};
            ((uint4*)s_K)[i] = zero;
            ((uint4*)s_V)[i] = zero;
        }
    }
    __syncthreads();
    
    int row = tid / 4; 
    int col_start = (tid % 4) * 32;
    
    float dK_acc[32] = {0.0f};
    float dV_acc[32] = {0.0f};
    
    for (int start_q = 0; start_q < S; start_q += Br) {
        long long q_base = base_idx + start_q * d;
        
        for (int i = tid; i < (Br * d) / 8; i += num_threads) {
            int r = i / 16;
            if (start_q + r < S) {
                ((uint4*)s_Q)[i] = ((const uint4*)(Q + q_base))[i];
                ((uint4*)s_O)[i] = ((const uint4*)(O + q_base))[i];
                ((uint4*)s_dO)[i] = ((const uint4*)(dO + q_base))[i];
            } else {
                uint4 zero = {0, 0, 0, 0};
                ((uint4*)s_Q)[i] = zero;
                ((uint4*)s_O)[i] = zero;
                ((uint4*)s_dO)[i] = zero;
            }
        }
        
        for (int i = tid; i < Br; i += num_threads) {
            if (start_q + i < S) {
                long long l_idx = ((long long)b * H + h) * S + start_q + i;
                s_L[i] = L[l_idx];
                
                float D_val = 0.0f;
                __nv_bfloat162* o_ptr = (__nv_bfloat162*)&s_O[i * d];
                __nv_bfloat162* do_ptr = (__nv_bfloat162*)&s_dO[i * d];
                for (int j = 0; j < 64; ++j) {
                    float2 o2 = __bfloat1622float2(o_ptr[j]);
                    float2 do2 = __bfloat1622float2(do_ptr[j]);
                    D_val += o2.x * do2.x + o2.y * do2.y;
                }
                s_D[i] = D_val;
            } else {
                s_L[i] = -1e20f;
                s_D[i] = 0.0f;
            }
        }
        __syncthreads();
        
        __nv_bfloat162* k_ptr = (__nv_bfloat162*)&s_K[row * d];
        __nv_bfloat162* v_ptr = (__nv_bfloat162*)&s_V[row * d];
        
        for (int i = 0; i < Br; ++i) {
            if (start_q + i >= S) continue;
            
            __nv_bfloat162* q_ptr = (__nv_bfloat162*)&s_Q[i * d];
            __nv_bfloat162* do_ptr = (__nv_bfloat162*)&s_dO[i * d];
            
            float s_ij = 0.0f;
            float dp_ij = 0.0f;
            
            for (int c = 0; c < 64; ++c) {
                float2 q2 = __bfloat1622float2(q_ptr[c]);
                float2 k2 = __bfloat1622float2(k_ptr[c]);
                s_ij += q2.x * k2.x + q2.y * k2.y;
                
                float2 do2 = __bfloat1622float2(do_ptr[c]);
                float2 v2 = __bfloat1622float2(v_ptr[c]);
                dp_ij += do2.x * v2.x + do2.y * v2.y;
            }
            s_ij *= scale;
            
            float p_ij = expf(s_ij - s_L[i]);
            float ds_ij = p_ij * (dp_ij - s_D[i]);
            
            __nv_bfloat162* q_col_ptr = (__nv_bfloat162*)&s_Q[i * d + col_start];
            __nv_bfloat162* do_col_ptr = (__nv_bfloat162*)&s_dO[i * d + col_start];
            for (int c = 0; c < 16; ++c) {
                float2 q2 = __bfloat1622float2(q_col_ptr[c]);
                float2 do2 = __bfloat1622float2(do_col_ptr[c]);
                dK_acc[c * 2 + 0] += ds_ij * q2.x;
                dK_acc[c * 2 + 1] += ds_ij * q2.y;
                dV_acc[c * 2 + 0] += p_ij * do2.x;
                dV_acc[c * 2 + 1] += p_ij * do2.y;
            }
        }
        __syncthreads();
    }
    
    if (start_k + row < S) {
        uint4* dK_ptr = (uint4*)(dK + k_base + row * d + col_start);
        uint4* dV_ptr = (uint4*)(dV + k_base + row * d + col_start);
        for (int c = 0; c < 4; ++c) {
            uint4 outK, outV;
            outK.x = pack_bf16(dK_acc[c * 8 + 0], dK_acc[c * 8 + 1]);
            outK.y = pack_bf16(dK_acc[c * 8 + 2], dK_acc[c * 8 + 3]);
            outK.z = pack_bf16(dK_acc[c * 8 + 4], dK_acc[c * 8 + 5]);
            outK.w = pack_bf16(dK_acc[c * 8 + 6], dK_acc[c * 8 + 7]);
            dK_ptr[c] = outK;
            
            outV.x = pack_bf16(dV_acc[c * 8 + 0], dV_acc[c * 8 + 1]);
            outV.y = pack_bf16(dV_acc[c * 8 + 2], dV_acc[c * 8 + 3]);
            outV.z = pack_bf16(dV_acc[c * 8 + 4], dV_acc[c * 8 + 5]);
            outV.w = pack_bf16(dV_acc[c * 8 + 6], dV_acc[c * 8 + 7]);
            dV_ptr[c] = outV;
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    
    float scale = 1.0f / sqrtf(128.0f); 
    
    dim3 grid1(B, H, (S + 31) / 32);
    dim3 block(128);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    const __nv_bfloat16* q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* k_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* v_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* o_ptr = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* do_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* l_ptr = static_cast<const float*>(L.data_ptr());
    
    __nv_bfloat16* dq_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dk_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dv_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    pass1_dQ<<<grid1, block, 0, stream>>>(q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, l_ptr, dq_ptr, B, H, S, scale);
    CUDA_CHECK(cudaGetLastError());
    
    pass2_dK_dV<<<grid1, block, 0, stream>>>(q_ptr, k_ptr, v_ptr, o_ptr, do_ptr, l_ptr, dk_ptr, dv_ptr, B, H, S, scale);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}  // namespace tvm_ffi_mha_bwd