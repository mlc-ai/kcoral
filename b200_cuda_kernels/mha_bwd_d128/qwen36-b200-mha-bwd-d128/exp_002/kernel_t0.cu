#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

constexpr int BLOCK_SIZE = 256;

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

__device__ __forceinline__ __nv_bfloat16 f32_to_bf16(float v) {
    return __float2bfloat16(v);
}

__global__ void mha_bwd_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* O, const __nv_bfloat16* dO, const float* L,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    float* deltas,
    int B, int H, int S, int d, float scale) 
{
    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    
    int stride_sd = d;
    int stride_hd = S * d;
    int stride_bhd = H * S * d;
    int stride_hs = S;
    int stride_bhs = H * S;
    
    const __nv_bfloat16* Q_h = Q + b * stride_bhd + h * stride_hd;
    const __nv_bfloat16* K_h = K + b * stride_bhd + h * stride_hd;
    const __nv_bfloat16* V_h = V + b * stride_bhd + h * stride_hd;
    const __nv_bfloat16* dO_h = dO + b * stride_bhd + h * stride_hd;
    const float* L_h = L + b * stride_bhs + h * stride_hs;
    __nv_bfloat16* dQ_h = dQ + b * stride_bhd + h * stride_hd;
    __nv_bfloat16* dK_h = dK + b * stride_bhd + h * stride_hd;
    __nv_bfloat16* dV_h = dV + b * stride_bhd + h * stride_hd;
    float* delta_h = deltas + b * stride_bhs + h * stride_hs;
    
    int tid = threadIdx.x;
    
    // Phase 1: Compute delta[i] = sum_j P[i][j] * dot(dO[i], V[j])
    // P[i][j] = exp(scale * dot(Q[i], K[j]) - L[i])
    for (int i = tid; i < S; i += BLOCK_SIZE) {
        float delta_val = 0.0f;
        float L_i = L_h[i];
        for (int j = 0; j < S; ++j) {
            float qk = 0.0f;
            float dov = 0.0f;
            for (int dd = 0; dd < d; ++dd) {
                qk += bf16_to_f32(Q_h[i * stride_sd + dd]) * bf16_to_f32(K_h[j * stride_sd + dd]);
                dov += bf16_to_f32(dO_h[i * stride_sd + dd]) * bf16_to_f32(V_h[j * stride_sd + dd]);
            }
            float p_ij = expf(scale * qk - L_i);
            delta_val += p_ij * dov;
        }
        delta_h[i] = delta_val;
    }
    __syncthreads();
    
    // Phase 2: Compute dV[j][dj] = sum_i P[i][j] * dO[i][dj]
    for (int out_idx = tid; out_idx < S * d; out_idx += BLOCK_SIZE) {
        int j = out_idx / d;
        int dj = out_idx % d;
        float acc = 0.0f;
        for (int i = 0; i < S; ++i) {
            float qk = 0.0f;
            for (int dd = 0; dd < d; ++dd) {
                qk += bf16_to_f32(Q_h[i * stride_sd + dd]) * bf16_to_f32(K_h[j * stride_sd + dd]);
            }
            float p_ij = expf(scale * qk - L_h[i]);
            acc += p_ij * bf16_to_f32(dO_h[i * stride_sd + dj]);
        }
        dV_h[out_idx] = f32_to_bf16(acc);
    }
    __syncthreads();
    
    // Phase 3: Compute dQ[i][di] = scale * sum_j ds_ij * K[j][di]
    // ds_ij = P[i][j] * (dot(dO[i], V[j]) - delta[i])
    for (int out_idx = tid; out_idx < S * d; out_idx += BLOCK_SIZE) {
        int i = out_idx / d;
        int di = out_idx % d;
        float acc = 0.0f;
        float L_i = L_h[i];
        float delta_i = delta_h[i];
        for (int j = 0; j < S; ++j) {
            float qk = 0.0f;
            float dov = 0.0f;
            for (int dd = 0; dd < d; ++dd) {
                qk += bf16_to_f32(Q_h[i * stride_sd + dd]) * bf16_to_f32(K_h[j * stride_sd + dd]);
                dov += bf16_to_f32(dO_h[i * stride_sd + dd]) * bf16_to_f32(V_h[j * stride_sd + dd]);
            }
            float p_ij = expf(scale * qk - L_i);
            float ds_ij = p_ij * (dov - delta_i);
            acc += ds_ij * bf16_to_f32(K_h[j * stride_sd + di]) * scale;
        }
        dQ_h[out_idx] = f32_to_bf16(acc);
    }
    __syncthreads();
    
    // Phase 4: Compute dK[j][dj] = scale * sum_i ds_ij * Q[i][dj]
    for (int out_idx = tid; out_idx < S * d; out_idx += BLOCK_SIZE) {
        int j = out_idx / d;
        int dj = out_idx % d;
        float acc = 0.0f;
        for (int i = 0; i < S; ++i) {
            float qk = 0.0f;
            float dov = 0.0f;
            for (int dd = 0; dd < d; ++dd) {
                qk += bf16_to_f32(Q_h[i * stride_sd + dd]) * bf16_to_f32(K_h[j * stride_sd + dd]);
                dov += bf16_to_f32(dO_h[i * stride_sd + dd]) * bf16_to_f32(V_h[j * stride_sd + dd]);
            }
            float p_ij = expf(scale * qk - L_h[i]);
            float ds_ij = p_ij * (dov - delta_h[i]);
            acc += ds_ij * bf16_to_f32(Q_h[i * stride_sd + dj]) * scale;
        }
        dK_h[out_idx] = f32_to_bf16(acc);
    }
}

namespace mha_bwd_impl {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = 4;
    int64_t H = 48;
    int64_t d = 128;
    int64_t S = Q.size(2);
    float scale = 1.0f / std::sqrt(static_cast<float>(d));
    
    // Allocate temporary buffer for delta values (size: B*H*S floats)
    size_t delta_bytes = B * H * S * sizeof(float);
    float* d_deltas = nullptr;
    CUDA_CHECK(cudaMalloc(&d_deltas, delta_bytes));
    CUDA_CHECK(cudaMemset(d_deltas, 0, delta_bytes));
    
    dim3 grid(B * H);
    dim3 block(BLOCK_SIZE);
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    mha_bwd_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(O.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        d_deltas,
        static_cast<int>(B), static_cast<int>(H), static_cast<int>(S), static_cast<int>(d), scale
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    CUDA_CHECK(cudaFree(d_deltas));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);

} // namespace mha_bwd_impl