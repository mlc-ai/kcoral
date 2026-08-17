#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define CUDA_CHECK(call) \
    do { cudaError_t e = call; if (e != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

constexpr int BLOCK_SIZE = 256;

__device__ static inline float bf162f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ static inline __nv_bfloat16 f2bf16(float x) { return __float2bfloat16(x); }

// Phase 1 kernel: compute delta[i] for each i
__global__ void mha_bwd_delta_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    float* __restrict__ delta_out,
    int B, int H, int S, int d, float scale)
{
    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    int tid = threadIdx.x;
    
    const __nv_bfloat16* Q_bh = &Q[((int64_t)b * H + h) * (int64_t)S * d];
    const __nv_bfloat16* K_bh = &K[((int64_t)b * H + h) * (int64_t)S * d];
    const __nv_bfloat16* V_bh = &V[((int64_t)b * H + h) * (int64_t)S * d];
    const __nv_bfloat16* dO_bh = &dO[((int64_t)b * H + h) * (int64_t)S * d];
    const float* L_bh = &L[((int64_t)b * H + h) * S];
    float* delta = &delta_out[((int64_t)b * H + h) * S];
    
    for (int i = tid; i < S; i += BLOCK_SIZE) {
        float acc = 0.0f;
        float Li = L_bh[i];
        int qi_off = i * d;
        int doi_off = i * d;
        
        for (int j = 0; j < S; ++j) {
            float qk = 0.0f;
            float dov = 0.0f;
            int kj_off = j * d;
            int vj_off = j * d;
            
            for (int dd = 0; dd < d; dd++) {
                qk += bf162f(Q_bh[qi_off + dd]) * bf162f(K_bh[kj_off + dd]);
                dov += bf162f(dO_bh[doi_off + dd]) * bf162f(V_bh[vj_off + dd]);
            }
            
            acc += expf(scale * qk - Li) * dov;
        }
        delta[i] = acc;
    }
}

// Phase 2 kernel: compute dV
__global__ void mha_bwd_dv_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int S, int d, float scale)
{
    int bh = blockIdx.x;
    int tid = threadIdx.x;
    
    const __nv_bfloat16* Q_bh = &Q[(int64_t)bh * S * d];
    const __nv_bfloat16* K_bh = &K[(int64_t)bh * S * d];
    const __nv_bfloat16* dO_bh = &dO[(int64_t)bh * S * d];
    const float* L_bh = &L[bh * S];
    __nv_bfloat16* dV_bh = &dV[(int64_t)bh * S * d];
    
    for (int out_idx = tid; out_idx < S * d; out_idx += BLOCK_SIZE) {
        int j = out_idx / d;
        int di = out_idx % d;
        float acc = 0.0f;
        int kj_off = j * d;
        
        for (int i = 0; i < S; ++i) {
            float qk = 0.0f;
            int qi_off = i * d;
            
            for (int dd = 0; dd < d; dd++) {
                qk += bf162f(Q_bh[qi_off + dd]) * bf162f(K_bh[kj_off + dd]);
            }
            
            acc += expf(scale * qk - L_bh[i]) * bf162f(dO_bh[i * d + di]);
        }
        dV_bh[out_idx] = f2bf16(acc);
    }
}

// Phase 3 kernel: compute dQ
__global__ void mha_bwd_dq_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ delta_in,
    __nv_bfloat16* __restrict__ dQ,
    int B, int H, int S, int d, float scale)
{
    int bh = blockIdx.x;
    int tid = threadIdx.x;
    
    const __nv_bfloat16* Q_bh = &Q[(int64_t)bh * S * d];
    const __nv_bfloat16* K_bh = &K[(int64_t)bh * S * d];
    const __nv_bfloat16* V_bh = &V[(int64_t)bh * S * d];
    const __nv_bfloat16* dO_bh = &dO[(int64_t)bh * S * d];
    const float* L_bh = &L[bh * S];
    const float* delta = &delta_in[bh * S];
    __nv_bfloat16* dQ_bh = &dQ[(int64_t)bh * S * d];
    
    for (int out_idx = tid; out_idx < S * d; out_idx += BLOCK_SIZE) {
        int i = out_idx / d;
        int di = out_idx % d;
        float acc = 0.0f;
        float Li = L_bh[i];
        float delta_i = delta[i];
        int qi_off = i * d;
        int doi_off = i * d;
        
        for (int j = 0; j < S; ++j) {
            float qk = 0.0f;
            float dov = 0.0f;
            int kj_off = j * d;
            int vj_off = j * d;
            
            for (int dd = 0; dd < d; dd++) {
                qk += bf162f(Q_bh[qi_off + dd]) * bf162f(K_bh[kj_off + dd]);
                dov += bf162f(dO_bh[doi_off + dd]) * bf162f(V_bh[vj_off + dd]);
            }
            
            float ds = expf(scale * qk - Li) * (dov - delta_i);
            acc += ds * bf162f(K_bh[kj_off + di]) * scale;
        }
        dQ_bh[out_idx] = f2bf16(acc);
    }
}

// Phase 4 kernel: compute dK
__global__ void mha_bwd_dk_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ delta_in,
    __nv_bfloat16* __restrict__ dK,
    int B, int H, int S, int d, float scale)
{
    int bh = blockIdx.x;
    int tid = threadIdx.x;
    
    const __nv_bfloat16* Q_bh = &Q[(int64_t)bh * S * d];
    const __nv_bfloat16* K_bh = &K[(int64_t)bh * S * d];
    const __nv_bfloat16* V_bh = &V[(int64_t)bh * S * d];
    const __nv_bfloat16* dO_bh = &dO[(int64_t)bh * S * d];
    const float* L_bh = &L[bh * S];
    const float* delta = &delta_in[bh * S];
    __nv_bfloat16* dK_bh = &dK[(int64_t)bh * S * d];
    
    for (int out_idx = tid; out_idx < S * d; out_idx += BLOCK_SIZE) {
        int j = out_idx / d;
        int dj = out_idx % d;
        float acc = 0.0f;
        int kj_off = j * d;
        int vj_off = j * d;
        
        for (int i = 0; i < S; ++i) {
            float qk = 0.0f;
            float dov = 0.0f;
            int qi_off = i * d;
            int doi_off = i * d;
            
            for (int dd = 0; dd < d; dd++) {
                qk += bf162f(Q_bh[qi_off + dd]) * bf162f(K_bh[kj_off + dd]);
                dov += bf162f(dO_bh[doi_off + dd]) * bf162f(V_bh[vj_off + dd]);
            }
            
            float ds = expf(scale * qk - L_bh[i]) * (dov - delta[i]);
            acc += ds * bf162f(Q_bh[qi_off + dj]) * scale;
        }
        dK_bh[out_idx] = f2bf16(acc);
    }
}

namespace mha_bwd_impl {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = 4, H = 48, d = 128;
    int64_t S = Q.size(2);
    float scale = 1.0f / std::sqrt(static_cast<float>(d));
    int64_t bh_total = B * H;
    
    // Allocate delta workspace
    float* d_delta = nullptr;
    CUDA_CHECK(cudaMalloc(&d_delta, bh_total * S * sizeof(float)));
    
    dim3 grid(bh_total);
    dim3 block(BLOCK_SIZE);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    // Phase 1: delta
    mha_bwd_delta_kernel<<<grid, block, 0, stream>>>(
        Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, d_delta,
        B, H, S, d, scale);
    CUDA_CHECK(cudaGetLastError());
    
    // Phase 2: dV
    mha_bwd_dv_kernel<<<grid, block, 0, stream>>>(
        Q_ptr, K_ptr, dO_ptr, L_ptr, dV_ptr,
        B, H, S, d, scale);
    CUDA_CHECK(cudaGetLastError());
    
    // Phase 3: dQ
    mha_bwd_dq_kernel<<<grid, block, 0, stream>>>(
        Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, d_delta, dQ_ptr,
        B, H, S, d, scale);
    CUDA_CHECK(cudaGetLastError());
    
    // Phase 4: dK
    mha_bwd_dk_kernel<<<grid, block, 0, stream>>>(
        Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, d_delta, dK_ptr,
        B, H, S, d, scale);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_delta));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);

} // namespace mha_bwd_impl