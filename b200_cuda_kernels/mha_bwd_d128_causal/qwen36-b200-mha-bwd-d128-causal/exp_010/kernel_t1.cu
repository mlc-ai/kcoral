#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <tvm/ffi/tvm_ffi.h>

#ifndef TVM_FFI_EXTERN_C
#define TVM_FFI_EXTERN_C extern "C"
#endif

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

namespace mha_bwd_impl {

static constexpr int DH = 128;

template<int MAX_S>
__global__ void mha_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    float* __restrict__ dQ_f,
    float* __restrict__ dK_f,
    float* __restrict__ dV_f,
    int S, int BH, int S_dh) {
    
    int b = blockIdx.y;
    int h = blockIdx.z;
    int bh = b * 48 + h;
    
    uint64_t slice_off = (uint64_t)bh * S_dh;
    uint64_t Q_off = slice_off;
    uint64_t K_off = slice_off;
    uint64_t V_off = slice_off;
    uint64_t dO_off = slice_off;
    int L_off = bh * S;
    
    float inv_sd = rsqrtf((float)DH);
    
    int tid = threadIdx.x;
    if (tid >= S) return;
    int q = tid;
    
    float lse_q = L[L_off + q];
    uint64_t qD_off = Q_off + (uint64_t)q * DH;
    
    float q_f[DH];
    for (int j = 0; j < DH; j++) {
        q_f[j] = __bfloat162float(Q[qD_off + j]);
    }
    
    float dq_accum[DH];
    for (int j = 0; j < DH; j++) dq_accum[j] = 0.0f;
    
    float drow = 0.0f;
    
    for (int k = 0; k < S; k++) {
        if (k > q) continue;
        
        uint64_t kD_off = K_off + (uint64_t)k * DH;
        
        float score = 0.0f;
        for (int j = 0; j < DH; j += 4) {
            float kj0 = __bfloat162float(K[kD_off + j]);
            float kj1 = __bfloat162float(K[kD_off + j + 1]);
            float kj2 = __bfloat162float(K[kD_off + j + 2]);
            float kj3 = __bfloat162float(K[kD_off + j + 3]);
            score += q_f[j]*kj0 + q_f[j+1]*kj1 + q_f[j+2]*kj2 + q_f[j+3]*kj3;
        }
        score *= inv_sd;
        float P_qk = expf(score - lse_q);
        
        uint64_t dO_k = dO_off + (uint64_t)k * DH;
        uint64_t V_k = V_off + (uint64_t)k * DH;
        
        float dp_sum = 0.0f;
        for (int j = 0; j < DH; j += 4) {
            float doj0 = __bfloat162float(dO[dO_k + j]);
            float doj1 = __bfloat162float(dO[dO_k + j + 1]);
            float doj2 = __bfloat162float(dO[dO_k + j + 2]);
            float doj3 = __bfloat162float(dO[dO_k + j + 3]);
            float vj0 = __bfloat162float(V[V_k + j]);
            float vj1 = __bfloat162float(V[V_k + j + 1]);
            float vj2 = __bfloat162float(V[V_k + j + 2]);
            float vj3 = __bfloat162float(V[V_k + j + 3]);
            dp_sum += doj0*vj0 + doj1*vj1 + doj2*vj2 + doj3*vj3;
        }
        drow += P_qk * dp_sum;
        
        float ds = P_qk * (dp_sum - drow);
        
        uint64_t dk_base = slice_off + (uint64_t)k * DH;
        uint64_t dv_base = slice_off + (uint64_t)k * DH;
        
        for (int j = 0; j < DH; j += 4) {
            float kj0 = __bfloat162float(K[kD_off + j]);
            float kj1 = __bfloat162float(K[kD_off + j + 1]);
            float kj2 = __bfloat162float(K[kD_off + j + 2]);
            float kj3 = __bfloat162float(K[kD_off + j + 3]);
            
            dq_accum[j]   += ds * kj0;
            dq_accum[j+1] += ds * kj1;
            dq_accum[j+2] += ds * kj2;
            dq_accum[j+3] += ds * kj3;
            
            float qj0 = q_f[j], qj1 = q_f[j+1], qj2 = q_f[j+2], qj3 = q_f[j+3];
            atomicAdd(&dK_f[dk_base+j],   ds * qj0);
            atomicAdd(&dK_f[dk_base+j+1], ds * qj1);
            atomicAdd(&dK_f[dk_base+j+2], ds * qj2);
            atomicAdd(&dK_f[dk_base+j+3], ds * qj3);
            
            float doj0 = __bfloat162float(dO[dO_k + j]);
            float doj1 = __bfloat162float(dO[dO_k + j + 1]);
            float doj2 = __bfloat162float(dO[dO_k + j + 2]);
            float doj3 = __bfloat162float(dO[dO_k + j + 3]);
            
            atomicAdd(&dV_f[dv_base+j],   P_qk * doj0);
            atomicAdd(&dV_f[dv_base+j+1], P_qk * doj1);
            atomicAdd(&dV_f[dv_base+j+2], P_qk * doj2);
            atomicAdd(&dV_f[dv_base+j+3], P_qk * doj3);
        }
    }
    
    uint64_t dq_out = slice_off + (uint64_t)q * DH;
    for (int j = 0; j < DH; j++) {
        dQ_f[dq_out + j] = dq_accum[j];
    }
}

__global__ void convert_float_to_bf16_kernel(
    const float* src, __nv_bfloat16* dst, int nelem) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < nelem; i += stride) {
        dst[i] = __float2bfloat16(src[i]);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    int64_t BH = B * H;
    int64_t total = BH * S * D;
    
    const __nv_bfloat16* Q_p = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_p = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_p = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* dO_p = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_p = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_p = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_p = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_p = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    cudaStream_t stream = nullptr;
    cudaStreamGetDefault(&stream);
    
    int nelem = (int)total;
    float* dQ_f; float* dK_f; float* dV_f;
    CUDA_CHECK(cudaMallocAsync(&dQ_f, nelem * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&dK_f, nelem * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&dV_f, nelem * sizeof(float), stream));
    
    CUDA_CHECK(cudaMemsetAsync(dK_f, 0, nelem * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dV_f, 0, nelem * sizeof(float), stream));
    
    int bs = (int)S < 1024 ? (int)S : 1024;
    dim3 grid(1, (int)B, (int)H);
    dim3 block(bs);
    
    mha_bwd_kernel<1024><<<grid, block, 0, stream>>>(
        Q_p, K_p, V_p, dO_p, L_p, dQ_f, dK_f, dV_f, (int)S, (int)BH, (int)(S * D));
    CUDA_CHECK(cudaGetLastError());
    
    int conv_threads = 256;
    int conv_blocks = (nelem + conv_threads - 1) / conv_threads;
    
    convert_float_to_bf16_kernel<<<conv_blocks, conv_threads, 0, stream>>>(dQ_f, dQ_p, nelem);
    CUDA_CHECK(cudaGetLastError());
    convert_float_to_bf16_kernel<<<conv_blocks, conv_threads, 0, stream>>>(dK_f, dK_p, nelem);
    CUDA_CHECK(cudaGetLastError());
    convert_float_to_bf16_kernel<<<conv_blocks, conv_threads, 0, stream>>>(dV_f, dV_p, nelem);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaFreeAsync(dQ_f, stream));
    CUDA_CHECK(cudaFreeAsync(dK_f, stream));
    CUDA_CHECK(cudaFreeAsync(dV_f, stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);

} // namespace mha_bwd_impl