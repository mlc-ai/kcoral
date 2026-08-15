#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
#include <tvm/ffi/tvm_ffi.h>

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

__global__ void mha_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int S) {
    
    int b = blockIdx.y;
    int h = blockIdx.z;
    int bh = b * 48 + h;
    
    uint64_t slice_off = (uint64_t)bh * S * DH;
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
    
    float dk_accum[DH];
    for (int j = 0; j < DH; j++) dk_accum[j] = 0.0f;
    
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
            score += q_f[j] * kj0 + q_f[j+1] * kj1 + q_f[j+2] * kj2 + q_f[j+3] * kj3;
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
        
        for (int j = 0; j < DH; j += 4) {
            float kj0 = __bfloat162float(K[kD_off + j]);
            float kj1 = __bfloat162float(K[kD_off + j + 1]);
            float kj2 = __bfloat162float(K[kD_off + j + 2]);
            float kj3 = __bfloat162float(K[kD_off + j + 3]);
            dq_accum[j]   += ds * kj0;
            dq_accum[j+1] += ds * kj1;
            dq_accum[j+2] += ds * kj2;
            dq_accum[j+3] += ds * kj3;
            
            dk_accum[j]   += ds * q_f[j];
            dk_accum[j+1] += ds * q_f[j+1];
            dk_accum[j+2] += ds * q_f[j+2];
            dk_accum[j+3] += ds * q_f[j+3];
            
            float doj0 = __bfloat162float(dO[dO_k + j]);
            float doj1 = __bfloat162float(dO[dO_k + j + 1]);
            float doj2 = __bfloat162float(dO[dO_k + j + 2]);
            float doj3 = __bfloat162float(dO[dO_k + j + 3]);
            atomicAdd(&((float*)dV)[slice_off / sizeof(float) * 2 + (uint64_t)k * DH + j], P_qk * doj0);
            atomicAdd(&((float*)dV)[slice_off / sizeof(float) * 2 + (uint64_t)k * DH + j+1], P_qk * doj1);
            atomicAdd(&((float*)dV)[slice_off / sizeof(float) * 2 + (uint64_t)k * DH + j+2], P_qk * doj2);
            atomicAdd(&((float*)dV)[slice_off / sizeof(float) * 2 + (uint64_t)k * DH + j+3], P_qk * doj3);
        }
    }
    
    uint64_t dq_out = slice_off + (uint64_t)q * DH;
    for (int j = 0; j < DH; j++) {
        dQ[dq_out + j] = __float2bfloat16(dq_accum[j]);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int64_t B = Q.size(0), H = Q.size(1), S = Q.size(2);
    const __nv_bfloat16* Q_p = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_p = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_p = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* dO_p = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_p = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_p = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_p = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_p = static_cast<__nv_bfloat16*>(dV.data_ptr());
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    int bs = (int)S < 1024 ? (int)S : 1024;
    mha_bwd_kernel<<<dim3(1,(int)B,(int)H), dim3(bs), 0, stream>>(
        Q_p, K_p, V_p, dO_p, L_p, dQ_p, dK_p, dV_p, (int)S);
    CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);

}