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
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    float* __restrict__ dQ_f,
    float* __restrict__ dK_f,
    float* __restrict__ dV_f,
    int S, int H) {
    
    int b = blockIdx.y;
    int h = blockIdx.z;
    int bh = b * H + h;
    uint64_t base = (uint64_t)bh * S * DH;
    
    float inv_sd = rsqrtf((float)DH);
    
    // Each thread handles one query position
    int q = threadIdx.x + blockIdx.x * blockDim.x;
    if (q >= S) return;
    
    uint64_t q_off = base + (uint64_t)q * DH;
    
    // Pre-load Q[q,:] into registers once
    float qreg[DH];
    for (int j = 0; j < DH; j++) {
        qreg[j] = __bfloat162float(Q[q_off + j]);
    }
    
    float lse_q = L[bh * S + q];
    
    // Compute D_q = sum_j O[q,j]*dO[q,j]
    float D_q = 0.0f;
    for (int j = 0; j < DH; j += 4) {
        float oj0  = __bfloat162float(O[q_off + j]);
        float oj1  = __bfloat162float(O[q_off + j+1]);
        float oj2  = __bfloat162float(O[q_off + j+2]);
        float oj3  = __bfloat162float(O[q_off + j+3]);
        float doj0 = __bfloat162float(dO[q_off + j]);
        float doj1 = __bfloat162float(dO[q_off + j+1]);
        float doj2 = __bfloat162float(dO[q_off + j+2]);
        float doj3 = __bfloat162float(dO[q_off + j+3]);
        D_q += oj0*doj0 + oj1*doj1 + oj2*doj2 + oj3*doj3;
    }
    
    float dq_acc[DH];
    for (int j = 0; j < DH; j++) dq_acc[j] = 0.0f;
    
    // Iterate over all k <= q (causal)
    for (int k = 0; k <= q; k++) {
        uint64_t k_off = base + (uint64_t)k * DH;
        
        // score = Q[q,:] . K[k,:]
        float score = 0.0f;
        for (int j = 0; j < DH; j += 4) {
            float kj0 = __bfloat162float(K[k_off + j]);
            float kj1 = __bfloat162float(K[k_off + j+1]);
            float kj2 = __bfloat162float(K[k_off + j+2]);
            float kj3 = __bfloat162float(K[k_off + j+3]);
            score += qreg[j]*kj0 + qreg[j+1]*kj1 + qreg[j+2]*kj2 + qreg[j+3]*kj3;
        }
        float P = expf(score * inv_sd - lse_q);
        
        // dp_sum = dO[q,:] . V[k,:]
        float dp_sum = 0.0f;
        for (int j = 0; j < DH; j += 4) {
            float doj0 = __bfloat162float(dO[q_off + j]);
            float doj1 = __bfloat162float(dO[q_off + j+1]);
            float doj2 = __bfloat162float(dO[q_off + j+2]);
            float doj3 = __bfloat162float(dO[q_off + j+3]);
            float vj0 = __bfloat162float(V[k_off + j]);
            float vj1 = __bfloat162float(V[k_off + j+1]);
            float vj2 = __bfloat162float(V[k_off + j+2]);
            float vj3 = __bfloat162float(V[k_off + j+3]);
            dp_sum += doj0*vj0 + doj1*vj1 + doj2*vj2 + doj3*vj3;
        }
        
        float ds = P * (dp_sum - D_q);
        
        // Accumulate gradients
        for (int j = 0; j < DH; j += 4) {
            float kj0 = __bfloat162float(K[k_off + j]);
            float kj1 = __bfloat162float(K[k_off + j+1]);
            float kj2 = __bfloat162float(K[k_off + j+2]);
            float kj3 = __bfloat162float(K[k_off + j+3]);
            
            dq_acc[j]   += ds * kj0;
            dq_acc[j+1] += ds * kj1;
            dq_acc[j+2] += ds * kj2;
            dq_acc[j+3] += ds * kj3;
            
            float qj0 = qreg[j], qj1 = qreg[j+1], qj2 = qreg[j+2], qj3 = qreg[j+3];
            float doj0 = __bfloat162float(dO[q_off + j]);
            float doj1 = __bfloat162float(dO[q_off + j+1]);
            float doj2 = __bfloat162float(dO[q_off + j+2]);
            float doj3 = __bfloat162float(dO[q_off + j+3]);
            
            atomicAdd(&dK_f[k_off+j],   ds * qj0);
            atomicAdd(&dK_f[k_off+j+1], ds * qj1);
            atomicAdd(&dK_f[k_off+j+2], ds * qj2);
            atomicAdd(&dK_f[k_off+j+3], ds * qj3);
            
            atomicAdd(&dV_f[k_off+j],   P * doj0);
            atomicAdd(&dV_f[k_off+j+1], P * doj1);
            atomicAdd(&dV_f[k_off+j+2], P * doj2);
            atomicAdd(&dV_f[k_off+j+3], P * doj3);
        }
    }
    
    // Write dQ output
    for (int j = 0; j < DH; j++) {
        dQ_f[q_off + j] = dq_acc[j];
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
    int64_t total = B * H * S * DH;

    const __nv_bfloat16* Q_p  = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_p  = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_p  = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_p  = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_p = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_p          = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_p       = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_p       = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_p       = static_cast<__nv_bfloat16*>(dV.data_ptr());

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    int nelem = (int)total;
    float* dQ_f; float* dK_f; float* dV_f;
    CUDA_CHECK(cudaMallocAsync(&dQ_f, nelem * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&dK_f, nelem * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&dV_f, nelem * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dK_f, 0, nelem * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dV_f, 0, nelem * sizeof(float), stream));

    int blk = 256;
    dim3 grid((S + blk - 1) / blk, (int)B, (int)H);
    dim3 block(blk);

    mha_bwd_kernel<<<grid, block, 0, stream>>>(
        Q_p, K_p, V_p, O_p, dO_p, L_p, dQ_f, dK_f, dV_f, (int)S, (int)H);
    CUDA_CHECK(cudaGetLastError());

    int ct = 256;
    int cb = (nelem + ct - 1) / ct;
    convert_float_to_bf16_kernel<<<cb, ct, 0, stream>>>(dQ_f, dQ_p, nelem);
    CUDA_CHECK(cudaGetLastError());
    convert_float_to_bf16_kernel<<<cb, ct, 0, stream>>>(dK_f, dK_p, nelem);
    CUDA_CHECK(cudaGetLastError());
    convert_float_to_bf16_kernel<<<cb, ct, 0, stream>>>(dV_f, dV_p, nelem);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFreeAsync(dQ_f, stream));
    CUDA_CHECK(cudaFreeAsync(dK_f, stream));
    CUDA_CHECK(cudaFreeAsync(dV_f, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);

} // namespace mha_bwd_impl