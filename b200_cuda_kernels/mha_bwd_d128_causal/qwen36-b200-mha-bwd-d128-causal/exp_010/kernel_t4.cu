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
static constexpr int TSIZE = 4;

__global__ void mha_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    float* __restrict__ dQ_f,
    float* __restrict__ dK_f,
    float* __restrict__ dV_f,
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
    
    // Grid-stride over query positions
    int q = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Each (b,h,q) triplet processes all k <= q for causal mask
    if (q >= S) return;
    
    float lse_q = L[L_off + q];
    uint64_t qD_off = Q_off + (uint64_t)q * DH;
    
    float dq_acc[DH];
    for (int j = 0; j < DH; j++) dq_acc[j] = 0.0f;
    
    float qv[TSIZE];
    float kv[TSIZE];
    float dov[TSIZE];
    float vv[TSIZE];
    
    // PASS 1: compute D[q] = sum_{k<=q} P_qk * dp_sum_k
    float D_q = 0.0f;
    
    for (int k = 0; k <= q; k++) {
        uint64_t kD_off = K_off + (uint64_t)k * DH;
        
        float score = 0.0f;
        for (int j = 0; j < DH; j += TSIZE) {
            qv[0] = __bfloat162float(Q[qD_off + j]);
            qv[1] = __bfloat162float(Q[qD_off + j + 1]);
            qv[2] = __bfloat162float(Q[qD_off + j + 2]);
            qv[3] = __bfloat162float(Q[qD_off + j + 3]);
            
            kv[0] = __bfloat162float(K[kD_off + j]);
            kv[1] = __bfloat162float(K[kD_off + j + 1]);
            kv[2] = __bfloat162float(K[kD_off + j + 2]);
            kv[3] = __bfloat162float(K[kD_off + j + 3]);
            
            score += qv[0]*kv[0] + qv[1]*kv[1] + qv[2]*kv[2] + qv[3]*kv[3];
        }
        score *= inv_sd;
        float P_qk = expf(score - lse_q);
        
        uint64_t dOk = dO_off + (uint64_t)k * DH;
        uint64_t Vk = V_off + (uint64_t)k * DH;
        
        float dp_sum = 0.0f;
        for (int j = 0; j < DH; j += TSIZE) {
            dov[0] = __bfloat162float(dO[dOk + j]);
            dov[1] = __bfloat162float(dO[dOk + j + 1]);
            dov[2] = __bfloat162float(dO[dOk + j + 2]);
            dov[3] = __bfloat162float(dO[dOk + j + 3]);
            vv[0] = __bfloat162float(V[Vk + j]);
            vv[1] = __bfloat162float(V[Vk + j + 1]);
            vv[2] = __bfloat162float(V[Vk + j + 2]);
            vv[3] = __bfloat162float(V[Vk + j + 3]);
            dp_sum += dov[0]*vv[0] + dov[1]*vv[1] + dov[2]*vv[2] + dov[3]*vv[3];
        }
        D_q += P_qk * dp_sum;
    }
    
    // PASS 2: compute gradients with complete D_q
    for (int k = 0; k <= q; k++) {
        uint64_t kD_off = K_off + (uint64_t)k * DH;
        
        float score = 0.0f;
        for (int j = 0; j < DH; j += TSIZE) {
            qv[0] = __bfloat162float(Q[qD_off + j]);
            qv[1] = __bfloat162float(Q[qD_off + j + 1]);
            qv[2] = __bfloat162float(Q[qD_off + j + 2]);
            qv[3] = __bfloat162float(Q[qD_off + j + 3]);
            kv[0] = __bfloat162float(K[kD_off + j]);
            kv[1] = __bfloat162float(K[kD_off + j + 1]);
            kv[2] = __bfloat162float(K[kD_off + j + 2]);
            kv[3] = __bfloat162float(K[kD_off + j + 3]);
            score += qv[0]*kv[0] + qv[1]*kv[1] + qv[2]*kv[2] + qv[3]*kv[3];
        }
        score *= inv_sd;
        float P_qk = expf(score - lse_q);
        
        uint64_t dOk = dO_off + (uint64_t)k * DH;
        uint64_t Vk = V_off + (uint64_t)k * DH;
        
        float dp_sum = 0.0f;
        for (int j = 0; j < DH; j += TSIZE) {
            dov[0] = __bfloat162float(dO[dOk + j]);
            dov[1] = __bfloat162float(dO[dOk + j + 1]);
            dov[2] = __bfloat162float(dO[dOk + j + 2]);
            dov[3] = __bfloat162float(dO[dOk + j + 3]);
            vv[0] = __bfloat162float(V[Vk + j]);
            vv[1] = __bfloat162float(V[Vk + j + 1]);
            vv[2] = __bfloat162float(V[Vk + j + 2]);
            vv[3] = __bfloat162float(V[Vk + j + 3]);
            dp_sum += dov[0]*vv[0] + dov[1]*vv[1] + dov[2]*vv[2] + dov[3]*vv[3];
        }
        
        float ds_factor = P_qk * (dp_sum - D_q);
        
        // Accumulate into outputs
        uint64_t dk_base = slice_off + (uint64_t)k * DH;
        uint64_t dv_base = slice_off + (uint64_t)k * DH;
        
        for (int j = 0; j < DH; j += TSIZE) {
            qv[0] = __bfloat162float(Q[qD_off + j]);
            qv[1] = __bfloat162float(Q[qD_off + j + 1]);
            qv[2] = __bfloat162float(Q[qD_off + j + 2]);
            qv[3] = __bfloat162float(Q[qD_off + j + 3]);
            
            kv[0] = __bfloat162float(K[kD_off + j]);
            kv[1] = __bfloat162float(K[kD_off + j + 1]);
            kv[2] = __bfloat162float(K[kD_off + j + 2]);
            kv[3] = __bfloat162float(K[kD_off + j + 3]);
            
            dq_acc[j]   += ds_factor * kv[0];
            dq_acc[j+1] += ds_factor * kv[1];
            dq_acc[j+2] += ds_factor * kv[2];
            dq_acc[j+3] += ds_factor * kv[3];
            
            atomicAdd(&dK_f[dk_base+j],   ds_factor * qv[0]);
            atomicAdd(&dK_f[dk_base+j+1], ds_factor * qv[1]);
            atomicAdd(&dK_f[dk_base+j+2], ds_factor * qv[2]);
            atomicAdd(&dK_f[dk_base+j+3], ds_factor * qv[3]);
            
            dov[0] = __bfloat162float(dO[dOk + j]);
            dov[1] = __bfloat162float(dO[dOk + j + 1]);
            dov[2] = __bfloat162float(dO[dOk + j + 2]);
            dov[3] = __bfloat162float(dO[dOk + j + 3]);
            
            atomicAdd(&dV_f[dv_base+j],   P_qk * dov[0]);
            atomicAdd(&dV_f[dv_base+j+1], P_qk * dov[1]);
            atomicAdd(&dV_f[dv_base+j+2], P_qk * dov[2]);
            atomicAdd(&dV_f[dv_base+j+3], P_qk * dov[3]);
        }
    }
    
    // Write dQ result
    uint64_t dq_out = slice_off + (uint64_t)q * DH;
    for (int j = 0; j < DH; j++) {
        dQ_f[dq_out + j] = dq_acc[j];
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
    int64_t total = B * H * S * D;
    
    const __nv_bfloat16* Q_p = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_p = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_p = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* dO_p = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_p = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_p = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_p = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_p = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    
    int nelem = (int)total;
    float* dQ_f; float* dK_f; float* dV_f;
    CUDA_CHECK(cudaMallocAsync(&dQ_f, nelem * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&dK_f, nelem * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&dV_f, nelem * sizeof(float), stream));
    
    CUDA_CHECK(cudaMemsetAsync(dK_f, 0, nelem * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dV_f, 0, nelem * sizeof(float), stream));
    
    // Grid layout: x=S (queries), y=B, z=H
    dim3 block(256);
    dim3 grid((S + block.x - 1) / block.x, (int)B, (int)H);
    
    mha_bwd_kernel<<<grid, block, 0, stream>>>(
        Q_p, K_p, V_p, dO_p, L_p, dQ_f, dK_f, dV_f, (int)S);
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
    
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);

} // namespace mha_bwd_impl