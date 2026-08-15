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

__device__ __forceinline__ void load_bf16_row(const __nv_bfloat16* src, float* dst, int n) {
    for (int j = 0; j < n; j += 4) {
        dst[j]   = __bfloat162float(src[j]);
        dst[j+1] = __bfloat162float(src[j+1]);
        dst[j+2] = __bfloat162float(src[j+2]);
        dst[j+3] = __bfloat162float(src[j+3]);
    }
}

__device__ __forceinline__ float dot4(const float* a, const float* b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
}

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
    
    int tid = threadIdx.x;
    if (tid >= S) return;
    int q = tid;
    
    float lse_q = L[L_off + q];
    uint64_t qD_off = Q_off + (uint64_t)q * DH;
    
    // Load Q[q,:] into registers once
    float q_f[DH];
    load_bf16_row(&Q[qD_off], q_f, DH);
    
    float dq_accum[DH];
    for (int j = 0; j < DH; j++) dq_accum[j] = 0.0f;
    
    // PASS 1: Compute drow = sum_{k<=q} P_qk * dp_sum_k
    float drow = 0.0f;
    for (int k = 0; k <= q; k++) {
        uint64_t kD_off = K_off + (uint64_t)k * DH;
        
        // QK score
        float score = 0.0f;
        for (int j = 0; j < DH; j += 4) {
            float kj[4];
            load_bf16_row(&K[kD_off + j], kj, 4);
            score += dot4(&q_f[j], kj);
        }
        score *= inv_sd;
        float P_qk = expf(score - lse_q);
        
        // dp_sum = sum_j(dO[k,j] * V[k,j])
        uint64_t dO_k_off = dO_off + (uint64_t)k * DH;
        uint64_t V_k_off = V_off + (uint64_t)k * DH;
        float dp_sum = 0.0f;
        for (int j = 0; j < DH; j += 4) {
            float doj[4], vj[4];
            load_bf16_row(&dO[dO_k_off + j], doj, 4);
            load_bf16_row(&V[V_k_off + j], vj, 4);
            dp_sum += dot4(doj, vj);
        }
        
        drow += P_qk * dp_sum;
    }
    
    // PASS 2: Compute gradients using complete drow
    for (int k = 0; k <= q; k++) {
        uint64_t kD_off = K_off + (uint64_t)k * DH;
        
        // Recompute score and P_qk
        float score = 0.0f;
        for (int j = 0; j < DH; j += 4) {
            float kj[4];
            load_bf16_row(&K[kD_off + j], kj, 4);
            score += dot4(&q_f[j], kj);
        }
        score *= inv_sd;
        float P_qk = expf(score - lse_q);
        
        // Recompute dp_sum and gather dO/V rows
        uint64_t dO_k_off = dO_off + (uint64_t)k * DH;
        uint64_t V_k_off = V_off + (uint64_t)k * DH;
        float dp_sum = 0.0f;
        float dO_k_f[DH];
        float V_k_f[DH];
        for (int j = 0; j < DH; j += 4) {
            load_bf16_row(&dO[dO_k_off + j], &dO_k_f[j], 4);
            load_bf16_row(&V[V_k_off + j], &V_k_f[j], 4);
            dp_sum += dot4(&dO_k_f[j], &V_k_f[j]);
        }
        
        float ds_factor = P_qk * (dp_sum - drow);
        
        // Accumulate dQ, dK, dV
        uint64_t dk_base = slice_off + (uint64_t)k * DH;
        uint64_t dv_base = slice_off + (uint64_t)k * DH;
        
        for (int j = 0; j < DH; j += 4) {
            float kj[4];
            load_bf16_row(&K[kD_off + j], kj, 4);
            
            // dQ += ds * K
            dq_accum[j]   += ds_factor * kj[0];
            dq_accum[j+1] += ds_factor * kj[1];
            dq_accum[j+2] += ds_factor * kj[2];
            dq_accum[j+3] += ds_factor * kj[3];
            
            // dK += ds * Q  (using registered q_f)
            atomicAdd(&dK_f[dk_base+j],   ds_factor * q_f[j]);
            atomicAdd(&dK_f[dk_base+j+1], ds_factor * q_f[j+1]);
            atomicAdd(&dK_f[dk_base+j+2], ds_factor * q_f[j+2]);
            atomicAdd(&dK_f[dk_base+j+3], ds_factor * q_f[j+3]);
            
            // dV += P * dO
            atomicAdd(&dV_f[dv_base+j],   P_qk * dO_k_f[j]);
            atomicAdd(&dV_f[dv_base+j+1], P_qk * dO_k_f[j+1]);
            atomicAdd(&dV_f[dv_base+j+2], P_qk * dO_k_f[j+2]);
            atomicAdd(&dV_f[dv_base+j+3], P_qk * dO_k_f[j+3]);
        }
    }
    
    // Write dQ result
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
    
    int bs = (int)S < 1024 ? (int)S : 1024;
    dim3 grid(1, (int)B, (int)H);
    dim3 block(bs);
    
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