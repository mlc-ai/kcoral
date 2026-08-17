#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdio.h>
#include <cstdint>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_mha_bwd {

constexpr int BLOCK_DIM_X = 128;

__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float v) {
    return __float2bfloat16(v);
}

// Copy FP32 staging -> BF16 output buffer
template<int D_DIM>
__global__ void copy_fp32_to_bf16(
    const float* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int total_bh, int S) {
    int elem = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_bh * S * D_DIM;
    for (int i = elem; i < total; i += blockDim.x * gridDim.x) {
        dst[i] = float_to_bf16(src[i]);
    }
}

template<int D_DIM>
__global__ void mha_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L_in,
    float* __restrict__ dQ_f32,
    float* __restrict__ dK_f32,
    float* __restrict__ dV_f32,
    int B, int H, int S) {
    
    static_assert(D_DIM == 128);
    
    int bh = blockIdx.x;
    if (bh >= B * H) return;
    
    int b = bh / H, h = bh % H;
    size_t base = (size_t)(b * H + h) * S * D_DIM;
    float inv_d = rsqrtf((float)D_DIM);
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    
    const __nv_bfloat16* Qg  = Q + base;
    const __nv_bfloat16* Kg  = K + base;
    const __nv_bfloat16* Vg  = V + base;
    const __nv_bfloat16* dOg = dO + base;
    const float* Lg          = L_in + b * H * S + h * S;
    
    float* dQ_f = dQ_f32 + bh * S * D_DIM;
    float* dK_f = dK_f32 + bh * S * D_DIM;
    float* dV_f = dV_f32 + bh * S * D_DIM;
    
    // Zero staging outputs
    for (int i = tid; i < S * D_DIM; i += nthreads) {
        dQ_f[i] = 0.f;
        dK_f[i] = 0.f;
        dV_f[i] = 0.f;
    }
    __syncthreads();
    
    // Each thread handles assigned qi indices
    for (int qi_start = tid; qi_start < S; qi_start += nthreads) {
        int qi = qi_start;
        float Li = Lg[qi];
        
        // Load Q[qi] and dO[qi] rows into registers
        float Qr[D_DIM];
        float dOr[D_DIM];
        for (int d = 0; d < D_DIM; d += 2) {
            __nv_bfloat162 qp = *reinterpret_cast<const __nv_bfloat162*>(&Qg[(size_t)qi * D_DIM + d]);
            __nv_bfloat162 dop = *reinterpret_cast<const __nv_bfloat162*>(&dOg[(size_t)qi * D_DIM + d]);
            Qr[d]     = bf16_to_float(qp.x);   Qr[d+1] = bf16_to_float(qp.y);
            dOr[d]    = bf16_to_float(dop.x);  dOr[d+1] = bf16_to_float(dop.y);
        }
        
        // ---- PASS 1: compute Dqi = sum_{kj<=qi} P[qi][kj]*dPV[qi][kj] ----
        float Dqi = 0.f;
        for (int kj = 0; kj <= qi; ++kj) {
            float sq = 0.f, sdv = 0.f;
            for (int d = 0; d < D_DIM; d += 2) {
                size_t off = (size_t)kj * D_DIM + d;
                __nv_bfloat162 kp = *reinterpret_cast<const __nv_bfloat162*>(&Kg[off]);
                __nv_bfloat162 vp = *reinterpret_cast<const __nv_bfloat162*>(&Vg[off]);
                float kx = bf16_to_float(kp.x), ky = bf16_to_float(kp.y);
                float vx = bf16_to_float(vp.x), vy = bf16_to_float(vp.y);
                sq  += fmaf(kx, Qr[d],   fmaf(ky, Qr[d+1], 0.f));
                sdv += fmaf(vx, dOr[d],  fmaf(vy, dOr[d+1], 0.f));
            }
            sq *= inv_d;
            Dqi += expf(sq - Li) * sdv;
        }
        
        // ---- PASS 2: accumulate gradients into FP32 staging ----
        float dq_reg[D_DIM];
        for (int d = 0; d < D_DIM; d++) dq_reg[d] = 0.f;
        
        for (int kj = 0; kj <= qi; ++kj) {
            size_t off_kj = (size_t)kj * D_DIM;
            float sq = 0.f, sdv = 0.f;
            float Kr[4], Vr[4];
            // Vectorized inner loop
            for (int d = 0; d < D_DIM; d += 4) {
                __nv_bfloat162 kp = *reinterpret_cast<const __nv_bfloat162*>(&Kg[off_kj + d]);
                __nv_bfloat162 vp = *reinterpret_cast<const __nv_bfloat162*>(&Vg[off_kj + d]);
                __nv_bfloat162 kp2 = *reinterpret_cast<const __nv_bfloat162*>(&Kg[off_kj + d + 2]);
                __nv_bfloat162 vp2 = *reinterpret_cast<const __nv_bfloat162*>(&Vg[off_kj + d + 2]);
                
                float k0 = bf16_to_float(kp.x), k1 = bf16_to_float(kp.y);
                float k2 = bf16_to_float(kp2.x), k3 = bf16_to_float(kp2.y);
                float v0 = bf16_to_float(vp.x), v1 = bf16_to_float(vp.y);
                float v2 = bf16_to_float(vp2.x), v3 = bf16_to_float(vp2.y);
                
                sq  += fmaf(k0, Qr[d],   fmaf(k1, Qr[d+1], 
                       fmaf(k2, Qr[d+2], fmaf(k3, Qr[d+3], 0.f))));
                sdv += fmaf(v0, dOr[d],  fmaf(v1, dOr[d+1], 
                       fmaf(v2, dOr[d+2], fmaf(v3, dOr[d+3], 0.f))));
            }
            sq *= inv_d;
            float p = expf(sq - Li);
            float diff = sdv - Dqi;
            
            // Accumulate dQ locally (unique per qi), dK/dV via atomics
            for (int d = 0; d < D_DIM; d += 4) {
                __nv_bfloat162 kp = *reinterpret_cast<const __nv_bfloat162*>(&Kg[off_kj + d]);
                __nv_bfloat162 kp2 = *reinterpret_cast<const __nv_bfloat162*>(&Kg[off_kj + d + 2]);
                dq_reg[d]     += p * bf16_to_float(kp.x)   * diff;
                dq_reg[d+1]   += p * bf16_to_float(kp.y)   * diff;
                dq_reg[d+2]   += p * bf16_to_float(kp2.x)  * diff;
                dq_reg[d+3]   += p * bf16_to_float(kp2.y)  * diff;
                
                atomicAdd(&dK_f[off_kj + d],     p * Qr[d] * diff);
                atomicAdd(&dK_f[off_kj + d + 1], p * Qr[d+1] * diff);
                atomicAdd(&dK_f[off_kj + d + 2], p * Qr[d+2] * diff);
                atomicAdd(&dK_f[off_kj + d + 3], p * Qr[d+3] * diff);
                
                atomicAdd(&dV_f[off_kj + d],     p * dOr[d]);
                atomicAdd(&dV_f[off_kj + d + 1], p * dOr[d+1]);
                atomicAdd(&dV_f[off_kj + d + 2], p * dOr[d+2]);
                atomicAdd(&dV_f[off_kj + d + 3], p * dOr[d+3]);
            }
        }
        
        // Store dQ (unique to this thread for this qi, no race condition!)
        for (int d = 0; d < D_DIM; d++) {
            dQ_f[(size_t)qi * D_DIM + d] = dq_reg[d];
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0), H = Q.size(1), S = Q.size(2), d = Q.size(3);
    int total_bh = static_cast<int>(B * H);
    
    dim3 grid(total_bh);
    dim3 block(BLOCK_DIM_X);
    
    // Allocate FP32 staging buffers (4-byte aligned guarantees clean atomics)
    float* d_dQ = nullptr, *d_dK = nullptr, *d_dV = nullptr;
    size_t buf_size = total_bh * S * d * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_dQ, buf_size));
    CUDA_CHECK(cudaMalloc(&d_dK, buf_size));
    CUDA_CHECK(cudaMalloc(&d_dV, buf_size));
    CUDA_CHECK(cudaMemset(d_dQ, 0, buf_size));
    CUDA_CHECK(cudaMemset(d_dK, 0, buf_size));
    CUDA_CHECK(cudaMemset(d_dV, 0, buf_size));
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    mha_bwd_kernel<128><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        d_dQ, d_dK, d_dV,
        static_cast<int>(B), static_cast<int>(H), static_cast<int>(S));
    
    CUDA_CHECK(cudaGetLastError());
    
    // Convert FP32 staging -> BF16 output
    dim3 grid_copy((total_bh * S * d + 255) / 256);
    dim3 block_copy(256);
    copy_fp32_to_bf16<128><<<grid_copy, block_copy, 0, stream>>>(d_dQ, static_cast<__nv_bfloat16*>(dQ.data_ptr()), total_bh, S);
    copy_fp32_to_bf16<128><<<grid_copy, block_copy, 0, stream>>>(d_dK, static_cast<__nv_bfloat16*>(dK.data_ptr()), total_bh, S);
    copy_fp32_to_bf16<128><<<grid_copy, block_copy, 0, stream>>>(d_dV, static_cast<__nv_bfloat16*>(dV.data_ptr()), total_bh, S);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    CUDA_CHECK(cudaFree(d_dQ));
    CUDA_CHECK(cudaFree(d_dK));
    CUDA_CHECK(cudaFree(d_dV));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

} // namespace tvm_ffi_mha_bwd