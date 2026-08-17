#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_bwd_impl {

__device__ __forceinline__ static float bf16tof(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ static __nv_bfloat16 f2bf16(float x) {
    return __float2bfloat16(x);
}

/**
 * MHA Backward Kernel (Causal Mask, BF16 inputs/outputs, FP32 intermediates)
 * 
 * Each block handles one (batch, head) combination. Within each block:
 * - Stage 1: Compute D[i] = sum_{j<=i} P[i,j] * (dO[i,:] . V[j,:])
 * - Stage 2: Compute dV[j,:] = sum_{i>=j} P[i,j] * dO[i,:]
 * - Stage 3: Compute dQ[i,:] = sum_{j<=i} dS[i,j] * K[j,:]
 * - Stage 4: Compute dK[j,:] = sum_{i>=j} dS[i,j] * Q[i,:]
 * where dS[i,j] = P[i,j] * (delta[i,j] - D[i])
 */
__global__ void mha_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const float* __restrict__ L,
    const __nv_bfloat16* __restrict__ dO,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int S, int d, int num_heads, float inv_sqrt_d,
    float* __restrict__ D_buf)
{
    int bh = blockIdx.x;
    if (bh >= (int)gridDim.x || S <= 0 || d <= 0) return;

    int b = bh / num_heads;
    int h = bh % num_heads;
    int tid = threadIdx.x;
    int nt = blockDim.x;

    // Linear index of this (batch, head) tile
    uint64_t off = (uint64_t)b * num_heads + h;

    const __nv_bfloat16* qb = Q + off * S * d;
    const __nv_bfloat16* kb = K + off * S * d;
    const __nv_bfloat16* vb = V + off * S * d;
    const float* lb = L + off * S;
    const __nv_bfloat16* dob = dO + off * S * d;
    __nv_bfloat16* dqb = dQ + off * S * d;
    __nv_bfloat16* dkb = dK + off * S * d;
    __nv_bfloat16* dvb = dV + off * S * d;
    float* db = D_buf + off * S;

    // Zero-initialize outputs
    for (int idx = tid; idx < S * d; idx += nt) {
        dqb[idx] = f2bf16(0.f);
        dkb[idx] = f2bf16(0.f);
        dvb[idx] = f2bf16(0.f);
    }
    __syncthreads();

    // =====================================================================
    // Stage 1: Compute D[i] for all query positions
    // D[i] = sum_{j=0}^{i} P[i,j] * (dO[i,:] . V[j,:])
    // Each thread computes one or more D[i] values independently.
    // =====================================================================
    for (int i = tid; i < S; i += nt) {
        float Di = 0.0f;
        float Li = lb[i];
        for (int j = 0; j <= i; j++) {
            float sim = 0.0f, delta = 0.0f;
            for (int f = 0; f < d; f++) {
                sim   += bf16tof(qb[i * d + f]) * bf16tof(kb[j * d + f]);
                delta += bf16tof(dob[i * d + f]) * bf16tof(vb[j * d + f]);
            }
            float Pij = expf(sim * inv_sqrt_d - Li);
            Di += Pij * delta;
        }
        db[i] = Di;
    }
    __syncthreads();

    // =====================================================================
    // Stage 2: Compute dV[j,:] = sum_{i=j}^{S-1} P[i,j] * dO[i,:]
    // Each thread handles one or more key positions j independently.
    // =====================================================================
    for (int j = tid; j < S; j += nt) {
        for (int f = 0; f < d; f++) {
            float acc = 0.0f;
            for (int i = j; i < S; i++) {
                float sim = 0.0f;
                for (int ff = 0; ff < d; ff++) {
                    sim += bf16tof(qb[i * d + ff]) * bf16tof(kb[j * d + ff]);
                }
                float Pij = expf(sim * inv_sqrt_d - lb[i]);
                acc += Pij * bf16tof(dob[i * d + f]);
            }
            dvb[j * d + f] = f2bf16(acc);
        }
    }
    // No __syncthreads() needed: each (j,f) location is written by exactly one thread

    // =====================================================================
    // Stage 3: Compute dQ[i,:] = sum_{j=0}^{i} dS[i,j] * K[j,:]
    // where dS[i,j] = P[i,j] * (delta[i,j] - D[i])
    // Each thread handles one or more query positions i independently.
    // =====================================================================
    for (int i = tid; i < S; i += nt) {
        float Di = db[i];
        float Li = lb[i];
        for (int f = 0; f < d; f++) {
            float acc = 0.0f;
            for (int j = 0; j <= i; j++) {
                float sim = 0.0f, delta = 0.0f;
                for (int ff = 0; ff < d; ff++) {
                    sim   += bf16tof(qb[i * d + ff]) * bf16tof(kb[j * d + ff]);
                    delta += bf16tof(dob[i * d + ff]) * bf16tof(vb[j * d + ff]);
                }
                float Pij  = expf(sim * inv_sqrt_d - Li);
                float dSij = Pij * (delta - Di);
                acc += dSij * bf16tof(kb[j * d + f]);
            }
            dqb[i * d + f] = f2bf16(acc);
        }
    }

    // =====================================================================
    // Stage 4: Compute dK[j,:] = sum_{i=j}^{S-1} dS[i,j] * Q[i,:]
    // where dS[i,j] = P[i,j] * (delta[i,j] - D[i])
    // Each thread handles one or more key positions j independently.
    // =====================================================================
    for (int j = tid; j < S; j += nt) {
        for (int f = 0; f < d; f++) {
            float acc = 0.0f;
            for (int i = j; i < S; i++) {
                float sim = 0.0f, delta = 0.0f;
                for (int ff = 0; ff < d; ff++) {
                    sim   += bf16tof(qb[i * d + ff]) * bf16tof(kb[j * d + ff]);
                    delta += bf16tof(dob[i * d + ff]) * bf16tof(vb[j * d + ff]);
                }
                float Pij  = expf(sim * inv_sqrt_d - lb[i]);
                float dSij = Pij * (delta - db[i]);
                acc += dSij * bf16tof(qb[i * d + f]);
            }
            dkb[j * d + f] = f2bf16(acc);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {

    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t ddim = Q.size(3);

    const __nv_bfloat16* ptr_Q  = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* ptr_K  = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* ptr_V  = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const float* ptr_L          = static_cast<const float*>(L.data_ptr());
    const __nv_bfloat16* ptr_dO = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    __nv_bfloat16* ptr_dQ       = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* ptr_dK       = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* ptr_dV       = static_cast<__nv_bfloat16*>(dV.data_ptr());

    // Allocate temporary D buffer [B, H, S] for softmax correction terms
    float* d_D = nullptr;
    if (B > 0 && H > 0 && S > 0 && ddim > 0) {
        size_t d_buf_bytes = B * H * S * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_D, d_buf_bytes));
    }

    int64_t total_bh = B * H;
    int threads = 128;
    int blocks = (int)total_bh;
    float inv_sqrt_d = 1.0f / sqrtf((float)ddim);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    if (total_bh > 0 && S > 0 && ddim > 0) {
        mha_bwd_kernel<<<blocks, threads, 0, stream>>>(
            ptr_Q, ptr_K, ptr_V, ptr_L, ptr_dO,
            ptr_dQ, ptr_dK, ptr_dV,
            (int)S, (int)ddim, (int)H, inv_sqrt_d, d_D
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    if (d_D) {
        CUDA_CHECK(cudaFree(d_D));
    }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);

} // namespace mha_bwd_impl