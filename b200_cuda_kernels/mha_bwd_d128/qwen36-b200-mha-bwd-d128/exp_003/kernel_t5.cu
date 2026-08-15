#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
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

namespace tvm_ffi_mha_bwd_d128 {

__device__ __forceinline__ float bf16_to_f32_nv(__nv_bfloat16 h) {
    return static_cast<float>(h);
}

// ==================== Kernel 1: Compute attention probs P ====================
// P[i,j] = exp(Q[i,:] · K[j,:] / sqrt(d) - L[i])

__global__ void compute_P_kernel(
    const __nv_bfloat16* Q_bh,
    const __nv_bfloat16* K_bh,
    const float* L_bh,
    float* P_bh,
    int S, int d) {
    
    int i_row = blockIdx.x;
    int lane = threadIdx.x;
    
    if (i_row >= S) return;
    
    extern __shared__ char smem_raw[];
    float* smem_Q = reinterpret_cast<float*>(smem_raw);
    
    for (int k = lane; k < d; k += blockDim.x) {
        smem_Q[k] = bf16_to_f32_nv(Q_bh[i_row * d + k]);
    }
    __syncthreads();
    
    float lsi = L_bh[i_row];
    float inv_sd = rsqrtf((float)d);
    
    int j = lane;
    if (j < S) {
        float dot = 0.0f;
        for (int k = 0; k < d; k += 4) {
            dot += smem_Q[k]     * bf16_to_f32_nv(K_bh[j * d + k]);
            dot += smem_Q[k + 1] * bf16_to_f32_nv(K_bh[j * d + k + 1]);
            dot += smem_Q[k + 2] * bf16_to_f32_nv(K_bh[j * d + k + 2]);
            dot += smem_Q[k + 3] * bf16_to_f32_nv(K_bh[j * d + k + 3]);
        }
        P_bh[i_row * S + j] = expf(dot * inv_sd - lsi);
    }
}

// ==================== Tiled FP32 GEMM: C[M,N] = A[M,K] @ B[K,N] ====================
template<int BM, int BN, int BK>
__global__ void gemm_rr_FP32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {
    
    constexpr int THREADS = 256;
    constexpr int EPT = (BM * BN) / THREADS;
    
    int tm = blockIdx.y * BM;
    int tn = blockIdx.x * BN;
    int tid = threadIdx.x;
    
    extern __shared__ float smem[];
    float* sa = smem;
    float* sb = smem + BK * BM;
    
    float acc[EPT];
    for (int e = 0; e < EPT; ++e) acc[e] = 0.0f;
    
    int nk = (K + BK - 1) / BK;
    for (int t = 0; t < nk; ++t) {
        int ks = t * BK;
        
        for (int idx = tid; idx < BK * BM; idx += THREADS) {
            int r = idx / BM, c = idx % BM;
            sa[idx] = (tm+c < M && ks+r < K) ? A[(tm+c)*K + ks+r] : 0.0f;
        }
        __syncthreads();
        
        for (int idx = tid; idx < BK * BN; idx += THREADS) {
            int r = idx / BN, c = idx % BN;
            sb[idx] = (ks+r < K && tn+c < N) ? B[(ks+r)*N + tn+c] : 0.0f;
        }
        __syncthreads();
        
        for (int e = 0; e < EPT; ++e) {
            int row = ((tid * EPT + e) / BN);
            int col = ((tid * EPT + e) % BN);
            #pragma unroll
            for (int k = 0; k < BK; ++k) {
                acc[e] += sa[k * BM + row] * sb[k * BN + col];
            }
        }
        __syncthreads();
    }
    
    for (int e = 0; e < EPT; ++e) {
        int idx = tid * EPT + e;
        int row = idx / BN, col = idx % BN;
        if (row < BM && col < BN && tm+row < M && tn+col < N)
            C[(tm+row)*N + tn+col] = acc[e];
    }
}

// GEMM transA: C[M,N] = A[K,M]^T @ B[K,N]
template<int BM, int BN, int BK>
__global__ void gemm_tA_FP32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {
    
    constexpr int THREADS = 256;
    constexpr int EPT = (BM * BN) / THREADS;
    
    int tm = blockIdx.y * BM;
    int tn = blockIdx.x * BN;
    int tid = threadIdx.x;
    
    extern __shared__ float smem[];
    float* sa = smem;
    float* sb = smem + BK * BM;
    
    float acc[EPT];
    for (int e = 0; e < EPT; ++e) acc[e] = 0.0f;
    
    int nk = (K + BK - 1) / BK;
    for (int t = 0; t < nk; ++t) {
        int ks = t * BK;
        
        for (int idx = tid; idx < BK * BM; idx += THREADS) {
            int r = idx / BM, c = idx % BM;
            sa[idx] = (ks+r < K && tm+c < M) ? A[(ks+r)*M + tm+c] : 0.0f;
        }
        __syncthreads();
        
        for (int idx = tid; idx < BK * BN; idx += THREADS) {
            int r = idx / BN, c = idx % BN;
            sb[idx] = (ks+r < K && tn+c < N) ? B[(ks+r)*N + tn+c] : 0.0f;
        }
        __syncthreads();
        
        for (int e = 0; e < EPT; ++e) {
            int row = ((tid * EPT + e) / BN);
            int col = ((tid * EPT + e) % BN);
            #pragma unroll
            for (int k = 0; k < BK; ++k) {
                acc[e] += sa[k * BM + row] * sb[k * BN + col];
            }
        }
        __syncthreads();
    }
    
    for (int e = 0; e < EPT; ++e) {
        int idx = tid * EPT + e;
        int row = idx / BN, col = idx % BN;
        if (row < BM && col < BN && tm+row < M && tn+col < N)
            C[(tm+row)*N + tn+col] = acc[e];
    }
}

// GEMM transB: C[M,N] = A[M,K] @ B[N,K]^T
template<int BM, int BN, int BK>
__global__ void gemm_tB_FP32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {
    
    constexpr int THREADS = 256;
    constexpr int EPT = (BM * BN) / THREADS;
    
    int tm = blockIdx.y * BM;
    int tn = blockIdx.x * BN;
    int tid = threadIdx.x;
    
    extern __shared__ float smem[];
    float* sa = smem;
    float* sb = smem + BK * BM;
    
    float acc[EPT];
    for (int e = 0; e < EPT; ++e) acc[e] = 0.0f;
    
    int nk = (K + BK - 1) / BK;
    for (int t = 0; t < nk; ++t) {
        int ks = t * BK;
        
        for (int idx = tid; idx < BK * BM; idx += THREADS) {
            int r = idx / BM, c = idx % BM;
            sa[idx] = (tm+c < M && ks+r < K) ? A[(tm+c)*K + ks+r] : 0.0f;
        }
        __syncthreads();
        
        for (int idx = tid; idx < BK * BN; idx += THREADS) {
            int r = idx / BN, c = idx % BN;
            sb[idx] = (tn+c < N && ks+r < K) ? B[(tn+c)*K + ks+r] : 0.0f;
        }
        __syncthreads();
        
        for (int e = 0; e < EPT; ++e) {
            int row = ((tid * EPT + e) / BN);
            int col = ((tid * EPT + e) % BN);
            #pragma unroll
            for (int k = 0; k < BK; ++k) {
                acc[e] += sa[k * BM + row] * sb[k * BN + col];
            }
        }
        __syncthreads();
    }
    
    for (int e = 0; e < EPT; ++e) {
        int idx = tid * EPT + e;
        int row = idx / BN, col = idx % BN;
        if (row < BM && col < BN && tm+row < M && tn+col < N)
            C[(tm+row)*N + tn+col] = acc[e];
    }
}

// Softmax backward: dS[i,j] = P[i,j] * (dP[i,j] - c[i])
__global__ void softmax_bw_kernel(const float* dP, const float* P, float* dS, int S) {
    int i = blockIdx.x;
    if (i >= S) return;
    
    int lane = threadIdx.x;
    extern __shared__ float smem[];
    
    float val = 0.0f;
    for (int j = lane; j < S; j += blockDim.x)
        val += P[i * S + j] * dP[i * S + j];
    smem[lane] = val;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) smem[lane] += smem[lane + stride];
        __syncthreads();
    }
    
    float ci = smem[0];
    for (int j = lane; j < S; j += blockDim.x)
        dS[i * S + j] = P[i * S + j] * (dP[i * S + j] - ci);
}

// Scale kernel for dQ and dK
__global__ void scale_and_convert(float* src, __nv_bfloat16* dst, int count, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = __float2bfloat16(src[i] * scale);
}

__global__ void bf16_to_f32_convert(const __nv_bfloat16* src, float* dst, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = bf16_to_f32_nv(src[i]);
}

__global__ void f32_to_bf16_convert(const float* src, __nv_bfloat16* dst, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = __float2bfloat16(src[i]);
}

void run(tvm::ffi::TensorView Q,
         tvm::ffi::TensorView K,
         tvm::ffi::TensorView V,
         tvm::ffi::TensorView O,
         tvm::ffi::TensorView dO,
         tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ,
         tvm::ffi::TensorView dK,
         tvm::ffi::TensorView dV) {
    
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = 4, H = 48, d = 128;
    int64_t S = Q.size(2);
    
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
    
    int64_t BH = B * H;
    int64_t SS = S * S;
    int64_t SD = S * d;
    
    // Single-batch-head buffers (process one bh at a time)
    float* tP = nullptr, *tdO = nullptr, *tV = nullptr;
    float* tdP = nullptr, *tdS = nullptr;
    float* tK = nullptr, *tQ = nullptr;
    float* tdV = nullptr, *tdQ = nullptr, *tdK = nullptr;
    
    size_t ssz = SS * sizeof(float);
    size_t sdz = SD * sizeof(float);
    
    CUDA_CHECK(cudaMallocAsync(&tP,   ssz, stream));
    CUDA_CHECK(cudaMallocAsync(&tdO,  sdz, stream));
    CUDA_CHECK(cudaMallocAsync(&tV,   sdz, stream));
    CUDA_CHECK(cudaMallocAsync(&tdP,  ssz, stream));
    CUDA_CHECK(cudaMallocAsync(&tdS,  ssz, stream));
    CUDA_CHECK(cudaMallocAsync(&tK,   sdz, stream));
    CUDA_CHECK(cudaMallocAsync(&tQ,   sdz, stream));
    CUDA_CHECK(cudaMallocAsync(&tdV,  sdz, stream));
    CUDA_CHECK(cudaMallocAsync(&tdQ,  sdz, stream));
    CUDA_CHECK(cudaMallocAsync(&tdK,  sdz, stream));
    
    constexpr int CONV_THREADS = 512;
    constexpr int BM_G = 128, BN_G = 64, BK_G = 16;
    int smem_sz_gemm = (BK_G * BM_G + BK_G * BN_G) * sizeof(float);
    
    float inv_sqrt_d = 1.0f / sqrtf((float)d);
    
    for (int64_t bh = 0; bh < BH; ++bh) {
        int64_t off_SD = bh * SD;
        
        // Pre-convert inputs to FP32
        int gb = (SD + CONV_THREADS - 1) / CONV_THREADS;
        bf16_to_f32_convert<<<gb, CONV_THREADS, 0, stream>>>(dO_p + off_SD, tdO, SD);
        bf16_to_f32_convert<<<gb, CONV_THREADS, 0, stream>>>(V_p + off_SD,  tV, SD);
        bf16_to_f32_convert<<<gb, CONV_THREADS, 0, stream>>>(K_p + off_SD,  tK, SD);
        bf16_to_f32_convert<<<gb, CONV_THREADS, 0, stream>>>(Q_p + off_SD,  tQ, SD);
        
        // Step 1: P[i,j] = exp(Q·K^T/sqrt(d) - L)
        int pb = std::min((int)S, 512);
        int pg = (S + pb - 1) / pb;
        compute_P_kernel<<<pg, pb, d * sizeof(float), stream>>>(
            Q_p + bh * SD, K_p + bh * SD, L_p + bh * S,
            tP, (int)S, (int)d);
        
        // Step 2: dV[S,d] = P[S,S]^T @ dO[S,d]
        dim3 gv((d + BN_G - 1) / BN_G, (S + BM_G - 1) / BM_G);
        gemm_tA_FP32<BM_G, BN_G, BK_G><<<gv, 256, smem_sz_gemm, stream>>>(
            tP, tdO, tdV, (int)S, (int)d, (int)S);
        
        // Step 3: dP[S,S] = dO[S,d] @ V[S,d]^T
        dim3 gp((S + BN_G - 1) / BN_G, (S + BM_G - 1) / BM_G);
        gemm_tB_FP32<BM_G, BN_G, BK_G><<<gp, 256, smem_sz_gemm, stream>>>(
            tdO, tV, tdP, (int)S, (int)S, (int)d);
        
        // Step 4: dS = softmax_backward(dP, P)
        softmax_bw_kernel<<<(int)S, pb, pb * sizeof(float), stream>>>(
            tdP, tP, tdS, (int)S);
        
        // Step 5: dQ_raw[S,d] = dS[S,S] @ K[S,d], then dQ = dQ_raw / sqrt(d)
        dim3 gq((d + BN_G - 1) / BN_G, (S + BM_G - 1) / BM_G);
        gemm_rr_FP32<BM_G, BN_G, BK_G><<<gq, 256, smem_sz_gemm, stream>>>(
            tdS, tK, tdQ, (int)S, (int)d, (int)S);
        
        // Step 6: dK_raw[S,d] = dS[S,S]^T @ Q[S,d], then dK = dK_raw / sqrt(d)
        dim3 gk((d + BN_G - 1) / BN_G, (S + BM_G - 1) / BM_G);
        gemm_tA_FP32<BM_G, BN_G, BK_G><<<gk, 256, smem_sz_gemm, stream>>>(
            tdS, tQ, tdK, (int)S, (int)d, (int)S);
        
        // Write outputs: dV no scale needed, dQ and dK need /sqrt(d)
        f32_to_bf16_convert<<<gb, CONV_THREADS, 0, stream>>>(tdV, dV_p + off_SD, SD);
        scale_and_convert<<<gb, CONV_THREADS, 0, stream>>>(tdQ, dQ_p + off_SD, SD, inv_sqrt_d);
        scale_and_convert<<<gb, CONV_THREADS, 0, stream>>>(tdK, dK_p + off_SD, SD, inv_sqrt_d);
    }
    
    CUDA_CHECK(cudaFreeAsync(tP, stream));
    CUDA_CHECK(cudaFreeAsync(tdO, stream));
    CUDA_CHECK(cudaFreeAsync(tV, stream));
    CUDA_CHECK(cudaFreeAsync(tdP, stream));
    CUDA_CHECK(cudaFreeAsync(tdS, stream));
    CUDA_CHECK(cudaFreeAsync(tK, stream));
    CUDA_CHECK(cudaFreeAsync(tQ, stream));
    CUDA_CHECK(cudaFreeAsync(tdV, stream));
    CUDA_CHECK(cudaFreeAsync(tdQ, stream));
    CUDA_CHECK(cudaFreeAsync(tdK, stream));
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace tvm_ffi_mha_bwd_d128

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd_d128::run);