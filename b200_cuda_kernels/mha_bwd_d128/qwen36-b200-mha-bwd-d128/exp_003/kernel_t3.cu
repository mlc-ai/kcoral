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

constexpr int WARP_SIZE = 32;

__device__ __forceinline__ float bf16_to_f32_nv(__nv_bfloat16 h) {
    return static_cast<float>(h);
}

// ==================== Kernel 1: Compute attention probs P ====================
// P[i,j] = exp(Q[i,:] · K[j,:] / sqrt(d) - L[i])
// Each block processes multiple rows using grid-stride loop over S dimension

__global__ void compute_P_kernel(
    const __nv_bfloat16* Q_bh,
    const __nv_bfloat16* K_bh,
    const float* L_bh,
    float* P_bh,
    int S, int d) {
    
    // Each block has blockDim.x threads, blockIdx.x iterates over query rows
    int i_row = blockIdx.x;
    int lane = threadIdx.x;
    
    if (i_row >= S) return;
    
    // Load Q[i_row, :] into shared memory
    extern __shared__ char smem_raw[];
    float* smem_Q = reinterpret_cast<float*>(smem_raw);
    
    for (int k = lane; k < d; k += blockDim.x) {
        smem_Q[k] = bf16_to_f32_nv(Q_bh[i_row * d + k]);
    }
    __syncthreads();
    
    float lsi = L_bh[i_row];
    float inv_sd = rsqrtf((float)d);
    
    // Each thread computes P[i_row, j] for its column j
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

// ==================== Tiled FP32 GEMM Kernels ====================

// Generic tiled GEMM: C[M,N] = A[M,K] @ B[K,N] in row-major FP32
template<int BM, int BN, int BK, int WM, int WN>
__global__ void gemm_FP32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {
    
    int tm = blockIdx.y * BM;
    int tn = blockIdx.x * BN;
    
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int wi = tid / WARP_SIZE;
    int wmi = (wi / (BN / WN)) * WM + lane / (WN / 2);
    int wni = (wi % (BN / WN)) * WN + (lane % (WN / 2)) * 2;
    
    if (wmi >= BM || wni >= BN) return;
    
    extern __shared__ float smem[];
    float* sa = smem;
    float* sb = smem + BK * BM;
    
    float acc[WM][WN] = {};
    
    int nk = (K + BK - 1) / BK;
    for (int t = 0; t < nk; ++t) {
        int ks = t * BK;
        
        // Load A tile
        for (int i = tid; i < BK * BM; i += blockDim.x) {
            int r = i / BM, c = i % BM;
            sa[i] = (tm + c < M && ks + r < K)
                ? A[(tm + c) * K + ks + r] : 0.0f;
        }
        __syncthreads();
        
        // Load B tile
        for (int i = tid; i < BK * BN; i += blockDim.x) {
            int r = i / BN, c = i % BN;
            sb[i] = (ks + r < K && tn + c < N)
                ? B[(ks + r) * N + tn + c] : 0.0f;
        }
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int im = 0; im < WM; ++im) {
                #pragma unroll
                for (int in = 0; in < WN; ++in) {
                    acc[im][in] += sa[k * BM + wmi + im - (wmi % WM)] *
                                   sb[k * BN + wni + in - (wni % WN)];
                }
            }
        }
        __syncthreads();
    }
    
    if (tm + wmi < M && tn + wni < N) {
        C[(tm + wmi) * N + tn + wni] = acc[0][0];
    }
}

// Simpler version with explicit accumulator per thread
template<int BM, int BN, int BK>
__global__ void simple_gemm_FP32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {
    
    constexpr int WM = 8, WN = 8;
    
    int tm = blockIdx.y * BM;
    int tn = blockIdx.x * BN;
    
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int wi = tid / WARP_SIZE;
    int wmi = (wi / (BN / WN)) * WM + lane / (WN / 2);
    int wni = (wi % (BN / WN)) * WN + (lane % (WN / 2)) * 2;
    
    if (wmi >= BM || wni >= BN) return;
    
    extern __shared__ float smem[];
    float* sa = smem;
    float* sb = smem + BK * BM;
    
    float acc = 0.0f;
    
    int nk = (K + BK - 1) / BK;
    for (int t = 0; t < nk; ++t) {
        int ks = t * BK;
        
        for (int i = tid; i < BK * BM; i += blockDim.x) {
            int r = i / BM, c = i % BM;
            sa[i] = (tm + c < M && ks + r < K)
                ? A[(tm + c) * K + ks + r] : 0.0f;
        }
        __syncthreads();
        
        for (int i = tid; i < BK * BN; i += blockDim.x) {
            int r = i / BN, c = i % BN;
            sb[i] = (ks + r < K && tn + c < N)
                ? B[(ks + r) * N + tn + c] : 0.0f;
        }
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            acc += sa[k * BM + wmi] * sb[k * BN + wni];
        }
        __syncthreads();
    }
    
    if (tm + wmi < M && tn + wni < N) {
        C[(tm + wmi) * N + tn + wni] = acc;
    }
}

// GEMM with transpose-B: C[M,N] = A[M,K] @ B[N,K]^T
template<int BM, int BN, int BK>
__global__ void gemm_F32TB(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {
    
    constexpr int WM = 8, WN = 8;
    
    int tm = blockIdx.y * BM;
    int tn = blockIdx.x * BN;
    
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int wi = tid / WARP_SIZE;
    int wmi = (wi / (BN / WN)) * WM + lane / (WN / 2);
    int wni = (wi % (BN / WN)) * WN + (lane % (WN / 2)) * 2;
    
    if (wmi >= BM || wni >= BN) return;
    
    extern __shared__ float smem[];
    float* sa = smem;
    float* sb = smem + BK * BM;
    
    float acc = 0.0f;
    
    int nk = (K + BK - 1) / BK;
    for (int t = 0; t < nk; ++t) {
        int ks = t * BK;
        
        for (int i = tid; i < BK * BM; i += blockDim.x) {
            int r = i / BM, c = i % BM;
            sa[i] = (tm + c < M && ks + r < K)
                ? A[(tm + c) * K + ks + r] : 0.0f;
        }
        __syncthreads();
        
        for (int i = tid; i < BK * BN; i += blockDim.x) {
            int r = i / BN, c = i % BN;
            sb[i] = (tn + c < N && ks + r < K)
                ? B[(tn + c) * K + ks + r] : 0.0f;
        }
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            acc += sa[k * BM + wmi] * sb[k * BN + wni];
        }
        __syncthreads();
    }
    
    if (tm + wmi < M && tn + wni < N) {
        C[(tm + wmi) * N + tn + wni] = acc;
    }
}

// GEMM with transpose-A: C[M,N] = A[K,M]^T @ B[K,N]
template<int BM, int BN, int BK>
__global__ void gemm_TBF32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {
    
    constexpr int WM = 8, WN = 8;
    
    int tm = blockIdx.y * BM;
    int tn = blockIdx.x * BN;
    
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int wi = tid / WARP_SIZE;
    int wmi = (wi / (BN / WN)) * WM + lane / (WN / 2);
    int wni = (wi % (BN / WN)) * WN + (lane % (WN / 2)) * 2;
    
    if (wmi >= BM || wni >= BN) return;
    
    extern __shared__ float smem[];
    float* sa = smem;
    float* sb = smem + BK * BM;
    
    float acc = 0.0f;
    
    int nk = (K + BK - 1) / BK;
    for (int t = 0; t < nk; ++t) {
        int ks = t * BK;
        
        for (int i = tid; i < BK * BM; i += blockDim.x) {
            int r = i / BM, c = i % BM;
            sa[i] = (ks + r < K && tm + c < M)
                ? A[(ks + r) * M + tm + c] : 0.0f;
        }
        __syncthreads();
        
        for (int i = tid; i < BK * BN; i += blockDim.x) {
            int r = i / BN, c = i % BN;
            sb[i] = (ks + r < K && tn + c < N)
                ? B[(ks + r) * N + tn + c] : 0.0f;
        }
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            acc += sa[k * BM + wmi] * sb[k * BN + wni];
        }
        __syncthreads();
    }
    
    if (tm + wmi < M && tn + wni < N) {
        C[(tm + wmi) * N + tn + wni] = acc;
    }
}

// BF16 to FP32 conversion kernel
__global__ void bf16_to_f32_convert(const __nv_bfloat16* src, float* dst, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = bf16_to_f32_nv(src[i]);
}

// FP32 to BF16 conversion kernel
__global__ void f32_to_bf16_convert(const float* src, __nv_bfloat16* dst, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = __float2bfloat16(src[i]);
}

// Element-wise multiply FP32
__global__ void fp32_mul(const float* A, const float* B, float* C, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) C[i] = A[i] * B[i];
}

// Softmax backward adjustment: dS[i,j] = P[i,j] * (dP[i,j] - c[i])
// where c[i] = sum_j(P[i,j] * dP[i,j])
__global__ void softmax_adj_kernel(const float* dP, const float* P, float* dS, int S) {
    int i = blockIdx.x;
    if (i >= S) return;
    
    // Reduce c[i] = sum_j(P[i,j] * dP[i,j])
    float* smem_c = nullptr;
    extern __shared__ char smem_raw[];
    float* smem_prod = reinterpret_cast<float*>(smem_raw);
    
    int lane = threadIdx.x;
    int n = S;
    
    // Compute product and partial reduce
    float val = 0.0f;
    for (int j = lane; j < n; j += blockDim.x) {
        val += P[i * S + j] * dP[i * S + j];
    }
    smem_prod[lane] = val;
    __syncthreads();
    
    // Parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            smem_prod[lane] += smem_prod[lane + stride];
        }
        __syncthreads();
    }
    
    float ci = smem_prod[0];
    
    // Write adjusted dS
    if (lane < S) {
        int j = lane;
        dS[i * S + j] = P[i * S + j] * (dP[i * S + j] - ci);
    }
}

// Host-side run function
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
    
    // Temp buffers in FP32
    float* tP = nullptr, *tdO = nullptr, *tV = nullptr;
    float* tdP = nullptr, *tdS = nullptr;
    float* tK = nullptr, *tQ = nullptr;
    float* tdV = nullptr, *tdQ = nullptr, *tdK = nullptr;
    
    size_t ssz = BH * SS * sizeof(float);
    size_t sdz = BH * SD * sizeof(float);
    
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
    
    constexpr int CONV_BLK = 512;
    
    // Pre-convert inputs to FP32
    for (int64_t bh = 0; bh < BH; ++bh) {
        int64_t off = bh * SD;
        int gb = (SD + CONV_BLK - 1) / CONV_BLK;
        bf16_to_f32_convert<<<gb, CONV_BLK, 0, stream>>>(dO_p + off, tdO + off, SD);
        bf16_to_f32_convert<<<gb, CONV_BLK, 0, stream>>>(V_p + off,  tV + off,  SD);
        bf16_to_f32_convert<<<gb, CONV_BLK, 0, stream>>>(K_p + off,  tK + off,  SD);
        bf16_to_f32_convert<<<gb, CONV_BLK, 0, stream>>>(Q_p + off,  tQ + off,  SD);
    }
    
    // Step 1: P[i,j] = exp(Q·K^T/sqrt(d) - L)
    for (int64_t bh = 0; bh < BH; ++bh) {
        int pb = std::min((int)S, 512);
        int pg = (S + pb - 1) / pb;
        compute_P_kernel<<<pg, pb, d * sizeof(float), stream>>>(
            Q_p + bh * SD, K_p + bh * SD, L_p + bh * S,
            tP + bh * SS, (int)S, (int)d);
    }
    
    // Step 2: dV = P^T @ dO  -> [S,d]
    // M=S, N=d, K=S
    constexpr int BM2 = 64, BN2 = 64, BK2 = 16;
    dim3 gv2((d + BN2 - 1) / BN2, (S + BM2 - 1) / BM2);
    dim3 bv2(BM2 * BN2 / 64);
    int smv2 = (BK2 * BM2 + BK2 * BN2) * sizeof(float);
    
    for (int64_t bh = 0; bh < BH; ++bh) {
        gemm_TBF32<BM2, BN2, BK2><<<gv2, bv2, smv2, stream>>>(
            tP + bh * SS, tdO + bh * SD, tdV + bh * SD,
            (int)S, (int)d, (int)S);
    }
    
    // Step 3: dP = dO @ V^T  -> [S,S]
    // M=S, N=S, K=d
    constexpr int BM3 = 64, BN3 = 64, BK3 = 16;
    dim3 gp3((S + BN3 - 1) / BN3, (S + BM3 - 1) / BM3);
    dim3 bp3(BM3 * BN3 / 64);
    int smdp3 = (BK3 * BM3 + BK3 * BN3) * sizeof(float);
    
    for (int64_t bh = 0; bh < BH; ++bh) {
        gemm_F32TB<BM3, BN3, BK3><<<gp3, bp3, smdp3, stream>>>(
            tdO + bh * SD, tV + bh * SD, tdP + bh * SS,
            (int)S, (int)S, (int)d);
    }
    
    // Step 4: dS = P * (dP - P.sum along last dim contracted)
    // Softmax backward: dS[i,j] = P[i,j] * (dP[i,j] - c[i])
    for (int64_t bh = 0; bh < BH; ++bh) {
        int pb = std::min((int)S, 512);
        softmax_adj_kernel<<<S, pb, pb * sizeof(float), stream>>>(
            tdP + bh * SS, tP + bh * SS, tdS + bh * SS, (int)S);
    }
    
    // Step 5: dQ = dS @ K  -> [S,d]
    constexpr int BM5 = 64, BN5 = 64, BK5 = 16;
    dim3 gq5((d + BN5 - 1) / BN5, (S + BM5 - 1) / BM5);
    dim3 bq5(BM5 * BN5 / 64);
    int smq5 = (BK5 * BM5 + BK5 * BN5) * sizeof(float);
    
    for (int64_t bh = 0; bh < BH; ++bh) {
        simple_gemm_FP32<BM5, BN5, BK5><<<gq5, bq5, smq5, stream>>>(
            tdS + bh * SS, tK + bh * SD, tdQ + bh * SD,
            (int)S, (int)d, (int)S);
    }
    
    // Step 6: dK = dS^T @ Q  -> [S,d]
    dim3 gk6((d + BN2 - 1) / BN2, (S + BM2 - 1) / BM2);
    dim3 bk6(BM2 * BN2 / 64);
    int smk6 = (BK2 * BM2 + BK2 * BN2) * sizeof(float);
    
    for (int64_t bh = 0; bh < BH; ++bh) {
        gemm_TBF32<BM2, BN2, BK2><<<gk6, bk6, smk6, stream>>>(
            tdS + bh * SS, tQ + bh * SD, tdK + bh * SD,
            (int)S, (int)d, (int)S);
    }
    
    // Post-convert outputs to BF16
    for (int64_t bh = 0; bh < BH; ++bh) {
        int64_t off = bh * SD;
        int gb = (SD + CONV_BLK - 1) / CONV_BLK;
        f32_to_bf16_convert<<<gb, CONV_BLK, 0, stream>>>(tdV + off, dV_p + off, SD);
        f32_to_bf16_convert<<<gb, CONV_BLK, 0, stream>>>(tdQ + off, dQ_p + off, SD);
        f32_to_bf16_convert<<<gb, CONV_BLK, 0, stream>>>(tdK + off, dK_p + off, SD);
    }
    
    // Cleanup temp memory
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