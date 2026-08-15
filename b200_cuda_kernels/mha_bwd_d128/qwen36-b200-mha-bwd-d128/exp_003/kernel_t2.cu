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

// Convert BF16 -> FP32 using standard cast (unambiguous via static_cast)
__device__ __forceinline__ float bf16_to_f32_nv(__nv_bfloat16 h) {
    return static_cast<float>(h);
}

// ==================== Kernel 1: Compute attention probs P ====================
// P[i,j] = exp(Q[i,:] · K[j,:] / sqrt(d) - L[i])
// One block per query row i within a (b,h) group

__global__ void compute_P_kernel(
    const __nv_bfloat16* Q_bh,
    const __nv_bfloat16* K_bh,
    const float* L_bh,
    __half* P_bh,
    int S, int d) {
    
    int i = blockIdx.x;
    if (i >= S) return;
    
    int j = threadIdx.x;
    
    // Load Q[i,:] into shared memory
    extern __shared__ char smem_raw[];
    float* smem_Q = reinterpret_cast<float*>(smem_raw);
    
    for (int k = j; k < d; k += blockDim.x) {
        smem_Q[k] = bf16_to_f32_nv(Q_bh[i * d + k]);
    }
    __syncthreads();
    
    float lsi = L_bh[i];
    float inv_sd = rsqrtf((float)d);
    
    if (j < S) {
        float dot = 0.0f;
        // Unrolled load from K and multiply with shared Q
        for (int k = 0; k < d; k += 4) {
            dot += smem_Q[k]     * bf16_to_f32_nv(K_bh[j * d + k]);
            dot += smem_Q[k + 1] * bf16_to_f32_nv(K_bh[j * d + k + 1]);
            dot += smem_Q[k + 2] * bf16_to_f32_nv(K_bh[j * d + k + 2]);
            dot += smem_Q[k + 3] * bf16_to_f32_nv(K_bh[j * d + k + 3]);
        }
        P_bh[i * S + j] = __float2half(expf(dot * inv_sd - lsi));
    }
}

// ==================== Tiled GEMM Kernels ====================

// Generic tiled GEMM: C[M,N] = A[M,K] @ B[K,N] in row-major FP16
template<int BM, int BN, int BK, int WM, int WN>
__global__ void gemm_FP16(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K) {
    
    int tm = blockIdx.y * BM;
    int tn = blockIdx.x * BN;
    
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int wi = tid / WARP_SIZE;
    int wmi = (wi / (BN / WN)) * WM + lane / (WN / 2);
    int wni = (wi % (BN / WN)) * WN + (lane % (WN / 2)) * 2;
    
    if (wmi >= BM || wni >= BN) return;
    
    extern __shared__ __half smem[];
    __half* sa = smem;
    __half* sb = smem + BK * BM;
    
    float acc = 0.0f;
    
    int nk = (K + BK - 1) / BK;
    for (int t = 0; t < nk; ++t) {
        int ks = t * BK;
        
        // Load A tile
        for (int i = tid; i < BK * BM; i += blockDim.x) {
            int r = i / BM, c = i % BM;
            sa[i] = (tm + c < M && ks + r < K)
                ? A[(tm + c) * K + ks + r] : __float2half(0.0f);
        }
        __syncthreads();
        
        // Load B tile
        for (int i = tid; i < BK * BN; i += blockDim.x) {
            int r = i / BN, c = i % BN;
            sb[i] = (ks + r < K && tn + c < N)
                ? B[(ks + r) * N + tn + c] : __float2half(0.0f);
        }
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float va = __half2float(sa[k * BM + wmi]);
            float vb = __half2float(sb[k * BN + wni]);
            acc += va * vb;
        }
        __syncthreads();
    }
    
    if (tm + wmi < M && tn + wni < N) {
        C[(tm + wmi) * N + tn + wni] = __float2half(acc);
    }
}

// GEMM with transpose-B: C[M,N] = A[M,K] @ B[N,K]^T
// C[i,j] = sum_k A[i,k] * B[j,k]
template<int BM, int BN, int BK, int WM, int WN>
__global__ void gemm_F16TB(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K) {
    
    int tm = blockIdx.y * BM;
    int tn = blockIdx.x * BN;
    
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int wi = tid / WARP_SIZE;
    int wmi = (wi / (BN / WN)) * WM + lane / (WN / 2);
    int wni = (wi % (BN / WN)) * WN + (lane % (WN / 2)) * 2;
    
    if (wmi >= BM || wni >= BN) return;
    
    extern __shared__ __half smem[];
    __half* sa = smem;
    __half* sb = smem + BK * BM;
    
    float acc = 0.0f;
    
    int nk = (K + BK - 1) / BK;
    for (int t = 0; t < nk; ++t) {
        int ks = t * BK;
        
        for (int i = tid; i < BK * BM; i += blockDim.x) {
            int r = i / BM, c = i % BM;
            sa[i] = (tm + c < M && ks + r < K)
                ? A[(tm + c) * K + ks + r] : __float2half(0.0f);
        }
        __syncthreads();
        
        // B is read as B[j,k] where j varies (rows of B) and k is the reduction
        for (int i = tid; i < BK * BN; i += blockDim.x) {
            int r = i / BN, c = i % BN;
            sb[i] = (tn + c < N && ks + r < K)
                ? B[(tn + c) * K + ks + r] : __float2half(0.0f);
        }
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float va = __half2float(sa[k * BM + wmi]);
            float vb = __half2float(sb[k * BN + wni]);
            acc += va * vb;
        }
        __syncthreads();
    }
    
    if (tm + wmi < M && tn + wni < N) {
        C[(tm + wmi) * N + tn + wni] = __float2half(acc);
    }
}

// GEMM with transpose-A: C[M,N] = A[K,M]^T @ B[K,N]
// C[i,j] = sum_k A[k,i] * B[k,j]
template<int BM, int BN, int BK, int WM, int WN>
__global__ void gemm_TBF16(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K) {
    
    int tm = blockIdx.y * BM;
    int tn = blockIdx.x * BN;
    
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int wi = tid / WARP_SIZE;
    int wmi = (wi / (BN / WN)) * WM + lane / (WN / 2);
    int wni = (wi % (BN / WN)) * WN + (lane % (WN / 2)) * 2;
    
    if (wmi >= BM || wni >= BN) return;
    
    extern __shared__ __half smem[];
    __half* sa = smem;
    __half* sb = smem + BK * BM;
    
    float acc = 0.0f;
    
    int nk = (K + BK - 1) / BK;
    for (int t = 0; t < nk; ++t) {
        int ks = t * BK;
        
        // A[K,M], access A[k,m] for shared memory indexed as [k_offset][m_local]
        for (int i = tid; i < BK * BM; i += blockDim.x) {
            int r = i / BM, c = i % BM;
            sa[i] = (ks + r < K && tm + c < M)
                ? A[(ks + r) * M + tm + c] : __float2half(0.0f);
        }
        __syncthreads();
        
        for (int i = tid; i < BK * BN; i += blockDim.x) {
            int r = i / BN, c = i % BN;
            sb[i] = (ks + r < K && tn + c < N)
                ? B[(ks + r) * N + tn + c] : __float2half(0.0f);
        }
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float va = __half2float(sa[k * BM + wmi]);
            float vb = __half2float(sb[k * BN + wni]);
            acc += va * vb;
        }
        __syncthreads();
    }
    
    if (tm + wmi < M && tn + wni < N) {
        C[(tm + wmi) * N + tn + wni] = __float2half(acc);
    }
}

// Conversion kernels
__global__ void bf16_to_half_convert(const __nv_bfloat16* src, __half* dst, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = __float2half(bf16_to_f32_nv(src[i]));
}

__global__ void half_to_bf16_convert(const __half* src, __nv_bfloat16* dst, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = __float2bfloat16(__half2float(src[i]));
}

__global__ void half_mul(const __half* A, const __half* B, __half* C, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) C[i] = __hmul(A[i], B[i]);
}

// ==================== Host-side run function ====================

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
    
    // Temp buffers
    __half* tP = nullptr, *tdO = nullptr, *tV = nullptr;
    __half* tdP = nullptr, *tdS = nullptr;
    __half* tK = nullptr, *tQ = nullptr;
    __half* tdV = nullptr, *tdQ = nullptr, *tdK = nullptr;
    
    size_t ssz = BH * SS * sizeof(__half);
    size_t sdz = BH * SD * sizeof(__half);
    
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
    
    // Pre-convert inputs to FP16
    for (int64_t bh = 0; bh < BH; ++bh) {
        int64_t off = bh * SD;
        int gb = (SD + CONV_BLK - 1) / CONV_BLK;
        bf16_to_half_convert<<<gb, CONV_BLK, 0, stream>>>(dO_p + off, tdO + off, SD);
        bf16_to_half_convert<<<gb, CONV_BLK, 0, stream>>>(V_p + off,  tV + off,  SD);
        bf16_to_half_convert<<<gb, CONV_BLK, 0, stream>>>(K_p + off,  tK + off,  SD);
        bf16_to_half_convert<<<gb, CONV_BLK, 0, stream>>>(Q_p + off,  tQ + off,  SD);
    }
    
    // Step 1: P[i,j] = exp(Q·K^T/sqrt(d) - L)
    for (int64_t bh = 0; bh < BH; ++bh) {
        int pb = std::min((int)S, 512);
        int pg = (S + pb - 1) / pb;
        compute_P_kernel<<<pg, pb, d * sizeof(float), stream>>>(
            Q_p + bh * SD, K_p + bh * SD, L_p + bh * S,
            tP + bh * SS, (int)S, (int)d);
    }
    
    // Step 2: dV = P^T @ dO  -> gemm_TBF16(P[S,S], dO[S,d]) -> [S,d]
    // M=S, N=d, K=S
    constexpr int BM2 = 64, BN2 = 64, BK2 = 16;
    dim3 gv2((d + BN2 - 1) / BN2, (S + BM2 - 1) / BM2);
    dim3 bv2(BM2 * BN2 / 64);
    int smv2 = (BK2 * BM2 + BK2 * BN2) * sizeof(__half);
    
    for (int64_t bh = 0; bh < BH; ++bh) {
        gemm_TBF16<BM2, BN2, BK2, 8, 8><<<gv2, bv2, smv2, stream>>>(
            tP + bh * SS, tdO + bh * SD, tdV + bh * SD,
            (int)S, (int)d, (int)S);
    }
    
    // Step 3: dP = dO @ V^T  -> gemm_F16TB(dO[S,d], V[S,d]) -> [S,S]
    // M=S, N=S, K=d
    constexpr int BM3 = 64, BN3 = 64, BK3 = 16;
    dim3 gp3((S + BN3 - 1) / BN3, (S + BM3 - 1) / BM3);
    dim3 bp3(BM3 * BN3 / 64);
    int smdp3 = (BK3 * BM3 + BK3 * BN3) * sizeof(__half);
    
    for (int64_t bh = 0; bh < BH; ++bh) {
        gemm_F16TB<BM3, BN3, BK3, 8, 8><<<gp3, bp3, smdp3, stream>>>(
            tdO + bh * SD, tV + bh * SD, tdP + bh * SS,
            (int)S, (int)S, (int)d);
    }
    
    // Step 4: dS = dP * P  (element-wise)
    for (int64_t bh = 0; bh < BH; ++bh) {
        int eg = (SS + CONV_BLK - 1) / CONV_BLK;
        half_mul<<<eg, CONV_BLK, 0, stream>>>(
            tdP + bh * SS, tP + bh * SS, tdS + bh * SS, SS);
    }
    
    // Step 5: dQ = dS @ K  -> gemm_FP16(dS[S,S], K[S,d]) -> [S,d]
    // M=S, N=d, K=S
    constexpr int BM5 = 64, BN5 = 64, BK5 = 16;
    dim3 gq5((d + BN5 - 1) / BN5, (S + BM5 - 1) / BM5);
    dim3 bq5(BM5 * BN5 / 64);
    int smq5 = (BK5 * BM5 + BK5 * BN5) * sizeof(__half);
    
    for (int64_t bh = 0; bh < BH; ++bh) {
        gemm_FP16<BM5, BN5, BK5, 8, 8><<<gq5, bq5, smq5, stream>>>(
            tdS + bh * SS, tK + bh * SD, tdQ + bh * SD,
            (int)S, (int)d, (int)S);
    }
    
    // Step 6: dK = dS^T @ Q  -> gemm_TBF16(dS[S,S], Q[S,d]) -> [S,d]
    // M=S, N=d, K=S
    dim3 gk6((d + BN2 - 1) / BN2, (S + BM2 - 1) / BM2);
    dim3 bk6(BM2 * BN2 / 64);
    int smk6 = (BK2 * BM2 + BK2 * BN2) * sizeof(__half);
    
    for (int64_t bh = 0; bh < BH; ++bh) {
        gemm_TBF16<BM2, BN2, BK2, 8, 8><<<gk6, bk6, smk6, stream>>>(
            tdS + bh * SS, tQ + bh * SD, tdK + bh * SD,
            (int)S, (int)d, (int)S);
    }
    
    // Post-convert outputs to BF16
    for (int64_t bh = 0; bh < BH; ++bh) {
        int64_t off = bh * SD;
        int gb = (SD + CONV_BLK - 1) / CONV_BLK;
        half_to_bf16_convert<<<gb, CONV_BLK, 0, stream>>>(tdV + off, dV_p + off, SD);
        half_to_bf16_convert<<<gb, CONV_BLK, 0, stream>>>(tdQ + off, dQ_p + off, SD);
        half_to_bf16_convert<<<gb, CONV_BLK, 0, stream>>>(tdK + off, dK_p + off, SD);
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