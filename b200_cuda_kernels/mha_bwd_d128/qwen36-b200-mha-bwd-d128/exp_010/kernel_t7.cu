#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
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

namespace mha_bwd_d128 {

constexpr int BM = 32;
constexpr int BN = 32;
constexpr int BD = 128;
constexpr int TPB = 128;

__device__ __forceinline__ float bf16tof(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 ftobf16(float x) {
    return __float2bfloat16(x);
}

__global__ void mha_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    float* __restrict__ dQ_acc,
    float* __restrict__ dK_acc,
    float* __restrict__ dV_acc,
    int B, int H, int S, int D,
    float scale) {

    extern __shared__ char smem[];
    
    // sQ: BM*BD bf16, sK: BN*BD bf16, sV: BN*BD bf16, sDO: BM*BD bf16, sLSE: BM float
    // Total = 2*(BM+BN)*BD*2 + BM*4 bytes
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sK = sQ + BM * BD;
    __nv_bfloat16* sV = sK + BN * BD;
    __nv_bfloat16* sDO = sV + BN * BD;
    float* sLSE = reinterpret_cast<float*>(sDO + BM * BD);
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = tx + ty * 32;
    
    int linear_id = blockIdx.x;
    int num_qtiles = (S + BM - 1) / BM;
    int num_ktiles = (S + BN - 1) / BN;
    
    int bh = linear_id / (num_qtiles * num_ktiles);
    int qktile = linear_id % (num_qtiles * num_ktiles);
    int qt = qktile / num_ktiles;
    int kt = qktile % num_ktiles;
    
    int b = bh / H;
    int h = bh % H;
    if (b >= B || h >= H) return;
    
    int qs = qt * BM;
    int ks = kt * BN;
    
    int base_bh = (b * H + h) * S * D;
    
    // Load Q tile
    for (int i = tid; i < BM * BD; i += TPB) {
        int r = i / BD;
        int c = i % BD;
        sQ[i] = (qs + r < S && c < D) ? Q[base_bh + (qs + r) * D + c] : ftobf16(0.0f);
    }
    
    // Load K tile
    for (int i = tid; i < BN * BD; i += TPB) {
        int r = i / BD;
        int c = i % BD;
        sK[i] = (ks + r < S && c < D) ? K[base_bh + (ks + r) * D + c] : ftobf16(0.0f);
    }
    
    // Load V tile
    for (int i = tid; i < BN * BD; i += TPB) {
        int r = i / BD;
        int c = i % BD;
        sV[i] = (ks + r < S && c < D) ? V[base_bh + (ks + r) * D + c] : ftobf16(0.0f);
    }
    
    // Load dO tile
    for (int i = tid; i < BM * BD; i += TPB) {
        int r = i / BD;
        int c = i % BD;
        sDO[i] = (qs + r < S && c < D) ? dO[base_bh + (qs + r) * D + c] : ftobf16(0.0f);
    }
    
    // Load LSE
    for (int i = tid; i < BM; i += TPB) {
        sLSE[i] = (qs + i < S) ? L[(b * H + h) * S + qs + i] : 0.0f;
    }
    
    __syncthreads();
    
    // Each thread handles one (m,n) pair
    int m = ty;
    int n = tx;
    
    int qm = qs + m;
    int kn = ks + n;
    
    if (qm < S && kn < S) {
        float lse_val = sLSE[m];
        
        // S[m,n] = dot(Q[qm], K[kn]) * scale
        float s_val = 0.0f;
        #pragma unroll
        for (int d = 0; d < BD; d++) {
            s_val += bf16tof(sQ[m * BD + d]) * bf16tof(sK[n * BD + d]);
        }
        s_val *= scale;
        
        float p_val = expf(s_val - lse_val);
        
        // dPV[m,n] = dot(dO[qm], V[kn])
        float dpv = 0.0f;
        #pragma unroll
        for (int d = 0; d < BD; d++) {
            dpv += bf16tof(sDO[m * BD + d]) * bf16tof(sV[n * BD + d]);
        }
        
        // Warp reduce for local_mean
        float p_dpvi = p_val * dpv;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            p_dpvi += __shfl_down_sync(0xFFFFFFFF, p_dpvi, offset);
        }
        #pragma unroll
        for (int offset = 8; offset > 0; offset /= 2) {
            p_dpvi += __shfl_down_sync(0xFFFFFFFF, p_dpvi, offset);
        }
        #pragma unroll
        for (int offset = 4; offset > 0; offset /= 2) {
            p_dpvi += __shfl_down_sync(0xFFFFFFFF, p_dpvi, offset);
        }
        #pragma unroll
        for (int offset = 2; offset > 0; offset /= 2) {
            p_dpvi += __shfl_down_sync(0xFFFFFFFF, p_dpvi, offset);
        }
        #pragma unroll
        for (int offset = 1; offset > 0; offset /= 2) {
            p_dpvi += __shfl_down_sync(0xFFFFFFFF, p_dpvi, offset);
        }
        float local_mean = p_dpvi;
        
        float ds = p_val * (dpv - local_mean);
        
        // Atomic accumulate to fp32 buffers
        for (int d = 0; d < BD; d++) {
            int dv_idx = base_bh + kn * D + d;
            int dq_idx = base_bh + qm * D + d;
            int dk_idx = base_bh + kn * D + d;
            
            atomicAdd(&dV_acc[dv_idx], p_val * bf16tof(sDO[m * BD + d]));
            atomicAdd(&dQ_acc[dq_idx], ds * bf16tof(sK[n * BD + d]));
            atomicAdd(&dK_acc[dk_idx], ds * bf16tof(sQ[m * BD + d]));
        }
    }
}

__global__ void to_bf16_kernel(const float* src, __nv_bfloat16* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = ftobf16(src[idx]);
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
    
    float scale = 1.0f / sqrtf(static_cast<float>(D));
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int64_t nout = B * H * S * D;
    size_t fsize = nout * sizeof(float);
    
    float* dQ_acc, *dK_acc, *dV_acc;
    CUDA_CHECK(cudaMallocAsync(&dQ_acc, fsize, stream));
    CUDA_CHECK(cudaMallocAsync(&dK_acc, fsize, stream));
    CUDA_CHECK(cudaMallocAsync(&dV_acc, fsize, stream));
    CUDA_CHECK(cudaMemsetAsync(dQ_acc, 0, fsize, stream));
    CUDA_CHECK(cudaMemsetAsync(dK_acc, 0, fsize, stream));
    CUDA_CHECK(cudaMemsetAsync(dV_acc, 0, fsize, stream));
    
    int num_qtiles = (S + BM - 1) / BM;
    int num_ktiles = (S + BN - 1) / BN;
    int total_blocks = B * H * num_qtiles * num_ktiles;
    
    dim3 block(32, 4);
    dim3 grid(total_blocks);
    
    size_t smem_size = 
        BM * BD * sizeof(__nv_bfloat16) +   // sQ
        BN * BD * sizeof(__nv_bfloat16) +   // sK
        BN * BD * sizeof(__nv_bfloat16) +   // sV
        BM * BD * sizeof(__nv_bfloat16) +   // sDO
        BM * sizeof(float);                 // sLSE
    
    mha_bwd_kernel<<<grid, block, smem_size, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        dQ_acc, dK_acc, dV_acc,
        B, H, S, D, scale);
    CUDA_CHECK(cudaGetLastError());
    
    int64_t threads = 256;
    int64_t blocks = (nout + threads - 1) / threads;
    
    to_bf16_kernel<<<blocks, threads, 0, stream>>>(dQ_acc,
        static_cast<__nv_bfloat16*>(dQ.data_ptr()), nout);
    to_bf16_kernel<<<blocks, threads, 0, stream>>>(dK_acc,
        static_cast<__nv_bfloat16*>(dK.data_ptr()), nout);
    to_bf16_kernel<<<blocks, threads, 0, stream>>>(dV_acc,
        static_cast<__nv_bfloat16*>(dV.data_ptr()), nout);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaFreeAsync(dQ_acc, stream));
    CUDA_CHECK(cudaFreeAsync(dK_acc, stream));
    CUDA_CHECK(cudaFreeAsync(dV_acc, stream));
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128::run);

}  // namespace mha_bwd_d128