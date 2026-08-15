#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

namespace mha_bwd_d128 {

constexpr int B_CONST = 4;
constexpr int H_CONST = 48;
constexpr int D_CONST = 128;
constexpr int BR = 64;
constexpr int BC = 64;
constexpr int THREADS = 256;
constexpr int D_PAD = 136;
constexpr int BC_PAD = 65;
constexpr int D8 = D_CONST / 8;

__global__ void compute_D_kernel(
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    float* __restrict__ D, int S) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B_CONST * H_CONST * S;
    if (idx >= total) return;
    int i = idx % S;
    int bh = idx / S;
    const __nv_bfloat16* O_ptr = O + (size_t)bh * S * D_CONST + (size_t)i * D_CONST;
    const __nv_bfloat16* dO_ptr = dO + (size_t)bh * S * D_CONST + (size_t)i * D_CONST;
    float d_val = 0.0f;
    #pragma unroll 8
    for (int k = 0; k < D_CONST; k += 2) {
        float2 o = __bfloat1622float2(*(const __nv_bfloat162*)&O_ptr[k]);
        float2 g = __bfloat1622float2(*(const __nv_bfloat162*)&dO_ptr[k]);
        d_val = fmaf(o.x, g.x, d_val);
        d_val = fmaf(o.y, g.y, d_val);
    }
    D[idx] = d_val;
}

__global__ void compute_dQ_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D,
    __nv_bfloat16* __restrict__ dQ, int S, float scale) {
    
    int num_q_blocks = (S + BR - 1) / BR;
    int bh = blockIdx.x / num_q_blocks;
    int qi_start = (blockIdx.x % num_q_blocks) * BR;
    int br = min(BR, S - qi_start);
    int tid = threadIdx.x;
    
    const __nv_bfloat16* Q_bh = Q + (size_t)bh * S * D_CONST;
    const __nv_bfloat16* K_bh = K + (size_t)bh * S * D_CONST;
    const __nv_bfloat16* V_bh = V + (size_t)bh * S * D_CONST;
    const __nv_bfloat16* dO_bh = dO + (size_t)bh * S * D_CONST;
    const float* L_bh = L + (size_t)bh * S;
    const float* D_bh = D + (size_t)bh * S;
    __nv_bfloat16* dQ_bh = dQ + (size_t)bh * S * D_CONST;
    
    extern __shared__ char smem[];
    __nv_bfloat16* sQ = (__nv_bfloat16*)smem;
    __nv_bfloat16* sdO = sQ + BR * D_PAD;
    __nv_bfloat16* sK = sdO + BR * D_PAD;
    __nv_bfloat16* sV = sK + BC * D_PAD;
    float* sP = (float*)(sV + BC * D_PAD);
    float* sdQ = sP + BR * BC_PAD;
    float* sL = sdQ + BR * D_PAD;
    float* sDrow = sL + BR;
    
    for (int idx = tid; idx < br * D8; idx += THREADS) {
        int i = idx / D8, k8 = idx % D8;
        *(int4*)&sQ[i * D_PAD + k8 * 8] = *(const int4*)&Q_bh[(qi_start + i) * D_CONST + k8 * 8];
        *(int4*)&sdO[i * D_PAD + k8 * 8] = *(const int4*)&dO_bh[(qi_start + i) * D_CONST + k8 * 8];
    }
    for (int i = tid; i < br; i += THREADS) {
        sL[i] = L_bh[qi_start + i];
        sDrow[i] = D_bh[qi_start + i];
    }
    for (int idx = tid; idx < br * D_CONST; idx += THREADS) {
        sdQ[(idx / D_CONST) * D_PAD + (idx % D_CONST)] = 0.0f;
    }
    __syncthreads();
    
    for (int kj = 0; kj < S; kj += BC) {
        int bc = min(BC, S - kj);
        for (int idx = tid; idx < bc * D8; idx += THREADS) {
            int j = idx / D8, k8 = idx % D8;
            *(int4*)&sK[j * D_PAD + k8 * 8] = *(const int4*)&K_bh[(kj + j) * D_CONST + k8 * 8];
            *(int4*)&sV[j * D_PAD + k8 * 8] = *(const int4*)&V_bh[(kj + j) * D_CONST + k8 * 8];
        }
        __syncthreads();
        
        for (int idx = tid; idx < br * bc; idx += THREADS) {
            int i = idx / bc, j = idx % bc;
            float s = 0.0f;
            const __nv_bfloat16* qrow = &sQ[i * D_PAD];
            const __nv_bfloat16* krow = &sK[j * D_PAD];
            #pragma unroll 8
            for (int k = 0; k < D_CONST; k += 2) {
                float2 q2 = __bfloat1622float2(*(const __nv_bfloat162*)&qrow[k]);
                float2 k2 = __bfloat1622float2(*(const __nv_bfloat162*)&krow[k]);
                s = fmaf(q2.x, k2.x, s);
                s = fmaf(q2.y, k2.y, s);
            }
            sP[i * BC_PAD + j] = __expf(s * scale - sL[i]);
        }
        __syncthreads();
        
        for (int idx = tid; idx < br * bc; idx += THREADS) {
            int i = idx / bc, j = idx % bc;
            float dp = 0.0f;
            const __nv_bfloat16* dorow = &sdO[i * D_PAD];
            const __nv_bfloat16* vrow = &sV[j * D_PAD];
            #pragma unroll 8
            for (int k = 0; k < D_CONST; k += 2) {
                float2 d2 = __bfloat1622float2(*(const __nv_bfloat162*)&dorow[k]);
                float2 v2 = __bfloat1622float2(*(const __nv_bfloat162*)&vrow[k]);
                dp = fmaf(d2.x, v2.x, dp);
                dp = fmaf(d2.y, v2.y, dp);
            }
            float p = sP[i * BC_PAD + j];
            sP[i * BC_PAD + j] = p * (dp - sDrow[i]);
        }
        __syncthreads();
        
        for (int idx = tid; idx < br * D_CONST; idx += THREADS) {
            int i = idx / D_CONST, k = idx % D_CONST;
            float dq = 0.0f;
            for (int j = 0; j < bc; j++)
                dq = fmaf(sP[i * BC_PAD + j], __bfloat162float(sK[j * D_PAD + k]), dq);
            sdQ[i * D_PAD + k] += dq * scale;
        }
        __syncthreads();
    }
    
    for (int idx = tid; idx < br * D_CONST; idx += THREADS) {
        int i = idx / D_CONST, k = idx % D_CONST;
        dQ_bh[(qi_start + i) * D_CONST + k] = __float2bfloat16(sdQ[i * D_PAD + k]);
    }
}

__global__ void compute_dKV_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV, int S, float scale) {
    
    int num_k_blocks = (S + BC - 1) / BC;
    int bh = blockIdx.x / num_k_blocks;
    int kj_start = (blockIdx.x % num_k_blocks) * BC;
    int bc = min(BC, S - kj_start);
    int tid = threadIdx.x;
    
    const __nv_bfloat16* Q_bh = Q + (size_t)bh * S * D_CONST;
    const __nv_bfloat16* K_bh = K + (size_t)bh * S * D_CONST;
    const __nv_bfloat16* V_bh = V + (size_t)bh * S * D_CONST;
    const __nv_bfloat16* dO_bh = dO + (size_t)bh * S * D_CONST;
    const float* L_bh = L + (size_t)bh * S;
    const float* D_bh = D + (size_t)bh * S;
    __nv_bfloat16* dK_bh = dK + (size_t)bh * S * D_CONST;
    __nv_bfloat16* dV_bh = dV + (size_t)bh * S * D_CONST;
    
    extern __shared__ char smem[];
    __nv_bfloat16* sK = (__nv_bfloat16*)smem;
    __nv_bfloat16* sV = sK + BC * D_PAD;
    __nv_bfloat16* sQ = sV + BC * D_PAD;
    __nv_bfloat16* sdO = sQ + BR * D_PAD;
    float* sP = (float*)(sdO + BR * D_PAD);
    float* sdK = sP + BR * BC_PAD;
    float* sdV = sdK + BC * D_PAD;
    float* sL = sdV + BC * D_PAD;
    float* sDrow = sL + BR;
    
    for (int idx = tid; idx < bc * D8; idx += THREADS) {
        int j = idx / D8, k8 = idx % D8;
        *(int4*)&sK[j * D_PAD + k8 * 8] = *(const int4*)&K_bh[(kj_start + j) * D_CONST + k8 * 8];
        *(int4*)&sV[j * D_PAD + k8 * 8] = *(const int4*)&V_bh[(kj_start + j) * D_CONST + k8 * 8];
    }
    for (int idx = tid; idx < bc * D_CONST; idx += THREADS) {
        sdK[(idx / D_CONST) * D_PAD + (idx % D_CONST)] = 0.0f;
        sdV[(idx / D_CONST) * D_PAD + (idx % D_CONST)] = 0.0f;
    }
    __syncthreads();
    
    for (int qi = 0; qi < S; qi += BR) {
        int br = min(BR, S - qi);
        for (int idx = tid; idx < br * D8; idx += THREADS) {
            int i = idx / D8, k8 = idx % D8;
            *(int4*)&sQ[i * D_PAD + k8 * 8] = *(const int4*)&Q_bh[(qi + i) * D_CONST + k8 * 8];
            *(int4*)&sdO[i * D_PAD + k8 * 8] = *(const int4*)&dO_bh[(qi + i) * D_CONST + k8 * 8];
        }
        for (int i = tid; i < br; i += THREADS) {
            sL[i] = L_bh[qi + i];
            sDrow[i] = D_bh[qi + i];
        }
        __syncthreads();
        
        for (int idx = tid; idx < br * bc; idx += THREADS) {
            int i = idx / bc, j = idx % bc;
            float s = 0.0f;
            const __nv_bfloat16* qrow = &sQ[i * D_PAD];
            const __nv_bfloat16* krow = &sK[j * D_PAD];
            #pragma unroll 8
            for (int k = 0; k < D_CONST; k += 2) {
                float2 q2 = __bfloat1622float2(*(const __nv_bfloat162*)&qrow[k]);
                float2 k2 = __bfloat1622float2(*(const __nv_bfloat162*)&krow[k]);
                s = fmaf(q2.x, k2.x, s);
                s = fmaf(q2.y, k2.y, s);
            }
            sP[i * BC_PAD + j] = __expf(s * scale - sL[i]);
        }
        __syncthreads();
        
        for (int idx = tid; idx < bc * D_CONST; idx += THREADS) {
            int j = idx / D_CONST, k = idx % D_CONST;
            float dv = 0.0f;
            for (int i = 0; i < br; i++)
                dv = fmaf(sP[i * BC_PAD + j], __bfloat162float(sdO[i * D_PAD + k]), dv);
            sdV[j * D_PAD + k] += dv;
        }
        __syncthreads();
        
        for (int idx = tid; idx < br * bc; idx += THREADS) {
            int i = idx / bc, j = idx % bc;
            float dp = 0.0f;
            const __nv_bfloat16* dorow = &sdO[i * D_PAD];
            const __nv_bfloat16* vrow = &sV[j * D_PAD];
            #pragma unroll 8
            for (int k = 0; k < D_CONST; k += 2) {
                float2 d2 = __bfloat1622float2(*(const __nv_bfloat162*)&dorow[k]);
                float2 v2 = __bfloat1622float2(*(const __nv_bfloat162*)&vrow[k]);
                dp = fmaf(d2.x, v2.x, dp);
                dp = fmaf(d2.y, v2.y, dp);
            }
            float p = sP[i * BC_PAD + j];
            sP[i * BC_PAD + j] = p * (dp - sDrow[i]);
        }
        __syncthreads();
        
        for (int idx = tid; idx < bc * D_CONST; idx += THREADS) {
            int j = idx / D_CONST, k = idx % D_CONST;
            float dk = 0.0f;
            for (int i = 0; i < br; i++)
                dk = fmaf(sP[i * BC_PAD + j], __bfloat162float(sQ[i * D_PAD + k]), dk);
            sdK[j * D_PAD + k] += dk * scale;
        }
        __syncthreads();
    }
    
    for (int idx = tid; idx < bc * D_CONST; idx += THREADS) {
        int j = idx / D_CONST, k = idx % D_CONST;
        dK_bh[(kj_start + j) * D_CONST + k] = __float2bfloat16(sdK[j * D_PAD + k]);
        dV_bh[(kj_start + j) * D_CONST + k] = __float2bfloat16(sdV[j * D_PAD + k]);
    }
}

void run(
    tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
    tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
    tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int S = (int)Q.size(2);
    float scale = 1.0f / sqrtf((float)D_CONST);
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_ptr = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    float* D_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&D_buf, (size_t)B_CONST * H_CONST * S * sizeof(float)));
    
    {   int total = B_CONST * H_CONST * S, t = 256;
        compute_D_kernel<<<(total+t-1)/t, t, 0, stream>>>(O_ptr, dO_ptr, D_buf, S);
    }
    {   int nqb = (S + BR - 1) / BR;
        size_t sz = (size_t)4*BR*D_PAD*2 + BR*BC_PAD*4 + BR*D_PAD*4 + 2*BR*4;
        CUDA_CHECK(cudaFuncSetAttribute(compute_dQ_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, sz));
        compute_dQ_kernel<<<B_CONST*H_CONST*nqb, THREADS, sz, stream>>>(
            Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_buf, dQ_ptr, S, scale);
    }
    {   int nkb = (S + BC - 1) / BC;
        size_t sz = (size_t)2*BC*D_PAD*2 + 2*BR*D_PAD*2 + BR*BC_PAD*4 + 2*BC*D_PAD*4 + 2*BR*4;
        CUDA_CHECK(cudaFuncSetAttribute(compute_dKV_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, sz));
        compute_dKV_kernel<<<B_CONST*H_CONST*nkb, THREADS, sz, stream>>>(
            Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_buf, dK_ptr, dV_ptr, S, scale);
    }
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(D_buf));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128::run);

}  // namespace mha_bwd_d128