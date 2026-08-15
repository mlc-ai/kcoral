#include <cuda_bf16.h>
#include <cuda_runtime.h>
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

namespace mha_bwd {

constexpr int HEAD_DIM = 128;
constexpr int BQ = 64;
constexpr int BK = 64;
constexpr float SCALE = 0.08838834764831845f; // 1/sqrt(128)

__global__ void compute_D_kernel(const __nv_bfloat16* O, const __nv_bfloat16* dO,
                                  float* D_data, int S, int BH) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = BH * S;
    if (idx >= total) return;
    int s = idx % S;
    int bh = idx / S;
    const __nv_bfloat16* O_ptr = O + (uint64_t)bh * S * HEAD_DIM + s * HEAD_DIM;
    const __nv_bfloat16* dO_ptr = dO + (uint64_t)bh * S * HEAD_DIM + s * HEAD_DIM;
    float sum = 0.0f;
    #pragma unroll 8
    for (int dd = 0; dd < HEAD_DIM; ++dd) {
        sum += __bfloat162float(O_ptr[dd]) * __bfloat162float(dO_ptr[dd]);
    }
    D_data[idx] = sum;
}

__global__ void compute_dQ_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* dO, const float* L, const float* D_data,
    float* dQ_out, int S) {
    
    int bh = blockIdx.x;
    int qi = blockIdx.y;
    int q_start = qi * BQ;
    int tid = threadIdx.x;
    
    extern __shared__ char smem[];
    __nv_bfloat16* Q_smem = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* dO_smem = Q_smem + BQ * HEAD_DIM;
    __nv_bfloat16* K_smem = dO_smem + BQ * HEAD_DIM;
    __nv_bfloat16* V_smem = K_smem + BK * HEAD_DIM;
    float* L_smem = reinterpret_cast<float*>(V_smem + BK * HEAD_DIM);
    float* D_smem = L_smem + BQ;
    
    for (int i = tid; i < BQ * HEAD_DIM; i += 64) {
        int row = i / HEAD_DIM;
        int col = i % HEAD_DIM;
        int q_idx = q_start + row;
        if (q_idx < S) {
            Q_smem[i] = Q[(uint64_t)bh * S * HEAD_DIM + q_idx * HEAD_DIM + col];
            dO_smem[i] = dO[(uint64_t)bh * S * HEAD_DIM + q_idx * HEAD_DIM + col];
        } else {
            Q_smem[i] = __float2bfloat16(0.0f);
            dO_smem[i] = __float2bfloat16(0.0f);
        }
    }
    if (tid < BQ) {
        int q_idx = q_start + tid;
        if (q_idx < S) {
            L_smem[tid] = L[(uint64_t)bh * S + q_idx];
            D_smem[tid] = D_data[(uint64_t)bh * S + q_idx];
        } else {
            L_smem[tid] = 0.0f;
            D_smem[tid] = 0.0f;
        }
    }
    __syncthreads();
    
    float dQ_reg[HEAD_DIM];
    #pragma unroll
    for (int dd = 0; dd < HEAD_DIM; ++dd) dQ_reg[dd] = 0.0f;
    
    int i = tid;
    int q_idx = q_start + i;
    if (q_idx >= S) {
        i = -1;
    }
    
    float L_i = (i >= 0) ? L_smem[i] : 0.0f;
    float D_i = (i >= 0) ? D_smem[i] : 0.0f;
    
    int nk_blocks = (S + BK - 1) / BK;
    for (int kj = 0; kj < nk_blocks; ++kj) {
        int k_start = kj * BK;
        for (int j = tid; j < BK * HEAD_DIM; j += 64) {
            int row = j / HEAD_DIM;
            int col = j % HEAD_DIM;
            int k_idx = k_start + row;
            if (k_idx < S) {
                K_smem[j] = K[(uint64_t)bh * S * HEAD_DIM + k_idx * HEAD_DIM + col];
                V_smem[j] = V[(uint64_t)bh * S * HEAD_DIM + k_idx * HEAD_DIM + col];
            } else {
                K_smem[j] = __float2bfloat16(0.0f);
                V_smem[j] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();
        
        if (i >= 0) {
            for (int j = 0; j < BK; ++j) {
                int k_idx = k_start + j;
                if (k_idx >= S) break;
                if (q_idx < k_idx) continue;
                float S_ij = 0.0f;
                #pragma unroll 8
                for (int dd = 0; dd < HEAD_DIM; ++dd) {
                    S_ij += __bfloat162float(Q_smem[i * HEAD_DIM + dd]) * __bfloat162float(K_smem[j * HEAD_DIM + dd]);
                }
                S_ij *= SCALE;
                float P_ij = expf(S_ij - L_i);
                float dP_ij = 0.0f;
                #pragma unroll 8
                for (int dd = 0; dd < HEAD_DIM; ++dd) {
                    dP_ij += __bfloat162float(dO_smem[i * HEAD_DIM + dd]) * __bfloat162float(V_smem[j * HEAD_DIM + dd]);
                }
                float dS_ij = P_ij * (dP_ij - D_i);
                #pragma unroll 8
                for (int dd = 0; dd < HEAD_DIM; ++dd) {
                    dQ_reg[dd] += dS_ij * __bfloat162float(K_smem[j * HEAD_DIM + dd]) * SCALE;
                }
            }
        }
        __syncthreads();
    }
    
    if (q_idx < S) {
        for (int dd = 0; dd < HEAD_DIM; ++dd) {
            dQ_out[(uint64_t)bh * S * HEAD_DIM + q_idx * HEAD_DIM + dd] = dQ_reg[dd];
        }
    }
}

__global__ void compute_dK_dV_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* dO, const float* L, const float* D_data,
    float* dK_out, float* dV_out, int S) {
    
    int bh = blockIdx.x;
    int kj = blockIdx.y;
    int k_start = kj * BK;
    int tid = threadIdx.x;
    
    extern __shared__ char smem[];
    __nv_bfloat16* K_smem = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* V_smem = K_smem + BK * HEAD_DIM;
    __nv_bfloat16* Q_smem = V_smem + BK * HEAD_DIM;
    __nv_bfloat16* dO_smem = Q_smem + BQ * HEAD_DIM;
    float* dK_smem = reinterpret_cast<float*>(dO_smem + BQ * HEAD_DIM);
    float* dV_smem = dK_smem + BK * HEAD_DIM;
    float* L_smem = dV_smem + BK * HEAD_DIM;
    float* D_smem = L_smem + BQ;
    
    for (int j = tid; j < BK * HEAD_DIM; j += 64) {
        int row = j / HEAD_DIM;
        int col = j % HEAD_DIM;
        int k_idx = k_start + row;
        if (k_idx < S) {
            K_smem[j] = K[(uint64_t)bh * S * HEAD_DIM + k_idx * HEAD_DIM + col];
            V_smem[j] = V[(uint64_t)bh * S * HEAD_DIM + k_idx * HEAD_DIM + col];
        } else {
            K_smem[j] = __float2bfloat16(0.0f);
            V_smem[j] = __float2bfloat16(0.0f);
        }
    }
    for (int j = tid; j < BK * HEAD_DIM; j += 64) {
        dK_smem[j] = 0.0f;
        dV_smem[j] = 0.0f;
    }
    __syncthreads();
    
    int j = tid;
    int k_idx = k_start + j;
    if (k_idx >= S) j = -1;
    
    int nq_blocks = (S + BQ - 1) / BQ;
    for (int qi = 0; qi < nq_blocks; ++qi) {
        int q_start = qi * BQ;
        for (int i = tid; i < BQ * HEAD_DIM; i += 64) {
            int row = i / HEAD_DIM;
            int col = i % HEAD_DIM;
            int q_idx = q_start + row;
            if (q_idx < S) {
                Q_smem[i] = Q[(uint64_t)bh * S * HEAD_DIM + q_idx * HEAD_DIM + col];
                dO_smem[i] = dO[(uint64_t)bh * S * HEAD_DIM + q_idx * HEAD_DIM + col];
            } else {
                Q_smem[i] = __float2bfloat16(0.0f);
                dO_smem[i] = __float2bfloat16(0.0f);
            }
        }
        if (tid < BQ) {
            int q_idx = q_start + tid;
            if (q_idx < S) {
                L_smem[tid] = L[(uint64_t)bh * S + q_idx];
                D_smem[tid] = D_data[(uint64_t)bh * S + q_idx];
            } else {
                L_smem[tid] = 0.0f;
                D_smem[tid] = 0.0f;
            }
        }
        __syncthreads();
        
        if (j >= 0) {
            for (int i = 0; i < BQ; ++i) {
                int q_idx = q_start + i;
                if (q_idx >= S) break;
                if (q_idx < k_idx) continue;
                float L_i = L_smem[i];
                float D_i = D_smem[i];
                float S_ij = 0.0f;
                #pragma unroll 8
                for (int dd = 0; dd < HEAD_DIM; ++dd) {
                    S_ij += __bfloat162float(Q_smem[i * HEAD_DIM + dd]) * __bfloat162float(K_smem[j * HEAD_DIM + dd]);
                }
                S_ij *= SCALE;
                float P_ij = expf(S_ij - L_i);
                float dP_ij = 0.0f;
                #pragma unroll 8
                for (int dd = 0; dd < HEAD_DIM; ++dd) {
                    dP_ij += __bfloat162float(dO_smem[i * HEAD_DIM + dd]) * __bfloat162float(V_smem[j * HEAD_DIM + dd]);
                }
                float dS_ij = P_ij * (dP_ij - D_i);
                #pragma unroll 8
                for (int dd = 0; dd < HEAD_DIM; ++dd) {
                    dK_smem[j * HEAD_DIM + dd] += dS_ij * __bfloat162float(Q_smem[i * HEAD_DIM + dd]) * SCALE;
                }
                #pragma unroll 8
                for (int dd = 0; dd < HEAD_DIM; ++dd) {
                    dV_smem[j * HEAD_DIM + dd] += P_ij * __bfloat162float(dO_smem[i * HEAD_DIM + dd]);
                }
            }
        }
        __syncthreads();
    }
    
    if (k_idx < S) {
        for (int dd = 0; dd < HEAD_DIM; ++dd) {
            dK_out[(uint64_t)bh * S * HEAD_DIM + k_idx * HEAD_DIM + dd] = dK_smem[j * HEAD_DIM + dd];
            dV_out[(uint64_t)bh * S * HEAD_DIM + k_idx * HEAD_DIM + dd] = dV_smem[j * HEAD_DIM + dd];
        }
    }
}

__global__ void convert_to_bf16_kernel(const float* src, __nv_bfloat16* dst, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dst[idx] = __float2bfloat16(src[idx]);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int BH = B * H;
    int total_elems = BH * S * HEAD_DIM;
    int total_rows = BH * S;
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    float *dQ_f, *dK_f, *dV_f, *D_f;
    CUDA_CHECK(cudaMalloc(&dQ_f, total_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dK_f, total_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dV_f, total_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&D_f, total_rows * sizeof(float)));
    
    int smem_dQ = BQ * HEAD_DIM * sizeof(__nv_bfloat16) * 2 + BK * HEAD_DIM * sizeof(__nv_bfloat16) * 2 + BQ * sizeof(float) * 2;
    int smem_dK = BK * HEAD_DIM * sizeof(__nv_bfloat16) * 2 + BQ * HEAD_DIM * sizeof(__nv_bfloat16) * 2 + BK * HEAD_DIM * sizeof(float) * 2 + BQ * sizeof(float) * 2;
    CUDA_CHECK(cudaFuncSetAttribute(compute_dQ_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_dQ));
    CUDA_CHECK(cudaFuncSetAttribute(compute_dK_dV_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_dK));
    
    {
        int threads = 256;
        int blocks = (total_rows + threads - 1) / threads;
        compute_D_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(O.data_ptr()),
            static_cast<const __nv_bfloat16*>(dO.data_ptr()),
            D_f, S, BH);
    }
    
    {
        dim3 grid(BH, (S + BQ - 1) / BQ);
        dim3 block(64);
        compute_dQ_kernel<<<grid, block, smem_dQ, stream>>>(
            static_cast<const __nv_bfloat16*>(Q.data_ptr()),
            static_cast<const __nv_bfloat16*>(K.data_ptr()),
            static_cast<const __nv_bfloat16*>(V.data_ptr()),
            static_cast<const __nv_bfloat16*>(dO.data_ptr()),
            static_cast<const float*>(L.data_ptr()),
            D_f, dQ_f, S);
    }
    
    {
        dim3 grid(BH, (S + BK - 1) / BK);
        dim3 block(64);
        compute_dK_dV_kernel<<<grid, block, smem_dK, stream>>>(
            static_cast<const __nv_bfloat16*>(Q.data_ptr()),
            static_cast<const __nv_bfloat16*>(K.data_ptr()),
            static_cast<const __nv_bfloat16*>(V.data_ptr()),
            static_cast<const __nv_bfloat16*>(dO.data_ptr()),
            static_cast<const float*>(L.data_ptr()),
            D_f, dK_f, dV_f, S);
    }
    
    {
        int threads = 256;
        int blocks = (total_elems + threads - 1) / threads;
        convert_to_bf16_kernel<<<blocks, threads, 0, stream>>>(
            dQ_f, static_cast<__nv_bfloat16*>(dQ.data_ptr()), total_elems);
        convert_to_bf16_kernel<<<blocks, threads, 0, stream>>>(
            dK_f, static_cast<__nv_bfloat16*>(dK.data_ptr()), total_elems);
        convert_to_bf16_kernel<<<blocks, threads, 0, stream>>>(
            dV_f, static_cast<__nv_bfloat16*>(dV.data_ptr()), total_elems);
    }
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    cudaFree(dQ_f);
    cudaFree(dK_f);
    cudaFree(dV_f);
    cudaFree(D_f);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd