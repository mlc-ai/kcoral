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
constexpr float SCALE = 0.08838834764831845f;
constexpr int WARPS = 4;
constexpr int THREADS = 128;

// dQ kernel config
constexpr int BQ = 32;
constexpr int BK = 64;
constexpr int ROWS_PER_WARP_DQ = BQ / WARPS; // 8

// dK/dV kernel config
constexpr int BK_DKV = 16;
constexpr int BQ_DKV = 64;
constexpr int ROWS_PER_WARP_DKV = BK_DKV / WARPS; // 4

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
    #pragma unroll
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
    int warp_id = tid / 32;
    int lane = tid % 32;

    extern __shared__ char smem[];
    __nv_bfloat16* Q_smem = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* dO_smem = Q_smem + BQ * HEAD_DIM;
    __nv_bfloat16* K_smem = dO_smem + BQ * HEAD_DIM;
    __nv_bfloat16* V_smem = K_smem + BK * HEAD_DIM;
    float* L_smem = reinterpret_cast<float*>(V_smem + BK * HEAD_DIM);
    float* D_smem = L_smem + BQ;

    // Load Q, dO, L, D
    for (int i = tid; i < BQ * HEAD_DIM; i += THREADS) {
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

    // Load Q, dO into registers for assigned rows (4 dims per thread)
    float Q_reg[ROWS_PER_WARP_DQ][4];
    float dO_reg[ROWS_PER_WARP_DQ][4];
    float dQ_reg[ROWS_PER_WARP_DQ][4];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP_DQ; ++r) {
        int local_row = warp_id * ROWS_PER_WARP_DQ + r;
        #pragma unroll
        for (int d = 0; d < 4; ++d) {
            int col = lane * 4 + d;
            Q_reg[r][d] = __bfloat162float(Q_smem[local_row * HEAD_DIM + col]);
            dO_reg[r][d] = __bfloat162float(dO_smem[local_row * HEAD_DIM + col]);
            dQ_reg[r][d] = 0.0f;
        }
    }

    int nk_blocks = (S + BK - 1) / BK;
    for (int kj = 0; kj < nk_blocks; ++kj) {
        int k_start = kj * BK;

        // Load K, V
        for (int j = tid; j < BK * HEAD_DIM; j += THREADS) {
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

        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP_DQ; ++r) {
            int local_row = warp_id * ROWS_PER_WARP_DQ + r;
            int q_idx = q_start + local_row;
            if (q_idx >= S) continue;

            float L_i = L_smem[local_row];
            float D_i = D_smem[local_row];

            for (int j = 0; j < BK; ++j) {
                int k_idx = k_start + j;
                if (k_idx > q_idx) break;
                if (k_idx >= S) break;

                float S_val = 0.0f;
                #pragma unroll
                for (int d = 0; d < 4; ++d) {
                    int col = lane * 4 + d;
                    S_val += Q_reg[r][d] * __bfloat162float(K_smem[j * HEAD_DIM + col]);
                }
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    S_val += __shfl_xor_sync(0xFFFFFFFF, S_val, offset);
                S_val *= SCALE;

                float P_val = __expf(S_val - L_i);

                float dP_val = 0.0f;
                #pragma unroll
                for (int d = 0; d < 4; ++d) {
                    int col = lane * 4 + d;
                    dP_val += dO_reg[r][d] * __bfloat162float(V_smem[j * HEAD_DIM + col]);
                }
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    dP_val += __shfl_xor_sync(0xFFFFFFFF, dP_val, offset);

                float dS_val = P_val * (dP_val - D_i);

                #pragma unroll
                for (int d = 0; d < 4; ++d) {
                    int col = lane * 4 + d;
                    dQ_reg[r][d] += dS_val * __bfloat162float(K_smem[j * HEAD_DIM + col]) * SCALE;
                }
            }
        }
        __syncthreads();
    }

    // Write dQ
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP_DQ; ++r) {
        int local_row = warp_id * ROWS_PER_WARP_DQ + r;
        int q_idx = q_start + local_row;
        if (q_idx < S) {
            #pragma unroll
            for (int d = 0; d < 4; ++d) {
                int col = lane * 4 + d;
                dQ_out[(uint64_t)bh * S * HEAD_DIM + q_idx * HEAD_DIM + col] = dQ_reg[r][d];
            }
        }
    }
}

__global__ void compute_dK_dV_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* dO, const float* L, const float* D_data,
    float* dK_out, float* dV_out, int S) {

    int bh = blockIdx.x;
    int kj = blockIdx.y;
    int k_start = kj * BK_DKV;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;

    extern __shared__ char smem[];
    __nv_bfloat16* Q_smem = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* dO_smem = Q_smem + BQ_DKV * HEAD_DIM;
    float* L_smem = reinterpret_cast<float*>(dO_smem + BQ_DKV * HEAD_DIM);
    float* D_smem = L_smem + BQ_DKV;

    // Load K, V into registers for assigned key rows
    float K_reg[ROWS_PER_WARP_DKV][4];
    float V_reg[ROWS_PER_WARP_DKV][4];
    float dK_reg[ROWS_PER_WARP_DKV][4];
    float dV_reg[ROWS_PER_WARP_DKV][4];
    int k_idx_arr[ROWS_PER_WARP_DKV];
    bool k_valid[ROWS_PER_WARP_DKV];

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP_DKV; ++r) {
        int local_k = warp_id * ROWS_PER_WARP_DKV + r;
        int k_idx = k_start + local_k;
        k_idx_arr[r] = k_idx;
        k_valid[r] = (k_idx < S);
        #pragma unroll
        for (int d = 0; d < 4; ++d) {
            int col = lane * 4 + d;
            if (k_idx < S) {
                K_reg[r][d] = __bfloat162float(K[(uint64_t)bh * S * HEAD_DIM + k_idx * HEAD_DIM + col]);
                V_reg[r][d] = __bfloat162float(V[(uint64_t)bh * S * HEAD_DIM + k_idx * HEAD_DIM + col]);
            } else {
                K_reg[r][d] = 0.0f;
                V_reg[r][d] = 0.0f;
            }
            dK_reg[r][d] = 0.0f;
            dV_reg[r][d] = 0.0f;
        }
    }

    int nq_blocks = (S + BQ_DKV - 1) / BQ_DKV;
    for (int qi = 0; qi < nq_blocks; ++qi) {
        int q_start = qi * BQ_DKV;

        // Load Q, dO, L, D
        for (int i = tid; i < BQ_DKV * HEAD_DIM; i += THREADS) {
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
        if (tid < BQ_DKV) {
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

        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP_DKV; ++r) {
            if (!k_valid[r]) continue;
            int k_idx = k_idx_arr[r];

            for (int i = 0; i < BQ_DKV; ++i) {
                int q_idx = q_start + i;
                if (q_idx >= S) break;
                if (q_idx < k_idx) continue;

                float L_i = L_smem[i];
                float D_i = D_smem[i];

                float S_val = 0.0f;
                #pragma unroll
                for (int d = 0; d < 4; ++d) {
                    int col = lane * 4 + d;
                    S_val += K_reg[r][d] * __bfloat162float(Q_smem[i * HEAD_DIM + col]);
                }
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    S_val += __shfl_xor_sync(0xFFFFFFFF, S_val, offset);
                S_val *= SCALE;

                float P_val = __expf(S_val - L_i);

                float dP_val = 0.0f;
                #pragma unroll
                for (int d = 0; d < 4; ++d) {
                    int col = lane * 4 + d;
                    dP_val += V_reg[r][d] * __bfloat162float(dO_smem[i * HEAD_DIM + col]);
                }
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    dP_val += __shfl_xor_sync(0xFFFFFFFF, dP_val, offset);

                float dS_val = P_val * (dP_val - D_i);

                #pragma unroll
                for (int d = 0; d < 4; ++d) {
                    int col = lane * 4 + d;
                    dK_reg[r][d] += dS_val * __bfloat162float(Q_smem[i * HEAD_DIM + col]) * SCALE;
                    dV_reg[r][d] += P_val * __bfloat162float(dO_smem[i * HEAD_DIM + col]);
                }
            }
        }
        __syncthreads();
    }

    // Write dK, dV
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP_DKV; ++r) {
        if (!k_valid[r]) continue;
        int k_idx = k_idx_arr[r];
        #pragma unroll
        for (int d = 0; d < 4; ++d) {
            int col = lane * 4 + d;
            dK_out[(uint64_t)bh * S * HEAD_DIM + k_idx * HEAD_DIM + col] = dK_reg[r][d];
            dV_out[(uint64_t)bh * S * HEAD_DIM + k_idx * HEAD_DIM + col] = dV_reg[r][d];
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

    int smem_dQ = (BQ + BK) * HEAD_DIM * sizeof(__nv_bfloat16) * 2 + BQ * sizeof(float) * 2;
    int smem_dK = BQ_DKV * HEAD_DIM * sizeof(__nv_bfloat16) * 2 + BQ_DKV * sizeof(float) * 2;
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
        dim3 block(THREADS);
        compute_dQ_kernel<<<grid, block, smem_dQ, stream>>>(
            static_cast<const __nv_bfloat16*>(Q.data_ptr()),
            static_cast<const __nv_bfloat16*>(K.data_ptr()),
            static_cast<const __nv_bfloat16*>(V.data_ptr()),
            static_cast<const __nv_bfloat16*>(dO.data_ptr()),
            static_cast<const float*>(L.data_ptr()),
            D_f, dQ_f, S);
    }

    {
        dim3 grid(BH, (S + BK_DKV - 1) / BK_DKV);
        dim3 block(THREADS);
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