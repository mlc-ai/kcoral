#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
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

using namespace nvcuda;

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
    int warp_row = warp_id / 2; // 0 or 1
    int warp_col = warp_id % 2; // 0 or 1

    extern __shared__ __align__(16) char smem[];
    __nv_bfloat16* Q_smem = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* dO_smem = Q_smem + BQ * HEAD_DIM;
    __nv_bfloat16* K_smem = dO_smem + BQ * HEAD_DIM;
    __nv_bfloat16* V_smem = K_smem + BK * HEAD_DIM;
    float* S_smem = reinterpret_cast<float*>(V_smem + BK * HEAD_DIM);
    float* dP_smem = S_smem + BQ * BK;
    float* dS_smem_f = dP_smem + BQ * BK;
    // Reuse S_smem for dS_smem_bf16 since S is no longer needed after dS computation
    __nv_bfloat16* dS_smem_bf16 = reinterpret_cast<__nv_bfloat16*>(S_smem);
    float* L_smem = reinterpret_cast<float*>(dS_smem_f + BQ * BK);
    float* D_smem = L_smem + BQ;
    float* out_smem = D_smem + BQ;

    for (int i = tid; i < BQ * HEAD_DIM; i += 128) {
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

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag_col;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag_row;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag[2][2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dQ_frag[2][4];

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            wmma::fill_fragment(dQ_frag[i][j], 0.0f);
        }
    }

    int nk_blocks = (S + BK - 1) / BK;
    for (int kj = 0; kj < nk_blocks; ++kj) {
        int k_start = kj * BK;

        for (int i = tid; i < BK * HEAD_DIM; i += 128) {
            int row = i / HEAD_DIM;
            int col = i % HEAD_DIM;
            int k_idx = k_start + row;
            if (k_idx < S) {
                K_smem[i] = K[(uint64_t)bh * S * HEAD_DIM + k_idx * HEAD_DIM + col];
                V_smem[i] = V[(uint64_t)bh * S * HEAD_DIM + k_idx * HEAD_DIM + col];
            } else {
                K_smem[i] = __float2bfloat16(0.0f);
                V_smem[i] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        // S = Q * K^T
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                wmma::fill_fragment(s_frag[i][j], 0.0f);
                for (int k = 0; k < HEAD_DIM; k += 16) {
                    wmma::load_matrix_sync(a_frag, &Q_smem[(warp_row * 32 + i * 16) * HEAD_DIM + k], HEAD_DIM);
                    wmma::load_matrix_sync(b_frag_col, &K_smem[(warp_col * 32 + j * 16) * HEAD_DIM + k], HEAD_DIM);
                    wmma::mma_sync(s_frag[i][j], a_frag, b_frag_col, s_frag[i][j]);
                }
                wmma::store_matrix_sync(&S_smem[(warp_row * 32 + i * 16) * BK + (warp_col * 32 + j * 16)], s_frag[i][j], BK, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // dP = dO * V^T
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                wmma::fill_fragment(s_frag[i][j], 0.0f);
                for (int k = 0; k < HEAD_DIM; k += 16) {
                    wmma::load_matrix_sync(a_frag, &dO_smem[(warp_row * 32 + i * 16) * HEAD_DIM + k], HEAD_DIM);
                    wmma::load_matrix_sync(b_frag_col, &V_smem[(warp_col * 32 + j * 16) * HEAD_DIM + k], HEAD_DIM);
                    wmma::mma_sync(s_frag[i][j], a_frag, b_frag_col, s_frag[i][j]);
                }
                wmma::store_matrix_sync(&dP_smem[(warp_row * 32 + i * 16) * BK + (warp_col * 32 + j * 16)], s_frag[i][j], BK, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // dS = exp(S - L) * (dP - D)
        for (int i = tid; i < BQ * BK; i += 128) {
            int row = i / BK;
            int col = i % BK;
            int q_idx = q_start + row;
            int k_idx = k_start + col;
            if (q_idx < S && k_idx <= q_idx && k_idx < S) {
                float s_val = S_smem[i] * SCALE;
                float p_val = expf(s_val - L_smem[row]);
                float dp_val = dP_smem[i];
                dS_smem_f[i] = p_val * (dp_val - D_smem[row]);
            } else {
                dS_smem_f[i] = 0.0f;
            }
        }
        __syncthreads();

        // Cast dS to bf16 on the fly into dS_smem_bf16
        for (int i = tid; i < BQ * BK; i += 128) {
            dS_smem_bf16[i] = __float2bfloat16(dS_smem_f[i]);
        }
        __syncthreads();

        // dQ += dS * K
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 4; ++j) {
                for (int k = 0; k < BK; k += 16) {
                    wmma::load_matrix_sync(a_frag, &dS_smem_bf16[(warp_row * 32 + i * 16) * BK + k], BK);
                    wmma::load_matrix_sync(b_frag_row, &K_smem[k * HEAD_DIM + warp_col * 64 + j * 16], HEAD_DIM);
                    wmma::mma_sync(dQ_frag[i][j], a_frag, b_frag_row, dQ_frag[i][j]);
                }
            }
        }
        __syncthreads();
    }

    // Store dQ to SMEM, then to global with bounds check
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int f = 0; f < dQ_frag[i][j].num_elements; ++f) {
                dQ_frag[i][j].x[f] *= SCALE;
            }
            wmma::store_matrix_sync(&out_smem[(warp_row * 32 + i * 16) * HEAD_DIM + (warp_col * 64 + j * 16)], dQ_frag[i][j], HEAD_DIM, wmma::mem_row_major);
        }
    }
    __syncthreads();
    for (int i = tid; i < BQ * HEAD_DIM; i += 128) {
        int row = i / HEAD_DIM;
        int col = i % HEAD_DIM;
        int q_idx = q_start + row;
        if (q_idx < S) {
            dQ_out[(uint64_t)bh * S * HEAD_DIM + q_idx * HEAD_DIM + col] = out_smem[i];
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
    int warp_id = tid / 32;
    int warp_row = warp_id / 2;
    int warp_col = warp_id % 2;

    extern __shared__ __align__(16) char smem[];
    __nv_bfloat16* K_smem = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* V_smem = K_smem + BK * HEAD_DIM;
    __nv_bfloat16* Q_smem = V_smem + BK * HEAD_DIM;
    __nv_bfloat16* dO_smem = Q_smem + BQ * HEAD_DIM;
    float* S_smem = reinterpret_cast<float*>(dO_smem + BQ * HEAD_DIM);
    float* dP_smem = S_smem + BK * BQ;
    float* P_smem_f = dP_smem + BK * BQ;
    float* dS_smem_f = P_smem_f + BK * BQ;
    // Reuse S_smem and dP_smem for bf16 versions
    __nv_bfloat16* dS_smem_bf16 = reinterpret_cast<__nv_bfloat16*>(S_smem);
    __nv_bfloat16* P_smem_bf16 = reinterpret_cast<__nv_bfloat16*>(dP_smem);
    float* L_smem = reinterpret_cast<float*>(dS_smem_f + BK * BQ);
    float* D_smem = L_smem + BQ;
    float* out_dK = D_smem + BQ;
    float* out_dV = out_dK + BK * HEAD_DIM;

    for (int i = tid; i < BK * HEAD_DIM; i += 128) {
        int row = i / HEAD_DIM;
        int col = i % HEAD_DIM;
        int k_idx = k_start + row;
        if (k_idx < S) {
            K_smem[i] = K[(uint64_t)bh * S * HEAD_DIM + k_idx * HEAD_DIM + col];
            V_smem[i] = V[(uint64_t)bh * S * HEAD_DIM + k_idx * HEAD_DIM + col];
        } else {
            K_smem[i] = __float2bfloat16(0.0f);
            V_smem[i] = __float2bfloat16(0.0f);
        }
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag_row;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag_col;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag_row;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag[2][2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dK_frag[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dV_frag[2][4];

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            wmma::fill_fragment(dK_frag[i][j], 0.0f);
            wmma::fill_fragment(dV_frag[i][j], 0.0f);
        }
    }

    int nq_blocks = (S + BQ - 1) / BQ;
    for (int qi = 0; qi < nq_blocks; ++qi) {
        int q_start = qi * BQ;

        for (int i = tid; i < BQ * HEAD_DIM; i += 128) {
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

        // S = K * Q^T
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                wmma::fill_fragment(s_frag[i][j], 0.0f);
                for (int k = 0; k < HEAD_DIM; k += 16) {
                    wmma::load_matrix_sync(a_frag_row, &K_smem[(warp_row * 32 + i * 16) * HEAD_DIM + k], HEAD_DIM);
                    wmma::load_matrix_sync(b_frag_col, &Q_smem[(warp_col * 32 + j * 16) * HEAD_DIM + k], HEAD_DIM);
                    wmma::mma_sync(s_frag[i][j], a_frag_row, b_frag_col, s_frag[i][j]);
                }
                wmma::store_matrix_sync(&S_smem[(warp_row * 32 + i * 16) * BQ + (warp_col * 32 + j * 16)], s_frag[i][j], BQ, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // dP = V * dO^T
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                wmma::fill_fragment(s_frag[i][j], 0.0f);
                for (int k = 0; k < HEAD_DIM; k += 16) {
                    wmma::load_matrix_sync(a_frag_row, &V_smem[(warp_row * 32 + i * 16) * HEAD_DIM + k], HEAD_DIM);
                    wmma::load_matrix_sync(b_frag_col, &dO_smem[(warp_col * 32 + j * 16) * HEAD_DIM + k], HEAD_DIM);
                    wmma::mma_sync(s_frag[i][j], a_frag_row, b_frag_col, s_frag[i][j]);
                }
                wmma::store_matrix_sync(&dP_smem[(warp_row * 32 + i * 16) * BQ + (warp_col * 32 + j * 16)], s_frag[i][j], BQ, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // dS = exp(S - L) * (dP - D), P = exp(S - L)
        for (int i = tid; i < BK * BQ; i += 128) {
            int row = i / BQ;
            int col = i % BQ;
            int k_idx = k_start + row;
            int q_idx = q_start + col;
            if (k_idx < S && q_idx < S && q_idx >= k_idx) {
                float s_val = S_smem[i] * SCALE;
                float p_val = expf(s_val - L_smem[col]);
                float dp_val = dP_smem[i];
                P_smem_f[i] = p_val;
                dS_smem_f[i] = p_val * (dp_val - D_smem[col]);
            } else {
                P_smem_f[i] = 0.0f;
                dS_smem_f[i] = 0.0f;
            }
        }
        __syncthreads();

        // Cast to bf16 on the fly
        for (int i = tid; i < BK * BQ; i += 128) {
            dS_smem_bf16[i] = __float2bfloat16(dS_smem_f[i]);
            P_smem_bf16[i] = __float2bfloat16(P_smem_f[i]);
        }
        __syncthreads();

        // dK += dS * Q
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 4; ++j) {
                for (int k = 0; k < BQ; k += 16) {
                    wmma::load_matrix_sync(a_frag_row, &dS_smem_bf16[(warp_row * 32 + i * 16) * BQ + k], BQ);
                    wmma::load_matrix_sync(b_frag_row, &Q_smem[k * HEAD_DIM + warp_col * 64 + j * 16], HEAD_DIM);
                    wmma::mma_sync(dK_frag[i][j], a_frag_row, b_frag_row, dK_frag[i][j]);
                }
            }
        }

        // dV += P * dO
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 4; ++j) {
                for (int k = 0; k < BQ; k += 16) {
                    wmma::load_matrix_sync(a_frag_row, &P_smem_bf16[(warp_row * 32 + i * 16) * BQ + k], BQ);
                    wmma::load_matrix_sync(b_frag_row, &dO_smem[k * HEAD_DIM + warp_col * 64 + j * 16], HEAD_DIM);
                    wmma::mma_sync(dV_frag[i][j], a_frag_row, b_frag_row, dV_frag[i][j]);
                }
            }
        }
        __syncthreads();
    }

    // Store dK, dV
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int f = 0; f < dK_frag[i][j].num_elements; ++f) {
                dK_frag[i][j].x[f] *= SCALE;
            }
            wmma::store_matrix_sync(&out_dK[(warp_row * 32 + i * 16) * HEAD_DIM + (warp_col * 64 + j * 16)], dK_frag[i][j], HEAD_DIM, wmma::mem_row_major);
            wmma::store_matrix_sync(&out_dV[(warp_row * 32 + i * 16) * HEAD_DIM + (warp_col * 64 + j * 16)], dV_frag[i][j], HEAD_DIM, wmma::mem_row_major);
        }
    }
    __syncthreads();
    for (int i = tid; i < BK * HEAD_DIM; i += 128) {
        int row = i / HEAD_DIM;
        int col = i % HEAD_DIM;
        int k_idx = k_start + row;
        if (k_idx < S) {
            dK_out[(uint64_t)bh * S * HEAD_DIM + k_idx * HEAD_DIM + col] = out_dK[i];
            dV_out[(uint64_t)bh * S * HEAD_DIM + k_idx * HEAD_DIM + col] = out_dV[i];
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

    int smem_dQ = (BQ + BK) * HEAD_DIM * sizeof(__nv_bfloat16) * 2 + BQ * BK * sizeof(float) * 3 + BQ * sizeof(float) * 2 + BQ * HEAD_DIM * sizeof(float);
    int smem_dK = (BK + BQ) * HEAD_DIM * sizeof(__nv_bfloat16) * 2 + BK * BQ * sizeof(float) * 4 + BQ * sizeof(float) * 2 + BK * HEAD_DIM * sizeof(float) * 2;
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
        dim3 block(128);
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
        dim3 block(128);
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