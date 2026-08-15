#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define CUDA_CHECK(call) \
    do {                                                         \
        cudaError_t e = call;                                    \
        if (e != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error: %s at %s:%d\n",         \
                    cudaGetErrorString(e), __FILE__, __LINE__);   \
            exit(1);                                              \
        }                                                        \
    } while(0)

namespace mha_bwd_impl {

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 f32_to_bf16(float x) {
    return __float2bfloat16(x);
}

__device__ __forceinline__ float fast_exp2f(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float fast_expf(float x) {
    return fast_exp2f(x * 1.4426950408889634f);
}

// ---- Generic GEMM: C[M,N] = A[M,K] @ B[N,K]^T, BF16->FP32, with optional causal mask ----
template<int BM, int BN, int BK_TILE, bool APPLY_CAUSAL>
__global__ void gemm_bf16_to_fp32_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K, float alpha) {

    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* As = smem;
    __nv_bfloat16* Bs = &smem[BM * BK_TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_global = blockIdx.x * BM + ty;
    int col_global = blockIdx.y * BN + tx;

    float acc = 0.0f;

    int num_tiles = (K + BK_TILE - 1) / BK_TILE;
    for (int t = 0; t < num_tiles; t++) {
        int k_start = t * BK_TILE;

        if (ty < BM && tx < BK_TILE) {
            int ar = row_global;
            int ac = k_start + tx;
            As[ty * BK_TILE + tx] = (ar < M && ac < K) ? A[ar * K + ac] : f32_to_bf16(0.0f);
        }

        if (tx < BN && ty < BK_TILE) {
            int br = col_global;
            int bc = k_start + ty;
            Bs[tx * BK_TILE + ty] = (br < N && bc < K) ? B[br * K + bc] : f32_to_bf16(0.0f);
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK_TILE; k++) {
            acc += bf16_to_f32(As[ty * BK_TILE + k]) * bf16_to_f32(Bs[tx * BK_TILE + k]);
        }

        __syncthreads();
    }

    if (row_global < M && col_global < N) {
        float val = acc * alpha;
        if (APPLY_CAUSAL && col_global >= row_global) {
            val = -1e30f;
        }
        C[row_global * N + col_global] = val;
    }
}

// ---- Softmax kernel: P[i,j] = exp(S[i,j] - L[i]), with causal mask already in S ----
__global__ void softmax_from_S_and_L_kernel(
    float* __restrict__ P,
    const float* __restrict__ L,
    int S_dim) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < S_dim && j < S_dim) {
        int idx = i * S_dim + j;
        float s_val = P[idx];
        float lse = L[i];
        P[idx] = fast_expf(s_val - lse);
    }
}

// ---- Element-wise multiply: C = A * B ----
__global__ void elemwise_mul_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                    float* __restrict__ C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] * B[idx];
    }
}

// ---- dO @ V^T: dP_raw[dO_dim, V_dim] ----
template<int BM, int BN, int BK_TILE>
__global__ void compute_dO_VT_kernel(
    const __nv_bfloat16* __restrict__ dO,
    const __nv_bfloat16* __restrict__ V,
    float* __restrict__ out,
    int S, int d) {

    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* dO_smem = smem;
    __nv_bfloat16* V_smem = &smem[BM * BK_TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_global = blockIdx.x * BM + ty;
    int col_global = blockIdx.y * BN + tx;

    float acc = 0.0f;

    int num_tiles = (d + BK_TILE - 1) / BK_TILE;
    for (int t = 0; t < num_tiles; t++) {
        int k_start = t * BK_TILE;

        if (ty < BM && tx < BK_TILE) {
            int dr = row_global;
            int dk = k_start + tx;
            dO_smem[ty * BK_TILE + tx] = (dr < S && dk < d) ? dO[dr * d + dk] : f32_to_bf16(0.0f);
        }

        if (tx < BN && ty < BK_TILE) {
            int vr = col_global;
            int vk = k_start + ty;
            V_smem[tx * BK_TILE + ty] = (vr < S && vk < d) ? V[vr * d + vk] : f32_to_bf16(0.0f);
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK_TILE; k++) {
            acc += bf16_to_f32(dO_smem[ty * BK_TILE + k]) * bf16_to_f32(V_smem[tx * BK_TILE + k]);
        }

        __syncthreads();
    }

    if (row_global < S && col_global < S) {
        out[row_global * S + col_global] = acc;
    }
}

// ---- P^T @ dO for dV ----
template<int BM, int BN, int BK_TILE>
__global__ void compute_PT_dO_kernel(
    const float* __restrict__ P,
    const __nv_bfloat16* __restrict__ dO,
    float* __restrict__ out,
    int S, int d) {

    extern __shared__ __nv_bfloat16 smem[];
    float* P_smem = (float*)smem;
    __nv_bfloat16* dO_smem = reinterpret_cast<__nv_bfloat16*>(
        (char*)smem + BM * BK_TILE * sizeof(float));

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_global = blockIdx.x * BM + ty;  // S dimension (output row)
    int col_global = blockIdx.y * BN + tx;  // d dimension (output col)

    if (row_global >= S || col_global >= d) return;

    float acc = 0.0f;

    int num_tiles = (S + BK_TILE - 1) / BK_TILE;
    for (int t = 0; t < num_tiles; t++) {
        int k_start = t * BK_TILE;

        // P^T[k, row] = P[row, k]
        if (ty < BM && tx < BK_TILE) {
            int pk = k_start + tx;
            if (pk < S) {
                P_smem[ty * BK_TILE + tx] = P[row_global * S + pk];
            } else {
                P_smem[ty * BK_TILE + tx] = 0.0f;
            }
        }

        // dO[k, col]
        if (tx < BN && ty < BK_TILE) {
            int dok = k_start + ty;
            if (dok < S) {
                dO_smem[tx * BK_TILE + ty] = dO[dok * d + col_global];
            } else {
                dO_smem[tx * BK_TILE + ty] = f32_to_bf16(0.0f);
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK_TILE; k++) {
            acc += P_smem[ty * BK_TILE + k] * bf16_to_f32(dO_smem[tx * BK_TILE + k]);
        }

        __syncthreads();
    }

    if (row_global < S && col_global < d) {
        out[row_global * d + col_global] = acc;
    }
}

// ---- dP^T @ Q for dK ----
template<int BM, int BN, int BK_TILE>
__global__ void compute_dPT_Q_kernel(
    const float* __restrict__ dP,
    const __nv_bfloat16* __restrict__ Q,
    float* __restrict__ out,
    int S, int d) {

    extern __shared__ char smem[];
    float* dP_smem = (float*)smem;
    __nv_bfloat16* Q_smem = reinterpret_cast<__nv_bfloat16*>((char*)smem + BM * BK_TILE * sizeof(float));

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_global = blockIdx.x * BM + ty;  // S dimension
    int col_global = blockIdx.y * BN + tx;  // d dimension

    if (row_global >= S || col_global >= d) return;

    float acc = 0.0f;

    int num_tiles = (S + BK_TILE - 1) / BK_TILE;
    for (int t = 0; t < num_tiles; t++) {
        int k_start = t * BK_TILE;

        // dP^T[k, row] = dP[row, k]
        if (ty < BM && tx < BK_TILE) {
            int dpk = k_start + tx;
            if (dpk < S) {
                dP_smem[ty * BK_TILE + tx] = dP[row_global * S + dpk];
            } else {
                dP_smem[ty * BK_TILE + tx] = 0.0f;
            }
        }

        // Q[k, col]
        if (tx < BN && ty < BK_TILE) {
            int qk = k_start + ty;
            if (qk < S) {
                Q_smem[tx * BK_TILE + ty] = Q[qk * d + col_global];
            } else {
                Q_smem[tx * BK_TILE + ty] = f32_to_bf16(0.0f);
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK_TILE; k++) {
            acc += dP_smem[ty * BK_TILE + k] * bf16_to_f32(Q_smem[tx * BK_TILE + k]);
        }

        __syncthreads();
    }

    if (row_global < S && col_global < d) {
        out[row_global * d + col_global] = acc;
    }
}

// ---- dP @ K for dQ ----
template<int BM, int BN, int BK_TILE>
__global__ void compute_dP_K_kernel(
    const float* __restrict__ dP,
    const __nv_bfloat16* __restrict__ K,
    float* __restrict__ out,
    int S, int d) {

    extern __shared__ char smem[];
    float* dP_smem = (float*)smem;
    __nv_bfloat16* K_smem = reinterpret_cast<__nv_bfloat16*>((char*)smem + BM * BK_TILE * sizeof(float));

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_global = blockIdx.x * BM + ty;  // S dimension
    int col_global = blockIdx.y * BN + tx;  // d dimension

    if (row_global >= S || col_global >= d) return;

    float acc = 0.0f;

    int num_tiles = (S + BK_TILE - 1) / BK_TILE;
    for (int t = 0; t < num_tiles; t++) {
        int k_start = t * BK_TILE;

        // dP[row, k]
        if (ty < BM && tx < BK_TILE) {
            int dpk = k_start + tx;
            if (dpk < S) {
                dP_smem[ty * BK_TILE + tx] = dP[row_global * S + dpk];
            } else {
                dP_smem[ty * BK_TILE + tx] = 0.0f;
            }
        }

        // K[k, col]
        if (tx < BN && ty < BK_TILE) {
            int kk = k_start + ty;
            if (kk < S) {
                K_smem[tx * BK_TILE + ty] = K[kk * d + col_global];
            } else {
                K_smem[tx * BK_TILE + ty] = f32_to_bf16(0.0f);
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK_TILE; k++) {
            acc += dP_smem[ty * BK_TILE + k] * bf16_to_f32(K_smem[tx * BK_TILE + k]);
        }

        __syncthreads();
    }

    if (row_global < S && col_global < d) {
        out[row_global * d + col_global] = acc;
    }
}

// ---- Zero fill and copy helpers ----
__global__ void bf16_zero_kernel(__nv_bfloat16* ptr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) ptr[idx] = f32_to_bf16(0.0f);
}

__global__ void fp32_add_to_bf16_kernel(const float* __restrict__ acc, __nv_bfloat16* __restrict__ dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = bf16_to_f32(dst[idx]) + acc[idx];
        dst[idx] = f32_to_bf16(v);
    }
}

// ================================================================
//                        HOST-SIDE RUN
// ================================================================

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {

    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);
    int d = (int)Q.size(3);

    const __nv_bfloat16* Q_d = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_d = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_d = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* dO_d = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_d = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_d = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_d = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_d = static_cast<__nv_bfloat16*>(dV.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    float inv_std = 1.0f / sqrtf((float)d);

    // Workspace allocations
    size_t sz_S2 = (size_t)S * S * sizeof(float);
    size_t Sz_d = (size_t)S * d * sizeof(float);

    size_t ws_size = B * H * (sz_S2 + sz_S2 + Sz_d + Sz_d + Sz_d);
    float* ws = nullptr;
    CUDA_CHECK(cudaMallocAsync(&ws, ws_size, stream));
    CUDA_CHECK(cudaMemsetAsync(ws, 0, ws_size, stream));

    float* S_score = ws;                 // [BH, S*S]
    float* dP_buf = S_score + B * H * sz_S2;  // [BH, S*S]
    float* dV_acc = dP_buf + B * H * sz_S2;   // [BH, S*d]
    float* dK_acc = dV_acc + B * H * Sz_d;    // [BH, S*d]
    float* dQ_acc = dK_acc + B * H * Sz_d;    // [BH, S*d]

    constexpr int BM = 16, BN = 16, BK = 32;
    dim3 blk_s(BN, BM);
    size_t gemm_smem = (size_t)(BM + BN) * BK * sizeof(__nv_bfloat16);

    // Step 1: S = Q @ K^T / sqrt(d), apply causal mask
    for (int bh = 0; bh < B * H; bh++) {
        const __nv_bfloat16* Q_b = Q_d + bh * S * d;
        const __nv_bfloat16* K_b = K_d + bh * S * d;
        float* S_b = S_score + bh * S * S;

        dim3 grid((S + BM - 1) / BM, (S + BN - 1) / BN);
        gemm_bf16_to_fp32_kernel<BM, BN, BK, true><<<grid, blk_s, gemm_smem, stream>>>(
            Q_b, K_b, S_b, S, S, d, inv_std);
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 2: Softmax P = exp(S - L), L is [B, H, S]
    dim3 softmax_blk(16, 16);
    dim3 softmax_grid((S + 15) / 16, (S + 15) / 16);
    for (int bh = 0; bh < B * H; bh++) {
        float* P_b = S_score + bh * S * S;
        const float* L_b = L_d + bh * S;
        softmax_from_S_and_L_kernel<<<softmax_grid, softmax_blk, 0, stream>>>(P_b, L_b, S);
        CUDA_CHECK(cudaGetLastError());
    }

    // S_score now holds P

    // Step 3: dP_raw = dO @ V^T
    dim3 dOVT_grid((S + BM - 1) / BM, (S + BN - 1) / BN);
    for (int bh = 0; bh < B * H; bh++) {
        const __nv_bfloat16* dO_b = dO_d + bh * S * d;
        const __nv_bfloat16* V_b = V_d + bh * S * d;
        float* dP_b = dP_buf + bh * S * S;
        compute_dO_VT_kernel<BM, BN, BK><<<dOVT_grid, blk_s, gemm_smem, stream>>>(
            dO_b, V_b, dP_b, S, d);
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 4: dP = dP_raw * P (element-wise)
    int n_elem_S2 = B * H * S * S;
    int ew_blks = (n_elem_S2 + 255) / 256;
    elemwise_mul_kernel<<<ew_blks, 256, 0, stream>>>(dP_buf, S_score, dP_buf, n_elem_S2);
    CUDA_CHECK(cudaGetLastError());

    // Step 5: dV = P^T @ dO
    dim3 PtDO_grid((S + BM - 1) / BM, (d + BN - 1) / BN);
    size_t PtDO_smem = (size_t)BM * BK * sizeof(float) + (size_t)BN * BK * sizeof(__nv_bfloat16);
    for (int bh = 0; bh < B * H; bh++) {
        const float* P_b = S_score + bh * S * S;
        const __nv_bfloat16* dO_b = dO_d + bh * S * d;
        float* dV_b = dV_acc + bh * S * d;
        compute_PT_dO_kernel<BM, BN, BK><<<PtDO_grid, blk_s, PtDO_smem, stream>>>(
            P_b, dO_b, dV_b, S, d);
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 6: dK = dP^T @ Q
    size_t dPTQ_smem = (size_t)BM * BK * sizeof(float) + (size_t)BN * BK * sizeof(__nv_bfloat16);
    for (int bh = 0; bh < B * H; bh++) {
        const float* dP_b = dP_buf + bh * S * S;
        const __nv_bfloat16* Q_b = Q_d + bh * S * d;
        float* dK_b = dK_acc + bh * S * d;
        compute_dPT_Q_kernel<BM, BN, BK><<<PtDO_grid, blk_s, dPTQ_smem, stream>>>(
            dP_b, Q_b, dK_b, S, d);
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 7: dQ = dP @ K
    for (int bh = 0; bh < B * H; bh++) {
        const float* dP_b = dP_buf + bh * S * S;
        const __nv_bfloat16* K_b = K_d + bh * S * d;
        float* dQ_b = dQ_acc + bh * S * d;
        compute_dP_K_kernel<BM, BN, BK><<<PtDO_grid, blk_s, dPTQ_smem, stream>>>(
            dP_b, K_b, dQ_b, S, d);
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 8: Zero outputs and add accumulators
    int n_elem_Sd = B * H * S * d;
    int conv_blks = (n_elem_Sd + 255) / 256;

    bf16_zero_kernel<<<conv_blks, 256, 0, stream>>>(dQ_d, n_elem_Sd);
    bf16_zero_kernel<<<conv_blks, 256, 0, stream>>>(dK_d, n_elem_Sd);
    bf16_zero_kernel<<<conv_blks, 256, 0, stream>>>(dV_d, n_elem_Sd);
    CUDA_CHECK(cudaGetLastError());

    fp32_add_to_bf16_kernel<<<conv_blks, 256, 0, stream>>>(dQ_acc, dQ_d, n_elem_Sd);
    fp32_add_to_bf16_kernel<<<conv_blks, 256, 0, stream>>>(dK_acc, dK_d, n_elem_Sd);
    fp32_add_to_bf16_kernel<<<conv_blks, 256, 0, stream>>>(dV_acc, dV_d, n_elem_Sd);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFreeAsync(ws, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace mha_bwd_impl

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);