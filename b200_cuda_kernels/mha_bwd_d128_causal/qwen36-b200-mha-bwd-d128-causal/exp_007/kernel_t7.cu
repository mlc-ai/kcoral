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

// ---- Generic GEMM: C[M,N] = A[M,K] @ B[N,K]^T, BF16->FP32 ----
template<int BM, int BN, int BK_TILE, bool APPLY_CAUSAL>
__global__ void gemm_bf16_to_fp32_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K, float alpha) {

    extern __shared__ __nv_bfloat16 gemm_smem[];
    __nv_bfloat16* As = gemm_smem;
    __nv_bfloat16* Bs = &gemm_smem[BM * BK_TILE];

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
            val = -1e10f;  // Very negative so exp(val - L) ≈ 0
        }
        C[row_global * N + col_global] = val;
    }
}

// ---- Softmax: P[i,j] = exp(S[i,j] - L[i]) with clamp ----
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
        
        float diff = s_val - lse;
        if (diff < -80.0f) {
            P[idx] = 0.0f;
        } else if (diff > 80.0f) {
            P[idx] = HUGE_VALF;
        } else {
            P[idx] = expf(diff);
        }
    }
}

// ---- dO @ V^T: dP_raw[S,S] ----
template<int BM, int BN, int BK_TILE>
__global__ void compute_dO_VT_kernel(
    const __nv_bfloat16* __restrict__ dO,
    const __nv_bfloat16* __restrict__ V,
    float* __restrict__ out,
    int S, int d) {

    extern __shared__ __nv_bfloat16 dov_smem[];
    __nv_bfloat16* dO_smem = dov_smem;
    __nv_bfloat16* V_smem = &dov_smem[BM * BK_TILE];

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

// ---- Atomic add helper: write FP32 sum into BF16 output via F32 scratch ----
template<int BM, int BN, int BK_TILE>
__global__ void compute_and_write_atomic(
    const float* __restrict__ matA,     // FP32 matrix
    const __nv_bfloat16* __restrict__ matB,  // BF16 matrix  
    __nv_bfloat16* __restrict__ out,
    int M, int N, int K,
    bool transpose_A) {

    extern __shared__ char smem[];
    
    float* A_smem = (float*)smem;
    __nv_bfloat16* B_smem = reinterpret_cast<__nv_bfloat16*>((char*)smem + BM * BK_TILE * sizeof(float));

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_global = blockIdx.x * BM + ty;
    int col_global = blockIdx.y * BN + tx;

    if (row_global >= M || col_global >= N) return;

    float acc = 0.0f;

    int num_tiles = (K + BK_TILE - 1) / BK_TILE;
    for (int t = 0; t < num_tiles; t++) {
        int k_start = t * BK_TILE;

        if (transpose_A) {
            // A^T[k, row] = A[row, k]
            if (ty < BM && tx < BK_TILE) {
                int ak = k_start + tx;
                if (ak < K) {
                    A_smem[ty * BK_TILE + tx] = matA[row_global * K + ak];
                } else {
                    A_smem[ty * BK_TILE + tx] = 0.0f;
                }
            }
        } else {
            if (ty < BM && tx < BK_TILE) {
                int ak = k_start + tx;
                if (row_global < M && ak < K) {
                    A_smem[ty * BK_TILE + tx] = matA[row_global * K + ak];
                } else {
                    A_smem[ty * BK_TILE + tx] = 0.0f;
                }
            }
        }

        if (tx < BN && ty < BK_TILE) {
            int bk = k_start + ty;
            if (bk < K && col_global < N) {
                B_smem[tx * BK_TILE + ty] = matB[bk * N + col_global];
            } else {
                B_smem[tx * BK_TILE + ty] = f32_to_bf16(0.0f);
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK_TILE; k++) {
            acc += A_smem[ty * BK_TILE + k] * bf16_to_f32(B_smem[tx * BK_TILE + k]);
        }

        __syncthreads();
    }

    if (row_global < M && col_global < N) {
        int idx = row_global * N + col_global;
        float old = bf16_to_f32(out[idx]);
        out[idx] = f32_to_bf16(old + acc);
    }
}

// Element-wise multiply: C = A * B
__global__ void elemwise_mul_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                    float* __restrict__ C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] * B[idx];
    }
}

__global__ void bf16_zero_kernel(__nv_bfloat16* ptr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) ptr[idx] = f32_to_bf16(0.0f);
}

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

    // Initialize outputs to zero
    int n_out = B * H * S * d;
    int conv_blks = (n_out + 255) / 256;
    bf16_zero_kernel<<<conv_blks, 256, 0, stream>>>(dQ_d, n_out);
    bf16_zero_kernel<<<conv_blks, 256, 0, stream>>>(dK_d, n_out);
    bf16_zero_kernel<<<conv_blks, 256, 0, stream>>>(dV_d, n_out);
    CUDA_CHECK(cudaGetLastError());

    // Workspace: SxS for scores/P, SxS for dP
    int64_t el_S2 = (int64_t)S * S;
    size_t ws_bytes = (size_t)(2 * el_S2) * sizeof(float);
    float* ws = nullptr;
    CUDA_CHECK(cudaMallocAsync(&ws, ws_bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(ws, 0, ws_bytes, stream));

    float* S_P     = ws;
    float* dP_buf  = ws + (int64_t)el_S2;

    constexpr int BM = 16, BN = 16, BK = 32;
    dim3 blk_s(BN, BM);
    size_t gemm_smem_bytes = (size_t)(BM + BN) * BK * sizeof(__nv_bfloat16);
    size_t mixed_smem_bytes = (size_t)BM * BK * sizeof(float) + (size_t)BN * BK * sizeof(__nv_bfloat16);

    dim3 gemm_grid((S + BM - 1) / BM, (S + BN - 1) / BN);
    dim3 mixed_grid((S + BM - 1) / BM, (d + BN - 1) / BN);

    for (int bh = 0; bh < B * H; bh++) {
        const __nv_bfloat16* Q_b = Q_d + bh * S * d;
        const __nv_bfloat16* K_b = K_d + bh * S * d;
        const __nv_bfloat16* V_b = V_d + bh * S * d;
        const __nv_bfloat16* dO_b = dO_d + bh * S * d;
        const float* L_b = L_d + bh * S;

        __nv_bfloat16* dQ_b = dQ_d + bh * S * d;
        __nv_bfloat16* dK_b = dK_d + bh * S * d;
        __nv_bfloat16* dV_b = dV_d + bh * S * d;

        // Zero workspace
        CUDA_CHECK(cudaMemsetAsync(S_P, 0, (size_t)el_S2 * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(dP_buf, 0, (size_t)el_S2 * sizeof(float), stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Phase 1: S = QK^T/sqrt(d) with causal mask (-1e10 for masked)
        gemm_bf16_to_fp32_kernel<BM, BN, BK, true><<<gemm_grid, blk_s, gemm_smem_bytes, stream>>>(
            Q_b, K_b, S_P, S, S, d, inv_std);
        CUDA_CHECK(cudaGetLastError());

        // Phase 2: Softmax P = exp(S - L) with clamping
        {
            dim3 sb(16, 16);
            dim3 sg((S + 15) / 16, (S + 15) / 16);
            softmax_from_S_and_L_kernel<<<sg, sb, 0, stream>>>(S_P, L_b, S);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Phase 3: dP_raw = dO @ V^T
        compute_dO_VT_kernel<BM, BN, BK><<<gemm_grid, blk_s, gemm_smem_bytes, stream>>>(
            dO_b, V_b, dP_buf, S, d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Phase 4: dP = dP_raw * P
        {
            int n_elem = S * S;
            int ew_blks = (n_elem + 255) / 256;
            elemwise_mul_kernel<<<ew_blks, 256, 0, stream>>>(dP_buf, S_P, dP_buf, n_elem);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Phase 5: dV += P^T @ dO
        compute_and_write_atomic<BM, BN, BK><<<mixed_grid, blk_s, mixed_smem_bytes, stream>>>(
            S_P, dO_b, dV_b, S, d, S, true);  // P^T @ dO
        CUDA_CHECK(cudaGetLastError());

        // Phase 6: dK += dP @ Q  (note: dP^T @ Q for dK)
        compute_and_write_atomic<BM, BN, BK><<<mixed_grid, blk_s, mixed_smem_bytes, stream>>>(
            dP_buf, Q_b, dK_b, S, d, S, true);  // dP^T @ Q
        CUDA_CHECK(cudaGetLastError());

        // Phase 7: dQ += dP @ K
        compute_and_write_atomic<BM, BN, BK><<<mixed_grid, blk_s, mixed_smem_bytes, stream>>>(
            dP_buf, K_b, dQ_b, S, d, S, false);  // dP @ K
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaFreeAsync(ws, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace mha_bwd_impl

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);