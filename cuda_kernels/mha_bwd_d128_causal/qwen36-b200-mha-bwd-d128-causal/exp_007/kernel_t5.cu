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

// ---- Generic GEMM: C[M,N] = A[M,K] @ B[N,K]^T, BF16->FP32, with optional causal mask ----
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
            val = 0.0f;  // Use 0 instead of -inf; softmax will clamp below
        }
        C[row_global * N + col_global] = val;
    }
}

// ---- Softmax: compute P[i,j] = exp(S[i,j] - L[i]) with causal awareness ----
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
        
        // Clamp to prevent numerical issues
        float diff = s_val - lse;
        if (diff < -50.0f) {
            P[idx] = 0.0f;
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

// ---- dO @ V^T @ K for dK: fused ----
template<int BM, int BN, int BK_TILE>
__global__ void compute_dOVT_K_kernel(
    const __nv_bfloat16* __restrict__ dO,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ K,
    float* __restrict__ dK_acc,     // FP32 partial result
    int S, int d) {

    // First compute tmp = dO @ V^T [S,S], then tmp @ K [S,d] -> [S,d]
    // We do it in two stages using shared memory
    
    extern __shared__ char smem[];
    __nv_bfloat16* tmp_smem = (__nv_bfloat16*)smem;
    
    // Stage 1: Compute tmp[i,j] = dO[i,:] @ V[j,:].T  (partial for this thread block)
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_out = blockIdx.x * BM + ty;  // S dim
    int col_out = blockIdx.y * BN + tx;  // d dim

    if (row_out >= S || col_out >= d) return;

    // We need: dK[row,col] = sum_j (tmp[i,j] * K[j,col])
    // where tmp[i,j] = sum_k dO[i,k] * V[j,k]
    // Combined: dK[i,c] = sum_j sum_k dO[i,k] * V[j,k] * K[j,c]
    
    // Reducing over j (S dim) tiled
    float acc = 0.0f;
    
    int num_tiles = (S + BK_TILE - 1) / BK_TILE;
    for (int t = 0; t < num_tiles; t++) {
        int j_start = t * BK_TILE;
        
        // For each tile, we need:
        // row_out: fixed
        // col_out: fixed 
        // j varying
        
        // Load K[j, col_out] for j in [j_start, j_start+BK_TILE)
        __nv_bfloat16* K_smem_row = &tmp_smem[0];
        __nv_bfloat16* V_col_smem = &tmp_smem[BK_TILE];
        __nv_bfloat16* dO_row_smem = &tmp_smem[BK_TILE * 2];
        
        // Load K slice for this block
        if (tx < BK_TILE && ty < BK_TILE) {
            int kj = j_start + tx;
            if (kj < S) {
                K_smem_row[tx] = K[kj * d + col_out];
            } else {
                K_smem_row[tx] = f32_to_bf16(0.0f);
            }
        }
        __syncthreads();
        
        // Inner reduction over k=d dimension
        float tile_acc = 0.0f;
        int d_tiles = (d + BK_TILE - 1) / BK_TILE;
        for (int dt = 0; dt < d_tiles; dt++) {
            int k_start = dt * BK_TILE;
            
            if (tx < BK_TILE) {
                int dj = j_start + tx;
                if (dj < S && k_start + dy < d) {
                    
                }
            }
            
            __syncthreads();
        }
        
        __syncthreads();
    }
    
    // Actually let me simplify: compute in 2 separate passes
    // Pass 1: dO @ V^T -> tmp
    // Pass 2: tmp @ K -> dK_acc
    
    // Simple but less efficient: just accumulate over j inline
    for (int j = 0; j < S; j++) {
        // tmp[i,j] = dO[i,:] . V[j,:]
        float tmp_ij = 0.0f;
        for (int k = 0; k < d; k++) {
            float do_ik = bf16_to_f32(dO[row_out * d + k]);
            float v_jk = bf16_to_f32(V[j * d + k]);
            tmp_ij += do_ik * v_jk;
        }
        float k_jc = bf16_to_f32(K[j * d + col_out]);
        acc += tmp_ij * k_jc;
    }
    
    if (row_out < S && col_out < d) {
        dK_acc[row_out * d + col_out] = acc;
    }
}

// Simpler fused kernel for dK = (dO @ V^T) @ K without intermediate SxS buffer
// Uses 2-stage tiling
template<int BM, int BN, int BK_TILE>
__global__ void compute_dK_fused_kernel(
    const __nv_bfloat16* __restrict__ dO,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ K,
    const float* __restrict__ P,
    __nv_bfloat16* __restrict__ dK_out,
    int S, int d) {

    // dK[j,c] = sum_i dP_ij * Q_ic  (from attention gradient)
    //          + sum_i sum_k P_ik * dO_ik * V_jk (from value gradient)
    //
    // Second term: (P @ dO)[i,:].T @ V[j,:] ... no
    // Actually: dK[j,c] = sum_i dP_ij * Q_ic + sum_i (sum_k P_ik * dO_ik) * V_jc... hmm
    //
    // Let me stick with the simpler separate computation
    // dK[i,c] = sum_j dP_ij * Q_jc
    // And compute dP separately
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.x * BM + ty;
    int col = blockIdx.y * BN + tx;

    if (row >= S || col >= d) return;
    
    extern __shared__ char smem[];
    float* P_smem = (float*)smem;
    __nv_bfloat16* K_smem = reinterpret_cast<__nv_bfloat16*>((char*)smem + BM * BK_TILE * sizeof(float));

    float acc = 0.0f;
    int num_tiles = (S + BK_TILE - 1) / BK_TILE;
    
    for (int t = 0; t < num_tiles; t++) {
        int k_start = t * BK_TILE;

        if (ty < BM && tx < BK_TILE) {
            int kk = k_start + tx;
            if (kk < S) {
                P_smem[ty * BK_TILE + tx] = P[row * S + kk];
            } else {
                P_smem[ty * BK_TILE + tx] = 0.0f;
            }
        }

        if (tx < BN && ty < BK_TILE) {
            int kk = k_start + ty;
            if (kk < S) {
                K_smem[tx * BK_TILE + ty] = K[kk * d + col];
            } else {
                K_smem[tx * BK_TILE + ty] = f32_to_bf16(0.0f);
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK_TILE; k++) {
            acc += P_smem[ty * BK_TILE + k] * bf16_to_f32(K_smem[tx * BK_TILE + k]);
        }

        __syncthreads();
    }

    if (row < S && col < d) {
        // Write directly, adding to existing value
        float old = bf16_to_f32(dK_out[row * d + col]);
        dK_out[row * d + col] = f32_to_bf16(old + acc);
    }
}

// dV = P^T @ dO
template<int BM, int BN, int BK_TILE>
__global__ void compute_dV_kernel(
    const float* __restrict__ P,
    const __nv_bfloat16* __restrict__ dO,
    __nv_bfloat16* __restrict__ dV_out,
    int S, int d) {

    extern __shared__ char ptdo_smem[];
    float* P_smem = (float*)ptdo_smem;
    __nv_bfloat16* dO_smem = reinterpret_cast<__nv_bfloat16*>(
        (char*)ptdo_smem + BM * BK_TILE * sizeof(float));

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_global = blockIdx.x * BM + ty;
    int col_global = blockIdx.y * BN + tx;

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
        float old = bf16_to_f32(dV_out[row_global * d + col_global]);
        dV_out[row_global * d + col_global] = f32_to_bf16(old + acc);
    }
}

// dQ = dP @ K where dP = (dO @ V^T) * P
// Write directly to dQ output
template<int BM, int BN, int BK_TILE>
__global__ void compute_dP_K_kernel(
    const float* __restrict__ dP,
    const __nv_bfloat16* __restrict__ K,
    __nv_bfloat16* __restrict__ dQ_out,
    int S, int d) {

    extern __shared__ char dpk_smem[];
    float* dP_smem = (float*)dpk_smem;
    __nv_bfloat16* K_smem = reinterpret_cast<__nv_bfloat16*>((char*)dpk_smem + BM * BK_TILE * sizeof(float));

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_global = blockIdx.x * BM + ty;
    int col_global = blockIdx.y * BN + tx;

    if (row_global >= S || col_global >= d) return;

    float acc = 0.0f;

    int num_tiles = (S + BK_TILE - 1) / BK_TILE;
    for (int t = 0; t < num_tiles; t++) {
        int k_start = t * BK_TILE;

        if (ty < BM && tx < BK_TILE) {
            int dpk = k_start + tx;
            if (dpk < S) {
                dP_smem[ty * BK_TILE + tx] = dP[row_global * S + dpk];
            } else {
                dP_smem[ty * BK_TILE + tx] = 0.0f;
            }
        }

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
        float old = bf16_to_f32(dQ_out[row_global * d + col_global]);
        dQ_out[row_global * d + col_global] = f32_to_bf16(old + acc);
    }
}

// dK = dP^T @ Q
template<int BM, int BN, int BK_TILE>
__global__ void compute_dK_from_dP_kernel(
    const float* __restrict__ dP,
    const __nv_bfloat16* __restrict__ Q,
    __nv_bfloat16* __restrict__ dK_out,
    int S, int d) {

    extern __shared__ char dptq_smem[];
    float* dP_smem = (float*)dptq_smem;
    __nv_bfloat16* Q_smem = reinterpret_cast<__nv_bfloat16*>((char*)dptq_smem + BM * BK_TILE * sizeof(float));

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_global = blockIdx.x * BM + ty;
    int col_global = blockIdx.y * BN + tx;

    if (row_global >= S || col_global >= d) return;

    float acc = 0.0f;

    int num_tiles = (S + BK_TILE - 1) / BK_TILE;
    for (int t = 0; t < num_tiles; t++) {
        int k_start = t * BK_TILE;

        if (ty < BM && tx < BK_TILE) {
            int dpk = k_start + tx;
            if (dpk < S) {
                dP_smem[ty * BK_TILE + tx] = dP[row_global * S + dpk];
            } else {
                dP_smem[ty * BK_TILE + tx] = 0.0f;
            }
        }

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
        float old = bf16_to_f32(dK_out[row_global * d + col_global]);
        dK_out[row_global * d + col_global] = f32_to_bf16(old + acc);
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

    // Initialize outputs to zero first
    int n_out = B * H * S * d;
    int conv_blks = (n_out + 255) / 256;
    bf16_zero_kernel<<<conv_blks, 256, 0, stream>>>(dQ_d, n_out);
    bf16_zero_kernel<<<conv_blks, 256, 0, stream>>>(dK_d, n_out);
    bf16_zero_kernel<<<conv_blks, 256, 0, stream>>>(dV_d, n_out);
    CUDA_CHECK(cudaGetLastError());

    // Workspace: only need SxS buffers (no Sxd accumulators anymore)
    int64_t el_S2 = (int64_t)S * S;
    size_t ws_bytes = (size_t)(2 * el_S2) * sizeof(float);  // S_P and dP

    float* ws = nullptr;
    CUDA_CHECK(cudaMallocAsync(&ws, ws_bytes, stream));

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

        // Phase 1: S = QK^T/sqrt(d) with causal mask → softmax → P
        gemm_bf16_to_fp32_kernel<BM, BN, BK, true><<<gemm_grid, blk_s, gemm_smem_bytes, stream>>>(
            Q_b, K_b, S_P, S, S, d, inv_std);
        CUDA_CHECK(cudaGetLastError());

        {
            dim3 sb(16, 16);
            dim3 sg((S + 15) / 16, (S + 15) / 16);
            softmax_from_S_and_L_kernel<<<sg, sb, 0, stream>>>(S_P, L_b, S);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Phase 2: dP_raw = dO @ V^T
        compute_dO_VT_kernel<BM, BN, BK><<<gemm_grid, blk_s, gemm_smem_bytes, stream>>>(
            dO_b, V_b, dP_buf, S, d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Phase 3: dP = dP_raw * P
        {
            int n_elem = S * S;
            int ew_blks = (n_elem + 255) / 256;
            elemwise_mul_kernel<<<ew_blks, 256, 0, stream>>>(dP_buf, S_P, dP_buf, n_elem);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Phase 4-6: Directly write to outputs
        compute_dV_kernel<BM, BN, BK><<<mixed_grid, blk_s, mixed_smem_bytes, stream>>>(
            S_P, dO_b, dV_b, S, d);
        CUDA_CHECK(cudaGetLastError());

        compute_dK_from_dP_kernel<BM, BN, BK><<<mixed_grid, blk_s, mixed_smem_bytes, stream>>>(
            dP_buf, Q_b, dK_b, S, d);
        CUDA_CHECK(cudaGetLastError());

        compute_dP_K_kernel<BM, BN, BK><<<mixed_grid, blk_s, mixed_smem_bytes, stream>>>(
            dP_buf, K_b, dQ_b, S, d);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaFreeAsync(ws, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace mha_bwd_impl

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);