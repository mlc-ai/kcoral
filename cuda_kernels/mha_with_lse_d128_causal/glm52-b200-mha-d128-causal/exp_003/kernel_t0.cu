#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;

namespace flash_attn_impl {

constexpr int D = 128;
constexpr int BR = 64;
constexpr int BC = 64;
constexpr int NUM_THREADS = 128;
constexpr int NUM_WARPS = 4;
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int BR_TILES = BR / WMMA_M;  // 4
constexpr int BC_TILES = BC / WMMA_N;  // 4
constexpr int D_TILES = D / WMMA_K;    // 8

// Shared memory layout:
// sQ:        bf16 [BR, D]  = 16384 bytes
// sKV:       bf16 [BC, D]  = 16384 bytes (reused for K and V)
// sS/sP:     fp32 [BR, BC] = 16384 bytes (S as fp32, then P as bf16 in first half)
// sO:        fp32 [BR, D]  = 32768 bytes
// sTmp:      fp32 [BR, D]  = 32768 bytes
// s_row_max: fp32 [BR]     =   256 bytes
// s_row_sum: fp32 [BR]     =   256 bytes
// Total: 115200 bytes
constexpr int SMEM_SIZE =
    BR * D * 2 +      // sQ
    BC * D * 2 +      // sKV
    BR * BC * 4 +     // sS (fp32, reused as sP bf16)
    BR * D * 4 +      // sO
    BR * D * 4 +      // sTmp
    BR * 4 * 2;       // s_row_max + s_row_sum

__global__ void flash_attn_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S)
{
    int bh = blockIdx.x;
    int q_tile = blockIdx.y;
    int q_start = q_tile * BR;

    if (q_start >= S) return;
    int br = min(BR, S - q_start);

    extern __shared__ char smem[];
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sKV = sQ + BR * D;
    float* sS = reinterpret_cast<float*>(sKV + BC * D);
    float* sO = sS + BR * BC;
    float* sTmp = sO + BR * D;
    float* s_row_max = sTmp + BR * D;
    float* s_row_sum = s_row_max + BR;

    const __nv_bfloat16* Q_base = Q + ((int64_t)bh * S * D);
    const __nv_bfloat16* K_base = K + ((int64_t)bh * S * D);
    const __nv_bfloat16* V_base = V + ((int64_t)bh * S * D);
    __nv_bfloat16* O_base = O + ((int64_t)bh * S * D);
    float* LSE_base = LSE + ((int64_t)bh * S);

    const float scale = 0.08838834764831845f;  // 1/sqrt(128)

    // Load Q tile
    for (int i = threadIdx.x; i < BR * D; i += NUM_THREADS) {
        int row = i / D;
        int col = i % D;
        sQ[row * D + col] = (row < br)
            ? Q_base[(int64_t)(q_start + row) * D + col]
            : __float2bfloat16(0.0f);
    }

    // Initialize sO and running stats
    for (int i = threadIdx.x; i < BR * D; i += NUM_THREADS) {
        sO[i] = 0.0f;
    }
    for (int i = threadIdx.x; i < BR; i += NUM_THREADS) {
        s_row_max[i] = -INFINITY;
        s_row_sum[i] = 0.0f;
    }

    __syncthreads();

    int warp_id = threadIdx.x / 32;
    int num_k_tiles = (S + BC - 1) / BC;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        int k_start = kt * BC;
        int bc = min(BC, S - k_start);

        // Causal: skip if all keys are beyond all queries
        if (k_start > q_start + br - 1) break;

        // Load K tile into sKV
        for (int i = threadIdx.x; i < BC * D; i += NUM_THREADS) {
            int row = i / D;
            int col = i % D;
            sKV[row * D + col] = (row < bc)
                ? K_base[(int64_t)(k_start + row) * D + col]
                : __float2bfloat16(0.0f);
        }
        __syncthreads();

        // S = Q @ K^T using wmma (bf16 x bf16 -> fp32)
        // A = Q [BR, D] row-major, B = K^T [D, BC] = K [BC, D] col-major
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_frag[4];
        for (int t = 0; t < 4; t++) wmma::fill_fragment(s_frag[t], 0.0f);

        for (int ki = 0; ki < D_TILES; ki++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;

            for (int t = 0; t < 4; t++) {
                int flat = warp_id * 4 + t;
                int m_tile = flat / BC_TILES;
                int n_tile = flat % BC_TILES;

                wmma::load_matrix_sync(a_frag, sQ + m_tile * WMMA_M * D + ki * WMMA_K, D);
                wmma::load_matrix_sync(b_frag, sKV + n_tile * WMMA_N * D + ki * WMMA_K, D);
                wmma::mma_sync(s_frag[t], a_frag, b_frag, s_frag[t]);
            }
        }

        // Store S to shared memory
        for (int t = 0; t < 4; t++) {
            int flat = warp_id * 4 + t;
            int m_tile = flat / BC_TILES;
            int n_tile = flat % BC_TILES;
            wmma::store_matrix_sync(sS + m_tile * WMMA_M * BC + n_tile * WMMA_N,
                                    s_frag[t], BC, wmma::mem_row_major);
        }
        __syncthreads();

        // Online softmax: each thread handles one row
        if (threadIdx.x < BR) {
            int row = threadIdx.x;
            if (row < br) {
                int q_pos = q_start + row;

                // Apply scale and causal mask, find row max
                float local_max = -INFINITY;
                for (int j = 0; j < BC; j++) {
                    int k_pos = k_start + j;
                    float val = sS[row * BC + j] * scale;
                    if (k_pos > q_pos || k_pos >= S) val = -INFINITY;
                    sS[row * BC + j] = val;
                    if (val > local_max) local_max = val;
                }

                float m_old = s_row_max[row];
                float m_new = fmaxf(m_old, local_max);
                float alpha = (m_new > -INFINITY)
                    ? ((m_old > -INFINITY) ? expf(m_old - m_new) : 0.0f)
                    : 0.0f;

                // Compute P = exp(S - m_new) and row sum
                float local_sum = 0.0f;
                for (int j = 0; j < BC; j++) {
                    float val = sS[row * BC + j];
                    float p = (val > -INFINITY) ? expf(val - m_new) : 0.0f;
                    sS[row * BC + j] = p;
                    local_sum += p;
                }

                // Rescale O
                for (int j = 0; j < D; j++) {
                    sO[row * D + j] *= alpha;
                }

                s_row_max[row] = m_new;
                s_row_sum[row] = s_row_sum[row] * alpha + local_sum;
            }
        }
        __syncthreads();

        // Convert P from fp32 to bf16 in-place (first half of sS buffer)
        __nv_bfloat16* sP = reinterpret_cast<__nv_bfloat16*>(sS);
        for (int i = threadIdx.x; i < BR * BC; i += NUM_THREADS) {
            sP[i] = __float2bfloat16(sS[i]);
        }
        __syncthreads();

        // Load V tile into sKV (reusing K's buffer)
        for (int i = threadIdx.x; i < BC * D; i += NUM_THREADS) {
            int row = i / D;
            int col = i % D;
            sKV[row * D + col] = (row < bc)
                ? V_base[(int64_t)(k_start + row) * D + col]
                : __float2bfloat16(0.0f);
        }
        __syncthreads();

        // tmp = P @ V using wmma (bf16 x bf16 -> fp32)
        // P: [BR, BC] bf16 row-major, V: [BC, D] bf16 row-major
        // 4x8 = 32 output tiles, 4 K-tiles, 8 tiles per warp
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> tmp_frag[8];
        for (int t = 0; t < 8; t++) wmma::fill_fragment(tmp_frag[t], 0.0f);

        for (int ki = 0; ki < BC_TILES; ki++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;

            for (int t = 0; t < 8; t++) {
                int flat = warp_id * 8 + t;
                int m_tile = flat / D_TILES;
                int n_tile = flat % D_TILES;

                wmma::load_matrix_sync(a_frag, sP + m_tile * WMMA_M * BC + ki * WMMA_K, BC);
                wmma::load_matrix_sync(b_frag, sKV + ki * WMMA_K * D + n_tile * WMMA_N, D);
                wmma::mma_sync(tmp_frag[t], a_frag, b_frag, tmp_frag[t]);
            }
        }

        // Store tmp to shared memory
        for (int t = 0; t < 8; t++) {
            int flat = warp_id * 8 + t;
            int m_tile = flat / D_TILES;
            int n_tile = flat % D_TILES;
            wmma::store_matrix_sync(sTmp + m_tile * WMMA_M * D + n_tile * WMMA_N,
                                    tmp_frag[t], D, wmma::mem_row_major);
        }
        __syncthreads();

        // O += tmp
        for (int i = threadIdx.x; i < BR * D; i += NUM_THREADS) {
            sO[i] += sTmp[i];
        }
        __syncthreads();
    }

    // Final normalization: O = O / row_sum, write to global
    for (int i = threadIdx.x; i < BR * D; i += NUM_THREADS) {
        int row = i / D;
        int col = i % D;
        if (row < br) {
            float sum = s_row_sum[row];
            float val = (sum > 0.0f) ? sO[i] / sum : 0.0f;
            O_base[(int64_t)(q_start + row) * D + col] = __float2bfloat16(val);
        }
    }

    // Write LSE = row_max + log(row_sum)
    if (threadIdx.x < BR) {
        int row = threadIdx.x;
        if (row < br) {
            float sum = s_row_sum[row];
            float lse = (sum > 0.0f) ? (s_row_max[row] + logf(sum)) : -INFINITY;
            LSE_base[q_start + row] = lse;
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    int device_id = Q.device().device_id;
    cudaSetDevice(device_id);

    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);

    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    dim3 grid(B * H, (S + BR - 1) / BR);
    dim3 block(NUM_THREADS);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, device_id));

    cudaFuncSetAttribute(flash_attn_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE);

    flash_attn_kernel<<<grid, block, SMEM_SIZE, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data, S);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaStreamSynchronize(stream);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flash_attn_impl::run);

}  // namespace flash_attn_impl