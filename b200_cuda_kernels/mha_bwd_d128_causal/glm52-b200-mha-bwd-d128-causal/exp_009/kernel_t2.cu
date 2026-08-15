#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace attn_bwd {

constexpr int D = 128;
constexpr int TILE = 64;
constexpr int THREADS = 128;
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

__global__ void attn_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    float* __restrict__ dQ_float,
    __nv_bfloat16* __restrict__ dK_out,
    __nv_bfloat16* __restrict__ dV_out,
    int B, int H, int S)
{
    using namespace nvcuda::wmma;

    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;

    int64_t base = (int64_t)(b * H + h) * S * D;
    int64_t lse_base = (int64_t)(b * H + h) * S;

    const __nv_bfloat16* Q_ = Q + base;
    const __nv_bfloat16* K_ = K + base;
    const __nv_bfloat16* V_ = V + base;
    const __nv_bfloat16* O_ = O + base;
    const __nv_bfloat16* dO_ = dO + base;
    const float* L_ = L + lse_base;
    float* dQ_f = dQ_float + base;
    __nv_bfloat16* dK_ = dK_out + base;
    __nv_bfloat16* dV_ = dV_out + base;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    const float scale = 0.0883883476f; // 1/sqrt(128)

    extern __shared__ char smem_raw[];
    float* dQ_acc = (float*)smem_raw;             // [TILE][D] float
    float* dK_acc = dQ_acc + TILE * D;             // [TILE][D] float
    float* dV_acc = dK_acc + TILE * D;             // [TILE][D] float
    float* sS = dV_acc + TILE * D;                 // [TILE][TILE] float
    __nv_bfloat16* sP_bf16 = (__nv_bfloat16*)(sS + TILE * TILE); // [TILE][TILE] bf16
    float* sD = (float*)(sP_bf16 + TILE * TILE);   // [TILE] float
    float* sL = sD + TILE;                         // [TILE] float
    __nv_bfloat16* sQ = (__nv_bfloat16*)(sL + TILE);     // [TILE][D] bf16
    __nv_bfloat16* sK = sQ + TILE * D;             // [TILE][D] bf16
    __nv_bfloat16* sV = sK + TILE * D;             // [TILE][D] bf16
    __nv_bfloat16* sdO = sV + TILE * D;            // [TILE][D] bf16

    for (int kj = 0; kj < S; kj += TILE) {
        int k_len = min(TILE, S - kj);

        // Load K, V tiles (pad with 0 for out-of-bounds)
        for (int idx = tid; idx < TILE * D; idx += THREADS) {
            int row = idx / D, col = idx % D;
            sK[idx] = (row < k_len) ? K_[(kj + row) * D + col] : __float2bfloat16(0.0f);
            sV[idx] = (row < k_len) ? V_[(kj + row) * D + col] : __float2bfloat16(0.0f);
        }
        // Zero dK, dV accumulators
        for (int idx = tid; idx < TILE * D; idx += THREADS) {
            dK_acc[idx] = 0.0f;
            dV_acc[idx] = 0.0f;
        }
        __syncthreads();

        for (int qi = kj; qi < S; qi += TILE) {
            int q_len = min(TILE, S - qi);

            // Load Q, dO tiles (pad with 0)
            for (int idx = tid; idx < TILE * D; idx += THREADS) {
                int row = idx / D, col = idx % D;
                sQ[idx] = (row < q_len) ? Q_[(qi + row) * D + col] : __float2bfloat16(0.0f);
                sdO[idx] = (row < q_len) ? dO_[(qi + row) * D + col] : __float2bfloat16(0.0f);
            }

            // Compute D[row] = rowsum(dO[row] * O[row]) and load L
            for (int row = tid; row < TILE; row += THREADS) {
                if (row < q_len) {
                    float d_val = 0.0f;
                    for (int col = 0; col < D; col++) {
                        d_val += __bfloat162float(sdO[row * D + col]) *
                                 __bfloat162float(O_[(qi + row) * D + col]);
                    }
                    sD[row] = d_val;
                    sL[row] = L_[qi + row];
                } else {
                    sD[row] = 0.0f;
                    sL[row] = 0.0f;
                }
            }
            __syncthreads();

            // Step 1: S = Q @ K^T (wmma: A=row_major Q, B=col_major K^T)
            for (int c = 0; c < TILE / WMMA_N; c++) {
                fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_frag;
                fill_fragment(s_frag, 0.0f);
                for (int kk = 0; kk < D / WMMA_K; kk++) {
                    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a_frag;
                    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> b_frag;
                    load_matrix_sync(a_frag, sQ + warp_id * WMMA_M * D + kk * WMMA_K, D);
                    load_matrix_sync(b_frag, sK + kk * WMMA_K + c * WMMA_N * D, D);
                    mma_sync(s_frag, a_frag, b_frag, s_frag);
                }
                store_matrix_sync(sS + warp_id * WMMA_M * TILE + c * WMMA_N, s_frag, TILE, mem_row_major);
            }
            __syncthreads();

            // Step 2: P = exp(S * scale - L) with causal mask
            for (int idx = tid; idx < TILE * TILE; idx += THREADS) {
                int i = idx / TILE, j = idx % TILE;
                if (i >= q_len || j >= k_len || qi + i < kj + j) {
                    sS[idx] = 0.0f;
                } else {
                    sS[idx] = __expf(sS[idx] * scale - sL[i]);
                }
            }
            __syncthreads();

            // Convert P to bf16
            for (int idx = tid; idx < TILE * TILE; idx += THREADS) {
                sP_bf16[idx] = __float2bfloat16(sS[idx]);
            }
            __syncthreads();

            // Step 3: dV += P^T @ dO (wmma: A=col_major P^T, B=row_major dO)
            for (int c = 0; c < D / WMMA_N; c++) {
                fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> dv_frag;
                load_matrix_sync(dv_frag, dV_acc + warp_id * WMMA_M * D + c * WMMA_N, D, mem_row_major);
                for (int kk = 0; kk < TILE / WMMA_K; kk++) {
                    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> a_frag;
                    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> b_frag;
                    load_matrix_sync(a_frag, sP_bf16 + warp_id * WMMA_M + kk * WMMA_K * TILE, TILE);
                    load_matrix_sync(b_frag, sdO + kk * WMMA_K * D + c * WMMA_N, D);
                    mma_sync(dv_frag, a_frag, b_frag, dv_frag);
                }
                store_matrix_sync(dV_acc + warp_id * WMMA_M * D + c * WMMA_N, dv_frag, D, mem_row_major);
            }
            __syncthreads();

            // Step 4: dP = dO @ V^T (wmma: A=row_major dO, B=col_major V^T)
            for (int c = 0; c < TILE / WMMA_N; c++) {
                fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> dp_frag;
                fill_fragment(dp_frag, 0.0f);
                for (int kk = 0; kk < D / WMMA_K; kk++) {
                    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a_frag;
                    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> b_frag;
                    load_matrix_sync(a_frag, sdO + warp_id * WMMA_M * D + kk * WMMA_K, D);
                    load_matrix_sync(b_frag, sV + kk * WMMA_K + c * WMMA_N * D, D);
                    mma_sync(dp_frag, a_frag, b_frag, dp_frag);
                }
                store_matrix_sync(sS + warp_id * WMMA_M * TILE + c * WMMA_N, dp_frag, TILE, mem_row_major);
            }
            __syncthreads();

            // Step 5: dS = P * (dP - D) * scale, with causal mask
            // Scale is applied here so dQ and dK wmma don't need to rescale accumulated values
            for (int idx = tid; idx < TILE * TILE; idx += THREADS) {
                int i = idx / TILE, j = idx % TILE;
                if (i >= q_len || j >= k_len || qi + i < kj + j) {
                    sS[idx] = 0.0f;
                } else {
                    float p_val = __bfloat162float(sP_bf16[idx]);
                    sS[idx] = p_val * (sS[idx] - sD[i]) * scale;
                }
            }
            __syncthreads();

            // Convert dS to bf16 (reuse sP_bf16 buffer)
            for (int idx = tid; idx < TILE * TILE; idx += THREADS) {
                sP_bf16[idx] = __float2bfloat16(sS[idx]);
            }
            __syncthreads();

            // Step 6: dQ += dS @ K (wmma: A=row_major dS, B=row_major K)
            // Zero dQ_acc first (dQ accumulates via atomicAdd to global)
            for (int idx = tid; idx < TILE * D; idx += THREADS) {
                dQ_acc[idx] = 0.0f;
            }
            __syncthreads();

            for (int c = 0; c < D / WMMA_N; c++) {
                fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> dq_frag;
                load_matrix_sync(dq_frag, dQ_acc + warp_id * WMMA_M * D + c * WMMA_N, D, mem_row_major);
                for (int kk = 0; kk < TILE / WMMA_K; kk++) {
                    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> a_frag;
                    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> b_frag;
                    load_matrix_sync(a_frag, sP_bf16 + warp_id * WMMA_M * TILE + kk * WMMA_K, TILE);
                    load_matrix_sync(b_frag, sK + kk * WMMA_K * D + c * WMMA_N, D);
                    mma_sync(dq_frag, a_frag, b_frag, dq_frag);
                }
                store_matrix_sync(dQ_acc + warp_id * WMMA_M * D + c * WMMA_N, dq_frag, D, mem_row_major);
            }
            __syncthreads();

            // AtomicAdd dQ to global float buffer
            for (int idx = tid; idx < q_len * D; idx += THREADS) {
                atomicAdd(&dQ_f[(qi + idx / D) * D + idx % D], dQ_acc[idx]);
            }

            // Step 7: dK += dS^T @ Q (wmma: A=col_major dS^T, B=row_major Q)
            for (int c = 0; c < D / WMMA_N; c++) {
                fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> dk_frag;
                load_matrix_sync(dk_frag, dK_acc + warp_id * WMMA_M * D + c * WMMA_N, D, mem_row_major);
                for (int kk = 0; kk < TILE / WMMA_K; kk++) {
                    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> a_frag;
                    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> b_frag;
                    load_matrix_sync(a_frag, sP_bf16 + warp_id * WMMA_M + kk * WMMA_K * TILE, TILE);
                    load_matrix_sync(b_frag, sQ + kk * WMMA_K * D + c * WMMA_N, D);
                    mma_sync(dk_frag, a_frag, b_frag, dk_frag);
                }
                store_matrix_sync(dK_acc + warp_id * WMMA_M * D + c * WMMA_N, dk_frag, D, mem_row_major);
            }
            __syncthreads();
        }

        // Write dK, dV for this key tile to global memory
        for (int idx = tid; idx < k_len * D; idx += THREADS) {
            dK_[(kj + idx / D) * D + idx % D] = __float2bfloat16(dK_acc[idx]);
            dV_[(kj + idx / D) * D + idx % D] = __float2bfloat16(dV_acc[idx]);
        }
        __syncthreads();
    }
}

__global__ void convert_dq_kernel(
    const float* __restrict__ dQ_float,
    __nv_bfloat16* __restrict__ dQ,
    int64_t total_elements)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        dQ[idx] = __float2bfloat16(dQ_float[idx]);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV)
{
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int d = Q.size(3);

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

    int64_t total_dq = (int64_t)B * H * S * d;
    float* dQ_float;
    CUDA_CHECK(cudaMalloc(&dQ_float, total_dq * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(dQ_float, 0, total_dq * sizeof(float), stream));

    // Shared memory layout:
    // dQ_acc + dK_acc + dV_acc: 3 * TILE * D * 4 = 98304
    // sS: TILE * TILE * 4 = 16384
    // sP_bf16: TILE * TILE * 2 = 8192
    // sD + sL: 2 * TILE * 4 = 512
    // sQ + sK + sV + sdO: 4 * TILE * D * 2 = 65536
    // Total: 188928 bytes
    size_t smem_size = (size_t)(3 * TILE * D * 4 + TILE * TILE * 4 + TILE * TILE * 2 + 2 * TILE * 4 + 4 * TILE * D * 2);

    CUDA_CHECK(cudaFuncSetAttribute(attn_bwd_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    int blocks = B * H;
    attn_bwd_kernel<<<blocks, THREADS, smem_size, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
        dQ_float, dK_ptr, dV_ptr, B, H, S);
    CUDA_CHECK(cudaGetLastError());

    int convert_threads = 256;
    int64_t convert_blocks = (total_dq + convert_threads - 1) / convert_threads;
    convert_dq_kernel<<<(int)convert_blocks, convert_threads, 0, stream>>>(
        dQ_float, dQ_ptr, total_dq);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(dQ_float));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, attn_bwd::run);

}  // namespace attn_bwd