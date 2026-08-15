#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;

#define CUDA_CHECK(call) do {                                    \
    cudaError_t _e = (call);                                     \
    if (_e != cudaSuccess) {                                     \
        fprintf(stderr, "CUDA error %s at %s:%d\n",              \
                cudaGetErrorString(_e), __FILE__, __LINE__);     \
        exit(1);                                                 \
    }                                                            \
} while(0)

namespace tvm_ffi_example_cuda {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int D = 128;
constexpr int WARPS = 4;
constexpr int THREADS = WARPS * 32;

constexpr int WM = 16;
constexpr int WN = 16;
constexpr int WK = 16;

__global__ void attention_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S) {

    int bh = blockIdx.y;
    int b = bh / H;
    int h = bh % H;
    int q_block = blockIdx.x;
    int q_start = q_block * BM;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    const int64_t base_offset = (int64_t)(b * H + h) * S * D;
    const float scale = 0.08838834764831845f;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sK = sQ + BM * D;
    __nv_bfloat16* sV = sK + BN * D;
    float* sS = reinterpret_cast<float*>(sV + BN * D);
    float* sO = sS + BM * BN;
    __nv_bfloat16* sP = reinterpret_cast<__nv_bfloat16*>(sO + BM * D);
    float* s_rowmax = reinterpret_cast<float*>(sP + BM * BN);
    float* s_rowsum = s_rowmax + BM;

    // Load Q tile [BM, D]
    for (int i = tid; i < BM * D; i += THREADS) {
        int row = i / D;
        int col = i % D;
        int q_pos = q_start + row;
        sQ[i] = (q_pos < S) ? Q[base_offset + (int64_t)q_pos * D + col]
                            : __float2bfloat16(0.0f);
    }

    // Initialize accumulators
    for (int i = tid; i < BM; i += THREADS) {
        s_rowmax[i] = -INFINITY;
        s_rowsum[i] = 0.0f;
    }
    for (int i = tid; i < BM * D; i += THREADS) {
        sO[i] = 0.0f;
    }
    __syncthreads();

    int max_k = min(S, q_start + BM);
    int num_k_blocks = (max_k + BN - 1) / BN;

    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        int k_start = k_block * BN;

        // Load K, V tiles [BN, D]
        for (int i = tid; i < BN * D; i += THREADS) {
            int row = i / D;
            int col = i % D;
            int k_pos = k_start + row;
            if (k_pos < S) {
                sK[i] = K[base_offset + (int64_t)k_pos * D + col];
                sV[i] = V[base_offset + (int64_t)k_pos * D + col];
            } else {
                sK[i] = __float2bfloat16(0.0f);
                sV[i] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        // S = Q @ K^T * scale using wmma
        // Warp w handles row tile w (rows w*16..w*16+15), all col tiles
        for (int j = 0; j < BN / WN; j++) {
            wmma::fragment<wmma::accumulator, WM, WN, WK, float> s_frag;
            wmma::fill_fragment(s_frag, 0.0f);

            for (int kk = 0; kk < D / WK; kk++) {
                wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::col_major> b_frag;

                wmma::load_matrix_sync(a_frag, sQ + warp_id * WM * D + kk * WK, D);
                wmma::load_matrix_sync(b_frag, sK + j * WN * D + kk * WK, D);
                wmma::mma_sync(s_frag, a_frag, b_frag, s_frag);
            }

            for (int i = 0; i < s_frag.num_elements; i++) {
                s_frag.x[i] *= scale;
            }

            wmma::store_matrix_sync(sS + warp_id * WM * BN + j * WN, s_frag, BN, wmma::mem_row_major);
        }
        __syncthreads();

        // Online softmax: each warp handles 16 rows, 32 lanes cooperate per row
        for (int i = 0; i < WM; i++) {
            int row = warp_id * WM + i;
            int q_pos = q_start + row;
            if (q_pos >= S) continue;

            // Each lane reads 2 elements from sS (BN=64, 32 lanes -> 2 per lane)
            float v0 = -INFINITY, v1 = -INFINITY;
            int j0 = lane_id;
            int j1 = lane_id + 32;
            int kp0 = k_start + j0;
            int kp1 = k_start + j1;

            if (j0 < BN && kp0 <= q_pos && kp0 < S) v0 = sS[row * BN + j0];
            if (j1 < BN && kp1 <= q_pos && kp1 < S) v1 = sS[row * BN + j1];

            // Warp reduce max
            float local_max = fmaxf(v0, v1);
            for (int offset = 16; offset > 0; offset >>= 1)
                local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, offset));

            float old_max = s_rowmax[row];
            float new_max = fmaxf(old_max, local_max);
            float exp_old = (old_max == -INFINITY) ? 0.0f : __expf(old_max - new_max);

            // Compute P and write to sP as BF16
            float p0 = (v0 == -INFINITY) ? 0.0f : __expf(v0 - new_max);
            float p1 = (v1 == -INFINITY) ? 0.0f : __expf(v1 - new_max);
            if (j0 < BN) sP[row * BN + j0] = __float2bfloat16(p0);
            if (j1 < BN) sP[row * BN + j1] = __float2bfloat16(p1);

            // Warp reduce sum
            float local_sum = p0 + p1;
            for (int offset = 16; offset > 0; offset >>= 1)
                local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);

            // Rescale O: 128 elements, 4 per lane
            for (int d = 0; d < 4; d++) {
                sO[row * D + lane_id * 4 + d] *= exp_old;
            }

            if (lane_id == 0) {
                s_rowmax[row] = new_max;
                s_rowsum[row] = s_rowsum[row] * exp_old + local_sum;
            }
        }
        __syncthreads();

        // O += P @ V using wmma
        // Warp w handles row tile w, all D/16=8 col tiles
        for (int j = 0; j < D / WN; j++) {
            wmma::fragment<wmma::accumulator, WM, WN, WK, float> o_frag;
            wmma::fill_fragment(o_frag, 0.0f);

            for (int kk = 0; kk < BN / WK; kk++) {
                wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::row_major> b_frag;

                wmma::load_matrix_sync(a_frag, sP + warp_id * WM * BN + kk * WK, BN);
                wmma::load_matrix_sync(b_frag, sV + kk * WK * D + j * WN, D);
                wmma::mma_sync(o_frag, a_frag, b_frag, o_frag);
            }

            // Load existing O, add, store back
            wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc_frag;
            wmma::load_matrix_sync(acc_frag, sO + warp_id * WM * D + j * WN, D, wmma::mem_row_major);
            for (int i = 0; i < o_frag.num_elements; i++)
                acc_frag.x[i] += o_frag.x[i];
            wmma::store_matrix_sync(sO + warp_id * WM * D + j * WN, acc_frag, D, wmma::mem_row_major);
        }
        __syncthreads();
    }

    // Final normalization and store
    for (int i = tid; i < BM; i += THREADS) {
        int q_pos = q_start + i;
        if (q_pos >= S) continue;
        LSE[(int64_t)(b * H + h) * S + q_pos] =
            s_rowmax[i] + logf(s_rowsum[i] + 1e-30f);
    }

    for (int i = tid; i < BM * D; i += THREADS) {
        int row = i / D;
        int col = i % D;
        int q_pos = q_start + row;
        if (q_pos < S) {
            float val = sO[i] / (s_rowsum[row] + 1e-30f);
            O[base_offset + (int64_t)q_pos * D + col] = __float2bfloat16(val);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K,
         tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    const int B = 4;
    const int H = 48;
    const int S = static_cast<int>(Q.size(2));

    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    dim3 grid((S + BM - 1) / BM, B * H);
    dim3 block(THREADS);

    int smem_size = BM * D * 2        // sQ (bf16)
                  + BN * D * 2        // sK (bf16)
                  + BN * D * 2        // sV (bf16)
                  + BM * BN * 4       // sS (float)
                  + BM * D * 4        // sO (float)
                  + BM * BN * 2       // sP (bf16)
                  + BM * 4            // s_rowmax
                  + BM * 4;           // s_rowsum

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    cudaFuncSetAttribute(
        attention_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    attention_kernel<<<grid, block, smem_size, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data, B, H, S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda