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

constexpr int BM = 128;
constexpr int BN = 64;
constexpr int D = 128;
constexpr int THREADS = 256;
constexpr int WARPS = 8;
constexpr int WM = 16;
constexpr int WN = 16;
constexpr int WK = 16;
constexpr int D_TILES = D / WN;  // 8

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
    __nv_bfloat16* sP = reinterpret_cast<__nv_bfloat16*>(sS + BM * BN);
    float* s_rowmax = reinterpret_cast<float*>(sP + BM * BN);
    float* s_rowsum = s_rowmax + BM;

    // Load Q tile
    #pragma unroll
    for (int i = tid; i < BM * D / 8; i += THREADS) {
        int row = i / (D / 8);
        int col_int4 = i % (D / 8);
        int q_pos = q_start + row;
        if (q_pos < S) {
            reinterpret_cast<int4*>(sQ)[i] = reinterpret_cast<const int4*>(
                Q + base_offset + (int64_t)q_pos * D)[col_int4];
        } else {
            reinterpret_cast<int4*>(sQ)[i] = make_int4(0, 0, 0, 0);
        }
    }

    // Initialize rowmax, rowsum
    for (int i = tid; i < BM; i += THREADS) {
        s_rowmax[i] = -INFINITY;
        s_rowsum[i] = 0.0f;
    }

    // O accumulators in registers - each warp has 8 col tiles
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> o_frag[D_TILES];
    #pragma unroll
    for (int tn = 0; tn < D_TILES; tn++)
        wmma::fill_fragment(o_frag[tn], 0.0f);

    __syncthreads();

    int max_k = min(S, q_start + BM);
    int num_k_blocks = (max_k + BN - 1) / BN;

    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        int k_start = k_block * BN;

        // Load K, V tiles
        #pragma unroll
        for (int i = tid; i < BN * D / 8; i += THREADS) {
            int row = i / (D / 8);
            int col_int4 = i % (D / 8);
            int k_pos = k_start + row;
            if (k_pos < S) {
                reinterpret_cast<int4*>(sK)[i] = reinterpret_cast<const int4*>(
                    K + base_offset + (int64_t)k_pos * D)[col_int4];
                reinterpret_cast<int4*>(sV)[i] = reinterpret_cast<const int4*>(
                    V + base_offset + (int64_t)k_pos * D)[col_int4];
            } else {
                reinterpret_cast<int4*>(sK)[i] = make_int4(0, 0, 0, 0);
                reinterpret_cast<int4*>(sV)[i] = make_int4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        // S = Q @ K^T * scale using wmma
        // Each warp handles 1 row tile (warp_id * 16)
        {
            int m = warp_id;
            #pragma unroll
            for (int tile_n = 0; tile_n < BN / WN; tile_n++) {
                wmma::fragment<wmma::accumulator, WM, WN, WK, float> s_frag;
                wmma::fill_fragment(s_frag, 0.0f);

                #pragma unroll
                for (int kk = 0; kk < D / WK; kk++) {
                    wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::col_major> b_frag;

                    wmma::load_matrix_sync(a_frag, sQ + m * WM * D + kk * WK, D);
                    wmma::load_matrix_sync(b_frag, sK + tile_n * WN * D + kk * WK, D);
                    wmma::mma_sync(s_frag, a_frag, b_frag, s_frag);
                }

                #pragma unroll
                for (int i = 0; i < s_frag.num_elements; i++)
                    s_frag.x[i] *= scale;

                wmma::store_matrix_sync(sS + m * WM * BN + tile_n * WN, s_frag, BN, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // Online softmax: 2 threads per row within each warp
        {
            int r = lane_id / 2;       // row within warp (0-15)
            int row = warp_id * WM + r; // global row within BM
            int q_pos = q_start + row;
            int col_half = lane_id % 2; // 0 or 1
            int col_start = col_half * (BN / 2);
            int col_end = col_start + BN / 2;

            float my_scale = 1.0f;

            if (q_pos < S) {
                // Compute local max with causal mask
                float local_max = -INFINITY;
                #pragma unroll
                for (int j = col_start; j < col_end; j++) {
                    int k_pos = k_start + j;
                    if (k_pos <= q_pos && k_pos < S)
                        local_max = fmaxf(local_max, sS[row * BN + j]);
                }

                // Reduce max across pair threads
                float partner_max = __shfl_xor_sync(0xFFFFFFFF, local_max, 1);
                local_max = fmaxf(local_max, partner_max);

                float old_max = s_rowmax[row];
                float new_max = fmaxf(old_max, local_max);
                float exp_old = (old_max == -INFINITY) ? 0.0f : __expf(old_max - new_max);

                // Compute P and local sum
                float local_sum = 0.0f;
                #pragma unroll
                for (int j = col_start; j < col_end; j++) {
                    int k_pos = k_start + j;
                    if (k_pos <= q_pos && k_pos < S) {
                        float val = __expf(sS[row * BN + j] - new_max);
                        sP[row * BN + j] = __float2bfloat16(val);
                        local_sum += val;
                    } else {
                        sP[row * BN + j] = __float2bfloat16(0.0f);
                    }
                }

                // Reduce sum across pair threads
                float partner_sum = __shfl_xor_sync(0xFFFFFFFF, local_sum, 1);
                local_sum += partner_sum;

                // Only even threads update shared state
                if (col_half == 0) {
                    s_rowmax[row] = new_max;
                    s_rowsum[row] = s_rowsum[row] * exp_old + local_sum;
                    my_scale = exp_old;
                }
            } else {
                // Write zeros to P for invalid rows
                #pragma unroll
                for (int j = col_start; j < col_end; j++) {
                    sP[row * BN + j] = __float2bfloat16(0.0f);
                }
            }

            // Rescale O fragments using per-row scale
            // Fragment layout: elems 0,1,4,5 -> row lane_id/4; elems 2,3,6,7 -> row lane_id/4+8
            float scale1 = __shfl_sync(0xFFFFFFFF, my_scale, 2 * (lane_id / 4));
            float scale2 = __shfl_sync(0xFFFFFFFF, my_scale, 2 * (lane_id / 4) + 16);

            #pragma unroll
            for (int tn = 0; tn < D_TILES; tn++) {
                o_frag[tn].x[0] *= scale1;
                o_frag[tn].x[1] *= scale1;
                o_frag[tn].x[4] *= scale1;
                o_frag[tn].x[5] *= scale1;
                o_frag[tn].x[2] *= scale2;
                o_frag[tn].x[3] *= scale2;
                o_frag[tn].x[6] *= scale2;
                o_frag[tn].x[7] *= scale2;
            }
        }
        __syncthreads();

        // O += P @ V using wmma
        {
            int m = warp_id;
            #pragma unroll
            for (int tile_n = 0; tile_n < D / WN; tile_n++) {
                #pragma unroll
                for (int kk = 0; kk < BN / WK; kk++) {
                    wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::row_major> b_frag;

                    wmma::load_matrix_sync(a_frag, sP + m * WM * BN + kk * WK, BN);
                    wmma::load_matrix_sync(b_frag, sV + kk * WK * D + tile_n * WN, D);
                    wmma::mma_sync(o_frag[tile_n], a_frag, b_frag, o_frag[tile_n]);
                }
            }
        }
        __syncthreads();
    }

    // Store O fragments to shared memory (reuse sQ and sS space)
    {
        float* sO_first = reinterpret_cast<float*>(sQ);   // 128*64 floats
        float* sO_second = reinterpret_cast<float*>(sS);  // 128*64 floats

        #pragma unroll
        for (int tn = 0; tn < 4; tn++) {
            wmma::store_matrix_sync(
                sO_first + warp_id * WM * 64 + tn * WN,
                o_frag[tn], 64, wmma::mem_row_major);
        }
        #pragma unroll
        for (int tn = 4; tn < 8; tn++) {
            wmma::store_matrix_sync(
                sO_second + warp_id * WM * 64 + (tn - 4) * WN,
                o_frag[tn], 64, wmma::mem_row_major);
        }
    }
    __syncthreads();

    // Final normalization and store
    {
        int row = tid % 128;
        int q_pos = q_start + row;
        if (q_pos < S) {
            float sum = s_rowsum[row] + 1e-30f;
            if (tid < 128) {
                LSE[(int64_t)(b * H + h) * S + q_pos] = s_rowmax[row] + logf(sum);
            }
            int col_half = tid / 128;
            int d_start = col_half * 64;
            float* sO_half = (col_half == 0)
                ? reinterpret_cast<float*>(sQ)
                : reinterpret_cast<float*>(sS);

            #pragma unroll
            for (int d = 0; d < 64; d += 8) {
                int4 vals;
                __nv_bfloat16* bf16_vals = reinterpret_cast<__nv_bfloat16*>(&vals);
                #pragma unroll
                for (int dd = 0; dd < 8; dd++) {
                    float val = sO_half[row * 64 + d + dd] / sum;
                    bf16_vals[dd] = __float2bfloat16(val);
                }
                *reinterpret_cast<int4*>(O + base_offset + (int64_t)q_pos * D + d_start + d) = vals;
            }
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