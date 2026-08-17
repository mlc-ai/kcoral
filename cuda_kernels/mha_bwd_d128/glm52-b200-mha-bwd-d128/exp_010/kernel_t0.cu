#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cmath>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while (0)

namespace attention_bwd {

constexpr int D_HEAD = 128;
constexpr int BQ = 32;
constexpr int BKV = 64;
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int NUM_WARPS = 4;
constexpr int NUM_THREADS = 128;

constexpr int SMEM_SIZE =
    BKV * D_HEAD * 2 +      // Kj (bf16) = 16384
    BKV * D_HEAD * 2 +      // Vj (bf16) = 16384
    BQ * D_HEAD * 2 +       // Qi (bf16) = 8192
    BQ * D_HEAD * 2 +       // dOi (bf16) = 8192
    BQ * BKV * 4 +          // S/dP_smem (float) = 8192
    BQ * BKV * 4 +          // P_smem (float) = 8192
    BQ * BKV * 4 +          // dS_smem (float) = 8192
    BQ * BKV * 2 +          // P_bf16_smem = 4096
    BQ * BKV * 2 +          // dS_bf16_smem = 4096
    BQ * 4 +                // Li = 128
    BQ * 4 +                // Di = 128
    NUM_WARPS * WMMA_M * WMMA_N * 4;  // dQ_stage = 4096
// Total = 86272

__global__ void compute_D_kernel(
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    float* __restrict__ D,
    int total_rows) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= total_rows) return;

    const __nv_bfloat16* O_ptr = O + (size_t)row * D_HEAD;
    const __nv_bfloat16* dO_ptr = dO + (size_t)row * D_HEAD;

    float sum = 0.0f;
    for (int i = tid; i < D_HEAD; i += blockDim.x) {
        sum += __bfloat162float(O_ptr[i]) * __bfloat162float(dO_ptr[i]);
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }
    if (tid % 32 == 0) {
        D[row] = sum;
    }
}

__global__ void zero_float_kernel(float* ptr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) ptr[idx] = 0.0f;
}

__global__ void convert_f32_to_bf16_kernel(
    const float* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2bfloat16(src[idx]);
}

__global__ __launch_bounds__(NUM_THREADS, 2) void attention_backward_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D_arr,
    float* __restrict__ dQ_float,
    __nv_bfloat16* __restrict__ dK_out,
    __nv_bfloat16* __restrict__ dV_out,
    int B, int H, int S) {

    extern __shared__ char smem[];
    __nv_bfloat16* Kj_smem = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* Vj_smem = Kj_smem + BKV * D_HEAD;
    __nv_bfloat16* Qi_smem = Vj_smem + BKV * D_HEAD;
    __nv_bfloat16* dOi_smem = Qi_smem + BQ * D_HEAD;
    float* S_smem = reinterpret_cast<float*>(dOi_smem + BQ * D_HEAD);
    float* P_smem = S_smem + BQ * BKV;
    float* dS_smem = P_smem + BQ * BKV;
    __nv_bfloat16* P_bf16_smem = reinterpret_cast<__nv_bfloat16*>(dS_smem + BQ * BKV);
    __nv_bfloat16* dS_bf16_smem = P_bf16_smem + BQ * BKV;
    float* Li_smem = reinterpret_cast<float*>(dS_bf16_smem + BQ * BKV);
    float* Di_smem = Li_smem + BQ;
    float* dQ_stage = Di_smem + BQ;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int num_kv_blocks = (S + BKV - 1) / BKV;
    int bh_idx = blockIdx.x / num_kv_blocks;
    int kv_block = blockIdx.x % num_kv_blocks;

    int b = bh_idx / H;
    int h = bh_idx % H;
    int kv_start = kv_block * BKV;

    size_t bh_offset = (size_t)(b * H + h) * S * D_HEAD;
    const __nv_bfloat16* Q_base = Q + bh_offset;
    const __nv_bfloat16* K_base = K + bh_offset;
    const __nv_bfloat16* V_base = V + bh_offset;
    const __nv_bfloat16* dO_base = dO + bh_offset;
    const float* L_base = L + (size_t)(b * H + h) * S;
    const float* D_base = D_arr + (size_t)(b * H + h) * S;
    float* dQ_base = dQ_float + bh_offset;
    __nv_bfloat16* dK_base = dK_out + bh_offset;
    __nv_bfloat16* dV_base = dV_out + bh_offset;

    // Load Kj, Vj into shared memory
    for (int i = tid; i < BKV * D_HEAD; i += NUM_THREADS) {
        int row = i / D_HEAD;
        int col = i % D_HEAD;
        int gr = kv_start + row;
        if (gr < S) {
            Kj_smem[i] = K_base[(size_t)gr * D_HEAD + col];
            Vj_smem[i] = V_base[(size_t)gr * D_HEAD + col];
        } else {
            Kj_smem[i] = __float2bfloat16(0.0f);
            Vj_smem[i] = __float2bfloat16(0.0f);
        }
    }
    __syncthreads();

    // dK/dV accumulators: BKV x D_HEAD = 64 x 128
    // 4 row tiles x 8 col tiles = 32 tiles, 8 per warp
    // warp 0: rows 0-31, cols 0-63; warp 1: rows 0-31, cols 64-127
    // warp 2: rows 32-63, cols 0-63; warp 3: rows 32-63, cols 64-127
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> dK_frag[8];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> dV_frag[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        wmma::fill_fragment(dK_frag[i], 0.0f);
        wmma::fill_fragment(dV_frag[i], 0.0f);
    }

    const float scale = 0.0883883476f; // 1/sqrt(128)

    // Loop over query blocks
    for (int q_start = 0; q_start < S; q_start += BQ) {
        // Load Qi, dOi
        for (int i = tid; i < BQ * D_HEAD; i += NUM_THREADS) {
            int row = i / D_HEAD;
            int col = i % D_HEAD;
            int gr = q_start + row;
            if (gr < S) {
                Qi_smem[i] = Q_base[(size_t)gr * D_HEAD + col];
                dOi_smem[i] = dO_base[(size_t)gr * D_HEAD + col];
            } else {
                Qi_smem[i] = __float2bfloat16(0.0f);
                dOi_smem[i] = __float2bfloat16(0.0f);
            }
        }
        if (tid < BQ) {
            int gr = q_start + tid;
            Li_smem[tid] = (gr < S) ? L_base[gr] : 0.0f;
            Di_smem[tid] = (gr < S) ? D_base[gr] : 0.0f;
        }
        __syncthreads();

        // Step 1: S = Qi @ Kj^T (BQ x BKV), 2 row tiles x 4 col tiles, 8 k-tiles
        // warp w: row tile w/2, col tiles w%2*2 to w%2*2+1
        {
            int wr = warp_id / 2;  // 0 or 1
            int wc = (warp_id % 2) * 2;  // 0 or 2
            #pragma unroll
            for (int wj = 0; wj < 2; wj++) {
                int rt = wr;
                int ct = wc + wj;
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
                wmma::fill_fragment(c_frag, 0.0f);
                #pragma unroll
                for (int kk = 0; kk < D_HEAD / WMMA_K; kk++) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;
                    wmma::load_matrix_sync(a_frag, Qi_smem + rt * 16 * D_HEAD + kk * 16, D_HEAD);
                    wmma::load_matrix_sync(b_frag, Kj_smem + ct * 16 * D_HEAD + kk * 16, D_HEAD);
                    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }
                wmma::store_matrix_sync(S_smem + rt * 16 * BKV + ct * 16, c_frag, BKV, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // Step 2: P = exp(S * scale - L)
        for (int i = tid; i < BQ * BKV; i += NUM_THREADS) {
            int row = i / BKV;
            int col = i % BKV;
            int qi = q_start + row;
            int ki = kv_start + col;
            if (qi < S && ki < S) {
                P_smem[i] = expf(S_smem[i] * scale - Li_smem[row]);
            } else {
                P_smem[i] = 0.0f;
            }
        }
        __syncthreads();

        // Step 3: dP = dOi @ Vj^T (same structure as S)
        {
            int wr = warp_id / 2;
            int wc = (warp_id % 2) * 2;
            #pragma unroll
            for (int wj = 0; wj < 2; wj++) {
                int rt = wr;
                int ct = wc + wj;
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
                wmma::fill_fragment(c_frag, 0.0f);
                #pragma unroll
                for (int kk = 0; kk < D_HEAD / WMMA_K; kk++) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;
                    wmma::load_matrix_sync(a_frag, dOi_smem + rt * 16 * D_HEAD + kk * 16, D_HEAD);
                    wmma::load_matrix_sync(b_frag, Vj_smem + ct * 16 * D_HEAD + kk * 16, D_HEAD);
                    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }
                wmma::store_matrix_sync(S_smem + rt * 16 * BKV + ct * 16, c_frag, BKV, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // Step 4: dS = P * (dP - D) * scale
        for (int i = tid; i < BQ * BKV; i += NUM_THREADS) {
            int row = i / BKV;
            int col = i % BKV;
            int qi = q_start + row;
            int ki = kv_start + col;
            if (qi < S && ki < S) {
                dS_smem[i] = P_smem[i] * (S_smem[i] - Di_smem[row]) * scale;
            } else {
                dS_smem[i] = 0.0f;
            }
        }
        __syncthreads();

        // Convert P and dS to bf16 for matmuls
        for (int i = tid; i < BQ * BKV; i += NUM_THREADS) {
            P_bf16_smem[i] = __float2bfloat16(P_smem[i]);
            dS_bf16_smem[i] = __float2bfloat16(dS_smem[i]);
        }
        __syncthreads();

        // Step 5: dV += P^T @ dO (BKV x D_HEAD)
        // A = P^T (col_major from P_bf16), B = dO (row_major)
        // 4 row tiles x 8 col tiles, 2 k-tiles (BQ/16=2)
        {
            int wrs = (warp_id / 2) * 32;
            int wcs = (warp_id % 2) * 64;
            #pragma unroll
            for (int wi = 0; wi < 2; wi++) {
                #pragma unroll
                for (int wj = 0; wj < 4; wj++) {
                    int rt = (wrs + wi * 16) / 16;
                    int ct = (wcs + wj * 16) / 16;
                    int fi = wi * 4 + wj;
                    #pragma unroll
                    for (int kk = 0; kk < BQ / WMMA_K; kk++) {
                        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> a_frag;
                        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
                        wmma::load_matrix_sync(a_frag, P_bf16_smem + kk * 16 * BKV + rt * 16, BKV);
                        wmma::load_matrix_sync(b_frag, dOi_smem + kk * 16 * D_HEAD + ct * 16, D_HEAD);
                        wmma::mma_sync(dV_frag[fi], a_frag, b_frag, dV_frag[fi]);
                    }
                }
            }
        }

        // Step 6: dK += dS^T @ Q (BKV x D_HEAD)
        {
            int wrs = (warp_id / 2) * 32;
            int wcs = (warp_id % 2) * 64;
            #pragma unroll
            for (int wi = 0; wi < 2; wi++) {
                #pragma unroll
                for (int wj = 0; wj < 4; wj++) {
                    int rt = (wrs + wi * 16) / 16;
                    int ct = (wcs + wj * 16) / 16;
                    int fi = wi * 4 + wj;
                    #pragma unroll
                    for (int kk = 0; kk < BQ / WMMA_K; kk++) {
                        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> a_frag;
                        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
                        wmma::load_matrix_sync(a_frag, dS_bf16_smem + kk * 16 * BKV + rt * 16, BKV);
                        wmma::load_matrix_sync(b_frag, Qi_smem + kk * 16 * D_HEAD + ct * 16, D_HEAD);
                        wmma::mma_sync(dK_frag[fi], a_frag, b_frag, dK_frag[fi]);
                    }
                }
            }
        }

        // Step 7: dQ += dS @ K (BQ x D_HEAD) -> atomic add to global
        // 2 row tiles x 8 col tiles, 4 k-tiles (BKV/16=4)
        // 4 tiles per warp
        {
            int wrs = (warp_id / 2) * 16;
            int wcs = (warp_id % 2) * 64;
            #pragma unroll
            for (int wj = 0; wj < 4; wj++) {
                int rt = wrs / 16;
                int ct = (wcs + wj * 16) / 16;

                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> dq_frag;
                wmma::fill_fragment(dq_frag, 0.0f);
                #pragma unroll
                for (int kk = 0; kk < BKV / WMMA_K; kk++) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
                    wmma::load_matrix_sync(a_frag, dS_bf16_smem + rt * 16 * BKV + kk * 16, BKV);
                    wmma::load_matrix_sync(b_frag, Kj_smem + kk * 16 * D_HEAD + ct * 16, D_HEAD);
                    wmma::mma_sync(dq_frag, a_frag, b_frag, dq_frag);
                }

                float* stage = dQ_stage + warp_id * WMMA_M * WMMA_N;
                wmma::store_matrix_sync(stage, dq_frag, WMMA_N, wmma::mem_row_major);
                __syncwarp();

                int grb = q_start + rt * 16;
                int gcb = ct * 16;
                for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {
                    int r = i / WMMA_N;
                    int c = i % WMMA_N;
                    int gr = grb + r;
                    int gc = gcb + c;
                    if (gr < S) {
                        atomicAdd(&dQ_base[(size_t)gr * D_HEAD + gc], stage[i]);
                    }
                }
            }
        }
        __syncthreads();
    }

    // Store dK, dV to global
    {
        int wrs = (warp_id / 2) * 32;
        int wcs = (warp_id % 2) * 64;
        #pragma unroll
        for (int wi = 0; wi < 2; wi++) {
            #pragma unroll
            for (int wj = 0; wj < 4; wj++) {
                int rt = (wrs + wi * 16) / 16;
                int ct = (wcs + wj * 16) / 16;
                int fi = wi * 4 + wj;
                int grb = kv_start + rt * 16;
                int gcb = ct * 16;
                float* stage = dQ_stage + warp_id * WMMA_M * WMMA_N;

                wmma::store_matrix_sync(stage, dK_frag[fi], WMMA_N, wmma::mem_row_major);
                __syncwarp();
                for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {
                    int r = i / WMMA_N;
                    int c = i % WMMA_N;
                    int gr = grb + r;
                    int gc = gcb + c;
                    if (gr < S) {
                        dK_base[(size_t)gr * D_HEAD + gc] = __float2bfloat16(stage[i]);
                    }
                }

                wmma::store_matrix_sync(stage, dV_frag[fi], WMMA_N, wmma::mem_row_major);
                __syncwarp();
                for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {
                    int r = i / WMMA_N;
                    int c = i % WMMA_N;
                    int gr = grb + r;
                    int gc = gcb + c;
                    if (gr < S) {
                        dV_base[(size_t)gr * D_HEAD + gc] = __float2bfloat16(stage[i]);
                    }
                }
            }
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {

    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    const int B = 4;
    const int H = 48;
    const int S = (int)Q.size(2);
    const int d = 128;

    int total_elements = B * H * S * d;
    int total_rows = B * H * S;

    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_ptr = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    float* D_buf;
    CUDA_CHECK(cudaMalloc(&D_buf, (size_t)total_rows * sizeof(float)));
    float* dQ_float;
    CUDA_CHECK(cudaMalloc(&dQ_float, (size_t)total_elements * sizeof(float)));

    {
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        zero_float_kernel<<<blocks, threads, 0, stream>>>(dQ_float, total_elements);
    }
    {
        int threads = 128;
        compute_D_kernel<<<total_rows, threads, 0, stream>>>(O_ptr, dO_ptr, D_buf, total_rows);
    }
    {
        int num_kv_blocks = (S + BKV - 1) / BKV;
        int grid = B * H * num_kv_blocks;
        CUDA_CHECK(cudaFuncSetAttribute(attention_backward_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));
        attention_backward_kernel<<<grid, NUM_THREADS, SMEM_SIZE, stream>>>(
            Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_buf,
            dQ_float, dK_ptr, dV_ptr, B, H, S);
    }
    {
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        convert_f32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(dQ_float, dQ_ptr, total_elements);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFree(D_buf));
    CUDA_CHECK(cudaFree(dQ_float));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, attention_bwd::run);

}  // namespace attention_bwd