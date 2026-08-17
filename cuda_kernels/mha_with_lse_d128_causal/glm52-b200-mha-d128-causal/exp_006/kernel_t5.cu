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
constexpr int THREADS = 128;
constexpr int WM = 16, WN = 16, WK = 16;
constexpr int D_TILES = D / WN;
constexpr int COL_TILES = BN / WN;
constexpr int K_STEPS = D / WK;

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
    float* sP_float = reinterpret_cast<float*>(sV + BN * D);
    float* s_rowmax = sP_float + 16 * 16;
    float* s_rowsum = s_rowmax + BM;

    // Load Q
    #pragma unroll
    for (int i = tid; i < BM * D / 8; i += THREADS) {
        int row = i / (D / 8);
        int col8 = i % (D / 8);
        int q_pos = q_start + row;
        if (q_pos < S) {
            reinterpret_cast<int4*>(sQ)[i] = reinterpret_cast<const int4*>(
                Q + base_offset + (int64_t)q_pos * D)[col8];
        } else {
            reinterpret_cast<int4*>(sQ)[i] = make_int4(0, 0, 0, 0);
        }
    }

    for (int i = tid; i < BM; i += THREADS) {
        s_rowmax[i] = -INFINITY;
        s_rowsum[i] = 0.0f;
    }

    wmma::fragment<wmma::accumulator, WM, WN, WK, float> o_frag[D_TILES];
    #pragma unroll
    for (int tn = 0; tn < D_TILES; tn++)
        wmma::fill_fragment(o_frag[tn], 0.0f);

    __syncthreads();

    int max_k = min(S, q_start + BM);
    int num_k_blocks = (max_k + BN - 1) / BN;

    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        int k_start = k_block * BN;

        // Load K, V
        #pragma unroll
        for (int i = tid; i < BN * D / 8; i += THREADS) {
            int row = i / (D / 8);
            int col8 = i % (D / 8);
            int k_pos = k_start + row;
            if (k_pos < S) {
                reinterpret_cast<int4*>(sK)[i] = reinterpret_cast<const int4*>(
                    K + base_offset + (int64_t)k_pos * D)[col8];
                reinterpret_cast<int4*>(sV)[i] = reinterpret_cast<const int4*>(
                    V + base_offset + (int64_t)k_pos * D)[col8];
            } else {
                reinterpret_cast<int4*>(sK)[i] = make_int4(0, 0, 0, 0);
                reinterpret_cast<int4*>(sV)[i] = make_int4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        // QK^T: compute S fragments (4 per warp)
        wmma::fragment<wmma::accumulator, WM, WN, WK, float> s_frag[COL_TILES];
        #pragma unroll
        for (int t = 0; t < COL_TILES; t++)
            wmma::fill_fragment(s_frag[t], 0.0f);

        #pragma unroll
        for (int col_tile = 0; col_tile < COL_TILES; col_tile++) {
            #pragma unroll
            for (int kk = 0; kk < K_STEPS; kk++) {
                wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::col_major> b_frag;

                wmma::load_matrix_sync(a_frag, sQ + warp_id * WM * D + kk * WK, D);
                wmma::load_matrix_sync(b_frag, sK + col_tile * WN * D + kk * WK, D);
                wmma::mma_sync(s_frag[col_tile], a_frag, b_frag, s_frag[col_tile]);
            }
            #pragma unroll
            for (int i = 0; i < s_frag[col_tile].num_elements; i++)
                s_frag[col_tile].x[i] *= scale;
        }

        // Softmax from fragments
        // Fragment layout: x[0,1]→row r, col cg*2, cg*2+1; x[2,3]→row r+8
        //                   x[4,5]→row r, col cg*2+8, cg*2+9; x[6,7]→row r+8
        int r = lane_id / 4;
        int cg = lane_id % 4;
        int row0 = warp_id * WM + r;
        int row1 = warp_id * WM + r + 8;
        int q_pos0 = q_start + row0;
        int q_pos1 = q_start + row1;

        float max0 = -INFINITY, max1 = -INFINITY;

        #pragma unroll
        for (int t = 0; t < COL_TILES; t++) {
            int base_col = t * WN + cg * 2;
            int kp0 = k_start + base_col;
            int kp1 = k_start + base_col + 1;
            int kp2 = k_start + base_col + 8;
            int kp3 = k_start + base_col + 9;

            float v0 = s_frag[t].x[0], v1 = s_frag[t].x[1];
            float v4 = s_frag[t].x[4], v5 = s_frag[t].x[5];
            float v2 = s_frag[t].x[2], v3 = s_frag[t].x[3];
            float v6 = s_frag[t].x[6], v7 = s_frag[t].x[7];

            if (kp0 > q_pos0 || kp0 >= S) v0 = -INFINITY;
            if (kp1 > q_pos0 || kp1 >= S) v1 = -INFINITY;
            if (kp2 > q_pos0 || kp2 >= S) v4 = -INFINITY;
            if (kp3 > q_pos0 || kp3 >= S) v5 = -INFINITY;
            if (kp0 > q_pos1 || kp0 >= S) v2 = -INFINITY;
            if (kp1 > q_pos1 || kp1 >= S) v3 = -INFINITY;
            if (kp2 > q_pos1 || kp2 >= S) v6 = -INFINITY;
            if (kp3 > q_pos1 || kp3 >= S) v7 = -INFINITY;

            max0 = fmaxf(max0, fmaxf(fmaxf(v0, v1), fmaxf(v4, v5)));
            max1 = fmaxf(max1, fmaxf(fmaxf(v2, v3), fmaxf(v6, v7)));

            s_frag[t].x[0] = v0; s_frag[t].x[1] = v1;
            s_frag[t].x[4] = v4; s_frag[t].x[5] = v5;
            s_frag[t].x[2] = v2; s_frag[t].x[3] = v3;
            s_frag[t].x[6] = v6; s_frag[t].x[7] = v7;
        }

        #pragma unroll
        for (int offset = 2; offset > 0; offset >>= 1) {
            max0 = fmaxf(max0, __shfl_xor_sync(0xFFFFFFFF, max0, offset));
            max1 = fmaxf(max1, __shfl_xor_sync(0xFFFFFFFF, max1, offset));
        }

        float old_max0 = (q_pos0 < S) ? s_rowmax[row0] : -INFINITY;
        float old_max1 = (q_pos1 < S) ? s_rowmax[row1] : -INFINITY;
        float new_max0 = fmaxf(old_max0, max0);
        float new_max1 = fmaxf(old_max1, max1);
        float exp_old0 = (old_max0 == -INFINITY) ? 0.0f : __expf(old_max0 - new_max0);
        float exp_old1 = (old_max1 == -INFINITY) ? 0.0f : __expf(old_max1 - new_max1);

        float sum0 = 0.0f, sum1 = 0.0f;
        #pragma unroll
        for (int t = 0; t < COL_TILES; t++) {
            s_frag[t].x[0] = __expf(s_frag[t].x[0] - new_max0);
            s_frag[t].x[1] = __expf(s_frag[t].x[1] - new_max0);
            s_frag[t].x[4] = __expf(s_frag[t].x[4] - new_max0);
            s_frag[t].x[5] = __expf(s_frag[t].x[5] - new_max0);
            s_frag[t].x[2] = __expf(s_frag[t].x[2] - new_max1);
            s_frag[t].x[3] = __expf(s_frag[t].x[3] - new_max1);
            s_frag[t].x[6] = __expf(s_frag[t].x[6] - new_max1);
            s_frag[t].x[7] = __expf(s_frag[t].x[7] - new_max1);

            sum0 += s_frag[t].x[0] + s_frag[t].x[1] + s_frag[t].x[4] + s_frag[t].x[5];
            sum1 += s_frag[t].x[2] + s_frag[t].x[3] + s_frag[t].x[6] + s_frag[t].x[7];
        }

        #pragma unroll
        for (int offset = 2; offset > 0; offset >>= 1) {
            sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, offset);
            sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, offset);
        }

        if (cg == 0) {
            if (q_pos0 < S) {
                s_rowmax[row0] = new_max0;
                s_rowsum[row0] = s_rowsum[row0] * exp_old0 + sum0;
            }
            if (q_pos1 < S) {
                s_rowmax[row1] = new_max1;
                s_rowsum[row1] = s_rowsum[row1] * exp_old1 + sum1;
            }
        }

        // Rescale O fragments
        #pragma unroll
        for (int tn = 0; tn < D_TILES; tn++) {
            o_frag[tn].x[0] *= exp_old0;
            o_frag[tn].x[1] *= exp_old0;
            o_frag[tn].x[4] *= exp_old0;
            o_frag[tn].x[5] *= exp_old0;
            o_frag[tn].x[2] *= exp_old1;
            o_frag[tn].x[3] *= exp_old1;
            o_frag[tn].x[6] *= exp_old1;
            o_frag[tn].x[7] *= exp_old1;
        }

        __syncwarp();

        // PV: store each P tile, convert to BF16, then matmul
        #pragma unroll
        for (int k_step = 0; k_step < COL_TILES; k_step++) {
            wmma::store_matrix_sync(sP_float, s_frag[k_step], WN, wmma::mem_row_major);
            __syncwarp();

            __nv_bfloat16* bf16_ptr = reinterpret_cast<__nv_bfloat16*>(sP_float);
            for (int i = lane_id; i < 16 * 16; i += 32) {
                bf16_ptr[i] = __float2bfloat16(sP_float[i]);
            }
            __syncwarp();

            #pragma unroll
            for (int d_tile = 0; d_tile < D_TILES; d_tile++) {
                wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> p_frag;
                wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::row_major> v_frag;

                wmma::load_matrix_sync(p_frag, bf16_ptr, WN);
                wmma::load_matrix_sync(v_frag, sV + k_step * WK * D + d_tile * WN, D);
                wmma::mma_sync(o_frag[d_tile], p_frag, v_frag, o_frag[d_tile]);
            }
        }
        __syncthreads();
    }

    // Store O fragments to shared memory for final normalization
    float* sO_first = reinterpret_cast<float*>(sQ);
    float* sO_second = reinterpret_cast<float*>(sK);

    #pragma unroll
    for (int tn = 0; tn < 4; tn++) {
        wmma::store_matrix_sync(sO_first + warp_id * WM * 64 + tn * WN,
            o_frag[tn], 64, wmma::mem_row_major);
    }
    #pragma unroll
    for (int tn = 4; tn < 8; tn++) {
        wmma::store_matrix_sync(sO_second + warp_id * WM * 64 + (tn - 4) * WN,
            o_frag[tn], 64, wmma::mem_row_major);
    }
    __syncthreads();

    // Store LSE
    for (int i = tid; i < BM; i += THREADS) {
        int q_pos = q_start + i;
        if (q_pos < S) {
            LSE[(int64_t)(b * H + h) * S + q_pos] =
                s_rowmax[i] + logf(s_rowsum[i] + 1e-30f);
        }
    }

    // Final normalization and coalesced store
    for (int i = tid; i < BM * D / 8; i += THREADS) {
        int row = i / (D / 8);
        int col8 = i % (D / 8);
        int col = col8 * 8;
        int q_pos = q_start + row;
        if (q_pos < S) {
            float* sO_half = (col < 64) ? sO_first : sO_second;
            int local_col = col % 64;
            float sum = s_rowsum[row] + 1e-30f;

            int4 out;
            __nv_bfloat16* bf16_out = reinterpret_cast<__nv_bfloat16*>(&out);
            #pragma unroll
            for (int dd = 0; dd < 8; dd++) {
                float val = sO_half[row * 64 + local_col + dd] / sum;
                bf16_out[dd] = __float2bfloat16(val);
            }
            *reinterpret_cast<int4*>(O + base_offset + (int64_t)q_pos * D + col) = out;
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

    int smem_size = BM * D * 2        // sQ (16KB)
                  + BN * D * 2        // sK (16KB)
                  + BN * D * 2        // sV (16KB)
                  + 16 * 16 * 4       // sP_float (1KB)
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