#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
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
} while(0)

namespace tvm_mha_bwd {

constexpr int D = 128;
constexpr int BLOCK_M = 32;
constexpr int BLOCK_N = 32;
constexpr int NUM_THREADS = 128;
constexpr int PAD = 8;
constexpr int D_STRIDE = D + PAD;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Pass 1: compute dK and dV
__global__ void mha_bwd_dk_dv_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int S) {

    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    int j_tile = blockIdx.y;
    int j_start = j_tile * BLOCK_N;

    if (j_start >= S) return;

    const __nv_bfloat16* Q_bh = Q + (b * H + h) * S * D;
    const __nv_bfloat16* K_bh = K + (b * H + h) * S * D;
    const __nv_bfloat16* V_bh = V + (b * H + h) * S * D;
    const __nv_bfloat16* O_bh = O + (b * H + h) * S * D;
    const __nv_bfloat16* dO_bh = dO + (b * H + h) * S * D;
    const float* L_bh = L + (b * H + h) * S;
    __nv_bfloat16* dK_bh = dK + (b * H + h) * S * D;
    __nv_bfloat16* dV_bh = dV + (b * H + h) * S * D;

    float scale = rsqrtf((float)D);

    __shared__ __nv_bfloat16 smem_K[BLOCK_N][D_STRIDE];
    __shared__ __nv_bfloat16 smem_V[BLOCK_N][D_STRIDE];
    __shared__ __nv_bfloat16 smem_Q[BLOCK_M][D_STRIDE];
    __shared__ __nv_bfloat16 smem_dO[BLOCK_M][D_STRIDE];
    __shared__ __nv_bfloat16 smem_O[BLOCK_M][D_STRIDE];
    __shared__ float smem_P[BLOCK_M][BLOCK_N];
    __shared__ float smem_dS[BLOCK_M][BLOCK_N];
    __shared__ __nv_bfloat16 smem_dS_bf16[BLOCK_M][BLOCK_N];
    __shared__ float smem_dV[BLOCK_N][D];
    __shared__ float smem_dK[BLOCK_N][D];
    __shared__ float smem_Di[BLOCK_M];
    __shared__ float smem_L[BLOCK_M];

    for (int idx = threadIdx.x; idx < BLOCK_N * D; idx += blockDim.x) {
        int r = idx / D;
        int c = idx % D;
        int gr = j_start + r;
        if (gr < S) {
            smem_K[r][c] = K_bh[gr * D + c];
            smem_V[r][c] = V_bh[gr * D + c];
        } else {
            smem_K[r][c] = __float2bfloat16(0.0f);
            smem_V[r][c] = __float2bfloat16(0.0f);
        }
    }

    for (int idx = threadIdx.x; idx < BLOCK_N * D; idx += blockDim.x) {
        int r = idx / D;
        int c = idx % D;
        smem_dV[r][c] = 0.0f;
        smem_dK[r][c] = 0.0f;
    }
    __syncthreads();

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    for (int i_start = j_start; i_start < S; i_start += BLOCK_M) {
        for (int idx = threadIdx.x; idx < BLOCK_M * D; idx += blockDim.x) {
            int r = idx / D;
            int c = idx % D;
            int gr = i_start + r;
            if (gr < S) {
                smem_Q[r][c] = Q_bh[gr * D + c];
                smem_dO[r][c] = dO_bh[gr * D + c];
                smem_O[r][c] = O_bh[gr * D + c];
            } else {
                smem_Q[r][c] = __float2bfloat16(0.0f);
                smem_dO[r][c] = __float2bfloat16(0.0f);
                smem_O[r][c] = __float2bfloat16(0.0f);
            }
        }
        if (threadIdx.x < BLOCK_M) {
            int r = threadIdx.x;
            int gr = i_start + r;
            smem_L[r] = (gr < S) ? L_bh[gr] : 0.0f;
        }
        __syncthreads();

        for (int r = 0; r < 8; ++r) {
            int row = warp_id * 8 + r;
            float acc = 0.0f;
            for (int k = lane_id * 4; k < lane_id * 4 + 4; ++k) {
                acc += __bfloat162float(smem_dO[row][k]) * __bfloat162float(smem_O[row][k]);
            }
            acc = warp_reduce_sum(acc);
            if (lane_id == 0) {
                smem_Di[row] = acc;
            }
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 8, 16, __nv_bfloat16, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 8, 16, __nv_bfloat16, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 8, 16, float> c_frag;

        int m = warp_id / 2;
        for (int n = (warp_id % 2) * 2; n < (warp_id % 2) * 2 + 2; n++) {
            wmma::fill_fragment(c_frag, 0.0f);
            for (int k = 0; k < 8; k++) {
                wmma::load_matrix_sync(a_frag, &smem_Q[m * 16][k * 16], D_STRIDE);
                wmma::load_matrix_sync(b_frag, &smem_K[n * 8][k * 16], D_STRIDE);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            wmma::store_matrix_sync(&smem_P[m * 16][n * 8], c_frag, BLOCK_N, wmma::mem_row_major);

            wmma::fill_fragment(c_frag, 0.0f);
            for (int k = 0; k < 8; k++) {
                wmma::load_matrix_sync(a_frag, &smem_dO[m * 16][k * 16], D_STRIDE);
                wmma::load_matrix_sync(b_frag, &smem_V[n * 8][k * 16], D_STRIDE);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            wmma::store_matrix_sync(&smem_dS[m * 16][n * 8], c_frag, BLOCK_N, wmma::mem_row_major);
        }
        __syncthreads();

        for (int idx = threadIdx.x; idx < BLOCK_M * BLOCK_N; idx += blockDim.x) {
            int r = idx / BLOCK_N;
            int c = idx % BLOCK_N;
            int i_row = i_start + r;
            int j_col = j_start + c;
            bool valid = (i_row < S && j_col < S && i_row >= j_col);
            float p = valid ? expf(smem_P[r][c] * scale - smem_L[r]) : 0.0f;
            smem_P[r][c] = p;
            smem_dS[r][c] = p * (smem_dS[r][c] - smem_Di[r]) * scale;
            smem_dS_bf16[r][c] = __float2bfloat16(smem_dS[r][c]);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 8, 16, __nv_bfloat16, wmma::col_major> a_T_frag;
        wmma::fragment<wmma::matrix_b, 16, 8, 16, __nv_bfloat16, wmma::row_major> b_T_frag;

        for (int t = warp_id; t < 32; t += 4) {
            int row_t = t / 4; 
            int col_t = t % 4; 

            wmma::load_matrix_sync(c_frag, &smem_dV[col_t * 8][row_t * 16], D, wmma::mem_col_major);
            for (int k = 0; k < 2; k++) {
                wmma::load_matrix_sync(a_T_frag, &smem_dO[col_t * 8][row_t * 16], D_STRIDE, wmma::mem_col_major);
                wmma::load_matrix_sync(b_T_frag, &smem_P[k * 16][col_t * 8], BLOCK_N, wmma::mem_row_major);
                wmma::mma_sync(c_frag, a_T_frag, b_T_frag, c_frag);
            }
            wmma::store_matrix_sync(&smem_dV[col_t * 8][row_t * 16], c_frag, D, wmma::mem_col_major);

            wmma::load_matrix_sync(c_frag, &smem_dK[col_t * 8][row_t * 16], D, wmma::mem_col_major);
            for (int k = 0; k < 2; k++) {
                wmma::load_matrix_sync(a_T_frag, &smem_Q[col_t * 8][row_t * 16], D_STRIDE, wmma::mem_col_major);
                wmma::load_matrix_sync(b_T_frag, &smem_dS_bf16[k * 16][col_t * 8], BLOCK_N, wmma::mem_row_major);
                wmma::mma_sync(c_frag, a_T_frag, b_T_frag, c_frag);
            }
            wmma::store_matrix_sync(&smem_dK[col_t * 8][row_t * 16], c_frag, D, wmma::mem_col_major);
        }
        __syncthreads();
    }

    for (int idx = threadIdx.x; idx < BLOCK_N * D; idx += blockDim.x) {
        int r = idx / D;
        int c = idx % D;
        int gr = j_start + r;
        if (gr < S) {
            dV_bh[gr * D + c] = __float2bfloat16(smem_dV[r][c]);
            dK_bh[gr * D + c] = __float2bfloat16(smem_dK[r][c]);
        }
    }
}

// Pass 2: compute dQ
__global__ void mha_bwd_dq_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    int B, int H, int S) {

    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    int i_tile = blockIdx.y;
    int i_start = i_tile * BLOCK_M;

    if (i_start >= S) return;

    const __nv_bfloat16* Q_bh = Q + (b * H + h) * S * D;
    const __nv_bfloat16* K_bh = K + (b * H + h) * S * D;
    const __nv_bfloat16* V_bh = V + (b * H + h) * S * D;
    const __nv_bfloat16* O_bh = O + (b * H + h) * S * D;
    const __nv_bfloat16* dO_bh = dO + (b * H + h) * S * D;
    const float* L_bh = L + (b * H + h) * S;
    __nv_bfloat16* dQ_bh = dQ + (b * H + h) * S * D;

    float scale = rsqrtf((float)D);

    __shared__ __nv_bfloat16 smem_Q[BLOCK_M][D_STRIDE];
    __shared__ __nv_bfloat16 smem_dO[BLOCK_M][D_STRIDE];
    __shared__ __nv_bfloat16 smem_O[BLOCK_M][D_STRIDE];
    __shared__ __nv_bfloat16 smem_K[BLOCK_N][D_STRIDE];
    __shared__ __nv_bfloat16 smem_V[BLOCK_N][D_STRIDE];
    __shared__ float smem_P[BLOCK_M][BLOCK_N];
    __shared__ float smem_dS[BLOCK_M][BLOCK_N];
    __shared__ __nv_bfloat16 smem_dS_bf16[BLOCK_M][BLOCK_N];
    __shared__ float smem_dQ[BLOCK_M][D];
    __shared__ float smem_Di[BLOCK_M];
    __shared__ float smem_L[BLOCK_M];

    for (int idx = threadIdx.x; idx < BLOCK_M * D; idx += blockDim.x) {
        int r = idx / D;
        int c = idx % D;
        int gr = i_start + r;
        if (gr < S) {
            smem_Q[r][c] = Q_bh[gr * D + c];
            smem_dO[r][c] = dO_bh[gr * D + c];
            smem_O[r][c] = O_bh[gr * D + c];
        } else {
            smem_Q[r][c] = __float2bfloat16(0.0f);
            smem_dO[r][c] = __float2bfloat16(0.0f);
            smem_O[r][c] = __float2bfloat16(0.0f);
        }
    }
    if (threadIdx.x < BLOCK_M) {
        int r = threadIdx.x;
        int gr = i_start + r;
        smem_L[r] = (gr < S) ? L_bh[gr] : 0.0f;
    }
    for (int idx = threadIdx.x; idx < BLOCK_M * D; idx += blockDim.x) {
        int r = idx / D;
        int c = idx % D;
        smem_dQ[r][c] = 0.0f;
    }
    __syncthreads();

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    for (int r = 0; r < 8; ++r) {
        int row = warp_id * 8 + r;
        float acc = 0.0f;
        for (int k = lane_id * 4; k < lane_id * 4 + 4; ++k) {
            acc += __bfloat162float(smem_dO[row][k]) * __bfloat162float(smem_O[row][k]);
        }
        acc = warp_reduce_sum(acc);
        if (lane_id == 0) {
            smem_Di[row] = acc;
        }
    }
    __syncthreads();

    for (int j_start = 0; j_start <= i_start + BLOCK_M - 1; j_start += BLOCK_N) {
        for (int idx = threadIdx.x; idx < BLOCK_N * D; idx += blockDim.x) {
            int r = idx / D;
            int c = idx % D;
            int gr = j_start + r;
            if (gr < S) {
                smem_K[r][c] = K_bh[gr * D + c];
                smem_V[r][c] = V_bh[gr * D + c];
            } else {
                smem_K[r][c] = __float2bfloat16(0.0f);
                smem_V[r][c] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 8, 16, __nv_bfloat16, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 8, 16, __nv_bfloat16, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 8, 16, float> c_frag;

        int m = warp_id / 2;
        for (int n = (warp_id % 2) * 2; n < (warp_id % 2) * 2 + 2; n++) {
            wmma::fill_fragment(c_frag, 0.0f);
            for (int k = 0; k < 8; k++) {
                wmma::load_matrix_sync(a_frag, &smem_Q[m * 16][k * 16], D_STRIDE);
                wmma::load_matrix_sync(b_frag, &smem_K[n * 8][k * 16], D_STRIDE);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            wmma::store_matrix_sync(&smem_P[m * 16][n * 8], c_frag, BLOCK_N, wmma::mem_row_major);

            wmma::fill_fragment(c_frag, 0.0f);
            for (int k = 0; k < 8; k++) {
                wmma::load_matrix_sync(a_frag, &smem_dO[m * 16][k * 16], D_STRIDE);
                wmma::load_matrix_sync(b_frag, &smem_V[n * 8][k * 16], D_STRIDE);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            wmma::store_matrix_sync(&smem_dS[m * 16][n * 8], c_frag, BLOCK_N, wmma::mem_row_major);
        }
        __syncthreads();

        for (int idx = threadIdx.x; idx < BLOCK_M * BLOCK_N; idx += blockDim.x) {
            int r = idx / BLOCK_N;
            int c = idx % BLOCK_N;
            int i_row = i_start + r;
            int j_col = j_start + c;
            bool valid = (i_row < S && j_col < S && i_row >= j_col);
            float p = valid ? expf(smem_P[r][c] * scale - smem_L[r]) : 0.0f;
            smem_P[r][c] = p;
            smem_dS[r][c] = p * (smem_dS[r][c] - smem_Di[r]) * scale;
            smem_dS_bf16[r][c] = __float2bfloat16(smem_dS[r][c]);
        }
        __syncthreads();

        for (int t = warp_id; t < 32; t += 4) {
            int row_t = t / 4; 
            int col_t = t % 4; 
            wmma::load_matrix_sync(c_frag, &smem_dQ[col_t * 8][row_t * 16], D, wmma::mem_row_major);
            for (int k = 0; k < 2; k++) {
                wmma::load_matrix_sync(a_frag, &smem_dS_bf16[col_t * 8][k * 16], BLOCK_N, wmma::mem_row_major);
                wmma::load_matrix_sync(b_frag, &smem_K[row_t * 8][k * 16], D_STRIDE, wmma::mem_col_major);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            wmma::store_matrix_sync(&smem_dQ[col_t * 8][row_t * 16], c_frag, D, wmma::mem_row_major);
        }
        __syncthreads();
    }

    for (int idx = threadIdx.x; idx < BLOCK_M * D; idx += blockDim.x) {
        int r = idx / D;
        int c = idx % D;
        int gr = i_start + r;
        if (gr < S) {
            dQ_bh[gr * D + c] = __float2bfloat16(smem_dQ[r][c]);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);

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

    int num_tiles = (S + BLOCK_N - 1) / BLOCK_N;
    dim3 grid1(B * H, num_tiles);
    dim3 block1(NUM_THREADS);
    mha_bwd_dk_dv_kernel<<<grid1, block1, 0, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr, B, H, S);
    CUDA_CHECK(cudaGetLastError());

    dim3 grid2(B * H, num_tiles);
    dim3 block2(NUM_THREADS);
    mha_bwd_dq_kernel<<<grid2, block2, 0, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr, B, H, S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_mha_bwd::run);

}  // namespace tvm_mha_bwd