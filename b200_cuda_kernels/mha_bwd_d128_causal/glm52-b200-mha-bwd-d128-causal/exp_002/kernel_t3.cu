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
constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;

__device__ __forceinline__ float reduce_sum4(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 1);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    return val;
}

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

    __shared__ __nv_bfloat16 smem_K[BLOCK_N][D];
    __shared__ __nv_bfloat16 smem_V[BLOCK_N][D];
    __shared__ __nv_bfloat16 smem_Q[BLOCK_M][D];
    __shared__ __nv_bfloat16 smem_dO[BLOCK_M][D];
    __shared__ __nv_bfloat16 smem_O[BLOCK_M][D];
    __shared__ float smem_P[BLOCK_M][BLOCK_N];
    __shared__ float smem_dS[BLOCK_M][BLOCK_N];
    __shared__ __nv_bfloat16 smem_P_bf16[BLOCK_M][BLOCK_N];
    __shared__ __nv_bfloat16 smem_dS_bf16[BLOCK_M][BLOCK_N];
    __shared__ float smem_dV[BLOCK_N][D];
    __shared__ float smem_dK[BLOCK_N][D];
    __shared__ float smem_Di[BLOCK_M];
    __shared__ float smem_L[BLOCK_M];

    for (int idx = threadIdx.x; idx < BLOCK_N * D / 8; idx += blockDim.x) {
        int r = idx / (D / 8);
        int c = (idx % (D / 8)) * 8;
        int gr = j_start + r;
        if (gr < S) {
            *reinterpret_cast<float4*>(&smem_K[r][c]) = *reinterpret_cast<const float4*>(&K_bh[gr * D + c]);
            *reinterpret_cast<float4*>(&smem_V[r][c]) = *reinterpret_cast<const float4*>(&V_bh[gr * D + c]);
        } else {
            *reinterpret_cast<float4*>(&smem_K[r][c]) = make_float4(0,0,0,0);
            *reinterpret_cast<float4*>(&smem_V[r][c]) = make_float4(0,0,0,0);
        }
    }
    for (int idx = threadIdx.x; idx < BLOCK_N * D; idx += blockDim.x) {
        smem_dV[idx / D][idx % D] = 0.0f;
        smem_dK[idx / D][idx % D] = 0.0f;
    }
    __syncthreads();

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int wm = warp_id / 2;
    int wn = warp_id % 2;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_row;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_col;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> a_col;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_row;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;

    for (int i_start = j_start; i_start < S; i_start += BLOCK_M) {
        for (int idx = threadIdx.x; idx < BLOCK_M * D / 8; idx += blockDim.x) {
            int r = idx / (D / 8);
            int c = (idx % (D / 8)) * 8;
            int gr = i_start + r;
            if (gr < S) {
                *reinterpret_cast<float4*>(&smem_Q[r][c]) = *reinterpret_cast<const float4*>(&Q_bh[gr * D + c]);
                *reinterpret_cast<float4*>(&smem_dO[r][c]) = *reinterpret_cast<const float4*>(&dO_bh[gr * D + c]);
                *reinterpret_cast<float4*>(&smem_O[r][c]) = *reinterpret_cast<const float4*>(&O_bh[gr * D + c]);
            } else {
                *reinterpret_cast<float4*>(&smem_Q[r][c]) = make_float4(0,0,0,0);
                *reinterpret_cast<float4*>(&smem_dO[r][c]) = make_float4(0,0,0,0);
                *reinterpret_cast<float4*>(&smem_O[r][c]) = make_float4(0,0,0,0);
            }
        }
        if (threadIdx.x < BLOCK_M) {
            int r = threadIdx.x;
            int gr = i_start + r;
            smem_L[r] = (gr < S) ? L_bh[gr] : 0.0f;
        }
        __syncthreads();

        {
            int row_in_warp = lane_id / 4;
            int sub = lane_id % 4;
            int row = warp_id * 8 + row_in_warp;
            float acc_val = 0.0f;
            for (int k = sub * 32; k < sub * 32 + 32; ++k) {
                acc_val += __bfloat162float(smem_dO[row][k]) * __bfloat162float(smem_O[row][k]);
            }
            acc_val = reduce_sum4(acc_val);
            if (sub == 0) smem_Di[row] = acc_val;
        }
        __syncthreads();

        wmma::fill_fragment(acc, 0.0f);
        for (int k = 0; k < D; k += WMMA_K) {
            wmma::load_matrix_sync(a_row, &smem_Q[wm*16][k], D);
            wmma::load_matrix_sync(b_col, &smem_K[wn*16][k], D);
            wmma::mma_sync(acc, a_row, b_col, acc);
        }
        wmma::store_matrix_sync(&smem_P[wm*16][wn*16], acc, BLOCK_N, wmma::mem_row_major);

        wmma::fill_fragment(acc, 0.0f);
        for (int k = 0; k < D; k += WMMA_K) {
            wmma::load_matrix_sync(a_row, &smem_dO[wm*16][k], D);
            wmma::load_matrix_sync(b_col, &smem_V[wn*16][k], D);
            wmma::mma_sync(acc, a_row, b_col, acc);
        }
        wmma::store_matrix_sync(&smem_dS[wm*16][wn*16], acc, BLOCK_N, wmma::mem_row_major);
        __syncthreads();

        for (int idx = threadIdx.x; idx < BLOCK_M * BLOCK_N; idx += blockDim.x) {
            int r = idx / BLOCK_N;
            int c = idx % BLOCK_N;
            int i_row = i_start + r;
            int j_col = j_start + c;
            bool valid = (i_row < S && j_col < S && i_row >= j_col);
            float p = valid ? __expf(smem_P[r][c] * scale - smem_L[r]) : 0.0f;
            smem_P[r][c] = p;
            float ds = p * (smem_dS[r][c] - smem_Di[r]) * scale;
            smem_dS[r][c] = ds;
            smem_P_bf16[r][c] = __float2bfloat16(p);
            smem_dS_bf16[r][c] = __float2bfloat16(ds);
        }
        __syncthreads();

        for (int t = warp_id; t < 16; t += 4) {
            int jb = t / 8;
            int db = t % 8;

            wmma::load_matrix_sync(acc, &smem_dV[jb*16][db*16], D, wmma::mem_row_major);
            for (int ib = 0; ib < 2; ib++) {
                wmma::load_matrix_sync(a_col, &smem_P_bf16[ib*16][jb*16], BLOCK_N);
                wmma::load_matrix_sync(b_row, &smem_dO[ib*16][db*16], D);
                wmma::mma_sync(acc, a_col, b_row, acc);
            }
            wmma::store_matrix_sync(&smem_dV[jb*16][db*16], acc, D, wmma::mem_row_major);

            wmma::load_matrix_sync(acc, &smem_dK[jb*16][db*16], D, wmma::mem_row_major);
            for (int ib = 0; ib < 2; ib++) {
                wmma::load_matrix_sync(a_col, &smem_dS_bf16[ib*16][jb*16], BLOCK_N);
                wmma::load_matrix_sync(b_row, &smem_Q[ib*16][db*16], D);
                wmma::mma_sync(acc, a_col, b_row, acc);
            }
            wmma::store_matrix_sync(&smem_dK[jb*16][db*16], acc, D, wmma::mem_row_major);
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

    __shared__ __nv_bfloat16 smem_Q[BLOCK_M][D];
    __shared__ __nv_bfloat16 smem_dO[BLOCK_M][D];
    __shared__ __nv_bfloat16 smem_O[BLOCK_M][D];
    __shared__ __nv_bfloat16 smem_K[BLOCK_N][D];
    __shared__ __nv_bfloat16 smem_V[BLOCK_N][D];
    __shared__ float smem_P[BLOCK_M][BLOCK_N];
    __shared__ float smem_dS[BLOCK_M][BLOCK_N];
    __shared__ __nv_bfloat16 smem_dS_bf16[BLOCK_M][BLOCK_N];
    __shared__ float smem_dQ[BLOCK_M][D];
    __shared__ float smem_Di[BLOCK_M];
    __shared__ float smem_L[BLOCK_M];

    for (int idx = threadIdx.x; idx < BLOCK_M * D / 8; idx += blockDim.x) {
        int r = idx / (D / 8);
        int c = (idx % (D / 8)) * 8;
        int gr = i_start + r;
        if (gr < S) {
            *reinterpret_cast<float4*>(&smem_Q[r][c]) = *reinterpret_cast<const float4*>(&Q_bh[gr * D + c]);
            *reinterpret_cast<float4*>(&smem_dO[r][c]) = *reinterpret_cast<const float4*>(&dO_bh[gr * D + c]);
            *reinterpret_cast<float4*>(&smem_O[r][c]) = *reinterpret_cast<const float4*>(&O_bh[gr * D + c]);
        } else {
            *reinterpret_cast<float4*>(&smem_Q[r][c]) = make_float4(0,0,0,0);
            *reinterpret_cast<float4*>(&smem_dO[r][c]) = make_float4(0,0,0,0);
            *reinterpret_cast<float4*>(&smem_O[r][c]) = make_float4(0,0,0,0);
        }
    }
    if (threadIdx.x < BLOCK_M) {
        int r = threadIdx.x;
        int gr = i_start + r;
        smem_L[r] = (gr < S) ? L_bh[gr] : 0.0f;
    }
    for (int idx = threadIdx.x; idx < BLOCK_M * D; idx += blockDim.x) {
        smem_dQ[idx / D][idx % D] = 0.0f;
    }
    __syncthreads();

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int wm = warp_id / 2;
    int wn = warp_id % 2;

    {
        int row_in_warp = lane_id / 4;
        int sub = lane_id % 4;
        int row = warp_id * 8 + row_in_warp;
        float acc_val = 0.0f;
        for (int k = sub * 32; k < sub * 32 + 32; ++k) {
            acc_val += __bfloat162float(smem_dO[row][k]) * __bfloat162float(smem_O[row][k]);
        }
        acc_val = reduce_sum4(acc_val);
        if (sub == 0) smem_Di[row] = acc_val;
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_row;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_col;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_row;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;

    for (int j_start = 0; j_start <= i_start + BLOCK_M - 1 && j_start < S; j_start += BLOCK_N) {
        for (int idx = threadIdx.x; idx < BLOCK_N * D / 8; idx += blockDim.x) {
            int r = idx / (D / 8);
            int c = (idx % (D / 8)) * 8;
            int gr = j_start + r;
            if (gr < S) {
                *reinterpret_cast<float4*>(&smem_K[r][c]) = *reinterpret_cast<const float4*>(&K_bh[gr * D + c]);
                *reinterpret_cast<float4*>(&smem_V[r][c]) = *reinterpret_cast<const float4*>(&V_bh[gr * D + c]);
            } else {
                *reinterpret_cast<float4*>(&smem_K[r][c]) = make_float4(0,0,0,0);
                *reinterpret_cast<float4*>(&smem_V[r][c]) = make_float4(0,0,0,0);
            }
        }
        __syncthreads();

        wmma::fill_fragment(acc, 0.0f);
        for (int k = 0; k < D; k += WMMA_K) {
            wmma::load_matrix_sync(a_row, &smem_Q[wm*16][k], D);
            wmma::load_matrix_sync(b_col, &smem_K[wn*16][k], D);
            wmma::mma_sync(acc, a_row, b_col, acc);
        }
        wmma::store_matrix_sync(&smem_P[wm*16][wn*16], acc, BLOCK_N, wmma::mem_row_major);

        wmma::fill_fragment(acc, 0.0f);
        for (int k = 0; k < D; k += WMMA_K) {
            wmma::load_matrix_sync(a_row, &smem_dO[wm*16][k], D);
            wmma::load_matrix_sync(b_col, &smem_V[wn*16][k], D);
            wmma::mma_sync(acc, a_row, b_col, acc);
        }
        wmma::store_matrix_sync(&smem_dS[wm*16][wn*16], acc, BLOCK_N, wmma::mem_row_major);
        __syncthreads();

        for (int idx = threadIdx.x; idx < BLOCK_M * BLOCK_N; idx += blockDim.x) {
            int r = idx / BLOCK_N;
            int c = idx % BLOCK_N;
            int i_row = i_start + r;
            int j_col = j_start + c;
            bool valid = (i_row < S && j_col < S && i_row >= j_col);
            float p = valid ? __expf(smem_P[r][c] * scale - smem_L[r]) : 0.0f;
            smem_P[r][c] = p;
            float ds = p * (smem_dS[r][c] - smem_Di[r]) * scale;
            smem_dS[r][c] = ds;
            smem_dS_bf16[r][c] = __float2bfloat16(ds);
        }
        __syncthreads();

        for (int t = warp_id; t < 16; t += 4) {
            int ib = t / 8;
            int db = t % 8;

            wmma::load_matrix_sync(acc, &smem_dQ[ib*16][db*16], D, wmma::mem_row_major);
            for (int jb = 0; jb < 2; jb++) {
                wmma::load_matrix_sync(a_row, &smem_dS_bf16[ib*16][jb*16], BLOCK_N);
                wmma::load_matrix_sync(b_row, &smem_K[jb*16][db*16], D);
                wmma::mma_sync(acc, a_row, b_row, acc);
            }
            wmma::store_matrix_sync(&smem_dQ[ib*16][db*16], acc, D, wmma::mem_row_major);
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