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
constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int NUM_THREADS = 128;
constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
constexpr float scale = 0.08838834764f; // 1.0f / sqrtf(128.0f)

__device__ __forceinline__ float reduce_sum4(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 1);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    return val;
}

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
    uint32_t smem_int = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem_int), "l"(gmem));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::: "memory");
}

__device__ void load_tile_async(__nv_bfloat16* dst, const __nv_bfloat16* src, int rows, int thread_id, int S, int tile_start) {
    for (int i = thread_id; i < rows * 16; i += blockDim.x) {
        int r = i / 16;
        int c = (i % 16) * 8;
        int gr = tile_start + r;
        if (gr < S) {
            cp_async_16(&dst[r * D + c], &src[r * D + c]);
        } else {
            uint32_t smem_int = (uint32_t)__cvta_generic_to_shared(&dst[r * D + c]);
            uint32_t zero = 0;
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n" :: "r"(smem_int), "r"(zero), "r"(zero), "r"(zero), "r"(zero));
        }
    }
}

__device__ void compute_Di(__nv_bfloat16* dO, __nv_bfloat16* O, float* Di) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    for (int row_base = warp_id; row_base < BLOCK_M / 8; row_base += 4) {
        int row = row_base * 8 + lane_id / 4;
        int sub = lane_id % 4;
        float acc = 0.0f;
        for (int k = sub * 32; k < sub * 32 + 32; ++k) {
            acc += __bfloat162float(dO[row * D + k]) * __bfloat162float(O[row * D + k]);
        }
        acc = reduce_sum4(acc);
        if (sub == 0) Di[row] = acc;
    }
}

__device__ void compute_P_dS(
    __nv_bfloat16* Q, __nv_bfloat16* K, __nv_bfloat16* dO, __nv_bfloat16* V,
    float* Di, float* L, int i_start, int j_start, int S, float* P, float* dS) {
    
    int warp_id = threadIdx.x / 32;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_row_Q, a_row_dO;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_col_K, b_col_V;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_P, acc_dS;

    for (int t = 0; t < 4; ++t) {
        int m = (warp_id * 4 + t) / 4;
        int n = (warp_id * 4 + t) % 4;
        
        wmma::fill_fragment(acc_P, 0.0f);
        wmma::fill_fragment(acc_dS, 0.0f);
        
        for (int k = 0; k < D; k += WMMA_K) {
            wmma::load_matrix_sync(a_row_Q, &Q[m*16*D + k], D);
            wmma::load_matrix_sync(b_col_K, &K[n*16*D + k], D);
            wmma::load_matrix_sync(a_row_dO, &dO[m*16*D + k], D);
            wmma::load_matrix_sync(b_col_V, &V[n*16*D + k], D);
            wmma::mma_sync(acc_P, a_row_Q, b_col_K, acc_P);
            wmma::mma_sync(acc_dS, a_row_dO, b_col_V, acc_dS);
        }
        wmma::store_matrix_sync(&P[m*16*BLOCK_N + n*16], acc_P, BLOCK_N, wmma::mem_row_major);
        wmma::store_matrix_sync(&dS[m*16*BLOCK_N + n*16], acc_dS, BLOCK_N, wmma::mem_row_major);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < BLOCK_M * BLOCK_N; idx += blockDim.x) {
        int r = idx / BLOCK_N;
        int c = idx % BLOCK_N;
        int i_row = i_start + r;
        int j_col = j_start + c;
        bool valid = (i_row < S && j_col < S && i_row >= j_col);
        float p = valid ? __expf(P[r*BLOCK_N + c] * scale - L[r]) : 0.0f;
        P[r*BLOCK_N + c] = p;
        dS[r*BLOCK_N + c] = p * (dS[r*BLOCK_N + c] - Di[r]) * scale;
    }
}

__device__ void convert_bf16(float* P, float* dS, __nv_bfloat16* P_bf16, __nv_bfloat16* dS_bf16) {
    for (int idx = threadIdx.x; idx < BLOCK_M * BLOCK_N; idx += blockDim.x) {
        int r = idx / BLOCK_N;
        int c = idx % BLOCK_N;
        P_bf16[r * BLOCK_N + c] = __float2bfloat16(P[r * BLOCK_N + c]);
        dS_bf16[r * BLOCK_N + c] = __float2bfloat16(dS[r * BLOCK_N + c]);
    }
}

__device__ void convert_dS_bf16(float* dS, __nv_bfloat16* dS_bf16) {
    for (int idx = threadIdx.x; idx < BLOCK_M * BLOCK_N; idx += blockDim.x) {
        int r = idx / BLOCK_N;
        int c = idx % BLOCK_N;
        dS_bf16[r * BLOCK_N + c] = __float2bfloat16(dS[r * BLOCK_N + c]);
    }
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

    __shared__ __nv_bfloat16 smem_K[BLOCK_N][D];
    __shared__ __nv_bfloat16 smem_V[BLOCK_N][D];
    __shared__ __nv_bfloat16 smem_Q[2][BLOCK_M][D];
    __shared__ __nv_bfloat16 smem_dO[2][BLOCK_M][D];
    __shared__ __nv_bfloat16 smem_O[BLOCK_M][D];
    __shared__ float smem_P[BLOCK_M][BLOCK_N];
    __shared__ float smem_dS[BLOCK_M][BLOCK_N];
    __shared__ __nv_bfloat16 smem_P_bf16[BLOCK_M][BLOCK_N];
    __shared__ __nv_bfloat16 smem_dS_bf16[BLOCK_M][BLOCK_N];
    __shared__ float smem_dV[BLOCK_N][D];
    __shared__ float smem_dK[BLOCK_N][D];
    __shared__ float smem_Di[BLOCK_M];
    __shared__ float smem_L[BLOCK_M];

    load_tile_async((__nv_bfloat16*)smem_K, &K_bh[j_start * D], BLOCK_N, threadIdx.x, S, j_start);
    load_tile_async((__nv_bfloat16*)smem_V, &V_bh[j_start * D], BLOCK_N, threadIdx.x, S, j_start);
    load_tile_async((__nv_bfloat16*)smem_Q[0], &Q_bh[j_start * D], BLOCK_M, threadIdx.x, S, j_start);
    load_tile_async((__nv_bfloat16*)smem_dO[0], &dO_bh[j_start * D], BLOCK_M, threadIdx.x, S, j_start);
    load_tile_async((__nv_bfloat16*)smem_O, &O_bh[j_start * D], BLOCK_M, threadIdx.x, S, j_start);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    int warp_id = threadIdx.x / 32;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> a_col_P, a_col_dS;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_row_dO, b_row_Q;
    
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_dV[8];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_dK[8];
    
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        wmma::fill_fragment(acc_dV[t], 0.0f);
        wmma::fill_fragment(acc_dK[t], 0.0f);
    }

    for (int i = j_start; i < S; i += BLOCK_M) {
        int buf = (i / BLOCK_M) % 2;
        if (i != j_start) {
            cp_async_wait_all();
            __syncthreads();
        }

        if (threadIdx.x < BLOCK_M) {
            int r = threadIdx.x;
            int gr = i + r;
            smem_L[r] = (gr < S) ? L_bh[gr] : 0.0f;
        }
        __syncthreads();

        compute_Di((__nv_bfloat16*)smem_dO[buf], (__nv_bfloat16*)smem_O, smem_Di);
        __syncthreads();

        int next_i = i + BLOCK_M;
        if (next_i < S) {
            int buf_next = (next_i / BLOCK_M) % 2;
            load_tile_async((__nv_bfloat16*)smem_Q[buf_next], &Q_bh[next_i * D], BLOCK_M, threadIdx.x, S, next_i);
            load_tile_async((__nv_bfloat16*)smem_dO[buf_next], &dO_bh[next_i * D], BLOCK_M, threadIdx.x, S, next_i);
            load_tile_async((__nv_bfloat16*)smem_O, &O_bh[next_i * D], BLOCK_M, threadIdx.x, S, next_i);
            cp_async_commit();
        }

        compute_P_dS((__nv_bfloat16*)smem_Q[buf], (__nv_bfloat16*)smem_K, (__nv_bfloat16*)smem_dO[buf], (__nv_bfloat16*)smem_V,
                     smem_Di, smem_L, i, j_start, S, &smem_P[0][0], &smem_dS[0][0]);
        __syncthreads();
        
        convert_bf16(&smem_P[0][0], &smem_dS[0][0], &smem_P_bf16[0][0], &smem_dS_bf16[0][0]);
        __syncthreads();

        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            int jb = (warp_id * 8 + t) / 8;
            int db = (warp_id * 8 + t) % 8;
            for (int ib = 0; ib < 4; ++ib) {
                wmma::load_matrix_sync(a_col_P, &smem_P_bf16[ib*16][jb*16], BLOCK_N);
                wmma::load_matrix_sync(b_row_dO, &smem_dO[buf][ib*16][db*16], D);
                wmma::load_matrix_sync(a_col_dS, &smem_dS_bf16[ib*16][jb*16], BLOCK_N);
                wmma::load_matrix_sync(b_row_Q, &smem_Q[buf][ib*16][db*16], D);
                wmma::mma_sync(acc_dV[t], a_col_P, b_row_dO, acc_dV[t]);
                wmma::mma_sync(acc_dK[t], a_col_dS, b_row_Q, acc_dK[t]);
            }
        }
    }

    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        int jb = (warp_id * 8 + t) / 8;
        int db = (warp_id * 8 + t) % 8;
        wmma::store_matrix_sync(&smem_dV[jb*16][db*16], acc_dV[t], D, wmma::mem_row_major);
        wmma::store_matrix_sync(&smem_dK[jb*16][db*16], acc_dK[t], D, wmma::mem_row_major);
    }
    __syncthreads();

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

    __shared__ __nv_bfloat16 smem_Q[BLOCK_M][D];
    __shared__ __nv_bfloat16 smem_dO[BLOCK_M][D];
    __shared__ __nv_bfloat16 smem_O[BLOCK_M][D];
    __shared__ __nv_bfloat16 smem_K[2][BLOCK_N][D];
    __shared__ __nv_bfloat16 smem_V[2][BLOCK_N][D];
    __shared__ float smem_P[BLOCK_M][BLOCK_N];
    __shared__ float smem_dS[BLOCK_M][BLOCK_N];
    __shared__ __nv_bfloat16 smem_dS_bf16[BLOCK_M][BLOCK_N];
    __shared__ float smem_dQ[BLOCK_M][D];
    __shared__ float smem_Di[BLOCK_M];
    __shared__ float smem_L[BLOCK_M];

    load_tile_async((__nv_bfloat16*)smem_Q, &Q_bh[i_start * D], BLOCK_M, threadIdx.x, S, i_start);
    load_tile_async((__nv_bfloat16*)smem_dO, &dO_bh[i_start * D], BLOCK_M, threadIdx.x, S, i_start);
    load_tile_async((__nv_bfloat16*)smem_O, &O_bh[i_start * D], BLOCK_M, threadIdx.x, S, i_start);
    if (threadIdx.x < BLOCK_M) {
        int r = threadIdx.x;
        int gr = i_start + r;
        smem_L[r] = (gr < S) ? L_bh[gr] : 0.0f;
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    compute_Di((__nv_bfloat16*)smem_dO, (__nv_bfloat16*)smem_O, smem_Di);
    __syncthreads();

    int warp_id = threadIdx.x / 32;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_row_dS;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_row_K;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_dQ[8];

    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        wmma::fill_fragment(acc_dQ[t], 0.0f);
    }

    for (int j = 0; j <= i_start + BLOCK_M - 1 && j < S; j += BLOCK_N) {
        int buf = (j / BLOCK_N) % 2;
        if (j == 0) {
            load_tile_async((__nv_bfloat16*)smem_K[0], &K_bh[j * D], BLOCK_N, threadIdx.x, S, j);
            load_tile_async((__nv_bfloat16*)smem_V[0], &V_bh[j * D], BLOCK_N, threadIdx.x, S, j);
            cp_async_commit();
            cp_async_wait_all();
            __syncthreads();
        }

        int next_j = j + BLOCK_N;
        if (next_j <= i_start + BLOCK_M - 1 && next_j < S) {
            int buf_next = (next_j / BLOCK_N) % 2;
            load_tile_async((__nv_bfloat16*)smem_K[buf_next], &K_bh[next_j * D], BLOCK_N, threadIdx.x, S, next_j);
            load_tile_async((__nv_bfloat16*)smem_V[buf_next], &V_bh[next_j * D], BLOCK_N, threadIdx.x, S, next_j);
            cp_async_commit();
        }

        compute_P_dS((__nv_bfloat16*)smem_Q, (__nv_bfloat16*)smem_K[buf], (__nv_bfloat16*)smem_dO, (__nv_bfloat16*)smem_V[buf],
                     smem_Di, smem_L, i_start, j, S, &smem_P[0][0], &smem_dS[0][0]);
        __syncthreads();
        
        convert_dS_bf16(&smem_dS[0][0], &smem_dS_bf16[0][0]);
        __syncthreads();

        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            int ib = (warp_id * 8 + t) / 8;
            int db = (warp_id * 8 + t) % 8;
            for (int jb = 0; jb < 4; ++jb) {
                wmma::load_matrix_sync(a_row_dS, &smem_dS_bf16[ib*16][jb*16], BLOCK_N);
                wmma::load_matrix_sync(b_row_K, &smem_K[buf][jb*16][db*16], D);
                wmma::mma_sync(acc_dQ[t], a_row_dS, b_row_K, acc_dQ[t]);
            }
        }

        if (next_j <= i_start + BLOCK_M - 1 && next_j < S) {
            cp_async_wait_all();
            __syncthreads();
        }
    }

    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        int ib = (warp_id * 8 + t) / 8;
        int db = (warp_id * 8 + t) % 8;
        wmma::store_matrix_sync(&smem_dQ[ib*16][db*16], acc_dQ[t], D, wmma::mem_row_major);
    }
    __syncthreads();

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