#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

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
    __shared__ float smem_dV[BLOCK_N][D];
    __shared__ float smem_dK[BLOCK_N][D];
    __shared__ float smem_Di[BLOCK_M];
    __shared__ float smem_L[BLOCK_M];

    // Load K and V for j tile
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

    // Initialize accumulators
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
        // Load Q, dO, O for i tile
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

        // Compute Di
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

        // Compute P and dS
        for (int r = 0; r < 8; ++r) {
            int row = warp_id * 8 + r;
            int i_row = i_start + row;
            float di = smem_Di[row];
            float l_val = smem_L[row];
            float qk = 0.0f;
            float dp = 0.0f;
            for (int k = 0; k < D; ++k) {
                qk += __bfloat162float(smem_Q[row][k]) * __bfloat162float(smem_K[lane_id][k]);
                dp += __bfloat162float(smem_dO[row][k]) * __bfloat162float(smem_V[lane_id][k]);
            }
            int j_col = j_start + lane_id;
            bool valid = (i_row < S && j_col < S && i_row >= j_col);
            float p = valid ? expf(qk * scale - l_val) : 0.0f;
            float ds = p * (dp - di);
            smem_P[row][lane_id] = p;
            smem_dS[row][lane_id] = ds;
        }
        __syncthreads();

        // Accumulate dV and dK
        for (int c = 0; c < 8; ++c) {
            int col = warp_id * 8 + c;
            for (int d = lane_id * 4; d < lane_id * 4 + 4; ++d) {
                float acc_v = 0.0f;
                float acc_k = 0.0f;
                for (int r = 0; r < BLOCK_M; ++r) {
                    acc_v += smem_P[r][col] * __bfloat162float(smem_dO[r][d]);
                    acc_k += smem_dS[r][col] * __bfloat162float(smem_Q[r][d]);
                }
                smem_dV[col][d] += acc_v;
                smem_dK[col][d] += acc_k * scale;
            }
        }
        __syncthreads();
    }

    // Store dV and dK
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
    __shared__ float smem_dS[BLOCK_M][BLOCK_N];
    __shared__ float smem_dQ[BLOCK_M][D];
    __shared__ float smem_Di[BLOCK_M];
    __shared__ float smem_L[BLOCK_M];

    // Load Q, dO, O for i tile
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
    // Initialize dQ
    for (int idx = threadIdx.x; idx < BLOCK_M * D; idx += blockDim.x) {
        int r = idx / D;
        int c = idx % D;
        smem_dQ[r][c] = 0.0f;
    }
    __syncthreads();

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // Compute Di (depends only on i tile)
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
        // Load K, V for j tile
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

        // Compute P and dS
        for (int r = 0; r < 8; ++r) {
            int row = warp_id * 8 + r;
            int i_row = i_start + row;
            float di = smem_Di[row];
            float l_val = smem_L[row];
            float qk = 0.0f;
            float dp = 0.0f;
            for (int k = 0; k < D; ++k) {
                qk += __bfloat162float(smem_Q[row][k]) * __bfloat162float(smem_K[lane_id][k]);
                dp += __bfloat162float(smem_dO[row][k]) * __bfloat162float(smem_V[lane_id][k]);
            }
            int j_col = j_start + lane_id;
            bool valid = (i_row < S && j_col < S && i_row >= j_col);
            float p = valid ? expf(qk * scale - l_val) : 0.0f;
            float ds = p * (dp - di);
            smem_dS[row][lane_id] = ds;
        }
        __syncthreads();

        // Accumulate dQ
        for (int r = 0; r < 8; ++r) {
            int row = warp_id * 8 + r;
            for (int d = lane_id * 4; d < lane_id * 4 + 4; ++d) {
                float acc = 0.0f;
                for (int c = 0; c < BLOCK_N; ++c) {
                    acc += smem_dS[row][c] * __bfloat162float(smem_K[c][d]);
                }
                smem_dQ[row][d] += acc * scale;
            }
        }
        __syncthreads();
    }

    // Store dQ
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
    // D is fixed to 128

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