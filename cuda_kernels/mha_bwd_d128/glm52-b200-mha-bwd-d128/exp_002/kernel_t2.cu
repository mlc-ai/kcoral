#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace mha_bwd {

constexpr int D = 128;
constexpr int B_R = 16;
constexpr int B_C = 64;
constexpr int THREADS = 128;
constexpr float SCALE = 0.07071067811865475f;

// ============ Pass 1: dV and Di ============
__global__ void dv_di_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dV,
    float* __restrict__ Di,
    int H, int S) {

    int bh = blockIdx.x;
    int b = bh / H, h = bh % H;
    size_t off = (size_t)(b * H + h) * S * D;
    const __nv_bfloat16* Q_bh = Q + off;
    const __nv_bfloat16* K_bh = K + off;
    const __nv_bfloat16* V_bh = V + off;
    const __nv_bfloat16* dO_bh = dO + off;
    const float* L_bh = L + (size_t)(b * H + h) * S;
    __nv_bfloat16* dV_bh = dV + off;
    float* Di_bh = Di + (size_t)(b * H + h) * S;

    int tid = threadIdx.x;
    int warp_id = tid / 32;

    extern __shared__ char smem[];
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sK = sQ + B_R * D;
    __nv_bfloat16* sV = sK + B_C * D;
    __nv_bfloat16* sdO = sV + B_C * D;
    float* sS = reinterpret_cast<float*>(sdO + B_R * D);
    float* sP = sS + B_R * B_C;
    float* sdP = sP + B_R * B_C;
    float* sdV = sdP + B_R * B_C;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_row;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_col;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a_col;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_row;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> d_frag[8];

    #pragma unroll
    for (int i = 0; i < 8; i++) wmma::fill_fragment(d_frag[i], 0.f);

    for (int kv_start = 0; kv_start < S; kv_start += B_C) {
        for (int i = tid; i < B_C * D; i += THREADS) {
            int j = i / D, d = i % D;
            sK[i] = (kv_start + j < S) ? K_bh[(size_t)(kv_start + j) * D + d] : __float2bfloat16(0.f);
            sV[i] = (kv_start + j < S) ? V_bh[(size_t)(kv_start + j) * D + d] : __float2bfloat16(0.f);
        }
        __syncthreads();

        for (int q_start = 0; q_start < S; q_start += B_R) {
            for (int i = tid; i < B_R * D; i += THREADS) {
                int r = i / D, d = i % D;
                sQ[i] = (q_start + r < S) ? Q_bh[(size_t)(q_start + r) * D + d] : __float2bfloat16(0.f);
                sdO[i] = (q_start + r < S) ? dO_bh[(size_t)(q_start + r) * D + d] : __float2bfloat16(0.f);
            }
            __syncthreads();

            // S = Q @ K^T (each warp: one 16x16 tile of 16x64)
            wmma::fill_fragment(c_frag, 0.f);
            for (int kk = 0; kk < D; kk += 16) {
                wmma::load_matrix_sync(a_row, &sQ[kk], D);
                wmma::load_matrix_sync(b_col, &sK[warp_id * 16 * D + kk], D);
                wmma::mma_sync(c_frag, a_row, b_col, c_frag);
            }
            wmma::store_matrix_sync(&sS[warp_id * 16], c_frag, B_C, wmma::mem_row_major);
            __syncthreads();

            // P = exp(S * scale - L)
            for (int i = tid; i < B_R * B_C; i += THREADS) {
                int r = i / B_C, c = i % B_C;
                sP[i] = (q_start + r < S && kv_start + c < S)
                    ? __expf(sS[i] * SCALE - L_bh[q_start + r]) : 0.f;
            }
            __syncthreads();

            // dP = dO @ V^T
            wmma::fill_fragment(c_frag, 0.f);
            for (int kk = 0; kk < D; kk += 16) {
                wmma::load_matrix_sync(a_row, &sdO[kk], D);
                wmma::load_matrix_sync(b_col, &sV[warp_id * 16 * D + kk], D);
                wmma::mma_sync(c_frag, a_row, b_col, c_frag);
            }
            wmma::store_matrix_sync(&sdP[warp_id * 16], c_frag, B_C, wmma::mem_row_major);
            __syncthreads();

            // Di += rowsum(P * dP)
            for (int i = tid; i < B_R; i += THREADS) {
                if (q_start + i < S) {
                    float di = 0.f;
                    for (int j = 0; j < B_C; j++) di += sP[i * B_C + j] * sdP[i * B_C + j];
                    atomicAdd(&Di_bh[q_start + i], di);
                }
            }

            // Convert P to bf16 (reuse sS buffer)
            __nv_bfloat16* sP_bf16 = reinterpret_cast<__nv_bfloat16*>(sS);
            for (int i = tid; i < B_R * B_C; i += THREADS)
                sP_bf16[i] = __float2bfloat16(sP[i]);
            __syncthreads();

            // dV += P^T @ dO (each warp: 4 M-tiles x 2 N-tiles = 8 tiles, K=16 = 1 step)
            for (int m = 0; m < 4; m++) {
                for (int n = 0; n < 2; n++) {
                    int idx = m * 2 + n;
                    int n_idx = warp_id * 2 + n;
                    wmma::load_matrix_sync(a_col, &sP_bf16[m * 16], B_C);
                    wmma::load_matrix_sync(b_row, &sdO[n_idx * 16], D);
                    wmma::mma_sync(d_frag[idx], a_col, b_row, d_frag[idx]);
                }
            }
            __syncthreads();
        }

        // Store dV fragments to shared memory then global
        for (int m = 0; m < 4; m++) {
            for (int n = 0; n < 2; n++) {
                int idx = m * 2 + n;
                int n_idx = warp_id * 2 + n;
                wmma::store_matrix_sync(&sdV[m * D + n_idx * 16], d_frag[idx], D, wmma::mem_row_major);
            }
        }
        __syncthreads();

        for (int i = tid; i < B_C * D; i += THREADS) {
            int j = i / D, d = i % D;
            if (kv_start + j < S) dV_bh[(size_t)(kv_start + j) * D + d] = __float2bfloat16(sdV[i]);
        }
        __syncthreads();
    }
}

// ============ Pass 2: dQ and dK ============
__global__ void dq_dk_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ Di,
    __nv_bfloat16* __restrict__ dQ,
    float* __restrict__ dK_ws,
    int H, int S) {

    int bh = blockIdx.x;
    int b = bh / H, h = bh % H;
    size_t off = (size_t)(b * H + h) * S * D;
    const __nv_bfloat16* Q_bh = Q + off;
    const __nv_bfloat16* K_bh = K + off;
    const __nv_bfloat16* V_bh = V + off;
    const __nv_bfloat16* dO_bh = dO + off;
    const float* L_bh = L + (size_t)(b * H + h) * S;
    const float* Di_bh = Di + (size_t)(b * H + h) * S;
    __nv_bfloat16* dQ_bh = dQ + off;
    float* dK_bh = dK_ws + off;

    int tid = threadIdx.x;
    int warp_id = tid / 32;

    extern __shared__ char smem[];
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sK = sQ + B_R * D;
    __nv_bfloat16* sV = sK + B_C * D;
    __nv_bfloat16* sdO = sV + B_C * D;
    float* sS = reinterpret_cast<float*>(sdO + B_R * D);
    float* sP = sS + B_R * B_C;
    float* sdP = sP + B_R * B_C;
    float* sdS = sdP + B_R * B_C;
    float* sdQ = sdS + B_R * B_C;
    float* sDi = sdQ + B_R * D;
    float* sdK_partial = sDi + B_R;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_row;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_col;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a_col;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_row;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dk_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dq_frag[2];

    for (int i = 0; i < 2; i++) wmma::fill_fragment(dq_frag[i], 0.f);

    for (int q_start = 0; q_start < S; q_start += B_R) {
        for (int i = tid; i < B_R * D; i += THREADS) {
            int r = i / D, d = i % D;
            sQ[i] = (q_start + r < S) ? Q_bh[(size_t)(q_start + r) * D + d] : __float2bfloat16(0.f);
            sdO[i] = (q_start + r < S) ? dO_bh[(size_t)(q_start + r) * D + d] : __float2bfloat16(0.f);
        }
        for (int i = tid; i < B_R; i += THREADS)
            sDi[i] = (q_start + i < S) ? Di_bh[q_start + i] : 0.f;
        __syncthreads();

        for (int kv_start = 0; kv_start < S; kv_start += B_C) {
            for (int i = tid; i < B_C * D; i += THREADS) {
                int j = i / D, d = i % D;
                sK[i] = (kv_start + j < S) ? K_bh[(size_t)(kv_start + j) * D + d] : __float2bfloat16(0.f);
                sV[i] = (kv_start + j < S) ? V_bh[(size_t)(kv_start + j) * D + d] : __float2bfloat16(0.f);
            }
            __syncthreads();

            // S = Q @ K^T
            wmma::fill_fragment(c_frag, 0.f);
            for (int kk = 0; kk < D; kk += 16) {
                wmma::load_matrix_sync(a_row, &sQ[kk], D);
                wmma::load_matrix_sync(b_col, &sK[warp_id * 16 * D + kk], D);
                wmma::mma_sync(c_frag, a_row, b_col, c_frag);
            }
            wmma::store_matrix_sync(&sS[warp_id * 16], c_frag, B_C, wmma::mem_row_major);
            __syncthreads();

            // P = exp(S * scale - L)
            for (int i = tid; i < B_R * B_C; i += THREADS) {
                int r = i / B_C, c = i % B_C;
                sP[i] = (q_start + r < S && kv_start + c < S)
                    ? __expf(sS[i] * SCALE - L_bh[q_start + r]) : 0.f;
            }
            __syncthreads();

            // dP = dO @ V^T
            wmma::fill_fragment(c_frag, 0.f);
            for (int kk = 0; kk < D; kk += 16) {
                wmma::load_matrix_sync(a_row, &sdO[kk], D);
                wmma::load_matrix_sync(b_col, &sV[warp_id * 16 * D + kk], D);
                wmma::mma_sync(c_frag, a_row, b_col, c_frag);
            }
            wmma::store_matrix_sync(&sdP[warp_id * 16], c_frag, B_C, wmma::mem_row_major);
            __syncthreads();

            // dS = P * (dP - Di)
            for (int i = tid; i < B_R * B_C; i += THREADS) {
                int r = i / B_C;
                sdS[i] = sP[i] * (sdP[i] - sDi[r]);
            }
            __syncthreads();

            // Convert dS to bf16 (reuse sS)
            __nv_bfloat16* sdS_bf16 = reinterpret_cast<__nv_bfloat16*>(sS);
            for (int i = tid; i < B_R * B_C; i += THREADS)
                sdS_bf16[i] = __float2bfloat16(sdS[i]);
            __syncthreads();

            // dQ += dS @ K (each warp: 2 N-tiles, 4 K-steps = 8 mma)
            for (int n = 0; n < 2; n++) {
                int n_idx = warp_id * 2 + n;
                for (int kk = 0; kk < 4; kk++) {
                    wmma::load_matrix_sync(a_row, &sdS_bf16[kk * 16], B_C);
                    wmma::load_matrix_sync(b_row, &sK[kk * 16 * D + n_idx * 16], D);
                    wmma::mma_sync(dq_frag[n], a_row, b_row, dq_frag[n]);
                }
            }

            // dK += dS^T @ Q (each warp: 4 M-tiles x 2 N-tiles, K=16 = 1 step)
            for (int m = 0; m < 4; m++) {
                for (int n = 0; n < 2; n++) {
                    int n_idx = warp_id * 2 + n;
                    wmma::fill_fragment(dk_frag, 0.f);
                    wmma::load_matrix_sync(a_col, &sdS_bf16[m * 16], B_C);
                    wmma::load_matrix_sync(b_row, &sQ[n_idx * 16], D);
                    wmma::mma_sync(dk_frag, a_col, b_row, dk_frag);
                    wmma::store_matrix_sync(&sdK_partial[m * D + n_idx * 16], dk_frag, D, wmma::mem_row_major);
                }
            }
            __syncthreads();

            // AtomicAdd dK_partial * scale to global
            for (int i = tid; i < B_C * D; i += THREADS) {
                int j = i / D, d = i % D;
                if (kv_start + j < S)
                    atomicAdd(&dK_bh[(size_t)(kv_start + j) * D + d], sdK_partial[i] * SCALE);
            }
            __syncthreads();
        }

        // Store dQ with scale
        for (int n = 0; n < 2; n++) {
            int n_idx = warp_id * 2 + n;
            wmma::store_matrix_sync(&sdQ[n_idx * 16], dq_frag[n], D, wmma::mem_row_major);
        }
        __syncthreads();

        for (int i = tid; i < B_R * D; i += THREADS) {
            int r = i / D, d = i % D;
            if (q_start + r < S) dQ_bh[(size_t)(q_start + r) * D + d] = __float2bfloat16(sdQ[i] * SCALE);
        }
        __syncthreads();
    }
}

__global__ void convert_f32_bf16(const float* __restrict__ src, __nv_bfloat16* __restrict__ dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2bfloat16(src[idx]);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = 4, H = 48;
    int S = (int)Q.size(2);
    int total = B * H;

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    float* Di = nullptr;
    float* dK_ws = nullptr;
    CUDA_CHECK(cudaMalloc(&Di, (size_t)total * S * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dK_ws, (size_t)total * S * D * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(Di, 0, (size_t)total * S * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dK_ws, 0, (size_t)total * S * D * sizeof(float), stream));

    size_t smem_dv = (size_t)(2 * B_R * D * 2 + 2 * B_C * D * 2 + 3 * B_R * B_C * 4 + B_C * D * 4);
    size_t smem_dq = (size_t)(2 * B_R * D * 2 + 2 * B_C * D * 2 + 4 * B_R * B_C * 4 + B_R * D * 4 + B_R * 4 + B_C * D * 4);

    CUDA_CHECK(cudaFuncSetAttribute(dv_di_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_dv));
    CUDA_CHECK(cudaFuncSetAttribute(dq_dk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_dq));

    dv_di_kernel<<<total, THREADS, smem_dv, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        Di, H, S);
    CUDA_CHECK(cudaGetLastError());

    dq_dk_kernel<<<total, THREADS, smem_dq, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        Di,
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        dK_ws, H, S);
    CUDA_CHECK(cudaGetLastError());

    int n_elems = total * S * D;
    convert_f32_bf16<<<(n_elems + 255) / 256, 256, 0, stream>>>(
        dK_ws, static_cast<__nv_bfloat16*>(dK.data_ptr()), n_elems);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(Di));
    CUDA_CHECK(cudaFree(dK_ws));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd