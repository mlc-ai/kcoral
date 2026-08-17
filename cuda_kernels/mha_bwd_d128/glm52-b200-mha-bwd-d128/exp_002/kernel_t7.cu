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
constexpr int B_R = 64;
constexpr int B_C = 64;
constexpr int THREADS = 128;
constexpr float SCALE = 0.08838834764831843f;

// ============ Kernel 1: dV and Di ============
__global__ void dv_di_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dV,
    float* __restrict__ Di,
    int H, int S) {

    int num_kv = S / B_C;
    int bh = blockIdx.x / num_kv;
    int kv_idx = blockIdx.x % num_kv;
    int b = bh / H, h = bh % H;
    int kv_start = kv_idx * B_C;

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
    float* sdP = sS + B_R * B_C;
    float* sdV = sdP + B_R * B_C;
    __nv_bfloat16* sP_bf16 = reinterpret_cast<__nv_bfloat16*>(sS);

    for (int i = tid; i < B_C * D; i += THREADS) {
        int j = i / D, d = i % D;
        sK[i] = K_bh[(size_t)(kv_start + j) * D + d];
        sV[i] = V_bh[(size_t)(kv_start + j) * D + d];
    }
    for (int i = tid; i < B_C * D; i += THREADS) sdV[i] = 0.f;
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_row;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_col;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a_col;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_row;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dv_frag[8];
    for (int i = 0; i < 8; i++) wmma::fill_fragment(dv_frag[i], 0.f);

    for (int q_start = 0; q_start < S; q_start += B_R) {
        for (int i = tid; i < B_R * D; i += THREADS) {
            int r = i / D, d = i % D;
            sQ[i] = Q_bh[(size_t)(q_start + r) * D + d];
            sdO[i] = dO_bh[(size_t)(q_start + r) * D + d];
        }
        __syncthreads();

        for (int n = 0; n < B_C / 16; n++) {
            wmma::fill_fragment(c_frag, 0.f);
            for (int k = 0; k < D; k += 16) {
                wmma::load_matrix_sync(a_row, &sQ[warp_id * 16 * D + k], D);
                wmma::load_matrix_sync(b_col, &sK[n * 16 * D + k], D);
                wmma::mma_sync(c_frag, a_row, b_col, c_frag);
            }
            wmma::store_matrix_sync(&sS[warp_id * 16 * B_C + n * 16], c_frag, B_C, wmma::mem_row_major);
        }
        __syncthreads();

        for (int i = tid; i < B_R * B_C; i += THREADS) {
            int r = i / B_C;
            sS[i] = expf(sS[i] * SCALE - L_bh[q_start + r]);
        }
        __syncthreads();

        for (int n = 0; n < B_C / 16; n++) {
            wmma::fill_fragment(c_frag, 0.f);
            for (int k = 0; k < D; k += 16) {
                wmma::load_matrix_sync(a_row, &sdO[warp_id * 16 * D + k], D);
                wmma::load_matrix_sync(b_col, &sV[n * 16 * D + k], D);
                wmma::mma_sync(c_frag, a_row, b_col, c_frag);
            }
            wmma::store_matrix_sync(&sdP[warp_id * 16 * B_C + n * 16], c_frag, B_C, wmma::mem_row_major);
        }
        __syncthreads();

        for (int i = tid; i < B_R; i += THREADS) {
            float di = 0.f;
            for (int j = 0; j < B_C; j++) di += sS[i * B_C + j] * sdP[i * B_C + j];
            atomicAdd(&Di_bh[q_start + i], di);
        }

        for (int i = tid; i < B_R * B_C; i += THREADS)
            sP_bf16[i] = __float2bfloat16(sS[i]);
        __syncthreads();

        for (int n = 0; n < D / 16; n++) {
            for (int k = 0; k < B_R / 16; k++) {
                wmma::load_matrix_sync(a_col, &sP_bf16[k * 16 * B_C + warp_id * 16], B_C);
                wmma::load_matrix_sync(b_row, &sdO[k * 16 * D + n * 16], D);
                wmma::mma_sync(dv_frag[n], a_col, b_row, dv_frag[n]);
            }
        }
        __syncthreads();
    }

    for (int n = 0; n < D / 16; n++)
        wmma::store_matrix_sync(&sdV[warp_id * 16 * D + n * 16], dv_frag[n], D, wmma::mem_row_major);
    __syncthreads();

    for (int i = tid; i < B_C * D; i += THREADS) {
        int j = i / D, d = i % D;
        dV_bh[(size_t)(kv_start + j) * D + d] = __float2bfloat16(sdV[i]);
    }
}

// ============ Kernel 2: dQ ============
__global__ void dq_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ Di,
    __nv_bfloat16* __restrict__ dQ,
    int H, int S) {

    int num_q = S / B_R;
    int bh = blockIdx.x / num_q;
    int q_idx = blockIdx.x % num_q;
    int b = bh / H, h = bh % H;
    int q_start = q_idx * B_R;

    size_t off = (size_t)(b * H + h) * S * D;
    const __nv_bfloat16* Q_bh = Q + off;
    const __nv_bfloat16* K_bh = K + off;
    const __nv_bfloat16* V_bh = V + off;
    const __nv_bfloat16* dO_bh = dO + off;
    const float* L_bh = L + (size_t)(b * H + h) * S;
    const float* Di_bh = Di + (size_t)(b * H + h) * S;
    __nv_bfloat16* dQ_bh = dQ + off;

    int tid = threadIdx.x;
    int warp_id = tid / 32;

    extern __shared__ char smem[];
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sK = sQ + B_R * D;
    __nv_bfloat16* sV = sK + B_C * D;
    __nv_bfloat16* sdO = sV + B_C * D;
    float* sS = reinterpret_cast<float*>(sdO + B_R * D);
    float* sdP = sS + B_R * B_C;
    float* sDi = sdP + B_R * B_C;
    float* sdQ = sDi + B_R;
    __nv_bfloat16* sdS_bf16 = reinterpret_cast<__nv_bfloat16*>(sS);

    for (int i = tid; i < B_R * D; i += THREADS) {
        int r = i / D, d = i % D;
        sQ[i] = Q_bh[(size_t)(q_start + r) * D + d];
        sdO[i] = dO_bh[(size_t)(q_start + r) * D + d];
    }
    for (int i = tid; i < B_R; i += THREADS)
        sDi[i] = Di_bh[q_start + i];
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_row;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_col;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_row;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dq_frag[8];
    for (int i = 0; i < 8; i++) wmma::fill_fragment(dq_frag[i], 0.f);

    for (int kv_start = 0; kv_start < S; kv_start += B_C) {
        for (int i = tid; i < B_C * D; i += THREADS) {
            int j = i / D, d = i % D;
            sK[i] = K_bh[(size_t)(kv_start + j) * D + d];
            sV[i] = V_bh[(size_t)(kv_start + j) * D + d];
        }
        __syncthreads();

        for (int n = 0; n < B_C / 16; n++) {
            wmma::fill_fragment(c_frag, 0.f);
            for (int k = 0; k < D; k += 16) {
                wmma::load_matrix_sync(a_row, &sQ[warp_id * 16 * D + k], D);
                wmma::load_matrix_sync(b_col, &sK[n * 16 * D + k], D);
                wmma::mma_sync(c_frag, a_row, b_col, c_frag);
            }
            wmma::store_matrix_sync(&sS[warp_id * 16 * B_C + n * 16], c_frag, B_C, wmma::mem_row_major);
        }
        __syncthreads();

        for (int i = tid; i < B_R * B_C; i += THREADS) {
            int r = i / B_C;
            sS[i] = expf(sS[i] * SCALE - L_bh[q_start + r]);
        }
        __syncthreads();

        for (int n = 0; n < B_C / 16; n++) {
            wmma::fill_fragment(c_frag, 0.f);
            for (int k = 0; k < D; k += 16) {
                wmma::load_matrix_sync(a_row, &sdO[warp_id * 16 * D + k], D);
                wmma::load_matrix_sync(b_col, &sV[n * 16 * D + k], D);
                wmma::mma_sync(c_frag, a_row, b_col, c_frag);
            }
            wmma::store_matrix_sync(&sdP[warp_id * 16 * B_C + n * 16], c_frag, B_C, wmma::mem_row_major);
        }
        __syncthreads();

        for (int i = tid; i < B_R * B_C; i += THREADS) {
            int r = i / B_C;
            float p = sS[i];
            sS[i] = p * (sdP[i] - sDi[r]);
        }
        __syncthreads();

        for (int i = tid; i < B_R * B_C; i += THREADS)
            sdS_bf16[i] = __float2bfloat16(sS[i]);
        __syncthreads();

        for (int n = 0; n < D / 16; n++) {
            for (int k = 0; k < B_C / 16; k++) {
                wmma::load_matrix_sync(a_row, &sdS_bf16[warp_id * 16 * B_C + k * 16], B_C);
                wmma::load_matrix_sync(b_row, &sK[k * 16 * D + n * 16], D);
                wmma::mma_sync(dq_frag[n], a_row, b_row, dq_frag[n]);
            }
        }
        __syncthreads();
    }

    for (int n = 0; n < D / 16; n++)
        wmma::store_matrix_sync(&sdQ[warp_id * 16 * D + n * 16], dq_frag[n], D, wmma::mem_row_major);
    __syncthreads();

    for (int i = tid; i < B_R * D; i += THREADS) {
        int r = i / D, d = i % D;
        dQ_bh[(size_t)(q_start + r) * D + d] = __float2bfloat16(sdQ[i] * SCALE);
    }
}

// ============ Kernel 3: dK ============
__global__ void dk_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ Di,
    __nv_bfloat16* __restrict__ dK,
    int H, int S) {

    int num_kv = S / B_C;
    int bh = blockIdx.x / num_kv;
    int kv_idx = blockIdx.x % num_kv;
    int b = bh / H, h = bh % H;
    int kv_start = kv_idx * B_C;

    size_t off = (size_t)(b * H + h) * S * D;
    const __nv_bfloat16* Q_bh = Q + off;
    const __nv_bfloat16* K_bh = K + off;
    const __nv_bfloat16* V_bh = V + off;
    const __nv_bfloat16* dO_bh = dO + off;
    const float* L_bh = L + (size_t)(b * H + h) * S;
    const float* Di_bh = Di + (size_t)(b * H + h) * S;
    __nv_bfloat16* dK_bh = dK + off;

    int tid = threadIdx.x;
    int warp_id = tid / 32;

    extern __shared__ char smem[];
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sK = sQ + B_R * D;
    __nv_bfloat16* sV = sK + B_C * D;
    __nv_bfloat16* sdO = sV + B_C * D;
    float* sS = reinterpret_cast<float*>(sdO + B_R * D);
    float* sdP = sS + B_R * B_C;
    float* sDi = sdP + B_R * B_C;
    float* sdK = sDi + B_R;
    __nv_bfloat16* sdS_bf16 = reinterpret_cast<__nv_bfloat16*>(sS);

    for (int i = tid; i < B_C * D; i += THREADS) {
        int j = i / D, d = i % D;
        sK[i] = K_bh[(size_t)(kv_start + j) * D + d];
        sV[i] = V_bh[(size_t)(kv_start + j) * D + d];
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_row;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_col;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a_col;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_row;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dk_frag[8];
    for (int i = 0; i < 8; i++) wmma::fill_fragment(dk_frag[i], 0.f);

    for (int q_start = 0; q_start < S; q_start += B_R) {
        for (int i = tid; i < B_R * D; i += THREADS) {
            int r = i / D, d = i % D;
            sQ[i] = Q_bh[(size_t)(q_start + r) * D + d];
            sdO[i] = dO_bh[(size_t)(q_start + r) * D + d];
        }
        for (int i = tid; i < B_R; i += THREADS)
            sDi[i] = Di_bh[q_start + i];
        __syncthreads();

        for (int n = 0; n < B_C / 16; n++) {
            wmma::fill_fragment(c_frag, 0.f);
            for (int k = 0; k < D; k += 16) {
                wmma::load_matrix_sync(a_row, &sQ[warp_id * 16 * D + k], D);
                wmma::load_matrix_sync(b_col, &sK[n * 16 * D + k], D);
                wmma::mma_sync(c_frag, a_row, b_col, c_frag);
            }
            wmma::store_matrix_sync(&sS[warp_id * 16 * B_C + n * 16], c_frag, B_C, wmma::mem_row_major);
        }
        __syncthreads();

        for (int i = tid; i < B_R * B_C; i += THREADS) {
            int r = i / B_C;
            sS[i] = expf(sS[i] * SCALE - L_bh[q_start + r]);
        }
        __syncthreads();

        for (int n = 0; n < B_C / 16; n++) {
            wmma::fill_fragment(c_frag, 0.f);
            for (int k = 0; k < D; k += 16) {
                wmma::load_matrix_sync(a_row, &sdO[warp_id * 16 * D + k], D);
                wmma::load_matrix_sync(b_col, &sV[n * 16 * D + k], D);
                wmma::mma_sync(c_frag, a_row, b_col, c_frag);
            }
            wmma::store_matrix_sync(&sdP[warp_id * 16 * B_C + n * 16], c_frag, B_C, wmma::mem_row_major);
        }
        __syncthreads();

        for (int i = tid; i < B_R * B_C; i += THREADS) {
            int r = i / B_C;
            float p = sS[i];
            sS[i] = p * (sdP[i] - sDi[r]);
        }
        __syncthreads();

        for (int i = tid; i < B_R * B_C; i += THREADS)
            sdS_bf16[i] = __float2bfloat16(sS[i]);
        __syncthreads();

        for (int n = 0; n < D / 16; n++) {
            for (int k = 0; k < B_R / 16; k++) {
                wmma::load_matrix_sync(a_col, &sdS_bf16[k * 16 * B_C + warp_id * 16], B_C);
                wmma::load_matrix_sync(b_row, &sQ[k * 16 * D + n * 16], D);
                wmma::mma_sync(dk_frag[n], a_col, b_row, dk_frag[n]);
            }
        }
        __syncthreads();
    }

    for (int n = 0; n < D / 16; n++)
        wmma::store_matrix_sync(&sdK[warp_id * 16 * D + n * 16], dk_frag[n], D, wmma::mem_row_major);
    __syncthreads();

    for (int i = tid; i < B_C * D; i += THREADS) {
        int j = i / D, d = i % D;
        dK_bh[(size_t)(kv_start + j) * D + d] = __float2bfloat16(sdK[i] * SCALE);
    }
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
    CUDA_CHECK(cudaMalloc(&Di, (size_t)total * S * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(Di, 0, (size_t)total * S * sizeof(float), stream));

    size_t smem_dv = (size_t)(B_R * D * 2 + B_C * D * 3 * 2 + B_R * B_C * 4 * 2 + B_C * D * 4);
    size_t smem_dq = (size_t)(B_R * D * 2 + B_C * D * 3 * 2 + B_R * B_C * 4 * 2 + B_R * 4 + B_R * D * 4);
    size_t smem_dk = smem_dq;

    CUDA_CHECK(cudaFuncSetAttribute(dv_di_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_dv));
    CUDA_CHECK(cudaFuncSetAttribute(dq_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_dq));
    CUDA_CHECK(cudaFuncSetAttribute(dk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_dk));

    int num_kv = S / B_C;
    int num_q = S / B_R;
    int grid_dv = total * num_kv;
    int grid_dq = total * num_q;
    int grid_dk = total * num_kv;

    dv_di_kernel<<<grid_dv, THREADS, smem_dv, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        Di, H, S);
    CUDA_CHECK(cudaGetLastError());

    dq_kernel<<<grid_dq, THREADS, smem_dq, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        Di,
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        H, S);
    CUDA_CHECK(cudaGetLastError());

    dk_kernel<<<grid_dk, THREADS, smem_dk, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        Di,
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        H, S);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(Di));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);

}  // namespace mha_bwd