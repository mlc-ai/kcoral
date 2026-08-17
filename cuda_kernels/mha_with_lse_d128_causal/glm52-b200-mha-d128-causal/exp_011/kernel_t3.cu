#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <stdio.h>
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

namespace tvm_ffi_mha {

constexpr int BR = 64;
constexpr int BC = 64;
constexpr int D = 128;
constexpr int M_TILE = 16;
constexpr int N_TILE = 16;
constexpr int K_TILE = 16;

constexpr int S_STRIDE = BC + 1;
constexpr int P_STRIDE = BC + 2;
constexpr int V_STRIDE = BC + 2;

constexpr int SMEM_SIZE =
    BR * D * 2 +
    BC * D * 2 +
    D * V_STRIDE * 2 +
    BR * S_STRIDE * 4 +
    BR * P_STRIDE * 2 +
    BR * D * 4 +
    BR * 4 * 3;

__device__ __forceinline__ float fast_exp2f_(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__global__ void __launch_bounds__(128, 2)
mha_causal_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int H, int S) {

    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    int64_t base = (int64_t)(b * H + h) * S * D;

    const __nv_bfloat16* Q_bh = Q + base;
    const __nv_bfloat16* K_bh = K + base;
    const __nv_bfloat16* V_bh = V + base;
    __nv_bfloat16* O_bh = O + base;
    float* LSE_bh = LSE + (int64_t)(b * H + h) * S;

    int warp_id = threadIdx.x / 32;

    extern __shared__ char smem[];
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sK = sQ + BR * D;
    __nv_bfloat16* sV = sK + BC * D;
    float* sS = reinterpret_cast<float*>(sV + D * V_STRIDE);
    __nv_bfloat16* sP = reinterpret_cast<__nv_bfloat16*>(sS + BR * S_STRIDE);
    float* sO = reinterpret_cast<float*>(sP + BR * P_STRIDE);
    float* sM = sO + BR * D;
    float* sL = sM + BR;
    float* sAlpha = sL + BR;

    const float scale_log2e = 0.12754844765175826f;
    const float inv_log2e = 0.6931471805599453f;

    for (int qi = 0; qi < S; qi += BR) {
        int q_rows = min(BR, S - qi);

        for (int i = threadIdx.x; i < BR * D / 8; i += 128) {
            int r = i / (D / 8);
            int d8 = i % (D / 8);
            if (qi + r < S) {
                reinterpret_cast<int4*>(&sQ[r * D])[d8] =
                    reinterpret_cast<const int4*>(&Q_bh[(int64_t)(qi + r) * D])[d8];
            } else {
                reinterpret_cast<int4*>(&sQ[r * D])[d8] = make_int4(0, 0, 0, 0);
            }
        }

        for (int i = threadIdx.x; i < BR * D; i += 128) {
            sO[i] = 0.0f;
        }
        for (int i = threadIdx.x; i < BR; i += 128) {
            sM[i] = -INFINITY;
            sL[i] = 0.0f;
        }
        __syncthreads();

        int max_k = min(qi + q_rows, S);

        for (int kj = 0; kj < max_k; kj += BC) {
            int k_cols = min(BC, S - kj);

            for (int i = threadIdx.x; i < BC * D / 8; i += 128) {
                int r = i / (D / 8);
                int d8 = i % (D / 8);
                if (kj + r < S) {
                    reinterpret_cast<int4*>(&sK[r * D])[d8] =
                        reinterpret_cast<const int4*>(&K_bh[(int64_t)(kj + r) * D])[d8];
                } else {
                    reinterpret_cast<int4*>(&sK[r * D])[d8] = make_int4(0, 0, 0, 0);
                }
            }

            for (int i = threadIdx.x; i < BC * D / 8; i += 128) {
                int r = i / (D / 8);
                int d8 = i % (D / 8);
                int d_base = d8 * 8;
                if (kj + r < S) {
                    int4 val = reinterpret_cast<const int4*>(&V_bh[(int64_t)(kj + r) * D])[d8];
                    __nv_bfloat16* bf16_val = reinterpret_cast<__nv_bfloat16*>(&val);
                    #pragma unroll
                    for (int k = 0; k < 8; k++) {
                        sV[(d_base + k) * V_STRIDE + r] = bf16_val[k];
                    }
                } else {
                    #pragma unroll
                    for (int k = 0; k < 8; k++) {
                        sV[(d_base + k) * V_STRIDE + r] = __float2bfloat16(0.0f);
                    }
                }
            }
            __syncthreads();

            {
                int m_tile = warp_id;
                for (int n_tile = 0; n_tile < BC / N_TILE; n_tile++) {
                    wmma::fragment<wmma::accumulator, M_TILE, N_TILE, K_TILE, float> c_frag;
                    wmma::fill_fragment(c_frag, 0.0f);

                    for (int k_tile = 0; k_tile < D / K_TILE; k_tile++) {
                        wmma::fragment<wmma::matrix_a, M_TILE, N_TILE, K_TILE, __nv_bfloat16, wmma::row_major> a_frag;
                        wmma::fragment<wmma::matrix_b, M_TILE, N_TILE, K_TILE, __nv_bfloat16, wmma::col_major> b_frag;

                        wmma::load_matrix_sync(a_frag, &sQ[m_tile * M_TILE * D + k_tile * K_TILE], D);
                        wmma::load_matrix_sync(b_frag, &sK[n_tile * N_TILE * D + k_tile * K_TILE], D);
                        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                    }

                    wmma::store_matrix_sync(&sS[m_tile * M_TILE * S_STRIDE + n_tile * N_TILE], c_frag, S_STRIDE, wmma::mem_row_major);
                }
            }
            __syncthreads();

            if (threadIdx.x < BR) {
                int row = threadIdx.x;
                if (row < q_rows) {
                    int q_pos = qi + row;

                    float m_old = sM[row];
                    float m_new = m_old;

                    for (int j = 0; j < BC; j++) {
                        float s = sS[row * S_STRIDE + j] * scale_log2e;
                        int k_pos = kj + j;
                        if (k_pos > q_pos || k_pos >= S) {
                            s = -INFINITY;
                        }
                        sS[row * S_STRIDE + j] = s;
                        m_new = fmaxf(m_new, s);
                    }

                    float alpha = (m_old == -INFINITY) ? 0.0f : fast_exp2f_(m_old - m_new);

                    float l_new = 0.0f;
                    for (int j = 0; j < BC; j++) {
                        float p = fast_exp2f_(sS[row * S_STRIDE + j] - m_new);
                        sP[row * P_STRIDE + j] = __float2bfloat16(p);
                        l_new += p;
                    }

                    sAlpha[row] = alpha;
                    sM[row] = m_new;
                    sL[row] = sL[row] * alpha + l_new;
                } else {
                    sAlpha[row] = 1.0f;
                }
            }
            __syncthreads();

            for (int i = threadIdx.x; i < BR * D; i += 128) {
                int row = i / D;
                sO[i] *= sAlpha[row];
            }
            __syncthreads();

            {
                int m_tile = warp_id;
                for (int n_tile = 0; n_tile < D / N_TILE; n_tile++) {
                    wmma::fragment<wmma::accumulator, M_TILE, N_TILE, K_TILE, float> c_frag;
                    wmma::load_matrix_sync(c_frag, &sO[m_tile * M_TILE * D + n_tile * N_TILE], D, wmma::mem_row_major);

                    for (int k_tile = 0; k_tile < BC / K_TILE; k_tile++) {
                        wmma::fragment<wmma::matrix_a, M_TILE, N_TILE, K_TILE, __nv_bfloat16, wmma::row_major> a_frag;
                        wmma::fragment<wmma::matrix_b, M_TILE, N_TILE, K_TILE, __nv_bfloat16, wmma::col_major> b_frag;

                        wmma::load_matrix_sync(a_frag, &sP[m_tile * M_TILE * P_STRIDE + k_tile * K_TILE], P_STRIDE);
                        wmma::load_matrix_sync(b_frag, &sV[n_tile * N_TILE * V_STRIDE + k_tile * K_TILE], V_STRIDE);
                        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                    }

                    wmma::store_matrix_sync(&sO[m_tile * M_TILE * D + n_tile * N_TILE], c_frag, D, wmma::mem_row_major);
                }
            }
            __syncthreads();
        }

        for (int i = threadIdx.x; i < q_rows * D; i += 128) {
            int row = i / D;
            int d = i % D;
            float val = sO[row * D + d] / sL[row];
            O_bh[(int64_t)(qi + row) * D + d] = __float2bfloat16(val);
        }
        if (threadIdx.x < q_rows) {
            LSE_bh[qi + threadIdx.x] = (sM[threadIdx.x] + log2f(sL[threadIdx.x])) * inv_log2e;
        }

        __syncthreads();
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);

    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    if (S == 0) return;

    int grid = B * H;
    int block = 128;

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(mha_causal_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));

    mha_causal_kernel<<<grid, block, SMEM_SIZE, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data, H, S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha::run);

}  // namespace tvm_ffi_mha