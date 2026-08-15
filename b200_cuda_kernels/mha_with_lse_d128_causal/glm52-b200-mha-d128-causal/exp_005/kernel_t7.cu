#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cmath>
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

namespace mha_kernel {

constexpr int D = 128;
constexpr int BM = 64;
constexpr int BK = 64;
constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
constexpr int NUM_WARPS = BM / WMMA_M;
constexpr int THREADS = NUM_WARPS * 32;

constexpr int SQ_STRIDE = D;
constexpr int SK_STRIDE = D;
constexpr int SV_STRIDE = D;
constexpr int SP_STRIDE = BK;
constexpr int SO_STRIDE = D;

__global__ void attn_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int H, int S)
{
    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    int qb = blockIdx.y;
    int q_start = qb * BM;

    int64_t off = (int64_t)(b * H + h) * (int64_t)S * D;
    const __nv_bfloat16* Qg = Q + off;
    const __nv_bfloat16* Kg = K + off;
    const __nv_bfloat16* Vg = V + off;
    __nv_bfloat16* Og = O + off;
    float* LSEg = LSE + (int64_t)(b * H + h) * (int64_t)S;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid & 31;
    int warp_row = warp_id * WMMA_M;

    extern __shared__ char smem_buf[];
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem_buf);
    __nv_bfloat16* sK = sQ + BM * SQ_STRIDE;
    __nv_bfloat16* sV = sK + BK * SK_STRIDE;
    float* sS = reinterpret_cast<float*>(sV + BK * SV_STRIDE);
    __nv_bfloat16* sP = reinterpret_cast<__nv_bfloat16*>(sS + BM * BK);
    float* sO = reinterpret_cast<float*>(sP + BM * SP_STRIDE);
    float* s_m = sO + BM * SO_STRIDE;
    float* s_l = s_m + BM;
    float* s_alpha = s_l + BM;

    const float scale = 0.08838834764831845f;

    // Load Q tile
    for (int i = tid; i < BM * D; i += THREADS) {
        int r = i / D, d = i % D;
        int gr = q_start + r;
        sQ[r * SQ_STRIDE + d] = (gr < S) ? Qg[(int64_t)gr * D + d] : __float2bfloat16(0.f);
    }

    // Initialize O, m, l
    for (int i = tid; i < BM * D; i += THREADS)
        sO[i] = 0.0f;
    for (int i = tid; i < BM; i += THREADS) {
        s_m[i] = -INFINITY;
        s_l[i] = 0.0f;
    }
    __syncthreads();

    int block_max_q = min(q_start + BM - 1, S - 1);
    if (block_max_q < 0) block_max_q = 0;
    int last_kb = block_max_q / BK;

    for (int kb = 0; kb <= last_kb; kb++) {
        int k_start = kb * BK;

        // Load K and V tiles
        for (int i = tid; i < BK * D; i += THREADS) {
            int k = i / D, d = i % D;
            int gr = k_start + k;
            if (gr < S) {
                sK[k * SK_STRIDE + d] = Kg[(int64_t)gr * D + d];
                sV[k * SV_STRIDE + d] = Vg[(int64_t)gr * D + d];
            } else {
                sK[k * SK_STRIDE + d] = __float2bfloat16(0.f);
                sV[k * SV_STRIDE + d] = __float2bfloat16(0.f);
            }
        }
        __syncthreads();

        // S = Q @ K^T using wmma
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_frag[4];
        for (int i = 0; i < 4; i++) wmma::fill_fragment(s_frag[i], 0.0f);

        for (int k_iter = 0; k_iter < D / WMMA_K; k_iter++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, &sQ[warp_row * SQ_STRIDE + k_iter * WMMA_K], SQ_STRIDE);

            for (int n_tile = 0; n_tile < BK / WMMA_N; n_tile++) {
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;
                wmma::load_matrix_sync(b_frag, &sK[n_tile * WMMA_N * SK_STRIDE + k_iter * WMMA_K], SK_STRIDE);
                wmma::mma_sync(s_frag[n_tile], a_frag, b_frag, s_frag[n_tile]);
            }
        }

        // Store S to shared memory
        for (int n_tile = 0; n_tile < 4; n_tile++)
            wmma::store_matrix_sync(&sS[warp_row * BK + n_tile * WMMA_N], s_frag[n_tile], BK, wmma::mem_row_major);
        __syncthreads();

        // Apply scale and causal mask
        for (int i = tid; i < BM * BK; i += THREADS) {
            int r = i / BK, c = i % BK;
            int q = q_start + r;
            int k = k_start + c;
            float val = sS[r * BK + c] * scale;
            if (k > q || q >= S) val = -INFINITY;
            sS[r * BK + c] = val;
        }
        __syncthreads();

        // Online softmax: rowmax
        int row0 = warp_row + lane / 4;
        int row1 = warp_row + lane / 4 + 8;
        int q0 = q_start + row0;
        int q1 = q_start + row1;
        int col_start = (lane % 4) * 16;

        float local_max0 = -INFINITY, local_max1 = -INFINITY;
        for (int i = 0; i < 16; i++) {
            int c = col_start + i;
            local_max0 = fmaxf(local_max0, sS[row0 * BK + c]);
            local_max1 = fmaxf(local_max1, sS[row1 * BK + c]);
        }
        local_max0 = fmaxf(local_max0, __shfl_xor_sync(0xffffffff, local_max0, 1));
        local_max0 = fmaxf(local_max0, __shfl_xor_sync(0xffffffff, local_max0, 2));
        local_max1 = fmaxf(local_max1, __shfl_xor_sync(0xffffffff, local_max1, 1));
        local_max1 = fmaxf(local_max1, __shfl_xor_sync(0xffffffff, local_max1, 2));

        float m_old0 = s_m[row0];
        float m_old1 = s_m[row1];
        float m_new0 = fmaxf(m_old0, local_max0);
        float m_new1 = fmaxf(m_old1, local_max1);
        float alpha0 = (m_old0 == -INFINITY) ? 0.f : __expf(m_old0 - m_new0);
        float alpha1 = (m_old1 == -INFINITY) ? 0.f : __expf(m_old1 - m_new1);

        if (lane % 4 == 0) {
            s_m[row0] = m_new0;
            s_m[row1] = m_new1;
            s_alpha[row0] = alpha0;
            s_alpha[row1] = alpha1;
        }
        __syncthreads();

        // Rescale O in shared memory
        for (int i = tid; i < BM * D; i += THREADS) {
            int r = i / D;
            sO[i] *= s_alpha[r];
        }
        __syncthreads();

        // Compute P = exp(S - m), rowsum
        float row_sum0 = 0.f, row_sum1 = 0.f;
        for (int i = 0; i < 16; i++) {
            int c = col_start + i;
            float s0 = sS[row0 * BK + c];
            float s1 = sS[row1 * BK + c];
            float p0 = (s0 == -INFINITY) ? 0.f : __expf(s0 - m_new0);
            float p1 = (s1 == -INFINITY) ? 0.f : __expf(s1 - m_new1);
            sS[row0 * BK + c] = p0;
            sS[row1 * BK + c] = p1;
            row_sum0 += p0;
            row_sum1 += p1;
        }
        row_sum0 += __shfl_xor_sync(0xffffffff, row_sum0, 1);
        row_sum0 += __shfl_xor_sync(0xffffffff, row_sum0, 2);
        row_sum1 += __shfl_xor_sync(0xffffffff, row_sum1, 1);
        row_sum1 += __shfl_xor_sync(0xffffffff, row_sum1, 2);

        if (lane % 4 == 0) {
            s_l[row0] = s_l[row0] * alpha0 + row_sum0;
            s_l[row1] = s_l[row1] * alpha1 + row_sum1;
        }
        __syncthreads();

        // Convert P to bf16 in sP
        for (int i = tid; i < BM * BK; i += THREADS) {
            int r = i / BK, c = i % BK;
            sP[r * SP_STRIDE + c] = __float2bfloat16(sS[r * BK + c]);
        }
        __syncthreads();

        // Load O from shared memory to fragments
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag[8];
        for (int n_tile = 0; n_tile < D / WMMA_N; n_tile++)
            wmma::load_matrix_sync(o_frag[n_tile], &sO[warp_row * SO_STRIDE + n_tile * WMMA_N], SO_STRIDE, wmma::mem_row_major);

        // O += P @ V
        for (int k_iter = 0; k_iter < BK / WMMA_K; k_iter++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, &sP[warp_row * SP_STRIDE + k_iter * WMMA_K], SP_STRIDE);

            for (int n_tile = 0; n_tile < D / WMMA_N; n_tile++) {
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;
                wmma::load_matrix_sync(b_frag, &sV[k_iter * WMMA_K * SV_STRIDE + n_tile * WMMA_N], SV_STRIDE);
                wmma::mma_sync(o_frag[n_tile], a_frag, b_frag, o_frag[n_tile]);
            }
        }

        // Store O back to shared memory
        for (int n_tile = 0; n_tile < D / WMMA_N; n_tile++)
            wmma::store_matrix_sync(&sO[warp_row * SO_STRIDE + n_tile * WMMA_N], o_frag[n_tile], SO_STRIDE, wmma::mem_row_major);
        __syncthreads();
    }

    // Normalize O and write to global
    for (int i = tid; i < BM * D; i += THREADS) {
        int r = i / D, d = i % D;
        int q = q_start + r;
        if (q < S) {
            float val = sO[r * SO_STRIDE + d] / s_l[r];
            Og[(int64_t)q * D + d] = __float2bfloat16(val);
        }
    }

    // Write LSE
    for (int i = tid; i < BM; i += THREADS) {
        int q = q_start + i;
        if (q < S) LSEg[q] = s_m[i] + logf(s_l[i]);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    const int B = 4;
    const int H = 48;
    int S = (int)Q.size(2);

    const __nv_bfloat16* Qd = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* Kd = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* Vd = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* Od = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSEd = static_cast<float*>(LSE.data_ptr());

    int smem_bytes = (int)(
        BM * SQ_STRIDE * sizeof(__nv_bfloat16) +
        BK * SK_STRIDE * sizeof(__nv_bfloat16) +
        BK * SV_STRIDE * sizeof(__nv_bfloat16) +
        BM * BK * sizeof(float) +
        BM * SP_STRIDE * sizeof(__nv_bfloat16) +
        BM * SO_STRIDE * sizeof(float) +
        BM * sizeof(float) * 3);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(
        attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    dim3 grid(B * H, (S + BM - 1) / BM);
    dim3 block(THREADS);
    attn_kernel<<<grid, block, smem_bytes, stream>>>(Qd, Kd, Vd, Od, LSEd, H, S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);

}  // namespace mha_kernel