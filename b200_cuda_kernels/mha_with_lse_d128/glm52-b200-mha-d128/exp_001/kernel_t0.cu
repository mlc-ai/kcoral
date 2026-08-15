#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;
using bf16 = __nv_bfloat16;

constexpr int D = 128;
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int WT = 16;

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

__device__ __forceinline__ float fast_expf(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x * 1.44269504088896340736f));
    return y;
}

__global__ void attn_kernel(
    const bf16* __restrict__ Q,
    const bf16* __restrict__ K,
    const bf16* __restrict__ V,
    bf16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S)
{
    int total_q = (S + BM - 1) / BM;
    int q_block = blockIdx.x % total_q;
    int bh = blockIdx.x / total_q;
    int h = bh % H;
    int b = bh / H;

    int q_start = q_block * BM;
    int tid = threadIdx.x;
    int warp_id = tid / 32;

    const float scale = 0.08838834764f; // 1/sqrt(128)

    extern __shared__ char smem_raw[];
    bf16*  Q_smem  = reinterpret_cast<bf16*>(smem_raw);                     // [BM][D]   32 KB
    bf16*  KV_smem = Q_smem + BM * D;                                       // [BN][D]   32 KB
    float* S_smem  = reinterpret_cast<float*>(KV_smem + BN * D);            // [BM][BN]  64 KB
    bf16*  P_smem  = reinterpret_cast<bf16*>(S_smem + BM * BN);             // [BM][BN]  32 KB
    float* O_smem  = reinterpret_cast<float*>(P_smem + BM * BN);            // [BM][D]   64 KB
    // total: 224 KB

    const bf16* Q_base = Q + (size_t)(b * H + h) * S * D;
    const bf16* K_base = K + (size_t)(b * H + h) * S * D;
    const bf16* V_base = V + (size_t)(b * H + h) * S * D;
    bf16*  O_base     = O + (size_t)(b * H + h) * S * D;
    float* LSE_base   = LSE + (size_t)(b * H + h) * S;

    // ---- Load Q tile with scaling, zero-pad OOB rows ----
    for (int i = tid * 8; i < BM * D; i += 128 * 8) {
        int row = i / D, col = i % D;
        int g_row = q_start + row;
        if (g_row < S) {
            int4 raw = *reinterpret_cast<const int4*>(&Q_base[(size_t)g_row * D + col]);
            bf16* vals = reinterpret_cast<bf16*>(&raw);
            #pragma unroll
            for (int v = 0; v < 8; v++)
                vals[v] = __float2bfloat16(__bfloat162float(vals[v]) * scale);
            *reinterpret_cast<int4*>(&Q_smem[i]) = raw;
        } else {
            *reinterpret_cast<int4*>(&Q_smem[i]) = make_int4(0, 0, 0, 0);
        }
    }

    // ---- Init O_smem = 0 ----
    for (int i = tid; i < BM * D; i += 128)
        O_smem[i] = 0.0f;

    float row_max = -INFINITY;
    float row_sum = 0.0f;

    __syncthreads();

    int num_kv_blocks = (S + BN - 1) / BN;

    for (int kv = 0; kv < num_kv_blocks; kv++) {
        int kv_start = kv * BN;

        // ---- Load K tile [BN][D] ----
        for (int i = tid * 8; i < BN * D; i += 128 * 8) {
            int row = i / D, col = i % D;
            int g_row = kv_start + row;
            if (g_row < S)
                *reinterpret_cast<int4*>(&KV_smem[i]) =
                    *reinterpret_cast<const int4*>(&K_base[(size_t)g_row * D + col]);
            else
                *reinterpret_cast<int4*>(&KV_smem[i]) = make_int4(0, 0, 0, 0);
        }
        __syncthreads();

        // ---- Compute S = Q @ K^T  (wmma 16x16x16, BF16->FP32) ----
        // 4 warps: warp w handles rows [w*32, w*32+32), all 128 cols
        wmma::fragment<wmma::accumulator, WT, WT, WT, float> s_frag[2][8];
        #pragma unroll
        for (int mi = 0; mi < 2; mi++)
            #pragma unroll
            for (int ni = 0; ni < 8; ni++)
                wmma::fill_fragment(s_frag[mi][ni], 0.0f);

        #pragma unroll
        for (int ki = 0; ki < D / WT; ki++) {          // 8 k-tiles
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                wmma::fragment<wmma::matrix_a, WT, WT, WT, bf16, wmma::row_major> q_frag;
                wmma::load_matrix_sync(q_frag,
                    Q_smem + (warp_id * 32 + mi * WT) * D + ki * WT, D);

                #pragma unroll
                for (int ni = 0; ni < 8; ni++) {
                    wmma::fragment<wmma::matrix_b, WT, WT, WT, bf16, wmma::col_major> k_frag;
                    // K_smem [BN][D] row-major == K^T [D][BN] col-major, ld=D
                    wmma::load_matrix_sync(k_frag,
                        KV_smem + ni * WT * D + ki * WT, D);
                    wmma::mma_sync(s_frag[mi][ni], q_frag, k_frag, s_frag[mi][ni]);
                }
            }
        }

        // ---- Store S to S_smem ----
        #pragma unroll
        for (int mi = 0; mi < 2; mi++)
            #pragma unroll
            for (int ni = 0; ni < 8; ni++)
                wmma::store_matrix_sync(
                    S_smem + (warp_id * 32 + mi * WT) * BN + ni * WT,
                    s_frag[mi][ni], BN, wmma::mem_row_major);
        __syncthreads();

        // ---- Online softmax: thread tid owns row tid ----
        float m_old = row_max;
        float m_new = m_old;
        for (int j = 0; j < BN; j++) {
            int kv_idx = kv_start + j;
            float s_val = (kv_idx < S) ? S_smem[tid * BN + j] : -INFINITY;
            m_new = fmaxf(m_new, s_val);
        }

        float rescale = fast_expf(m_old - m_new);

        // Rescale O for this row
        for (int d = 0; d < D; d++)
            O_smem[tid * D + d] *= rescale;

        // Compute P = exp(S - m_new), accumulate block_sum
        float block_sum = 0.0f;
        for (int j = 0; j < BN; j++) {
            int kv_idx = kv_start + j;
            float s_val = (kv_idx < S) ? S_smem[tid * BN + j] : -INFINITY;
            float p_val = fast_expf(s_val - m_new);
            block_sum += p_val;
            P_smem[tid * BN + j] = __float2bfloat16(p_val);
        }

        row_sum = row_sum * rescale + block_sum;
        row_max = m_new;

        __syncthreads();

        // ---- Load V tile [BN][D] (reuse KV_smem) ----
        for (int i = tid * 8; i < BN * D; i += 128 * 8) {
            int row = i / D, col = i % D;
            int g_row = kv_start + row;
            if (g_row < S)
                *reinterpret_cast<int4*>(&KV_smem[i]) =
                    *reinterpret_cast<const int4*>(&V_base[(size_t)g_row * D + col]);
            else
                *reinterpret_cast<int4*>(&KV_smem[i]) = make_int4(0, 0, 0, 0);
        }
        __syncthreads();

        // ---- Compute O += P @ V  (wmma 16x16x16, BF16->FP32) ----
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            #pragma unroll
            for (int ni = 0; ni < D / WT; ni++) {       // 8 col-tiles
                wmma::fragment<wmma::accumulator, WT, WT, WT, float> o_frag;
                wmma::load_matrix_sync(o_frag,
                    O_smem + (warp_id * 32 + mi * WT) * D + ni * WT, D,
                    wmma::mem_row_major);

                #pragma unroll
                for (int ki = 0; ki < BN / WT; ki++) {  // 8 k-tiles
                    wmma::fragment<wmma::matrix_a, WT, WT, WT, bf16, wmma::row_major> p_frag;
                    wmma::load_matrix_sync(p_frag,
                        P_smem + (warp_id * 32 + mi * WT) * BN + ki * WT, BN);

                    wmma::fragment<wmma::matrix_b, WT, WT, WT, bf16, wmma::row_major> v_frag;
                    wmma::load_matrix_sync(v_frag,
                        KV_smem + ki * WT * D + ni * WT, D);

                    wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
                }

                wmma::store_matrix_sync(
                    O_smem + (warp_id * 32 + mi * WT) * D + ni * WT,
                    o_frag, D, wmma::mem_row_major);
            }
        }
        __syncthreads();
    }

    // ---- Final normalisation and output ----
    int q_row = q_start + tid;
    if (q_row < S) {
        float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
        for (int d = 0; d < D; d += 8) {
            bf16 out_vals[8];
            #pragma unroll
            for (int v = 0; v < 8; v++)
                out_vals[v] = __float2bfloat16(O_smem[tid * D + d + v] * inv_sum);
            *reinterpret_cast<int4*>(&O_base[(size_t)q_row * D + d]) =
                *reinterpret_cast<int4*>(out_vals);
        }
        LSE_base[q_row] = (row_sum > 0.0f) ? (row_max + logf(row_sum)) : -INFINITY;
    }
}

namespace attn_impl {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);

    const bf16* Q_ptr = static_cast<const bf16*>(Q.data_ptr());
    const bf16* K_ptr = static_cast<const bf16*>(K.data_ptr());
    const bf16* V_ptr = static_cast<const bf16*>(V.data_ptr());
    bf16*  O_ptr     = static_cast<bf16*>(O.data_ptr());
    float* LSE_ptr   = static_cast<float*>(LSE.data_ptr());

    int total_q = (S + BM - 1) / BM;
    int blocks  = B * H * total_q;
    int threads = 128;

    // Shared: Q(32K) + KV(32K) + S(64K) + P(32K) + O(64K) = 224 KB
    size_t smem_size = (size_t)BM * D * 2 + (size_t)BN * D * 2 +
                       (size_t)BM * BN * 4 + (size_t)BM * BN * 2 +
                       (size_t)BM * D * 4;

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    cudaFuncSetAttribute(attn_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size);

    attn_kernel<<<blocks, threads, smem_size, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr, B, H, S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, attn_impl::run);

}  // namespace attn_impl