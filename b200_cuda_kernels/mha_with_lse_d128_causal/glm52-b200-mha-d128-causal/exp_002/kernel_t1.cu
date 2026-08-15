#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;

#define CUDA_CHECK(call) do {                                       \
    cudaError_t _e = (call);                                        \
    if (_e != cudaSuccess) {                                        \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                 \
                cudaGetErrorString(_e), __FILE__, __LINE__);        \
        exit(1);                                                    \
    }                                                               \
} while (0)

namespace flash_attn {

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int D = 128;
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int WM = 16, WN = 16, WK = 16;

constexpr int SMEM_Q_SIZE  = BM * D * sizeof(__nv_bfloat16);
constexpr int SMEM_KV_SIZE = BN * D * sizeof(__nv_bfloat16);
constexpr int SMEM_S_SIZE  = BM * BN * sizeof(float);
constexpr int SMEM_P_SIZE  = BM * BN * sizeof(__nv_bfloat16);
constexpr int SMEM_STATS   = 3 * BM * sizeof(float);

__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem_src));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::: "memory");
}

__global__ void flash_attn_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S)
{
    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    int q_block = blockIdx.y;
    int q_start = q_block * BM;

    if (q_start >= S) return;

    int q_end = min(q_start + BM, S);
    int q_rows = q_end - q_start;

    const __nv_bfloat16* Q_bh = Q + (size_t)(b * H + h) * S * D;
    const __nv_bfloat16* K_bh = K + (size_t)(b * H + h) * S * D;
    const __nv_bfloat16* V_bh = V + (size_t)(b * H + h) * S * D;
    __nv_bfloat16* O_bh = O + (size_t)(b * H + h) * S * D;
    float* LSE_bh = LSE + (size_t)(b * H + h) * S;

    extern __shared__ char smem[];
    __nv_bfloat16* smem_q  = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* smem_kv = smem_q + BM * D;
    float* smem_s          = reinterpret_cast<float*>(smem_kv + BN * D);
    __nv_bfloat16* smem_p  = reinterpret_cast<__nv_bfloat16*>(smem_s + BM * BN);
    float* smem_m          = reinterpret_cast<float*>(smem_p + BM * BN);
    float* smem_l          = smem_m + BM;
    float* smem_rescale    = smem_l + BM;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    const float scale = 0.08838834764831845f;

    // Load Q tile
    for (int i = tid * 8; i < BM * D; i += THREADS * 8) {
        int row = i / D, col = i % D;
        if (row < q_rows) {
            cp_async_16(smem_q + i, Q_bh + (q_start + row) * D + col);
        } else {
            *reinterpret_cast<uint4*>(smem_q + i) = make_uint4(0, 0, 0, 0);
        }
    }
    cp_async_commit();
    cp_async_wait_all();

    if (tid < BM) {
        smem_m[tid] = -INFINITY;
        smem_l[tid] = 0.0f;
    }

    wmma::fragment<wmma::accumulator, WM, WN, WK, float> o_frag[D / WN];
    #pragma unroll
    for (int i = 0; i < D / WN; i++) wmma::fill_fragment(o_frag[i], 0.0f);

    __syncthreads();

    for (int kv_start = 0; kv_start < q_end; kv_start += BN) {
        int kv_len = min(kv_start + BN, S) - kv_start;
        bool need_mask = (kv_start + BN > q_start);

        // Load K tile
        for (int i = tid * 8; i < BN * D; i += THREADS * 8) {
            int row = i / D, col = i % D;
            int load_row = min(row, kv_len - 1);
            cp_async_16(smem_kv + i, K_bh + (kv_start + load_row) * D + col);
        }
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();

        // S = Q @ K^T
        {
            wmma::fragment<wmma::accumulator, WM, WN, WK, float> s_frag[BN / WN];
            wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::col_major> b_frag;

            #pragma unroll
            for (int i = 0; i < BN / WN; i++) wmma::fill_fragment(s_frag[i], 0.0f);

            #pragma unroll
            for (int k = 0; k < D / WK; k++) {
                wmma::load_matrix_sync(a_frag, smem_q + warp_id * WM * D + k * WK, D);
                #pragma unroll
                for (int n = 0; n < BN / WN; n++) {
                    wmma::load_matrix_sync(b_frag, smem_kv + n * WN * D + k * WK, D);
                    wmma::mma_sync(s_frag[n], a_frag, b_frag, s_frag[n]);
                }
            }

            #pragma unroll
            for (int n = 0; n < BN / WN; n++) {
                wmma::store_matrix_sync(smem_s + warp_id * WM * BN + n * WN, s_frag[n], BN, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // Softmax
        if (tid < BM) {
            int row = tid;
            int q_row = q_start + row;

            float row_max = -INFINITY;
            for (int j = 0; j < BN; j++) {
                int k_col = kv_start + j;
                float s = smem_s[row * BN + j] * scale;
                if (k_col >= S || (need_mask && k_col > q_row)) s = -INFINITY;
                smem_s[row * BN + j] = s;
                row_max = fmaxf(row_max, s);
            }

            float m_old = smem_m[row];
            float m_new = fmaxf(m_old, row_max);
            float rescale = (m_old == -INFINITY) ? 0.0f : __expf(m_old - m_new);

            float row_sum = 0.0f;
            for (int j = 0; j < BN; j++) {
                float s = smem_s[row * BN + j];
                float p = (s == -INFINITY) ? 0.0f : __expf(s - m_new);
                smem_p[row * BN + j] = __float2bfloat16(p);
                row_sum += p;
            }

            smem_m[row] = m_new;
            smem_l[row] = smem_l[row] * rescale + row_sum;
            smem_rescale[row] = rescale;
        }
        __syncthreads();

        // Rescale O: store to smem_s, rescale, load back
        #pragma unroll
        for (int n = 0; n < D / WN; n++) {
            wmma::store_matrix_sync(smem_s + warp_id * WM * D + n * WN, o_frag[n], D, wmma::mem_row_major);
        }
        __syncthreads();

        for (int i = tid; i < BM * D; i += THREADS) {
            int row = i / D;
            smem_s[i] *= smem_rescale[row];
        }
        __syncthreads();

        #pragma unroll
        for (int n = 0; n < D / WN; n++) {
            wmma::load_matrix_sync(o_frag[n], smem_s + warp_id * WM * D + n * WN, D, wmma::mem_row_major);
        }

        // Load V tile
        for (int i = tid * 8; i < BN * D; i += THREADS * 8) {
            int row = i / D, col = i % D;
            int load_row = min(row, kv_len - 1);
            cp_async_16(smem_kv + i, V_bh + (kv_start + load_row) * D + col);
        }
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();

        // O += P @ V
        {
            wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> p_frag;
            wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::col_major> b_frag;

            #pragma unroll
            for (int k = 0; k < BN / WK; k++) {
                wmma::load_matrix_sync(p_frag, smem_p + warp_id * WM * BN + k * WK, BN);
                #pragma unroll
                for (int n = 0; n < D / WN; n++) {
                    wmma::load_matrix_sync(b_frag, smem_kv + k * WK * D + n * WN, D);
                    wmma::mma_sync(o_frag[n], p_frag, b_frag, o_frag[n]);
                }
            }
        }
        __syncthreads();
    }

    // Store O to shared memory (reuse smem_q space)
    {
        float* smem_o = reinterpret_cast<float*>(smem);
        #pragma unroll
        for (int n = 0; n < D / WN; n++) {
            wmma::store_matrix_sync(smem_o + warp_id * WM * D + n * WN, o_frag[n], D, wmma::mem_row_major);
        }
    }
    __syncthreads();

    // Normalize and store to global
    {
        float* smem_o = reinterpret_cast<float*>(smem);
        for (int i = tid; i < BM * D; i += THREADS) {
            int row = i / D, col = i % D;
            if (q_start + row < S) {
                float l = smem_l[row];
                float o_val = smem_o[row * D + col];
                O_bh[(q_start + row) * D + col] = __float2bfloat16(
                    (l > 0.0f) ? (o_val / l) : 0.0f);
            }
        }
    }

    // Store LSE
    if (tid < BM) {
        int row = tid;
        if (q_start + row < S) {
            float l = smem_l[row];
            float m = smem_m[row];
            LSE_bh[q_start + row] = (l > 0.0f) ? (m + logf(l)) : -INFINITY;
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);

    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    dim3 grid(B * H, (S + BM - 1) / BM);
    dim3 block(THREADS);
    int smem_bytes = SMEM_Q_SIZE + SMEM_KV_SIZE + SMEM_S_SIZE + SMEM_P_SIZE + SMEM_STATS;

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    cudaFuncSetAttribute(flash_attn_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    flash_attn_kernel<<<grid, block, smem_bytes, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data, B, H, S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flash_attn::run);

}  // namespace flash_attn