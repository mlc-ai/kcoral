#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdio.h>
#include <mma.h>
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

namespace tvm_ffi_mha_cuda {

constexpr int D_HEAD = 128;
constexpr int BM = 64;
constexpr int BK = 64;
constexpr int NUM_THREADS = 128;
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int P_STRIDE = BK + 4;
constexpr float SCALE = 0.08838834764f;

constexpr int Q_SMEM_SIZE = BM * D_HEAD * 2;       // 16384
constexpr int KV_SMEM_SIZE = BK * D_HEAD * 2;      // 16384
constexpr int P_SMEM_SIZE = BM * P_STRIDE * 2;     // 8704
constexpr int ML_SMEM_SIZE = BM * 4 * 2;           // 512
constexpr int SMEM_SIZE = Q_SMEM_SIZE + KV_SMEM_SIZE * 2 + P_SMEM_SIZE + ML_SMEM_SIZE;

__device__ __forceinline__ void cp_async_16(void* dst, const void* src) {
    uint32_t d = (uint32_t)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
        :: "r"(d), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
template<int N>
__device__ __forceinline__ void cp_async_wait_n() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

__global__ __launch_bounds__(128, 4) void attention_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S, int B, int H, int num_q_blocks)
{
    const int block_idx = blockIdx.x;
    const int q_block_idx = block_idx % num_q_blocks;
    const int bh = block_idx / num_q_blocks;
    const int b = bh / H;
    const int h = bh % H;
    const int q_start = q_block_idx * BM;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int m_row_start = warp_id * WMMA_M;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* Q_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* K_smem = Q_smem + BM * D_HEAD;
    __nv_bfloat16* V_smem = K_smem + BK * D_HEAD;
    __nv_bfloat16* P_smem = V_smem + BK * D_HEAD;
    float* m_smem = reinterpret_cast<float*>(P_smem + BM * P_STRIDE);
    float* l_smem = m_smem + BM;

    const size_t bh_offset = (static_cast<size_t>(b) * H + h) * S * D_HEAD;
    const __nv_bfloat16* Q_bh = Q + bh_offset;
    const __nv_bfloat16* K_bh = K + bh_offset;
    const __nv_bfloat16* V_bh = V + bh_offset;
    __nv_bfloat16* O_bh = O + bh_offset;
    float* LSE_bh = LSE + (static_cast<size_t>(b) * H + h) * S;

    // ---- Load Q ----
    {
        const uint4* Q_g = reinterpret_cast<const uint4*>(Q_bh + static_cast<size_t>(q_start) * D_HEAD);
        uint4* Q_s = reinterpret_cast<uint4*>(Q_smem);
        const int total_vecs = BM * D_HEAD / 8;
        for (int i = tid; i < total_vecs; i += blockDim.x) {
            const int row = i / (D_HEAD / 8);
            Q_s[i] = (q_start + row < S) ? Q_g[i] : make_uint4(0, 0, 0, 0);
        }
    }

    if (tid < BM) {
        m_smem[tid] = -INFINITY;
        l_smem[tid] = 0.0f;
    }

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag[D_HEAD / WMMA_N];
    #pragma unroll
    for (int n = 0; n < D_HEAD / WMMA_N; n++) {
        wmma::fill_fragment(o_frag[n], 0.0f);
    }

    __syncthreads();

    const int row0 = m_row_start + lane_id / 4;
    const int row1 = m_row_start + lane_id / 4 + 8;
    const int sub_col = (lane_id % 4) * 2;
    const bool valid0 = (q_start + row0 < S);
    const bool valid1 = (q_start + row1 < S);

    for (int kv_start = 0; kv_start < S; kv_start += BK) {
        const int kv_len = min(kv_start + BK, S) - kv_start;

        // ---- Load K and V via cp.async ----
        {
            const uint4* K_g = reinterpret_cast<const uint4*>(K_bh + static_cast<size_t>(kv_start) * D_HEAD);
            const uint4* V_g = reinterpret_cast<const uint4*>(V_bh + static_cast<size_t>(kv_start) * D_HEAD);
            uint4* K_s = reinterpret_cast<uint4*>(K_smem);
            uint4* V_s = reinterpret_cast<uint4*>(V_smem);
            const int total_vecs = BK * D_HEAD / 8;

            for (int i = tid; i < total_vecs; i += blockDim.x) {
                const int row = i / (D_HEAD / 8);
                if (kv_start + row < S) {
                    cp_async_16(&K_s[i], &K_g[i]);
                    cp_async_16(&V_s[i], &V_g[i]);
                } else {
                    K_s[i] = make_uint4(0, 0, 0, 0);
                    V_s[i] = make_uint4(0, 0, 0, 0);
                }
            }
            cp_async_commit();  // Group 0: K+V
        }

        cp_async_wait_n<0>();
        __syncthreads();

        // ---- Compute S = Q @ K^T ----
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_frag[BK / WMMA_N];
        #pragma unroll
        for (int n_tile = 0; n_tile < BK / WMMA_N; n_tile++) {
            wmma::fill_fragment(s_frag[n_tile], 0.0f);
            #pragma unroll
            for (int k_tile = 0; k_tile < D_HEAD / WMMA_K; k_tile++) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;
                wmma::load_matrix_sync(a_frag, Q_smem + m_row_start * D_HEAD + k_tile * WMMA_K, D_HEAD);
                wmma::load_matrix_sync(b_frag, K_smem + n_tile * WMMA_N * D_HEAD + k_tile * WMMA_K, D_HEAD);
                wmma::mma_sync(s_frag[n_tile], a_frag, b_frag, s_frag[n_tile]);
            }
        }

        // ---- Online softmax ----
        float m_old0 = valid0 ? m_smem[row0] : -INFINITY;
        float m_old1 = valid1 ? m_smem[row1] : -INFINITY;
        float l_old0 = valid0 ? l_smem[row0] : 0.0f;
        float l_old1 = valid1 ? l_smem[row1] : 0.0f;

        float row_max0 = -INFINITY, row_max1 = -INFINITY;
        #pragma unroll
        for (int n = 0; n < BK / WMMA_N; n++) {
            const int n_col_base = n * WMMA_N + sub_col;
            float vals[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) vals[i] = s_frag[n].x[i] * SCALE;

            if (n_col_base     >= kv_len) { vals[0] = -INFINITY; vals[2] = -INFINITY; }
            if (n_col_base + 1 >= kv_len) { vals[1] = -INFINITY; vals[3] = -INFINITY; }
            if (n_col_base + 8 >= kv_len) { vals[4] = -INFINITY; vals[6] = -INFINITY; }
            if (n_col_base + 9 >= kv_len) { vals[5] = -INFINITY; vals[7] = -INFINITY; }

            row_max0 = fmaxf(row_max0, fmaxf(fmaxf(vals[0], vals[1]), fmaxf(vals[4], vals[5])));
            row_max1 = fmaxf(row_max1, fmaxf(fmaxf(vals[2], vals[3]), fmaxf(vals[6], vals[7])));
        }

        row_max0 = fmaxf(row_max0, __shfl_xor_sync(0xFFFFFFFF, row_max0, 1));
        row_max0 = fmaxf(row_max0, __shfl_xor_sync(0xFFFFFFFF, row_max0, 2));
        row_max1 = fmaxf(row_max1, __shfl_xor_sync(0xFFFFFFFF, row_max1, 1));
        row_max1 = fmaxf(row_max1, __shfl_xor_sync(0xFFFFFFFF, row_max1, 2));

        float m_new0 = valid0 ? fmaxf(m_old0, row_max0) : -INFINITY;
        float m_new1 = valid1 ? fmaxf(m_old1, row_max1) : -INFINITY;

        float row_sum0 = 0.0f, row_sum1 = 0.0f;
        #pragma unroll
        for (int n = 0; n < BK / WMMA_N; n++) {
            const int n_col_base = n * WMMA_N + sub_col;
            float p[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) p[i] = 0.0f;

            if (valid0) {
                p[0] = (n_col_base     < kv_len) ? __expf(s_frag[n].x[0] * SCALE - m_new0) : 0.0f;
                p[1] = (n_col_base + 1 < kv_len) ? __expf(s_frag[n].x[1] * SCALE - m_new0) : 0.0f;
                p[4] = (n_col_base + 8 < kv_len) ? __expf(s_frag[n].x[4] * SCALE - m_new0) : 0.0f;
                p[5] = (n_col_base + 9 < kv_len) ? __expf(s_frag[n].x[5] * SCALE - m_new0) : 0.0f;
                row_sum0 += p[0] + p[1] + p[4] + p[5];
            }
            if (valid1) {
                p[2] = (n_col_base     < kv_len) ? __expf(s_frag[n].x[2] * SCALE - m_new1) : 0.0f;
                p[3] = (n_col_base + 1 < kv_len) ? __expf(s_frag[n].x[3] * SCALE - m_new1) : 0.0f;
                p[6] = (n_col_base + 8 < kv_len) ? __expf(s_frag[n].x[6] * SCALE - m_new1) : 0.0f;
                p[7] = (n_col_base + 9 < kv_len) ? __expf(s_frag[n].x[7] * SCALE - m_new1) : 0.0f;
                row_sum1 += p[2] + p[3] + p[6] + p[7];
            }

            const int c0 = n * WMMA_N + sub_col;
            P_smem[row0 * P_STRIDE + c0]     = __float2bfloat16(p[0]);
            P_smem[row0 * P_STRIDE + c0 + 1] = __float2bfloat16(p[1]);
            P_smem[row1 * P_STRIDE + c0]     = __float2bfloat16(p[2]);
            P_smem[row1 * P_STRIDE + c0 + 1] = __float2bfloat16(p[3]);
            P_smem[row0 * P_STRIDE + c0 + 8] = __float2bfloat16(p[4]);
            P_smem[row0 * P_STRIDE + c0 + 9] = __float2bfloat16(p[5]);
            P_smem[row1 * P_STRIDE + c0 + 8] = __float2bfloat16(p[6]);
            P_smem[row1 * P_STRIDE + c0 + 9] = __float2bfloat16(p[7]);
        }

        row_sum0 += __shfl_xor_sync(0xFFFFFFFF, row_sum0, 1);
        row_sum0 += __shfl_xor_sync(0xFFFFFFFF, row_sum0, 2);
        row_sum1 += __shfl_xor_sync(0xFFFFFFFF, row_sum1, 1);
        row_sum1 += __shfl_xor_sync(0xFFFFFFFF, row_sum1, 2);

        const float rescale0 = (m_old0 != -INFINITY) ? __expf(m_old0 - m_new0) : 1.0f;
        const float rescale1 = (m_old1 != -INFINITY) ? __expf(m_old1 - m_new1) : 1.0f;
        #pragma unroll
        for (int n = 0; n < D_HEAD / WMMA_N; n++) {
            o_frag[n].x[0] *= rescale0; o_frag[n].x[1] *= rescale0;
            o_frag[n].x[4] *= rescale0; o_frag[n].x[5] *= rescale0;
            o_frag[n].x[2] *= rescale1; o_frag[n].x[3] *= rescale1;
            o_frag[n].x[6] *= rescale1; o_frag[n].x[7] *= rescale1;
        }

        if ((lane_id & 3) == 0) {
            if (valid0) {
                m_smem[row0] = m_new0;
                l_smem[row0] = l_old0 * rescale0 + row_sum0;
            }
            if (valid1) {
                m_smem[row1] = m_new1;
                l_smem[row1] = l_old1 * rescale1 + row_sum1;
            }
        }

        // ---- Compute O += P @ V ----
        #pragma unroll
        for (int n_tile = 0; n_tile < D_HEAD / WMMA_N; n_tile++) {
            #pragma unroll
            for (int k_tile = 0; k_tile < BK / WMMA_K; k_tile++) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
                wmma::load_matrix_sync(a_frag, P_smem + m_row_start * P_STRIDE + k_tile * WMMA_K, P_STRIDE);
                wmma::load_matrix_sync(b_frag, V_smem + k_tile * WMMA_K * D_HEAD + n_tile * WMMA_N, D_HEAD);
                wmma::mma_sync(o_frag[n_tile], a_frag, b_frag, o_frag[n_tile]);
            }
        }
        __syncthreads();
    }

    // ---- Final normalization ----
    const float final_l0 = valid0 ? l_smem[row0] : 0.0f;
    const float final_l1 = valid1 ? l_smem[row1] : 0.0f;
    const float final_m0 = valid0 ? m_smem[row0] : -INFINITY;
    const float final_m1 = valid1 ? m_smem[row1] : -INFINITY;
    const float inv_l0 = (final_l0 > 0.0f) ? (1.0f / final_l0) : 0.0f;
    const float inv_l1 = (final_l1 > 0.0f) ? (1.0f / final_l1) : 0.0f;

    if (valid0 && final_l0 > 0.0f) {
        LSE_bh[q_start + row0] = final_m0 + logf(final_l0);
    }
    if (valid1 && final_l1 > 0.0f) {
        LSE_bh[q_start + row1] = final_m1 + logf(final_l1);
    }

    #pragma unroll
    for (int n = 0; n < D_HEAD / WMMA_N; n++) {
        o_frag[n].x[0] *= inv_l0; o_frag[n].x[1] *= inv_l0;
        o_frag[n].x[4] *= inv_l0; o_frag[n].x[5] *= inv_l0;
        o_frag[n].x[2] *= inv_l1; o_frag[n].x[3] *= inv_l1;
        o_frag[n].x[6] *= inv_l1; o_frag[n].x[7] *= inv_l1;
    }

    // ---- Store O ----
    __nv_bfloat16* O_smem = Q_smem;
    const int col_base = (lane_id % 4) * 2;
    #pragma unroll
    for (int n = 0; n < D_HEAD / WMMA_N; n++) {
        const int n_col = n * WMMA_N;
        if (valid0) {
            O_smem[row0 * D_HEAD + n_col + col_base]     = __float2bfloat16(o_frag[n].x[0]);
            O_smem[row0 * D_HEAD + n_col + col_base + 1] = __float2bfloat16(o_frag[n].x[1]);
            O_smem[row0 * D_HEAD + n_col + col_base + 8] = __float2bfloat16(o_frag[n].x[4]);
            O_smem[row0 * D_HEAD + n_col + col_base + 9] = __float2bfloat16(o_frag[n].x[5]);
        }
        if (valid1) {
            O_smem[row1 * D_HEAD + n_col + col_base]     = __float2bfloat16(o_frag[n].x[2]);
            O_smem[row1 * D_HEAD + n_col + col_base + 1] = __float2bfloat16(o_frag[n].x[3]);
            O_smem[row1 * D_HEAD + n_col + col_base + 8] = __float2bfloat16(o_frag[n].x[6]);
            O_smem[row1 * D_HEAD + n_col + col_base + 9] = __float2bfloat16(o_frag[n].x[7]);
        }
    }
    __syncthreads();

    {
        const uint4* O_s = reinterpret_cast<const uint4*>(O_smem);
        uint4* O_g = reinterpret_cast<uint4*>(O_bh + static_cast<size_t>(q_start) * D_HEAD);
        const int total_vecs = BM * D_HEAD / 8;
        for (int i = tid; i < total_vecs; i += blockDim.x) {
            const int row = i / (D_HEAD / 8);
            if (q_start + row < S) {
                O_g[i] = O_s[i];
            }
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    const int B = static_cast<int>(Q.size(0));
    const int H = static_cast<int>(Q.size(1));
    const int S = static_cast<int>(Q.size(2));
    const int D = static_cast<int>(Q.size(3));
    (void)D;

    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    const int num_q_blocks = (S + BM - 1) / BM;
    const int total_blocks = num_q_blocks * B * H;
    dim3 grid(total_blocks);
    dim3 block(NUM_THREADS);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(attention_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));

    attention_kernel<<<grid, block, SMEM_SIZE, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data, S, B, H, num_q_blocks);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_cuda::run);

}  // namespace tvm_ffi_mha_cuda