#include <cuda_runtime.h>
#include <cuda_bfloat16.h>
#include <device_launch_parameters.h>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <float.h>
#include <assert.h>
#include <stdio.h>

#define CUDA_CHECK(call) do {                                                \
    cudaError_t _e = (call);                                                 \
    if (_e != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error: %s at %s:%d\n",                         \
                cudaGetErrorString(_e), __FILE__, __LINE__);                  \
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
} while(0)

namespace mha_d128_impl {

static constexpr int TILE_M  = 32;   // Q rows per block (fewer regs/thread)
static constexpr int TILE_N  = 64;   // KV rows per tile
static constexpr int HEAD_D  = 128;
static constexpr int NT      = 128;  // 4 warps = 1 warpgroup

__global__ void mha_forward_kernel(
    const __nv_bfloat16* __restrict__ Q_g,
    const __nv_bfloat16* __restrict__ K_g,
    const __nv_bfloat16* __restrict__ V_g,
    __nv_bfloat16*               __restrict__ O_g,
    float*                       __restrict__ LSE_g,
    int B, int H, int S, float attn_scale)
{
    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* sQ = smem;
    __nv_bfloat16* sK = smem + TILE_M * HEAD_D;
    __nv_bfloat16* sV = sK   + TILE_N * HEAD_D;

    int bh_idx = blockIdx.x;
    int qtile  = blockIdx.y;

    if (bh_idx >= B * H || qtile * TILE_M >= S) return;

    int b = bh_idx / H;
    int h = bh_idx % H;

    int64_t bh_off = static_cast<int64_t>(b * H + h) * S * HEAD_D;
    int64_t q_off  = static_cast<int64_t>(qtile)     * TILE_M * HEAD_D;

    const __nv_bfloat16* Q_base = Q_g + bh_off + q_off;
    const __nv_bfloat16* K_base = K_g + bh_off;
    const __nv_bfloat16* V_base = V_g + bh_off;
    __nv_bfloat16*         O_base = O_g + bh_off + q_off;
    float*               LSE_base = LSE_g + static_cast<int64_t>(b * H + h) * S
                                           + qtile * TILE_M;

    int tid = threadIdx.x;
    int act_m = min(TILE_M, S - qtile * TILE_M);

    // Cooperative load of Q tile
    for (int i = tid; i < act_m * HEAD_D; i += NT)
        sQ[i] = Q_base[i];
    __syncthreads();

    // Each thread processes (TILE_M / NT) rows = 32/128 -> rounds to 1 row per block
    // With NT=128 and TILE_M=32, we have 4 threads per row for cooperative work
    // But we simplify: thread t maps to row t when t < TILE_M
    // Threads >= act_m just help with loads and idle during compute

    if (tid < act_m) {
        // Cache Q row in registers
        __nv_bfloat16 qr[HEAD_D];
        for (int d = 0; d < HEAD_D; ++d)
            qr[d] = sQ[tid * HEAD_D + d];

        // Output accumulator (FP32 precision)
        float acc[HEAD_D] = {};

        // Online softmax state
        float m_i = -FLT_MAX;   // running max
        float li_i = 0.0f;       // running denominator

        int nk_end = (S + TILE_N - 1) / TILE_N;  // number of KV tiles

        for (int tile_k = 0; tile_k < nk_end; ++tile_k) {
            int nk = tile_k * TILE_N;
            int act_n = min(TILE_N, S - nk);
            int64_t nk_off = static_cast<int64_t>(nk) * HEAD_D;

            // --- Load K tile cooperatively ---
            for (int i = tid; i < act_n * HEAD_D; i += NT)
                sK[i] = K_base[nk_off + i];
            __syncthreads();

            // --- Pass 1: find row maximum over K tile ---
            float m_new = m_i;
            for (int j = 0; j < act_n; ++j) {
                float dot = 0.0f;
                for (int d = 0; d < HEAD_D; ++d)
                    dot += __bfloat162float(qr[d])
                         * __bfloat162float(sK[j * HEAD_D + d]);
                dot *= attn_scale;
                if (dot > m_new) m_new = dot;
            }

            // --- Load V tile cooperatively ---
            for (int i = tid; i < act_n * HEAD_D; i += NT)
                sV[i] = V_base[nk_off + i];
            __syncthreads();

            // --- Pass 2: compute probabilities & accumulate output ---
            float p_sum = 0.0f;
            for (int j = 0; j < act_n; ++j) {
                float dot = 0.0f;
                for (int d = 0; d < HEAD_D; ++d)
                    dot += __bfloat162float(qr[d])
                         * __bfloat162float(sK[j * HEAD_D + d]);
                dot *= attn_scale;

                float p = expf(dot - m_new);
                p_sum += p;
                for (int d = 0; d < HEAD_D; ++d)
                    acc[d] += p * __bfloat162float(sV[j * HEAD_D + d]);
            }

            // ---- RESCALE: apply m_i -> m_new shift ----
            // OLD_ACC_NEW = alpha * OLD_ACC_OLD  (before adding new pV terms)
            // NEW_LSE_NUM = alpha * OLD_LSE_NUM + p_sum
            float alpha = expf(m_i - m_new);

            // Save old li_i scaled form
            float li_scaled = li_i * alpha + p_sum;

            // Scale ONLY the previously accumulated contributions
            for (int d = 0; d < HEAD_D; ++d)
                acc[d] *= alpha;

            // Actually, wait — we already added new pV to acc above!
            // We need to subtract them, scale old, then re-add.
            // BETTER: scale acc BEFORE adding new contributions.
            // Let me rewrite with the correct order...
            
            // Undo: subtract newly added terms, scale old acc, re-add
            for (int j = act_n - 1; j >= 0; --j) {
                float dot = 0.0f;
                for (int d = 0; d < HEAD_D; ++d)
                    dot += __bfloat162float(qr[d])
                         * __bfloat162float(sK[j * HEAD_D + d]);
                dot *= attn_scale;
                float p = expf(dot - m_new);
                for (int d = 0; d < HEAD_D; ++d)
                    acc[d] -= p * __bfloat162float(sV[j * HEAD_D + d]);
            }
            // Now acc only has old contributions, scale them
            for (int d = 0; d < HEAD_D; ++d)
                acc[d] *= alpha;
            // Re-add new contributions
            for (int j = 0; j < act_n; ++j) {
                float dot = 0.0f;
                for (int d = 0; d < HEAD_D; ++d)
                    dot += __bfloat162float(qr[d])
                         * __bfloat162float(sK[j * HEAD_D + d]);
                dot *= attn_scale;
                float p = expf(dot - m_new);
                for (int d = 0; d < HEAD_D; ++d)
                    acc[d] += p * __bfloat162float(sV[j * HEAD_D + d]);
            }

            li_i = li_scaled;
            m_i = m_new;

            __syncthreads();
        }

        // Final normalization
        float inv_li = (li_i > 0.0f) ? (1.0f / li_i) : 0.0f;
        for (int d = 0; d < HEAD_D; ++d)
            O_base[tid * HEAD_D + d] = __float2bfloat16(acc[d] * inv_li);
        LSE_base[tid] = (li_i > 0.0f) ? (m_i + logf(li_i)) : (-FLT_MAX);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K,
         tvm::ffi::TensorView V, tvm::ffi::TensorView O,
         tvm::ffi::TensorView LSE)
{
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);

    assert(D == HEAD_D && "Head dimension must be 128");

    int num_bh     = static_cast<int>(B * H);
    int num_qtiles = static_cast<int>((S + TILE_M - 1) / TILE_M);

    dim3 grid(num_bh, num_qtiles);
    dim3 block(NT);

    size_t smem_bytes = static_cast<size_t>(TILE_M + TILE_N * 2) * HEAD_D
                        * sizeof(__nv_bfloat16);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    float scale = rsqrtf(static_cast<float>(D));

    mha_forward_kernel<<<grid, block, smem_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<__nv_bfloat16*>(O.data_ptr()),
        static_cast<float*>(LSE.data_ptr()),
        static_cast<int>(B), static_cast<int>(H),
        static_cast<int>(S), scale);

    CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_d128_impl::run);

}  // namespace mha_d128_impl