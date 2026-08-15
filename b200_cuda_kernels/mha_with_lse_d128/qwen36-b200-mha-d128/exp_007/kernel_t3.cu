#include <cuda_runtime.h>
#include <cuda_bf16.h>
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

static constexpr int TILE_M  = 16;   // Q rows per block (lower reg pressure)
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

    // Cooperative load of Q tile into shared memory
    for (int i = tid; i < act_m * HEAD_D; i += NT)
        sQ[i] = Q_base[i];
    __syncthreads();

    // Each active thread owns one Q row and iterates over KV tiles
    if (tid < act_m) {
        // Read Q row directly from shared memory each time (avoids register spilling)
        // Base pointer to this thread's Q row in smem
        __nv_bfloat16* qrow = sQ + tid * HEAD_D;

        // FP32 output accumulator - reduced to 64 elements (still fits comfortably)
        // We'll split accumulation into two halves to stay safe
        float acc_lo[64] = {};  // dimensions 0..63
        float acc_hi[64] = {};  // dimensions 64..127

        // Online softmax state
        float m_i  = -FLT_MAX;  // running max
        float li_i = 0.0f;       // running lse numerator

        int nk_end = (S + TILE_N - 1) / TILE_N;

        for (int tile_k = 0; tile_k < nk_end; ++tile_k) {
            int nk = tile_k * TILE_N;
            int act_n = min(TILE_N, S - nk);
            int64_t nk_off = static_cast<int64_t>(nk) * HEAD_D;

            // Load K tile cooperatively
            for (int i = tid; i < act_n * HEAD_D; i += NT)
                sK[i] = K_base[nk_off + i];
            __syncthreads();

            // --- Step 1: scan K tile to find row maximum ---
            float m_new = m_i;
            for (int j = 0; j < act_n; ++j) {
                float dot = 0.0f;
                for (int d = 0; d < HEAD_D; ++d)
                    dot += __bfloat162float(qrow[d])
                         * __bfloat162float(sK[j * HEAD_D + d]);
                dot *= attn_scale;
                if (dot > m_new) m_new = dot;
            }

            // Guard against degenerate case
            if (m_new < -1e10f) continue;

            // --- Step 2: load V tile cooperatively ---
            for (int i = tid; i < act_n * HEAD_D; i += NT)
                sV[i] = V_base[nk_off + i];
            __syncthreads();

            // --- Step 3: rescale previous accumulator to new basis ---
            float alpha = expf(m_i - m_new);

            // Handle first iteration or extreme rescaling
            bool prev_valid = (m_i > -FLT_MAX / 2.0f);

            if (prev_valid && alpha > 0.0f) {
                for (int d = 0; d < 64; ++d) {
                    acc_lo[d] *= alpha;
                    acc_hi[d] *= alpha;
                }
                li_i *= alpha;
            } else {
                // Previous contributions are negligible
                for (int d = 0; d < 64; ++d) {
                    acc_lo[d] = 0.0f;
                    acc_hi[d] = 0.0f;
                }
                li_i = 0.0f;
            }

            // --- Step 4: compute probabilities & accumulate ---
            float p_sum = 0.0f;
            for (int j = 0; j < act_n; ++j) {
                float dot = 0.0f;
                for (int d = 0; d < HEAD_D; ++d)
                    dot += __bfloat162float(qrow[d])
                         * __bfloat162float(sK[j * HEAD_D + d]);
                dot *= attn_scale;

                float diff = dot - m_new;
                if (diff > -100.0f) {
                    float p = expf(diff);
                    p_sum += p;

                    // Accumulate lo
                    for (int d = 0; d < 64; ++d)
                        acc_lo[d] += p * __bfloat162float(sV[j * HEAD_D + d]);
                    // Accumulate hi
                    for (int d = 0; d < 64; ++d)
                        acc_hi[d] += p * __bfloat162float(sV[j * HEAD_D + 64 + d]);
                }
            }

            li_i += p_sum;
            m_i = m_new;

            __syncthreads();
        }

        // --- Write back: normalize and convert to BF16 ---
        float inv_li = (li_i > 1e-30f) ? (1.0f / li_i) : 0.0f;
        for (int d = 0; d < 64; ++d)
            O_base[tid * HEAD_D + d] = __float2bfloat16(acc_lo[d] * inv_li);
        for (int d = 0; d < 64; ++d)
            O_base[tid * HEAD_D + 64 + d] = __float2bfloat16(acc_hi[d] * inv_li);

        LSE_base[tid] = (li_i > 1e-30f) ? (m_i + logf(li_i)) : (-FLT_MAX);
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

    // Shared memory: sQ(TILE_M*D) + sK(TILE_N*D) + sV(TILE_N*D) bytes of bf16
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