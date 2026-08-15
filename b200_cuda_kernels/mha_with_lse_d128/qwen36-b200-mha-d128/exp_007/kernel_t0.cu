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

// ---------------------------------------------------------------------------
// Tile constants (optimized for D=128, Blackwell SM100)
// ---------------------------------------------------------------------------
// 128 threads = 4 warps per block (warpgroup granularity for future tcgen05)
// TILE_M = 64 Q rows per block -> 2 threads per row (moderate register pressure)
// TILE_N = 64 KV rows per tile  -> balanced tiling for S > 64
// HEAD_D = 128                   -> fixed by benchmark definition
// ---------------------------------------------------------------------------
static constexpr int TILE_M  = 64;
static constexpr int TILE_N  = 64;
static constexpr int HEAD_D  = 128;
static constexpr int NT      = 128;

// ---------------------------------------------------------------------------
// Forward kernel: flash-attention style online softmax
// ---------------------------------------------------------------------------
__global__ void mha_forward_kernel(
    const __nv_bfloat16* __restrict__ Q_g,
    const __nv_bfloat16* __restrict__ K_g,
    const __nv_bfloat16* __restrict__ V_g,
    __nv_bfloat16*               __restrict__ O_g,
    float*                       __restrict__ LSE_g,
    int B, int H, int S, float attn_scale)
{
    // ================================================================
    // Dynamic shared memory: three staging buffers, row-major layout
    //   sQ[TILE_M][HEAD_D]  ->  64*128*2 = 16 KiB
    //   sK[TILE_N][HEAD_D]  ->  64*128*2 = 16 KiB
    //   sV[TILE_N][HEAD_D]  ->  64*128*2 = 16 KiB
    //   Total: 48 KiB per block (well within SM100's 228 KiB/SM)
    // ================================================================
    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* sQ = smem;
    __nv_bfloat16* sK = smem + TILE_M * HEAD_D;
    __nv_bfloat16* sV = sK   + TILE_N * HEAD_D;

    int bh_idx = blockIdx.x;       // packed (batch, head) identifier
    int qtile  = blockIdx.y;       // tile index along Q-sequence axis

    if (bh_idx >= B * H || qtile * TILE_M >= S) return;

    int b = bh_idx / H;
    int h = bh_idx % H;

    // Base offsets for this (batch, head) slice
    int64_t bh_off = static_cast<int64_t>(b * H + h) * S * HEAD_D;
    int64_t q_off  = static_cast<int64_t>(qtile)     * TILE_M * HEAD_D;

    const __nv_bfloat16* Q_base = Q_g + bh_off + q_off;
    const __nv_bfloat16* K_base = K_g + bh_off;
    const __nv_bfloat16* V_base = V_g + bh_off;
    __nv_bfloat16*         O_base = O_g + bh_off + q_off;
    float*               LSE_base = LSE_g + static_cast<int64_t>(b * H + h) * S
                                           + qtile * TILE_M;

    int tid = threadIdx.x;
    int act_m = min(TILE_M, S - qtile * TILE_M);   // actual Q rows in this tile

    // ---------------------------------------------------------------
    // PHASE 1: Cooperative load of Q tile into shared memory
    // Each thread strides over the TILE_M * HEAD_D element space
    // ---------------------------------------------------------------
    for (int i = tid; i < act_m * HEAD_D; i += NT) {
        sQ[i] = Q_base[i];
    }
    __syncthreads();

    // ---------------------------------------------------------------
    // PHASE 2: Per-row online-softmax attention
    // Each active thread (tid < act_m) owns one Q row.
    // It iterates over all KV tiles, maintaining:
    //   m_i  : running maximum of scaled scores for this row
    //   l_i  : running denominator sum of exp(score - m_i)
    //   acc[]: FP32 accumulator for the output row
    //
    // Register budget per active thread:
    //   q_buf[128] bf16  = ~64  regs  (cached Q row from smem)
    //   acc[128]   fp32  = ~128 regs  (output accumulator)
    //   scalars              ~4 regs
    //   Total                      ~196 regs  < 255 limit
    // ---------------------------------------------------------------
    if (tid < act_m) {
        // Cache this thread's Q row in registers (BF16 to save register space)
        __nv_bfloat16 q_buf[HEAD_D];
        // FP32 accumulator for the output row (precision-critical)
        float acc[HEAD_D] = {};

        for (int d = 0; d < HEAD_D; ++d)
            q_buf[d] = sQ[tid * HEAD_D + d];

        float m_i = -FLT_MAX;   // row-wise running maximum
        float l_i = 0.0f;       // row-wise running denominator

        // Iterate over KV sequence in tiles
        for (int nk = 0; nk < S; nk += TILE_N) {
            int act_n = min(TILE_N, S - nk);
            int64_t nk_off = static_cast<int64_t>(nk) * HEAD_D;

            // --- Load K tile cooperatively ---
            for (int i = tid; i < act_n * HEAD_D; i += NT)
                sK[i] = K_base[nk_off + i];
            __syncthreads();

            // --- Pass 1: scan K tile to find new row maximum ---
            // (avoids allocating a score[64] array in registers)
            float m_new = m_i;
            for (int j = 0; j < act_n; ++j) {
                float dot = 0.0f;
                for (int d = 0; d < HEAD_D; ++d)
                    dot += __bfloat162float(q_buf[d])
                         * __bfloat162float(sK[j * HEAD_D + d]);
                dot *= attn_scale;
                if (dot > m_new) m_new = dot;
            }

            // --- Load V tile cooperatively ---
            for (int i = tid; i < act_n * HEAD_D; i += NT)
                sV[i] = V_base[nk_off + i];
            __syncthreads();

            // --- Pass 2: compute probabilities & accumulate output ---
            float alpha = expf(m_i - m_new);    // rescale factor (<= 1)
            float p_sum = 0.0f;

            for (int j = 0; j < act_n; ++j) {
                // Recompute dot product (trades registers for a few smem reads)
                float dot = 0.0f;
                for (int d = 0; d < HEAD_D; ++d)
                    dot += __bfloat162float(q_buf[d])
                         * __bfloat162float(sK[j * HEAD_D + d]);
                dot *= attn_scale;

                float p = expf(dot - m_new);
                p_sum += p;

                // Accumulate: acc[d] += p * V[j][d]
                for (int d = 0; d < HEAD_D; ++d)
                    acc[d] += p * __bfloat162float(sV[j * HEAD_D + d]);
            }

            // Stabilize: rescale previous accumulator contributions
            for (int d = 0; d < HEAD_D; ++d)
                acc[d] *= alpha;

            // Update running softmax stats
            l_i = l_i * alpha + p_sum;
            m_i = m_new;

            __syncthreads();   // flush before next K-load cycle
        }

        // ---------------------------------------------------------------
        // PHASE 3: Final normalization and write-back
        //   O[row] = acc / l_i          (BF16 output)
        //   LSE[row] = m_i + ln(l_i)    (FP32 natural-log LSE)
        // ---------------------------------------------------------------
        float inv_l = 1.0f / l_i;
        for (int d = 0; d < HEAD_D; ++d)
            O_base[tid * HEAD_D + d] = __float2bfloat16(acc[d] * inv_l);
        LSE_base[tid] = m_i + logf(l_i);
    }
}

// ---------------------------------------------------------------------------
// Host-side launcher with TVM-FFI binding
// ---------------------------------------------------------------------------
void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K,
         tvm::ffi::TensorView V, tvm::ffi::TensorView O,
         tvm::ffi::TensorView LSE)
{
    // Ensure correct GPU context for multi-GPU deployments
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);

    assert(D == HEAD_D && "Head dimension D must equal 128");

    // Grid: one block per (batch, head, Q-tile) triple
    int num_bh     = static_cast<int>(B * H);
    int num_qtiles = static_cast<int>((S + TILE_M - 1) / TILE_M);

    dim3 grid(num_bh, num_qtiles);
    dim3 block(NT);

    // Dynamic shared memory: (TILE_M + 2*TILE_N) * HEAD_D * sizeof(bf16)
    size_t smem_bytes = static_cast<size_t>(TILE_M + TILE_N * 2) * HEAD_D
                        * sizeof(__nv_bfloat16);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    // Attention scaling factor: 1 / sqrt(D)  (applied inside the QK dot product)
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