#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
#include <cfloat>
#include <cmath>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                         \
    cudaError_t _e = (call);                                          \
    if (_e != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                  \
                cudaGetErrorString(_e), __FILE__, __LINE__);          \
        exit(1);                                                      \
    }                                                                 \
} while(0)

namespace mha_d128 {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int NT = 256;      // threads per block
constexpr int TPR = NT / BM; // threads per row of output tile = 4
constexpr int EPT = BN / TPR;// elements per thread = 16

__global__ void attention_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S, int D)
{
    extern __shared__ __nv_bfloat16 smem[];

    // Shared memory layout:
    //   Q_smem: [BM][D]     = 64*128 bf16  = 16 KB
    //   K_smem: [BK][D]     = 16*128 bf16  =  4 KB
    //   V_smem: [BK][BN]    = 16*  64 bf16 =  2 KB
    __nv_bfloat16* Q_smem = smem;                      // offset 0
    __nv_bfloat16* K_smem = smem + BM * D;             // offset 8192
    __nv_bfloat16* V_smem = smem + BM * D + BK * D;    // offset 10240

    int bh      = blockIdx.x;
    int q_blk   = blockIdx.y;
    int o_blk   = blockIdx.z;

    int q_row_start = q_blk * BM;
    int o_col_start = o_blk * BN;

    int tid       = threadIdx.x;
    int row_id    = tid / TPR;        // which output row this thread owns (0..BM-1)
    int elem_base = (tid % TPR) * EPT;// which output columns this thread owns (stride EPT)

    // Base pointers for this (batch, head) combination
    const __nv_bfloat16* Q_bh  = Q + (uint64_t)bh * S * D;
    const __nv_bfloat16* K_bh  = K + (uint64_t)bh * S * D;
    const __nv_bfloat16* V_bh  = V + (uint64_t)bh * S * D;
    __nv_bfloat16*         O_bh = O + (uint64_t)bh * S * D;
    float*                 LSE_bh = LSE + bh * S;

    float inv_sd = rsqrtf((float)D);

    // ==================================================================
    // Phase 1: Cooperative load of Q tile into shared memory
    // ==================================================================
    for (int i = tid; i < BM * D; i += NT) {
        int r = i / D;
        int c = i % D;
        int gr = q_row_start + r;
        Q_smem[r * D + c] = (gr < S)
            ? Q_bh[(uint64_t)gr * D + c]
            : __float2bfloat16(0.0f);
    }
    __syncthreads();

    // Per-thread accumulators (FP32 for numerical stability)
    float  acc[EPT] = {};        // output accumulator for owned columns
    float  m_val = -FLT_MAX;     // running softmax max for this row
    float  l_val = 0.0f;         // running softmax sum-exp for this row

    int nk = (S + BK - 1) / BK;  // number of key chunks to iterate

    // ==========================================================================
    // Phase 2: Streamed iteration over K / V chunks — online softmax
    // ==========================================================================
    for (int ki = 0; ki < nk; ki++) {
        int k_start = ki * BK;
        int kv = (BK < S - k_start) ? BK : (S - k_start);

        // Cooperative load of K tile  [BK][D]
        for (int i = tid; i < BK * D; i += NT) {
            int r = i / D;
            int c = i % D;
            int gr = k_start + r;
            K_smem[r * D + c] = (gr < S)
                ? K_bh[(uint64_t)gr * D + c]
                : __float2bfloat16(0.0f);
        }

        // Cooperative load of V tile  [BK][BN]
        for (int i = tid; i < BK * BN; i += NT) {
            int r = i / BN;
            int c = i % BN;
            int gr = k_start + r;
            int gc = o_col_start + c;
            V_smem[r * BN + c] = (gr < S && gc < D)
                ? V_bh[(uint64_t)gr * D + gc]
                : __float2bfloat16(0.0f);
        }
        __syncthreads();

        // ------ Attention scores: S[row][k] = Q[row]·K[k] / sqrt(D) ------
        float scores[BK];
        bool row_ok = (q_row_start + row_id < S);

        for (int k = 0; k < BK; k++) {
            if (!row_ok || (k_start + k >= S)) {
                scores[k] = -FLT_MAX;
            } else {
                float dot = 0.0f;
                // Vectorised bf16->fp32 unpack + FMA: 8 bf16 per inner-step
                // D=128 → 16 groups of 8 bf16 (each group = 4 × __nv_bfloat162)
                const __nv_bfloat162* qp =
                    reinterpret_cast<const __nv_bfloat162*>(&Q_smem[row_id * D]);
                const __nv_bfloat162* kp =
                    reinterpret_cast<const __nv_bfloat162*>(&K_smem[k * D]);

                for (int g = 0; g < D / 8; g++) {
                    int idx = g * 4;
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        dot += __low2float(qp[idx + j]) * __low2float(kp[idx + j]);
                        dot += __high2float(qp[idx + j]) * __high2float(kp[idx + j]);
                    }
                }
                // tail (D % 8 elements; dead code for D=128, kept for generality)
                for (int d = (D / 8) * 8; d < D; d++) {
                    dot += __bfloat162float(Q_smem[row_id * D + d])
                         * __bfloat162float(K_smem[k * D + d]);
                }
                scores[k] = dot * inv_sd;
            }
        }

        // ------ Online softmax state update ------
        float m_new = -FLT_MAX;
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            m_new = (m_new > scores[k]) ? m_new : scores[k];
        }

        float m_old = m_val;
        m_val = (m_old > m_new) ? m_old : m_new;
        float scale = expf(m_old - m_val);

        // Rescale previously accumulated output
        #pragma unroll
        for (int i = 0; i < EPT; i++) {
            acc[i] *= scale;
        }

        // Accumulate new attention-weighted V
        float l_inc = 0.0f;
        for (int k = 0; k < kv; k++) {
            float pk = expf(scores[k] - m_val);
            l_inc += pk;
            #pragma unroll
            for (int i = 0; i < EPT; i++) {
                acc[i] += pk * __bfloat162float(V_smem[k * BN + elem_base + i]);
            }
        }
        l_val = l_val * scale + l_inc;
    }

    // ==========================================================================
    // Phase 3: Normalise, store O (bf16) and LSE (fp32)
    // ==========================================================================
    float norm = (l_val > 0) ? (1.0f / l_val) : 0.0f;
    int out_r = q_row_start + row_id;

    #pragma unroll
    for (int i = 0; i < EPT; i++) {
        int out_c = o_col_start + elem_base + i;
        if (out_r < S && out_c < D) {
            O_bh[(uint64_t)out_r * D + out_c] =
                __float2bfloat16(acc[i] * norm);
        }
    }

    // LSE = m_val + log(l_val); only first column-block writes to avoid duplicates
    if (o_blk == 0 && (tid % TPR == 0)) {
        if (out_r < S) {
            LSE_bh[out_r] = (l_val > 0) ? (m_val + logf(l_val)) : (-FLT_MAX);
        }
    }
}

void run(tvm::ffi::TensorView Q_tv, tvm::ffi::TensorView K_tv,
         tvm::ffi::TensorView V_tv, tvm::ffi::TensorView O_tv,
         tvm::ffi::TensorView LSE_tv) {
    CUDA_CHECK(cudaSetDevice(Q_tv.device().device_id));

    constexpr int64_t B = 4, H = 48, D = 128;
    int64_t S = Q_tv.size(2);

    const __nv_bfloat16* Q   = static_cast<const __nv_bfloat16*>(Q_tv.data_ptr());
    const __nv_bfloat16* K   = static_cast<const __nv_bfloat16*>(K_tv.data_ptr());
    const __nv_bfloat16* V   = static_cast<const __nv_bfloat16*>(V_tv.data_ptr());
    __nv_bfloat16*         O = static_cast<__nv_bfloat16*>(O_tv.data_ptr());
    float*                 LSE = static_cast<float*>(LSE_tv.data_ptr());

    dim3 grid(B * H, (S + BM - 1) / BM, (D + BN - 1) / BN);
    dim3 block(NT);
    size_t smem_bytes = (BM * D + BK * D + BK * BN) * sizeof(__nv_bfloat16);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q_tv.device().device_type, Q_tv.device().device_id));

    attention_kernel<<<grid, block, smem_bytes, stream>>>(Q, K, V, O, LSE, B, H, S, D);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace mha_d128

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_d128::run);