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

// Tile dimensions
constexpr int BM = 64;       // output rows per CTA
constexpr int BK = 16;       // key tokens per chunk
constexpr int NT = 256;      // threads per block
constexpr int TPR = NT / BM; // threads per output row = 4
constexpr int D = 128;       // head dimension
constexpr int EPT = D / TPR; // elements per thread = 32

__global__ void attention_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S)
{
    extern __shared__ __nv_bfloat16 smem[];

    // Shared memory layout (no overlap):
    //   Q_tile: [BM][D]   = 8192 bf16  at offset 0
    //   K_tile: [BK][D]   = 2048 bf16  at offset 8192
    //   V_tile: [BK][D]   = 2048 bf16  at offset 10240
    // Total: 12288 bf16 = 24576 bytes ~ 24 KB
    __nv_bfloat16* Q_tile = smem;
    __nv_bfloat16* K_tile = smem + BM * D;
    __nv_bfloat16* V_tile = smem + BM * D + BK * D;

    int bh      = blockIdx.x;      // combined batch-head index
    int q_blk   = blockIdx.y;      // query block index
    int row_off = q_blk * BM;      // starting global query row

    int tid     = threadIdx.x;
    int r_idx   = tid / TPR;       // thread's row within tile [0,BM)
    int lane    = tid % TPR;       // thread's lane within row [0,TPR)

    // Base pointers for this (batch, head) pair
    const __nv_bfloat16* Q_bh = Q + (uint64_t)bh * S * D;
    const __nv_bfloat16* K_bh = K + (uint64_t)bh * S * D;
    const __nv_bfloat16* V_bh = V + (uint64_t)bh * S * D;
    __nv_bfloat16*         O_bh = O + (uint64_t)bh * S * D;
    float*                 LSE_bh = LSE + bh * S;

    float inv_sqrt_d = rsqrtf(static_cast<float>(D));
    int global_qr = row_off + r_idx;
    bool q_row_in_bounds = (global_qr < S);

    // ================================================================
    // Phase 1: Cooperative load of Q tile into shared memory
    // ================================================================
    for (int idx = tid; idx < BM * D; idx += NT) {
        int tr = idx / D;
        int tc = idx % D;
        int gqr = row_off + tr;
        Q_tile[tr * D + tc] = (gqr < S)
            ? Q_bh[(uint64_t)gqr * D + tc]
            : __float2bfloat16(0.0f);
    }
    __syncthreads();

    // ================================================================
    // Per-thread state for online softmax (FP32 accumulation)
    // ================================================================
    float acc[EPT];       // output accumulators (one per owned column)
    for (int e = 0; e < EPT; e++) acc[e] = 0.0f;

    float m_prev = -FLT_MAX; // running row max
    float l_prev = 0.0f;     // running sum-exp

    int num_chunks = (S + BK - 1) / BK;

    // ================================================================
    // Phase 2: Streaming iteration over K/V chunks
    // ================================================================
    for (int ki = 0; ki < num_chunks; ki++) {
        int k_start = ki * BK;
        int kv = (k_start + BK <= S) ? BK : (S - k_start);

        // --- Cooperative load of K tile [BK][D] ---
        for (int idx = tid; idx < BK * D; idx += NT) {
            int tr = idx / D;
            int tc = idx % D;
            int gkr = k_start + tr;
            K_tile[tr * D + tc] = (gkr < S)
                ? K_bh[(uint64_t)gkr * D + tc]
                : __float2bfloat16(0.0f);
        }

        // --- Cooperative load of V tile [BK][D] ---
        for (int idx = tid; idx < BK * D; idx += NT) {
            int tr = idx / D;
            int tc = idx % D;
            int gkr = k_start + tr;
            V_tile[tr * D + tc] = (gkr < S)
                ? V_bh[(uint64_t)gkr * D + tc]
                : __float2bfloat16(0.0f);
        }
        __syncthreads();

        // --- Compute Q·K^T scores for this row and this chunk ---
        float cur_max = -FLT_MAX;
        float p[BK];  // attention weights (after subtraction of new row max)

        if (q_row_in_bounds) {
            const __nv_bfloat162* qp =
                reinterpret_cast<const __nv_bfloat162*>(&Q_tile[r_idx * D]);
            for (int k = 0; k < kv; k++) {
                float dot = 0.0f;
                const __nv_bfloat162* kp =
                    reinterpret_cast<const __nv_bfloat162*>(&K_tile[k * D]);
                // D/2 = 64 packed pairs; unroll-friendly
                for (int dp = 0; dp < D / 2; dp++) {
                    float qa_lo = __low2float(qp[dp]);
                    float qa_hi = __high2float(qp[dp]);
                    float ka_lo = __low2float(kp[dp]);
                    float ka_hi = __high2float(kp[dp]);
                    dot += qa_lo * ka_lo + qa_hi * ka_hi;
                }
                float s = dot * inv_sqrt_d;
                p[k] = s;
                if (s > cur_max) cur_max = s;
            }
            // Padded keys: mark as excluded
            for (int k = kv; k < BK; k++) {
                p[k] = -FLT_MAX;
            }
        } else {
            // Query row out of global bounds: all scores -inf
            for (int k = 0; k < BK; k++) {
                p[k] = -FLT_MAX;
            }
        }

        // --- Online softmax state update ---
        float m_cur = max(m_prev, cur_max);
        float alpha = expf(m_prev - m_cur);  // rescale factor

        // Rescale previous accumulator and partial sum
        for (int e = 0; e < EPT; e++) {
            acc[e] *= alpha;
        }
        float l_cur = l_prev * alpha;

        // Accumulate new chunk contribution (only over valid keys)
        for (int k = 0; k < kv; k++) {
            float pw = expf(p[k] - m_cur);
            l_cur += pw;
            for (int e = 0; e < EPT; e++) {
                int col = lane * EPT + e;
                acc[e] += pw * __bfloat162float(V_tile[k * D + col]);
            }
        }

        m_prev = m_cur;
        l_prev = l_cur;
    }

    // ================================================================
    // Phase 3: Normalize and write output O (bf16) and LSE (fp32)
    // ================================================================
    float l_final = l_prev;
    float m_final = m_prev;

    // Normalize
    float inv_l = (l_final > 0.0f) ? (1.0f / l_final) : 0.0f;

    for (int e = 0; e < EPT; e++) {
        int col = lane * EPT + e;
        if (q_row_in_bounds && global_qr < S && col < D) {
            O_bh[(uint64_t)global_qr * D + col] =
                __float2bfloat16(acc[e] * inv_l);
        }
    }

    // Write LSE (exactly once per query row: lane==0 is arbitrary representative)
    if (lane == 0 && q_row_in_bounds) {
        LSE_bh[global_qr] = (l_final > 0.0f)
            ? (m_final + logf(l_final))
            : (-FLT_MAX);
    }
}

void run(tvm::ffi::TensorView Q_tv, tvm::ffi::TensorView K_tv,
         tvm::ffi::TensorView V_tv, tvm::ffi::TensorView O_tv,
         tvm::ffi::TensorView LSE_tv) {
    CUDA_CHECK(cudaSetDevice(Q_tv.device().device_id));

    constexpr int64_t B = 4, H = 48, D_const = 128;
    int64_t S = Q_tv.size(2);

    const __nv_bfloat16* Q_gptr =
        static_cast<const __nv_bfloat16*>(Q_tv.data_ptr());
    const __nv_bfloat16* K_gptr =
        static_cast<const __nv_bfloat16*>(K_tv.data_ptr());
    const __nv_bfloat16* V_gptr =
        static_cast<const __nv_bfloat16*>(V_tv.data_ptr());
    __nv_bfloat16* O_gptr =
        static_cast<__nv_bfloat16*>(O_tv.data_ptr());
    float* LSE_gptr =
        static_cast<float*>(LSE_tv.data_ptr());

    // Grid: one block per (batch-head, query-tile)
    // Each block handles BM=64 query rows and all D=128 output columns.
    dim3 grid(B * H, (S + BM - 1) / BM);
    dim3 block(NT);

    // Shared memory: Q[BM][D] + K[BK][D] + V[BK][D]
    size_t smem_bytes = (BM * D + BK * D + BK * D) * sizeof(__nv_bfloat16);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q_tv.device().device_type, Q_tv.device().device_id));

    attention_kernel<<<grid, block, smem_bytes, stream>>>(
        Q_gptr, K_gptr, V_gptr, O_gptr, LSE_gptr, B, H, S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace mha_d128

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_d128::run);