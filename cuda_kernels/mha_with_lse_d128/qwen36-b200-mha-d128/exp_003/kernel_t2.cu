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

constexpr int BM = 64;       // output rows per CTA
constexpr int BK = 16;       // key tokens per chunk
constexpr int NT = 256;      // threads per block
constexpr int TPR = NT / BM; // threads per output row = 4
constexpr int DIM = 128;     // head dimension
constexpr int EPS = DIM / TPR; // elements per thread per row = 32

__global__ void attn_kernel(
    const __nv_bfloat16* __restrict__ Q_gmem,
    const __nv_bfloat16* __restrict__ K_gmem,
    const __nv_bfloat16* __restrict__ V_gmem,
    __nv_bfloat16* __restrict__ O_gmem,
    float* __restrict__ LSE_gmem,
    int B, int H, int S)
{
    // Shared memory: Q[BM][DIM] + K[BK][DIM] + V[BK][DIM]
    extern __shared__ __align__(256) unsigned char smem_raw[];
    __nv_bfloat16* Q_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* K_smem = Q_smem + BM * DIM;
    __nv_bfloat16* V_smem = K_smem + BK * DIM;

    int bh     = blockIdx.x;             // batch-head combo [0..B*H)
    int qb_idx = blockIdx.y;             // query-block index [0..ceil(S/BM))

    int tid = threadIdx.x;               // [0..NT)
    int thr_rid = tid / TPR;             // row-id within tile [0..BM)
    int thr_lane = tid % TPR;            // lane-id [0..TPR)

    // Global offset for this Q row
    int gqr = qb_idx * BM + thr_rid;
    bool valid_qrow = (gqr < S);

    // Base pointers for this (batch, head)
    uint64_t bh_offset = static_cast<uint64_t>(bh) * S * DIM;
    const __nv_bfloat16* Q_base = Q_gmem + bh_offset;
    const __nv_bfloat16* K_base = K_gmem + bh_offset;
    const __nv_bfloat16* V_base = V_gmem + bh_offset;
    __nv_bfloat16*         O_base = O_gmem + bh_offset;
    float*                 LSE_base = LSE_gmem + bh * S;

    float inv_sqrt_d = rsqrtf(static_cast<float>(DIM));

    // =====================================================================
    // Step 1: Cooperative load Q tile into shared memory
    // =====================================================================
    for (int i = tid; i < BM * DIM; i += NT) {
        int sr = i / DIM;
        int sc = i % DIM;
        int gr = qb_idx * BM + sr;
        Q_smem[sr * DIM + sc] = (gr < S)
            ? Q_base[static_cast<uint64_t>(gr) * DIM + sc]
            : __float2bfloat16(0.0f);
    }
    __syncthreads();

    // Per-thread FP32 accumulators for EPS output elements
    float acc[EPS];
    for (int e = 0; e < EPS; ++e) acc[e] = 0.0f;

    // Online softmax state
    float mi = -FLT_MAX;   // running row max
    float li = 0.0f;       // running sum-exp

    int nk = (S + BK - 1) / BK;  // number of key-chunks

    // =====================================================================
    // Step 2: Streaming K/V chunks
    // =====================================================================
    for (int ki = 0; ki < nk; ++ki) {
        int ks = ki * BK;                      // global key-start
        int kv_eff = (ks + BK <= S) ? BK : S - ks;  // effective keys (clamped)

        // --- Cooperative load K chunk [BK][DIM] ---
        for (int i = tid; i < BK * DIM; i += NT) {
            int sr = i / DIM;
            int sc = i % DIM;
            int gk = ks + sr;
            K_smem[sr * DIM + sc] = (gk < S)
                ? K_base[static_cast<uint64_t>(gk) * DIM + sc]
                : __float2bfloat16(0.0f);
        }

        // --- Cooperative load V chunk [BK][DIM] ---
        for (int i = tid; i < BK * DIM; i += NT) {
            int sr = i / DIM;
            int sc = i % DIM;
            int gk = ks + sr;
            V_smem[sr * DIM + sc] = (gk < S)
                ? V_base[static_cast<uint64_t>(gk) * DIM + sc]
                : __float2bfloat16(0.0f);
        }
        __syncthreads();

        // --- Compute Q(row) · K(k)^T for each key in this chunk ---
        float cmax = -FLT_MAX;  // local chunk-max for this row
        float attn_w[BK];       // attention weights for this chunk

        const __nv_bfloat162* qp =
            reinterpret_cast<const __nv_bfloat162*>(&Q_smem[thr_rid * DIM]);

        if (valid_qrow) {
            for (int k = 0; k < BK; ++k) {
                float s = 0.0f;
                const __nv_bfloat162* kp =
                    reinterpret_cast<const __nv_bfloat162*>(&K_smem[k * DIM]);
                // Unrolled bf16->fp32 unpack with FMA (DIM/2 = 64 packed pairs)
                #pragma unroll
                for (int dp = 0; dp < DIM / 2; ++dp) {
                    float qa = __low2float(qp[dp]);
                    float qb = __high2float(qp[dp]);
                    float ka = __low2float(kp[dp]);
                    float kb = __high2float(kp[dp]);
                    s += qa * ka + qb * kb;
                }
                attn_w[k] = s * inv_sqrt_d;
                if (attn_w[k] > cmax) cmax = attn_w[k];
            }
        } else {
            // Padding row → all scores = -FLT_MAX
            for (int k = 0; k < BK; ++k) attn_w[k] = -FLT_MAX;
        }

        // --- Online softmax: compare chunk-max with running max ---
        float m_prev = mi;
        mi = (m_prev > cmax) ? m_prev : cmax;
        float scale = expf(m_prev - mi);  // ≤ 1.0

        // Rescale previous accumulator
        #pragma unroll
        for (int e = 0; e < EPS; ++e) acc[e] *= scale;

        // Accumulate new chunk (only over effective keys)
        float new_li = li * scale;
        int col_base = thr_lane * EPS;

        for (int k = 0; k < kv_eff; ++k) {
            float w = expf(attn_w[k] - mi);
            new_li += w;
            const __nv_bfloat16* vk = &V_smem[k * DIM + col_base];
            #pragma unroll
            for (int e = 0; e < EPS; ++e) {
                acc[e] += w * __bfloat162float(vk[e]);
            }
        }
        li = new_li;
    }

    // =====================================================================
    // Step 3: Normalize, convert to BF16, write output O and LSE
    // =====================================================================
    float inv_li = (li > 0.0f) ? (1.0f / li) : 0.0f;
    int col_base = thr_lane * EPS;

    for (int e = 0; e < EPS; ++e) {
        int gc = col_base + e;
        if (valid_qrow && gqr < S && gc < DIM) {
            O_base[static_cast<uint64_t>(gqr) * DIM + gc] =
                __float2bfloat16(acc[e] * inv_li);
        }
    }

    // Write LSE once per query row (any single lane)
    if (thr_lane == 0 && valid_qrow) {
        LSE_base[gqr] = (li > 0.0f) ? (mi + logf(li)) : (-FLT_MAX);
    }
}

void run(tvm::ffi::TensorView Q_tv, tvm::ffi::TensorView K_tv,
         tvm::ffi::TensorView V_tv, tvm::ffi::TensorView O_tv,
         tvm::ffi::TensorView LSE_tv) {
    CUDA_CHECK(cudaSetDevice(Q_tv.device().device_id));

    constexpr int64_t B = 4, H = 48;
    int64_t S = Q_tv.size(2);

    const __nv_bfloat16* Q = static_cast<const __nv_bfloat16*>(Q_tv.data_ptr());
    const __nv_bfloat16* K = static_cast<const __nv_bfloat16*>(K_tv.data_ptr());
    const __nv_bfloat16* V = static_cast<const __nv_bfloat16*>(V_tv.data_ptr());
    __nv_bfloat16*        O = static_cast<__nv_bfloat16*>(O_tv.data_ptr());
    float*               LSE = static_cast<float*>(LSE_tv.data_ptr());

    dim3 grid(B * H, (S + BM - 1) / BM);
    dim3 block(NT);

    // SMEM: Q[BM][DIM] + K[BK][DIM] + V[BK][DIM]
    size_t smem_bytes = (BM * DIM + BK * DIM + BK * DIM) * sizeof(__nv_bfloat16);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q_tv.device().device_type, Q_tv.device().device_id));

    attn_kernel<<<grid, block, smem_bytes, stream>>>(
        Q, K, V, O, LSE, B, H, S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace mha_d128

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_d128::run);