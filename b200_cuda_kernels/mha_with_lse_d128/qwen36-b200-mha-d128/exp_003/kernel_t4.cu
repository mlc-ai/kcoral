#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
#include <cfloat>
#include <cmath>
#include <cassert>
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

constexpr int BM = 64;       // query rows per CTA
constexpr int BK = 16;       // key rows per chunk
constexpr int NT = 256;      // threads per block (= 4 warps)
constexpr int TPQR = NT / BM;  // threads per query-row = 4
constexpr int DIM = 128;     // head dimension
constexpr int EPT = DIM / TPQR; // elements per thread per row = 32

// Shared memory layout: Q[BM][DIM], K[BK][DIM], V[BK][DIM]
constexpr int SMEM_Q_ELTS = BM * DIM;   // 8192
constexpr int SMEM_K_ELTS = BK * DIM;   // 2048
constexpr int SMEM_V_ELTS = BK * DIM;   // 2048
constexpr int SMEM_TOTAL  = SMEM_Q_ELTS + SMEM_K_ELTS + SMEM_V_ELTS; // 12288

__global__ void attn_forward_kernel(
    const __nv_bfloat16* __restrict__ Q_gm,
    const __nv_bfloat16* __restrict__ K_gm,
    const __nv_bfloat16* __restrict__ V_gm,
    __nv_bfloat16* __restrict__ O_gm,
    float* __restrict__ LSE_out,
    int B, int H, int S)
{
    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* sQ = &smem[SMEM_Q_ELTS * 0];
    __nv_bfloat16* sK = &smem[SMEM_Q_ELTS * 1];
    __nv_bfloat16* sV = &smem[SMEM_Q_ELTS + SMEM_K_ELTS];

    int bh    = blockIdx.x;           // batch*H + head  [0..B*H)
    int qtile = blockIdx.y;           // query tile index [0..ceil(S/BM))

    int tid  = threadIdx.x;            // [0..256)
    int ridx = tid / TPQR;             // local row in [0..BM)
    int lane = tid % TPQR;             // lane in [0..TPQR)

    int gqr = qtile * BM + ridx;       // global query row index
    uint64_t base = static_cast<uint64_t>(bh) * S * DIM;  // start of (batch,head) slice

    float inv_sqrt_dim = rsqrtf(static_cast<float>(DIM));

    // ================================================================
    // STEP 1: cooperatively load Q tile into shared mem
    // ================================================================
    for (int idx = tid; idx < SMEM_Q_ELTS; idx += NT) {
        int rr = idx / DIM;
        int cc = idx % DIM;
        int global_row = qtile * BM + rr;
        sQ[idx] = (global_row < S)
            ? Q_gm[base + static_cast<uint64_t>(global_row) * DIM + cc]
            : __float2bfloat16(0.0f);
    }
    __syncthreads();

    // Thread-local state: FP32 accumulators for EPT output cols
    float acc[EPT];
    for (int e = 0; e < EPT; ++e) acc[e] = 0.0f;

    float mi = -FLT_MAX;  // running per-row max
    float li = 0.0f;      // running per-row sum-exp
    bool q_valid = (gqr < S);

    int num_chunks = (S + BK - 1) / BK;
    int col_start = lane * EPT;

    // ================================================================
    // STEP 2: stream K/V chunks through online softmax
    // ================================================================
    for (int ki = 0; ki < num_chunks; ++ki) {
        int k_off  = ki * BK;              // global key start
        int kv_eff = min(BK, S - k_off);   // actual valid keys in this chunk

        // Load K chunk
        for (int idx = tid; idx < SMEM_K_ELTS; idx += NT) {
            int rr = idx / DIM;
            int cc = idx % DIM;
            int gkr = k_off + rr;
            sK[idx] = (gkr < S)
                ? K_gm[base + static_cast<uint64_t>(gkr) * DIM + cc]
                : __float2bfloat16(0.0f);
        }

        // Load V chunk
        for (int idx = tid; idx < SMEM_V_ELTS; idx += NT) {
            int rr = idx / DIM;
            int cc = idx % DIM;
            int gkr = k_off + rr;
            sV[idx] = (gkr < S)
                ? V_gm[base + static_cast<uint64_t>(gkr) * DIM + cc]
                : __float2bfloat16(0.0f);
        }
        __syncthreads();

        // Compute Q·K^T scores for each key in this chunk
        float scores[BK];
        float cmax = -FLT_MAX;

        if (q_valid) {
            const __nv_bfloat162* qp = reinterpret_cast<const __nv_bfloat162*>(
                &sQ[ridx * DIM]);
            for (int k = 0; k < BK; ++k) {
                float dot = 0.0f;
                const __nv_bfloat162* kp = reinterpret_cast<const __nv_bfloat162*>(
                    &sK[k * DIM]);
                for (int gp = 0; gp < DIM / 2; ++gp) {
                    float qa_lo = __low2float(qp[gp]);
                    float qa_hi = __high2float(qp[gp]);
                    float ka_lo = __low2float(kp[gp]);
                    float ka_hi = __high2float(kp[gp]);
                    dot += qa_lo * ka_lo + qa_hi * ka_hi;
                }
                float sc = dot * inv_sqrt_dim;
                scores[k] = sc;
                if (sc > cmax) cmax = sc;
            }
        } else {
            for (int k = 0; k < BK; ++k) scores[k] = -FLT_MAX;
        }

        // --- Online softmax update ---
        float mi_old = mi;
        mi = max(mi, cmax);
        float alpha = expf(mi_old - mi);

        // Rescale accumulated output
        for (int e = 0; e < EPT; ++e) acc[e] *= alpha;
        li *= alpha;

        // Accumulate new attention-weighted contributions
        for (int k = 0; k < kv_eff; ++k) {
            float pw = expf(scores[k] - mi);
            li += pw;
            for (int e = 0; e < EPT; ++e) {
                acc[e] += pw * __bfloat162float(sV[k * DIM + col_start + e]);
            }
        }
    }

    // ================================================================
    // STEP 3: normalize and write output
    // ================================================================
    float scale_final = (li > 0.0f) ? (1.0f / li) : 0.0f;

    for (int e = 0; e < EPT; ++e) {
        int gc = col_start + e;
        if (q_valid && gc < DIM) {
            O_gm[base + static_cast<uint64_t>(gqr) * DIM + gc] =
                __float2bfloat16(acc[e] * scale_final);
        }
    }

    // Write LSE once per query row (lane 0 does it)
    if (lane == 0 && q_valid) {
        LSE_out[bh * S + gqr] = (li > 0.0f) ? (mi + logf(li)) : (-FLT_MAX);
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
    size_t smem_bytes = SMEM_TOTAL * sizeof(__nv_bfloat16);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q_tv.device().device_type, Q_tv.device().device_id));

    attn_forward_kernel<<<grid, block, smem_bytes, stream>>>(
        Q, K, V, O, LSE, B, H, S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace mha_d128

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_d128::run);