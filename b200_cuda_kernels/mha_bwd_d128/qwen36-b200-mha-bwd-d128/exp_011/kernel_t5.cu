#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdio.h>
#include <cstdint>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call)                                           \
    do {                                                           \
        cudaError_t _e = (call);                                   \
        if (_e != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error %s at %s:%d\n",           \
                    cudaGetErrorString(_e), __FILE__, __LINE__);   \
            exit(1);                                               \
        }                                                          \
    } while (0)

static constexpr int TILE_Q = 32;
static constexpr int TILE_K = 32;
static constexpr int NUM_THREADS = 128;
static constexpr int PAIRS_PER_THREAD = (TILE_Q * TILE_K) / NUM_THREADS;

__global__ void mha_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O_fwd,
    const __nv_bfloat16* __restrict__ dO_in,
    const float* __restrict__ LSE,
    __nv_bfloat16* __restrict__ dQ_out,
    __nv_bfloat16* __restrict__ dK_out,
    __nv_bfloat16* __restrict__ dV_out,
    int64_t B, int64_t H, int64_t S, int64_t D,
    float inv_sqrt_d
) {
    // Shared memory layout for D=128:
    // sh_Q[TILE_Q][D]     bf16 = 8KB
    // sh_O[TILE_Q][D]     bf16 = 8KB
    // sh_dO[TILE_Q][D]    bf16 = 8KB
    // sh_K[TILE_K][D]     bf16 = 8KB
    // sh_V[TILE_K][D]     bf16 = 8KB
    // sh_dk[TILE_K*D]     fp32 = 16KB (per-tile accumulation)
    // Total ~= 56KB
    
    extern __shared__ __align__(16) unsigned char smem_raw[];
    
    __nv_bfloat16* sh_Q     = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sh_O     = sh_Q + TILE_Q * D;
    __nv_bfloat16* sh_dO    = sh_O + TILE_Q * D;
    __nv_bfloat16* sh_K     = sh_dO + TILE_Q * D;
    __nv_bfloat16* sh_V     = sh_K + TILE_K * D;
    float* sh_dk_acc        = reinterpret_cast<float*>(sh_V + TILE_K * D);

    int tid = threadIdx.x;
    int64_t bh       = blockIdx.z;
    int64_t ktile    = blockIdx.y;
    int64_t qtile    = blockIdx.x;
    int64_t k_s      = ktile * TILE_K;
    int64_t q_s      = qtile * TILE_Q;
    int64_t bh_off   = bh * S * D;

    int64_t nq_tiles = (S + TILE_Q - 1) / TILE_Q;
    int64_t nk_tiles = (S + TILE_K - 1) / TILE_K;
    
    int64_t k_end = min(k_s + TILE_K, S);
    int64_t nk = k_end - k_s;

    // Per-block accumulators for dK and dV (reduced, written at end)
    // Each thread handles specific sk rows, accumulates into registers then flushes to shared mem

    // Pre-allocate register arrays
    float reg_dk[NUM_THREADS];      // Accumulate dK contributions for assigned elements
    float reg_dv[NUM_THREADS];      // Accumulate dV contributions for assigned elements

    // Initialize global outputs will be done before kernel call via memset
    // Each kernel invocation writes exactly one tile of each output

    // Load K and V tiles ONCE at start (constant across q tiles)
    for (int64_t i = tid; i < nk * D; i += NUM_THREADS) {
        int64_t r = i / D, c = i % D;
        int64_t g = bh_off + (k_s + r) * D + c;
        sh_K[i] = K[g];
        sh_V[i] = V[g];
    }
    __syncthreads();

    // Iterate over q-tiles contributing to this k-tile
    for (int64_t qt = 0; qt < nq_tiles; qt++) {
        int64_t qs_qt = qt * TILE_Q;
        int64_t qend_qt = min(qs_qt + TILE_Q, S);
        int64_t nqq = qend_qt - qs_qt;
        if (nqq == 0) continue;

        // Cooperative load Q, O, dO tile
        for (int64_t i = tid; i < nqq * D; i += NUM_THREADS) {
            int64_t r = i / D, c = i % D;
            int64_t g = bh_off + (qs_qt + r) * D + c;
            sh_Q[i]  = Q[g];
            sh_O[i]  = O_fwd[g];
            sh_dO[i] = dO_in[g];
        }
        __syncthreads();

        // Precompute D[q] for softmax backward
        float reg_Dq[TILE_Q];
        for (int64_t sq = tid; sq < nqq; sq += NUM_THREADS) {
            float s = 0.0f;
            for (int dd = 0; dd < D; dd++) {
                s += __bfloat162float(sh_dO[sq * D + dd]) * __bfloat162float(sh_O[sq * D + dd]);
            }
            reg_Dq[sq] = s;
        }
        __syncthreads();

        // Compute attention scores and gradients
        #pragma unroll 4
        for (int p = 0; p < PAIRS_PER_THREAD; p++) {
            int64_t gp = tid * PAIRS_PER_THREAD + p;
            if (gp >= static_cast<int64_t>(nqq) * nk) break;

            int64_t sq = gp / TILE_K;
            int64_t sk = gp % TILE_K;

            // Score: Q[q,:] . K[k,:] / sqrt(D)
            float score = 0.0f;
            for (int dd = 0; dd < D; dd++) {
                score += __bfloat162float(sh_Q[sq * D + dd]) * __bfloat162float(sh_K[sk * D + dd]);
            }
            score *= inv_sqrt_d;

            // P = exp(score - LSE)
            int64_t qg = qs_qt + sq;
            float pval = expf(score - LSE[bh * S + qg]);

            // dAttn = dO[q,:] . V[k,:]
            float dattn = 0.0f;
            for (int dd = 0; dd < D; dd++) {
                dattn += __bfloat162float(sh_dO[sq * D + dd]) * __bfloat162float(sh_V[sk * D + dd]);
            }

            // dScore = P * (dAttn - D[q]) / sqrt(D)
            float dscore = pval * (dattn - reg_Dq[sq]) * inv_sqrt_d;

            // Now store back contribution pointers
            // We'll write per-thread partial results later
        }
        
        __syncthreads();
    }
}

namespace mha_bwd_impl {

void run(
    tvm::ffi::TensorView Q,
    tvm::ffi::TensorView K,
    tvm::ffi::TensorView V,
    tvm::ffi::TensorView O_fwd,
    tvm::ffi::TensorView dO_in,
    tvm::ffi::TensorView L,
    tvm::ffi::TensorView dQ_out,
    tvm::ffi::TensorView dK_out,
    tvm::ffi::TensorView dV_out
) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    float inv_sqrt_d = 1.0f / sqrtf(static_cast<float>(D));

    dim3 block(NUM_THREADS);
    dim3 grid(
        static_cast<uint32_t>((S + TILE_Q - 1) / TILE_Q),
        static_cast<uint32_t>((S + TILE_K - 1) / TILE_K),
        static_cast<uint32_t>(B * H)
    );

    size_t smem_bytes = 3LL * TILE_Q * D * sizeof(__nv_bfloat16) 
                      + 2LL * TILE_K * D * sizeof(__nv_bfloat16)
                      + TILE_K * D * sizeof(float);

    mha_bwd_kernel<<<grid, block, smem_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(O_fwd.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO_in.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ_out.data_ptr()),
        static_cast<__nv_bfloat16*>(dK_out.data_ptr()),
        static_cast<__nv_bfloat16*>(dV_out.data_ptr()),
        B, H, S, D, inv_sqrt_d
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);

}  // namespace mha_bwd_impl