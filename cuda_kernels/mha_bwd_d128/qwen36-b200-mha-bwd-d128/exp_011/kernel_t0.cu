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

static constexpr int TILE_M = 32;
static constexpr int TILE_N = 32;
static constexpr int NUM_THREADS = 256;

/**
 * MHA backward kernel using 2D tiling with atomic accumulations.
 * 
 * For each (b,h), tiles the S×S attention space into TILE_M×TILE_N blocks.
 * Each block computes:
 *   score[q,k] = Q[q,:]·K[k,:] / sqrt(d)
 *   p[q,k] = exp(score - L[q])                       (softmax prob)
 *   dAttn[q,k] = dO[q,:]·V[k,:]                      (grad w.r.t. P)
 *   D[q] = Σ_d dO[q,d]*O[q,d]                        (softmax correction)
 *   dscore[q,k] = p*(dAttn - D[q]) * 1/sqrt(d)
 * 
 * Then atomically accumulates:
 *   dQ[q,d] += dscore[q,k] * K[k,d]
 *   dK[k,d] += dscore[q,k] * Q[q,d]
 *   dV[k,d] += p * dO[q,d]
 */
__global__ void mha_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O_fwd,
    const __nv_bfloat16* __restrict__ dO_in,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int64_t B, int64_t H, int64_t S, int64_t D,
    float inv_sqrt_d
) {
    static constexpr int PAIRS = (TILE_M * TILE_N) / NUM_THREADS;  // 4 pairs per thread

    // Shared memory layout:
    // sh_Q  [TILE_M][D]  -- 8KB
    // sh_K  [TILE_N][D]  -- 8KB
    // sh_V  [TILE_N][D]  -- 8KB
    // sh_O  [TILE_M][D]  -- 8KB
    // sh_dO [TILE_M][D]  -- 8KB
    // sh_Dq [TILE_M]     -- 128B  (fp32)
    // Total: ~41KB
    extern __shared__ __align__(16) unsigned char smem_raw[];

    __nv_bfloat16* sh_Q   = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sh_K   = sh_Q + TILE_M * D;
    __nv_bfloat16* sh_V   = sh_K + TILE_N * D;
    __nv_bfloat16* sh_O   = sh_V + TILE_N * D;
    __nv_bfloat16* sh_dO  = sh_O + TILE_M * D;
    float* sh_Dq           = reinterpret_cast<float*>(sh_dO + TILE_M * D);

    int64_t bh = blockIdx.z;
    int64_t q_s = blockIdx.x * TILE_M;
    int64_t k_s = blockIdx.y * TILE_N;

    int64_t q_e = (q_s + TILE_M < S) ? q_s + TILE_M : S;
    int64_t k_e = (k_s + TILE_N < S) ? k_s + TILE_N : S;
    int64_t nq = q_e - q_s;
    int64_t nk = k_e - k_s;

    int64_t bh_base = bh * S * D;

    // Cooperative load of Q, O, dO tiles into shared memory
    for (int64_t i = threadIdx.x; i < nq * D; i += NUM_THREADS) {
        int64_t r = i / D;
        int64_t c = i % D;
        int64_t g = bh_base + (q_s + r) * D + c;
        sh_Q[i]    = Q[g];
        sh_O[i]    = O_fwd[g];
        sh_dO[i]   = dO_in[g];
    }

    // Cooperative load of K, V tiles into shared memory
    for (int64_t i = threadIdx.x; i < nk * D; i += NUM_THREADS) {
        int64_t r = i / D;
        int64_t c = i % D;
        int64_t g = bh_base + (k_s + r) * D + c;
        sh_K[i]   = K[g];
        sh_V[i]   = V[g];
    }

    __syncthreads();

    // Precompute D[q] = sum_d(dO[q,d] * O_fwd[q,d]) for softmax backward
    for (int64_t sq = threadIdx.x; sq < nq; sq += NUM_THREADS) {
        float sum = 0.0f;
        for (int64_t dd = 0; dd < D; dd++) {
            sum += __bfloat162float(sh_dO[sq * D + dd]) * __bfloat162float(sh_O[sq * D + dd]);
        }
        sh_Dq[sq] = sum;
    }
    __syncthreads();

    // Each thread processes PAIRS=4 (q,k) pairs within this tile
    int64_t base = threadIdx.x * PAIRS;
    for (int64_t p = 0; p < PAIRS; p++) {
        int64_t idx = base + p;
        if (idx >= nq * nk) break;

        int64_t sq = idx / TILE_N;
        int64_t sk = idx % TILE_N;
        if (sq >= nq || sk >= nk) continue;

        int64_t qg = q_s + sq;  // global query index
        int64_t kg = k_s + sk;  // global key index

        const __nv_bfloat16* qrow  = &sh_Q[sq * D];
        const __nv_bfloat16* krow  = &sh_K[sk * D];
        const __nv_bfloat16* vrow  = &sh_V[sk * D];
        const __nv_bfloat16* dorow = &sh_dO[sq * D];

        // Compute score = Q[q,:] · K[k,:] / sqrt(d)
        float score = 0.0f;
        for (int64_t dd = 0; dd < D; dd++) {
            score += __bfloat162float(qrow[dd]) * __bfloat162float(krow[dd]);
        }
        score *= inv_sqrt_d;

        // Softmax probability: P[q,k] = exp(score - LSE[q])
        float lse = L[bh * S + qg];
        float pval = expf(score - lse);

        // Compute dAttn[q,k] = dO[q,:] · V[k,:]
        float dattn = 0.0f;
        for (int64_t dd = 0; dd < D; dd++) {
            dattn += __bfloat162float(dorow[dd]) * __bfloat162float(vrow[dd]);
        }

        // Backprop through softmax: dscore = P * (dAttn - D[q]) * 1/sqrt(d)
        float dscore = pval * (dattn - sh_Dq[sq]) * inv_sqrt_d;

        // Output offsets in global memory
        int64_t off_dV = bh_base + kg * D;
        int64_t off_dQ = bh_base + qg * D;
        int64_t off_dK = bh_base + kg * D;

        // Atomic accumulation of gradient contributions
        for (int64_t dd = 0; dd < D; dd++) {
            float fd = __bfloat162float(dorow[dd]);
            float fk = __bfloat162float(krow[dd]);
            float fq = __bfloat162float(qrow[dd]);

            // dV[k,d] += P[q,k] * dO[q,d]
            atomicAdd(&dV[off_dV + dd], __float2bfloat16(pval * fd));
            // dQ[q,d] += dscore[q,k] * K[k,d]
            atomicAdd(&dQ[off_dQ + dd], __float2bfloat16(dscore * fk));
            // dK[k,d] += dscore[q,k] * Q[q,d]
            atomicAdd(&dK[off_dK + dd], __float2bfloat16(dscore * fq));
        }
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

    // Zero-initialize output tensors before atomic accumulation
    size_t out_bytes = static_cast<size_t>(B) * H * S * D * 2;
    CUDA_CHECK(cudaMemsetAsync(dQ_out.data_ptr(), 0, out_bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(dK_out.data_ptr(), 0, out_bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(dV_out.data_ptr(), 0, out_bytes, stream));

    // Grid: x=q-tiles, y=k-tiles, z=batch*head
    dim3 block(NUM_THREADS, 1, 1);
    dim3 grid(
        static_cast<int>((S + TILE_M - 1) / TILE_M),
        static_cast<int>((S + TILE_N - 1) / TILE_N),
        static_cast<int>(B * H)
    );

    // Dynamic shared memory: 5 bf16 tiles + 1 float array
    size_t smem_bytes = static_cast<size_t>(5) * TILE_M * D * 2 + TILE_M * 4;

    float inv_sqrt_d = 1.0f / sqrtf(static_cast<float>(D));

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