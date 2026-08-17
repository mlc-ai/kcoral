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
static constexpr int NUM_THREADS = 128;
static constexpr int PAIRS_PER_THREAD = (TILE_M * TILE_N) / NUM_THREADS;

template <int D>
__global__ void mha_bwd_dq_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O_fwd,
    const __nv_bfloat16* __restrict__ dO_in,
    const float* __restrict__ LSE,
    __nv_bfloat16* __restrict__ dQ_out,
    int64_t B, int64_t H, int64_t S,
    float inv_sqrt_d
) {
    // Shared memory for one q-tile: Q, O, dO
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16* smem_q   = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* smem_o   = smem_q + TILE_M * D;
    __nv_bfloat16* smem_do  = smem_o + TILE_M * D;
    __nv_bfloat16* smem_k   = smem_do + TILE_M * D;
    __nv_bfloat16* smem_v   = smem_k + TILE_N * D;
    float* smem_dq_acc      = reinterpret_cast<float*>(smem_v + TILE_N * D);

    int tid = threadIdx.x;
    int64_t bh = blockIdx.z;
    int64_t qtile = blockIdx.x;
    int64_t q_s = qtile * TILE_M;
    int64_t q_end = min(q_s + TILE_M, S);
    int64_t nq = q_end - q_s;
    int64_t bh_off = bh * S * D;
    int64_t n_ktiles = (S + TILE_N - 1) / TILE_N;

    // Cooperative load Q tile
    for (int64_t i = tid; i < nq * D; i += NUM_THREADS) {
        int64_t r = i / D, c = i % D;
        int64_t g = bh_off + (q_s + r) * D + c;
        smem_q[i]  = Q[g];
        smem_o[i]  = O_fwd[g];
        smem_do[i] = dO_in[g];
    }

    // Initialize dQ accumulator in shared memory
    for (int64_t i = tid; i < nq * D; i += NUM_THREADS) {
        smem_dq_acc[i] = 0.0f;
    }
    __syncthreads();

    // Precompute D[q] for softmax backward
    // Store in registers since we need it per-thread
    float reg_Dq[nq > 128 ? 128 : nq];
    if (nq <= 128) {
        for (int64_t sq = tid; sq < nq; sq += NUM_THREADS) {
            float s = 0.0f;
            for (int dd = 0; dd < D; dd++) {
                s += __bfloat162float(smem_do[sq * D + dd]) * __bfloat162float(smem_o[sq * D + dd]);
            }
            reg_Dq[sq] = s;
        }
    } else {
        // Fallback: use shared memory for Dq
        for (int64_t sq = tid; sq < nq; sq += NUM_THREADS) {
            float s = 0.0f;
            for (int dd = 0; dd < D; dd++) {
                s += __bfloat162float(smem_do[sq * D + dd]) * __bfloat162float(smem_o[sq * D + dd]);
            }
            // Store in unused part of shared mem - but this won't work cleanly
            // Since nq <= 32 always (TILE_M=32), nq <= 128 branch is always taken
        }
    }

    // Sweep over all k tiles
    for (int64_t kt = 0; kt < n_ktiles; kt++) {
        int64_t k_s = kt * TILE_N;
        int64_t k_end_k = min(k_s + TILE_N, S);
        int64_t nk = k_end_k - k_s;
        if (nk == 0) continue;

        // Cooperative load K, V tile
        for (int64_t i = tid; i < nk * D; i += NUM_THREADS) {
            int64_t r = i / D, c = i % D;
            int64_t g = bh_off + (k_s + r) * D + c;
            smem_k[i] = K[g];
            smem_v[i] = V[g];
        }
        __syncthreads();

        // Each thread processes PAIRS_PER_THREAD (sq, sk) pairs
        #pragma unroll
        for (int p = 0; p < PAIRS_PER_THREAD; p++) {
            int64_t gp = tid * PAIRS_PER_THREAD + p;
            if (gp >= nq * nk) break;

            int64_t sq = gp / TILE_N;
            int64_t sk = gp % TILE_N;

            const __nv_bfloat16* qrow  = &smem_q[sq * D];
            const __nv_bfloat16* krow  = &smem_k[sk * D];
            const __nv_bfloat16* vrow  = &smem_v[sk * D];
            const __nv_bfloat16* dorow = &smem_do[sq * D];

            // Score: Q . K / sqrt(D)
            float score = 0.0f;
            #pragma unroll
            for (int dd = 0; dd < D; dd++) {
                score += __bfloat162float(qrow[dd]) * __bfloat162float(krow[dd]);
            }
            score *= inv_sqrt_d;

            // P = exp(score - LSE)
            int64_t qg = q_s + sq;
            float lse_val = LSE[bh * S + qg];
            float pval = expf(score - lse_val);

            // dAttn = dO . V
            float dattn = 0.0f;
            #pragma unroll
            for (int dd = 0; dd < D; dd++) {
                dattn += __bfloat162float(dorow[dd]) * __bfloat162float(vrow[dd]);
            }

            // dScore = P * (dAttn - D[q]) / sqrt(D)
            float dscore = pval * (dattn - reg_Dq[sq]) * inv_sqrt_d;

            // Accumulate dQ: dQ[q,d] += dScore * K[k,d]
            #pragma unroll
            for (int dd = 0; dd < D; dd++) {
                float fk = __bfloat162float(krow[dd]);
                smem_dq_acc[sq * D + dd] += dscore * fk;
            }
        }
        __syncthreads();
    }

    // Write dQ cohesively (no atomics!)
    for (int64_t i = tid; i < nq * D; i += NUM_THREADS) {
        int64_t r = i / D, c = i % D;
        dQ_out[bh_off + (q_s + r) * D + c] = __float2bfloat16(smem_dq_acc[i]);
    }
}

template <int D>
__global__ void mha_bwd_dk_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O_fwd,
    const __nv_bfloat16* __restrict__ dO_in,
    const float* __restrict__ LSE,
    __nv_bfloat16* __restrict__ dK_out,
    int64_t B, int64_t H, int64_t S,
    float inv_sqrt_d
) {
    // Shared memory for one k-tile: K, dK accumulator
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16* smem_q   = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* smem_o   = smem_q + TILE_M * D;
    __nv_bfloat16* smem_do  = smem_o + TILE_M * D;
    __nv_bfloat16* smem_k   = smem_do + TILE_M * D;
    __nv_bfloat16* smem_v   = smem_k + TILE_N * D;
    float* smem_dk_acc      = reinterpret_cast<float*>(smem_v + TILE_N * D);

    int tid = threadIdx.x;
    int64_t bh = blockIdx.z;
    int64_t ktile = blockIdx.x;
    int64_t k_s = ktile * TILE_N;
    int64_t k_end = min(k_s + TILE_N, S);
    int64_t nk = k_end - k_s;
    int64_t bh_off = bh * S * D;
    int64_t n_qtiles = (S + TILE_M - 1) / TILE_M;

    // Cooperative load K tile (once)
    for (int64_t i = tid; i < nk * D; i += NUM_THREADS) {
        int64_t r = i / D, c = i % D;
        int64_t g = bh_off + (k_s + r) * D + c;
        smem_k[i] = K[g];
    }

    // Initialize dK accumulator in shared memory
    for (int64_t i = tid; i < nk * D; i += NUM_THREADS) {
        smem_dk_acc[i] = 0.0f;
    }
    __syncthreads();

    // Sweep over all q tiles
    for (int64_t qt = 0; qt < n_qtiles; qt++) {
        int64_t q_s_qt = qt * TILE_M;
        int64_t q_end_qt = min(q_s_qt + TILE_M, S);
        int64_t nq_qt = q_end_qt - q_s_qt;
        if (nq_qt == 0) continue;

        // Cooperative load Q, O, dO tile
        for (int64_t i = tid; i < nq_qt * D; i += NUM_THREADS) {
            int64_t r = i / D, c = i % D;
            int64_t g = bh_off + (q_s_qt + r) * D + c;
            smem_q[i]  = Q[g];
            smem_o[i]  = O_fwd[g];
            smem_do[i] = dO_in[g];
        }
        __syncthreads();

        // Cooperative load V tile
        for (int64_t i = tid; i < nk * D; i += NUM_THREADS) {
            int64_t r = i / D, c = i % D;
            int64_t g = bh_off + (k_s + r) * D + c;
            smem_v[i] = V[g];
        }
        __syncthreads();

        // Precompute D[q] for this q tile
        float reg_Dq[nq_qt];
        for (int64_t sq = tid; sq < nq_qt; sq += NUM_THREADS) {
            float s = 0.0f;
            for (int dd = 0; dd < D; dd++) {
                s += __bfloat162float(smem_do[sq * D + dd]) * __bfloat162float(smem_o[sq * D + dd]);
            }
            reg_Dq[sq] = s;
        }
        __syncthreads();

        // Each thread processes pairs
        #pragma unroll
        for (int p = 0; p < PAIRS_PER_THREAD; p++) {
            int64_t gp = tid * PAIRS_PER_THREAD + p;
            if (gp >= nq_qt * nk) break;

            int64_t sq = gp / TILE_N;
            int64_t sk = gp % TILE_N;

            const __nv_bfloat16* qrow  = &smem_q[sq * D];
            const __nv_bfloat16* krow  = &smem_k[sk * D];
            const __nv_bfloat16* vrow  = &smem_v[sk * D];
            const __nv_bfloat16* dorow = &smem_do[sq * D];

            // Score: Q . K / sqrt(D)
            float score = 0.0f;
            #pragma unroll
            for (int dd = 0; dd < D; dd++) {
                score += __bfloat162float(qrow[dd]) * __bfloat162float(krow[dd]);
            }
            score *= inv_sqrt_d;

            // P = exp(score - LSE)
            int64_t qg = q_s_qt + sq;
            float lse_val = LSE[bh * S + qg];
            float pval = expf(score - lse_val);

            // dAttn = dO . V
            float dattn = 0.0f;
            #pragma unroll
            for (int dd = 0; dd < D; dd++) {
                dattn += __bfloat162float(dorow[dd]) * __bfloat162float(vrow[dd]);
            }

            // dScore = P * (dAttn - D[q]) / sqrt(D)
            float dscore = pval * (dattn - reg_Dq[sq]) * inv_sqrt_d;

            // Accumulate dK: dK[k,d] += dScore * Q[q,d]
            #pragma unroll
            for (int dd = 0; dd < D; dd++) {
                float fq = __bfloat162float(qrow[dd]);
                smem_dk_acc[sk * D + dd] += dscore * fq;
            }
        }
        __syncthreads();
    }

    // Write dK cohesively (no atomics!)
    for (int64_t i = tid; i < nk * D; i += NUM_THREADS) {
        int64_t r = i / D, c = i % D;
        dK_out[bh_off + (k_s + r) * D + c] = __float2bfloat16(smem_dk_acc[i]);
    }
}

template <int D>
__global__ void mha_bwd_dv_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O_fwd,
    const __nv_bfloat16* __restrict__ dO_in,
    const float* __restrict__ LSE,
    __nv_bfloat16* __restrict__ dV_out,
    int64_t B, int64_t H, int64_t S,
    float inv_sqrt_d
) {
    // Shared memory for one k-tile: V, dV accumulator
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16* smem_q   = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* smem_o   = smem_q + TILE_M * D;
    __nv_bfloat16* smem_do  = smem_o + TILE_M * D;
    __nv_bfloat16* smem_k   = smem_do + TILE_M * D;
    __nv_bfloat16* smem_v   = smem_k + TILE_N * D;
    float* smem_dv_acc      = reinterpret_cast<float*>(smem_v + TILE_N * D);

    int tid = threadIdx.x;
    int64_t bh = blockIdx.z;
    int64_t ktile = blockIdx.x;
    int64_t k_s = ktile * TILE_N;
    int64_t k_end = min(k_s + TILE_N, S);
    int64_t nk = k_end - k_s;
    int64_t bh_off = bh * S * D;
    int64_t n_qtiles = (S + TILE_M - 1) / TILE_M;

    // Initialize dV accumulator in shared memory
    for (int64_t i = tid; i < nk * D; i += NUM_THREADS) {
        smem_dv_acc[i] = 0.0f;
    }
    __syncthreads();

    // Sweep over all q tiles
    for (int64_t qt = 0; qt < n_qtiles; qt++) {
        int64_t q_s_qt = qt * TILE_M;
        int64_t q_end_qt = min(q_s_qt + TILE_M, S);
        int64_t nq_qt = q_end_qt - q_s_qt;
        if (nq_qt == 0) continue;

        // Cooperative load Q, O, dO tile
        for (int64_t i = tid; i < nq_qt * D; i += NUM_THREADS) {
            int64_t r = i / D, c = i % D;
            int64_t g = bh_off + (q_s_qt + r) * D + c;
            smem_q[i]  = Q[g];
            smem_o[i]  = O_fwd[g];
            smem_do[i] = dO_in[g];
        }
        __syncthreads();

        // Cooperative load K tile
        for (int64_t i = tid; i < nk * D; i += NUM_THREADS) {
            int64_t r = i / D, c = i % D;
            int64_t g = bh_off + (k_s + r) * D + c;
            smem_k[i] = K[g];
        }
        __syncthreads();

        // Cooperative load V tile
        for (int64_t i = tid; i < nk * D; i += NUM_THREADS) {
            int64_t r = i / D, c = i % D;
            int64_t g = bh_off + (k_s + r) * D + c;
            smem_v[i] = V[g];
        }
        __syncthreads();

        // Precompute D[q] for this q tile
        float reg_Dq[nq_qt];
        for (int64_t sq = tid; sq < nq_qt; sq += NUM_THREADS) {
            float s = 0.0f;
            for (int dd = 0; dd < D; dd++) {
                s += __bfloat162float(smem_do[sq * D + dd]) * __bfloat162float(smem_o[sq * D + dd]);
            }
            reg_Dq[sq] = s;
        }
        __syncthreads();

        // Each thread processes pairs
        #pragma unroll
        for (int p = 0; p < PAIRS_PER_THREAD; p++) {
            int64_t gp = tid * PAIRS_PER_THREAD + p;
            if (gp >= nq_qt * nk) break;

            int64_t sq = gp / TILE_N;
            int64_t sk = gp % TILE_N;

            const __nv_bfloat16* qrow  = &smem_q[sq * D];
            const __nv_bfloat16* krow  = &smem_k[sk * D];
            const __nv_bfloat16* vrow  = &smem_v[sk * D];
            const __nv_bfloat16* dorow = &smem_do[sq * D];

            // Score: Q . K / sqrt(D)
            float score = 0.0f;
            #pragma unroll
            for (int dd = 0; dd < D; dd++) {
                score += __bfloat162float(qrow[dd]) * __bfloat162float(krow[dd]);
            }
            score *= inv_sqrt_d;

            // P = exp(score - LSE)
            int64_t qg = q_s_qt + sq;
            float lse_val = LSE[bh * S + qg];
            float pval = expf(score - lse_val);

            // dAttn = dO . V
            float dattn = 0.0f;
            #pragma unroll
            for (int dd = 0; dd < D; dd++) {
                dattn += __bfloat162float(dorow[dd]) * __bfloat162float(vrow[dd]);
            }

            // dScore = P * (dAttn - D[q]) / sqrt(D)
            float dscore = pval * (dattn - reg_Dq[sq]) * inv_sqrt_d;

            // Accumulate dV: dV[k,d] += P * dO[q,d]
            #pragma unroll
            for (int dd = 0; dd < D; dd++) {
                float fdo = __bfloat162float(dorow[dd]);
                smem_dv_acc[sk * D + dd] += pval * fdo;
            }
        }
        __syncthreads();
    }

    // Write dV cohesively (no atomics!)
    for (int64_t i = tid; i < nk * D; i += NUM_THREADS) {
        int64_t r = i / D, c = i % D;
        dV_out[bh_off + (k_s + r) * D + c] = __float2bfloat16(smem_dv_acc[i]);
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

    // Zero initialize outputs
    size_t out_bytes = static_cast<size_t>(B) * H * S * D * sizeof(__nv_bfloat16);
    CUDA_CHECK(cudaMemsetAsync(dQ_out.data_ptr(), 0, out_bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(dK_out.data_ptr(), 0, out_bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(dV_out.data_ptr(), 0, out_bytes, stream));

    float inv_sqrt_d = 1.0f / sqrtf(static_cast<float>(D));

    // Shared memory: 
    // dQ kernel: Q(8KB) + O(8KB) + dO(8KB) + K(8KB) + V(8KB) + dq_acc(32*128*4=16KB) = ~56KB
    // dk/dv kernel: same + dk/dv_acc = ~56KB
    size_t smem_bytes = static_cast<size_t>(5) * TILE_M * D * sizeof(__nv_bfloat16) + TILE_M * D * sizeof(float);

    dim3 block(NUM_THREADS);

    // Kernel 1: dQ - each block handles one (bh, q_tile), sweeps all k_tiles
    {
        int64_t n_qtiles = (S + TILE_M - 1) / TILE_M;
        dim3 grid(n_qtiles, 1, B * H);
        mha_bwd_dq_kernel<128><<<grid, block, smem_bytes, stream>>>(
            static_cast<const __nv_bfloat16*>(Q.data_ptr()),
            static_cast<const __nv_bfloat16*>(K.data_ptr()),
            static_cast<const __nv_bfloat16*>(V.data_ptr()),
            static_cast<const __nv_bfloat16*>(O_fwd.data_ptr()),
            static_cast<const __nv_bfloat16*>(dO_in.data_ptr()),
            static_cast<const float*>(L.data_ptr()),
            static_cast<__nv_bfloat16*>(dQ_out.data_ptr()),
            B, H, S, inv_sqrt_d
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Kernel 2: dK - each block handles one (bh, k_tile), sweeps all q_tiles
    {
        int64_t n_ktiles = (S + TILE_N - 1) / TILE_N;
        dim3 grid(n_ktiles, 1, B * H);
        mha_bwd_dk_kernel<128><<<grid, block, smem_bytes, stream>>>(
            static_cast<const __nv_bfloat16*>(Q.data_ptr()),
            static_cast<const __nv_bfloat16*>(K.data_ptr()),
            static_cast<const __nv_bfloat16*>(V.data_ptr()),
            static_cast<const __nv_bfloat16*>(O_fwd.data_ptr()),
            static_cast<const __nv_bfloat16*>(dO_in.data_ptr()),
            static_cast<const float*>(L.data_ptr()),
            static_cast<__nv_bfloat16*>(dK_out.data_ptr()),
            B, H, S, inv_sqrt_d
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Kernel 3: dV - each block handles one (bh, k_tile), sweeps all q_tiles
    {
        int64_t n_ktiles = (S + TILE_N - 1) / TILE_N;
        dim3 grid(n_ktiles, 1, B * H);
        mha_bwd_dv_kernel<128><<<grid, block, smem_bytes, stream>>>(
            static_cast<const __nv_bfloat16*>(Q.data_ptr()),
            static_cast<const __nv_bfloat16*>(K.data_ptr()),
            static_cast<const __nv_bfloat16*>(V.data_ptr()),
            static_cast<const __nv_bfloat16*>(O_fwd.data_ptr()),
            static_cast<const __nv_bfloat16*>(dO_in.data_ptr()),
            static_cast<const float*>(L.data_ptr()),
            static_cast<__nv_bfloat16*>(dV_out.data_ptr()),
            B, H, S, inv_sqrt_d
        );
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);

}  // namespace mha_bwd_impl