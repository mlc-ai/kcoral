#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t _e = (call); \
        if (_e != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
            exit(1); \
        } \
    } while(0)

namespace mha_bwd_impl {

// Tile sizes and thread configuration
// TILE_M = query tile height, TILE_N = KV tile height
constexpr int TILE_M = 32;
constexpr int TILE_N = 32;
constexpr int TPB = 128;

__device__ __forceinline__ float bf162f32(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

__device__ __forceinline__ __nv_bfloat16 f322bf16(float v) {
    return __float2bfloat16(v);
}

__global__ void mha_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K_g,
    const __nv_bfloat16* __restrict__ V_g,
    const __nv_bfloat16* __restrict__ O_g,
    const __nv_bfloat16* __restrict__ dO_g,
    const float* __restrict__ LSE_g,
    __nv_bfloat16* __restrict__ dQ_out,
    __nv_bfloat16* __restrict__ dK_out,
    __nv_bfloat16* __restrict__ dV_out,
    int B, int H, int S, int d,
    float scale) {

    // One block per (batch, head) pair
    int bid = blockIdx.x;
    int b = bid / H;
    int h = bid % H;
    if (b >= B || h >= H) return;

    int tid = threadIdx.x;

    uint64_t bh_off = ((uint64_t)b * H + h) * (uint64_t)S * d;
    const __nv_bfloat16* Q_bh = Q + bh_off;
    const __nv_bfloat16* K_bh = K_g + bh_off;
    const __nv_bfloat16* V_bh = V_g + bh_off;
    const __nv_bfloat16* O_bh = O_g + bh_off;
    const __nv_bfloat16* dO_bh = dO_g + bh_off;
    const float* LSE_bh = LSE_g + (b * H + h) * S;
    __nv_bfloat16* dQ_bh = dQ_out + bh_off;
    __nv_bfloat16* dK_bh = dK_out + bh_off;
    __nv_bfloat16* dV_bh = dV_out + bh_off;

    int n_qtiles = (S + TILE_M - 1) / TILE_M;
    int n_kvtiles = (S + TILE_N - 1) / TILE_N;

    // =========================================================
    // Dynamic shared memory layout:
    // sQ      [TILE_M][d]    bf16   offset 0
    // sK      [TILE_N][d]    bf16   offset TILE_M*d*2
    // sV      [TILE_N][d]    bf16   offset (TILE_M+TILE_N)*d*2
    // sdO     [TILE_M][d]    bf16   offset (TILE_M+2*TILE_N)*d*2
    // sLSE    [TILE_M]       fp32   after sdO
    // sD      [TILE_M]       fp32   after sLSE
    // sP      [TILE_M][TILE_N] fp32 attention probabilities
    // sdS     [TILE_M][TILE_N] fp32 score gradients
    // sDQ     [TILE_M][d]    fp32   accumulator (Phase 1)
    // sDK     (alias sDQ)    fp32   accumulator (Phase 2, reused)
    // sDV     [TILE_N][d]    fp32   accumulator (Phase 2)
    // =========================================================
    extern __shared__ char smem[];

    __nv_bfloat16* sQ  = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sK  = sQ + TILE_M * d;
    __nv_bfloat16* sV  = sK + TILE_N * d;
    __nv_bfloat16* sdO = sV + TILE_N * d;
    float* sLSE        = reinterpret_cast<float*>(sdO + TILE_M * d);
    float* sD          = sLSE + TILE_M;
    float* sP          = sD + TILE_M;
    float* sdS         = sP + TILE_M * TILE_N;
    float* sDQ         = sdS + TILE_M * TILE_N;
    float* sDK         = sDQ;               // reused alias in Phase 2
    float* sDV         = sDK + TILE_N * d;

    // ---------------------------------------------------------
    // Initialize dQ, dK, dV to zero
    // ---------------------------------------------------------
    for (int idx = tid; idx < S * d; idx += TPB) {
        dQ_bh[idx] = f322bf16(0.0f);
        dK_bh[idx] = f322bf16(0.0f);
        dV_bh[idx] = f322bf16(0.0f);
    }
    __syncthreads();

    // ===================================================================
    // PHASE 1: Compute dQ
    // dQ[i] = sum_j dS[i][j] * K[j]
    // Strategy: query-tile outer loop, accumulate dQ across all KV tiles
    // ===================================================================
    for (int tq = 0; tq < n_qtiles; tq++) {
        int qs = tq * TILE_M;
        int qe = (qs + TILE_M < S) ? qs + TILE_M : S;
        int qh = qe - qs;

        // --- Load Q and dO tiles ---
        for (int idx = tid; idx < TILE_M * d; idx += TPB) {
            int r = idx / d;
            int c = idx % d;
            if (r < qh) {
                sQ[idx]  = Q_bh[(uint64_t)(qs + r) * d + c];
                sdO[idx] = dO_bh[(uint64_t)(qs + r) * d + c];
            } else {
                sQ[idx]  = f322bf16(0.0f);
                sdO[idx] = f322bf16(0.0f);
            }
        }
        __syncthreads();

        // --- Load LSE and compute D[i] = dO[i] · O[i] ---
        for (int i = tid; i < TILE_M; i += TPB) {
            if (i < qh) {
                sLSE[i] = LSE_bh[qs + i];
                float D_val = 0.0f;
                for (int c = 0; c < d; c++) {
                    D_val += bf162f32(sdO[i * d + c]) * bf162f32(O_bh[(uint64_t)(qs + i) * d + c]);
                }
                sD[i] = D_val;
            } else {
                sLSE[i] = 0.0f;
                sD[i]   = 0.0f;
            }
        }
        __syncthreads();

        // --- Clear dQ accumulator for this query tile ---
        for (int idx = tid; idx < TILE_M * d; idx += TPB) {
            sDQ[idx] = 0.0f;
        }
        __syncthreads();

        // --- Iterate over all KV tiles ---
        for (int tk = 0; tk < n_kvtiles; tk++) {
            int ks = tk * TILE_N;
            int ke = (ks + TILE_N < S) ? ks + TILE_N : S;
            int kh = ke - ks;

            // Load K and V tiles
            for (int idx = tid; idx < TILE_N * d; idx += TPB) {
                int r = idx / d;
                int c = idx % d;
                if (r < kh) {
                    sK[idx] = K_bh[(uint64_t)(ks + r) * d + c];
                    sV[idx] = V_bh[(uint64_t)(ks + r) * d + c];
                } else {
                    sK[idx] = f322bf16(0.0f);
                    sV[idx] = f322bf16(0.0f);
                }
            }
            __syncthreads();

            // Compute attention probabilities P[i][j] = exp(Q[i]·K[j]*scale - LSE[i])
            for (int ij = tid; ij < TILE_M * TILE_N; ij += TPB) {
                int i = ij / TILE_N;
                int j = ij % TILE_N;
                float S_val = 0.0f;
                for (int c = 0; c < d; c++) {
                    S_val += bf162f32(sQ[i * d + c]) * bf162f32(sK[j * d + c]);
                }
                sP[ij] = expf(S_val * scale - sLSE[i]);
            }
            __syncthreads();

            // Compute dOV[i][j] = dO[i]·V[j] and dS[i][j] = P[i][j] * (dOV - D[i])
            for (int ij = tid; ij < TILE_M * TILE_N; ij += TPB) {
                int i = ij / TILE_N;
                int j = ij % TILE_N;
                float dOV = 0.0f;
                for (int c = 0; c < d; c++) {
                    dOV += bf162f32(sdO[i * d + c]) * bf162f32(sV[j * d + c]);
                }
                sdS[ij] = sP[ij] * (dOV - sD[i]);
            }
            __syncthreads();

            // Accumulate dQ[i][c] += sum_j dS[i][j] * K[j][c]
            for (int ic = tid; ic < TILE_M * d; ic += TPB) {
                int i = ic / d;
                int c = ic % d;
                float val = 0.0f;
                for (int j = 0; j < TILE_N; j++) {
                    val += sdS[i * TILE_N + j] * bf162f32(sK[j * d + c]);
                }
                sDQ[ic] += val;
            }
            __syncthreads();
        }

        // --- Write dQ for this query tile to global memory ---
        for (int idx = tid; idx < qh * d; idx += TPB) {
            int r = idx / d;
            int c = idx % d;
            dQ_bh[(uint64_t)(qs + r) * d + c] = f322bf16(sDQ[idx]);
        }
        __syncthreads();
    }

    // ===================================================================
    // PHASE 2: Compute dK and dV
    // dK[j] = sum_i dS[i][j] * Q[i]
    // dV[j] = sum_i P[i][j] * dO[i]
    // Strategy: KV-tile outer loop, accumulate across all query tiles
    // ===================================================================
    for (int tk = 0; tk < n_kvtiles; tk++) {
        int ks = tk * TILE_N;
        int ke = (ks + TILE_N < S) ? ks + TILE_N : S;
        int kh = ke - ks;

        // Load K and V tiles (held constant for all query tiles in this iteration)
        for (int idx = tid; idx < TILE_N * d; idx += TPB) {
            int r = idx / d;
            int c = idx % d;
            if (r < kh) {
                sK[idx] = K_bh[(uint64_t)(ks + r) * d + c];
                sV[idx] = V_bh[(uint64_t)(ks + r) * d + c];
            } else {
                sK[idx] = f322bf16(0.0f);
                sV[idx] = f322bf16(0.0f);
            }
        }
        __syncthreads();

        // Clear dK and dV accumulators for this KV tile
        for (int idx = tid; idx < TILE_N * d; idx += TPB) {
            sDK[idx] = 0.0f;
            sDV[idx] = 0.0f;
        }
        __syncthreads();

        for (int tq = 0; tq < n_qtiles; tq++) {
            int qs = tq * TILE_M;
            int qe = (qs + TILE_M < S) ? qs + TILE_M : S;
            int qh = qe - qs;

            // Load Q and dO tiles
            for (int idx = tid; idx < TILE_M * d; idx += TPB) {
                int r = idx / d;
                int c = idx % d;
                if (r < qh) {
                    sQ[idx]  = Q_bh[(uint64_t)(qs + r) * d + c];
                    sdO[idx] = dO_bh[(uint64_t)(qs + r) * d + c];
                } else {
                    sQ[idx]  = f322bf16(0.0f);
                    sdO[idx] = f322bf16(0.0f);
                }
            }
            __syncthreads();

            // Load LSE and compute D[i] = dO[i] · O[i]
            for (int i = tid; i < TILE_M; i += TPB) {
                if (i < qh) {
                    sLSE[i] = LSE_bh[qs + i];
                    float D_val = 0.0f;
                    for (int c = 0; c < d; c++) {
                        D_val += bf162f32(sdO[i * d + c]) * bf162f32(O_bh[(uint64_t)(qs + i) * d + c]);
                    }
                    sD[i] = D_val;
                } else {
                    sLSE[i] = 0.0f;
                    sD[i]   = 0.0f;
                }
            }
            __syncthreads();

            // Compute P[i][j] = exp(Q[i]·K[j]*scale - LSE[i])
            for (int ij = tid; ij < TILE_M * TILE_N; ij += TPB) {
                int i = ij / TILE_N;
                int j = ij % TILE_N;
                float S_val = 0.0f;
                for (int c = 0; c < d; c++) {
                    S_val += bf162f32(sQ[i * d + c]) * bf162f32(sK[j * d + c]);
                }
                sP[ij] = expf(S_val * scale - sLSE[i]);
            }
            __syncthreads();

            // Compute dS[i][j] = P[i][j] * (dO[i]·V[j] - D[i])
            for (int ij = tid; ij < TILE_M * TILE_N; ij += TPB) {
                int i = ij / TILE_N;
                int j = ij % TILE_N;
                float dOV = 0.0f;
                for (int c = 0; c < d; c++) {
                    dOV += bf162f32(sdO[i * d + c]) * bf162f32(sV[j * d + c]);
                }
                sdS[ij] = sP[ij] * (dOV - sD[i]);
            }
            __syncthreads();

            // Accumulate dK[j][c] += sum_i dS[i][j] * Q[i][c]
            for (int jc = tid; jc < TILE_N * d; jc += TPB) {
                int j = jc / d;
                int c = jc % d;
                float val = 0.0f;
                for (int i = 0; i < TILE_M; i++) {
                    val += sdS[i * TILE_N + j] * bf162f32(sQ[i * d + c]);
                }
                sDK[jc] += val;
            }
            __syncthreads();

            // Accumulate dV[j][c] += sum_i P[i][j] * dO[i][c]
            for (int jc = tid; jc < TILE_N * d; jc += TPB) {
                int j = jc / d;
                int c = jc % d;
                float val = 0.0f;
                for (int i = 0; i < TILE_M; i++) {
                    val += sP[i * TILE_N + j] * bf162f32(sdO[i * d + c]);
                }
                sDV[jc] += val;
            }
            __syncthreads();
        }

        // Write dK and dV for this KV tile to global memory
        for (int idx = tid; idx < kh * d; idx += TPB) {
            int r = idx / d;
            int c = idx % d;
            dK_bh[(uint64_t)(ks + r) * d + c] = f322bf16(sDK[idx]);
            dV_bh[(uint64_t)(ks + r) * d + c] = f322bf16(sDV[idx]);
        }
        __syncthreads();
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S_val = Q.size(2);
    int64_t d = Q.size(3);

    int num_bh = static_cast<int>(B * H);
    dim3 grid(num_bh);
    dim3 block(TPB);

    // Compute dynamic shared memory size in bytes
    // BF16 buffers: sQ[TILE_M][d], sK[TILE_N][d], sV[TILE_N][d], sdO[TILE_M][d]
    size_t smem_bytes = (size_t)TILE_M * d * sizeof(__nv_bfloat16);  // sQ
    smem_bytes += (size_t)TILE_N * d * sizeof(__nv_bfloat16);        // sK
    smem_bytes += (size_t)TILE_N * d * sizeof(__nv_bfloat16);        // sV
    smem_bytes += (size_t)TILE_M * d * sizeof(__nv_bfloat16);        // sdO
    smem_bytes += (size_t)TILE_M * sizeof(float);                    // sLSE
    smem_bytes += (size_t)TILE_M * sizeof(float);                    // sD
    smem_bytes += (size_t)TILE_M * TILE_N * sizeof(float);           // sP
    smem_bytes += (size_t)TILE_M * TILE_N * sizeof(float);           // sdS
    smem_bytes += (size_t)TILE_M * d * sizeof(float);                // sDQ (Phase 1) / sDK (Phase 2 alias)
    smem_bytes += (size_t)TILE_N * d * sizeof(float);                // sDV

    float attn_scale = 1.0f / std::sqrt(static_cast<float>(d));

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    // Optionally raise dynamic SMEM limit if needed (for configs with <~80KB default)
    int smem_avail = 0;
    cudaDeviceGetAttribute(&smem_avail, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    int smem_optin = 0;
    cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    if (smem_bytes > static_cast<size_t>(smem_avail) && smem_bytes <= static_cast<size_t>(smem_optin)) {
        cudaFuncSetAttribute(reinterpret_cast<cudaFunction_t>(&mha_bwd_kernel),
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_bytes));
    }

    mha_bwd_kernel<<<grid, block, smem_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(O.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        static_cast<int>(B), static_cast<int>(H), static_cast<int>(S_val), static_cast<int>(d),
        attn_scale
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);

}  // namespace mha_bwd_impl