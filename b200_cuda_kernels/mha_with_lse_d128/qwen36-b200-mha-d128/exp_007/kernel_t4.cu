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
    extern __shared__ __align__(128) char smem_raw[];

    // Shared memory layout (all bf16):
    // sQ[HEAD_D]       -> 128 bf16 = 256 bytes   (one Q row at a time)
    // sK[TILE_N*D]     -> 64*128 bf16 = 16 KiB
    // sV[TILE_N*D]     -> 64*128 bf16 = 16 KiB
    // sScores[TILE_N]  -> 64 fp32   = 256 bytes   (KV scores for softmax)
    // Total ~= 32.7 KiB
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sK = sQ + HEAD_D;
    __nv_bfloat16* sV = sK + TILE_N * HEAD_D;
    float*         sScores = reinterpret_cast<float*>(
            reinterpret_cast<char*>(sV) + TILE_N * HEAD_D * sizeof(__nv_bfloat16));

    int bh_idx = blockIdx.x;
    int tid    = threadIdx.x;

    if (bh_idx >= B * H) return;

    int b = bh_idx / H;
    int h = bh_idx % H;

    int64_t bh_off = static_cast<int64_t>(b * H + h) * S * HEAD_D;
    const __nv_bfloat16* K_base = K_g + bh_off;
    const __nv_bfloat16* V_base = V_g + bh_off;
    __nv_bfloat16*         O_base = O_g + bh_off;
    float*               LSE_base = LSE_g + static_cast<int64_t>(b * H + h) * S;

    // This kernel processes ALL query rows for this (b,h) pair via a loop
    for (int qi = 0; qi < S; ++qi) {
        // --- Cooperative load of ONE Q row into sQ (HEAD_D elements) ---
        for (int d = tid; d < HEAD_D; d += NT)
            sQ[d] = Q_g[bh_off + static_cast<int64_t>(qi) * HEAD_D + d];
        __syncthreads();

        // Per-query-row online softmax state
        float m_i = -FLT_MAX;
        float li_i = 0.0f;

        // Output accumulator stored in shared memory (avoids register blowup)
        // acc[HEAD_D] lives in smem -- but we can't afford another 256B
        // Instead, each thread accumulates a slice of D dimensions
        // Thread t handles dimensions [t*(D/NT) .. (t+1)*(D/NT))
        int elems_per_thread = HEAD_D / NT;  // 128/128 = 1 element per thread!

        float acc_local[elems_per_thread] = {};

        // Iterate over KV tiles
        int nk_end = (S + TILE_N - 1) / TILE_N;
        for (int tile_k = 0; tile_k < nk_end; ++tile_k) {
            int nk = tile_k * TILE_N;
            int act_n = min(TILE_N, S - nk);
            int64_t nk_off = static_cast<int64_t>(nk) * HEAD_D;

            // --- Load K tile cooperatively ---
            for (int i = tid; i < act_n * HEAD_D; i += NT)
                sK[i] = K_base[nk_off + i];
            __syncthreads();

            // Each thread computes its contribution to the dot products
            // Thread tid contributes partial sums for dimensions it owns
            // Each KV row j: partial dot product over dims owned by this thread
            // Then warp-level reduce to get full dot product for each KV row

            // Store partial scores per KV row
            float local_scores[TILE_N] = {};
            #pragma unroll
            for (int j = 0; j < act_n; ++j) {
                float partial = 0.0f;
                #pragma unroll
                for (int e = 0; e < elems_per_thread; ++e) {
                    int d = tid * elems_per_thread + e;
                    partial += __bfloat162float(sQ[d])
                             * __bfloat162float(sK[j * HEAD_D + d]);
                }
                local_scores[j] = partial;
            }

            // Warp-level reduction: sum partial scores across all threads
            // Using shuffle-down reduction within warp of 32
            #pragma unroll
            for (int j = 0; j < act_n; ++j) {
                for (int offset = 16; offset > 0; offset /= 2) {
                    float val = __shfl_down_sync(0xFFFFFFFF, local_scores[j], offset);
                    // Only add if the lane exists in our warp (for last warp if NT%32!=0)
                    if ((tid % 32) < 32 - offset || tid >= NT - offset)
                        local_scores[j] += val;
                }
            }
            // Lane 0 in each warp has the complete score. Broadcast to sScores.
            if ((tid % 32) == 0 && tid < act_n) {
                local_scores[tid] *= attn_scale;
                sScores[tid] = local_scores[tid];
            }
            // All lanes in warp should write for coalescing
            if ((tid % 32) == 0 && tid < act_n) {
                sScores[tid] = local_scores[tid] * attn_scale;
            }
            __syncthreads();

            // Find new maximum across all KV scores in this tile
            float m_new = m_i;
            #pragma unroll
            for (int j = 0; j < act_n; ++j) {
                if (sScores[j] > m_new) m_new = sScores[j];
            }

            // Guard against extreme values
            if (m_new < -1e10f) {
                __syncthreads();
                continue;
            }

            // --- Load V tile cooperatively ---
            for (int i = tid; i < act_n * HEAD_D; i += NT)
                sV[i] = V_base[nk_off + i];
            __syncthreads();

            // Rescale previous accumulator
            float alpha = expf(m_i - m_new);
            bool prev_valid = (m_i > -FLT_MAX / 2.0f);

            if (prev_valid && alpha > 1e-30f) {
                #pragma unroll
                for (int e = 0; e < elems_per_thread; ++e)
                    acc_local[e] *= alpha;
                li_i *= alpha;
            } else {
                #pragma unroll
                for (int e = 0; e < elems_per_thread; ++e)
                    acc_local[e] = 0.0f;
                li_i = 0.0f;
            }

            // Compute probabilities & accumulate output for dims owned by this thread
            float p_sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < act_n; ++j) {
                float diff = sScores[j] - m_new;
                if (diff > -100.0f) {
                    float p = expf(diff);
                    p_sum += p;

                    // Accumulate only the dimension slice this thread owns
                    #pragma unroll
                    for (int e = 0; e < elems_per_thread; ++e) {
                        int d = tid * elems_per_thread + e;
                        acc_local[e] += p * __bfloat162float(sV[j * HEAD_D + d]);
                    }
                }
            }

            li_i += p_sum;
            m_i = m_new;

            __syncthreads();
        }

        // Write back the accumulated result to global memory
        float inv_li = (li_i > 1e-30f) ? (1.0f / li_i) : 0.0f;
        #pragma unroll
        for (int e = 0; e < elems_per_thread; ++e) {
            int d = tid * elems_per_thread + e;
            O_base[static_cast<int64_t>(qi) * HEAD_D + d] =
                __float2bfloat16(acc_local[e] * inv_li);
        }

        // Write LSE (only thread 0 writes to avoid race, since it's per-row scalar)
        if (tid == 0) {
            LSE_base[qi] = (li_i > 1e-30f) ? (m_i + logf(li_i)) : (-FLT_MAX);
        }
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

    // One block per (batch, head). Block loops over all Q rows internally.
    dim3 grid(static_cast<int>(B * H));
    dim3 block(NT);

    // Shared memory: sQ(128) + sK(64*128) + sV(64*128) bf16 + sScores(64) fp32
    size_t smem_bytes = static_cast<size_t>(HEAD_D + TILE_N * HEAD_D * 2) 
                            * sizeof(__nv_bfloat16)
                       + TILE_N * sizeof(float);

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