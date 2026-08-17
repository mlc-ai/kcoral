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

// Warp-level reduce-sum of a single float using shuffle-down
template<int N>
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = N / 2; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xFFFFFFFF, val, offset);
        val += other;
    }
    return val;
}

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
    int lane   = tid % 32;

    if (bh_idx >= B * H) return;

    int b = bh_idx / H;
    int h = bh_idx % H;

    int64_t bh_off = static_cast<int64_t>(b * H + h) * S * HEAD_D;
    const __nv_bfloat16* K_base = K_g + bh_off;
    const __nv_bfloat16* V_base = V_g + bh_off;
    __nv_bfloat16*         O_base = O_g + bh_off;
    float*               LSE_base = LSE_g + static_cast<int64_t>(b * H + h) * S;

    // Each thread owns exactly HEAD_D / NT = 1 element of the output vector
    int my_d = tid;  // since HEAD_D == NT == 128, thread t handles dim t

    // This kernel processes ALL query rows for this (b,h) pair via a loop
    for (int qi = 0; qi < S; ++qi) {
        // --- Cooperative load of ONE Q row into sQ (HEAD_D elements) ---
        for (int d = tid; d < HEAD_D; d += NT)
            sQ[d] = Q_g[bh_off + static_cast<int64_t>(qi) * HEAD_D + d];
        __syncthreads();

        // Per-query-row online softmax state
        float m_i = -FLT_MAX;
        float li_i = 0.0f;

        // Output accumulator for the dimension owned by this thread
        float acc = 0.0f;

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

            // --- Compute dot products: each thread contributes one partial element ---
            // Thread tid computes q[my_d] * k[j][my_d] for all j, then reduce across warp
            // Since my_d == tid, each thread already has its own Q value cached... but let's read from smem
            
            float q_val = __bfloat162float(sQ[my_d]);

            // Local accumulators for reducing dot products
            float local_scores[TILE_N] = {};
            for (int j = 0; j < act_n; ++j) {
                local_scores[j] = q_val * __bfloat162float(sK[j * HEAD_D + my_d]);
            }

            // 4-way warp reduce across the 4 warps (NT=128 threads, 4 warps of 32)
            // Reduce within each warp first, then cross-warp
            #pragma unroll
            for (int j = 0; j < act_n; ++j) {
                float sum = local_scores[j];
                // Reduce within warp of 32
                for (int offset = 16; offset > 0; offset /= 2) {
                    float other = __shfl_down_sync(0xFFFFFFFF, sum, offset);
                    sum += other;
                }
                local_scores[j] = sum;
            }

            // Now lane 0 of each warp has the partial sum for that warp
            // Cross-warp reduce: write partial sums to shared memory, then sum
            if (lane == 0) {
                #pragma unroll
                for (int j = 0; j < act_n; ++j) {
                    sScores[j + tid] = local_scores[j];  // offset by warp base = tid
                }
            }
            __syncthreads();
            
            // No — this approach is getting complicated with partial writes.
            // Let me just have lane-0 of warp-0 collect from all warp lane-0s.
            // Actually, simplest: since we have only 4 warps, let each warp lane-0 
            // write its partial, then one warp reads them all. But we want simplicity.
            
            // Simpler approach: All 128 threads, each contributes to full-dot via shuffles
            // Across all 4 warps simultaneously using __shfl_down_sync on full mask
            // Actually we need to reduce across 128 threads for each score[j].
            
            // Reset and do it right: accumulate across all NT threads using 4-stage reduce
            for (int j = 0; j < act_n; ++j) {
                float sum = q_val * __bfloat162float(sK[j * HEAD_D + my_d]);
                
                // Reduce within own warp (32 lanes)
                for (int offset = 16; offset > 0; offset /= 2) {
                    sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
                }
                
                // Only lane 0 of each warp proceeds
                if (lane == 0) {
                    // Cross-warp: 4 lanes (lane 0 of each of 4 warps)
                    // These are threads 0, 32, 64, 96
                    float cross = sum;
                    if (tid <= 96) {  // warp 0, 1, or 2
                        cross += __shfl_down_sync(0x0000FFFF, sum, 32);
                    }
                    if (tid <= 64) {  // warp 0 or 1
                        cross += __shfl_down_sync(0x0000FFFF, sum, 64);
                    }
                    // thread 0 now has complete sum
                    if (tid == 0) {
                        sScores[j] = cross * attn_scale;
                    }
                }
            }
            __syncthreads();

            // --- Find new maximum across all KV scores in this tile ---
            float m_new = m_i;
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
                acc *= alpha;
                li_i *= alpha;
            } else {
                acc = 0.0f;
                li_i = 0.0f;
            }

            // Compute probabilities & accumulate output for dim owned by this thread
            float p_sum = 0.0f;
            for (int j = 0; j < act_n; ++j) {
                float diff = sScores[j] - m_new;
                if (diff > -100.0f) {
                    float p = expf(diff);
                    p_sum += p;

                    // Accumulate only my dimension
                    acc += p * __bfloat162float(sV[j * HEAD_D + my_d]);
                }
            }

            li_i += p_sum;
            m_i = m_new;

            __syncthreads();
        }

        // Write back the accumulated result to global memory
        float inv_li = (li_i > 1e-30f) ? (1.0f / li_i) : 0.0f;
        O_base[static_cast<int64_t>(qi) * HEAD_D + my_d] =
            __float2bfloat16(acc * inv_li);

        // Write LSE (only thread 0 writes to avoid race)
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