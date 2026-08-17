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

static constexpr int TILE_M  = 64;   // Q rows per block
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

    // Shared memory layout:
    // sQ[TILE_M*HEAD_D]   -> 64*128 bf16 = 16 KiB
    // sK[TILE_N*HEAD_D]   -> 64*128 bf16 = 16 KiB  
    // sV[TILE_N*HEAD_D]   -> 64*128 bf16 = 16 KiB
    // sWarpPartial[TILE_N*NUM_WARPS] -> 64*4 fp32 = 1 KiB
    // Total ~= 49 KiB
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sK = sQ + TILE_M * HEAD_D;
    __nv_bfloat16* sV = sK + TILE_N * HEAD_D;
    float*         sWarpPartial = reinterpret_cast<float*>(
            reinterpret_cast<char*>(sV) + TILE_N * HEAD_D * sizeof(__nv_bfloat16));

    int bh_idx  = blockIdx.x;       // (batch, head) packed
    int qtile   = blockIdx.y;       // Q tile index along sequence axis

    if (bh_idx >= B * H || qtile * TILE_M >= S) return;

    int b   = bh_idx / H;
    int h   = bh_idx % H;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp = tid / 32;

    int64_t bh_off = static_cast<int64_t>(b * H + h) * S * HEAD_D;
    int64_t q_off  = static_cast<int64_t>(qtile)     * TILE_M * HEAD_D;
    
    int act_m = min(TILE_M, S - qtile * TILE_M);

    const __nv_bfloat16* Q_base = Q_g + bh_off + q_off;
    const __nv_bfloat16* K_base = K_g + bh_off;
    const __nv_bfloat16* V_base = V_g + bh_off;
    __nv_bfloat16*         O_base = O_g + bh_off + q_off;
    float*               LSE_base = LSE_g + static_cast<int64_t>(b * H + h) * S + qtile * TILE_M;

    // --- Load Q tile cooperatively ---
    for (int i = tid; i < act_m * HEAD_D; i += NT)
        sQ[i] = Q_base[i];
    __syncthreads();

    // Only first 64 threads are "active" for softmax+accumulation
    // Each active thread handles 1 Q row and 2 output dimensions (to cover D=128)
    int D_per_thread = HEAD_D / TILE_M;  // 128 / 64 = 2

    if (tid < act_m) {
        // Per-row online softmax state
        float m_i  = -FLT_MAX;
        float li_i = 0.0f;
        
        // Output accumulator: 2 FP32 values for the two D-dimensions owned
        float acc[D_per_thread] = {};

        int nk_end = (S + TILE_N - 1) / TILE_N;

        for (int tile_k = 0; tile_k < nk_end; ++tile_k) {
            int nk = tile_k * TILE_N;
            int act_n = min(TILE_N, S - nk);
            int64_t nk_off = static_cast<int64_t>(nk) * HEAD_D;

            // --- Load K tile cooperatively (all threads participate) ---
            for (int i = tid; i < act_n * HEAD_D; i += NT)
                sK[i] = K_base[nk_off + i];
            __syncthreads();

            // --- Compute Q·K^T scores for this row ---
            // Active thread tid owns 1 Q row, must reduce over all HEAD_D elements
            // Strategy: each of NT threads contributes partial (reading sQ[tid*D+col]),
            // reduce across warps via sWarpPartial
            
            #pragma unroll
            for (int j = 0; j < act_n; ++j) {
                float partial = 0.0f;
                // Each thread contributes its slice of HEAD_D dimensions
                for (int d = tid; d < HEAD_D; d += NT) {
                    partial += __bfloat162float(sQ[tid * HEAD_D + d])
                             * __bfloat162float(sK[j * HEAD_D + d]);
                }
                
                // Warp-level reduction
                for (int offset = 16; offset > 0; offset /= 2) {
                    partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
                }
                
                // Write warp partial
                if (lane == 0) {
                    sWarpPartial[j + warp * act_n] = partial;
                }
            }
            __syncthreads();

            // Warp 0 combines the 4 warp partials
            if (warp == 0) {
                #pragma unroll
                for (int j = 0; j < act_n; ++j) {
                    float total = sWarpPartial[j + 0 * act_n]
                                + sWarpPartial[j + 1 * act_n]
                                + sWarpPartial[j + 2 * act_n]
                                + sWarpPartial[j + 3 * act_n];
                    
                    if (lane < act_n) {
                        sWarpPartial[lane] = total * attn_scale;
                    }
                }
            }
            __syncthreads();

            // Read combined scores and find new max
            float m_new = m_i;
            #pragma unroll
            for (int j = 0; j < act_n; ++j) {
                float sj = sWarpPartial[j];
                if (sj > m_new) m_new = sj;
            }

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
                for (int e = 0; e < D_per_thread; ++e) acc[e] *= alpha;
                li_i *= alpha;
            } else {
                #pragma unroll
                for (int e = 0; e < D_per_thread; ++e) acc[e] = 0.0f;
                li_i = 0.0f;
            }

            // Compute probabilities & accumulate for this thread's D dimensions
            float p_sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < act_n; ++j) {
                float diff = sWarpPartial[j] - m_new;
                if (diff > -100.0f) {
                    float p = expf(diff);
                    p_sum += p;
                    
                    #pragma unroll
                    for (int e = 0; e < D_per_thread; ++e) {
                        int d = tid * D_per_thread + e;
                        acc[e] += p * __bfloat162float(sV[j * HEAD_D + d]);
                    }
                }
            }

            li_i += p_sum;
            m_i = m_new;

            __syncthreads();
        }

        // Normalize and write output
        float inv_li = (li_i > 1e-30f) ? (1.0f / li_i) : 0.0f;
        #pragma unroll
        for (int e = 0; e < D_per_thread; ++e) {
            int d = tid * D_per_thread + e;
            O_base[tid * HEAD_D + d] = __float2bfloat16(acc[e] * inv_li);
        }

        // Write LSE (only thread 0 per block)
        if (tid == 0) {
            for (int r = 0; r < act_m; ++r) {
                // Need each thread's li/m — this requires per-row storage in smem
                // Simplest: thread 0 waits and reads from other threads
                // For now, each thread writes its own LSE position
            }
        }
        
        if (tid < act_m) {
            LSE_base[tid] = (li_i > 1e-30f) ? (m_i + logf(li_i)) : (-FLT_MAX);
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

    int num_bh     = static_cast<int>(B * H);
    int num_qtiles = static_cast<int>((S + TILE_M - 1) / TILE_M);

    dim3 grid(num_bh, num_qtiles);
    dim3 block(NT);

    // Shared memory: sQ(64*128) + sK(64*128) + sV(64*128) bf16 + sWarpPartial(64*4) fp32
    size_t smem_bytes = static_cast<size_t>(TILE_M * HEAD_D + TILE_N * HEAD_D * 2) 
                            * sizeof(__nv_bfloat16)
                       + TILE_N * (NT / 32) * sizeof(float);

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