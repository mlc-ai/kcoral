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
static constexpr int NUM_WARPS = NT / 32;  // = 4

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
    // sQ[HEAD_D]            -> 128 bf16       =  256 bytes
    // sK[TILE_N*HEAD_D]     -> 64*128 bf16    = 16 KiB
    // sV[TILE_N*HEAD_D]     -> 64*128 bf16    = 16 KiB  
    // sScores[TILE_N]       -> 64 fp32        =  256 bytes
    // sWarpPartial[NUM_WARPS] -> 4 fp32       =   16 bytes
    // Total ~= 33 KiB
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sK = sQ + HEAD_D;
    __nv_bfloat16* sV = sK + TILE_N * HEAD_D;
    float*         sScores = reinterpret_cast<float*>(
            reinterpret_cast<char*>(sV) + TILE_N * HEAD_D * sizeof(__nv_bfloat16));
    float*         sWarpPartial = sScores + TILE_N;

    int bh_idx = blockIdx.x;
    int tid    = threadIdx.x;
    int lane   = tid % 32;
    int warp   = tid / 32;

    if (bh_idx >= B * H) return;

    int b = bh_idx / H;
    int h = bh_idx % H;

    int64_t bh_off = static_cast<int64_t>(b * H + h) * S * HEAD_D;
    const __nv_bfloat16* K_base = K_g + bh_off;
    const __nv_bfloat16* V_base = V_g + bh_off;
    __nv_bfloat16*         O_base = O_g + bh_off;
    float*               LSE_base = LSE_g + static_cast<int64_t>(b * H + h) * S;

    // Thread owns dimension my_d = tid (since HEAD_D == NT == 128)
    int my_d = tid;

    for (int qi = 0; qi < S; ++qi) {
        // --- Load ONE Q row into sQ cooperatively ---
        for (int d = tid; d < HEAD_D; d += NT)
            sQ[d] = Q_g[bh_off + static_cast<int64_t>(qi) * HEAD_D + d];
        __syncthreads();

        // Per-query-row online softmax state
        float m_i = -FLT_MAX;
        float li_i = 0.0f;
        float acc = 0.0f;  // accumulator for this thread's dimension

        float q_val = __bfloat162float(sQ[my_d]);

        int nk_end = (S + TILE_N - 1) / TILE_N;

        for (int tile_k = 0; tile_k < nk_end; ++tile_k) {
            int nk = tile_k * TILE_N;
            int act_n = min(TILE_N, S - nk);
            int64_t nk_off = static_cast<int64_t>(nk) * HEAD_D;

            // --- Load K tile cooperatively ---
            for (int i = tid; i < act_n * HEAD_D; i += NT)
                sK[i] = K_base[nk_off + i];
            __syncthreads();

            // ---- Compute dot products with cross-warp reduction ----
            // Each thread: compute q_val * k[j][my_d], reduce within warp,
            // write to sWarpPartial, then combine 4 warp partials
            #pragma unroll
            for (int j = 0; j < act_n; ++j) {
                // Partial contribution from this thread
                float partial = q_val * __bfloat162float(sK[j * HEAD_D + my_d]);

                // Reduce within own warp of 32 lanes
                for (int offset = 16; offset > 0; offset /= 2) {
                    partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
                }

                // Write warp-leader partial to shared memory
                if (lane == 0) {
                    sWarpPartial[j + warp * act_n] = partial;
                }
            }
            __syncthreads();

            // Now warp 0 reads the 4 partials and combines them
            if (warp == 0) {
                #pragma unroll
                for (int j = 0; j < act_n; ++j) {
                    float total = sWarpPartial[j + 0 * act_n]
                                + sWarpPartial[j + 1 * act_n]
                                + sWarpPartial[j + 2 * act_n]
                                + sWarpPartial[j + 3 * act_n];
                    
                    // If lane matches a valid KV row index, store the score
                    if (lane < act_n) {
                        sScores[lane] = total * attn_scale;
                    }
                }
            }
            __syncthreads();

            // --- Find new maximum across all KV scores ---
            float m_new = m_i;
            #pragma unroll
            for (int j = 0; j < act_n; ++j) {
                if (sScores[j] > m_new) m_new = sScores[j];
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
                acc *= alpha;
                li_i *= alpha;
            } else {
                acc = 0.0f;
                li_i = 0.0f;
            }

            // Compute probabilities & accumulate output for my dimension
            float p_sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < act_n; ++j) {
                float diff = sScores[j] - m_new;
                if (diff > -100.0f) {
                    float p = expf(diff);
                    p_sum += p;
                    acc += p * __bfloat162float(sV[j * HEAD_D + my_d]);
                }
            }

            li_i += p_sum;
            m_i = m_new;

            __syncthreads();
        }

        // Write back normalized result
        float inv_li = (li_i > 1e-30f) ? (1.0f / li_i) : 0.0f;
        O_base[static_cast<int64_t>(qi) * HEAD_D + my_d] =
            __float2bfloat16(acc * inv_li);

        // Write LSE (only thread 0)
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

    dim3 grid(static_cast<int>(B * H));
    dim3 block(NT);

    // Shared memory layout sizes
    size_t smem_bytes = static_cast<size_t>(HEAD_D + TILE_N * HEAD_D * 2) 
                            * sizeof(__nv_bfloat16)
                       + TILE_N * sizeof(float)           // sScores
                       + TILE_N * NUM_WARPS * sizeof(float); // sWarpPartial

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