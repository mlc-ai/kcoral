#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <cfloat>
#include <cmath>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_d128_causal {

constexpr int BM = 32;
constexpr int BN = 32;
constexpr int D = 128;
constexpr int NT = 128;

// Shared mem layout (in bf16 words):
//   0..BM*D-1      : sQ[BM][D]
//   BM*D..BM*D+BN*D-1 : sK[BN][D]
//   ...+BN*D..+2*BN*D-1 : sV[BN][D]
// Total: (BM + 2*BN)*D bf16 = (32+64)*128 = 12288 bf16 = 24.5KB
constexpr int SHARED_BF16_WORDS = (BM + 2*BN) * D;

__global__ void fa_fwd_kernel(
    const __nv_bfloat16* __restrict__ Q_g,
    const __nv_bfloat16* __restrict__ K_g,
    const __nv_bfloat16* __restrict__ V_g,
    __nv_bfloat16* __restrict__ O_g,
    float* __restrict__ LSE_g,
    int S, int B, int H)
{
    extern __shared__ __nv_bfloat16 smem[];

    __nv_bfloat16* sQ = smem;
    __nv_bfloat16* sK = smem + BM * D;
    __nv_bfloat16* sV = smem + BM * D + BN * D;

    int bid = blockIdx.x;
    int num_q_tiles = (S + BM - 1) / BM;
    int q_tile = bid % num_q_tiles;
    int bh     = bid / num_q_tiles;
    int batch  = bh / H;
    int head   = bh % H;

    int q_start = q_tile * BM;
    int tid     = threadIdx.x;

    // Per-thread registers: one row of output accumulator
    float oacc[D];
    #pragma unroll
    for (int d = 0; d < D; ++d) oacc[d] = 0.0f;

    size_t base_off = (size_t)batch * H * S * D + head * S * D;
    
    const __nv_bfloat16* Qbase = Q_g + base_off;
    const __nv_bfloat16* Kbase = K_g + base_off;
    const __nv_bfloat16* Vbase = V_g + base_off;
    __nv_bfloat16*         Obse = O_g + base_off;
    float*                 LSE  = LSE_g + (size_t)batch * H * S + head * S;

    float inv_sqrt_d = rsqrtf((float)D);
    int num_k_steps  = (S + BN - 1) / BN;

    // Load Q tile cooperatively: sQ[row*D + col]
    for (int col = tid; col < D; col += NT) {
        for (int row = 0; row < BM; ++row) {
            int q_abs = q_start + row;
            sQ[row * D + col] = (q_abs < S) ? Qbase[(size_t)q_abs * D + col] : __float2bfloat16(0.0f);
        }
    }
    __syncthreads();

    // Each thread handles one query row
    for (int ri = tid; ri < BM; ri += NT) {
        int q_row = ri;
        int q_abs = q_start + q_row;

        // Reset this thread's accumulator (redundant for first call but safe)
        #pragma unroll
        for (int d = 0; d < D; ++d) oacc[d] = 0.0f;

        float cur_max = -FLT_MAX;
        float cur_sum = 0.0f;

        float scores[BN];

        for (int ks = 0; ks < num_k_steps; ++ks) {
            int k_start = ks * BN;

            // Cooperatively load K and V tiles
            if (tid < BN) {
                int kr = tid;
                int k_abs = k_start + kr;
                uint2* sk_ptr = (uint2*)(sK + kr * D);
                uint2* sv_ptr = (uint2*)(sV + kr * D);
                for (int d = 0; d < D; d += 8) {
                    if (k_abs < S) {
                        sk_ptr[d/8] = ((const uint2*)Kbase)[(size_t)k_abs * D / 8 + d / 8];
                        sv_ptr[d/8] = ((const uint2*)Vbase)[(size_t)k_abs * D / 8 + d / 8];
                    } else {
                        sk_ptr[d/8] = make_uint2(0, 0);
                        sv_ptr[d/8] = make_uint2(0, 0);
                    }
                }
            }
            __syncthreads();

            // Compute attention scores
            for (int k = 0; k < BN; ++k) {
                float s = 0.0f;
                #pragma unroll
                for (int d = 0; d < D; d += 4) {
                    float qv = __bfloat162float(sQ[q_row * D + d]);
                    float kv = __bfloat162float(sK[k * D + d]);
                    float qv1 = __bfloat162float(sQ[q_row * D + d + 1]);
                    float kv1 = __bfloat162float(sK[k * D + d + 1]);
                    float qv2 = __bfloat162float(sQ[q_row * D + d + 2]);
                    float kv2 = __bfloat162float(sK[k * D + d + 2]);
                    float qv3 = __bfloat162float(sQ[q_row * D + d + 3]);
                    float kv3 = __bfloat162float(sK[k * D + d + 3]);
                    s += qv*kv + qv1*kv1 + qv2*kv2 + qv3*kv3;
                }
                s *= inv_sqrt_d;

                int k_abs_local = k_start + k;
                scores[k] = ((q_abs < S) && (k_abs_local <= q_abs)) ? s : -FLT_MAX;
            }

            // Find new max
            float new_max = -FLT_MAX;
            for (int k = 0; k < BN; ++k) {
                if (scores[k] > new_max) new_max = scores[k];
            }

            // Rescale if max increased
            if (cur_max > -FLT_MAX && new_max > cur_max) {
                float alpha = expf(cur_max - new_max);
                #pragma unroll
                for (int d = 0; d < D; ++d) {
                    oacc[d] *= alpha;
                }
                cur_sum *= alpha;
            }
            cur_max = new_max;

            // Accumulate weighted V
            for (int k = 0; k < BN; ++k) {
                float sc = scores[k];
                if (sc == -FLT_MAX || cur_max == -FLT_MAX) continue;

                float w = expf(sc - cur_max);
                cur_sum += w;

                #pragma unroll
                for (int d = 0; d < D; ++d) {
                    oacc[d] += w * __bfloat162float(sV[k * D + d]);
                }
            }

            __syncthreads();
        }

        // Epilogue
        if (q_abs < S && cur_sum > 0.0f) {
            float inv_l = 1.0f / cur_sum;
            LSE[q_abs] = cur_max + logf(cur_sum);
            for (int d = 0; d < D; ++d) {
                Obse[(size_t)q_abs * D + d] = __float2bfloat16(oacc[d] * inv_l);
            }
        } else if (q_abs < S) {
            LSE[q_abs] = -FLT_MAX;
            for (int d = 0; d < D; ++d) {
                Obse[(size_t)q_abs * D + d] = __float2bfloat16(0.0f);
            }
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);

    int num_q_tiles = (int)((S + BM - 1) / BM);
    int total_blocks = (int)(B * H * num_q_tiles);

    dim3 grid(total_blocks);
    dim3 block(NT);

    size_t smem_size = SHARED_BF16_WORDS * sizeof(__nv_bfloat16);

    cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id);

    fa_fwd_kernel<<<grid, block, smem_size, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<__nv_bfloat16*>(O.data_ptr()),
        static_cast<float*>(LSE.data_ptr()),
        (int)S, (int)B, (int)H);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_d128_causal::run);

}  // namespace mha_d128_causal