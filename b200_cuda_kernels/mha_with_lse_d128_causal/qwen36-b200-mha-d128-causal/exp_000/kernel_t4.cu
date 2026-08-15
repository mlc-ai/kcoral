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

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int D = 128;
constexpr int NT = 64;

__global__ void fa_fwd_kernel(
    const __nv_bfloat16* __restrict__ Q_g,
    const __nv_bfloat16* __restrict__ K_g,
    const __nv_bfloat16* __restrict__ V_g,
    __nv_bfloat16* __restrict__ O_g,
    float* __restrict__ LSE_g,
    int S, int B, int H)
{
    extern __shared__ char smem[];

    // Layout in shared mem (bytes):
    // sQ[BM][D]     bf16 => 64*128*2 = 16KB @ offset 0
    // sK[BN][D]     bf16 => 64*128*2 = 16KB @ offset 16KB
    // sV[BN][D]     bf16 => 64*128*2 = 16KB @ offset 32KB
    // o_acc[BM][D]  fp32 => 64*128*4 = 32KB @ offset 48KB
    // Total ~= 80 KB per CTA
    
    __nv_bfloat16* sQ   = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sK   = sQ   + BM * D;
    __nv_bfloat16* sV   = sK   + BN * D;
    float*         o_acc= reinterpret_cast<float*>(sV + BN * D);

    int bid = blockIdx.x;
    int num_q_tiles = (S + BM - 1) / BM;
    int q_tile      = bid % num_q_tiles;
    int bh          = bid / num_q_tiles;
    int batch_idx   = bh / H;
    int head_idx    = bh % H;

    int q_start = q_tile * BM;
    int tid     = threadIdx.x;

    size_t base_off = (size_t)batch_idx * H * S * D + head_idx * S * D;
    
    const __nv_bfloat16* Qbase = Q_g + base_off;
    const __nv_bfloat16* Kbase = K_g + base_off;
    const __nv_bfloat16* Vbase = V_g + base_off;
    __nv_bfloat16*         Obse = O_g + base_off;
    float*                 LSE  = LSE_g + (size_t)batch_idx * H * S + head_idx * S;

    const float inv_sqrt_d = rsqrtf((float)D);
    const int num_k_steps  = (S + BN - 1) / BN;

    // Initialize accumulator
    for (int i = tid; i < BM * D; i += NT) {
        o_acc[i] = 0.0f;
    }
    __syncthreads();

    // Load Q tile cooperatively
    for (int d = tid; d < D; d += NT) {
        for (int row = 0; row < BM; ++row) {
            int q_abs = q_start + row;
            sQ[row * D + d] = (q_abs < S) ? Qbase[(size_t)q_abs * D + d] : __float2bfloat16(0.0f);
        }
    }
    __syncthreads();

    // Each thread handles one query row
    int q_row = tid;
    int q_abs = q_start + q_row;

    float cur_max = -FLT_MAX;
    float cur_sum = 0.0f;

    float scores[BN];

    for (int ks = 0; ks < num_k_steps; ++ks) {
        int k_start = ks * BN;

        // Cooperatively load K tile
        for (int d = tid; d < D; d += NT) {
            for (int row = 0; row < BN; ++row) {
                int k_abs = k_start + row;
                sK[row * D + d] = (k_abs < S) ? Kbase[(size_t)k_abs * D + d] : __float2bfloat16(0.0f);
            }
        }
        // Cooperatively load V tile
        for (int d = tid; d < D; d += NT) {
            for (int row = 0; row < BN; ++row) {
                int k_abs = k_start + row;
                sV[row * D + d] = (k_abs < S) ? Vbase[(size_t)k_abs * D + d] : __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        // Compute attention scores S[q_row][k]
        for (int k = 0; k < BN; ++k) {
            float s = 0.0f;
            for (int d = 0; d < D; d += 4) {
                float q0 = __bfloat162float(sQ[q_row * D + d]);
                float q1 = __bfloat162float(sQ[q_row * D + d + 1]);
                float q2 = __bfloat162float(sQ[q_row * D + d + 2]);
                float q3 = __bfloat162float(sQ[q_row * D + d + 3]);

                float k0 = __bfloat162float(sK[k * D + d]);
                float k1 = __bfloat162float(sK[k * D + d + 1]);
                float k2 = __bfloat162float(sK[k * D + d + 2]);
                float k3 = __bfloat162float(sK[k * D + d + 3]);

                s += q0*k0 + q1*k1 + q2*k2 + q3*k3;
            }
            s *= inv_sqrt_d;

            // Apply causal mask
            int k_abs_local = k_start + k;
            scores[k] = ((q_abs < S) && (k_abs_local <= q_abs)) ? s : -FLT_MAX;
        }

        // Find new max across this block of keys
        float new_max = -FLT_MAX;
        for (int k = 0; k < BN; ++k) {
            if (scores[k] > new_max) new_max = scores[k];
        }

        // Rescale old accumulator entries if max increased
        if (cur_max > -FLT_MAX && new_max > cur_max) {
            float alpha = expf(cur_max - new_max);
            for (int d = 0; d < D; ++d) {
                o_acc[q_row * D + d] *= alpha;
            }
            cur_sum *= alpha;
        }
        cur_max = new_max;

        // Accumulate weighted V contributions
        for (int k = 0; k < BN; ++k) {
            float sc = scores[k];
            if (sc == -FLT_MAX) continue;

            float w = expf(sc - cur_max);
            cur_sum += w;

            for (int d = 0; d < D; ++d) {
                o_acc[q_row * D + d] += w * __bfloat162float(sV[k * D + d]);
            }
        }

        __syncthreads();
    }

    // Epilogue: normalize, compute LSE, store O
    if (q_abs < S) {
        if (cur_sum > 0.0f) {
            float inv_l = 1.0f / cur_sum;
            LSE[q_abs] = cur_max + logf(cur_sum);
            for (int d = 0; d < D; ++d) {
                Obse[(size_t)q_abs * D + d] = __float2bfloat16(o_acc[q_row * D + d] * inv_l);
            }
        } else {
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

    size_t smem_size = static_cast<size_t>(BM)*D*sizeof(__nv_bfloat16)
                     + static_cast<size_t>(BN)*D*sizeof(__nv_bfloat16)
                     + static_cast<size_t>(BN)*D*sizeof(__nv_bfloat16)
                     + static_cast<size_t>(BM)*D*sizeof(float);

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