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

// Shared memory layout (all tiles are BN x D):
// sQ:     [BN][D] bf16  = 64*128*2 = 16KB  
// sKVT:   [BN][D] bf16  = 64*128*2 = 16KB  (K transposed conceptually)
// sV:     [BN][D] bf16  = 64*128*2 = 16KB
// o_acc:  [BN][D] fp32  = 64*128*4 = 32KB
// Total:  80 KB

__global__ void fa_fwd_kernel(
    const __nv_bfloat16* __restrict__ Q_g,
    const __nv_bfloat16* __restrict__ K_g,
    const __nv_bfloat16* __restrict__ V_g,
    __nv_bfloat16* __restrict__ O_g,
    float* __restrict__ LSE_g,
    int S, int B, int H)
{
    extern __shared__ char smem[];

    __nv_bfloat16* sQ   = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sKV  = sQ   + BM * D;
    __nv_bfloat16* sV   = sKV  + BN * D;
    float* o_acc        = reinterpret_cast<float*>(sV + BN * D);

    int bid = blockIdx.x;
    int q_blocks_per_bh = (S + BM - 1) / BM;
    int q_block_idx = bid % q_blocks_per_bh;
    int bh_idx      = bid / q_blocks_per_bh;
    int batch_idx   = bh_idx / H;
    int head_idx    = bh_idx % H;

    int q_start = q_block_idx * BM;
    int tid     = threadIdx.x;

    // Base pointers for this (batch, head)
    size_t offset_bh = (size_t)batch_idx * H * S * D + head_idx * S * D;
    
    const __nv_bfloat16* Qbase = Q_g + offset_bh;
    const __nv_bfloat16* Kbase = K_g + offset_bh;
    const __nv_bfloat16* Vbase = V_g + offset_bh;
    __nv_bfloat16*         Obase = O_g + offset_bh;
    float*                 LSE = LSE_g + (size_t)batch_idx * H * S + head_idx * S;

    float inv_sqrt_d = rsqrtf((float)D);
    int num_k_steps  = (S + BN - 1) / BN;

    // Init output accumulators for this CTA's query rows
    #pragma unroll
    for (int i = tid; i < BM * D; i += NT) {
        o_acc[i] = 0.0f;
    }
    __syncthreads();

    // Load Q tile cooperatively: sQ[row][col] = Q[q_start+row][col]
    for (int col = tid; col < D; col += NT) {
        for (int row = 0; row < BM; row++) {
            int q_abs = q_start + row;
            if (q_abs < S) {
                sQ[row * D + col] = Qbase[(size_t)q_abs * D + col];
            }
        }
    }
    __syncthreads();

    // Each thread handles one query row
    int my_q_row  = tid;
    int my_q_abs  = q_start + my_q_row;
    
    // Online softmax state
    float cur_max  = -FLT_MAX;
    float cur_sum  = 0.0f;

    float S_local[BN];  // scores for current key block

    for (int ks = 0; ks < num_k_steps; ks++) {
        int k_start = ks * BN;
        bool last_k = (ks == num_k_steps - 1);

        // Cooperatively load K tile: sKV[row][col] = K[k_start+row][col]
        // Also load V tile: sV[row][col] = V[k_start+row][col]
        for (int col = tid; col < D; col += NT) {
            for (int row = 0; row < BN; row++) {
                int k_abs = k_start + row;
                if (k_abs < S) {
                    sKV[row * D + col] = Kbase[(size_t)k_abs * D + col];
                    sV[row * D + col]  = Vbase[(size_t)k_abs * D + col];
                } else {
                    sKV[row * D + col] = __float2bfloat16(0.0f);
                    sV[row * D + col]  = __float2bfloat16(0.0f);
                }
            }
        }
        __syncthreads();

        // Compute S[my_q_row][k] for k in [0, BN)
        for (int k = 0; k < BN; k++) {
            float s = 0.0f;
            for (int d = 0; d < D; d += 4) {
                float q0 = __bfloat162float(sQ[my_q_row * D + d]);
                float q1 = __bfloat162float(sQ[my_q_row * D + d + 1]);
                float q2 = __bfloat162float(sQ[my_q_row * D + d + 2]);
                float q3 = __bfloat162float(sQ[my_q_row * D + d + 3]);

                float kv0 = __bfloat162float(sKV[k * D + d]);
                float kv1 = __bfloat162float(sKV[k * D + d + 1]);
                float kv2 = __bfloat162float(sKV[k * D + d + 2]);
                float kv3 = __bfloat162float(sKV[k * D + d + 3]);

                s += q0*kv0 + q1*kv1 + q2*kv2 + q3*kv3;
            }
            s *= inv_sqrt_d;

            int k_abs_local = k_start + k;
            if (my_q_abs >= S || k_abs_local > my_q_abs) {
                S_local[k] = -FLT_MAX;
            } else {
                S_local[k] = s;
            }
        }

        // Find new max
        float new_max = -FLT_MAX;
        for (int k = 0; k < BN; k++) {
            if (S_local[k] > new_max) new_max = S_local[k];
        }

        // Rescale accumulator if needed
        if (new_max > cur_max && cur_max > -FLT_MAX) {
            float alpha = expf(cur_max - new_max);
            for (int d = 0; d < D; d++) {
                o_acc[my_q_row * D + d] *= alpha;
            }
            cur_sum *= alpha;
        }
        cur_max = new_max;

        // Accumulate weighted V
        for (int k = 0; k < BN; k++) {
            float sv = S_local[k];
            if (sv == -FLT_MAX) continue;

            float w = expf(sv - cur_max);
            cur_sum += w;

            for (int d = 0; d < D; d++) {
                o_acc[my_q_row * D + d] += w * __bfloat162float(sV[k * D + d]);
            }
        }
        __syncthreads();
    }

    // Epilogue: normalize and write back
    if (my_q_abs < S && cur_sum > 0.0f) {
        float inv_l = 1.0f / cur_sum;
        float logsumexp = cur_max + logf(cur_sum);
        
        // Use atomicAdd for LSE in case multiple blocks write (shouldn't happen with our grid layout)
        LSE[my_q_abs] = logsumexp;

        for (int d = 0; d < D; d++) {
            Obase[(size_t)my_q_abs * D + d] = __float2bfloat16(o_acc[my_q_row * D + d] * inv_l);
        }
    } else if (my_q_abs < S) {
        // No valid keys (edge case)
        LSE[my_q_abs] = -FLT_MAX;
        for (int d = 0; d < D; d++) {
            Obase[(size_t)my_q_abs * D + d] = __float2bfloat16(0.0f);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);

    int q_blocks_per_bh = (int)((S + BM - 1) / BM);
    int total_blocks = (int)(B * H * q_blocks_per_bh);

    dim3 grid(total_blocks);
    dim3 block(NT);

    // Shared memory: 3 * (tile * D * sizeof(bf16)) + 1 * (tile * D * sizeof(fp32))
    size_t smem_size = 3ULL * (size_t)BM * D * sizeof(__nv_bfloat16)
                    + 1ULL * (size_t)BM * D * sizeof(float);

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