#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                    \
    cudaError_t _e = (call);                                     \
    if (_e != cudaSuccess) {                                     \
        fprintf(stderr, "CUDA error %s at %s:%d\n",              \
                cudaGetErrorString(_e), __FILE__, __LINE__);     \
        exit(1);                                                 \
    }                                                            \
} while(0)

namespace tvm_ffi_example_cuda {

// Tile sizes
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int D = 128;
constexpr int THREADS = 256;

__global__ void attention_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S) {

    int bh = blockIdx.y;
    int b = bh / H;
    int h = bh % H;
    int q_block = blockIdx.x;
    int q_start = q_block * BM;
    int tid = threadIdx.x;

    const int64_t base_offset = (int64_t)(b * H + h) * S * D;
    const float scale = 0.08838834764831845f;  // 1/sqrt(128)

    extern __shared__ char smem_raw[];
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sK = sQ + BM * D;
    __nv_bfloat16* sV = sK + BN * D;
    float* sS = reinterpret_cast<float*>(sV + BN * D);
    float* sO = sS + BM * BN;
    float* s_rowmax = sO + BM * D;
    float* s_rowsum = s_rowmax + BM;

    // Load Q tile [BM, D]
    #pragma unroll
    for (int i = tid; i < BM * D; i += THREADS) {
        int row = i / D;
        int col = i % D;
        int q_pos = q_start + row;
        sQ[i] = (q_pos < S) ? Q[base_offset + (int64_t)q_pos * D + col]
                            : __float2bfloat16(0.0f);
    }

    // Initialize accumulators
    for (int i = tid; i < BM; i += THREADS) {
        s_rowmax[i] = -INFINITY;
        s_rowsum[i] = 0.0f;
    }
    for (int i = tid; i < BM * D; i += THREADS) {
        sO[i] = 0.0f;
    }
    __syncthreads();

    // Causal: iterate key blocks from 0 to the block containing q_start+BM-1
    int max_k = min(S, q_start + BM);
    int num_k_blocks = (max_k + BN - 1) / BN;

    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        int k_start = k_block * BN;

        // Load K, V tiles [BN, D]
        for (int i = tid; i < BN * D; i += THREADS) {
            int row = i / D;
            int col = i % D;
            int k_pos = k_start + row;
            if (k_pos < S) {
                sK[i] = K[base_offset + (int64_t)k_pos * D + col];
                sV[i] = V[base_offset + (int64_t)k_pos * D + col];
            } else {
                sK[i] = __float2bfloat16(0.0f);
                sV[i] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        // Compute S = Q @ K^T * scale, with causal mask
        for (int idx = tid; idx < BM * BN; idx += THREADS) {
            int i = idx / BN;
            int j = idx % BN;
            float sum = 0.0f;
            #pragma unroll 8
            for (int d = 0; d < D; d++) {
                sum += __bfloat162float(sQ[i * D + d]) *
                       __bfloat162float(sK[j * D + d]);
            }
            sum *= scale;
            int q_pos = q_start + i;
            int k_pos = k_start + j;
            if (k_pos > q_pos) sum = -INFINITY;
            sS[idx] = sum;
        }
        __syncthreads();

        // Online softmax: update rowmax, rowsum, rescale O, compute P
        for (int i = tid; i < BM; i += THREADS) {
            int q_pos = q_start + i;
            if (q_pos >= S) continue;

            // Local max over valid entries
            float local_max = -INFINITY;
            for (int j = 0; j < BN; j++) {
                int k_pos = k_start + j;
                if (k_pos <= q_pos && k_pos < S) {
                    local_max = fmaxf(local_max, sS[i * BN + j]);
                }
            }

            float old_max = s_rowmax[i];
            float new_max = fmaxf(old_max, local_max);

            float exp_old = (old_max == -INFINITY) ? 0.0f
                                                    : __expf(old_max - new_max);

            float local_sum = 0.0f;
            for (int j = 0; j < BN; j++) {
                int k_pos = k_start + j;
                if (k_pos <= q_pos && k_pos < S) {
                    float val = __expf(sS[i * BN + j] - new_max);
                    sS[i * BN + j] = val;
                    local_sum += val;
                } else {
                    sS[i * BN + j] = 0.0f;
                }
            }

            float old_sum = s_rowsum[i];
            // Rescale O accumulator
            for (int d = 0; d < D; d++) {
                sO[i * D + d] *= exp_old;
            }

            s_rowmax[i] = new_max;
            s_rowsum[i] = old_sum * exp_old + local_sum;
        }
        __syncthreads();

        // O += P @ V  where P is in sS (float), V is in sV (bf16)
        for (int idx = tid; idx < BM * D; idx += THREADS) {
            int i = idx / D;
            int d = idx % D;
            int q_pos = q_start + i;
            if (q_pos >= S) continue;

            float sum = 0.0f;
            for (int j = 0; j < BN; j++) {
                sum += sS[i * BN + j] * __bfloat162float(sV[j * D + d]);
            }
            sO[idx] += sum;
        }
        __syncthreads();
    }

    // Final normalization: O = O / rowsum, store LSE = rowmax + log(rowsum)
    for (int i = tid; i < BM; i += THREADS) {
        int q_pos = q_start + i;
        if (q_pos >= S) continue;
        LSE[(int64_t)(b * H + h) * S + q_pos] =
            s_rowmax[i] + logf(s_rowsum[i]);
    }

    for (int idx = tid; idx < BM * D; idx += THREADS) {
        int i = idx / D;
        int d = idx % D;
        int q_pos = q_start + i;
        if (q_pos < S) {
            float val = sO[idx] / s_rowsum[i];
            O[base_offset + (int64_t)q_pos * D + d] = __float2bfloat16(val);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K,
         tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    const int B = 4;
    const int H = 48;
    const int D_val = 128;
    const int S = static_cast<int>(Q.size(2));  // Q shape: [B, H, S, D]

    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    dim3 grid((S + BM - 1) / BM, B * H);
    dim3 block(THREADS);

    // Shared memory: sQ + sK + sV + sS + sO + s_rowmax + s_rowsum
    int smem_size = BM * D * 2   // sQ (bf16)
                  + BN * D * 2   // sK (bf16)
                  + BN * D * 2   // sV (bf16)
                  + BM * BN * 4  // sS (float)
                  + BM * D * 4   // sO (float)
                  + BM * 4       // s_rowmax
                  + BM * 4;      // s_rowsum

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    cudaFuncSetAttribute(
        attention_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    attention_kernel<<<grid, block, smem_size, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data, B, H, S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda