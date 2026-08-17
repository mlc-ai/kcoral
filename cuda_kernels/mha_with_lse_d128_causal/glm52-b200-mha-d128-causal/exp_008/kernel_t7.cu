#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
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
} while (0)

namespace mha_lse_d128 {

constexpr int D = 128;
constexpr int BQ = 32;
constexpr int BK = 32;
constexpr int THREADS = 128;

__global__ void mha_causal_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S)
{
    const int bh_idx = blockIdx.x;
    const int b = bh_idx / H;
    const int h = bh_idx % H;
    const int q_block = blockIdx.y;
    const int tid = threadIdx.x;

    extern __shared__ char smem_buf[];
    __nv_bfloat16* Q_smem = reinterpret_cast<__nv_bfloat16*>(smem_buf);
    __nv_bfloat16* K_smem = Q_smem + BQ * D;
    __nv_bfloat16* V_smem = K_smem + BK * D;

    const int64_t bh_offset = (int64_t)(b * H + h) * S * D;
    const __nv_bfloat16* Q_base = Q + bh_offset;
    const __nv_bfloat16* K_base = K + bh_offset;
    const __nv_bfloat16* V_base = V + bh_offset;
    __nv_bfloat16* O_base = O + bh_offset;
    float* LSE_base = LSE + (int64_t)(b * H + h) * S;

    const int q_start = q_block * BQ;
    constexpr float scale = 0.08838834764831845f;

    // Load Q tile to shared memory
    for (int i = tid; i < BQ * D / 8; i += THREADS) {
        int row = i / (D / 8);
        int col8 = i % (D / 8);
        int q_idx = q_start + row;
        if (q_idx < S) {
            reinterpret_cast<uint4*>(Q_smem)[i] =
                reinterpret_cast<const uint4*>(Q_base + (int64_t)q_idx * D)[col8];
        } else {
            reinterpret_cast<uint4*>(Q_smem)[i] = make_uint4(0, 0, 0, 0);
        }
    }
    __syncthreads();

    const int q_idx = q_start + tid;
    const bool valid = (q_idx < S);

    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float o_acc[D];
    #pragma unroll
    for (int d = 0; d < D; d++) o_acc[d] = 0.0f;

    int max_q = min(q_start + BQ, S);
    int num_k_blocks = (max_q + BK - 1) / BK;

    for (int kb = 0; kb < num_k_blocks; kb++) {
        int k_start = kb * BK;

        for (int i = tid; i < BK * D / 8; i += THREADS) {
            int row = i / (D / 8);
            int col8 = i % (D / 8);
            int k_idx = k_start + row;
            if (k_idx < S) {
                reinterpret_cast<uint4*>(K_smem)[i] =
                    reinterpret_cast<const uint4*>(K_base + (int64_t)k_idx * D)[col8];
                reinterpret_cast<uint4*>(V_smem)[i] =
                    reinterpret_cast<const uint4*>(V_base + (int64_t)k_idx * D)[col8];
            } else {
                reinterpret_cast<uint4*>(K_smem)[i] = make_uint4(0, 0, 0, 0);
                reinterpret_cast<uint4*>(V_smem)[i] = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        if (valid && q_idx >= k_start) {
            float scores[BK];
            for (int k = 0; k < BK; k++) {
                int k_idx = k_start + k;
                if (k_idx > q_idx || k_idx >= S) {
                    scores[k] = -INFINITY;
                } else {
                    float dot = 0.0f;
                    #pragma unroll
                    for (int d = 0; d < D; d += 4) {
                        float2 q0 = __bfloat1622float2(
                            *reinterpret_cast<__nv_bfloat162*>(&Q_smem[tid * D + d]));
                        float2 q1 = __bfloat1622float2(
                            *reinterpret_cast<__nv_bfloat162*>(&Q_smem[tid * D + d + 2]));
                        float2 k0 = __bfloat1622float2(
                            *reinterpret_cast<__nv_bfloat162*>(&K_smem[k * D + d]));
                        float2 k1 = __bfloat1622float2(
                            *reinterpret_cast<__nv_bfloat162*>(&K_smem[k * D + d + 2]));
                        dot += q0.x * k0.x + q0.y * k0.y + q1.x * k1.x + q1.y * k1.y;
                    }
                    scores[k] = dot * scale;
                }
            }

            float block_max = -INFINITY;
            #pragma unroll
            for (int k = 0; k < BK; k++) {
                block_max = fmaxf(block_max, scores[k]);
            }

            float new_max = fmaxf(row_max, block_max);
            float exp_diff = (row_max > -INFINITY) ? __expf(row_max - new_max) : 0.0f;

            #pragma unroll
            for (int d = 0; d < D; d++) {
                o_acc[d] *= exp_diff;
            }

            float block_sum = 0.0f;
            #pragma unroll
            for (int k = 0; k < BK; k++) {
                float p = __expf(scores[k] - new_max);
                scores[k] = p;
                block_sum += p;
            }

            row_sum = row_sum * exp_diff + block_sum;
            row_max = new_max;

            for (int k = 0; k < BK; k++) {
                float p = scores[k];
                #pragma unroll
                for (int d = 0; d < D; d += 4) {
                    float2 v0 = __bfloat1622float2(
                        *reinterpret_cast<__nv_bfloat162*>(&V_smem[k * D + d]));
                    float2 v1 = __bfloat1622float2(
                        *reinterpret_cast<__nv_bfloat162*>(&V_smem[k * D + d + 2]));
                    o_acc[d]     += p * v0.x;
                    o_acc[d + 1] += p * v0.y;
                    o_acc[d + 2] += p * v1.x;
                    o_acc[d + 3] += p * v1.y;
                }
            }
        }

        __syncthreads();
    }

    if (valid) {
        float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;

        for (int d = 0; d < D; d += 8) {
            __nv_bfloat16 vals[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                vals[i] = __float2bfloat16(o_acc[d + i] * inv_sum);
            }
            *reinterpret_cast<uint4*>(O_base + (int64_t)q_idx * D + d) =
                *reinterpret_cast<uint4*>(vals);
        }

        LSE_base[q_idx] = (row_sum > 0.0f) ? (row_max + logf(row_sum)) : -INFINITY;
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    const int B = 4;
    const int H = 48;
    const int S = static_cast<int>(Q.size(2));

    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    dim3 grid(B * H, (S + BQ - 1) / BQ);
    dim3 block(THREADS);

    int smem_size = (BQ * D + BK * D + BK * D) * sizeof(__nv_bfloat16);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(mha_causal_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    mha_causal_kernel<<<grid, block, smem_size, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data, B, H, S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_lse_d128::run);

}  // namespace mha_lse_d128