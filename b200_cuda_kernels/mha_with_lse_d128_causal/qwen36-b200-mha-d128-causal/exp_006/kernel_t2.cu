#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
#include <float.h>
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

namespace mha_causal_impl {

static constexpr int HEAD_DIM = 128;
static constexpr int BM = 64;
static constexpr int BN = 64;
static constexpr float INV_SQRT_D = 0.08838834764831844f;

__global__ void mha_causal_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S, int B, int H)
{
    static_assert(BM == blockDim.x && BN <= 64);
    constexpr int D = HEAD_DIM;
    constexpr float NEG_INF = -1e20f;

    // Shared memory for K and V staging
    extern __shared__ __nv_bfloat16 shared_mem[];
    __nv_bfloat16* sK = shared_mem;
    __nv_bfloat16* sV = shared_mem + BN * D;

    const int b = blockIdx.y / H;
    const int h = blockIdx.y % H;
    const int bm_start = blockIdx.x * BM;
    const int tid = threadIdx.x;  // 0..63, each thread owns one query row

    const int bh_offset = (b * H + h) * S * D;

    int eff_bm = min(BM, S - bm_start);
    int my_q_row = tid;  // query row index within this tile

    bool active = (my_q_row < eff_bm);
    int q_pos_global = bm_start + my_q_row;

    // Per-thread accumulators
    float my_row_max = NEG_INF;
    float my_row_sum = 0.0f;

    // Accumulator output vector
    float my_row_out[D];
    for (int d = 0; d < D; d++) my_row_out[d] = 0.0f;

    // Load Q row into registers once
    float q_reg[D];
    if (active) {
        int q_base = bh_offset + q_pos_global * D;
        for (int d = 0; d < D; d++) {
            q_reg[d] = static_cast<float>(Q[q_base + d]);
        }
    } else {
        for (int d = 0; d < D; d++) q_reg[d] = 0.0f;
    }

    // Loop over BN tiles of K/V
    for (int bn_idx = 0; bn_idx < S; bn_idx += BN) {
        int cur_bn = bn_idx;
        int eff_bn = min(BN, S - cur_bn);

        // Cooperative load K tile into shared memory
        for (int idx = tid; idx < eff_bn * D; idx += BM) {
            int j = idx / D;
            int d = idx % D;
            sK[j * D + d] = K[bh_offset + (cur_bn + j) * D + d];
        }

        // Cooperative load V tile into shared memory
        for (int idx = tid; idx < eff_bn * D; idx += BM) {
            int j = idx / D;
            int d = idx % D;
            sV[j * D + d] = V[bh_offset + (cur_bn + j) * D + d];
        }
        __syncthreads();

        if (!active) {
            __syncthreads();
            continue;
        }

        // Compute attention scores for this Q row against loaded K tile
        // and update softmax state online
        float local_new_m = NEG_INF;

        // First pass: compute raw scores and find new local max
        float scores[BN];
        for (int j = 0; j < eff_bn; j++) {
            int k_pos = cur_bn + j;
            if (k_pos > q_pos_global) {
                scores[j] = NEG_INF;
                continue;
            }
            float sim = 0.0f;
            for (int d = 0; d < D; d++) {
                sim += q_reg[d] * static_cast<float>(sK[j * D + d]);
            }
            sim *= INV_SQRT_D;
            scores[j] = sim;
            if (sim > local_new_m) local_new_m = sim;
        }

        // Old accumulator values
        float old_m = my_row_max;
        float old_s = my_row_sum;

        // Apply exp(scores - new_m), compute partial sum, accumulate OV
        float local_sum = 0.0f;
        for (int j = 0; j < eff_bn; j++) {
            if (scores[j] == NEG_INF) {
                scores[j] = 0.0f;
                continue;
            }
            float p = expf(scores[j] - local_new_m);
            local_sum += p;
            for (int d = 0; d < D; d++) {
                my_row_out[d] += p * static_cast<float>(sV[j * D + d]);
            }
        }

        // Scale previous outputs by exp(old_m - new_m)
        float ratio = expf(old_m - local_new_m);
        for (int d = 0; d < D; d++) {
            my_row_out[d] *= ratio;
        }

        // Update running max and sum
        my_row_max = local_new_m;
        my_row_sum = old_s * ratio + local_sum;

        __syncthreads();
    }

    // Write final output
    if (!active) return;

    float denom = 1.0f / my_row_sum;
    float lse_val = my_row_max + logf(my_row_sum);

    int out_base = bh_offset + q_pos_global * D;
    for (int d = 0; d < D; d++) {
        O[out_base + d] = __float2bfloat16(my_row_out[d] * denom);
    }
    LSE[(b * H + h) * S + q_pos_global] = lse_val;
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = static_cast<int>(Q.size(0));
    int H = static_cast<int>(Q.size(1));
    int S = static_cast<int>(Q.size(2));

    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE.data_ptr());

    dim3 block(BM);
    dim3 grid((S + BM - 1) / BM, B * H);
    int smem_size = 2 * BN * HEAD_DIM * sizeof(__nv_bfloat16);

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    mha_causal_kernel<<<grid, block, smem_size, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr, S, B, H);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_causal_impl::run);

}  // namespace mha_causal_impl