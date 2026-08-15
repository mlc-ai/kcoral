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
    constexpr int D = HEAD_DIM;
    constexpr float NEG_INF = -5e4f;

    // Shared memory: K_tile[BN][D], V_tile[BN][D]
    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* sK = smem;
    __nv_bfloat16* sV = smem + BN * D;

    int b = blockIdx.y / H;
    int h = blockIdx.y % H;
    int bm_start = blockIdx.x * BM;
    int tid = threadIdx.x;

    int bh_offset = (b * H + h) * S * D;
    int eff_bm = min(BM, S - bm_start);

    // Threads 0..BM-1 each own one query row; threads BM..BSZ-1 help with loads only
    bool compute = (tid < eff_bm);
    int q_global = bm_start + tid;

    float my_max = NEG_INF;
    float my_sum = 0.0f;
    
    // Only active computation threads need the accumulator array
    // We declare it conditionally to save registers for helper threads
    float* acc = nullptr;
    float q_vec[D];

    if (compute) {
        // Allocate per-thread acc on stack via local array
        // Load Q row into registers
        int q_off = bh_offset + q_global * D;
        for (int d = 0; d < D; ++d) {
            q_vec[d] = static_cast<float>(Q[q_off + d]);
        }
    }

    // Use a large-enough local buffer that all threads can write
    // Helper threads just don't read it after
    float acc_buf[D];
    for (int d = 0; d < D; ++d) acc_buf[d] = 0.0f;

    // Main loop over KV tiles
    for (int bn_idx = 0; bn_idx < S; bn_idx += BN) {
        int cur_bn = bn_idx;
        int eff_bn = min(BN, S - cur_bn);

        // All threads help load K and V into shared memory
        // Total elements per matrix: eff_bn * D = up to 8192
        int total_elems = eff_bn * D;
        for (int idx = tid; idx < total_elems; idx += blockDim.x) {
            int r = idx / D;
            int c = idx % D;
            sK[r * D + c] = K[bh_offset + (cur_bn + r) * D + c];
            sV[r * D + c] = V[bh_offset + (cur_bn + r) * D + c];
        }
        __syncthreads();

        if (!compute) {
            __syncthreads();
            continue;
        }

        // Compute QK^T for this query row against loaded K tile
        float local_m = NEG_INF;
        
        // First pass: compute raw attention scores and find max
        float scores[BN];
        for (int j = 0; j < eff_bn; j++) {
            int k_pos = cur_bn + j;
            if (k_pos > q_global) {
                scores[j] = NEG_INF;
                continue;
            }
            float sim = 0.0f;
            for (int d = 0; d < D; d++) {
                sim += q_vec[d] * static_cast<float>(sK[j * D + d]);
            }
            sim *= INV_SQRT_D;
            scores[j] = sim;
            if (sim > local_m) local_m = sim;
        }

        float old_m = my_max;
        float old_s = my_sum;

        // Second pass: exp(scores - local_m), accumulate PV
        float local_sum = 0.0f;
        for (int j = 0; j < eff_bn; j++) {
            if (scores[j] <= NEG_INF + 1.0f) {
                scores[j] = 0.0f;
                continue;
            }
            float p = expf(scores[j] - local_m);
            local_sum += p;
            for (int d = 0; d < D; d++) {
                acc_buf[d] += p * static_cast<float>(sV[j * D + d]);
            }
        }

        // Scale previous accumulation by exp(old_m - new_m)
        float ratio = expf(old_m - local_m);
        for (int d = 0; d < D; ++d) {
            acc_buf[d] *= ratio;
        }

        my_max = local_m;
        my_sum = old_s * ratio + local_sum;

        __syncthreads();
    }

    if (!compute) return;

    // Finalize and write output
    if (my_sum < 1e-30f) {
        my_sum = 1e-30f;
    }
    
    float inv_sum = 1.0f / my_sum;
    float lse = my_max + logf(my_sum);

    int out_base = bh_offset + q_global * D;
    for (int d = 0; d < D; d++) {
        O[out_base + d] = __float2bfloat16(acc_buf[d] * inv_sum);
    }
    LSE[(b * H + h) * S + q_global] = lse;
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

    // 128 threads per block: 64 compute + 64 helpers for SMEM loads
    dim3 block(128);
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