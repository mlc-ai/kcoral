#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_impl {

constexpr int HEAD_DIM = 128;

__global__ void mha_fwd_kernel(
    const __nv_bfloat16* __restrict__ Q_gm,
    const __nv_bfloat16* __restrict__ K_gm,
    const __nv_bfloat16* __restrict__ V_gm,
    __nv_bfloat16* __restrict__ O_gm,
    float* __restrict__ LSE_gm,
    int B, int H, int S, int D,
    int seg_size)
{
    // Each thread handles ONE output row (one q position) for ONE (b, h) pair
    // Sequential scan over K/V segments using online softmax

    uint64_t bhq_offset = blockIdx.x;
    uint64_t n_output_rows = (uint64_t)B * H * S;
    if (bhq_offset >= n_output_rows) return;

    int s_idx = (int)(bhq_offset % S);
    uint64_t rem = bhq_offset / S;
    int h_idx = (int)(rem % H);
    int b_idx = (int)(rem / H);

    int row_str = D;
    int head_str = S * D;
    int batch_str = H * S * D;

    const __nv_bfloat16* Q_row = Q_gm + bhq_offset * row_str;
    const __nv_bfloat16* K_base = K_gm + ((uint64_t)b_idx * H + h_idx) * head_str;
    const __nv_bfloat16* V_base = V_gm + ((uint64_t)b_idx * H + h_idx) * head_str;
    __nv_bfloat16*       O_row = O_gm + bhq_offset * row_str;
    float*              LSE_loc = LSE_gm + (b_idx * H + h_idx) * S + s_idx;

    // Load Q into registers
    float q_reg[D];
    #pragma unroll
    for (int d = 0; d < D; ++d) {
        q_reg[d] = static_cast<float>(Q_row[d]);
    }

    float scale = rsqrtf((float)D);

    // Online softmax accumulators
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float o_acc[D];
    #pragma unroll
    for (int d = 0; d < D; ++d) {
        o_acc[d] = 0.0f;
    }

    int n_segs = (S + seg_size - 1) / seg_size;

    for (int seg = 0; seg < n_segs; ++seg) {
        int k_start = seg * seg_size;
        int n_k = min(k_start + seg_size, S) - k_start;

        // Find segment max for numerical stability
        float seg_max = -INFINITY;
        for (int j = 0; j < n_k; ++j) {
            int kpos = k_start + j;
            float dot = 0.0f;
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                dot += q_reg[d] * static_cast<float>(K_base[kpos * row_str + d]);
            }
            float pval = dot * scale;
            if (pval > seg_max) seg_max = pval;
        }

        // If seg_max is -INF (empty segment), skip
        if (seg_max == -INFINITY) continue;

        // Merge with running softmax state
        bool first_segment = (running_max == -INFINITY);
        float alpha;

        if (!first_segment) {
            alpha = expf(running_max - seg_max);
            running_sum *= alpha;
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                o_acc[d] *= alpha;
            }
        }

        running_max = seg_max;

        // Accumulate current segment
        for (int j = 0; j < n_k; ++j) {
            int kpos = k_start + j;
            float dot = 0.0f;
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                dot += q_reg[d] * static_cast<float>(K_base[kpos * row_str + d]);
            }
            float pval = dot * scale;
            float e = expf(pval - seg_max);
            running_sum += e;
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                o_acc[d] += e * static_cast<float>(V_base[kpos * row_str + d]);
            }
        }
    }

    // Finalize: normalize and write output
    if (running_sum > 0.0f) {
        float inv_denom = 1.0f / running_sum;
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            O_row[d] = __float2bfloat16(o_acc[d] * inv_denom);
        }
        *LSE_loc = running_max + logf(running_sum);
    } else {
        // Should not happen with valid input, but guard anyway
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            O_row[d] = __float2bfloat16(0.0f);
        }
        *LSE_loc = -INFINITY;
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = static_cast<int>(Q.size(0));
    int H = static_cast<int>(Q.size(1));
    int S = static_cast<int>(Q.size(2));
    int D = static_cast<int>(Q.size(3));

    if (S == 0 || B == 0 || H == 0 || D != HEAD_DIM) return;

    const __nv_bfloat16* Q_d = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_d = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_d = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_d = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_d = static_cast<float*>(LSE.data_ptr());

    // Segment size for scanning over K/V (power of 2 for good memory alignment)
    int seg_size = 64;
    int threads_per_block = 128;
    uint64_t total_elems = (uint64_t)B * H * S;
    dim3 grid(((total_elems + threads_per_block - 1) / threads_per_block));
    dim3 block(threads_per_block);

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    mha_fwd_kernel<<<grid, block, 0, stream>>>(
        Q_d, K_d, V_d, O_d, LSE_d, B, H, S, D, seg_size);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace mha_impl

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_impl::run);