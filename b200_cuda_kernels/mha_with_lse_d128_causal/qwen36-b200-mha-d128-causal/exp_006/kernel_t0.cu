#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
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
static constexpr int BM = 64;   // Query tile size
static constexpr int BN = 64;   // Key/Value tile size
static constexpr float INV_SQRT_D = 0.08838834764831844f; // 1/sqrt(128)

template <int BLOCK_SIZE, int BM_, int BN_>
__global__ void mha_causal_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S, int B, int H)
{
    constexpr int BM_VAL = BM_;
    constexpr int BN_VAL = BN_;
    constexpr int D = HEAD_DIM;
    static_assert(BLOCK_SIZE == 128);

    // Shared memory for K and V staging: BN*D bytes each
    __shared__ __align__(128) unsigned int sK_smem[(BN_VAL * D) / sizeof(unsigned int)];
    __shared__ __align__(128) unsigned int sV_smem[(BN_VAL * D) / sizeof(unsigned int)];
    __nv_bfloat16* sK = reinterpret_cast<__nv_bfloat16*>(sK_smem);
    __nv_bfloat16* sV = reinterpret_cast<__nv_bfloat16*>(sV_smem);

    const int b = blockIdx.y / H;
    const int h = blockIdx.y % H;
    const int bm_start = blockIdx.x * BM_VAL;
    const int tid = threadIdx.x;
    const int lane = tid % 32;

    const int bh_offset = (b * H + h) * S * D;
    const int stride_d = S * D;

    // Temporary local storage for Q rows assigned to this thread
    int num_q_rows = (BM_VAL + 31) / 32;  // At most 2 for BM=64
    // Use fixed size for simplicity: each thread processes up to 2 rows
    float my_q[2][D];
    float my_out[2][D];
    float my_max[2];
    float my_sum[2];

    // Process each query tile
    for (int bm_idx = 0; bm_idx < S; bm_idx += BM_VAL) {
        int cur_bm = bm_idx;
        if (b * H + h >= B * H) break;

        // Initialize per-row state
        for (int rr = 0; rr < 2; rr++) {
            my_max[rr] = -FLT_MAX;
            my_sum[rr] = 0.0f;
            for (int d = 0; d < D; d++) my_out[rr][d] = 0.0f;
        }

        // Which query rows does this thread own?
        int q_row_starts[2];
        int actual_num_q = 0;
        for (int w = 0; w < (BLOCK_SIZE / 32); w++) {
            if (tid >= w * 32 && tid < (w + 1) * 32) {
                int base_row = cur_bm + w * 16 + lane;
                q_row_starts[w] = base_row;
                if (base_row < cur_bm + BM_VAL && base_row < S) {
                    actual_num_q++;
                }
            }
        }

        // Actually simplify: warp 0 -> rows 0..15 of BM, warp 1 -> rows 16..31, etc
        // So warp_id = tid / 32, lane = tid % 32, each lane owns 2 rows: lane*2, lane*2+1
        int warp_id = tid / 32;
        int first_q = cur_bm + warp_id * 16 + lane;

        for (int qi = 0; qi < 2; qi++) {
            int q_pos = first_q + qi;
            if (q_pos < S && q_pos < cur_bm + BM_VAL) {
                int q_off = bh_offset + q_pos * D;
                for (int d = 0; d < D; d += 2) {
                    float2 tmp = reinterpret_cast<const float2*>(&Q[q_off + d])[lane % 1];
                    // Better: load element by element
                }
            }
        }

        // Simpler Q loading: each thread loads elements for its 2 rows
        for (int qi = 0; qi < 2; qi++) {
            int q_pos = first_q + qi;
            if (q_pos >= S || q_pos >= cur_bm + BM_VAL) {
                for (int d = 0; d < D; d++) my_q[qi][d] = 0.0f;
                my_max[qi] = -FLT_MAX;
                my_sum[qi] = 0.0f;
                for (int d = 0; d < D; d++) my_out[qi][d] = 0.0f;
                continue;
            }
            int q_base = bh_offset + q_pos * D;
            for (int d = tid; d < D; d += BLOCK_SIZE) {
                my_q[qi][d] = static_cast<float>(Q[q_base + d]);
            }
            my_max[qi] = -FLT_MAX;
            my_sum[qi] = 0.0f;
            for (int d = 0; d < D; d++) my_out[qi][d] = 0.0f;
        }
        __syncthreads();

        // Loop over key/value tiles
        for (int bn_idx = 0; bn_idx < S; bn_idx += BN_VAL) {
            int cur_bn = bn_idx;
            int effective_bn = min(BN_VAL, S - cur_bn);

            // Cooperative load K tile into shared memory
            for (int d = 0; d < D; d += 4) {
                for (int j = tid; j < effective_bn; j += BLOCK_SIZE) {
                    int src_k = bh_offset + (cur_bn + j) * D + d;
                    uint4 val;
                    memcpy(&val, &K[src_k], sizeof(uint4));
                    reinterpret_cast<uint4*>(&sK[j * D + d])[0] = val;
                }
                for (int j = tid; j < effective_bn; j += BLOCK_SIZE) {
                    int src_v = bh_offset + (cur_bn + j) * D + d;
                    uint4 val;
                    memcpy(&val, &V[src_v], sizeof(uint4));
                    reinterpret_cast<uint4*>(&sV[j * D + d])[0] = val;
                }
            }
            __syncthreads();

            // Compute QK^T and update softmax + accumulate OV
            for (int qi = 0; qi < 2; qi++) {
                int q_pos = first_q + qi;
                if (q_pos >= S || q_pos >= cur_bm + BM_VAL) continue;

                float old_m = my_max[qi];
                float old_s = my_sum[qi];

                // Compute attention scores for this Q row against K tile
                float local_p[BN_VAL];  // Scores after exp
                float new_m = -FLT_MAX;

                for (int j = 0; j < effective_bn; j++) {
                    int k_pos = cur_bn + j;
                    if (k_pos > q_pos) {
                        local_p[j] = 0.0f;  // Causal mask: zero out
                        continue;
                    }
                    float sim = 0.0f;
                    #pragma unroll 8
                    for (int d = 0; d < D; d += 8) {
                        float k0 = static_cast<float>(sK[j * D + d]);
                        float k1 = static_cast<float>(sK[j * D + d + 1]);
                        float k2 = static_cast<float>(sK[j * D + d + 2]);
                        float k3 = static_cast<float>(sK[j * D + d + 3]);
                        float k4 = static_cast<float>(sK[j * D + d + 4]);
                        float k5 = static_cast<float>(sK[j * D + d + 5]);
                        float k6 = static_cast<float>(sK[j * D + d + 6]);
                        float k7 = static_cast<float>(sK[j * D + d + 7]);
                        sim += my_q[qi][d] * k0 + my_q[qi][d+1] * k1 + my_q[qi][d+2] * k2 + my_q[qi][d+3] * k3;
                        sim += my_q[qi][d+4] * k4 + my_q[qi][d+5] * k5 + my_q[qi][d+6] * k6 + my_q[qi][d+7] * k7;
                    }
                    sim *= INV_SQRT_D;
                    local_p[j] = sim;  // Store raw score
                    if (sim > new_m) new_m = sim;
                }

                // Reduce new_m across warps
                for (int offset = 16; offset > 0; offset /= 2) {
                    float other = __shfl_down_sync(0xFFFFFFFF, new_m, offset);
                    if (other > new_m) new_m = other;
                }

                // Now compute P = exp(score - old_m) scaled properly
                // Standard online softmax: rescale previous output, add new contribution
                for (int j = 0; j < effective_bn; j++) {
                    int k_pos = cur_bn + j;
                    if (k_pos > q_pos) {
                        local_p[j] = 0.0f;
                        continue;
                    }
                    // Scale: exp(new_score - new_m)
                    local_p[j] = expf(local_p[j] - new_m);
                }

                // Find local sum
                float local_sum = 0.0f;
                for (int j = 0; j < effective_bn; j++) {
                    local_sum += local_p[j];
                }
                // Reduce sum across warps
                for (int offset = 16; offset > 0; offset /= 2) {
                    local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
                }

                // Combine with previous state
                float ratio = expf(old_m - new_m);
                float combined_sum = old_s * ratio + local_sum;

                // Update output: scale previous by ratio, add new contributions
                for (int d = 0; d < D; d++) {
                    my_out[qi][d] *= ratio;
                }

                for (int j = 0; j < effective_bn; j++) {
                    if (local_p[j] == 0.0f) continue;
                    float pj = local_p[j];
                    #pragma unroll 8
                    for (int d = 0; d < D; d += 8) {
                        float v0 = static_cast<float>(sV[j * D + d]);
                        float v1 = static_cast<float>(sV[j * D + d + 1]);
                        float v2 = static_cast<float>(sV[j * D + d + 2]);
                        float v3 = static_cast<float>(sV[j * D + d + 3]);
                        float v4 = static_cast<float>(sV[j * D + d + 4]);
                        float v5 = static_cast<float>(sV[j * D + d + 5]);
                        float v6 = static_cast<float>(sV[j * D + d + 6]);
                        float v7 = static_cast<float>(sV[j * D + d + 7]);
                        my_out[qi][d] += pj * v0;
                        my_out[qi][d+1] += pj * v1;
                        my_out[qi][d+2] += pj * v2;
                        my_out[qi][d+3] += pj * v3;
                        my_out[qi][d+4] += pj * v4;
                        my_out[qi][d+5] += pj * v5;
                        my_out[qi][d+6] += pj * v6;
                        my_out[qi][d+7] += pj * v7;
                    }
                }

                my_max[qi] = new_m;
                my_sum[qi] = combined_sum;
            }
            __syncthreads();
        }

        // Final normalize and write output
        for (int qi = 0; qi < 2; qi++) {
            int q_pos = first_q + qi;
            if (q_pos >= S || q_pos >= cur_bm + BM_VAL) continue;

            float denom = 1.0f / my_sum[qi];
            float lse_val = my_max[qi] + logf(my_sum[qi]);

            int out_base = bh_offset + q_pos * D;
            for (int d = 0; d < D; d++) {
                O[out_base + d] = __float2bfloat16(my_out[qi][d] * denom);
            }
            LSE[(b * H + h) * S + q_pos] = lse_val;
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = static_cast<int>(Q.size(0));
    int H = static_cast<int>(Q.size(1));
    int S = static_cast<int>(Q.size(2));
    int D = static_cast<int>(Q.size(3));

    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE.data_ptr());

    dim3 block(BLOCK_SIZE);
    dim3 grid((S + BM - 1) / BM, B * H);
    int smem_size = 2 * BN * D * sizeof(__nv_bfloat16);

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    mha_causal_kernel<128, 64, 64><<<grid, block, smem_size, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr, S, B, H);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_causal_impl::run);

}  // namespace mha_causal_impl