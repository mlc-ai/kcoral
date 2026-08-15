#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace flash_attn {

constexpr int D = 128;
constexpr int BM = 128;
constexpr int BK = 64;
constexpr int THREADS = 128;

__global__ void attention_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S) {

    int b = blockIdx.x;
    int h = blockIdx.y;
    int q_block = blockIdx.z;
    int q_start = q_block * BM;
    int tid = threadIdx.x;
    int q_idx = q_start + tid;

    extern __shared__ char smem_raw[];
    float* O_acc = reinterpret_cast<float*>(smem_raw);          // [BM][D] float
    __nv_bfloat16* Q_tile = reinterpret_cast<__nv_bfloat16*>(O_acc + BM * D);  // [BM][D] bf16
    __nv_bfloat16* K_tile = Q_tile + BM * D;                     // [BK][D] bf16
    __nv_bfloat16* V_tile = K_tile + BK * D;                     // [BK][D] bf16

    int bh = (b * H + h);
    int bh_offset = bh * S * D;
    const __nv_bfloat16* Q_base = Q + bh_offset;
    const __nv_bfloat16* K_base = K + bh_offset;
    const __nv_bfloat16* V_base = V + bh_offset;
    __nv_bfloat16* O_base = O + bh_offset;
    float* LSE_base = LSE + bh * S;

    // Load Q tile to shared memory
    for (int i = tid; i < BM * D / 8; i += THREADS) {
        int row = i / (D / 8);
        if (q_start + row < S) {
            reinterpret_cast<uint4*>(Q_tile)[i] = reinterpret_cast<const uint4*>(Q_base + (q_start + row) * D)[i % (D / 8)];
        } else {
            reinterpret_cast<uint4*>(Q_tile)[i] = make_uint4(0, 0, 0, 0);
        }
    }
    __syncthreads();

    // Initialize O_acc
    for (int i = tid; i < BM * D; i += THREADS) {
        O_acc[i] = 0.0f;
    }
    __syncthreads();

    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    const float scale = 0.0883883476f;  // 1/sqrt(128)

    int num_k_blocks = (S + BK - 1) / BK;

    for (int kb = 0; kb < num_k_blocks; kb++) {
        int k_start = kb * BK;

        // Load K and V tiles
        for (int i = tid; i < BK * D / 8; i += THREADS) {
            int row = i / (D / 8);
            if (k_start + row < S) {
                reinterpret_cast<uint4*>(K_tile)[i] = reinterpret_cast<const uint4*>(K_base + (k_start + row) * D)[i % (D / 8)];
                reinterpret_cast<uint4*>(V_tile)[i] = reinterpret_cast<const uint4*>(V_base + (k_start + row) * D)[i % (D / 8)];
            } else {
                reinterpret_cast<uint4*>(K_tile)[i] = make_uint4(0, 0, 0, 0);
                reinterpret_cast<uint4*>(V_tile)[i] = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        if (q_idx < S) {
            // Compute attention scores: Q[q] . K[k]^T * scale
            float scores[BK];
            #pragma unroll
            for (int k = 0; k < BK; k++) {
                int g_k = k_start + k;
                if (g_k >= S) {
                    scores[k] = -INFINITY;
                } else {
                    float dot = 0.0f;
                    #pragma unroll
                    for (int d = 0; d < D; d += 2) {
                        __nv_bfloat162 q_vec = *reinterpret_cast<__nv_bfloat162*>(&Q_tile[tid * D + d]);
                        __nv_bfloat162 k_vec = *reinterpret_cast<__nv_bfloat162*>(&K_tile[k * D + d]);
                        float2 qf = __bfloat1622float2(q_vec);
                        float2 kf = __bfloat1622float2(k_vec);
                        dot = fmaf(qf.x, kf.x, dot);
                        dot = fmaf(qf.y, kf.y, dot);
                    }
                    scores[k] = dot * scale;
                }
            }

            // Online softmax: find block max, update running max and sum
            float block_max = -INFINITY;
            #pragma unroll
            for (int k = 0; k < BK; k++) {
                block_max = fmaxf(block_max, scores[k]);
            }

            float new_max = fmaxf(max_score, block_max);
            float correction = (max_score > -INFINITY) ? __expf(max_score - new_max) : 0.0f;

            float block_sum = 0.0f;
            #pragma unroll
            for (int k = 0; k < BK; k++) {
                if (scores[k] > -INFINITY) {
                    float p = __expf(scores[k] - new_max);
                    scores[k] = p;
                    block_sum += p;
                } else {
                    scores[k] = 0.0f;
                }
            }

            sum_exp = sum_exp * correction + block_sum;

            // Update O accumulator
            #pragma unroll
            for (int d = 0; d < D; d += 4) {
                float o0 = O_acc[tid * D + d] * correction;
                float o1 = O_acc[tid * D + d + 1] * correction;
                float o2 = O_acc[tid * D + d + 2] * correction;
                float o3 = O_acc[tid * D + d + 3] * correction;

                #pragma unroll
                for (int k = 0; k < BK; k++) {
                    if (scores[k] > 0.0f) {
                        float p = scores[k];
                        o0 = fmaf(p, __bfloat162float(V_tile[k * D + d]), o0);
                        o1 = fmaf(p, __bfloat162float(V_tile[k * D + d + 1]), o1);
                        o2 = fmaf(p, __bfloat162float(V_tile[k * D + d + 2]), o2);
                        o3 = fmaf(p, __bfloat162float(V_tile[k * D + d + 3]), o3);
                    }
                }
                O_acc[tid * D + d] = o0;
                O_acc[tid * D + d + 1] = o1;
                O_acc[tid * D + d + 2] = o2;
                O_acc[tid * D + d + 3] = o3;
            }

            max_score = new_max;
        }
        __syncthreads();
    }

    // Finalize: normalize and write output
    if (q_idx < S) {
        float inv_sum = 1.0f / sum_exp;
        __nv_bfloat16* O_row = O_base + q_idx * D;

        #pragma unroll
        for (int d = 0; d < D; d += 4) {
            O_row[d]     = __float2bfloat16(O_acc[tid * D + d]     * inv_sum);
            O_row[d + 1] = __float2bfloat16(O_acc[tid * D + d + 1] * inv_sum);
            O_row[d + 2] = __float2bfloat16(O_acc[tid * D + d + 2] * inv_sum);
            O_row[d + 3] = __float2bfloat16(O_acc[tid * D + d + 3] * inv_sum);
        }

        LSE_base[q_idx] = max_score + __logf(sum_exp);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);

    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    int num_q_blocks = (S + BM - 1) / BM;
    dim3 grid((int)B, (int)H, num_q_blocks);
    dim3 block(THREADS);

    // Shared memory: O_acc (64KB) + Q_tile (32KB) + K_tile (16KB) + V_tile (16KB) = 128KB
    size_t smem_size = BM * D * sizeof(float) + (BM + 2 * BK) * D * sizeof(__nv_bfloat16);

    cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size);

    attention_kernel<<<grid, block, smem_size, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data, (int)B, (int)H, (int)S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flash_attn::run);

}  // namespace flash_attn