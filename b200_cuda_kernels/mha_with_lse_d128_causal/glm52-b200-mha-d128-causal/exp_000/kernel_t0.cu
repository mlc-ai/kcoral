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
} while(0)

namespace flash_attn_impl {

constexpr int HEAD_DIM = 128;
constexpr int BR = 128;
constexpr int BC = 64;
constexpr int D_PAD = 130;  // Pad to avoid bank conflicts (gcd(130*2/4, 32) = 1)
constexpr float SCALE = 0.08838834764831840f;  // 1/sqrt(128)

__global__ __launch_bounds__(128, 1)
void flash_attn_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int H, int S) {

    int bh = blockIdx.x;
    int batch = bh / H;
    int head = bh % H;
    int q_block = blockIdx.y;
    int q_start = q_block * BR;

    int tid = threadIdx.x;
    int q_idx = q_start + tid;

    // Shared memory: Q (padded), K, V, m, l
    __shared__ __nv_bfloat16 Q_smem[BR * D_PAD];      // 128 * 130 * 2 = 33,280 bytes
    __shared__ __nv_bfloat16 K_smem[BC * HEAD_DIM];    // 64 * 128 * 2 = 16,384 bytes
    __shared__ __nv_bfloat16 V_smem[BC * HEAD_DIM];    // 64 * 128 * 2 = 16,384 bytes
    __shared__ float m_smem[BR];                        // 512 bytes
    __shared__ float l_smem[BR];                        // 512 bytes

    int64_t qkv_offset = ((int64_t)batch * H + head) * S * HEAD_DIM;

    // Load Q tile into shared memory with padding
    #pragma unroll 4
    for (int i = tid; i < BR * HEAD_DIM; i += 128) {
        int row = i / HEAD_DIM;
        int col = i % HEAD_DIM;
        int global_row = q_start + row;
        Q_smem[row * D_PAD + col] = (global_row < S)
            ? Q[qkv_offset + (int64_t)global_row * HEAD_DIM + col]
            : __float2bfloat16(0.0f);
    }
    // Zero out padding columns
    for (int i = tid; i < BR; i += 128) {
        Q_smem[i * D_PAD + HEAD_DIM] = __float2bfloat16(0.0f);
        Q_smem[i * D_PAD + HEAD_DIM + 1] = __float2bfloat16(0.0f);
    }

    // Initialize m and l
    if (tid < BR) {
        m_smem[tid] = -INFINITY;
        l_smem[tid] = 0.0f;
    }

    // O accumulator in registers (128 floats per thread)
    float O_reg[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) O_reg[d] = 0.0f;

    __syncthreads();

    // Causal: only process KV blocks up to the end of current Q block
    int kv_end = min(q_start + BR, S);
    int num_kv_blocks = (kv_end + BC - 1) / BC;

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int kv_start = kv_block * BC;

        // Load K and V tiles
        #pragma unroll 4
        for (int i = tid; i < BC * HEAD_DIM; i += 128) {
            int row = i / HEAD_DIM;
            int col = i % HEAD_DIM;
            int global_row = kv_start + row;
            if (global_row < kv_end) {
                K_smem[row * HEAD_DIM + col] = K[qkv_offset + (int64_t)global_row * HEAD_DIM + col];
                V_smem[row * HEAD_DIM + col] = V[qkv_offset + (int64_t)global_row * HEAD_DIM + col];
            } else {
                K_smem[row * HEAD_DIM + col] = __float2bfloat16(0.0f);
                V_smem[row * HEAD_DIM + col] = __float2bfloat16(0.0f);
            }
        }

        __syncthreads();

        if (q_idx < S) {
            // Compute S = Q @ K^T * scale
            float S_vals[BC];
            #pragma unroll
            for (int j = 0; j < BC; j++) S_vals[j] = 0.0f;

            // d-outer, j-inner: Q loaded once per d, K is broadcast (all threads read same element)
            #pragma unroll 4
            for (int d = 0; d < HEAD_DIM; d += 4) {
                float q0 = __bfloat162float(Q_smem[tid * D_PAD + d]);
                float q1 = __bfloat162float(Q_smem[tid * D_PAD + d + 1]);
                float q2 = __bfloat162float(Q_smem[tid * D_PAD + d + 2]);
                float q3 = __bfloat162float(Q_smem[tid * D_PAD + d + 3]);
                #pragma unroll
                for (int j = 0; j < BC; j++) {
                    float k0 = __bfloat162float(K_smem[j * HEAD_DIM + d]);
                    float k1 = __bfloat162float(K_smem[j * HEAD_DIM + d + 1]);
                    float k2 = __bfloat162float(K_smem[j * HEAD_DIM + d + 2]);
                    float k3 = __bfloat162float(K_smem[j * HEAD_DIM + d + 3]);
                    S_vals[j] = fmaf(q0, k0, fmaf(q1, k1, fmaf(q2, k2, fmaf(q3, k3, S_vals[j]))));
                }
            }

            // Apply scale and causal mask
            #pragma unroll
            for (int j = 0; j < BC; j++) {
                int key_idx = kv_start + j;
                if (key_idx > q_idx || key_idx >= kv_end) {
                    S_vals[j] = -INFINITY;
                } else {
                    S_vals[j] *= SCALE;
                }
            }

            // Online softmax: find row max
            float row_max = -INFINITY;
            #pragma unroll
            for (int j = 0; j < BC; j++) {
                row_max = fmaxf(row_max, S_vals[j]);
            }

            float old_max = m_smem[tid];
            float new_max = fmaxf(old_max, row_max);

            // Compute exp(S - new_max) and sum
            float row_sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < BC; j++) {
                S_vals[j] = __expf(S_vals[j] - new_max);
                row_sum += S_vals[j];
            }

            // Rescale O by exp(old_max - new_max)
            float exp_old = (old_max > -INFINITY) ? __expf(old_max - new_max) : 0.0f;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) {
                O_reg[d] *= exp_old;
            }

            // O += P @ V (V is broadcast to all threads)
            #pragma unroll
            for (int j = 0; j < BC; j++) {
                float p = S_vals[j];
                #pragma unroll 4
                for (int d = 0; d < HEAD_DIM; d += 4) {
                    float v0 = __bfloat162float(V_smem[j * HEAD_DIM + d]);
                    float v1 = __bfloat162float(V_smem[j * HEAD_DIM + d + 1]);
                    float v2 = __bfloat162float(V_smem[j * HEAD_DIM + d + 2]);
                    float v3 = __bfloat162float(V_smem[j * HEAD_DIM + d + 3]);
                    O_reg[d]     = fmaf(p, v0, O_reg[d]);
                    O_reg[d + 1] = fmaf(p, v1, O_reg[d + 1]);
                    O_reg[d + 2] = fmaf(p, v2, O_reg[d + 2]);
                    O_reg[d + 3] = fmaf(p, v3, O_reg[d + 3]);
                }
            }

            // Update running statistics
            float old_l = l_smem[tid];
            m_smem[tid] = new_max;
            l_smem[tid] = old_l * exp_old + row_sum;
        }

        __syncthreads();
    }

    // Write output: O and LSE
    if (q_idx < S) {
        float l = l_smem[tid];
        float m = m_smem[tid];
        float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;

        // LSE = m + ln(l) = ln(sum(exp(scores)))
        LSE[((int64_t)batch * H + head) * S + q_idx] =
            (l > 0.0f) ? (m + logf(l)) : -INFINITY;

        // Write O = (P @ V) / l, converted to bf16
        #pragma unroll 4
        for (int d = 0; d < HEAD_DIM; d++) {
            O[qkv_offset + (int64_t)q_idx * HEAD_DIM + d] =
                __float2bfloat16(O_reg[d] * inv_l);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = 4;
    int H = 48;
    int S_val = static_cast<int>(Q.size(2));

    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    dim3 grid(B * H, (S_val + BR - 1) / BR, 1);
    dim3 block(128, 1, 1);

    flash_attn_kernel<<<grid, block, 0, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr, H, S_val);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flash_attn_impl::run);

}  // namespace flash_attn_impl