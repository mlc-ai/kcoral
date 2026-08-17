#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace flash_attn_impl {

constexpr int HEAD_DIM = 128;
constexpr int BR = 128;
constexpr int BC = 64;
constexpr int D_PAD = 130;          // Pad Q columns to avoid 32-way bank conflicts
constexpr int O_STRIDE = BR + 1;    // Transposed O layout [D][BR+1] for conflict-free access
constexpr float SCALE = 0.08838834764831840f;  // 1/sqrt(128)

__global__ __launch_bounds__(128, 1)
void flash_attn_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int H, int S) {

    const int bh = blockIdx.x;
    const int batch = bh / H;
    const int head = bh % H;
    const int q_block = blockIdx.y;
    const int q_start = q_block * BR;
    const int tid = threadIdx.x;
    const int q_idx = q_start + tid;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* Q_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* K_smem = Q_smem + BR * D_PAD;
    __nv_bfloat16* V_smem = K_smem + BC * HEAD_DIM;
    float* O_smem = reinterpret_cast<float*>(V_smem + BC * HEAD_DIM);
    float* m_smem = O_smem + HEAD_DIM * O_STRIDE;
    float* l_smem = m_smem + BR;

    const int64_t qkv_offset = ((int64_t)batch * H + head) * S * HEAD_DIM;

    // Load Q tile with padding
    for (int i = tid; i < BR * HEAD_DIM; i += 128) {
        int row = i / HEAD_DIM;
        int col = i % HEAD_DIM;
        int global_row = q_start + row;
        Q_smem[row * D_PAD + col] = (global_row < S)
            ? Q[qkv_offset + (int64_t)global_row * HEAD_DIM + col]
            : __float2bfloat16(0.0f);
    }
    for (int i = tid; i < BR * 2; i += 128) {
        Q_smem[(i / 2) * D_PAD + HEAD_DIM + (i % 2)] = __float2bfloat16(0.0f);
    }

    // Initialize m, l, O
    if (tid < BR) {
        m_smem[tid] = -INFINITY;
        l_smem[tid] = 0.0f;
    }
    for (int i = tid; i < HEAD_DIM * O_STRIDE; i += 128) {
        O_smem[i] = 0.0f;
    }

    __syncthreads();

    const int kv_end = min(q_start + BR, S);
    const int num_kv_blocks = (kv_end + BC - 1) / BC;

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        const int kv_start = kv_block * BC;

        // Load K and V tiles
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

            // Compute P = exp(S - new_max) and sum (reuse S_vals as P)
            float row_sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < BC; j++) {
                S_vals[j] = __expf(S_vals[j] - new_max);
                row_sum += S_vals[j];
            }

            // Rescale O in shared memory (transposed layout, no bank conflicts)
            float exp_old = (old_max > -INFINITY) ? __expf(old_max - new_max) : 0.0f;
            #pragma unroll 4
            for (int d = 0; d < HEAD_DIM; d++) {
                O_smem[d * O_STRIDE + tid] *= exp_old;
            }

            // Accumulate P @ V in chunks of 32 to limit register pressure
            // Peak registers: 64 (P) + 32 (O_chunk) = 96 data registers
            #pragma unroll
            for (int d_chunk = 0; d_chunk < 4; d_chunk++) {
                float O_chunk[32];
                #pragma unroll
                for (int d = 0; d < 32; d++) {
                    O_chunk[d] = O_smem[(d_chunk * 32 + d) * O_STRIDE + tid];
                }

                #pragma unroll
                for (int j = 0; j < BC; j++) {
                    float p = S_vals[j];
                    #pragma unroll 4
                    for (int d = 0; d < 32; d += 4) {
                        float v0 = __bfloat162float(V_smem[j * HEAD_DIM + d_chunk * 32 + d]);
                        float v1 = __bfloat162float(V_smem[j * HEAD_DIM + d_chunk * 32 + d + 1]);
                        float v2 = __bfloat162float(V_smem[j * HEAD_DIM + d_chunk * 32 + d + 2]);
                        float v3 = __bfloat162float(V_smem[j * HEAD_DIM + d_chunk * 32 + d + 3]);
                        O_chunk[d]     = fmaf(p, v0, O_chunk[d]);
                        O_chunk[d + 1] = fmaf(p, v1, O_chunk[d + 1]);
                        O_chunk[d + 2] = fmaf(p, v2, O_chunk[d + 2]);
                        O_chunk[d + 3] = fmaf(p, v3, O_chunk[d + 3]);
                    }
                }

                #pragma unroll
                for (int d = 0; d < 32; d++) {
                    O_smem[(d_chunk * 32 + d) * O_STRIDE + tid] = O_chunk[d];
                }
            }

            // Update m, l
            float old_l = l_smem[tid];
            m_smem[tid] = new_max;
            l_smem[tid] = old_l * exp_old + row_sum;
        }

        __syncthreads();
    }

    // Write output: cooperative coalesced writes
    // All 128 threads write 1 element per row → 32 threads write 64 bytes (1 cache line)
    for (int row = 0; row < BR; row++) {
        int global_row = q_start + row;
        if (global_row < S) {
            float l = l_smem[row];
            float m = m_smem[row];
            float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;

            if (tid == 0) {
                LSE[((int64_t)batch * H + head) * S + global_row] =
                    (l > 0.0f) ? (m + logf(l)) : -INFINITY;
            }

            if (tid < HEAD_DIM) {
                O[qkv_offset + (int64_t)global_row * HEAD_DIM + tid] =
                    __float2bfloat16(O_smem[tid * O_STRIDE + row] * inv_l);
            }
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    const int B = 4;
    const int H = 48;
    const int S_val = static_cast<int>(Q.size(2));

    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    const int smem_size = BR * D_PAD * sizeof(__nv_bfloat16)      // Q: 33,280
                        + BC * HEAD_DIM * sizeof(__nv_bfloat16)    // K: 16,384
                        + BC * HEAD_DIM * sizeof(__nv_bfloat16)    // V: 16,384
                        + HEAD_DIM * O_STRIDE * sizeof(float)      // O: 66,048
                        + BR * sizeof(float)                       // m: 512
                        + BR * sizeof(float);                      // l: 512
                        // Total: 133,120 bytes

    CUDA_CHECK(cudaFuncSetAttribute(flash_attn_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    dim3 grid(B * H, (S_val + BR - 1) / BR, 1);
    dim3 block(128, 1, 1);

    flash_attn_kernel<<<grid, block, smem_size, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr, H, S_val);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flash_attn_impl::run);

}  // namespace flash_attn_impl