#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace attention_kernel {

constexpr int Br = 128;
constexpr int Bc = 32;
constexpr int D = 128;

__global__ __launch_bounds__(128, 2)
void attention_fwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S) {

    const float scale = 0.08838834764831840f; // 1/sqrt(128)

    int bh = blockIdx.x;
    int q_block = blockIdx.y;
    int q_start = q_block * Br;
    int tid = threadIdx.x;

    int b = bh / H;
    int h = bh % H;

    int64_t offset = (int64_t)b * H * S * D + (int64_t)h * S * D;
    const __nv_bfloat16* Q_base = Q + offset;
    const __nv_bfloat16* K_base = K + offset;
    const __nv_bfloat16* V_base = V + offset;
    __nv_bfloat16* O_base = O + offset;
    float* LSE_base = LSE + (int64_t)b * H * S + (int64_t)h * S;

    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* Q_tile = smem;
    __nv_bfloat16* K_tile = smem + Br * D;
    __nv_bfloat16* V_tile = smem + Br * D + Bc * D;

    int q_idx = q_start + tid;

    // Load Q tile: each thread loads one row (128 bf16 = 4 x uint4)
    if (q_idx < S) {
        const uint4* src = reinterpret_cast<const uint4*>(&Q_base[(int64_t)q_idx * D]);
        uint4* dst = reinterpret_cast<uint4*>(&Q_tile[tid * D]);
        dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; dst[3] = src[3];
    } else {
        uint4 zero = make_uint4(0, 0, 0, 0);
        uint4* dst = reinterpret_cast<uint4*>(&Q_tile[tid * D]);
        dst[0] = zero; dst[1] = zero; dst[2] = zero; dst[3] = zero;
    }
    __syncthreads();

    // Per-thread accumulators (in registers / local memory)
    float O_acc[D];
    float S_val[Bc];
    float m_prev = -INFINITY;
    float l_prev = 0.0f;

    #pragma unroll 8
    for (int d = 0; d < D; d++) O_acc[d] = 0.0f;

    int n_blocks = (S + Bc - 1) / Bc;

    for (int kv_block = 0; kv_block < n_blocks; kv_block++) {
        int kv_start = kv_block * Bc;

        // Cooperatively load K and V tiles
        // Bc * D / 8 = 32 * 128 / 8 = 512 uint4s, 128 threads -> 4 per thread
        for (int i = tid; i < Bc * (D / 8); i += 128) {
            int row = i / (D / 8);
            int col8 = i % (D / 8);
            int kv_idx = kv_start + row;
            int smem_off = row * D + col8 * 8;
            if (kv_idx < S) {
                reinterpret_cast<uint4*>(&K_tile[smem_off])[0] =
                    reinterpret_cast<const uint4*>(&K_base[(int64_t)kv_idx * D + col8 * 8])[0];
                reinterpret_cast<uint4*>(&V_tile[smem_off])[0] =
                    reinterpret_cast<const uint4*>(&V_base[(int64_t)kv_idx * D + col8 * 8])[0];
            } else {
                *reinterpret_cast<uint4*>(&K_tile[smem_off]) = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(&V_tile[smem_off]) = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        if (q_idx < S) {
            // S = Q @ K^T * scale
            float row_max = -INFINITY;
            #pragma unroll
            for (int j = 0; j < Bc; j++) {
                int kv_idx = kv_start + j;
                if (kv_idx >= S) {
                    S_val[j] = -INFINITY;
                    continue;
                }
                float dot_val = 0.0f;
                #pragma unroll 8
                for (int d = 0; d < D; d += 2) {
                    __nv_bfloat162 q_v = *reinterpret_cast<__nv_bfloat162*>(&Q_tile[tid * D + d]);
                    __nv_bfloat162 k_v = *reinterpret_cast<__nv_bfloat162*>(&K_tile[j * D + d]);
                    dot_val += __bfloat162float(q_v.x) * __bfloat162float(k_v.x);
                    dot_val += __bfloat162float(q_v.y) * __bfloat162float(k_v.y);
                }
                S_val[j] = dot_val * scale;
                row_max = fmaxf(row_max, S_val[j]);
            }

            // Online softmax update
            float m_new = fmaxf(m_prev, row_max);
            float alpha = expf(m_prev - m_new);

            #pragma unroll 8
            for (int d = 0; d < D; d++) O_acc[d] *= alpha;

            float row_sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < Bc; j++) {
                if (S_val[j] > -INFINITY) {
                    S_val[j] = expf(S_val[j] - m_new);
                    row_sum += S_val[j];
                } else {
                    S_val[j] = 0.0f;
                }
            }
            l_prev = l_prev * alpha + row_sum;
            m_prev = m_new;

            // O += P @ V  (iterate over j for sequential V access)
            #pragma unroll
            for (int j = 0; j < Bc; j++) {
                if (S_val[j] == 0.0f) continue;
                float p = S_val[j];
                #pragma unroll 8
                for (int d = 0; d < D; d++) {
                    O_acc[d] += p * __bfloat162float(V_tile[j * D + d]);
                }
            }
        }
        __syncthreads();
    }

    // Write output
    if (q_idx < S) {
        float inv_l = 1.0f / l_prev;
        for (int d = 0; d < D; d += 2) {
            __nv_bfloat162 o_v;
            o_v.x = __float2bfloat16(O_acc[d] * inv_l);
            o_v.y = __float2bfloat16(O_acc[d + 1] * inv_l);
            *reinterpret_cast<__nv_bfloat162*>(&O_base[(int64_t)q_idx * D + d]) = o_v;
        }
        LSE_base[q_idx] = m_prev + logf(l_prev);
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

    dim3 grid(B * H, (S + Br - 1) / Br);
    dim3 block(128);

    int smem_size = (Br * D + 2 * Bc * D) * (int)sizeof(__nv_bfloat16);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(attention_fwd_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    attention_fwd_kernel<<<grid, block, smem_size, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data, B, H, S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, attention_kernel::run);

}  // namespace attention_kernel