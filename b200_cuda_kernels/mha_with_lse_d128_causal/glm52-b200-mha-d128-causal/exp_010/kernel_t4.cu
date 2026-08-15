#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <cmath>
#include <stdio.h>
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

namespace mha_causal_d128 {

constexpr int D = 128;
constexpr int Br = 16;
constexpr int Bc = 64;
constexpr int THREADS = 128;  // 4 warps

__global__ void mha_causal_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S,
    float scale)
{
    int batch = blockIdx.x;
    int head  = blockIdx.y;
    int q_block = blockIdx.z;
    int q_start = q_block * Br;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    __shared__ __nv_bfloat16 Q_smem[Br * D];
    __shared__ __nv_bfloat16 K_smem[Bc * D];
    __shared__ __nv_bfloat16 V_smem[Bc * D];
    __shared__ float S_smem[Br * Bc];

    const __nv_bfloat16* Q_gptr = Q + ((batch * 48 + head) * S + q_start) * D;
    const __nv_bfloat16* K_base = K + (batch * 48 + head) * S * D;
    const __nv_bfloat16* V_base = V + (batch * 48 + head) * S * D;
    __nv_bfloat16* O_gptr = O + ((batch * 48 + head) * S + q_start) * D;
    float* LSE_gptr = LSE + (batch * 48 + head) * S + q_start;

    // Load Q tile
    for (int c = tid; c < (Br * D) / 8; c += THREADS) {
        int offset = c * 8;
        int row = offset / D;
        int qg = q_start + row;
        if (qg < S) {
            *reinterpret_cast<uint4*>(Q_smem + offset) =
                *reinterpret_cast<const uint4*>(Q_gptr + offset);
        } else {
            *reinterpret_cast<uint4*>(Q_smem + offset) = make_uint4(0, 0, 0, 0);
        }
    }
    __syncthreads();

    // Process query rows in groups of 4 (one per warp)
    for (int q_sub = 0; q_sub < Br / 4; ++q_sub) {
        int q_row = q_sub * 4 + warp_id;
        int q_idx = q_start + q_row;
        bool valid = (q_idx < S);

        // Load Q row into registers
        float q0 = 0.f, q1 = 0.f, q2 = 0.f, q3 = 0.f;
        if (valid) {
            int d_base = lane_id * 4;
            q0 = __bfloat162float(Q_smem[q_row * D + d_base + 0]);
            q1 = __bfloat162float(Q_smem[q_row * D + d_base + 1]);
            q2 = __bfloat162float(Q_smem[q_row * D + d_base + 2]);
            q3 = __bfloat162float(Q_smem[q_row * D + d_base + 3]);
        }

        float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
        float m = -INFINITY;
        float l = 0.f;

        int max_kv_block = valid ? (q_idx / Bc) : -1;

        for (int kv = 0; kv <= max_kv_block; ++kv) {
            int kv_start = kv * Bc;

            // Load K and V tiles (collaborative across all threads)
            for (int c = tid; c < (Bc * D) / 8; c += THREADS) {
                int offset = c * 8;
                int row = offset / D;
                int col_offset = offset % D;
                int kg = kv_start + row;
                if (kg < S) {
                    *reinterpret_cast<uint4*>(K_smem + offset) =
                        *reinterpret_cast<const uint4*>(K_base + kg * D + col_offset);
                    *reinterpret_cast<uint4*>(V_smem + offset) =
                        *reinterpret_cast<const uint4*>(V_base + kg * D + col_offset);
                } else {
                    *reinterpret_cast<uint4*>(K_smem + offset) = make_uint4(0, 0, 0, 0);
                    *reinterpret_cast<uint4*>(V_smem + offset) = make_uint4(0, 0, 0, 0);
                }
            }
            __syncthreads();

            if (valid) {
                // Compute scores
                for (int j = 0; j < Bc; ++j) {
                    int k_idx = kv_start + j;
                    float s = 0.f;
                    if (k_idx <= q_idx && k_idx < S) {
                        int d_base = lane_id * 4;
                        float k0 = __bfloat162float(K_smem[j * D + d_base + 0]);
                        float k1 = __bfloat162float(K_smem[j * D + d_base + 1]);
                        float k2 = __bfloat162float(K_smem[j * D + d_base + 2]);
                        float k3 = __bfloat162float(K_smem[j * D + d_base + 3]);
                        s = q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;
                        for (int off = 16; off > 0; off >>= 1)
                            s += __shfl_down_sync(0xffffffff, s, off);
                    }
                    s = __shfl_sync(0xffffffff, s, 0);
                    if (k_idx > q_idx || k_idx >= S) s = -INFINITY;
                    s *= scale;
                    if (lane_id == 0)
                        S_smem[q_row * Bc + j] = s;
                }
                __syncwarp();

                // Online softmax: find max
                float m_new = m;
                for (int j = 0; j < Bc; ++j)
                    m_new = fmaxf(m_new, S_smem[q_row * Bc + j]);

                float alpha = (m > -INFINITY) ? __expf(m - m_new) : 1.f;
                l = alpha * l;
                o0 = alpha * o0;
                o1 = alpha * o1;
                o2 = alpha * o2;
                o3 = alpha * o3;

                // Accumulate weighted V
                for (int j = 0; j < Bc; ++j) {
                    float p = __expf(S_smem[q_row * Bc + j] - m_new);
                    l += p;
                    int d_base = lane_id * 4;
                    float v0 = __bfloat162float(V_smem[j * D + d_base + 0]);
                    float v1 = __bfloat162float(V_smem[j * D + d_base + 1]);
                    float v2 = __bfloat162float(V_smem[j * D + d_base + 2]);
                    float v3 = __bfloat162float(V_smem[j * D + d_base + 3]);
                    o0 += p * v0;
                    o1 += p * v1;
                    o2 += p * v2;
                    o3 += p * v3;
                }
                m = m_new;
            }
            __syncthreads();
        }

        if (valid) {
            float inv_l = (l > 0.f) ? 1.f / l : 0.f;
            o0 *= inv_l; o1 *= inv_l; o2 *= inv_l; o3 *= inv_l;
            int d_base = lane_id * 4;
            O_gptr[q_row * D + d_base + 0] = __float2bfloat16(o0);
            O_gptr[q_row * D + d_base + 1] = __float2bfloat16(o1);
            O_gptr[q_row * D + d_base + 2] = __float2bfloat16(o2);
            O_gptr[q_row * D + d_base + 3] = __float2bfloat16(o3);
            if (lane_id == 0)
                LSE_gptr[q_row] = m + logf(l);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = 4;
    int H = 48;
    int D_val = 128;
    int S = static_cast<int>(Q.size(2));

    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE.data_ptr());

    float scale = 1.f / sqrtf((float)D_val);

    dim3 grid(B, H, (S + Br - 1) / Br);
    dim3 block(THREADS);

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    mha_causal_kernel<<<grid, block, 0, stream>>>(Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr, S, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_causal_d128::run);

}  // namespace mha_causal_d128