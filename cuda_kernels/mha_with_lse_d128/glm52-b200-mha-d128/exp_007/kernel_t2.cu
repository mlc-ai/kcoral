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

constexpr int Br = 64;
constexpr int Bc = 32;
constexpr int D = 128;
constexpr int D_pad = 132;   // Pad bf16 arrays (multiple of 4 for uint2 alignment, avoids bank conflicts)
constexpr int D_pad_O = 132; // Pad float array (multiple of 4 for float4 alignment)
constexpr int D_chunk = 32;  // Process O accumulation in chunks to limit register usage

__global__ __launch_bounds__(64, 2)
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

    extern __shared__ char smem_raw[];
    __nv_bfloat16* Q_tile = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* K_tile = Q_tile + Br * D_pad;
    __nv_bfloat16* V_tile = K_tile + Bc * D_pad;
    float* O_smem = reinterpret_cast<float*>(V_tile + Bc * D_pad);

    int q_idx = q_start + tid;

    // Load Q tile: each thread loads one row (128 bf16 = 32 uint2s)
    if (q_idx < S) {
        for (int i = 0; i < D / 4; i++) {
            reinterpret_cast<uint2*>(&Q_tile[tid * D_pad + i * 4])[0] =
                reinterpret_cast<const uint2*>(&Q_base[(int64_t)q_idx * D + i * 4])[0];
        }
    } else {
        for (int i = 0; i < D / 4; i++) {
            *reinterpret_cast<uint2*>(&Q_tile[tid * D_pad + i * 4]) = make_uint2(0, 0);
        }
    }
    // Zero Q padding (4 elements)
    *reinterpret_cast<uint2*>(&Q_tile[tid * D_pad + D]) = make_uint2(0, 0);

    // Initialize O_smem to 0 (each thread initializes its row)
    for (int d = 0; d < D; d += 4) {
        *reinterpret_cast<float4*>(&O_smem[tid * D_pad_O + d]) =
            make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    *reinterpret_cast<float4*>(&O_smem[tid * D_pad_O + D]) =
        make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    __syncthreads();

    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float S_val[Bc]; // 32 floats in registers

    int n_blocks = (S + Bc - 1) / Bc;

    for (int kv_block = 0; kv_block < n_blocks; kv_block++) {
        int kv_start = kv_block * Bc;

        // Cooperatively load K, V tiles (all 64 threads participate)
        // Bc * (D/4) = 32 * 32 = 1024 uint2 loads, 64 threads -> 16 per thread
        for (int i = tid; i < Bc * (D / 4); i += 64) {
            int row = i / (D / 4);
            int col4 = i % (D / 4);
            int kv_idx = kv_start + row;
            int smem_off = row * D_pad + col4 * 4;
            if (kv_idx < S) {
                reinterpret_cast<uint2*>(&K_tile[smem_off])[0] =
                    reinterpret_cast<const uint2*>(&K_base[(int64_t)kv_idx * D + col4 * 4])[0];
                reinterpret_cast<uint2*>(&V_tile[smem_off])[0] =
                    reinterpret_cast<const uint2*>(&V_base[(int64_t)kv_idx * D + col4 * 4])[0];
            } else {
                *reinterpret_cast<uint2*>(&K_tile[smem_off]) = make_uint2(0, 0);
                *reinterpret_cast<uint2*>(&V_tile[smem_off]) = make_uint2(0, 0);
            }
        }
        // Zero K, V padding (4 elements per row, Bc rows)
        for (int i = tid; i < Bc; i += 64) {
            *reinterpret_cast<uint2*>(&K_tile[i * D_pad + D]) = make_uint2(0, 0);
            *reinterpret_cast<uint2*>(&V_tile[i * D_pad + D]) = make_uint2(0, 0);
        }
        __syncthreads();

        if (q_idx < S) {
            // Compute S = Q @ K^T * scale
            float row_max = -INFINITY;
            #pragma unroll 4
            for (int j = 0; j < Bc; j++) {
                int kv_idx_j = kv_start + j;
                if (kv_idx_j >= S) {
                    S_val[j] = -INFINITY;
                    continue;
                }
                float dot_val = 0.0f;
                #pragma unroll 8
                for (int d = 0; d < D; d += 2) {
                    __nv_bfloat162 q_v = *reinterpret_cast<__nv_bfloat162*>(&Q_tile[tid * D_pad + d]);
                    __nv_bfloat162 k_v = *reinterpret_cast<__nv_bfloat162*>(&K_tile[j * D_pad + d]);
                    dot_val += __bfloat162float(q_v.x) * __bfloat162float(k_v.x);
                    dot_val += __bfloat162float(q_v.y) * __bfloat162float(k_v.y);
                }
                S_val[j] = dot_val * scale;
                row_max = fmaxf(row_max, S_val[j]);
            }

            // Online softmax update
            float m_new = fmaxf(m_prev, row_max);
            float alpha = __expf(m_prev - m_new);

            // Rescale O in shared memory
            for (int d = 0; d < D; d += 4) {
                float4 o_v = *reinterpret_cast<float4*>(&O_smem[tid * D_pad_O + d]);
                o_v.x *= alpha; o_v.y *= alpha; o_v.z *= alpha; o_v.w *= alpha;
                *reinterpret_cast<float4*>(&O_smem[tid * D_pad_O + d]) = o_v;
            }

            // Compute P and row_sum
            float row_sum = 0.0f;
            #pragma unroll 4
            for (int j = 0; j < Bc; j++) {
                if (S_val[j] > -INFINITY) {
                    S_val[j] = __expf(S_val[j] - m_new);
                    row_sum += S_val[j];
                } else {
                    S_val[j] = 0.0f;
                }
            }
            l_prev = l_prev * alpha + row_sum;
            m_prev = m_new;

            // O += P @ V (process D in chunks of 32 to limit register usage)
            for (int dc = 0; dc < D; dc += D_chunk) {
                float O_chunk[D_chunk];
                // Load current O values from shared memory
                for (int d = 0; d < D_chunk; d += 4) {
                    float4 v = *reinterpret_cast<float4*>(&O_smem[tid * D_pad_O + dc + d]);
                    O_chunk[d] = v.x; O_chunk[d+1] = v.y;
                    O_chunk[d+2] = v.z; O_chunk[d+3] = v.w;
                }

                // Accumulate P @ V for this D chunk
                #pragma unroll 4
                for (int j = 0; j < Bc; j++) {
                    float p = S_val[j];
                    if (p == 0.0f) continue;
                    #pragma unroll 4
                    for (int d = 0; d < D_chunk; d += 2) {
                        __nv_bfloat162 v_v = *reinterpret_cast<__nv_bfloat162*>(&V_tile[j * D_pad + dc + d]);
                        O_chunk[d]   += p * __bfloat162float(v_v.x);
                        O_chunk[d+1] += p * __bfloat162float(v_v.y);
                    }
                }

                // Store updated O values back to shared memory
                for (int d = 0; d < D_chunk; d += 4) {
                    float4 v = make_float4(O_chunk[d], O_chunk[d+1], O_chunk[d+2], O_chunk[d+3]);
                    *reinterpret_cast<float4*>(&O_smem[tid * D_pad_O + dc + d]) = v;
                }
            }
        }
        __syncthreads();
    }

    // Write output: O and LSE
    if (q_idx < S) {
        float inv_l = 1.0f / l_prev;
        for (int d = 0; d < D; d += 4) {
            float4 o_v = *reinterpret_cast<float4*>(&O_smem[tid * D_pad_O + d]);
            __nv_bfloat162 o0, o1;
            o0.x = __float2bfloat16(o_v.x * inv_l);
            o0.y = __float2bfloat16(o_v.y * inv_l);
            o1.x = __float2bfloat16(o_v.z * inv_l);
            o1.y = __float2bfloat16(o_v.w * inv_l);
            *reinterpret_cast<__nv_bfloat162*>(&O_base[(int64_t)q_idx * D + d]) = o0;
            *reinterpret_cast<__nv_bfloat162*>(&O_base[(int64_t)q_idx * D + d + 2]) = o1;
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
    dim3 block(64);

    int smem_size = (Br * D_pad + 2 * Bc * D_pad) * (int)sizeof(__nv_bfloat16)
                  + Br * D_pad_O * (int)sizeof(float);

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