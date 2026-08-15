#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                   \
    cudaError_t _e = (call);                                     \
    if (_e != cudaSuccess) {                                     \
        fprintf(stderr, "CUDA error %s at %s:%d\n",              \
                cudaGetErrorString(_e), __FILE__, __LINE__);      \
        exit(1);                                                  \
    }                                                             \
} while(0)

namespace attn_fwd {

constexpr int BM = 128;
constexpr int BN = 64;
constexpr int D  = 128;
constexpr float SCALE = 0.08838834764831845f;  // 1.0f / sqrtf(128.0f)

__global__ __launch_bounds__(128, 1)
void attention_fwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S)
{
    const int bh      = blockIdx.x;
    const int b       = bh / H;
    const int h       = bh % H;
    const int q_block = blockIdx.y;
    const int m_start = q_block * BM;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sK = sQ + BM * D;
    __nv_bfloat16* sV = sK + BN * D;

    const int tid = threadIdx.x;
    const int row = tid;
    const int m   = m_start + row;

    const int64_t base = ((int64_t)b * H + h) * (int64_t)S * D;

    // ---- Load Q tile (each thread loads one row: 128 bf16 = 256 B = 4×int4) ----
    if (m < S) {
        #pragma unroll
        for (int d = 0; d < D; d += 8) {
            *reinterpret_cast<int4*>(&sQ[row * D + d]) =
                *reinterpret_cast<const int4*>(&Q[base + (int64_t)m * D + d]);
        }
    } else {
        #pragma unroll
        for (int d = 0; d < D; d += 8) {
            *reinterpret_cast<int4*>(&sQ[row * D + d]) = make_int4(0, 0, 0, 0);
        }
    }

    // ---- O accumulator in registers ----
    float o[D];
    #pragma unroll
    for (int d = 0; d < D; d++) o[d] = 0.0f;

    float rowmax = -INFINITY;
    float rowsum = 0.0f;

    __syncthreads();

    // ---- Iterate over KV tiles ----
    for (int kv_start = 0; kv_start < S; kv_start += BN) {
        // Load K and V tiles (first BN threads each load one row)
        for (int i = tid; i < BN; i += BM) {
            int kv = kv_start + i;
            if (kv < S) {
                #pragma unroll
                for (int d = 0; d < D; d += 8) {
                    *reinterpret_cast<int4*>(&sK[i * D + d]) =
                        *reinterpret_cast<const int4*>(&K[base + (int64_t)kv * D + d]);
                    *reinterpret_cast<int4*>(&sV[i * D + d]) =
                        *reinterpret_cast<const int4*>(&V[base + (int64_t)kv * D + d]);
                }
            } else {
                #pragma unroll
                for (int d = 0; d < D; d += 8) {
                    *reinterpret_cast<int4*>(&sK[i * D + d]) = make_int4(0, 0, 0, 0);
                    *reinterpret_cast<int4*>(&sV[i * D + d]) = make_int4(0, 0, 0, 0);
                }
            }
        }
        __syncthreads();

        if (m < S) {
            // ---- Compute S = Q @ K^T (1×BN dot products of length D) ----
            float s[BN];
            #pragma unroll
            for (int n = 0; n < BN; n++) s[n] = 0.0f;

            #pragma unroll 8
            for (int d = 0; d < D; d += 2) {
                __nv_bfloat162 qv = *reinterpret_cast<__nv_bfloat162*>(&sQ[row * D + d]);
                float2 qf = __bfloat1622float2(qv);
                #pragma unroll
                for (int n = 0; n < BN; n++) {
                    __nv_bfloat162 kv = *reinterpret_cast<__nv_bfloat162*>(&sK[n * D + d]);
                    float2 kf = __bfloat1622float2(kv);
                    s[n] += qf.x * kf.x + qf.y * kf.y;
                }
            }

            // Apply scale
            #pragma unroll
            for (int n = 0; n < BN; n++) s[n] *= SCALE;

            // Mask out-of-bounds KV positions
            int kv_end = min(kv_start + BN, S);
            #pragma unroll
            for (int n = 0; n < BN; n++) {
                if (kv_start + n >= kv_end) s[n] = -INFINITY;
            }

            // ---- Online softmax ----
            float new_max = rowmax;
            #pragma unroll
            for (int n = 0; n < BN; n++) {
                new_max = fmaxf(new_max, s[n]);
            }

            float alpha = __expf(rowmax - new_max);
            float p_sum = 0.0f;
            #pragma unroll
            for (int n = 0; n < BN; n++) {
                float p = __expf(s[n] - new_max);
                s[n] = p;
                p_sum += p;
            }

            // Rescale O
            #pragma unroll
            for (int d = 0; d < D; d++) o[d] *= alpha;

            rowsum = rowsum * alpha + p_sum;
            rowmax = new_max;

            // ---- O += P @ V ----
            #pragma unroll
            for (int n = 0; n < BN; n++) {
                float p = s[n];
                if (p > 0.0f) {
                    #pragma unroll 8
                    for (int d = 0; d < D; d += 2) {
                        __nv_bfloat162 vv = *reinterpret_cast<__nv_bfloat162*>(&sV[n * D + d]);
                        float2 vf = __bfloat1622float2(vv);
                        o[d]     += p * vf.x;
                        o[d + 1] += p * vf.y;
                    }
                }
            }
        }
        __syncthreads();
    }

    // ---- Final normalization and store ----
    if (m < S) {
        float inv_sum = 1.0f / rowsum;
        #pragma unroll
        for (int d = 0; d < D; d += 2) {
            __nv_bfloat162 val;
            val.x = __float2bfloat16(o[d]     * inv_sum);
            val.y = __float2bfloat16(o[d + 1] * inv_sum);
            *reinterpret_cast<__nv_bfloat162*>(&O[base + (int64_t)m * D + d]) = val;
        }
        // LSE = rowmax + log(rowsum)  (natural log)
        LSE[((int64_t)b * H + h) * S + m] = rowmax + __logf(rowsum);
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
    __nv_bfloat16* O_data       = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data             = static_cast<float*>(LSE.data_ptr());

    const int num_q_blocks = (S + BM - 1) / BM;
    dim3 grid(B * H, num_q_blocks);
    dim3 block(BM);

    size_t smem_size = (size_t)(BM * D + 2 * BN * D) * sizeof(__nv_bfloat16);

    CUDA_CHECK(cudaFuncSetAttribute(
        attention_fwd_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem_size));

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    attention_fwd_kernel<<<grid, block, smem_size, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data, B, H, S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, attn_fwd::run);

} // namespace attn_fwd