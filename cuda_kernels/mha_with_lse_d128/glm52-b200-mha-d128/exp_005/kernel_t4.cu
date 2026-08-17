#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
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
constexpr int THREADS = 128;
constexpr float SCALE = 0.08838834764831845f;

__global__ __launch_bounds__(THREADS, 1)
void attention_kernel(
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
    const int tid     = threadIdx.x;
    const int row     = tid;
    const int m       = m_start + row;

    extern __shared__ char smem_raw[];
    // Pad sO rows by 1 float to avoid bank conflicts
    float* sO          = reinterpret_cast<float*>(smem_raw);                    // [BM, D+1]
    __nv_bfloat16* sQ  = reinterpret_cast<__nv_bfloat16*>(sO + BM * (D + 1));   // [BM, D]
    __nv_bfloat16* sK  = sQ + BM * D;                                            // [BN, D]
    __nv_bfloat16* sV  = sK + BN * D;                                            // [BN, D]

    const int64_t base = ((int64_t)b * H + h) * S * D;

    // Initialize sO to 0
    for (int i = tid; i < BM * (D + 1); i += THREADS) {
        sO[i] = 0.0f;
    }

    // Load Q tile: each thread loads one row (128 bf16 = 256 bytes = 4 x int4)
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

    float my_rowmax = -INFINITY;
    float my_rowsum = 0.0f;

    __syncthreads();

    for (int kv_start = 0; kv_start < S; kv_start += BN) {
        // Load K and V tiles cooperatively
        for (int i = tid; i < BN; i += THREADS) {
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
            // Compute S[n] = dot(Q[row], K[n]) * scale for n=0..BN-1
            float s[BN];
            #pragma unroll
            for (int n = 0; n < BN; n++) s[n] = 0.0f;

            #pragma unroll 4
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

            #pragma unroll
            for (int n = 0; n < BN; n++) s[n] *= SCALE;

            // Mask out-of-bounds
            int kv_end = min(kv_start + BN, S);
            #pragma unroll
            for (int n = 0; n < BN; n++) {
                if (kv_start + n >= kv_end) s[n] = -INFINITY;
            }

            // Online softmax
            float old_max = my_rowmax;
            float new_max = old_max;
            #pragma unroll
            for (int n = 0; n < BN; n++) {
                new_max = fmaxf(new_max, s[n]);
            }

            float alpha = expf(old_max - new_max);
            float p_sum = 0.0f;

            // Rescale sO by alpha
            #pragma unroll 4
            for (int d = 0; d < D; d++) {
                sO[row * (D + 1) + d] *= alpha;
            }

            // Compute P and accumulate O += P @ V
            #pragma unroll
            for (int n = 0; n < BN; n++) {
                float p = expf(s[n] - new_max);
                p_sum += p;
                #pragma unroll 4
                for (int d = 0; d < D; d += 2) {
                    __nv_bfloat162 vv = *reinterpret_cast<__nv_bfloat162*>(&sV[n * D + d]);
                    float2 vf = __bfloat1622float2(vv);
                    sO[row * (D + 1) + d]     += p * vf.x;
                    sO[row * (D + 1) + d + 1] += p * vf.y;
                }
            }

            my_rowmax = new_max;
            my_rowsum = my_rowsum * alpha + p_sum;
        }
        __syncthreads();
    }

    // Final normalization and store
    if (m < S) {
        float inv_sum = 1.0f / my_rowsum;
        #pragma unroll 4
        for (int d = 0; d < D; d += 2) {
            __nv_bfloat162 val;
            val.x = __float2bfloat16(sO[row * (D + 1) + d]     * inv_sum);
            val.y = __float2bfloat16(sO[row * (D + 1) + d + 1] * inv_sum);
            *reinterpret_cast<__nv_bfloat162*>(&O[base + (int64_t)m * D + d]) = val;
        }
        LSE[((int64_t)b * H + h) * S + m] = my_rowmax + logf(my_rowsum);
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
    dim3 block(THREADS);

    size_t smem_size =
        (size_t)(BM * (D + 1)) * sizeof(float)            // sO (padded)
      + (size_t)(BM * D) * sizeof(__nv_bfloat16)           // sQ
      + (size_t)(BN * D) * sizeof(__nv_bfloat16) * 2;      // sK + sV

    CUDA_CHECK(cudaFuncSetAttribute(
        attention_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem_size));

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    attention_kernel<<<grid, block, smem_size, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data, B, H, S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, attn_fwd::run);

} // namespace attn_fwd