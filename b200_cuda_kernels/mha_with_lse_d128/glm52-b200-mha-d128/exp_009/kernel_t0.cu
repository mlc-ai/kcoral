#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
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

namespace mha_cuda {

constexpr int D = 128;
constexpr int D_PAD = 132;  // Padded for 2-way bank conflict avoidance
constexpr int BQ = 128;     // Query tile size
constexpr int BK = 64;      // Key tile size

__device__ __forceinline__ float fast_expf(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x * 1.4426950408889634f));
    return y;
}

__global__ __launch_bounds__(128, 2)
void attention_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int H, int S
) {
    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* Q_tile = smem;                    // [BQ, D_PAD] bf16
    __nv_bfloat16* KV_tile = Q_tile + BQ * D_PAD;    // [BK, D] bf16 (reused for K then V)

    int q_start = blockIdx.x * BQ;
    int bh = blockIdx.y;
    int b = bh / H;
    int h = bh % H;
    int tid = threadIdx.x;
    int q_idx = q_start + tid;
    bool valid = (q_idx < S);

    int64_t base = (int64_t)(b * H + h) * S * D;
    const __nv_bfloat16* Q_bh = Q + base;
    const __nv_bfloat16* K_bh = K + base;
    const __nv_bfloat16* V_bh = V + base;

    // Load Q_tile [BQ, D] -> padded [BQ, D_PAD] using int2 (8-byte aligned)
    {
        int total = BQ * (D / 4);  // 128 * 32 = 4096 int2 elements
        for (int i = tid; i < total; i += 128) {
            int row = i / (D / 4);
            int col = i % (D / 4);
            int q_row = q_start + row;
            int2* dst = reinterpret_cast<int2*>(Q_tile + row * D_PAD + col * 4);
            if (q_row < S) {
                const int2* src = reinterpret_cast<const int2*>(Q_bh + q_row * D + col * 4);
                *dst = *src;
            } else {
                *dst = make_int2(0, 0);
            }
        }
    }
    __syncthreads();

    // Online softmax state in registers
    float m = -INFINITY;
    float l = 0.0f;
    float o[D];
    #pragma unroll
    for (int d = 0; d < D; d++) o[d] = 0.0f;

    // Iterate over key tiles
    for (int k_start = 0; k_start < S; k_start += BK) {
        int actual_bk = min(BK, S - k_start);

        // Load K_tile [BK, D] using int4 (16-byte aligned, no padding needed)
        {
            int total = BK * (D / 8);  // 64 * 16 = 1024 int4
            for (int i = tid; i < total; i += 128) {
                int row = i / (D / 8);
                int col = i % (D / 8);
                int k_row = k_start + row;
                int4* dst = reinterpret_cast<int4*>(KV_tile + row * D + col * 8);
                if (k_row < S) {
                    const int4* src = reinterpret_cast<const int4*>(K_bh + k_row * D + col * 8);
                    *dst = *src;
                } else {
                    *dst = make_int4(0, 0, 0, 0);
                }
            }
        }
        __syncthreads();

        // Compute scores: scores[j] = dot(Q[tid], K[j]) / sqrt(D)
        float scores[BK];
        if (valid) {
            for (int j = 0; j < BK; j++) {
                float dot = 0.0f;
                #pragma unroll
                for (int d = 0; d < D; d += 2) {
                    __nv_bfloat162 q2 = *reinterpret_cast<__nv_bfloat162*>(Q_tile + tid * D_PAD + d);
                    __nv_bfloat162 k2 = *reinterpret_cast<__nv_bfloat162*>(KV_tile + j * D + d);
                    float2 qf = __bfloat1622float2(q2);
                    float2 kf = __bfloat1622float2(k2);
                    dot = __fmaf_rn(qf.x, kf.x, dot);
                    dot = __fmaf_rn(qf.y, kf.y, dot);
                }
                scores[j] = dot * 0.08838834764831845f;  // 1/sqrt(128)
            }
        }
        __syncthreads();

        // Load V_tile [BK, D] (overwrite K_tile buffer)
        {
            int total = BK * (D / 8);
            for (int i = tid; i < total; i += 128) {
                int row = i / (D / 8);
                int col = i % (D / 8);
                int k_row = k_start + row;
                int4* dst = reinterpret_cast<int4*>(KV_tile + row * D + col * 8);
                if (k_row < S) {
                    const int4* src = reinterpret_cast<const int4*>(V_bh + k_row * D + col * 8);
                    *dst = *src;
                } else {
                    *dst = make_int4(0, 0, 0, 0);
                }
            }
        }
        __syncthreads();

        if (valid) {
            // Online softmax: find new max
            float m_new = m;
            for (int j = 0; j < actual_bk; j++) {
                m_new = fmaxf(m_new, scores[j]);
            }

            // Rescale running statistics
            float scale = (m > -INFINITY) ? fast_expf(m - m_new) : 0.0f;
            l *= scale;
            #pragma unroll
            for (int d = 0; d < D; d++) o[d] *= scale;

            // Compute softmax weights and update l
            for (int j = 0; j < BK; j++) {
                if (j >= actual_bk) {
                    scores[j] = 0.0f;
                } else {
                    float p = fast_expf(scores[j] - m_new);
                    scores[j] = p;
                    l += p;
                }
            }
            m = m_new;

            // Accumulate O += P @ V
            for (int j = 0; j < BK; j++) {
                float p = scores[j];
                if (p == 0.0f) continue;
                #pragma unroll
                for (int d = 0; d < D; d += 2) {
                    __nv_bfloat162 v2 = *reinterpret_cast<__nv_bfloat162*>(KV_tile + j * D + d);
                    float2 vf = __bfloat1622float2(v2);
                    o[d]   = __fmaf_rn(p, vf.x, o[d]);
                    o[d+1] = __fmaf_rn(p, vf.y, o[d+1]);
                }
            }
        }
        __syncthreads();
    }

    // Write output: O = o / l, LSE = m + log(l)
    if (valid) {
        float inv_l = 1.0f / l;
        __nv_bfloat16* O_out = O + (int64_t)(b * H + h) * S * D + (int64_t)q_idx * D;
        #pragma unroll
        for (int d = 0; d < D; d += 2) {
            __nv_bfloat162 o2;
            o2.x = __float2bfloat16(o[d] * inv_l);
            o2.y = __float2bfloat16(o[d+1] * inv_l);
            *reinterpret_cast<__nv_bfloat162*>(O_out + d) = o2;
        }
        LSE[(int64_t)(b * H + h) * S + q_idx] = m + logf(l);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    const int B = 4;
    const int H = 48;
    int S = static_cast<int>(Q.size(2));

    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    dim3 grid((S + BQ - 1) / BQ, B * H);
    dim3 block(128);

    size_t smem_size = (BQ * D_PAD + BK * D) * sizeof(__nv_bfloat16);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(attention_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_size)));

    attention_kernel<<<grid, block, smem_size, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data, H, S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_cuda::run);

}  // namespace mha_cuda