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
} while (0)

namespace mha_cuda {

constexpr int D = 128;
constexpr int D_PAD = 136;  // Padded to avoid bank conflicts and maintain 16B alignment
constexpr int BQ = 128;
constexpr int BK = 64;
constexpr int NUM_THREADS = 128;
constexpr float SCALE = 0.08838834764831845f;  // 1/sqrt(128)
constexpr float LOG2E = 1.4426950408889634f;

__device__ __forceinline__ float fast_expf(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x * LOG2E));
    return y;
}

__device__ __forceinline__ void cp_async_16(uint32_t smem_addr, const void* gmem_addr) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem_addr));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__global__ __launch_bounds__(NUM_THREADS, 2)
void attention_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int H, int S
) {
    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* Q_smem = smem;
    __nv_bfloat16* KV_smem[3];
    KV_smem[0] = Q_smem + BQ * D_PAD;
    KV_smem[1] = KV_smem[0] + BK * D;
    KV_smem[2] = KV_smem[1] + BK * D;

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

    // Load Q to Q_smem [BQ, D_PAD] row-major
    {
        int total = BQ * (D / 8);
        for (int i = tid; i < total; i += NUM_THREADS) {
            int row = i / (D / 8), col = i % (D / 8);
            int q_row = q_start + row;
            int4* dst = reinterpret_cast<int4*>(Q_smem + row * D_PAD + col * 8);
            if (q_row < S) {
                *dst = *reinterpret_cast<const int4*>(Q_bh + (int64_t)q_row * D + col * 8);
            } else {
                *dst = make_int4(0, 0, 0, 0);
            }
        }
    }
    __syncthreads();

    // Prologue: load K_0 into KV_smem[0]
    {
        int total = BK * (D / 8);
        for (int i = tid; i < total; i += NUM_THREADS) {
            int row = i / (D / 8), col = i % (D / 8);
            int k_row = 0 + row;
            if (k_row < S) {
                uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(
                    KV_smem[0] + row * D + col * 8);
                const void* gmem_addr = K_bh + (int64_t)k_row * D + col * 8;
                cp_async_16(smem_addr, gmem_addr);
            }
        }
        cp_async_commit();
    }

    float m = -INFINITY;
    float l = 0.0f;
    float o[D];
    #pragma unroll
    for (int d = 0; d < D; d++) o[d] = 0.0f;

    int k_buf_idx = 0;
    int v_buf_idx = 1;
    int next_k_buf_idx = 2;

    for (int k_start = 0; k_start < S; k_start += BK) {
        __nv_bfloat16* K_buf = KV_smem[k_buf_idx];
        __nv_bfloat16* V_buf = KV_smem[v_buf_idx];
        __nv_bfloat16* next_K_buf = KV_smem[next_k_buf_idx];

        // Wait for K_i
        if (k_start == 0) {
            cp_async_wait_group<0>();
        } else {
            cp_async_wait_group<1>();
        }
        __syncthreads();

        // Issue V_i into V_buf
        {
            int total = BK * (D / 8);
            for (int i = tid; i < total; i += NUM_THREADS) {
                int row = i / (D / 8), col = i % (D / 8);
                int k_row = k_start + row;
                if (k_row < S) {
                    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(
                        V_buf + row * D + col * 8);
                    const void* gmem_addr = V_bh + (int64_t)k_row * D + col * 8;
                    cp_async_16(smem_addr, gmem_addr);
                }
            }
            cp_async_commit();
        }

        // Issue K_{i+1} into next_K_buf
        bool has_next = (k_start + BK < S);
        if (has_next) {
            int next_k_start = k_start + BK;
            int total = BK * (D / 8);
            for (int i = tid; i < total; i += NUM_THREADS) {
                int row = i / (D / 8), col = i % (D / 8);
                int k_row = next_k_start + row;
                if (k_row < S) {
                    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(
                        next_K_buf + row * D + col * 8);
                    const void* gmem_addr = K_bh + (int64_t)k_row * D + col * 8;
                    cp_async_16(smem_addr, gmem_addr);
                }
            }
            cp_async_commit();
        }

        // Compute QK_i
        float scores[BK];
        if (valid) {
            for (int j = 0; j < BK; j++) {
                int k_row = k_start + j;
                if (k_row >= S) {
                    scores[j] = -INFINITY;
                    continue;
                }
                float dot = 0.0f;
                #pragma unroll
                for (int d = 0; d < D; d += 2) {
                    __nv_bfloat162 q2 = *reinterpret_cast<__nv_bfloat162*>(Q_smem + tid * D_PAD + d);
                    __nv_bfloat162 k2 = *reinterpret_cast<__nv_bfloat162*>(K_buf + j * D + d);
                    float2 qf = __bfloat1622float2(q2);
                    float2 kf = __bfloat1622float2(k2);
                    dot = __fmaf_rn(qf.x, kf.x, dot);
                    dot = __fmaf_rn(qf.y, kf.y, dot);
                }
                scores[j] = dot * SCALE;
            }
        }

        // Wait for V_i
        if (has_next) {
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }
        __syncthreads();

        // Online softmax + PV_i
        if (valid) {
            float m_new = m;
            for (int j = 0; j < BK; j++) {
                if (k_start + j < S) {
                    m_new = fmaxf(m_new, scores[j]);
                }
            }

            float scale = (m > -INFINITY) ? fast_expf(m - m_new) : 0.0f;
            l *= scale;
            #pragma unroll
            for (int d = 0; d < D; d++) o[d] *= scale;

            for (int j = 0; j < BK; j++) {
                if (k_start + j >= S) {
                    scores[j] = 0.0f;
                } else {
                    float p = fast_expf(scores[j] - m_new);
                    scores[j] = p;
                    l += p;
                }
            }
            m = m_new;

            // Accumulate PV_i
            for (int j = 0; j < BK; j++) {
                float p = scores[j];
                if (p == 0.0f) continue;
                #pragma unroll
                for (int d = 0; d < D; d += 2) {
                    __nv_bfloat162 v2 = *reinterpret_cast<__nv_bfloat162*>(V_buf + j * D + d);
                    float2 vf = __bfloat1622float2(v2);
                    o[d]   = __fmaf_rn(p, vf.x, o[d]);
                    o[d+1] = __fmaf_rn(p, vf.y, o[d+1]);
                }
            }
        }

        // Rotate buffers
        int old_k = k_buf_idx;
        k_buf_idx = next_k_buf_idx;
        next_k_buf_idx = v_buf_idx;
        v_buf_idx = old_k;

        __syncthreads();
    }

    // Write output
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
    dim3 block(NUM_THREADS);

    size_t smem_size = (BQ * D_PAD + 3 * BK * D) * sizeof(__nv_bfloat16);

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