#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cfloat>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_impl {

static constexpr int D = 128;
static constexpr int MQ = 32;
static constexpr int BK = 64;
static constexpr int THD = 128;
static constexpr int NW = THD / 32;

template <>
struct type_identity<__nv_bfloat16> { using type = __nv_bfloat16; };

__global__ void mha_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S, float scale) {

    __shared__ __nv_bfloat16 sK[BK][D];
    __shared__ __nv_bfloat16 sV[BK][D];
    __shared__ float sWScore[NW * BK * MQ];
    __shared__ float sS[BK * MQ];

    int bh = blockIdx.x;
    int q_start = blockIdx.y * MQ;
    int tid = threadIdx.x;
    int lid = tid % 32;
    int wid = tid / 32;
    uint32_t FULL_MASK = 0xFFFFFFFF;

    uint64_t base_off = (uint64_t)bh * S * D;

    float q_reg[MQ];
    float m_reg[MQ], d_reg[MQ], o_reg[MQ];

    for (int i = 0; i < MQ; ++i) {
        int r = q_start + i;
        if (r < S) {
            q_reg[i] = __bfloat162float(Q[base_off + (uint64_t)r * D + tid]);
        } else {
            q_reg[i] = 0.0f;
        }
        m_reg[i] = -FLT_MAX;
        d_reg[i] = 0.0f;
        o_reg[i] = 0.0f;
    }

    for (int kv_start = 0; kv_start < S; kv_start += BK) {
        int bk_end = min(BK, S - kv_start);

        // Clear warp-score accumulator
        for (int i = tid; i < NW * BK * MQ; i += THD) {
            sWScore[i] = 0.0f;
        }
        __syncthreads();

        // Load K and V tile cooperatively
        for (int kv = 0; kv < BK; ++kv) {
            int kv_idx = kv_start + kv;
            if (kv_idx < S) {
                uint64_t off = base_off + (uint64_t)kv_idx * D;
                sK[kv][tid] = K[off + tid];
                sV[kv][tid] = V[off + tid];
            }
        }
        __syncthreads();

        // Compute partial scores with warp reduction
        for (int kv = 0; kv < BK; ++kv) {
            float kval[D];
            kval[tid] = __bfloat162float(sK[kv][tid]);
            // Each thread computes partials for MQ rows, warp-reduces each
            for (int mq = 0; mq < MQ; ++mq) {
                float partial = q_reg[mq] * kval[tid] * scale;
                for (int s = 16; s > 0; s >>= 1) {
                    partial += __shfl_down_sync(FULL_MASK, partial, s);
                }
                if (lid == 0) {
                    sWScore[wid * BK * MQ + kv * MQ + mq] = partial;
                }
            }
        }
        __syncthreads();

        // Merge warp partials into final scores
        for (int i = tid; i < BK * MQ; i += THD) {
            float sum = 0.0f;
            for (int w = 0; w < NW; ++w) {
                sum += sWScore[w * BK * MQ + i];
            }
            sS[i] = sum;
        }
        __syncthreads();

        // Online softmax + weighted V accumulation
        for (int mq = 0; mq < MQ; ++mq) {
            int r = q_start + mq;
            if (r >= S) continue;

            float tmax = -FLT_MAX;
            for (int kv = 0; kv < bk_end; ++kv) {
                float s = sS[kv * MQ + mq];
                if (s > tmax) tmax = s;
            }

            float old_m = m_reg[mq];
            m_reg[mq] = tmax;
            float alpha = expf(old_m - tmax);
            d_reg[mq] *= alpha;
            o_reg[mq] *= alpha;

            for (int kv = 0; kv < bk_end; ++kv) {
                float s = sS[kv * MQ + mq];
                float p = expf(s - tmax);
                d_reg[mq] += p;
                o_reg[mq] += p * __bfloat162float(sV[kv][tid]);
            }
        }
        __syncthreads();
    }

    // Normalize and store output
    for (int mq = 0; mq < MQ; ++mq) {
        int r = q_start + mq;
        if (r < S) {
            float inv = 1.0f / d_reg[mq];
            O[base_off + (uint64_t)r * D + tid] = __float2bfloat16(o_reg[mq] * inv);
        }
    }

    // Store LSE (one thread per block)
    if (tid == 0) {
        uint64_t lse_base = (uint64_t)bh * S;
        for (int mq = 0; mq < MQ; ++mq) {
            int r = q_start + mq;
            if (r < S) {
                LSE[lse_base + r] = m_reg[mq] + logf(d_reg[mq]);
            }
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D_dim = Q.size(3);

    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    float sc = 1.0f / sqrtf(static_cast<float>(D_dim));

    int64_t num_bh = B * H;
    int64_t num_qblocks = (S + MQ - 1) / MQ;

    dim3 grid(num_bh, num_qblocks);
    dim3 block(THD);
    size_t smem_bytes = sizeof(__nv_bfloat16) * BK * D
                       + sizeof(__nv_bfloat16) * BK * D
                       + sizeof(float) * NW * BK * MQ
                       + sizeof(float) * BK * MQ;

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    mha_kernel<<<grid, block, smem_bytes, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data,
        static_cast<int>(B), static_cast<int>(H), static_cast<int>(S), sc);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

} // namespace mha_impl

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_impl::run);