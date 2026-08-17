#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                    \
    cudaError_t _e = (call);                                     \
    if (_e != cudaSuccess) {                                     \
        fprintf(stderr, "CUDA error %s at %s:%d\n",              \
                cudaGetErrorString(_e), __FILE__, __LINE__);      \
        exit(1);                                                  \
    }                                                             \
} while (0)

namespace mha_cuda {

constexpr int BQ = 64;
constexpr int BK = 32;
constexpr int D = 128;
constexpr int THREADS = 64;
constexpr int H_CONST = 48;
constexpr int B_CONST = 4;

__global__ __launch_bounds__(64, 4)
void mha_causal_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S) {

    int bh = blockIdx.x;
    int b = bh / H_CONST;
    int h = bh % H_CONST;
    int q_block = blockIdx.y * BQ;
    int tid = threadIdx.x;
    int q = q_block + tid;

    extern __shared__ char smem[];
    __nv_bfloat16* Q_smem = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* KV_smem = Q_smem + BQ * D;
    float* S_smem = reinterpret_cast<float*>(KV_smem + BK * D);

    constexpr float scale = 0.08838834764831845f;

    size_t head_offset = (size_t)b * H_CONST * S * D + (size_t)h * S * D;
    const __nv_bfloat16* Q_ptr = Q + head_offset;
    const __nv_bfloat16* K_ptr = K + head_offset;
    const __nv_bfloat16* V_ptr = V + head_offset;
    __nv_bfloat16* O_ptr = O + head_offset;
    float* LSE_ptr = LSE + (size_t)b * H_CONST * S + (size_t)h * S;

    bool active = (q < S);
    int causal_limit = active ? (q + 1) : 0;

    // Load Q into shared memory (zero-pad if out of bounds)
    for (int i = tid; i < BQ * (D / 8); i += THREADS) {
        int q_idx = i / (D / 8);
        int d_chunk = i % (D / 8);
        if (q_block + q_idx < S) {
            *(reinterpret_cast<int4*>(Q_smem + q_idx * D + d_chunk * 8)) =
                *(reinterpret_cast<const int4*>(Q_ptr + (size_t)(q_block + q_idx) * D + d_chunk * 8));
        } else {
            *(reinterpret_cast<int4*>(Q_smem + q_idx * D + d_chunk * 8)) = make_int4(0, 0, 0, 0);
        }
    }
    __syncthreads();

    // Load Q row into registers (all 128 values = 64 bf16x2)
    __nv_bfloat162 q_reg[64];
    {
        const __nv_bfloat162* Q_row = reinterpret_cast<const __nv_bfloat162*>(Q_smem + tid * D);
        #pragma unroll
        for (int i = 0; i < 64; i++) {
            q_reg[i] = Q_row[i];
        }
    }

    float m = -INFINITY;
    float l = 0.0f;
    float o_acc[128];
    #pragma unroll
    for (int i = 0; i < 128; i++) o_acc[i] = 0.0f;

    int block_max_k = min(q_block + BQ, S);

    for (int k_block = 0; k_block < block_max_k; k_block += BK) {
        int num_k = min(BK, block_max_k - k_block);

        // Load K tile (cooperative)
        for (int i = tid; i < num_k * (D / 8); i += THREADS) {
            int k = i / (D / 8);
            int d_chunk = i % (D / 8);
            *(reinterpret_cast<int4*>(KV_smem + k * D + d_chunk * 8)) =
                *(reinterpret_cast<const int4*>(K_ptr + (size_t)(k_block + k) * D + d_chunk * 8));
        }
        __syncthreads();

        // Compute scores: each thread computes full 128-dim dot product independently
        float tile_max = -INFINITY;
        for (int k = 0; k < num_k; k++) {
            float dot = 0.0f;
            bool valid = active && (k_block + k < causal_limit);
            if (valid) {
                const __nv_bfloat162* K_row = reinterpret_cast<const __nv_bfloat162*>(KV_smem + k * D);
                #pragma unroll
                for (int i = 0; i < 64; i++) {
                    float2 qf = __bfloat1622float2(q_reg[i]);
                    float2 kf = __bfloat1622float2(K_row[i]);
                    dot = fmaf(qf.x, kf.x, dot);
                    dot = fmaf(qf.y, kf.y, dot);
                }
            }
            float score = valid ? dot * scale : -INFINITY;
            S_smem[tid * BK + k] = score;
            tile_max = fmaxf(tile_max, score);
        }

        // Online softmax update
        float m_new = fmaxf(m, tile_max);
        float rescale = (m > -INFINITY) ? expf(m - m_new) : 0.0f;

        // Compute P and row_sum
        float row_sum = 0.0f;
        for (int k = 0; k < num_k; k++) {
            float s = S_smem[tid * BK + k];
            float p = (s == -INFINITY) ? 0.0f : expf(s - m_new);
            S_smem[tid * BK + k] = p;
            row_sum += p;
        }

        l = l * rescale + row_sum;
        m = m_new;

        // Rescale o_acc
        #pragma unroll
        for (int i = 0; i < 128; i++) o_acc[i] *= rescale;

        // Load V tile (reuse KV_smem)
        for (int i = tid; i < num_k * (D / 8); i += THREADS) {
            int k = i / (D / 8);
            int d_chunk = i % (D / 8);
            *(reinterpret_cast<int4*>(KV_smem + k * D + d_chunk * 8)) =
                *(reinterpret_cast<const int4*>(V_ptr + (size_t)(k_block + k) * D + d_chunk * 8));
        }
        __syncthreads();

        // P @ V accumulation (full 128 D dims per thread)
        for (int k = 0; k < num_k; k++) {
            float p = S_smem[tid * BK + k];
            if (p > 0.0f) {
                const __nv_bfloat162* V_row = reinterpret_cast<const __nv_bfloat162*>(KV_smem + k * D);
                #pragma unroll
                for (int i = 0; i < 64; i++) {
                    float2 vf = __bfloat1622float2(V_row[i]);
                    o_acc[i * 2]     = fmaf(p, vf.x, o_acc[i * 2]);
                    o_acc[i * 2 + 1] = fmaf(p, vf.y, o_acc[i * 2 + 1]);
                }
            }
        }
    }

    // Write output
    if (active) {
        float inv_l = 1.0f / l;
        #pragma unroll
        for (int i = 0; i < 64; i++) {
            O_ptr[(size_t)q * D + i * 2]     = __float2bfloat16_rn(o_acc[i * 2] * inv_l);
            O_ptr[(size_t)q * D + i * 2 + 1] = __float2bfloat16_rn(o_acc[i * 2 + 1] * inv_l);
        }
        LSE_ptr[q] = m + logf(l);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t S = Q.size(2);

    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    int num_q_blocks = ((int)S + BQ - 1) / BQ;
    dim3 grid(B_CONST * H_CONST, num_q_blocks);
    dim3 block(THREADS);

    size_t smem_size = (size_t)BQ * D * sizeof(__nv_bfloat16) +   // Q_smem
                       (size_t)BK * D * sizeof(__nv_bfloat16) +   // KV_smem
                       (size_t)BQ * BK * sizeof(float);            // S_smem

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    mha_causal_kernel<<<grid, block, smem_size, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data, (int)S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_cuda::run);

}  // namespace mha_cuda