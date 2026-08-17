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
constexpr int THREADS = 256;
constexpr int H_CONST = 48;
constexpr int B_CONST = 4;

__global__ __launch_bounds__(256, 2)
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
    int q_local = tid >> 2;   // 0..63
    int d_quarter = tid & 3;  // 0..3
    int q = q_block + q_local;
    int d_offset = d_quarter * 32;

    extern __shared__ char smem[];
    __nv_bfloat16* KV_smem = reinterpret_cast<__nv_bfloat16*>(smem);

    constexpr float scale = 0.08838834764831845f;

    size_t head_offset = (size_t)b * H_CONST * S * D + (size_t)h * S * D;
    const __nv_bfloat16* Q_ptr = Q + head_offset;
    const __nv_bfloat16* K_ptr = K + head_offset;
    const __nv_bfloat16* V_ptr = V + head_offset;
    __nv_bfloat16* O_ptr = O + head_offset;
    float* LSE_ptr = LSE + (size_t)b * H_CONST * S + (size_t)h * S;

    bool active = (q < S);

    // Load Q quarter (16 bf16x2 = 32 bf16 values) into registers
    __nv_bfloat162 q_reg[16];
    if (active) {
        const __nv_bfloat162* Q_row = reinterpret_cast<const __nv_bfloat162*>(Q_ptr + (size_t)q * D + d_offset);
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            q_reg[i] = Q_row[i];
        }
    }

    float m = -INFINITY;
    float l = 0.0f;
    float o_acc[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) o_acc[i] = 0.0f;

    int causal_limit = active ? (q + 1) : 0;
    int max_k = min(q_block + BQ, S);

    for (int k_block = 0; k_block < max_k; k_block += BK) {
        int num_k = min(BK, max_k - k_block);

        // Load K tile: BK * D bf16 values using 128-bit vectorized loads
        // BK * (D/8) = 32 * 16 = 512 int4 chunks, 256 threads => 2 chunks per thread
        for (int i = tid; i < BK * (D / 8); i += THREADS) {
            int k = i / (D / 8);
            int d_chunk = i % (D / 8);
            if (k < num_k) {
                *(reinterpret_cast<int4*>(KV_smem + k * D + d_chunk * 8)) =
                    *(reinterpret_cast<const int4*>(K_ptr + (size_t)(k_block + k) * D + d_chunk * 8));
            }
        }
        __syncthreads();

        // Compute scores: each thread computes 1/4 of dot product, then reduce via shfl
        float scores[BK];
        float local_max = -INFINITY;

        for (int k = 0; k < num_k; k++) {
            float dot = 0.0f;
            bool valid = active && (k_block + k < causal_limit);
            if (valid) {
                const __nv_bfloat162* K_row = reinterpret_cast<const __nv_bfloat162*>(KV_smem + k * D + d_offset);
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    float2 qf = __bfloat1622float2(q_reg[i]);
                    float2 kf = __bfloat1622float2(K_row[i]);
                    dot = fmaf(qf.x, kf.x, dot);
                    dot = fmaf(qf.y, kf.y, dot);
                }
            }
            // Reduce across 4 threads (uniform - all threads participate to avoid deadlock)
            dot += __shfl_xor_sync(0xFFFFFFFF, dot, 1);
            dot += __shfl_xor_sync(0xFFFFFFFF, dot, 2);
            scores[k] = valid ? dot * scale : -INFINITY;
            local_max = fmaxf(local_max, scores[k]);
        }

        // Online softmax update
        float m_new = fmaxf(m, local_max);
        float rescale = __expf(m - m_new);

        float row_sum = 0.0f;
        for (int k = 0; k < num_k; k++) {
            scores[k] = (scores[k] == -INFINITY) ? 0.0f : __expf(scores[k] - m_new);
            row_sum += scores[k];
        }

        l = l * rescale + row_sum;
        m = m_new;

        #pragma unroll
        for (int i = 0; i < 32; i++) o_acc[i] *= rescale;

        // Load V tile (reuse same shared memory)
        for (int i = tid; i < BK * (D / 8); i += THREADS) {
            int k = i / (D / 8);
            int d_chunk = i % (D / 8);
            if (k < num_k) {
                *(reinterpret_cast<int4*>(KV_smem + k * D + d_chunk * 8)) =
                    *(reinterpret_cast<const int4*>(V_ptr + (size_t)(k_block + k) * D + d_chunk * 8));
            }
        }
        __syncthreads();

        // P @ V accumulation: each thread accumulates 32 output dimensions
        for (int k = 0; k < num_k; k++) {
            float p = scores[k];
            if (p > 0.0f) {
                const __nv_bfloat162* V_row = reinterpret_cast<const __nv_bfloat162*>(KV_smem + k * D + d_offset);
                #pragma unroll
                for (int i = 0; i < 16; i++) {
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
        __nv_bfloat162* O_row = reinterpret_cast<__nv_bfloat162*>(O_ptr + (size_t)q * D + d_offset);
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            O_row[i] = __floats2bfloat162_rn(o_acc[i * 2] * inv_l, o_acc[i * 2 + 1] * inv_l);
        }
        if (d_quarter == 0) {
            LSE_ptr[q] = m + __logf(l);
        }
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

    size_t smem_size = (size_t)BK * D * sizeof(__nv_bfloat16);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    mha_causal_kernel<<<grid, block, smem_size, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data, (int)S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_cuda::run);

}  // namespace mha_cuda