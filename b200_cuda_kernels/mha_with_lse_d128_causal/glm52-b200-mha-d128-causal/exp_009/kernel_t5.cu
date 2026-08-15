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

constexpr int BQ = 16;
constexpr int BK = 32;
constexpr int D = 128;
constexpr int THREADS = 512;  // 16 warps, one per query row
constexpr int H_CONST = 48;
constexpr int B_CONST = 4;

__global__ __launch_bounds__(512, 2)
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
    int warp_id = tid / 32;   // maps to q_local (0..15)
    int lane = tid % 32;      // 0..31
    int q = q_block + warp_id;
    int d_offset = lane * 4;  // each lane handles 4 bf16 values

    extern __shared__ char smem[];
    __nv_bfloat16* KV_smem = reinterpret_cast<__nv_bfloat16*>(smem);

    constexpr float scale = 0.08838834764831845f;  // 1/sqrt(128)

    size_t head_offset = (size_t)b * H_CONST * S * D + (size_t)h * S * D;
    const __nv_bfloat16* Q_ptr = Q + head_offset;
    const __nv_bfloat16* K_ptr = K + head_offset;
    const __nv_bfloat16* V_ptr = V + head_offset;
    __nv_bfloat16* O_ptr = O + head_offset;
    float* LSE_ptr = LSE + (size_t)b * H_CONST * S + (size_t)h * S;

    bool active = (q < S);
    int causal_limit = active ? (q + 1) : 0;

    // Load Q: each lane loads 4 bf16 = 2 bf16x2
    __nv_bfloat162 q_reg[2];
    if (active) {
        const __nv_bfloat162* Q_row = reinterpret_cast<const __nv_bfloat162*>(Q_ptr + (size_t)q * D + d_offset);
        q_reg[0] = Q_row[0];
        q_reg[1] = Q_row[1];
    }

    float m = -INFINITY;
    float l = 0.0f;
    float o_acc[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) o_acc[i] = 0.0f;

    int block_max_k = min(q_block + BQ, S);

    for (int k_block = 0; k_block < block_max_k; k_block += BK) {
        int num_k = min(BK, block_max_k - k_block);

        // Load K tile: BK*D/8 = 32*16 = 512 chunks, 512 threads = 1 per thread
        if (tid < num_k * (D / 8)) {
            int k = tid / (D / 8);
            int d_chunk = tid % (D / 8);
            *(reinterpret_cast<int4*>(KV_smem + k * D + d_chunk * 8)) =
                *(reinterpret_cast<const int4*>(K_ptr + (size_t)(k_block + k) * D + d_chunk * 8));
        }
        __syncthreads();

        // Compute scores: each lane computes 4-dim partial dot, then reduce across 32 lanes
        float scores[BK];
        float tile_max = -INFINITY;
        
        for (int k = 0; k < num_k; k++) {
            float dot = 0.0f;
            bool valid = active && (k_block + k < causal_limit);
            if (valid) {
                const __nv_bfloat162* K_row = reinterpret_cast<const __nv_bfloat162*>(KV_smem + k * D + d_offset);
                float2 qf0 = __bfloat1622float2(q_reg[0]);
                float2 kf0 = __bfloat1622float2(K_row[0]);
                float2 qf1 = __bfloat1622float2(q_reg[1]);
                float2 kf1 = __bfloat1622float2(K_row[1]);
                dot = fmaf(qf0.x, kf0.x, 0.0f);
                dot = fmaf(qf0.y, kf0.y, dot);
                dot = fmaf(qf1.x, kf1.x, dot);
                dot = fmaf(qf1.y, kf1.y, dot);
            }
            // Warp shuffle reduction across 32 lanes (5 steps)
            dot += __shfl_xor_sync(0xFFFFFFFF, dot, 1);
            dot += __shfl_xor_sync(0xFFFFFFFF, dot, 2);
            dot += __shfl_xor_sync(0xFFFFFFFF, dot, 4);
            dot += __shfl_xor_sync(0xFFFFFFFF, dot, 8);
            dot += __shfl_xor_sync(0xFFFFFFFF, dot, 16);
            
            scores[k] = valid ? dot * scale : -INFINITY;
            tile_max = fmaxf(tile_max, scores[k]);
        }

        // Online softmax update
        float m_new = fmaxf(m, tile_max);
        float rescale = (m > -INFINITY) ? expf(m - m_new) : 0.0f;

        float row_sum = 0.0f;
        for (int k = 0; k < num_k; k++) {
            scores[k] = (scores[k] == -INFINITY) ? 0.0f : expf(scores[k] - m_new);
            row_sum += scores[k];
        }

        l = l * rescale + row_sum;
        m = m_new;

        #pragma unroll
        for (int i = 0; i < 4; i++) o_acc[i] *= rescale;

        // Load V tile (reuse same shared memory)
        if (tid < num_k * (D / 8)) {
            int k = tid / (D / 8);
            int d_chunk = tid % (D / 8);
            *(reinterpret_cast<int4*>(KV_smem + k * D + d_chunk * 8)) =
                *(reinterpret_cast<const int4*>(V_ptr + (size_t)(k_block + k) * D + d_chunk * 8));
        }
        __syncthreads();

        // P @ V accumulation: each lane accumulates 4 output dims
        for (int k = 0; k < num_k; k++) {
            float p = scores[k];
            if (p > 0.0f) {
                const __nv_bfloat162* V_row = reinterpret_cast<const __nv_bfloat162*>(KV_smem + k * D + d_offset);
                float2 vf0 = __bfloat1622float2(V_row[0]);
                float2 vf1 = __bfloat1622float2(V_row[1]);
                o_acc[0] = fmaf(p, vf0.x, o_acc[0]);
                o_acc[1] = fmaf(p, vf0.y, o_acc[1]);
                o_acc[2] = fmaf(p, vf1.x, o_acc[2]);
                o_acc[3] = fmaf(p, vf1.y, o_acc[3]);
            }
        }
    }

    // Write output
    if (active) {
        float inv_l = 1.0f / l;
        __nv_bfloat162* O_row = reinterpret_cast<__nv_bfloat162*>(O_ptr + (size_t)q * D + d_offset);
        O_row[0] = __floats2bfloat162_rn(o_acc[0] * inv_l, o_acc[1] * inv_l);
        O_row[1] = __floats2bfloat162_rn(o_acc[2] * inv_l, o_acc[3] * inv_l);
        if (lane == 0) {
            LSE_ptr[q] = m + logf(l);
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