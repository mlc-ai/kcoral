#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
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

constexpr int BQ = 32;
constexpr int BK = 64;
constexpr int D = 128;
constexpr int THREADS = 128;
constexpr int H_CONST = 48;
constexpr int B_CONST = 4;

__global__ void mha_causal_kernel(
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
    int warp_id = tid / 32;
    int lane = tid % 32;
    int q_local = lane % 8;
    int k_group = lane / 8;
    int q = warp_id * 8 + q_local;
    int global_q = q_block + q;
    int k_base = k_group * 16;
    int d_base = k_group * 32;

    extern __shared__ char smem[];
    __nv_bfloat16* Q_smem  = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* KV_smem = Q_smem + BQ * D;
    float*         S_smem  = reinterpret_cast<float*>(KV_smem + BK * D);

    constexpr float scale = 0.08838834764831845f; // 1/sqrt(128)

    size_t head_offset = (size_t)b * H_CONST * S * D + (size_t)h * S * D;
    const __nv_bfloat16* Q_ptr = Q + head_offset;
    const __nv_bfloat16* K_ptr = K + head_offset;
    const __nv_bfloat16* V_ptr = V + head_offset;
    __nv_bfloat16* O_ptr       = O + head_offset;
    float* LSE_ptr             = LSE + (size_t)b * H_CONST * S + (size_t)h * S;

    int q_end = min(BQ, S - q_block);
    int causal_key_limit = min(q_block + BQ, S);

    // Load Q tile into shared memory
    for (int i = tid; i < BQ * D; i += THREADS) {
        int qi = i / D;
        int di = i % D;
        Q_smem[qi * D + di] = (qi < q_end)
            ? Q_ptr[(size_t)(q_block + qi) * D + di]
            : __float2bfloat16(0.0f);
    }
    __syncthreads();

    // Load Q row for this thread into registers
    __nv_bfloat162 q_reg[64];
    if (q < q_end) {
        #pragma unroll
        for (int i = 0; i < 64; i++) {
            q_reg[i] = *reinterpret_cast<const __nv_bfloat162*>(&Q_smem[q * D + i * 2]);
        }
    }

    // Per-thread state (all 4 threads per q row stay in sync via shfl)
    float m = -INFINITY;
    float l = 0.0f;
    float o_acc[32];
    #pragma unroll
    for (int dd = 0; dd < 32; dd++) o_acc[dd] = 0.0f;

    for (int k_block = 0; k_block < causal_key_limit; k_block += BK) {
        int num_k = min(BK, causal_key_limit - k_block);

        // Load K tile
        for (int i = tid; i < BK * D; i += THREADS) {
            int ki = i / D;
            int di = i % D;
            KV_smem[ki * D + di] = (ki < num_k)
                ? K_ptr[(size_t)(k_block + ki) * D + di]
                : __float2bfloat16(0.0f);
        }
        __syncthreads();

        // Compute scores S[q][k] = (Q[q] . K[k]) * scale, apply causal mask
        float local_max = -INFINITY;
        if (q < q_end) {
            #pragma unroll
            for (int kk = 0; kk < 16; kk++) {
                int k = k_base + kk;
                int global_k = k_block + k;
                float score;
                if (k >= num_k || global_k > global_q) {
                    score = -INFINITY;
                } else {
                    float dot = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < 64; i++) {
                        __nv_bfloat162 kv = *reinterpret_cast<const __nv_bfloat162*>(&KV_smem[k * D + i * 2]);
                        float2 qf = __bfloat1622float2(q_reg[i]);
                        float2 kf = __bfloat1622float2(kv);
                        dot = fmaf(qf.x, kf.x, dot);
                        dot = fmaf(qf.y, kf.y, dot);
                    }
                    score = dot * scale;
                }
                S_smem[q * BK + k] = score;
                local_max = fmaxf(local_max, score);
            }
        }
        __syncthreads();

        // Reduce row max across 4 threads (k_group 0..3) via warp shuffles
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, 8));
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, 16));

        float m_new = fmaxf(m, local_max);
        float rescale = expf(m - m_new);

        // Compute P = exp(S - m_new) and row sum
        float row_sum = 0.0f;
        if (q < q_end) {
            #pragma unroll
            for (int kk = 0; kk < 16; kk++) {
                int k = k_base + kk;
                float s = S_smem[q * BK + k];
                float p = (s == -INFINITY) ? 0.0f : expf(s - m_new);
                S_smem[q * BK + k] = p;
                row_sum += p;
            }
        }
        __syncthreads();

        // Reduce row sum across 4 threads
        row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, 8);
        row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, 16);

        l = l * rescale + row_sum;
        m = m_new;

        // Rescale O accumulators
        #pragma unroll
        for (int dd = 0; dd < 32; dd++) o_acc[dd] *= rescale;

        // Load V tile (overwrite K in shared memory)
        for (int i = tid; i < BK * D; i += THREADS) {
            int ki = i / D;
            int di = i % D;
            KV_smem[ki * D + di] = (ki < num_k)
                ? V_ptr[(size_t)(k_block + ki) * D + di]
                : __float2bfloat16(0.0f);
        }
        __syncthreads();

        // O += P @ V  (each thread: 32 D columns x num_k keys)
        if (q < q_end) {
            for (int k = 0; k < num_k; k++) {
                float p = S_smem[q * BK + k];
                #pragma unroll
                for (int dd = 0; dd < 32; dd += 2) {
                    int d = d_base + dd;
                    __nv_bfloat162 vv = *reinterpret_cast<const __nv_bfloat162*>(&KV_smem[k * D + d]);
                    float2 vf = __bfloat1622float2(vv);
                    o_acc[dd]     = fmaf(p, vf.x, o_acc[dd]);
                    o_acc[dd + 1] = fmaf(p, vf.y, o_acc[dd + 1]);
                }
            }
        }
    }

    // Write outputs
    if (q < q_end) {
        float inv_l = 1.0f / l;
        #pragma unroll
        for (int dd = 0; dd < 32; dd++) {
            int d = d_base + dd;
            O_ptr[(size_t)global_q * D + d] = __float2bfloat16(o_acc[dd] * inv_l);
        }
        if (k_group == 0) {
            LSE_ptr[global_q] = m + logf(l);
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
    __nv_bfloat16*       O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float*               LSE_data = static_cast<float*>(LSE.data_ptr());

    int num_q_blocks = ((int)S + BQ - 1) / BQ;
    dim3 grid(B_CONST * H_CONST, num_q_blocks);
    dim3 block(THREADS);

    size_t smem_size =
        (size_t)BQ * D * sizeof(__nv_bfloat16) +   // Q_smem
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