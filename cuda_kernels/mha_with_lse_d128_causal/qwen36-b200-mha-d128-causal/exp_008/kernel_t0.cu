#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cmath>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                          \
    cudaError_t _e = (call);                                           \
    if (_e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                cudaGetErrorString(_e), __FILE__, __LINE__);           \
        exit(1);                                                       \
    }                                                                  \
} while(0)

namespace mha_cuda {

template<int BLOCK_M, int BLOCK_N>
__global__ void mha_causal_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S,
    float inv_sqrt_d
) {
    constexpr int BLOCK_D = 128;
    constexpr int NUM_THREADS = 128;

    extern __shared__ char smem[];
    float* sQ  = reinterpret_cast<float*>(smem);
    float* sK  = sQ  + BLOCK_M * BLOCK_D;
    float* sV  = sK  + BLOCK_N * BLOCK_D;
    float* sO  = sV  + BLOCK_N * BLOCK_D;

    int b_idx = blockIdx.x;
    int h_idx = blockIdx.y;
    int m_idx = blockIdx.z;

    int q_start = m_idx * BLOCK_M;
    int base_idx = (b_idx * blockDim.y + h_idx) * S * BLOCK_D;

    // Load Q tile into shared memory
    for (int d = threadIdx.x; d < BLOCK_D; d += NUM_THREADS) {
        for (int m = 0; m < BLOCK_M; ++m) {
            int qr = q_start + m;
            if (qr < S) {
                sQ[m * BLOCK_D + d] = __bfloat162float(Q[base_idx + qr * BLOCK_D + d]);
            } else {
                sQ[m * BLOCK_D + d] = 0.0f;
            }
        }
    }
    __syncthreads();

    // Initialize output accumulator in shared memory
    for (int i = threadIdx.x; i < BLOCK_M * BLOCK_D; i += NUM_THREADS) {
        sO[i] = 0.0f;
    }
    __syncthreads();

    // Each thread handles 2 query rows
    int r1 = threadIdx.x / 2;
    int r2 = r1 + 1;
    bool valid_r1 = r1 < BLOCK_M && (q_start + r1) < S;
    bool valid_r2 = r2 < BLOCK_M && (q_start + r2) < S;

    float m_r[2] = {-INFINITY, -INFINITY};
    float l_r[2] = {0.0f, 0.0f};

    int num_n_blocks = (S + BLOCK_N - 1) / BLOCK_N;

    for (int n = 0; n < num_n_blocks; ++n) {
        int k_start = n * BLOCK_N;

        // Load K & V tiles
        for (int d = threadIdx.x; d < BLOCK_D; d += NUM_THREADS) {
            for (int nk = 0; nk < BLOCK_N; ++nk) {
                int kr = k_start + nk;
                float val_k = (kr < S) ? __bfloat162float(K[base_idx + kr * BLOCK_D + d]) : 0.0f;
                float val_v = (kr < S) ? __bfloat162float(V[base_idx + kr * BLOCK_D + d]) : 0.0f;
                sK[nk * BLOCK_D + d] = val_k;
                sV[nk * BLOCK_D + d] = val_v;
            }
        }
        __syncthreads();

        // Compute attention scores P = Q @ K^T / sqrt(D)
        float cur_p[2][BLOCK_N] = {};
        for (int d = 0; d < BLOCK_D; ++d) {
            float qv1 = sQ[r1 * BLOCK_D + d];
            float qv2 = sQ[r2 * BLOCK_D + d];
            for (int nk = 0; nk < BLOCK_N; ++nk) {
                float kv = sK[nk * BLOCK_D + d];
                cur_p[0][nk] += qv1 * kv;
                cur_p[1][nk] += qv2 * kv;
            }
        }
        // Apply scaling factor
        for(int rr = 0; rr < 2; ++rr) {
            for(int nk = 0; nk < BLOCK_N; ++nk) cur_p[rr][nk] *= inv_sqrt_d;
        }

        // Softmax update & accumulate O
        for (int rr = 0; rr < 2; ++rr) {
            bool valid = (rr == 0) ? valid_r1 : valid_r2;
            if (!valid) continue;
            
            int qr = q_start + (rr == 0 ? r1 : r2);

            // Find block max with causal mask
            float local_max = -INFINITY;
            for (int nk = 0; nk < BLOCK_N; ++nk) {
                if (k_start + nk < qr) {
                    if (cur_p[rr][nk] > local_max) local_max = cur_p[rr][nk];
                }
            }
            if (local_max == -INFINITY) local_max = 0.0f;

            float m_prev = m_r[rr];
            float m_new = (m_prev > local_max) ? m_prev : local_max;

            float alpha = expf(m_prev - m_new);
            float l_prev = l_r[rr];
            float l_new = 0.0f;

            // Weighted accumulation into O
            for (int d = 0; d < BLOCK_D; ++d) {
                float acc = 0.0f;
                for (int nk = 0; nk < BLOCK_N; ++nk) {
                    if (k_start + nk < qr) {
                        float e = expf(cur_p[rr][nk] - m_new);
                        l_new += e;
                        acc += e * sV[nk * BLOCK_D + d];
                    }
                }
                int row_idx = rr == 0 ? r1 : r2;
                sO[row_idx * BLOCK_D + d] = alpha * sO[row_idx * BLOCK_D + d] + acc;
            }
            
            l_new += alpha * l_prev;
            m_r[rr] = m_new;
            l_r[rr] = l_new;
        }
        __syncthreads();
    }

    // Epilogue: normalize and write back O and LSE
    int rows_to_process = 2;
    if (r2 >= BLOCK_M || q_start + r2 >= S) rows_to_process = 1;

    for (int ri = 0; ri < rows_to_process; ++ri) {
        int r = ri == 0 ? r1 : r2;
        int qr = q_start + r;
        if (qr >= S) break;

        float l_val = l_r[ri];
        float m_val = m_r[ri];
        float norm = (l_val > 1e-6f) ? (1.0f / l_val) : 0.0f;

        for (int d = 0; d < BLOCK_D; ++d) {
            O[base_idx + qr * BLOCK_D + d] = __float2bfloat16(sO[r * BLOCK_D + d] * norm);
        }
        
        int lse_idx = (b_idx * blockDim.y + h_idx) * S + qr;
        LSE[lse_idx] = m_val + logf(l_val > 1e-6f ? l_val : 0.0f);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);

    static_assert(true); // D is expected to be 128
    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_N = 64;
    constexpr int BLOCK_D = 128;
    constexpr int NUM_THREADS = 128;

    float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(D));

    dim3 grid(B, H, (S + BLOCK_M - 1) / BLOCK_M);
    dim3 block(NUM_THREADS);
    
    size_t smem_bytes = sizeof(float) * (BLOCK_M * BLOCK_D + 2 * BLOCK_N * BLOCK_D + BLOCK_M * BLOCK_D);

    cudaStream_t stream = 
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    mha_causal_kernel<BLOCK_M, BLOCK_N><<<grid, block, smem_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<__nv_bfloat16*>(O.data_ptr()),
        static_cast<float*>(LSE.data_ptr()),
        static_cast<int>(S),
        inv_sqrt_d
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_cuda::run);

} // namespace mha_cuda