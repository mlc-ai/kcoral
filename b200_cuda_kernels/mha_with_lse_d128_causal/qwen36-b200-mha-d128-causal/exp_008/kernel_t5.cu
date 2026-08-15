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

template<int BLOCK_M, int BLOCK_N, int BLOCK_D>
__global__ void mha_causal_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S,
    int D,
    float inv_sqrt_d
) {
    constexpr int NUM_THREADS = 128;
    static_assert(BLOCK_D == 128);
    static_assert(BLOCK_M == 64);
    static_assert(BLOCK_N == 64);

    extern __shared__ char smem[];
    // Layout: sQ[BLOCK_M*BLOCK_D], sK[BLOCK_N*BLOCK_D], sV[BLOCK_N*BLOCK_D], sO[BLOCK_M*BLOCK_D]
    float* sQ  = reinterpret_cast<float*>(smem);
    float* sK  = sQ  + BLOCK_M * BLOCK_D;
    float* sV  = sK  + BLOCK_N * BLOCK_D;
    float* sO  = sV  + BLOCK_N * BLOCK_D;

    int b_idx = blockIdx.x;
    int h_idx = blockIdx.y;
    int m_idx = blockIdx.z;
    int num_heads = gridDim.y;

    int q_start = m_idx * BLOCK_M;
    int base_QKV = (b_idx * num_heads + h_idx) * S * D;

    // Load Q tile into shared memory using vectorized loads
    for (int tid = threadIdx.x; tid < BLOCK_M * (BLOCK_D / 4); tid += NUM_THREADS) {
        int r = tid / (BLOCK_D / 4);
        int c4 = (tid % (BLOCK_D / 4)) * 4;
        int qr = q_start + r;
        if (qr < S) {
            *(reinterpret_cast<float4*>(sQ) + tid) = reinterpret_cast<const float4*>(Q)[base_QKV / 4 + qr * (D / 4) + c4 / 4];
        } else {
            float4 zero = {0.0f, 0.0f, 0.0f, 0.0f};
            *(reinterpret_cast<float4*>(sQ) + tid) = zero;
        }
    }
    __syncthreads();

    // Initialize output accumulator
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

    // Registers for attention scores - keep them small by processing one row at a time
    float p_block[BLOCK_N];

    int num_n_blocks = (S + BLOCK_N - 1) / BLOCK_N;

    for (int n = 0; n < num_n_blocks; ++n) {
        int k_start = n * BLOCK_N;

        // Load K & V tiles
        for (int tid = threadIdx.x; tid < BLOCK_N * (BLOCK_D / 4); tid += NUM_THREADS) {
            int r = tid / (BLOCK_D / 4);
            int c4 = (tid % (BLOCK_D / 4)) * 4;
            int kr = k_start + r;
            float4 zk, zv;
            if (kr < S) {
                zk = reinterpret_cast<const float4*>(K)[base_QKV / 4 + kr * (D / 4) + c4 / 4];
                zv = reinterpret_cast<const float4*>(V)[base_QKV / 4 + kr * (D / 4) + c4 / 4];
            } else {
                zk = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                zv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            *(reinterpret_cast<float4*>(sK) + tid) = zk;
            *(reinterpret_cast<float4*>(sV) + tid) = zv;
        }
        __syncthreads();

        // Compute P = Q @ K^T * inv_sqrt_d
        for (int rr = 0; rr < 2; ++rr) {
            int rm = rr == 0 ? r1 : r2;
            const float* q_row = sQ + rm * BLOCK_D;
            for (int nk = 0; nk < BLOCK_N; ++nk) {
                float acc = 0.0f;
                const float* k_row = sK + nk * BLOCK_D;
#pragma unroll
                for (int d = 0; d < BLOCK_D; ++d) {
                    acc += q_row[d] * k_row[d];
                }
                p_block[nk] = acc * inv_sqrt_d;
            }

            bool valid = (rr == 0) ? valid_r1 : valid_r2;
            if (!valid) continue;
            
            int qr = q_start + (rr == 0 ? r1 : r2);
            int row_idx = rr == 0 ? r1 : r2;

            // Find block max respecting causal mask
            float local_max = -INFINITY;
            for (int nk = 0; nk < BLOCK_N; ++nk) {
                if (k_start + nk < qr && p_block[nk] > local_max) {
                    local_max = p_block[nk];
                }
            }
            if (local_max == -INFINITY) continue;

            float m_prev = m_r[rr];
            float m_new = (m_prev > local_max) ? m_prev : local_max;
            float alpha = expf(m_prev - m_new);
            float l_prev = l_r[rr];
            float l_new = 0.0f;

            float* o_row = sO + row_idx * BLOCK_D;
            for (int d = 0; d < BLOCK_D; ++d) {
                float acc_o = 0.0f;
                for (int nk = 0; nk < BLOCK_N; ++nk) {
                    if (k_start + nk < qr) {
                        float e = expf(p_block[nk] - m_new);
                        l_new += e;
                        acc_o += e * sV[nk * BLOCK_D + d];
                    }
                }
                o_row[d] = alpha * o_row[d] + acc_o;
            }
            
            l_new += alpha * l_prev;
            m_r[rr] = m_new;
            l_r[rr] = l_new;
        }
        __syncthreads();
    }

    // Epilogue: normalize and write back O and LSE
    for (int ri = 0; ri < 2; ++ri) {
        int r = ri == 0 ? r1 : r2;
        if (r >= BLOCK_M) break;
        int qr = q_start + r;
        if (qr >= S) break;

        float l_val = l_r[ri];
        float m_val = m_r[ri];
        float norm = (l_val > 1e-6f) ? (1.0f / l_val) : 0.0f;
        
        const float* o_row = sO + r * BLOCK_D;
        for (int d = 0; d < BLOCK_D / 4; ++d) {
            float4 out4;
            out4.x = __float2bfloat16(o_row[d * 4 + 0] * norm);
            out4.y = __float2bfloat16(o_row[d * 4 + 1] * norm);
            out4.z = __float2bfloat16(o_row[d * 4 + 2] * norm);
            out4.w = __float2bfloat16(o_row[d * 4 + 3] * norm);
            reinterpret_cast<__nv_bfloat16*>(O)[(base_QKV + qr * D) / 4 + d * 4] = reinterpret_cast<const __nv_bfloat16*>(&out4)[0];
            reinterpret_cast<__nv_bfloat16*>(O)[(base_QKV + qr * D) / 4 + d * 4 + 1] = reinterpret_cast<const __nv_bfloat16*>(&out4)[1];
            reinterpret_cast<__nv_bfloat16*>(O)[(base_QKV + qr * D) / 4 + d * 4 + 2] = reinterpret_cast<const __nv_bfloat16*>(&out4)[2];
            reinterpret_cast<__nv_bfloat16*>(O)[(base_QKV + qr * D) / 4 + d * 4 + 3] = reinterpret_cast<const __nv_bfloat16*>(&out4)[3];
        }
        int lse_idx = (b_idx * num_heads + h_idx) * S + qr;
        LSE[lse_idx] = (l_val > 1e-6f) ? (m_val + logf(l_val)) : (-INFINITY);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);

    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_N = 64;
    constexpr int BLOCK_D = 128;
    constexpr int NUM_THREADS = 128;

    float inv_sqrt_d = rsqrtf(static_cast<float>(D));

    dim3 grid((unsigned int)B, (unsigned int)H, (unsigned int)((S + BLOCK_M - 1) / BLOCK_M));
    dim3 block(NUM_THREADS);
    
    // Shared memory layout: Q[BLOCK_M x D], K[BLOCK_N x D], V[BLOCK_N x D], O[BLOCK_M x D]
    size_t smem_bytes = sizeof(float) * ((BLOCK_M + 2 * BLOCK_N + BLOCK_M) * BLOCK_D);

    mha_causal_kernel<BLOCK_M, BLOCK_N, BLOCK_D><<<grid, block, smem_bytes>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<__nv_bfloat16*>(O.data_ptr()),
        static_cast<float*>(LSE.data_ptr()),
        static_cast<int>(S),
        static_cast<int>(D),
        inv_sqrt_d
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace mha_cuda

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_cuda::run);