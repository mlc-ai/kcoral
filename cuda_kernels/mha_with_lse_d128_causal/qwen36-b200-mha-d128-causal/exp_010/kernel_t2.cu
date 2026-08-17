#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_mha_causal_d128 {

// Tiled causal FlashAttention kernel for BF16 inputs
// Template parameters allow specialization for D=128
template <int BM, int BN, int D, int NUM_THREADS>
__global__ void causal_attn_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S,
    float inv_sqrt_d)
{
    // Shared memory layout: Q_tile[B M x D] + K_tile[BN x D] + V_tile[BN x D]
    extern __shared__ char shared_mem[];
    __nv_bfloat16* s_Q = reinterpret_cast<__nv_bfloat16*>(shared_mem);
    __nv_bfloat16* s_K = s_Q + BM * D;
    __nv_bfloat16* s_V = s_K + BN * D;

    int tid = threadIdx.x;
    int bid_bh = blockIdx.x;
    int batch = bid_bh / H;
    int head = bid_bh % H;
    if (batch >= B || head >= H) return;

    int64_t bh_off = (static_cast<int64_t>(batch) * H + head) * static_cast<int64_t>(S) * D;
    const __nv_bfloat16* Q_base = Q + bh_off;
    const __nv_bfloat16* K_base = K + bh_off;
    const __nv_bfloat16* V_base = V + bh_off;
    __nv_bfloat16* O_base = O + bh_off;
    float* LSE_base = LSE + (batch * H + head) * S;

    int num_kv_blocks = (S + BN - 1) / BN;

    // Process one query block (BM rows)
    int b_m = blockIdx.y;
    int q_start = b_m * BM;
    if (q_start >= S) return;
    int bm_actual = min(q_start + BM, S) - q_start;

    // Load Q tile into shared memory cooperatively
    for (int idx = tid; idx < bm_actual * D; idx += NUM_THREADS) {
        int r = idx / D;
        int c = idx % D;
        s_Q[r * D + c] = Q_base[(q_start + r) * D + c];
    }
    __syncthreads();

    // Each thread computes output for a subset of query rows.
    // Distribute BM rows among NUM_THREADS: rows_per_thread = ceil(BM / NUM_THREADS)
    // Most threads get either floor or ceil rows
    int rows_per_thread_floor = BM / NUM_THREADS;
    int rows_per_thread_ceil = (BM + NUM_THREADS - 1) / NUM_THREADS;
    bool this_thread_gets_extra = (tid < (BM - rows_per_thread_floor * NUM_THREADS));
    int my_num_rows = rows_per_thread_floor + (this_thread_gets_extra ? 1 : 0);

    // Starting row index in [0, BM) for this thread
    int my_row_offset = tid * rows_per_thread_floor + min(tid, BM - rows_per_thread_floor * NUM_THREADS);

    // Per-row state: max (m), sum (l), and output accumulator (o)
    // Register-limited: each thread keeps only its own rows' states
    float m_reg[max(rows_per_thread_ceil, 1)];
    float l_reg[max(rows_per_thread_ceil, 1)];
    float o_reg[max(rows_per_thread_ceil, 1)][D];

    // Initialize
    #pragma unroll
    for (int ri = 0; ri < (my_num_rows > 0 ? my_num_rows : 1); ri++) {
        m_reg[ri] = -INFINITY;
        l_reg[ri] = 0.0f;
        #pragma unroll
        for (int d = 0; d < D; d++) {
            o_reg[ri][d] = 0.0f;
        }
    }

    // Iterate over all KV blocks
    for (int b_n = 0; b_n < num_kv_blocks; b_n++) {
        int kv_start = b_n * BN;
        int bn_actual = min(kv_start + BN, S) - kv_start;

        // Load K tile
        for (int idx = tid; idx < bn_actual * D; idx += NUM_THREADS) {
            int r = idx / D;
            int c = idx % D;
            s_K[r * D + c] = K_base[(kv_start + r) * D + c];
        }
        // Load V tile
        for (int idx = tid; idx < bn_actual * D; idx += NUM_THREADS) {
            int r = idx / D;
            int c = idx % D;
            s_V[r * D + c] = V_base[(kv_start + r) * D + c];
        }
        __syncthreads();

        // Pass 1: compute row-local max over new attn scores
        float new_m[max(rows_per_thread_ceil, 1)];
        #pragma unroll
        for (int ri = 0; ri < (my_num_rows > 0 ? my_num_rows : 1); ri++) {
            new_m[ri] = -INFINITY;
        }

        #pragma unroll
        for (int ri = 0; ri < (my_num_rows > 0 ? my_num_rows : 1); ri++) {
            int qr = my_row_offset + ri;
            if (qr >= bm_actual) continue;
            int qp = q_start + qr;  // absolute query position

            // Compute Q[qr,:] @ K[k,: for k in kv_block]^T
            #pragma unroll
            for (int k_col = 0; k_col < bn_actual; k_col++) {
                int kp = kv_start + k_col;
                float dot = 0.0f;

                // Unrolled dot product for D=128 (32 iterations of 4)
                #pragma unroll
                for (int dd = 0; dd < D; dd += 4) {
                    float q_vals[4] = {
                        __bfloat162float(s_Q[qr * D + dd]),
                        __bfloat162float(s_Q[qr * D + dd + 1]),
                        __bfloat162float(s_Q[qr * D + dd + 2]),
                        __bfloat162float(s_Q[qr * D + dd + 3])
                    };
                    float k_vals[4] = {
                        __bfloat162float(s_K[k_col * D + dd]),
                        __bfloat162float(s_K[k_col * D + dd + 1]),
                        __bfloat162float(s_K[k_col * D + dd + 2]),
                        __bfloat162float(s_K[k_col * D + dd + 3])
                    };
                    dot += q_vals[0]*k_vals[0] + q_vals[1]*k_vals[1] + q_vals[2]*k_vals[2] + q_vals[3]*k_vals[3];
                }

                dot *= inv_sqrt_d;

                // Causal mask: mask if kp > qp
                if (kp <= qp) {
                    if (dot > new_m[ri]) new_m[ri] = dot;
                }
            }
        }

        // Warp reduce to find block max for each row
        // Each warp of 32 threads does intra-warp reduction
        #pragma unroll
        for (int ri = 0; ri < (my_num_rows > 0 ? my_num_rows : 1); ri++) {
            float val = new_m[ri];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                float other = __shfl_down_sync(0xFFFFFFFF, val, offset);
                if (other > val) val = other;
            }
            new_m[ri] = val;
        }

        // Share across warps via shared memory (one value per thread used)
        __shared__ float smem_block_m[32];
        int warp_id = tid / 32;
        int lane = tid % 32;

        // Warp leader writes its new_m to shared mem
        if (lane == 0) {
            #pragma unroll
            for (int ri = 0; ri < (my_num_rows > 0 ? my_num_rows : 1); ri++) {
                smem_block_m[warp_id * max(rows_per_thread_ceil, 1) + ri] = new_m[ri];
            }
        }
        __threadfence_block();

        // All threads read their max from shared mem
        #pragma unroll
        for (int ri = 0; ri < (my_num_rows > 0 ? my_num_rows : 1); ri++) {
            float block_m = -INFINITY;
            #pragma unroll
            for (int w = 0; w < (NUM_THREADS / 32); w++) {
                float candidate = smem_block_m[w * max(rows_per_thread_ceil, 1) + ri];
                if (candidate > block_m) block_m = candidate;
            }
            new_m[ri] = block_m;
        }

        // Pass 2: scale old accumulators, compute attn, accumulate
        #pragma unroll
        for (int ri = 0; ri < (my_num_rows > 0 ? my_num_rows : 1); ri++) {
            int qr = my_row_offset + ri;
            if (qr >= bm_actual) continue;
            int qp = q_start + qr;

            float m_old = m_reg[ri];
            float alpha = expf(m_old - new_m[ri]);

            float local_l = 0.0f;

            #pragma unroll
            for (int k_col = 0; k_col < bn_actual; k_col++) {
                int kp = kv_start + k_col;
                float dot = 0.0f;

                #pragma unroll
                for (int dd = 0; dd < D; dd += 4) {
                    float q_vals[4] = {
                        __bfloat162float(s_Q[qr * D + dd]),
                        __bfloat162float(s_Q[qr * D + dd + 1]),
                        __bfloat162float(s_Q[qr * D + dd + 2]),
                        __bfloat162float(s_Q[qr * D + dd + 3])
                    };
                    float k_vals[4] = {
                        __bfloat162float(s_K[k_col * D + dd]),
                        __bfloat162float(s_K[k_col * D + dd + 1]),
                        __bfloat162float(s_K[k_col * D + dd + 2]),
                        __bfloat162float(s_K[k_col * D + dd + 3])
                    };
                    dot += q_vals[0]*k_vals[0] + q_vals[1]*k_vals[1] + q_vals[2]*k_vals[2] + q_vals[3]*k_vals[3];
                }

                dot *= inv_sqrt_d;

                float attn;
                if (kp <= qp) {
                    attn = expf(dot - new_m[ri]);
                    local_l += attn;

                    // Accumulate output: o += attn * V[k_col, :]
                    #pragma unroll
                    for (int dd = 0; dd < D; dd++) {
                        o_reg[ri][dd] += attn * __bfloat162float(s_V[k_col * D + dd]);
                    }
                }
            }

            // Update accumulators
            m_reg[ri] = new_m[ri];
            l_reg[ri] = l_reg[ri] * alpha + local_l;
            #pragma unroll
            for (int dd = 0; dd < D; dd++) {
                o_reg[ri][dd] = o_reg[ri][dd] * alpha;
            }
        }
    }  // end KV blocks

    // Write output and LSE
    #pragma unroll
    for (int ri = 0; ri < (my_num_rows > 0 ? my_num_rows : 1); ri++) {
        int qr = my_row_offset + ri;
        if (qr >= bm_actual) continue;
        int out_row = q_start + qr;
        float inv_l = 1.0f / fmaxf(l_reg[ri], 1e-12f);

        // Store output
        #pragma unroll
        for (int dd = 0; dd < D; dd++) {
            O_base[static_cast<int64_t>(out_row) * D + dd] = __float2bfloat16(o_reg[ri][dd] * inv_l);
        }
        // Store LSE
        LSE_base[out_row] = m_reg[ri] + logf(fmaxf(l_reg[ri], 1e-12f));
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);

    const __nv_bfloat16* q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* k_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* v_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());

    // Kernel configuration for D=128
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int NUM_THREADS = 128;

    int num_q_blocks = (S + BM - 1) / BM;
    dim3 grid(static_cast<int>(B * H), num_q_blocks);
    dim3 block(NUM_THREADS);

    // Shared memory: BM*D + BN*D + BN*D bf16 elements + some extra for block reduce
    size_t smem_size = (BM * D + BN * D + BN * D) * sizeof(__nv_bfloat16) + NUM_THREADS * sizeof(float);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    float inv_sqrt_d = rsqrtf(static_cast<float>(D));

    causal_attn_kernel<BM, BN, D, NUM_THREADS><<<grid, block, smem_size, stream>>>(
        q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr,
        static_cast<int>(B), static_cast<int>(H),
        static_cast<int>(S),
        inv_sqrt_d
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_causal_d128::run);

}  // namespace tvm_ffi_mha_causal_d128