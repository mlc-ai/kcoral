#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
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

namespace mha_causal_d128 {

constexpr int BLOCK_M = 64;  // Number of Q rows per tile
constexpr int BLOCK_N = 64;  // Number of K rows per tile per step
constexpr int BD      = 128; // Head dimension D=128

template <unsigned int BD_>
__global__ void mha_causal_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
          __nv_bfloat16* __restrict__ O,
    float*                 __restrict__ LSE_out,
    int B, int H, int S,
    float scale_inv_sqrt_d)
{
    extern __shared__ __align__(256) unsigned char smem_raw[];

    // Layout: smem_Q [64][128] bf16, smem_K [64][128] bf16, smem_V [64][128] bf16
    __nv_bfloat16 (*smem_Q)[BD_] = reinterpret_cast<__nv_bfloat16(*)[BD_]>(smem_raw);
    __nv_bfloat16 (*smem_K)[BD_] = reinterpret_cast<__nv_bfloat16(*)[BD_]>(smem_raw + BLOCK_M * BD_ * sizeof(__nv_bfloat16));
    __nv_bfloat16 (*smem_V)[BD_] = reinterpret_cast<__nv_bfloat16(*)[BD_]>(smem_raw + 2 * BLOCK_M * BD_ * sizeof(__nv_bfloat16));

    // Map thread to (q_row_in_block, k_col_in_block)
    // 256 threads -> 4 sweeps of 64 threads each
    int tid = threadIdx.x;   // 0..255

    // Global batch, head indices
    int bh_idx = blockIdx.x;
    int b = bh_idx / H;
    int h = bh_idx % H;

    // Base offsets into global tensors (contiguous [B,H,S,D])
    int64_t b_stride = (int64_t)b * H * S * BD_;
    int64_t h_stride = h * S * BD_;

    // Query position in global sequence (this thread's Q row within the BM-sized output tile)
    // We process 64 Q rows per block, sweeping over 4 k-col positions
    int sweep = tid / 64;       // 0..3
    int thread_in_sweep = tid % 64;  // 0..63
    
    // Each thread computes a unique (q_idx_in_block, k_idx_in_block) pair
    int q_idx_in_block = thread_in_sweep;
    int k_idx_in_block = sweep;

    int q_global = blockIdx.y * BLOCK_M + q_idx_in_block;

    // Early exit if q_global is out of bounds
    if (q_global >= S) return;

    // Per-thread output accumulator (fp32) for the BD=128 output dimensions
    float acc_v[BD_];
    for (int d = 0; d < BD_; ++d) {
        acc_v[d] = 0.f;
    }

    // Online softmax state
    float row_max = -1e20f;
    float row_sum = 0.f;

    // Load Q[q_idx_in_block, :] into registers once
    __nv_bfloat16 q_reg[BD_];
    {
        int d_base = b_stride + h_stride + q_global * BD_;
        for (int i = 0; i < BD_; i += 4) {
            const unsigned short* src = reinterpret_cast<const unsigned short*>(&Q[d_base + i]);
            q_reg[i]   = reinterpret_cast<const __nv_bfloat16&>(src[0]);
            q_reg[i+1] = reinterpret_cast<const __nv_bfloat16&>(src[1]);
            q_reg[i+2] = reinterpret_cast<const __nv_bfloat16&>(src[2]);
            q_reg[i+3] = reinterpret_cast<const __nv_bfloat16&>(src[3]);
        }
    }

    // Loop over K/V tile blocks
    int num_k_steps = (S + BLOCK_N - 1) / BLOCK_N;
    for (int step = 0; step < num_k_steps; ++step) {
        int k_global_base = step * BLOCK_N;

        // ---- Load Q_tile into shared memory (threads 0..63) ----
        {
            int thread_id = tid % 64;
            for (int i = 0; i < BD_; i += 2) {
                int d = i + thread_id;
                if (d < BD_) {
                    int64_t off = b_stride + h_stride + q_global * BD_ + d;
                    volatile __nv_bfloat16* dst = &smem_Q[thread_id][d];
                    *dst = Q[off];
                }
            }
        }

        // ---- Load K_tile into shared memory (threads 0..63) ----
        {
            int thread_id = tid % 64;
            for (int i = 0; i < BD_; i += 2) {
                int d = i + thread_id;
                if (d < BD_) {
                    int k_g = k_global_base + thread_id;
                    int64_t off = b_stride + h_stride + k_g * BD_ + d;
                    volatile __nv_bfloat16* dst = &smem_K[thread_id][d];
                    *dst = (k_g < S) ? K[off] : __float2bfloat16(0.f);
                }
            }
        }

        // ---- Load V_tile into shared memory (threads 64..127) ----
        {
            int thread_id = (tid % 128);
            if (thread_id < 64) {
                for (int i = 0; i < BD_; i += 2) {
                    int d = i + thread_id;
                    if (d < BD_) {
                        int k_g = k_global_base + thread_id;
                        int64_t off = b_stride + h_stride + k_g * BD_ + d;
                        volatile __nv_bfloat16* dst = &smem_V[thread_id][d];
                        *dst = (k_g < S) ? V[off] : __float2bfloat16(0.f);
                    }
                }
            }
        }

        __syncthreads();

        // ---- Compute s = Q[q_idx_in_block, :] dot K[k_global_base + k_idx_in_block, :] ----
        int k_idx = k_global_base + k_idx_in_block;
        float s = 0.f;
        {
            const __nv_bfloat16* k_ptr = smem_K[k_idx_in_block];
            for (int d = 0; d < BD_; d += 4) {
                float qa = __bfloat162float(q_reg[d]);
                float qb = __bfloat162float(q_reg[d+1]);
                float qc = __bfloat162float(q_reg[d+2]);
                float qd = __bfloat162float(q_reg[d+3]);
                float ka = __bfloat162float(k_ptr[d]);
                float kb = __bfloat162float(k_ptr[d+1]);
                float kc = __bfloat162float(k_ptr[d+2]);
                float kd = __bfloat162float(k_ptr[d+3]);
                s += qa * ka + qb * kb + qc * kc + qd * kd;
            }
            s *= scale_inv_sqrt_d;
        }

        // ---- Apply causal mask ----
        if (k_idx > q_global) {
            s = -1e20f;
        }

        // ---- Online softmax: compute alpha ----
        float alpha;
        if (s > row_max) {
            float old_exp_diff = expf(row_max - s);
            row_sum = row_sum * old_exp_diff + 1.f;
            // Scale previous accumulations
            for (int d = 0; d < BD_; ++d) {
                acc_v[d] *= old_exp_diff;
            }
            alpha = 1.f;
            row_max = s;
        } else {
            alpha = expf(s - row_max);
            row_sum += alpha;
        }

        // ---- Accumulate alpha * V[k_idx, :] ----
        {
            const __nv_bfloat16* v_ptr = smem_V[k_idx_in_block];
            for (int d = 0; d < BD_; d += 4) {
                float va = __bfloat162float(v_ptr[d]);
                float vb = __bfloat162float(v_ptr[d+1]);
                float vc = __bfloat162float(v_ptr[d+2]);
                float vd = __bfloat162float(v_ptr[d+3]);
                acc_v[d]   += alpha * va;
                acc_v[d+1] += alpha * vb;
                acc_v[d+2] += alpha * vc;
                acc_v[d+3] += alpha * vd;
            }
        }

        __syncthreads();
    }

    // ---- Finalize: normalize and write O ----
    if (q_global < S && row_sum > 0.f) {
        float inv_sum = 1.f / row_sum;
        int64_t o_off = b_stride + h_stride + q_global * BD_;
        for (int d = 0; d < BD_; d += 4) {
            O[o_off + d]   = __float2bfloat16(acc_v[d]   * inv_sum);
            O[o_off + d+1] = __float2bfloat16(acc_v[d+1] * inv_sum);
            O[o_off + d+2] = __float2bfloat16(acc_v[d+2] * inv_sum);
            O[o_off + d+3] = __float2bfloat16(acc_v[d+3] * inv_sum);
        }
    }

    // ---- Write LSE: only thread 0 of each Q row writes ----
    if (tid % 64 == 0) {
        float lse_val = (row_sum > 0.f) ? (row_max + logf(row_sum)) : (-1e10f);
        // Find the canonical Q row this thread writes
        int q_write = blockIdx.y * BLOCK_M + (tid / 64);
        if (q_write < S) {
            int64_t lse_off = (int64_t)b * H * S + h * S + q_write;
            LSE_out[lse_off] = lse_val;
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE)
{
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);  // Expected to be 128

    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    // Shared memory: 3 tiles of 64*128 bf16 = 3 * 32KB = 96KB
    int64_t smem_bytes = 3LL * BLOCK_M * BD * sizeof(__nv_bfloat16);

    int64_t num_bh = B * H;
    int64_t num_m_tiles = (S + BLOCK_M - 1) / BLOCK_M;
    dim3 grid(num_bh * num_m_tiles, 1, 1);
    dim3 block(256, 1, 1);

    float scale_inv_sqrt_d = rsqrtf(static_cast<float>(D));

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    // Dispatch kernel specialization for BD=128
    mha_causal_kernel<128><<<grid, block, static_cast<size_t>(smem_bytes), stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data,
        static_cast<int>(B), static_cast<int>(H), static_cast<int>(S),
        scale_inv_sqrt_d);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_causal_d128::run);

}  // namespace mha_causal_d128