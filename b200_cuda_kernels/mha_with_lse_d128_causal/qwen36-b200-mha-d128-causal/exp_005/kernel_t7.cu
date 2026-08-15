#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
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

namespace mha_causal_d128 {

constexpr int BLOCK_M = 64;   // Q rows per block tile
constexpr int BLOCK_N = 64;   // K/V rows per step tile
constexpr int GROUP_D = 4;    // Output dims per thread

__global__ void mha_causal_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
          __nv_bfloat16* __restrict__ O,
    float*                 __restrict__ LSE_out,
    int B, int H, int S, int D,
    int64_t q_stride_q, int64_t q_stride_d,
    int64_t k_stride_q, int64_t k_stride_d,
    int64_t v_stride_q, int64_t v_stride_d,
    int64_t o_stride_q, int64_t o_stride_d,
    float scale)
{
    extern __shared__ __align__(256) unsigned char smem_raw[];

    __nv_bfloat16* smem_Q = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* smem_K = reinterpret_cast<__nv_bfloat16*>(smem_raw + BLOCK_M * D * sizeof(__nv_bfloat16));
    __nv_bfloat16* smem_V = reinterpret_cast<__nv_bfloat16*>(smem_raw + (BLOCK_M + BLOCK_N) * D * sizeof(__nv_bfloat16));

    int bh_idx = blockIdx.x;
    int b = bh_idx / H;
    int h = bh_idx % H;

    int64_t base_Q = static_cast<int64_t>(b) * q_stride_q * S + static_cast<int64_t>(h) * q_stride_q;
    int64_t base_K = base_Q;  // Same assumed layout
    int64_t base_V = base_Q;
    int64_t base_O = (int64_t)b * o_stride_q * S + (int64_t)h * o_stride_q;

    int q_tile_start = blockIdx.y * BLOCK_M;
    if (q_tile_start >= S) return;

    int tid = threadIdx.x;      // 0..63
    int d0 = tid * GROUP_D;     // Output dim group [d0..d0+GROUP_D-1]
    if (d0 + GROUP_D > D) return;

    int q_local = tid;           // Query row within this tile (0..63)
    int q_global = q_tile_start + q_local;

    // Online softmax state
    float row_max = -FLT_MAX;
    float row_sum = 0.f;
    float acc[GROUP_D] = {0.f, 0.f, 0.f, 0.f};

    // Load Q[q_global, :] into shared memory once at beginning
    {
        int ldim = (D + 2) / 4;  // Unroll by 4 bf16 = 8 bytes
        for (int i = tid; i < ldim; i += BLOCK_M) {
            int idx = i * 4;
            int off = base_Q + q_global * q_stride_q + idx;
            smem_Q[q_local * D + idx]     = Q[off];
            smem_Q[q_local * D + idx + 1] = Q[off + 1];
            smem_Q[q_local * D + idx + 2] = Q[off + 2];
            smem_Q[q_local * D + idx + 3] = Q[off + 3];
        }
    }
    __syncthreads();

    // Pointers into our row of smem_Q
    const __nv_bfloat16* q_row = smem_Q + q_local * D;

    // Iterate over K/V tiles
    int num_steps = (S + BLOCK_N - 1) / BLOCK_N;
    for (int step = 0; step < num_steps; ++step) {
        int k_tile_start = step * BLOCK_N;

        // ---- Cooperative load K tile ----
        {
            int ldim = (D + 2) / 4;
            for (int i = tid; i < BLOCK_N * ldim; i += BLOCK_M) {
                int kn = i / ldim;
                int kd = (i % ldim) * 4;
                int k_global_val = k_tile_start + kn;
                int off = base_K + k_global_val * k_stride_q + kd;
                smem_K[kn * D + kd]     = (k_global_val < S) ? K[off] : __float2bfloat16(0.f);
                smem_K[kn * D + kd + 1] = (k_global_val < S) ? K[off + 1] : __float2bfloat16(0.f);
                smem_K[kn * D + kd + 2] = (k_global_val < S) ? K[off + 2] : __float2bfloat16(0.f);
                smem_K[kn * D + kd + 3] = (k_global_val < S) ? K[off + 3] : __float2bfloat16(0.f);
            }
        }

        // ---- Cooperative load V tile ----
        {
            int ldim = (D + 2) / 4;
            for (int i = tid; i < BLOCK_N * ldim; i += BLOCK_M) {
                int kn = i / ldim;
                int kd = (i % ldim) * 4;
                int k_global_val = k_tile_start + kn;
                int off = base_V + k_global_val * v_stride_q + kd;
                smem_V[kn * D + kd]     = (k_global_val < S) ? V[off] : __float2bfloat16(0.f);
                smem_V[kn * D + kd + 1] = (k_global_val < S) ? V[off + 1] : __float2bfloat16(0.f);
                smem_V[kn * D + kd + 2] = (k_global_val < S) ? V[off + 2] : __float2bfloat16(0.f);
                smem_V[kn * D + kd + 3] = (k_global_val < S) ? V[off + 3] : __float2bfloat16(0.f);
            }
        }
        __syncthreads();

        // ---- Compute attention and accumulate ----
        for (int kn = 0; kn < BLOCK_N; ++kn) {
            int k_global = k_tile_start + kn;

            // Dot product: s = Q[q_global,:] · K[k_global,:]
            float s = 0.f;
            const __nv_bfloat16* k_row = smem_K + kn * D;
            for (int d = 0; d < D; d += 4) {
                float qa = __bfloat162float(q_row[d]);
                float qb = __bfloat162float(q_row[d + 1]);
                float qc = __bfloat162float(q_row[d + 2]);
                float qd = __bfloat162float(q_row[d + 3]);
                float ka = __bfloat162float(k_row[d]);
                float kb = __bfloat162float(k_row[d + 1]);
                float kc = __bfloat162float(k_row[d + 2]);
                float kd_val = __bfloat162float(k_row[d + 3]);
                s += qa*ka + qb*kb + qc*kc + qd*kd_val;
            }
            s *= scale;

            // Causal mask
            if (k_global > q_global) {
                s = -FLT_MAX;
            }

            // Online softmax update
            if (s > row_max) {
                float ratio = expf(row_max - s);
                for (int di = 0; di < GROUP_D; ++di) {
                    acc[di] *= ratio;
                }
                row_sum *= ratio;
                row_max = s;
                row_sum += 1.f;
                // Accumulate V
                const __nv_bfloat16* v_row = smem_V + kn * D;
                for (int di = 0; di < GROUP_D; ++di) {
                    acc[di] += __bfloat162float(v_row[d0 + di]);
                }
            } else {
                float alpha = expf(s - row_max);
                row_sum += alpha;
                const __nv_bfloat16* v_row = smem_V + kn * D;
                for (int di = 0; di < GROUP_D; ++di) {
                    acc[di] += alpha * __bfloat162float(v_row[d0 + di]);
                }
            }
        }
        __syncthreads();
    }

    // Normalize and write output
    if (q_global < S && row_sum > 0.f) {
        float inv_sum = 1.f / row_sum;
        int64_t o_off = base_O + q_global * o_stride_q + d0;
        for (int di = 0; di < GROUP_D; ++di) {
            O[o_off + di * o_stride_d] = __float2bfloat16(acc[di] * inv_sum);
        }
        if (tid == 0) {
            int64_t lse_off = static_cast<int64_t>(bh_idx) * S + q_global;
            LSE_out[lse_off] = row_max + logf(row_sum);
        }
    } else if (q_global < S) {
        int64_t o_off = base_O + q_global * o_stride_q + d0;
        for (int di = 0; di < GROUP_D; ++di) {
            O[o_off + di * o_stride_d] = __float2bfloat16(0.f);
        }
        if (tid == 0) {
            int64_t lse_off = static_cast<int64_t>(bh_idx) * S + q_global;
            LSE_out[lse_off] = -1e10f;
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
    int64_t D = Q.size(3);

    int64_t q_stride_q = Q.stride(2);
    int64_t q_stride_d = Q.stride(3);
    int64_t k_stride_q = K.stride(2);
    int64_t k_stride_d = K.stride(3);
    int64_t v_stride_q = V.stride(2);
    int64_t v_stride_d = V.stride(3);
    int64_t o_stride_q = O.stride(2);
    int64_t o_stride_d = O.stride(3);

    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    int64_t num_bh = B * H;
    int64_t num_m_tiles = (S + BLOCK_M - 1) / BLOCK_M;

    // Shared memory: smem_Q[BLOCK_M][D] + smem_K[BLOCK_N][D] + smem_V[BLOCK_N][D]
    int64_t smem_bytes = (BLOCK_M + 2LL * BLOCK_N) * D * sizeof(__nv_bfloat16);

    dim3 grid(num_bh, num_m_tiles, 1);
    dim3 block(BLOCK_M, 1, 1);

    float scale = rsqrtf(static_cast<float>(D));

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    mha_causal_kernel<<<grid, block, static_cast<size_t>(smem_bytes), stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data,
        static_cast<int>(B), static_cast<int>(H), static_cast<int>(S), static_cast<int>(D),
        q_stride_q, q_stride_d,
        k_stride_q, k_stride_d,
        v_stride_q, v_stride_d,
        o_stride_q, o_stride_d,
        scale);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_causal_d128::run);

}  // namespace mha_causal_d128