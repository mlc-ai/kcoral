#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                       \
    cudaError_t _e = (call);                                        \
    if (_e != cudaSuccess) {                                        \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                 \
                cudaGetErrorString(_e), __FILE__, __LINE__);        \
        exit(1);                                                    \
    }                                                               \
} while (0)

namespace flash_attn {

// Tile sizes
constexpr int BM = 64;          // Q rows per block
constexpr int BN = 64;          // KV cols per tile
constexpr int D = 128;          // Head dimension
constexpr int KV_STRIDE = 130;  // Padded stride to reduce bank conflicts
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int ROWS_PER_WARP = BM / WARPS;    // 8
constexpr int COLS_S_PER_LANE = BN / 32;     // 2
constexpr int COLS_O_PER_LANE = D / 32;      // 4

__global__ void flash_attn_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S)
{
    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    int q_block = blockIdx.y;
    int q_start = q_block * BM;

    if (q_start >= S) return;

    int q_end_val = min(q_start + BM, S);
    int q_rows = q_end_val - q_start;

    const __nv_bfloat16* Q_bh = Q + (size_t)(b * H + h) * S * D;
    const __nv_bfloat16* K_bh = K + (size_t)(b * H + h) * S * D;
    const __nv_bfloat16* V_bh = V + (size_t)(b * H + h) * S * D;
    __nv_bfloat16* O_bh = O + (size_t)(b * H + h) * S * D;
    float* LSE_bh = LSE + (size_t)(b * H + h) * S;

    extern __shared__ char smem[];
    __nv_bfloat16* smem_q  = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* smem_kv = smem_q + BM * D;
    __nv_bfloat16* smem_p  = smem_kv + BN * KV_STRIDE;

    int tid        = threadIdx.x;
    int warp_id    = tid / 32;
    int lane_id    = tid % 32;
    int q_row_base = warp_id * ROWS_PER_WARP;
    int k_col_base = lane_id * COLS_S_PER_LANE;
    int d_col_base = lane_id * COLS_O_PER_LANE;

    // Register accumulators
    float acc_o[ROWS_PER_WARP][COLS_O_PER_LANE];
    float m_i[ROWS_PER_WARP];
    float l_i[ROWS_PER_WARP];

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        m_i[r] = -INFINITY;
        l_i[r] = 0.0f;
        #pragma unroll
        for (int c = 0; c < COLS_O_PER_LANE; c++)
            acc_o[r][c] = 0.0f;
    }

    const float scale = 0.08838834764831845f;  // 1/sqrt(128)

    // Load Q tile to shared memory (vectorized 4-byte loads)
    for (int i = tid * 2; i < BM * D; i += THREADS * 2) {
        int row = i / D, col = i % D;
        uint32_t val = (row < q_rows)
            ? *reinterpret_cast<const uint32_t*>(Q_bh + (q_start + row) * D + col)
            : 0u;
        *reinterpret_cast<uint32_t*>(smem_q + row * D + col) = val;
    }
    __syncthreads();

    // Iterate over KV tiles (causal: only up to q_end)
    for (int kv_start = 0; kv_start < q_end_val; kv_start += BN) {
        int kv_len = min(kv_start + BN, S) - kv_start;

        // ---- Load K tile ----
        for (int i = tid * 2; i < BN * D; i += THREADS * 2) {
            int row = i / D, col = i % D;
            uint32_t val = (row < kv_len)
                ? *reinterpret_cast<const uint32_t*>(K_bh + (kv_start + row) * D + col)
                : 0u;
            *reinterpret_cast<uint32_t*>(smem_kv + row * KV_STRIDE + col) = val;
        }
        __syncthreads();

        // ---- S = Q @ K^T  (in registers) ----
        float s_val[ROWS_PER_WARP][COLS_S_PER_LANE];
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; r++)
            #pragma unroll
            for (int c = 0; c < COLS_S_PER_LANE; c++)
                s_val[r][c] = 0.0f;

        #pragma unroll
        for (int d = 0; d < D; d += 2) {
            __nv_bfloat162 q_val[ROWS_PER_WARP];
            #pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; r++)
                q_val[r] = *reinterpret_cast<__nv_bfloat162*>(
                    smem_q + (q_row_base + r) * D + d);

            __nv_bfloat162 k_val[COLS_S_PER_LANE];
            #pragma unroll
            for (int c = 0; c < COLS_S_PER_LANE; c++)
                k_val[c] = *reinterpret_cast<__nv_bfloat162*>(
                    smem_kv + (k_col_base + c) * KV_STRIDE + d);

            #pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; r++) {
                float2 qf = __bfloat1622float2(q_val[r]);
                #pragma unroll
                for (int c = 0; c < COLS_S_PER_LANE; c++) {
                    float2 kf = __bfloat1622float2(k_val[c]);
                    s_val[r][c] += qf.x * kf.x + qf.y * kf.y;
                }
            }
        }

        // Scale + causal mask
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; r++) {
            int q_row = q_start + q_row_base + r;
            #pragma unroll
            for (int c = 0; c < COLS_S_PER_LANE; c++) {
                int k_col = kv_start + k_col_base + c;
                s_val[r][c] *= scale;
                if (k_col > q_row) s_val[r][c] = -INFINITY;
            }
        }

        // ---- Online softmax + P store ----
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; r++) {
            float s0 = s_val[r][0], s1 = s_val[r][1];

            // Row max via warp reduction
            float local_max = fmaxf(s0, s1);
            local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, 16));
            local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, 8));
            local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, 4));
            local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, 2));
            local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, 1));

            float m_old   = m_i[r];
            float m_new   = fmaxf(m_old, local_max);
            float rescale = (m_old == -INFINITY) ? 0.0f : __expf(m_old - m_new);

            float p0 = (s0 == -INFINITY) ? 0.0f : __expf(s0 - m_new);
            float p1 = (s1 == -INFINITY) ? 0.0f : __expf(s1 - m_new);

            // Store P as bf16
            smem_p[(q_row_base + r) * BN + k_col_base]     = __float2bfloat16(p0);
            smem_p[(q_row_base + r) * BN + k_col_base + 1] = __float2bfloat16(p1);

            // Row sum via warp reduction
            float local_sum = p0 + p1;
            local_sum += __shfl_xor_sync(0xffffffff, local_sum, 16);
            local_sum += __shfl_xor_sync(0xffffffff, local_sum, 8);
            local_sum += __shfl_xor_sync(0xffffffff, local_sum, 4);
            local_sum += __shfl_xor_sync(0xffffffff, local_sum, 2);
            local_sum += __shfl_xor_sync(0xffffffff, local_sum, 1);

            l_i[r] = l_i[r] * rescale + local_sum;
            m_i[r] = m_new;

            #pragma unroll
            for (int c = 0; c < COLS_O_PER_LANE; c++)
                acc_o[r][c] *= rescale;
        }
        __syncthreads();

        // ---- Load V tile (overwrites K in smem_kv) ----
        for (int i = tid * 2; i < BN * D; i += THREADS * 2) {
            int row = i / D, col = i % D;
            uint32_t val = (row < kv_len)
                ? *reinterpret_cast<const uint32_t*>(V_bh + (kv_start + row) * D + col)
                : 0u;
            *reinterpret_cast<uint32_t*>(smem_kv + row * KV_STRIDE + col) = val;
        }
        __syncthreads();

        // ---- O += P @ V ----
        #pragma unroll
        for (int j = 0; j < BN; j++) {
            uint32_t v01_packed = *reinterpret_cast<uint32_t*>(
                smem_kv + j * KV_STRIDE + d_col_base);
            uint32_t v23_packed = *reinterpret_cast<uint32_t*>(
                smem_kv + j * KV_STRIDE + d_col_base + 2);
            float2 vf01 = __bfloat1622float2(
                *reinterpret_cast<__nv_bfloat162*>(&v01_packed));
            float2 vf23 = __bfloat1622float2(
                *reinterpret_cast<__nv_bfloat162*>(&v23_packed));

            #pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; r++) {
                float p = __bfloat162float(smem_p[(q_row_base + r) * BN + j]);
                acc_o[r][0] += p * vf01.x;
                acc_o[r][1] += p * vf01.y;
                acc_o[r][2] += p * vf23.x;
                acc_o[r][3] += p * vf23.y;
            }
        }
        __syncthreads();
    }

    // ---- Store O and LSE ----
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        int q_row = q_start + q_row_base + r;
        if (q_row < S) {
            float inv_l = (l_i[r] > 0.0f) ? (1.0f / l_i[r]) : 0.0f;
            #pragma unroll
            for (int c = 0; c < COLS_O_PER_LANE; c++)
                O_bh[q_row * D + d_col_base + c] =
                    __float2bfloat16(acc_o[r][c] * inv_l);
        }
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; r++) {
            int q_row = q_start + q_row_base + r;
            if (q_row < S)
                LSE_bh[q_row] =
                    (l_i[r] > 0.0f) ? (m_i[r] + logf(l_i[r])) : -INFINITY;
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = (int)Q.size(0);
    int H = (int)Q.size(1);
    int S = (int)Q.size(2);
    // D = 128 is fixed by constexpr in the kernel

    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data       = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data             = static_cast<float*>(LSE.data_ptr());

    dim3 grid(B * H, (S + BM - 1) / BM);
    dim3 block(THREADS);
    int smem_bytes = BM * D * 2 + BN * KV_STRIDE * 2 + BN * BN * 2;

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    cudaFuncSetAttribute(flash_attn_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    flash_attn_kernel<<<grid, block, smem_bytes, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data, B, H, S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flash_attn::run);

}  // namespace flash_attn