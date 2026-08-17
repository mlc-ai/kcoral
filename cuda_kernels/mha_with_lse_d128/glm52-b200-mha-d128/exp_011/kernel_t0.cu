#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace mha_cuda {

constexpr int D = 128;
constexpr int HALF_D = 64;
constexpr int D_PAD = 2;
constexpr int D_STRIDE = D + D_PAD;  // 130, bank-conflict-free for row-parallel access
constexpr int BQ = 64;
constexpr int BK = 32;
constexpr int THREADS = 128;

constexpr int Q_SMEM = BQ * D_STRIDE;
constexpr int K_SMEM = BK * D_STRIDE;
constexpr int V_SMEM = BK * D_STRIDE;
constexpr int SMEM_BYTES = (Q_SMEM + K_SMEM + V_SMEM) * (int)sizeof(__nv_bfloat16);

__global__ __launch_bounds__(THREADS, 2)
void mha_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S_val)
{
    extern __shared__ __nv_bfloat16 smem_buf[];
    __nv_bfloat16* Q_tile = smem_buf;
    __nv_bfloat16* K_tile = smem_buf + Q_SMEM;
    __nv_bfloat16* V_tile = smem_buf + Q_SMEM + K_SMEM;

    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;

    int tid = threadIdx.x;
    int q_local = tid % BQ;
    int d_half = tid / BQ;
    int d_offset = d_half * HALF_D;

    constexpr float scale = 0.08838834764831845f;  // 1/sqrt(128)

    int64_t bh_offset = (int64_t)(b * H + h) * S_val * D;
    const __nv_bfloat16* Q_bh = Q + bh_offset;
    const __nv_bfloat16* K_bh = K + bh_offset;
    const __nv_bfloat16* V_bh = V + bh_offset;
    __nv_bfloat16* O_bh = O + bh_offset;
    float* LSE_bh = LSE + (int64_t)(b * H + h) * S_val;

    for (int q_block = 0; q_block < S_val; q_block += BQ) {
        int q_idx = q_block + q_local;

        // Load Q tile [BQ, D] with padding
        for (int i = tid; i < BQ * D; i += THREADS) {
            int row = i / D;
            int col = i % D;
            int gidx = q_block + row;
            Q_tile[row * D_STRIDE + col] =
                (gidx < S_val) ? Q_bh[(int64_t)gidx * D + col] : __float2bfloat16(0.0f);
        }
        __syncthreads();

        const __nv_bfloat162* Q_row =
            reinterpret_cast<const __nv_bfloat162*>(Q_tile + q_local * D_STRIDE);

        float O_acc[HALF_D];
        float m_val = -INFINITY;
        float l_val = 0.0f;
        #pragma unroll
        for (int d = 0; d < HALF_D; d++) O_acc[d] = 0.0f;

        for (int k_block = 0; k_block < S_val; k_block += BK) {
            // Load K and V tiles [BK, D] with padding
            for (int i = tid; i < BK * D; i += THREADS) {
                int row = i / D;
                int col = i % D;
                int gidx = k_block + row;
                if (gidx < S_val) {
                    K_tile[row * D_STRIDE + col] = K_bh[(int64_t)gidx * D + col];
                    V_tile[row * D_STRIDE + col] = V_bh[(int64_t)gidx * D + col];
                } else {
                    K_tile[row * D_STRIDE + col] = __float2bfloat16(0.0f);
                    V_tile[row * D_STRIDE + col] = __float2bfloat16(0.0f);
                }
            }
            __syncthreads();

            // Compute S = Q @ K^T * scale
            float P[BK];
            float m_block = -INFINITY;

            #pragma unroll
            for (int k = 0; k < BK; k++) {
                const __nv_bfloat162* K_row =
                    reinterpret_cast<const __nv_bfloat162*>(K_tile + k * D_STRIDE);
                float sum = 0.0f;
                #pragma unroll
                for (int d = 0; d < HALF_D; d++) {
                    float2 qf = __bfloat1622float2(Q_row[d]);
                    float2 kf = __bfloat1622float2(K_row[d]);
                    sum += qf.x * kf.x + qf.y * kf.y;
                }
                float s_val = sum * scale;
                if (k_block + k >= S_val) s_val = -INFINITY;
                P[k] = s_val;
                m_block = fmaxf(m_block, s_val);
            }

            // Online softmax update
            float m_new = fmaxf(m_val, m_block);
            float correction = (m_val == -INFINITY) ? 0.0f : expf(m_val - m_new);

            float l_block = 0.0f;
            #pragma unroll
            for (int k = 0; k < BK; k++) {
                P[k] = (k_block + k >= S_val) ? 0.0f : expf(P[k] - m_new);
                l_block += P[k];
            }

            l_val = l_val * correction + l_block;

            // Rescale O accumulator
            #pragma unroll
            for (int d = 0; d < HALF_D; d++) {
                O_acc[d] *= correction;
            }

            // O += P @ V
            #pragma unroll
            for (int k = 0; k < BK; k++) {
                float p = P[k];
                const __nv_bfloat162* V_row =
                    reinterpret_cast<const __nv_bfloat162*>(V_tile + k * D_STRIDE + d_offset);
                #pragma unroll
                for (int d = 0; d < HALF_D / 2; d++) {
                    float2 vf = __bfloat1622float2(V_row[d]);
                    O_acc[2*d]     += p * vf.x;
                    O_acc[2*d+1]   += p * vf.y;
                }
            }

            m_val = m_new;
            __syncthreads();
        }

        // Finalize: O = O / l, LSE = m + log(l)
        if (q_idx < S_val) {
            float inv_l = 1.0f / l_val;
            #pragma unroll
            for (int d = 0; d < HALF_D; d++) {
                O_bh[(int64_t)q_idx * D + d_offset + d] =
                    __float2bfloat16(O_acc[d] * inv_l);
            }
            if (d_half == 0) {
                LSE_bh[q_idx] = m_val + logf(l_val);
            }
        }

        __syncthreads();
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE)
{
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = static_cast<int>(Q.size(0));
    int H = static_cast<int>(Q.size(1));
    int S = static_cast<int>(Q.size(2));

    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    int grid = B * H;
    int block = THREADS;

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(mha_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));

    mha_kernel<<<grid, block, SMEM_BYTES, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data, B, H, S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_cuda::run);

}  // namespace mha_cuda