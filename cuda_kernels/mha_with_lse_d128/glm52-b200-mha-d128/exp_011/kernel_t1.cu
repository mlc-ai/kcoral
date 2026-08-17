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

constexpr int D_VAL = 128;
constexpr int BQ = 16;
constexpr int BK = 32;
constexpr int THREADS = 128;
constexpr int TPR = THREADS / BQ;  // 8 threads per row
constexpr int DPT = D_VAL / TPR;   // 16 elements per thread

constexpr int SMEM_Q = BQ * D_VAL;
constexpr int SMEM_K = BK * D_VAL;
constexpr int SMEM_V = BK * D_VAL;
constexpr int SMEM_TOTAL = (SMEM_Q + SMEM_K + SMEM_V) * (int)sizeof(__nv_bfloat16);

__global__ __launch_bounds__(THREADS, 4)
void mha_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S_val)
{
    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* Q_smem = smem;
    __nv_bfloat16* K_smem = smem + SMEM_Q;
    __nv_bfloat16* V_smem = smem + SMEM_Q + SMEM_K;

    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    int q_block = blockIdx.y;
    int q_start = q_block * BQ;

    int tid = threadIdx.x;
    int row = tid / TPR;
    int d_grp = tid % TPR;
    int d_offset = d_grp * DPT;

    constexpr float scale = 0.08838834764831845f;

    int64_t bh_off = (int64_t)(b * H + h) * S_val * D_VAL;
    const __nv_bfloat16* Q_bh = Q + bh_off;
    const __nv_bfloat16* K_bh = K + bh_off;
    const __nv_bfloat16* V_bh = V + bh_off;
    __nv_bfloat16* O_bh = O + bh_off;
    float* LSE_bh = LSE + (int64_t)(b * H + h) * S_val;

    // Load Q tile with vectorized int4 loads
    {
        const int4* src = reinterpret_cast<const int4*>(Q_bh);
        int4* dst = reinterpret_cast<int4*>(Q_smem);
        int total = BQ * D_VAL / 8;
        for (int i = tid; i < total; i += THREADS) {
            int r = i / (D_VAL / 8);
            int c = i % (D_VAL / 8);
            int g_row = q_start + r;
            if (g_row < S_val)
                dst[i] = src[g_row * (D_VAL / 8) + c];
            else
                dst[i] = make_int4(0, 0, 0, 0);
        }
    }
    __syncthreads();

    // Load Q row into registers (8 bf16x2 = 8 registers)
    __nv_bfloat162 Q_reg[DPT / 2];
    {
        const __nv_bfloat162* Q_row = reinterpret_cast<const __nv_bfloat162*>(Q_smem + row * D_VAL + d_offset);
        #pragma unroll
        for (int d = 0; d < DPT / 2; d++)
            Q_reg[d] = Q_row[d];
    }
    __syncthreads();

    float O_acc[DPT];
    #pragma unroll
    for (int d = 0; d < DPT; d++) O_acc[d] = 0.0f;

    float m_val = -INFINITY;
    float l_val = 0.0f;

    for (int k_block = 0; k_block < S_val; k_block += BK) {
        // Load K and V tiles with vectorized int4 loads
        {
            const int4* K_src = reinterpret_cast<const int4*>(K_bh);
            const int4* V_src = reinterpret_cast<const int4*>(V_bh);
            int4* K_dst = reinterpret_cast<int4*>(K_smem);
            int4* V_dst = reinterpret_cast<int4*>(V_smem);
            int total = BK * D_VAL / 8;
            for (int i = tid; i < total; i += THREADS) {
                int r = i / (D_VAL / 8);
                int c = i % (D_VAL / 8);
                int g_row = k_block + r;
                if (g_row < S_val) {
                    K_dst[i] = K_src[g_row * (D_VAL / 8) + c];
                    V_dst[i] = V_src[g_row * (D_VAL / 8) + c];
                } else {
                    K_dst[i] = make_int4(0, 0, 0, 0);
                    V_dst[i] = make_int4(0, 0, 0, 0);
                }
            }
        }
        __syncthreads();

        // Compute P = Q @ K^T * scale
        float P[BK];
        float m_block = -INFINITY;

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            const __nv_bfloat162* K_row = reinterpret_cast<const __nv_bfloat162*>(K_smem + k * D_VAL + d_offset);
            float sum = 0.0f;
            #pragma unroll
            for (int d = 0; d < DPT / 2; d++) {
                float2 qf = __bfloat1622float2(Q_reg[d]);
                float2 kf = __bfloat1622float2(K_row[d]);
                sum = __fmaf_rn(qf.x, kf.x, sum);
                sum = __fmaf_rn(qf.y, kf.y, sum);
            }
            // Reduce across 8 threads using warp shuffles
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, 1, TPR);
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, 2, TPR);
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, 4, TPR);

            float s_val = sum * scale;
            if (k_block + k >= S_val) s_val = -INFINITY;
            P[k] = s_val;
            m_block = fmaxf(m_block, s_val);
        }

        // Online softmax update
        float m_new = fmaxf(m_val, m_block);
        float correction = (m_val == -INFINITY) ? 0.0f : __expf(m_val - m_new);

        float l_block = 0.0f;
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            P[k] = (k_block + k >= S_val) ? 0.0f : __expf(P[k] - m_new);
            l_block += P[k];
        }

        l_val = l_val * correction + l_block;

        // Rescale O accumulator
        #pragma unroll
        for (int d = 0; d < DPT; d++)
            O_acc[d] *= correction;

        // O += P @ V
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float p = P[k];
            const __nv_bfloat162* V_row = reinterpret_cast<const __nv_bfloat162*>(V_smem + k * D_VAL + d_offset);
            #pragma unroll
            for (int d = 0; d < DPT / 2; d++) {
                float2 vf = __bfloat1622float2(V_row[d]);
                O_acc[2*d]     = __fmaf_rn(p, vf.x, O_acc[2*d]);
                O_acc[2*d + 1] = __fmaf_rn(p, vf.y, O_acc[2*d + 1]);
            }
        }

        m_val = m_new;
        __syncthreads();
    }

    // Write output
    int q_idx = q_start + row;
    if (q_idx < S_val) {
        float inv_l = 1.0f / l_val;
        #pragma unroll
        for (int d = 0; d < DPT; d++)
            O_bh[(int64_t)q_idx * D_VAL + d_offset + d] = __float2bfloat16(O_acc[d] * inv_l);
        if (d_grp == 0)
            LSE_bh[q_idx] = m_val + __logf(l_val);
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

    int q_blocks = (S + BQ - 1) / BQ;
    dim3 grid(B * H, q_blocks);
    dim3 block(THREADS);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(mha_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOTAL));

    mha_kernel<<<grid, block, SMEM_TOTAL, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data, B, H, S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_cuda::run);

}  // namespace mha_cuda