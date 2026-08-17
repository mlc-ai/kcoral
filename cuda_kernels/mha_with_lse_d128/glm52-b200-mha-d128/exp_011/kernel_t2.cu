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
constexpr int BQ = 128;
constexpr int BK = 64;
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int Q_PER_WARP = BQ / WARPS;  // 16
constexpr int TPR = 32 / Q_PER_WARP;    // 2
constexpr int DPT = D_VAL / TPR;        // 64

constexpr int SMEM_Q = BQ * D_VAL;
constexpr int SMEM_K = BK * D_VAL;
constexpr int SMEM_V = BK * D_VAL;
constexpr int SMEM_TOTAL = (SMEM_Q + 2 * SMEM_K + 2 * SMEM_V) * (int)sizeof(__nv_bfloat16);

constexpr float LOG2E = 1.4426950408889634f;

__device__ __forceinline__ float fast_exp2f(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src) {
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem_src));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

__global__ __launch_bounds__(THREADS, 2)
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
    __nv_bfloat16* V_smem = K_smem + 2 * SMEM_K;

    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    int q_start = blockIdx.y * BQ;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int q_row = lane_id / TPR;
    int d_part = lane_id % TPR;
    int d_offset = d_part * DPT;
    int q_local = warp_id * Q_PER_WARP + q_row;
    int q_idx = q_start + q_local;

    constexpr float scale = 0.08838834764831845f;

    int64_t bh_off = (int64_t)(b * H + h) * S_val * D_VAL;
    const __nv_bfloat16* Q_bh = Q + bh_off;
    const __nv_bfloat16* K_bh = K + bh_off;
    const __nv_bfloat16* V_bh = V + bh_off;
    __nv_bfloat16* O_bh = O + bh_off;
    float* LSE_bh = LSE + (int64_t)(b * H + h) * S_val;

    // Issue first K/V load via cp.async (overlaps with Q load below)
    {
        __nv_bfloat16* Kb = K_smem;
        __nv_bfloat16* Vb = V_smem;
        int total = BK * D_VAL / 8;
        for (int i = tid; i < total; i += THREADS) {
            int r = i / (D_VAL / 8);
            int c = i % (D_VAL / 8);
            int g_row = r;
            if (g_row < S_val) {
                cp_async_16(Kb + r * D_VAL + c * 8, K_bh + (int64_t)g_row * D_VAL + c * 8);
                cp_async_16(Vb + r * D_VAL + c * 8, V_bh + (int64_t)g_row * D_VAL + c * 8);
            }
        }
        cp_async_commit();
    }

    // Load Q tile [BQ, D] with int4 vectorized loads (overlaps with K/V cp.async)
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

    // Load Q row into registers
    __nv_bfloat162 Q_reg[DPT / 2];
    {
        const __nv_bfloat162* Q_row = reinterpret_cast<const __nv_bfloat162*>(
            Q_smem + q_local * D_VAL + d_offset);
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

    int num_k_blocks = (S_val + BK - 1) / BK;
    int buf = 0;

    for (int iter = 0; iter < num_k_blocks; iter++) {
        int k_block = iter * BK;
        int next_buf = 1 - buf;

        // Issue next K/V load (if not last)
        if (iter + 1 < num_k_blocks) {
            __nv_bfloat16* Kb = K_smem + next_buf * SMEM_K;
            __nv_bfloat16* Vb = V_smem + next_buf * SMEM_V;
            int next_k = k_block + BK;
            int total = BK * D_VAL / 8;
            for (int i = tid; i < total; i += THREADS) {
                int r = i / (D_VAL / 8);
                int c = i % (D_VAL / 8);
                int g_row = next_k + r;
                if (g_row < S_val) {
                    cp_async_16(Kb + r * D_VAL + c * 8, K_bh + (int64_t)g_row * D_VAL + c * 8);
                    cp_async_16(Vb + r * D_VAL + c * 8, V_bh + (int64_t)g_row * D_VAL + c * 8);
                }
            }
            cp_async_commit();
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();

        __nv_bfloat16* K_cur = K_smem + buf * SMEM_K;
        __nv_bfloat16* V_cur = V_smem + buf * SMEM_V;

        // Pass 1: Compute P[k] = Q@K^T * scale, find m_block
        float m_block = -INFINITY;
        #pragma unroll 1
        for (int k = 0; k < BK; k++) {
            float sum = 0.0f;
            const __nv_bfloat162* K_row = reinterpret_cast<const __nv_bfloat162*>(
                K_cur + k * D_VAL + d_offset);
            #pragma unroll
            for (int d = 0; d < DPT / 2; d++) {
                float2 qf = __bfloat1622float2(Q_reg[d]);
                float2 kf = __bfloat1622float2(K_row[d]);
                sum = __fmaf_rn(qf.x, kf.x, sum);
                sum = __fmaf_rn(qf.y, kf.y, sum);
            }
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, 1);
            float p = sum * scale;
            if (k_block + k >= S_val) p = -INFINITY;
            m_block = fmaxf(m_block, p);
        }

        float m_new = fmaxf(m_val, m_block);
        float correction = (m_val == -INFINITY) ? 0.0f : fast_exp2f((m_val - m_new) * LOG2E);

        #pragma unroll
        for (int d = 0; d < DPT; d++)
            O_acc[d] *= correction;

        // Pass 2: Recompute P[k], accumulate l_block and O += P @ V
        float l_block = 0.0f;
        #pragma unroll 1
        for (int k = 0; k < BK; k++) {
            float sum = 0.0f;
            const __nv_bfloat162* K_row = reinterpret_cast<const __nv_bfloat162*>(
                K_cur + k * D_VAL + d_offset);
            #pragma unroll
            for (int d = 0; d < DPT / 2; d++) {
                float2 qf = __bfloat1622float2(Q_reg[d]);
                float2 kf = __bfloat1622float2(K_row[d]);
                sum = __fmaf_rn(qf.x, kf.x, sum);
                sum = __fmaf_rn(qf.y, kf.y, sum);
            }
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, 1);
            float p = sum * scale;
            if (k_block + k >= S_val) p = 0.0f;
            else p = fast_exp2f((p - m_new) * LOG2E);

            l_block += p;

            const __nv_bfloat162* V_row = reinterpret_cast<const __nv_bfloat162*>(
                V_cur + k * D_VAL + d_offset);
            #pragma unroll
            for (int d = 0; d < DPT / 2; d++) {
                float2 vf = __bfloat1622float2(V_row[d]);
                O_acc[2*d]     = __fmaf_rn(p, vf.x, O_acc[2*d]);
                O_acc[2*d + 1] = __fmaf_rn(p, vf.y, O_acc[2*d + 1]);
            }
        }

        l_val = l_val * correction + l_block;
        m_val = m_new;

        buf = next_buf;
        __syncthreads();
    }

    if (q_idx < S_val) {
        float inv_l = 1.0f / l_val;
        #pragma unroll
        for (int d = 0; d < DPT; d++)
            O_bh[(int64_t)q_idx * D_VAL + d_offset + d] = __float2bfloat16(O_acc[d] * inv_l);
        if (d_part == 0)
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