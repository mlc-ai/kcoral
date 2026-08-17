#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t _e = (call); \
        if (_e != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
            exit(1); \
        } \
    } while(0)

namespace mha_bwd_ns {

static constexpr int HEAD_DIM = 128;
static constexpr int NTHREADS = 1024;
static constexpr int NWARPS = NTHREADS / 32;   // 32 warps
static constexpr int COLS_PER_THREAD = HEAD_DIM / 32; // 4

__device__ __forceinline__ float warp_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return __shfl_sync(0xFFFFFFFF, val, 0);
}

__device__ __forceinline__ void load_4cols(const __nv_bfloat16* ptr, int col_idx,
    float& v0, float& v1, float& v2, float& v3) {
    uint4 u = reinterpret_cast<const uint4*>(ptr)[col_idx];
    v0 = __uint2float_rn(__bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&u.x)));
    v1 = __uint2float_rn(__bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&u.y)));
    v2 = __uint2float_rn(__bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&u.z)));
    v3 = __uint2float_rn(__bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&u.w)));
}

__global__ void mha_bwd_dQ(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    float* __restrict__ m_buf,
    int B, int H, int S, int D)
{
    int bh = blockIdx.y;
    if (bh >= B * H) return;
    int b = bh / H;
    int h = bh % H;

    int64_t base = (int64_t)(b * H + h) * S * D;
    int64_t l_off = (int64_t)(b * H + h) * S;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    int qi = blockIdx.x * NWARPS + warp_id;
    if (qi >= S) return;

    float scale = rsqrtf(static_cast<float>(D));
    float L_qi = L[l_off + qi];

    // Base pointer to q-th row start for each matrix
    const __nv_bfloat16* Q_row = Q + base + qi * D;
    const __nv_bfloat16* dO_row = dO + base + qi * D;
    __nv_bfloat16* dQ_row = dQ + base + qi * D;

    // Load Q[qi] and dO[qi] (constant for this warp)
    float q_vals[COLS_PER_THREAD];
    float dO_vals[COLS_PER_THREAD];
    for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
        int col = lane_id * COLS_PER_THREAD + cd;
        q_vals[cd] = __bfloat162float(Q_row[col]);
        dO_vals[cd] = __bfloat162float(dO_row[col]);
    }

    float dq_vals[COLS_PER_THREAD] = {};
    float ak_vals[COLS_PER_THREAD] = {};
    float m_sum = 0.0f;

    // Iterate k-tiles: load tile of K,V into registers via warp-level cooperation
    for (int kj = 0; kj <= qi; kj++) {
        const __nv_bfloat16* Kj = K + base + kj * D;
        const __nv_bfloat16* Vj = V + base + kj * D;

        float k_vals[COLS_PER_THREAD];
        float v_vals[COLS_PER_THREAD];
        for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
            int col = lane_id * COLS_PER_THREAD + cd;
            k_vals[cd] = __bfloat162float(Kj[col]);
            v_vals[cd] = __bfloat162float(Vj[col]);
        }

        float score_tid = 0.0f;
        float dP_tid = 0.0f;
        #pragma unroll
        for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
            score_tid += q_vals[cd] * k_vals[cd];
            dP_tid += dO_vals[cd] * v_vals[cd];
        }

        float score_full = warp_sum(score_tid);
        float dP_full = warp_sum(dP_tid);

        float attn = expf(score_full * scale - L_qi);
        float adP = attn * dP_full;
        m_sum += adP;

        float scaled_adP = adP * scale;
        #pragma unroll
        for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
            dq_vals[cd] += scaled_adP * k_vals[cd];
            ak_vals[cd] += attn * scale * k_vals[cd];
        }
    }

    for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
        int col = lane_id * COLS_PER_THREAD + cd;
        dQ_row[col] = __float2bfloat16(dq_vals[cd] - m_sum * ak_vals[cd]);
    }

    if (lane_id == 0) {
        m_buf[l_off + qi] = m_sum;
    }
}

__global__ void mha_bwd_dK(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ m_buf,
    __nv_bfloat16* __restrict__ dK,
    int B, int H, int S, int D)
{
    int bh = blockIdx.y;
    if (bh >= B * H) return;
    int b = bh / H;
    int h = bh % H;

    int64_t base = (int64_t)(b * H + h) * S * D;
    int64_t l_off = (int64_t)(b * H + h) * S;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    int kj = blockIdx.x * NWARPS + warp_id;
    if (kj >= S) return;

    float scale = rsqrtf(static_cast<float>(D));

    const __nv_bfloat16* Kj = K + base + kj * D;
    const __nv_bfloat16* Vj = V + base + kj * D;

    float k_vals[COLS_PER_THREAD];
    float v_vals[COLS_PER_THREAD];
    for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
        int col = lane_id * COLS_PER_THREAD + cd;
        k_vals[cd] = __bfloat162float(Kj[col]);
        v_vals[cd] = __bfloat162float(Vj[col]);
    }

    float dk_vals[COLS_PER_THREAD] = {};

    for (int qi = kj; qi < S; qi++) {
        const __nv_bfloat16* Qi = Q + base + qi * D;
        const __nv_bfloat16* dOi = dO + base + qi * D;

        float q_vals[COLS_PER_THREAD];
        float dO_vals[COLS_PER_THREAD];
        for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
            int col = lane_id * COLS_PER_THREAD + cd;
            q_vals[cd] = __bfloat162float(Qi[col]);
            dO_vals[cd] = __bfloat162float(dOi[col]);
        }

        float score_tid = 0.0f;
        float dP_tid = 0.0f;
        #pragma unroll
        for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
            score_tid += q_vals[cd] * k_vals[cd];
            dP_tid += dO_vals[cd] * v_vals[cd];
        }

        float score_full = warp_sum(score_tid);
        float dP_full = warp_sum(dP_tid);

        float attn = expf(score_full * scale - L[l_off + qi]);
        float m_qi = m_buf[l_off + qi];
        float dScore = attn * (dP_full - m_qi) * scale;

        #pragma unroll
        for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
            dk_vals[cd] += dScore * q_vals[cd];
        }
    }

    __nv_bfloat16* dKj = dK + base + kj * D;
    for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
        int col = lane_id * COLS_PER_THREAD + cd;
        dKj[col] = __float2bfloat16(dk_vals[cd]);
    }
}

__global__ void mha_bwd_dV(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int S, int D)
{
    int bh = blockIdx.y;
    if (bh >= B * H) return;
    int b = bh / H;
    int h = bh % H;

    int64_t base = (int64_t)(b * H + h) * S * D;
    int64_t l_off = (int64_t)(b * H + h) * S;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    int kj = blockIdx.x * NWARPS + warp_id;
    if (kj >= S) return;

    float scale = rsqrtf(static_cast<float>(D));

    const __nv_bfloat16* Kj = K + base + kj * D;

    float k_vals[COLS_PER_THREAD];
    for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
        int col = lane_id * COLS_PER_THREAD + cd;
        k_vals[cd] = __bfloat162float(Kj[col]);
    }

    float dv_vals[COLS_PER_THREAD] = {};

    for (int qi = kj; qi < S; qi++) {
        const __nv_bfloat16* Qi = Q + base + qi * D;
        const __nv_bfloat16* dOi = dO + base + qi * D;

        float q_vals[COLS_PER_THREAD];
        float dO_vals[COLS_PER_THREAD];
        for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
            int col = lane_id * COLS_PER_THREAD + cd;
            q_vals[cd] = __bfloat162float(Qi[col]);
            dO_vals[cd] = __bfloat162float(dOi[col]);
        }

        float score_tid = 0.0f;
        #pragma unroll
        for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
            score_tid += q_vals[cd] * k_vals[cd];
        }

        float score_full = warp_sum(score_tid);
        float attn = expf(score_full * scale - L[l_off + qi]);

        #pragma unroll
        for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
            dv_vals[cd] += attn * dO_vals[cd];
        }
    }

    __nv_bfloat16* dVj = dV + base + kj * D;
    for (int cd = 0; cd < COLS_PER_THREAD; cd++) {
        int col = lane_id * COLS_PER_THREAD + cd;
        dVj[col] = __float2bfloat16(dv_vals[cd]);
    }
}

void run(tvm::ffi::TensorView Q_tv, tvm::ffi::TensorView K_tv, tvm::ffi::TensorView V_tv,
         tvm::ffi::TensorView O_tv, tvm::ffi::TensorView dO_tv, tvm::ffi::TensorView L_tv,
         tvm::ffi::TensorView dQ_tv, tvm::ffi::TensorView dK_tv, tvm::ffi::TensorView dV_tv) {
    CUDA_CHECK(cudaSetDevice(Q_tv.device().device_id));
    (void)O_tv;

    auto shape = Q_tv.shape();
    int B = static_cast<int>(shape[0]);
    int H = static_cast<int>(shape[1]);
    int S = static_cast<int>(shape[2]);
    int D = static_cast<int>(shape[3]);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q_tv.device().device_type, Q_tv.device().device_id));

    const __nv_bfloat16* Q  = static_cast<const __nv_bfloat16*>(Q_tv.data_ptr());
    const __nv_bfloat16* K  = static_cast<const __nv_bfloat16*>(K_tv.data_ptr());
    const __nv_bfloat16* V  = static_cast<const __nv_bfloat16*>(V_tv.data_ptr());
    const __nv_bfloat16* dO = static_cast<const __nv_bfloat16*>(dO_tv.data_ptr());
    const float* L          = static_cast<const float*>(L_tv.data_ptr());
    __nv_bfloat16* dQ       = static_cast<__nv_bfloat16*>(dQ_tv.data_ptr());
    __nv_bfloat16* dK       = static_cast<__nv_bfloat16*>(dK_tv.data_ptr());
    __nv_bfloat16* dV       = static_cast<__nv_bfloat16*>(dV_tv.data_ptr());

    size_t m_size = static_cast<size_t>(B) * H * S * sizeof(float);
    float* m_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&m_buf, m_size));

    int n_bh = B * H;
    int n_row_blocks = (S + NWARPS - 1) / NWARPS;
    dim3 grid(n_row_blocks, n_bh);
    dim3 block(NTHREADS);

    mha_bwd_dQ<<<grid, block, 0, stream>>>(Q, K, V, dO, L, dQ, m_buf, B, H, S, D);
    CUDA_CHECK(cudaGetLastError());
    mha_bwd_dK<<<grid, block, 0, stream>>>(Q, K, V, dO, L, m_buf, dK, B, H, S, D);
    CUDA_CHECK(cudaGetLastError());
    mha_bwd_dV<<<grid, block, 0, stream>>>(Q, K, dO, L, dV, B, H, S, D);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(m_buf));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_ns::run);

}  // namespace