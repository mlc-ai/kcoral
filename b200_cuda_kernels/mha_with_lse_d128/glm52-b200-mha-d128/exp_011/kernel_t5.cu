#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;

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
constexpr int Q_PER_WARP = BQ / WARPS;
constexpr int D_PER_WARP = D_VAL / WARPS;
constexpr int WM = 16, WN = 16, WK = 16;
constexpr int NK_QK = D_VAL / WK;
constexpr int NN_QK = BK / WN;
constexpr int NN_PV = D_PER_WARP / WN;
constexpr int NK_PV = BK / WK;

constexpr int SMEM_Q = BQ * D_VAL;
constexpr int SMEM_KV = BK * D_VAL;
constexpr int SMEM_P = BQ * BK;
constexpr int SMEM_BYTES = (SMEM_Q + 2 * SMEM_KV + 2 * SMEM_KV + SMEM_P) * (int)sizeof(__nv_bfloat16);

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
    __nv_bfloat16* K_smem = Q_smem + SMEM_Q;
    __nv_bfloat16* V_smem = K_smem + 2 * SMEM_KV;
    __nv_bfloat16* P_smem = V_smem + 2 * SMEM_KV;

    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    int q_start = blockIdx.y * BQ;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_row = warp_id * Q_PER_WARP;
    int d_warp = warp_id * D_PER_WARP;

    constexpr float scale = 0.08838834764831845f;

    int64_t bh_off = (int64_t)(b * H + h) * S_val * D_VAL;
    const __nv_bfloat16* Q_bh = Q + bh_off;
    const __nv_bfloat16* K_bh = K + bh_off;
    const __nv_bfloat16* V_bh = V + bh_off;
    __nv_bfloat16* O_bh = O + bh_off;
    float* LSE_bh = LSE + (int64_t)(b * H + h) * S_val;

    int r0 = lane_id / 4;
    int r8 = r0 + 8;
    int c0 = (lane_id % 4) * 2;

    {
        int total = BK * D_VAL / 8;
        for (int i = tid; i < total; i += THREADS) {
            int r = i / (D_VAL / 8);
            int c = i % (D_VAL / 8);
            if (r < S_val) {
                cp_async_16(K_smem + r * D_VAL + c * 8, K_bh + (int64_t)r * D_VAL + c * 8);
                cp_async_16(V_smem + r * D_VAL + c * 8, V_bh + (int64_t)r * D_VAL + c * 8);
            }
        }
        cp_async_commit();
    }

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

    wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> q_frag[NK_QK];
    for (int k = 0; k < NK_QK; k++)
        wmma::load_matrix_sync(q_frag[k], Q_smem + warp_row * D_VAL + k * WK, D_VAL);
    __syncthreads();

    wmma::fragment<wmma::accumulator, WM, WN, WK, float> o_frag[NN_PV];
    for (int n = 0; n < NN_PV; n++)
        wmma::fill_fragment(o_frag[n], 0.0f);

    float m_val0 = -INFINITY, m_val8 = -INFINITY;
    float l_val0 = 0.0f, l_val8 = 0.0f;

    int num_k_blocks = (S_val + BK - 1) / BK;
    int buf = 0;

    wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::col_major> b_qk;
    wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::row_major> b_pv;
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> s_frag[NN_QK];
    wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> p_frag;

    for (int iter = 0; iter < num_k_blocks; iter++) {
        int k_block = iter * BK;
        int next_buf = 1 - buf;

        if (iter + 1 < num_k_blocks) {
            __nv_bfloat16* Kb = K_smem + next_buf * SMEM_KV;
            __nv_bfloat16* Vb = V_smem + next_buf * SMEM_KV;
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

        __nv_bfloat16* K_cur = K_smem + buf * SMEM_KV;
        __nv_bfloat16* V_cur = V_smem + buf * SMEM_KV;

        for (int n = 0; n < NN_QK; n++) {
            wmma::fill_fragment(s_frag[n], 0.0f);
            for (int k = 0; k < NK_QK; k++) {
                wmma::load_matrix_sync(b_qk, K_cur + n * WN * D_VAL + k * WK, D_VAL);
                wmma::mma_sync(s_frag[n], q_frag[k], b_qk, s_frag[n]);
            }
            for (int i = 0; i < s_frag[n].num_elements; i++) {
                int col = n * WN + c0 + (i % 2) + (i / 4) * 8;
                int k_idx = k_block + col;
                if (k_idx >= S_val)
                    s_frag[n].x[i] = -INFINITY;
                else
                    s_frag[n].x[i] *= scale;
            }
        }

        float max0 = -INFINITY, max8 = -INFINITY;
        for (int n = 0; n < NN_QK; n++) {
            max0 = fmaxf(max0, fmaxf(s_frag[n].x[0], s_frag[n].x[1]));
            max0 = fmaxf(max0, fmaxf(s_frag[n].x[4], s_frag[n].x[5]));
            max8 = fmaxf(max8, fmaxf(s_frag[n].x[2], s_frag[n].x[3]));
            max8 = fmaxf(max8, fmaxf(s_frag[n].x[6], s_frag[n].x[7]));
        }
        max0 = fmaxf(max0, __shfl_xor_sync(0xFFFFFFFF, max0, 1));
        max0 = fmaxf(max0, __shfl_xor_sync(0xFFFFFFFF, max0, 2));
        max8 = fmaxf(max8, __shfl_xor_sync(0xFFFFFFFF, max8, 1));
        max8 = fmaxf(max8, __shfl_xor_sync(0xFFFFFFFF, max8, 2));

        float m_new0 = fmaxf(m_val0, max0);
        float m_new8 = fmaxf(m_val8, max8);
        float corr0 = (m_val0 == -INFINITY) ? 0.0f : fast_exp2f((m_val0 - m_new0) * LOG2E);
        float corr8 = (m_val8 == -INFINITY) ? 0.0f : fast_exp2f((m_val8 - m_new8) * LOG2E);

        for (int n = 0; n < NN_PV; n++) {
            o_frag[n].x[0] *= corr0; o_frag[n].x[1] *= corr0;
            o_frag[n].x[4] *= corr0; o_frag[n].x[5] *= corr0;
            o_frag[n].x[2] *= corr8; o_frag[n].x[3] *= corr8;
            o_frag[n].x[6] *= corr8; o_frag[n].x[7] *= corr8;
        }

        float sum0 = 0.0f, sum8 = 0.0f;
        for (int n = 0; n < NN_QK; n++) {
            float p0 = fast_exp2f((s_frag[n].x[0] - m_new0) * LOG2E);
            float p1 = fast_exp2f((s_frag[n].x[1] - m_new0) * LOG2E);
            float p4 = fast_exp2f((s_frag[n].x[4] - m_new0) * LOG2E);
            float p5 = fast_exp2f((s_frag[n].x[5] - m_new0) * LOG2E);
            float p2 = fast_exp2f((s_frag[n].x[2] - m_new8) * LOG2E);
            float p3 = fast_exp2f((s_frag[n].x[3] - m_new8) * LOG2E);
            float p6 = fast_exp2f((s_frag[n].x[6] - m_new8) * LOG2E);
            float p7 = fast_exp2f((s_frag[n].x[7] - m_new8) * LOG2E);

            sum0 += p0 + p1 + p4 + p5;
            sum8 += p2 + p3 + p6 + p7;

            int base = (warp_row + r0) * BK + n * WN + c0;
            int base8 = (warp_row + r8) * BK + n * WN + c0;
            P_smem[base + 0] = __float2bfloat16(p0);
            P_smem[base + 1] = __float2bfloat16(p1);
            P_smem[base + 8] = __float2bfloat16(p4);
            P_smem[base + 9] = __float2bfloat16(p5);
            P_smem[base8 + 0] = __float2bfloat16(p2);
            P_smem[base8 + 1] = __float2bfloat16(p3);
            P_smem[base8 + 8] = __float2bfloat16(p6);
            P_smem[base8 + 9] = __float2bfloat16(p7);
        }
        __syncwarp();

        sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 1);
        sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 2);
        sum8 += __shfl_xor_sync(0xFFFFFFFF, sum8, 1);
        sum8 += __shfl_xor_sync(0xFFFFFFFF, sum8, 2);

        l_val0 = l_val0 * corr0 + sum0;
        l_val8 = l_val8 * corr8 + sum8;
        m_val0 = m_new0;
        m_val8 = m_new8;

        for (int n = 0; n < NN_PV; n++) {
            for (int k = 0; k < NK_PV; k++) {
                wmma::load_matrix_sync(p_frag, P_smem + warp_row * BK + k * WK, BK);
                wmma::load_matrix_sync(b_pv, V_cur + k * WK * D_VAL + d_warp + n * WN, D_VAL);
                wmma::mma_sync(o_frag[n], p_frag, b_pv, o_frag[n]);
            }
        }

        buf = next_buf;
        __syncthreads();
    }

    float* O_smem = reinterpret_cast<float*>(K_smem);
    float inv_l0 = 1.0f / l_val0;
    float inv_l8 = 1.0f / l_val8;
    for (int n = 0; n < NN_PV; n++) {
        o_frag[n].x[0] *= inv_l0; o_frag[n].x[1] *= inv_l0;
        o_frag[n].x[4] *= inv_l0; o_frag[n].x[5] *= inv_l0;
        o_frag[n].x[2] *= inv_l8; o_frag[n].x[3] *= inv_l8;
        o_frag[n].x[6] *= inv_l8; o_frag[n].x[7] *= inv_l8;
        wmma::store_matrix_sync(O_smem + warp_row * D_VAL + d_warp + n * WN, o_frag[n], D_VAL, wmma::mem_row_major);
    }
    __syncthreads();

    {
        int total = BQ * D_VAL / 8;
        for (int i = tid; i < total; i += THREADS) {
            int r = i / (D_VAL / 8);
            int c = i % (D_VAL / 8);
            int g_row = q_start + r;
            if (g_row < S_val) {
                float* src = O_smem + r * D_VAL + c * 8;
                __nv_bfloat16 vals[8];
                for (int j = 0; j < 8; j++)
                    vals[j] = __float2bfloat16(src[j]);
                *reinterpret_cast<int4*>(O_bh + (int64_t)g_row * D_VAL + c * 8) =
                    *reinterpret_cast<int4*>(vals);
            }
        }
    }

    if (lane_id % 4 == 0) {
        int g_row0 = q_start + warp_row + r0;
        int g_row8 = q_start + warp_row + r8;
        if (g_row0 < S_val) LSE_bh[g_row0] = m_val0 + __logf(l_val0);
        if (g_row8 < S_val) LSE_bh[g_row8] = m_val8 + __logf(l_val8);
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
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));

    mha_kernel<<<grid, block, SMEM_BYTES, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data, B, H, S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_cuda::run);

}  // namespace mha_cuda