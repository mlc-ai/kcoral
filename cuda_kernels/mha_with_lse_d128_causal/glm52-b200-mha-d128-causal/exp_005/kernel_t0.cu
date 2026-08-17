#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <cstdio>
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

namespace mha_kernel {

constexpr int BQ = 64;
constexpr int BK = 64;
constexpr int D = 128;
constexpr int THREADS = 256;

__global__ void attn_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int H, int S)
{
    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    int qb = blockIdx.y;
    int q_start = qb * BQ;

    int64_t off = (int64_t)(b * H + h) * (int64_t)S * D;
    const __nv_bfloat16* Qg = Q + off;
    const __nv_bfloat16* Kg = K + off;
    const __nv_bfloat16* Vg = V + off;
    __nv_bfloat16* Og = O + off;
    float* LSEg = LSE + (int64_t)(b * H + h) * (int64_t)S;

    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int row_in_warp = lane >> 2;
    int col_group = lane & 3;
    int my_row = (warp << 3) + row_in_warp;
    int q_global = q_start + my_row;

    extern __shared__ char smem_buf[];
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem_buf);
    __nv_bfloat16* sK = sQ + BQ * D;
    __nv_bfloat16* sV = sK + BK * D;
    __nv_bfloat16* sP = sV + BK * D;

    const float scale = 0.08838834764831845f; // 1/sqrt(128)

    // Load Q tile [BQ x D]
    for (int i = tid; i < BQ * D; i += THREADS) {
        int r = i / D;
        int d = i % D;
        int gr = q_start + r;
        sQ[i] = (gr < S) ? Qg[(int64_t)gr * D + d] : __float2bfloat16(0.f);
    }
    __syncthreads();

    float o_acc[32];
    #pragma unroll
    for (int j = 0; j < 32; j++) o_acc[j] = 0.f;
    float m_row = -INFINITY;
    float l_row = 0.f;

    int last_kb = (q_start + BQ - 1) / BK;

    for (int kb = 0; kb <= last_kb; kb++) {
        int k_start = kb * BK;

        // Load K and V tiles [BK x D]
        for (int i = tid; i < BK * D; i += THREADS) {
            int r = i / D;
            int d = i % D;
            int gr = k_start + r;
            if (gr < S) {
                sK[i] = Kg[(int64_t)gr * D + d];
                sV[i] = Vg[(int64_t)gr * D + d];
            } else {
                sK[i] = __float2bfloat16(0.f);
                sV[i] = __float2bfloat16(0.f);
            }
        }
        __syncthreads();

        // Load Q row into registers (64 bf162)
        __nv_bfloat162 q_row[D / 2];
        for (int d = 0; d < D; d += 2) {
            q_row[d >> 1] = *reinterpret_cast<__nv_bfloat162*>(&sQ[my_row * D + d]);
        }

        // Compute S[my_row][col_group*16+j]
        float s[16];
        if (q_global < S) {
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                int c = col_group * 16 + j;
                int k_global = k_start + c;
                float acc = 0.f;
                for (int d = 0; d < D; d += 2) {
                    __nv_bfloat162 k2 = *reinterpret_cast<__nv_bfloat162*>(&sK[c * D + d]);
                    float2 qf = __bfloat1622float2(q_row[d >> 1]);
                    float2 kf = __bfloat1622float2(k2);
                    acc += qf.x * kf.x + qf.y * kf.y;
                }
                acc *= scale;
                if (k_global > q_global) acc = -INFINITY;
                s[j] = acc;
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 16; j++) s[j] = -INFINITY;
        }

        // row max (reduce across 4 lanes sharing a row)
        float row_max_local = -INFINITY;
        #pragma unroll
        for (int j = 0; j < 16; j++) row_max_local = fmaxf(row_max_local, s[j]);
        float m_new = row_max_local;
        m_new = fmaxf(m_new, __shfl_xor_sync(0xffffffff, m_new, 1));
        m_new = fmaxf(m_new, __shfl_xor_sync(0xffffffff, m_new, 2));

        float m_prev = m_row;
        m_row = m_new;
        float alpha = (m_prev == -INFINITY) ? 0.f : expf(m_prev - m_new);

        // Rescale O
        #pragma unroll
        for (int j = 0; j < 32; j++) o_acc[j] *= alpha;

        // P = exp(s - m_new), accumulate row sum
        float row_sum_local = 0.f;
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            float p = (s[j] == -INFINITY) ? 0.f : expf(s[j] - m_new);
            s[j] = p;
            row_sum_local += p;
        }
        row_sum_local += __shfl_xor_sync(0xffffffff, row_sum_local, 1);
        row_sum_local += __shfl_xor_sync(0xffffffff, row_sum_local, 2);
        l_row = l_row * alpha + row_sum_local;

        // Store P to shared memory (bf16)
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            int c = col_group * 16 + j;
            sP[my_row * BK + c] = __float2bfloat16(s[j]);
        }
        __syncthreads();

        // Load P row into registers
        float p_row[BK];
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            p_row[k] = __bfloat162float(sP[my_row * BK + k]);
        }

        // O += P @ V,  my d range = [col_group*32, col_group*32+32)
        for (int k = 0; k < BK; k++) {
            float p = p_row[k];
            #pragma unroll
            for (int j = 0; j < 32; j += 2) {
                int d = col_group * 32 + j;
                __nv_bfloat162 v2 = *reinterpret_cast<__nv_bfloat162*>(&sV[k * D + d]);
                float2 vf = __bfloat1622float2(v2);
                o_acc[j]     += p * vf.x;
                o_acc[j + 1] += p * vf.y;
            }
        }
        __syncthreads();
    }

    // Finalize: normalize and write out
    if (q_global < S) {
        float inv_l = 1.f / l_row;
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            int d = col_group * 32 + j;
            Og[(int64_t)q_global * D + d] = __float2bfloat16(o_acc[j] * inv_l);
        }
        if (col_group == 0) {
            float lse = m_row + logf(l_row);
            LSEg[q_global] = lse;
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    const int B = 4;
    const int H = 48;
    const int D = 128;
    int S = (int)Q.size(2); // Q is (B, H, S, D)

    const __nv_bfloat16* Qd = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* Kd = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* Vd = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* Od = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSEd = static_cast<float*>(LSE.data_ptr());

    int smem_bytes = (int)(BQ * D * sizeof(__nv_bfloat16)      // sQ
                          + BK * D * sizeof(__nv_bfloat16) * 2  // sK, sV
                          + BQ * BK * sizeof(__nv_bfloat16));    // sP

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(
        attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    dim3 grid(B * H, (S + BQ - 1) / BQ);
    dim3 block(THREADS);
    attn_kernel<<<grid, block, smem_bytes, stream>>>(Qd, Kd, Vd, Od, LSEd, H, S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel::run);

}  // namespace mha_kernel