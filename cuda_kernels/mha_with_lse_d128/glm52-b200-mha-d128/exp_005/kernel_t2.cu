#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda::wmma;

#define CUDA_CHECK(call) do {                                   \
    cudaError_t _e = (call);                                     \
    if (_e != cudaSuccess) {                                     \
        fprintf(stderr, "CUDA error %s at %s:%d\n",              \
                cudaGetErrorString(_e), __FILE__, __LINE__);      \
        exit(1);                                                  \
    }                                                             \
} while(0)

namespace attn_fwd {

constexpr int BM = 128;
constexpr int BN = 64;
constexpr int D = 128;
constexpr int THREADS = 128;
constexpr float SCALE = 0.08838834764831845f;

__global__ __launch_bounds__(THREADS, 1)
void attention_kernel(
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
    int m_start = q_block * BM;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    extern __shared__ char smem[];
    float* sO = (float*)smem;
    __nv_bfloat16* sQ = (__nv_bfloat16*)(sO + BM * D);
    __nv_bfloat16* sK = sQ + BM * D;
    __nv_bfloat16* sV = sK + BN * D;
    float* sS = (float*)(sV + BN * D);
    __nv_bfloat16* sP = (__nv_bfloat16*)sS;
    float* sScratch = sS + BM * BN;

    int64_t base = ((int64_t)b * H + h) * S * D;

    for (int i = tid; i < BM * D; i += THREADS) {
        sO[i] = 0.0f;
    }

    for (int i = tid; i < BM; i += THREADS) {
        int m = m_start + i;
        if (m < S) {
            #pragma unroll
            for (int d = 0; d < D; d += 8) {
                *((int4*)&sQ[i * D + d]) =
                    *((const int4*)&Q[base + (int64_t)m * D + d]);
            }
        } else {
            #pragma unroll
            for (int d = 0; d < D; d += 8) {
                *((int4*)&sQ[i * D + d]) = make_int4(0, 0, 0, 0);
            }
        }
    }

    float my_rowmax = -INFINITY;
    float my_rowsum = 0.0f;

    __syncthreads();

    for (int kv_start = 0; kv_start < S; kv_start += BN) {
        for (int i = tid; i < BN; i += THREADS) {
            int kv = kv_start + i;
            if (kv < S) {
                #pragma unroll
                for (int d = 0; d < D; d += 8) {
                    *((int4*)&sK[i * D + d]) =
                        *((const int4*)&K[base + (int64_t)kv * D + d]);
                    *((int4*)&sV[i * D + d]) =
                        *((const int4*)&V[base + (int64_t)kv * D + d]);
                }
            } else {
                #pragma unroll
                for (int d = 0; d < D; d += 8) {
                    *((int4*)&sK[i * D + d]) = make_int4(0, 0, 0, 0);
                    *((int4*)&sV[i * D + d]) = make_int4(0, 0, 0, 0);
                }
            }
        }
        __syncthreads();

        fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a_frag;
        fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> b_frag;
        fragment<accumulator, 16, 16, 16, float> s_frag[2][4];

        #pragma unroll
        for (int mi = 0; mi < 2; mi++)
            #pragma unroll
            for (int ni = 0; ni < 4; ni++)
                fill_fragment(s_frag[mi][ni], 0.0f);

        #pragma unroll
        for (int ki = 0; ki < D / 16; ki++) {
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                load_matrix_sync(a_frag,
                    &sQ[(warp_id * 2 + mi) * 16 * D + ki * 16], D);
                #pragma unroll
                for (int ni = 0; ni < 4; ni++) {
                    load_matrix_sync(b_frag,
                        &sK[ni * 16 * D + ki * 16], D);
                    mma_sync(s_frag[mi][ni], a_frag, b_frag, s_frag[mi][ni]);
                }
            }
        }

        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                #pragma unroll
                for (int e = 0; e < s_frag[mi][ni].num_elements; e++) {
                    s_frag[mi][ni].x[e] *= SCALE;
                }
            }
        }

        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                store_matrix_sync(
                    &sS[(warp_id * 2 + mi) * 16 * BN + ni * 16],
                    s_frag[mi][ni], BN, mem_row_major);
            }
        }
        __syncthreads();

        int row = tid;
        int m = m_start + row;
        float alpha = 1.0f;
        if (m < S) {
            int kv_end = min(kv_start + BN, S);
            float old_max = my_rowmax;
            float new_max = old_max;
            for (int n = 0; n < BN; n++) {
                if (kv_start + n < kv_end) {
                    new_max = fmaxf(new_max, sS[row * BN + n]);
                }
            }

            alpha = __expf(old_max - new_max);
            float p_sum = 0.0f;
            for (int n = 0; n < BN; n++) {
                float s_val = (kv_start + n < kv_end)
                    ? sS[row * BN + n] : -INFINITY;
                float p = __expf(s_val - new_max);
                sP[row * BN + n] = __float2bfloat16(p);
                p_sum += p;
            }

            my_rowmax = new_max;
            my_rowsum = my_rowsum * alpha + p_sum;
        }
        sScratch[row] = alpha;
        __syncthreads();

        for (int i = tid; i < BM * D; i += THREADS) {
            sO[i] *= sScratch[i / D];
        }
        __syncthreads();

        fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> p_frag;
        fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major> v_frag;
        fragment<accumulator, 16, 16, 16, float> acc;

        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                load_matrix_sync(acc,
                    &sO[(warp_id * 2 + mi) * 16 * D + ni * 16],
                    D, mem_row_major);
                #pragma unroll
                for (int ki = 0; ki < BN / 16; ki++) {
                    load_matrix_sync(p_frag,
                        &sP[(warp_id * 2 + mi) * 16 * BN + ki * 16], BN);
                    load_matrix_sync(v_frag,
                        &sV[ki * 16 * D + ni * 16], D);
                    mma_sync(acc, p_frag, v_frag, acc);
                }
                store_matrix_sync(
                    &sO[(warp_id * 2 + mi) * 16 * D + ni * 16],
                    acc, D, mem_row_major);
            }
        }
        __syncthreads();
    }

    sScratch[tid] = (m_start + tid < S) ? (1.0f / my_rowsum) : 0.0f;
    __syncthreads();

    for (int i = tid; i < BM * D; i += THREADS) {
        sO[i] *= sScratch[i / D];
    }
    __syncthreads();

    int row = tid;
    int m = m_start + row;
    if (m < S) {
        #pragma unroll
        for (int d = 0; d < D; d += 2) {
            __nv_bfloat162 val;
            val.x = __float2bfloat16(sO[row * D + d]);
            val.y = __float2bfloat16(sO[row * D + d + 1]);
            *((__nv_bfloat162*)&O[base + (int64_t)m * D + d]) = val;
        }
        LSE[((int64_t)b * H + h) * S + m] = my_rowmax + __logf(my_rowsum);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    const int B = 4;
    const int H = 48;
    const int S = static_cast<int>(Q.size(2));

    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data       = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data             = static_cast<float*>(LSE.data_ptr());

    const int num_q_blocks = (S + BM - 1) / BM;
    dim3 grid(B * H, num_q_blocks);
    dim3 block(THREADS);

    size_t smem_size =
        (size_t)(BM * D) * sizeof(float)
      + (size_t)(BM * D) * sizeof(__nv_bfloat16)
      + (size_t)(BN * D) * sizeof(__nv_bfloat16) * 2
      + (size_t)(BM * BN) * sizeof(float)
      + (size_t)(BM) * sizeof(float);

    CUDA_CHECK(cudaFuncSetAttribute(
        attention_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem_size));

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    attention_kernel<<<grid, block, smem_size, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data, B, H, S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, attn_fwd::run);

} // namespace attn_fwd