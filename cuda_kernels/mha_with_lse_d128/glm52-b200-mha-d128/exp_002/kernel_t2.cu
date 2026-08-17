#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cmath>
#include <cstdint>
#include <stdio.h>
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

using namespace nvcuda;

namespace mha_kernel_ns {

constexpr int D = 128;
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int Q_STRIDE = 130;    // D + 2 padding
constexpr int S_STRIDE = 65;     // BN + 1 padding (fp32 elements)
constexpr int P_STRIDE = 130;    // 2*S_STRIDE (bf16 elements)
constexpr float INV_SQRT_D = 0.0883883476483184f;

__global__ __launch_bounds__(128, 4)
void mha_wmma_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S) {

    int b = blockIdx.x;
    int h = blockIdx.y;
    int q_block = blockIdx.z;
    int q_start = q_block * BM;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;

    size_t bh_offset = ((size_t)b * H + h) * S * D;
    const __nv_bfloat16* Q_bh = Q + bh_offset;
    const __nv_bfloat16* K_bh = K + bh_offset;
    const __nv_bfloat16* V_bh = V + bh_offset;
    __nv_bfloat16* O_bh = O + bh_offset;
    float* LSE_bh = LSE + ((size_t)b * H + h) * S;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* smem_k = smem_q + (size_t)BM * Q_STRIDE;
    __nv_bfloat16* smem_v = smem_k + (size_t)BN * D;
    float* smem_s = reinterpret_cast<float*>(smem_v + (size_t)BN * D);
    __nv_bfloat16* smem_p = reinterpret_cast<__nv_bfloat16*>(smem_s);
    float* smem_rescale = smem_s + (size_t)BM * S_STRIDE;
    float* smem_rowsum = smem_rescale + BM;

    // Load Q tile
    {
        int total = BM * D / 8;
        const int4* q_src = reinterpret_cast<const int4*>(Q_bh + (size_t)q_start * D);
        int4* q_dst = reinterpret_cast<int4*>(smem_q);
        for (int i = tid; i < total; i += 128) {
            int row = i / (D / 8);
            if (q_start + row < S) {
                q_dst[i] = q_src[i];
            } else {
                q_dst[i] = make_int4(0, 0, 0, 0);
            }
        }
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag_qk;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag[4];

    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag_pv;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frag[8];

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        wmma::fill_fragment(o_frag[j], 0.0f);
    }

    float rowmax = -INFINITY;
    float rowsum = 0.0f;

    for (int kv = 0; kv < S; kv += BN) {
        int actual_bn = min(BN, S - kv);

        // Load K tile
        {
            int total = BN * D / 8;
            const int4* k_src = reinterpret_cast<const int4*>(K_bh + (size_t)kv * D);
            int4* k_dst = reinterpret_cast<int4*>(smem_k);
            for (int i = tid; i < total; i += 128) {
                int row = i / (D / 8);
                if (kv + row < S) {
                    k_dst[i] = k_src[i];
                } else {
                    k_dst[i] = make_int4(0, 0, 0, 0);
                }
            }
        }
        // Load V tile
        {
            int total = BN * D / 8;
            const int4* v_src = reinterpret_cast<const int4*>(V_bh + (size_t)kv * D);
            int4* v_dst = reinterpret_cast<int4*>(smem_v);
            for (int i = tid; i < total; i += 128) {
                int row = i / (D / 8);
                if (kv + row < S) {
                    v_dst[i] = v_src[i];
                } else {
                    v_dst[i] = make_int4(0, 0, 0, 0);
                }
            }
        }
        __syncthreads();

        // QK^T: each warp handles 1 M-tile (16 rows), 4 N-tiles, 8 K-tiles
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            wmma::fill_fragment(s_frag[n], 0.0f);
        }

        #pragma unroll
        for (int k_t = 0; k_t < D / 16; k_t++) {
            wmma::load_matrix_sync(a_frag,
                smem_q + warp_id * 16 * Q_STRIDE + k_t * 16, Q_STRIDE);

            #pragma unroll
            for (int n_t = 0; n_t < 4; n_t++) {
                // col_major fragment: layout is already in the type, no extra arg
                wmma::load_matrix_sync(b_frag_qk,
                    smem_k + n_t * 16 * D + k_t * 16, D);
                wmma::mma_sync(s_frag[n_t], a_frag, b_frag_qk, s_frag[n_t]);
            }
        }

        // Scale by 1/sqrt(D)
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                s_frag[n].x[i] *= INV_SQRT_D;
            }
        }

        // Store S to shared memory
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            wmma::store_matrix_sync(
                smem_s + warp_id * 16 * S_STRIDE + n * 16,
                s_frag[n], S_STRIDE, wmma::mem_row_major);
        }
        __syncthreads();

        // Online softmax: threads 0-63 each handle one row
        if (tid < BM) {
            int row = tid;
            float* s_row = smem_s + row * S_STRIDE;
            __nv_bfloat16* p_row = smem_p + row * P_STRIDE;

            float old_max = rowmax;
            float new_max = old_max;
            #pragma unroll
            for (int j = 0; j < BN; j++) {
                float val = (j < actual_bn) ? s_row[j] : -INFINITY;
                new_max = fmaxf(new_max, val);
            }

            float rescale = expf(old_max - new_max);
            smem_rescale[row] = rescale;
            rowsum *= rescale;

            #pragma unroll
            for (int j = 0; j < BN; j++) {
                float val = (j < actual_bn) ? s_row[j] : -INFINITY;
                float p = expf(val - new_max);
                p_row[j] = __float2bfloat16(p);
                rowsum += p;
            }

            rowmax = new_max;
            smem_rowsum[row] = rowsum;
        }
        __syncthreads();

        // Rescale O fragments
        {
            int row1 = warp_id * 16 + lane / 4;
            int row2 = warp_id * 16 + lane / 4 + 8;
            float r1 = smem_rescale[row1];
            float r2 = smem_rescale[row2];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                o_frag[j].x[0] *= r1;
                o_frag[j].x[1] *= r1;
                o_frag[j].x[4] *= r1;
                o_frag[j].x[5] *= r1;
                o_frag[j].x[2] *= r2;
                o_frag[j].x[3] *= r2;
                o_frag[j].x[6] *= r2;
                o_frag[j].x[7] *= r2;
            }
        }

        // PV: A=P(row_major), B=V(row_major), C=O
        #pragma unroll
        for (int k_t = 0; k_t < BN / 16; k_t++) {
            wmma::load_matrix_sync(a_frag,
                smem_p + warp_id * 16 * P_STRIDE + k_t * 16, P_STRIDE);

            #pragma unroll
            for (int n_t = 0; n_t < 8; n_t++) {
                wmma::load_matrix_sync(b_frag_pv,
                    smem_v + k_t * 16 * D + n_t * 16, D);
                wmma::mma_sync(o_frag[n_t], a_frag, b_frag_pv, o_frag[n_t]);
            }
        }

        __syncthreads();
    }

    // Store O to shared memory (reuse K/V space for fp32 staging)
    float* smem_o1 = reinterpret_cast<float*>(smem_k);
    float* smem_o2 = reinterpret_cast<float*>(smem_v);

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        float* target = (warp_id < 2) ? smem_o1 : smem_o2;
        int row_offset = (warp_id % 2) * 16;
        wmma::store_matrix_sync(target + row_offset * 128 + j * 16,
            o_frag[j], 128, wmma::mem_row_major);
    }
    __syncthreads();

    // Vectorized store to global
    {
        int total = BM * D / 8;
        int4* o_dst = reinterpret_cast<int4*>(O_bh + (size_t)q_start * D);
        for (int i = tid; i < total; i += 128) {
            int row = i / (D / 8);
            int col_start = (i % (D / 8)) * 8;
            if (q_start + row < S) {
                float* src = (row < 32) ? smem_o1 : smem_o2;
                int local_row = row % 32;
                float rs = smem_rowsum[row];
                float inv_rs = (rs > 0.0f) ? (1.0f / rs) : 0.0f;
                __nv_bfloat16 tmp[8];
                #pragma unroll
                for (int d = 0; d < 8; d++) {
                    tmp[d] = __float2bfloat16(src[local_row * 128 + col_start + d] * inv_rs);
                }
                o_dst[i] = *reinterpret_cast<int4*>(tmp);
            }
        }
    }

    // Store LSE
    if (tid < BM && q_start + tid < S) {
        LSE_bh[q_start + tid] = rowmax + logf(rowsum);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = static_cast<int>(Q.size(0));
    int H = static_cast<int>(Q.size(1));
    int S = static_cast<int>(Q.size(2));

    if (S == 0) return;

    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    dim3 grid(B, H, (S + BM - 1) / BM);
    dim3 block(128);

    size_t smem_size = (size_t)BM * Q_STRIDE * sizeof(__nv_bfloat16)
                     + (size_t)BN * D * sizeof(__nv_bfloat16)
                     + (size_t)BN * D * sizeof(__nv_bfloat16)
                     + (size_t)BM * S_STRIDE * sizeof(float)
                     + (size_t)BM * sizeof(float)
                     + (size_t)BM * sizeof(float);

    CUDA_CHECK(cudaFuncSetAttribute(mha_wmma_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    mha_wmma_kernel<<<grid, block, smem_size, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data, B, H, S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_kernel_ns::run);

}  // namespace mha_kernel_ns