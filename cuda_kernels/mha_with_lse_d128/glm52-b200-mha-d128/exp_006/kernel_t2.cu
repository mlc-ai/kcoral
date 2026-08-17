#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;

constexpr int D = 128;
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int NUM_THREADS = 256;
constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace mha_lse_d128 {

__global__ __launch_bounds__(256)
void attention_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K_g,
    const __nv_bfloat16* __restrict__ V_g,
    __nv_bfloat16* __restrict__ O_g,
    float* __restrict__ LSE_g,
    int S) {

    int bh = blockIdx.x;
    int q_blk = blockIdx.y;

    int m_start = q_blk * BM;
    if (m_start >= S) return;

    int actual_bm = min(BM, S - m_start);
    float scale = rsqrtf((float)D);

    int64_t offset = (int64_t)bh * S * D;
    const __nv_bfloat16* Q_base = Q + offset;
    const __nv_bfloat16* K_base = K_g + offset;
    const __nv_bfloat16* V_base = V_g + offset;
    __nv_bfloat16* O_base = O_g + offset;
    float* LSE_base = LSE_g + (int64_t)bh * S;

    extern __shared__ char smem_buf[];
    char* ptr = smem_buf;
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)ptr;  ptr += BM * D * 2;
    float* smem_SP = (float*)ptr;                 ptr += BM * BN * 4;
    __nv_bfloat16* smem_KV = (__nv_bfloat16*)ptr; ptr += BN * D * 2;
    __nv_bfloat16* smem_P = (__nv_bfloat16*)ptr;  ptr += BM * BN * 2;
    float* smem_rm = (float*)ptr;                 ptr += BM * 4;
    float* smem_rs = (float*)ptr;                 ptr += BM * 4;
    float* smem_rescale = (float*)ptr;            ptr += BM * 4;
    float* smem_O_out = (float*)smem_Q;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_m = warp_id / 2;
    int warp_n = warp_id % 2;

    // Load Q tile to SMEM (zero-pad rows beyond S)
    for (int i = tid; i < BM * D / 8; i += NUM_THREADS) {
        int row = i / (D / 8);
        int gm_row = m_start + row;
        if (gm_row < S)
            ((int4*)smem_Q)[i] = ((const int4*)Q_base)[gm_row * (D / 8) + (i % (D / 8))];
        else
            ((int4*)smem_Q)[i] = make_int4(0, 0, 0, 0);
    }

    if (tid < BM) {
        smem_rm[tid] = -INFINITY;
        smem_rs[tid] = 0.0f;
    }

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        wmma::fill_fragment(o_frag[i], 0.0f);

    __syncthreads();

    for (int kv_start = 0; kv_start < S; kv_start += BN) {
        int actual_kv = min(BN, S - kv_start);

        // Load K chunk
        for (int i = tid; i < BN * D / 8; i += NUM_THREADS) {
            int row = i / (D / 8);
            int gm_row = kv_start + row;
            if (gm_row < S)
                ((int4*)smem_KV)[i] = ((const int4*)K_base)[gm_row * (D / 8) + (i % (D / 8))];
            else
                ((int4*)smem_KV)[i] = make_int4(0, 0, 0, 0);
        }
        __syncthreads();

        // S = Q @ K^T  (M=64, N=64, K=128)
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_frag[2];
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            wmma::fill_fragment(s_frag[i], 0.0f);

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;

        #pragma unroll
        for (int k_step = 0; k_step < D; k_step += WMMA_K) {
            wmma::load_matrix_sync(a_frag, smem_Q + warp_m * WMMA_M * D + k_step, D);
            #pragma unroll
            for (int n = 0; n < 2; ++n) {
                int n_tile = warp_n * 2 + n;
                wmma::load_matrix_sync(b_frag, smem_KV + n_tile * WMMA_N * D + k_step, D);
                wmma::mma_sync(s_frag[n], a_frag, b_frag, s_frag[n]);
            }
        }

        #pragma unroll
        for (int n = 0; n < 2; ++n) {
            int n_tile = warp_n * 2 + n;
            wmma::store_matrix_sync(smem_SP + warp_m * WMMA_M * BN + n_tile * WMMA_N,
                                    s_frag[n], BN, wmma::mem_row_major);
        }
        __syncthreads();

        // Apply scale and mask invalid columns
        for (int i = tid; i < BM * BN; i += NUM_THREADS) {
            int col = i % BN;
            smem_SP[i] = (col >= actual_kv) ? -INFINITY : smem_SP[i] * scale;
        }
        __syncthreads();

        // Online softmax (4 threads per row, 64 rows)
        // P is UNNORMALIZED: P = exp(S - m), will normalize at the end
        {
            int row = tid / 4;
            int col_start = (tid % 4) * 16;
            bool valid = (row < actual_bm);

            float old_max = valid ? smem_rm[row] : 0.0f;
            float local_max = -INFINITY;
            if (valid) {
                #pragma unroll
                for (int c = 0; c < 16; ++c)
                    local_max = fmaxf(local_max, smem_SP[row * BN + col_start + c]);
            }

            local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, 1));
            local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, 2));

            float new_max = valid ? fmaxf(old_max, local_max) : 0.0f;
            float rescale = valid ? expf(old_max - new_max) : 1.0f;

            float vals[16];
            float local_sum = 0.0f;
            if (valid) {
                #pragma unroll
                for (int c = 0; c < 16; ++c) {
                    vals[c] = __expf(smem_SP[row * BN + col_start + c] - new_max);
                    local_sum += vals[c];
                }
            }

            local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, 1);
            local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, 2);

            if (valid) {
                float old_sum = smem_rs[row];
                float new_sum = old_sum * rescale + local_sum;

                // Store UNNORMALIZED P (no division by sum)
                #pragma unroll
                for (int c = 0; c < 16; ++c)
                    smem_P[row * BN + col_start + c] = __float2bfloat16(vals[c]);

                if (tid % 4 == 0) {
                    smem_rm[row] = new_max;
                    smem_rs[row] = new_sum;
                    smem_rescale[row] = rescale;
                }
            } else {
                if (tid % 4 == 0)
                    smem_rescale[row] = 1.0f;
            }
        }
        __syncthreads();

        // Rescale O accumulators: O *= exp(m_old - m_new)
        // wmma f32 accumulator layout for 16x16:
        //   x[0,1,4,5] -> row = lane/4,      x[2,3,6,7] -> row = lane/4 + 8
        // Must use lane_id (0-31), not tid, and add warp_m * WMMA_M for tile offset
        {
            int frag_row0 = warp_m * WMMA_M + lane_id / 4;
            int frag_row1 = frag_row0 + 8;
            float r0 = (frag_row0 < actual_bm) ? smem_rescale[frag_row0] : 1.0f;
            float r1 = (frag_row1 < actual_bm) ? smem_rescale[frag_row1] : 1.0f;

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    if (j < 2 || (j >= 4 && j < 6))
                        o_frag[i].x[j] *= r0;
                    else
                        o_frag[i].x[j] *= r1;
                }
            }
        }

        // Load V chunk (reuse smem_KV)
        for (int i = tid; i < BN * D / 8; i += NUM_THREADS) {
            int row = i / (D / 8);
            int gm_row = kv_start + row;
            if (gm_row < S)
                ((int4*)smem_KV)[i] = ((const int4*)V_base)[gm_row * (D / 8) + (i % (D / 8))];
            else
                ((int4*)smem_KV)[i] = make_int4(0, 0, 0, 0);
        }
        __syncthreads();

        // O += P @ V  (M=64, N=128, K=64)
        {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> p_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> v_frag;

            #pragma unroll
            for (int k_step = 0; k_step < BN; k_step += WMMA_K) {
                wmma::load_matrix_sync(p_frag, smem_P + warp_m * WMMA_M * BN + k_step, BN);
                #pragma unroll
                for (int n = 0; n < 4; ++n) {
                    int n_tile = warp_n * 4 + n;
                    wmma::load_matrix_sync(v_frag, smem_KV + k_step * D + n_tile * WMMA_N, D);
                    wmma::mma_sync(o_frag[n], p_frag, v_frag, o_frag[n]);
                }
            }
        }
        __syncthreads();
    }

    // Final normalization: O /= l (the running sum of unnormalized P)
    {
        int frag_row0 = warp_m * WMMA_M + lane_id / 4;
        int frag_row1 = frag_row0 + 8;
        float is0 = (frag_row0 < actual_bm && smem_rs[frag_row0] > 0.0f) ? 1.0f / smem_rs[frag_row0] : 0.0f;
        float is1 = (frag_row1 < actual_bm && smem_rs[frag_row1] > 0.0f) ? 1.0f / smem_rs[frag_row1] : 0.0f;

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                if (j < 2 || (j >= 4 && j < 6))
                    o_frag[i].x[j] *= is0;
                else
                    o_frag[i].x[j] *= is1;
            }
        }
    }

    // Store O to SMEM (reuses smem_Q + smem_SP = 32KB)
    #pragma unroll
    for (int n = 0; n < 4; ++n) {
        int n_tile = warp_n * 4 + n;
        wmma::store_matrix_sync(smem_O_out + warp_m * WMMA_M * D + n_tile * WMMA_N,
                                o_frag[n], D, wmma::mem_row_major);
    }
    __syncthreads();

    // Convert fp32 -> bf16 and write to global O
    for (int i = tid; i < BM * D / 8; i += NUM_THREADS) {
        int row = i / (D / 8);
        int col8 = i % (D / 8);
        int gm_row = m_start + row;
        if (gm_row < S) {
            __nv_bfloat16 bv[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                bv[j] = __float2bfloat16(smem_O_out[row * D + col8 * 8 + j]);
            ((int4*)O_base)[gm_row * (D / 8) + col8] = *((int4*)bv);
        }
    }

    // LSE = m + log(l)
    if (tid < actual_bm) {
        int gm_row = m_start + tid;
        float mv = smem_rm[tid];
        float lv = smem_rs[tid];
        LSE_base[gm_row] = (lv > 0.0f) ? (mv + logf(lv)) : mv;
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K,
         tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B_val = Q.size(0);
    int64_t H_val = Q.size(1);
    int64_t S_val = Q.size(2);

    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    int num_q_blocks = (S_val + BM - 1) / BM;
    dim3 grid((int)(B_val * H_val), num_q_blocks);
    dim3 block(NUM_THREADS);

    int smem_size = BM * D * 2 + BM * BN * 4 + BN * D * 2 + BM * BN * 2 + BM * 4 * 3;

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(attention_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    attention_kernel<<<grid, block, smem_size, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data, (int)S_val);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_lse_d128::run);

}  // namespace mha_lse_d128