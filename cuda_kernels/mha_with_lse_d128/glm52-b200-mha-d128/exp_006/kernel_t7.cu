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
constexpr int STAGES = 2;

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace mha_lse_d128 {

__device__ __forceinline__ void cp_async_16B(uint32_t smem_addr, const void* gmem_addr) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem_addr));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

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
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)ptr;  ptr += BM * D * 2;       // 16KB
    __nv_bfloat16* smem_K0 = (__nv_bfloat16*)ptr; ptr += BN * D * 2;       // 16KB
    __nv_bfloat16* smem_K1 = (__nv_bfloat16*)ptr; ptr += BN * D * 2;       // 16KB
    __nv_bfloat16* smem_V = (__nv_bfloat16*)ptr;  ptr += BN * D * 2;       // 16KB
    float* smem_S = (float*)ptr;                  ptr += BM * BN * 4;      // 16KB
    __nv_bfloat16* smem_P = (__nv_bfloat16*)ptr;  ptr += BM * BN * 2;      // 8KB (SEPARATE buffer!)
    float* smem_rm = (float*)ptr;                 ptr += BM * 4;           // 256B
    float* smem_rs = (float*)ptr;                 ptr += BM * 4;           // 256B
    float* smem_rescale = (float*)ptr;            // 256B

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_m = warp_id / 2;
    int warp_n = warp_id % 2;

    // Load Q to SMEM with zero-padding
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

    int num_kv = (S + BN - 1) / BN;

    // Issue first K load
    {
        int kv_start = 0;
        for (int i = tid; i < BN * D / 8; i += NUM_THREADS) {
            int row = i / (D / 8);
            int gm_row = kv_start + row;
            if (gm_row < S) {
                uint32_t s_addr = (uint32_t)__cvta_generic_to_shared(&smem_K0[i * 8]);
                cp_async_16B(s_addr, &K_base[gm_row * D + (i % (D / 8)) * 8]);
            } else {
                ((int4*)smem_K0)[i] = make_int4(0, 0, 0, 0);
            }
        }
        cp_async_commit();
        cp_async_wait_all();
    }
    __syncthreads();

    for (int kv_idx = 0; kv_idx < num_kv; ++kv_idx) {
        int kv_start = kv_idx * BN;
        int actual_kv = min(BN, S - kv_start);
        __nv_bfloat16* smem_K = (kv_idx % 2 == 0) ? smem_K0 : smem_K1;

        // Issue next K load if exists
        if (kv_idx + 1 < num_kv) {
            __nv_bfloat16* next_K = (kv_idx % 2 == 0) ? smem_K1 : smem_K0;
            int next_kv_start = (kv_idx + 1) * BN;
            for (int i = tid; i < BN * D / 8; i += NUM_THREADS) {
                int row = i / (D / 8);
                int gm_row = next_kv_start + row;
                if (gm_row < S) {
                    uint32_t s_addr = (uint32_t)__cvta_generic_to_shared(&next_K[i * 8]);
                    cp_async_16B(s_addr, &K_base[gm_row * D + (i % (D / 8)) * 8]);
                } else {
                    ((int4*)next_K)[i] = make_int4(0, 0, 0, 0);
                }
            }
            cp_async_commit();
        }

        // S = Q @ K^T
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_frag[2];
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            wmma::fill_fragment(s_frag[i], 0.0f);

        {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;

            #pragma unroll
            for (int k_step = 0; k_step < D; k_step += WMMA_K) {
                wmma::load_matrix_sync(a_frag, smem_Q + warp_m * WMMA_M * D + k_step, D);
                #pragma unroll
                for (int n = 0; n < 2; ++n) {
                    int n_tile = warp_n * 2 + n;
                    wmma::load_matrix_sync(b_frag, smem_K + n_tile * WMMA_N * D + k_step, D);
                    wmma::mma_sync(s_frag[n], a_frag, b_frag, s_frag[n]);
                }
            }
        }

        #pragma unroll
        for (int n = 0; n < 2; ++n) {
            int n_tile = warp_n * 2 + n;
            wmma::store_matrix_sync(smem_S + warp_m * WMMA_M * BN + n_tile * WMMA_N,
                                    s_frag[n], BN, wmma::mem_row_major);
        }
        __syncthreads();

        // Scale and mask
        for (int i = tid; i < BM * BN; i += NUM_THREADS) {
            int col = i % BN;
            smem_S[i] = (col >= actual_kv) ? -INFINITY : smem_S[i] * scale;
        }
        __syncthreads();

        // Issue V load (overlaps with softmax)
        for (int i = tid; i < BN * D / 8; i += NUM_THREADS) {
            int row = i / (D / 8);
            int gm_row = kv_start + row;
            if (gm_row < S) {
                uint32_t s_addr = (uint32_t)__cvta_generic_to_shared(&smem_V[i * 8]);
                cp_async_16B(s_addr, &V_base[gm_row * D + (i % (D / 8)) * 8]);
            } else {
                ((int4*)smem_V)[i] = make_int4(0, 0, 0, 0);
            }
        }
        cp_async_commit();

        // Softmax: 4 threads per row, 64 rows. S and P are SEPARATE buffers.
        {
            int row = tid / 4;
            int col_start = (tid % 4) * 16;
            bool valid = (row < actual_bm);

            float old_max = valid ? smem_rm[row] : 0.0f;
            float local_max = -INFINITY;
            if (valid) {
                #pragma unroll
                for (int c = 0; c < 16; ++c)
                    local_max = fmaxf(local_max, smem_S[row * BN + col_start + c]);
            }
            local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, 1));
            local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, 2));

            float new_max = valid ? fmaxf(old_max, local_max) : 0.0f;
            float rescale = valid ? __expf(old_max - new_max) : 1.0f;

            float local_sum = 0.0f;
            if (valid) {
                #pragma unroll
                for (int c = 0; c < 16; ++c) {
                    float val = __expf(smem_S[row * BN + col_start + c] - new_max);
                    local_sum += val;
                    smem_P[row * BN + col_start + c] = __float2bfloat16(val);
                }
            }
            local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, 1);
            local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, 2);

            if (valid) {
                float old_sum = smem_rs[row];
                float new_sum = old_sum * rescale + local_sum;
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

        // Wait for V and next K
        cp_async_wait_all();
        __syncthreads();

        // Rescale O fragments
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

        // O += P @ V
        {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> p_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> v_frag;

            #pragma unroll
            for (int k_step = 0; k_step < BN; k_step += WMMA_K) {
                wmma::load_matrix_sync(p_frag, smem_P + warp_m * WMMA_M * BN + k_step, BN);
                #pragma unroll
                for (int n = 0; n < 4; ++n) {
                    int n_tile = warp_n * 4 + n;
                    wmma::load_matrix_sync(v_frag, smem_V + k_step * D + n_tile * WMMA_N, D);
                    wmma::mma_sync(o_frag[n], p_frag, v_frag, o_frag[n]);
                }
            }
        }
        __syncthreads();
    }

    // Final normalization O /= sum
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

    // Store O to SMEM (reuse smem_Q for rows 0-31, smem_S for rows 32-63)
    #pragma unroll
    for (int n = 0; n < 4; ++n) {
        int n_tile = warp_n * 4 + n;
        int abs_row = warp_m * WMMA_M;
        float* out_ptr;
        if (abs_row < 32) {
            out_ptr = (float*)smem_Q + abs_row * D + n_tile * WMMA_N;
        } else {
            out_ptr = (float*)smem_S + (abs_row - 32) * D + n_tile * WMMA_N;
        }
        wmma::store_matrix_sync(out_ptr, o_frag[n], D, wmma::mem_row_major);
    }
    __syncthreads();

    // Convert FP32 -> BF16 and write to global O
    for (int i = tid; i < BM * D / 8; i += NUM_THREADS) {
        int row = i / (D / 8);
        int col8 = i % (D / 8);
        int gm_row = m_start + row;
        if (gm_row < S) {
            float* fp32_ptr;
            if (row < 32) {
                fp32_ptr = (float*)smem_Q + row * D + col8 * 8;
            } else {
                fp32_ptr = (float*)smem_S + (row - 32) * D + col8 * 8;
            }
            __nv_bfloat16 bv[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                bv[j] = __float2bfloat16(fp32_ptr[j]);
            ((int4*)O_base)[gm_row * (D / 8) + col8] = *((int4*)bv);
        }
    }

    // LSE = m + log(l)
    if (tid < actual_bm) {
        int gm_row = m_start + tid;
        float mv = smem_rm[tid];
        float lv = smem_rs[tid];
        LSE_base[gm_row] = (lv > 0.0f) ? (mv + __logf(lv)) : mv;
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

    int smem_size = BM * D * 2 + STAGES * BN * D * 2 + BN * D * 2 + BM * BN * 4 + BM * BN * 2 + BM * 4 * 3;

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