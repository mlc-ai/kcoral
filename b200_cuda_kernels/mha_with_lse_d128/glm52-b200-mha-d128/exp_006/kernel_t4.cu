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
constexpr int BM = 128;
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

__global__ __launch_bounds__(256, 2)
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
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)smem_buf;              // 128*128*2 = 32KB
    __nv_bfloat16* smem_K = smem_Q + BM * D;                       // 64*128*2 = 16KB
    __nv_bfloat16* smem_V = smem_K + BN * D;                       // 64*128*2 = 16KB
    float* smem_S = (float*)(smem_V + BN * D);                     // 128*64*4 = 32KB
    __nv_bfloat16* smem_P = (__nv_bfloat16*)smem_S;                // reuses S buffer (BF16, first 16KB)
    float* smem_rm = (float*)(smem_S + BM * BN);                   // 128*4 = 512B
    float* smem_rs = smem_rm + BM;                                 // 128*4 = 512B
    float* smem_rescale = smem_rs + BM;                            // 128*4 = 512B

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_m = warp_id / 2;  // 0-3
    int warp_n = warp_id % 2;  // 0-1

    // Load Q to SMEM with zero-padding
    {
        int row = tid / 2;
        int col_chunk = tid % 2;
        int gm_row = m_start + row;
        __nv_bfloat16* smem_ptr = smem_Q + row * D + col_chunk * 64;
        if (gm_row < S) {
            const __nv_bfloat16* gmem_ptr = Q_base + gm_row * D + col_chunk * 64;
            for (int i = 0; i < 8; i++) {
                uint32_t s_addr = (uint32_t)__cvta_generic_to_shared(smem_ptr + i * 8);
                cp_async_16B(s_addr, gmem_ptr + i * 8);
            }
        } else {
            for (int i = 0; i < 8; i++) {
                ((int4*)smem_ptr)[i] = make_int4(0, 0, 0, 0);
            }
        }
        cp_async_commit();
        cp_async_wait_all();
    }

    // Init row stats
    if (tid < BM) {
        smem_rm[tid] = -INFINITY;
        smem_rs[tid] = 0.0f;
    }

    // O fragments: 2 M-tiles × 4 N-tiles = 8 per warp
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        wmma::fill_fragment(o_frag[i], 0.0f);

    __syncthreads();

    int num_kv = (S + BN - 1) / BN;

    for (int kv_idx = 0; kv_idx < num_kv; ++kv_idx) {
        int kv_start = kv_idx * BN;
        int actual_kv = min(BN, S - kv_start);

        // Load K
        {
            int row = tid / 4;
            int col_chunk = tid % 4;
            int gm_row = kv_start + row;
            __nv_bfloat16* smem_ptr = smem_K + row * D + col_chunk * 32;
            if (gm_row < S) {
                const __nv_bfloat16* gmem_ptr = K_base + gm_row * D + col_chunk * 32;
                for (int i = 0; i < 4; i++) {
                    uint32_t s_addr = (uint32_t)__cvta_generic_to_shared(smem_ptr + i * 8);
                    cp_async_16B(s_addr, gmem_ptr + i * 8);
                }
            } else {
                for (int i = 0; i < 4; i++) {
                    ((int4*)smem_ptr)[i] = make_int4(0, 0, 0, 0);
                }
            }
            cp_async_commit();
            cp_async_wait_all();
        }
        __syncthreads();

        // S = Q @ K^T  (M=128, N=64, K=128)
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_frag[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            wmma::fill_fragment(s_frag[i], 0.0f);

        {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;

            #pragma unroll
            for (int k_step = 0; k_step < D; k_step += WMMA_K) {
                #pragma unroll
                for (int m = 0; m < 2; ++m) {
                    wmma::load_matrix_sync(a_frag, smem_Q + (warp_m * 32 + m * 16) * D + k_step, D);
                    #pragma unroll
                    for (int n = 0; n < 2; ++n) {
                        int n_tile = warp_n * 2 + n;
                        wmma::load_matrix_sync(b_frag, smem_K + n_tile * 16 * D + k_step, D);
                        wmma::mma_sync(s_frag[m * 2 + n], a_frag, b_frag, s_frag[m * 2 + n]);
                    }
                }
            }
        }

        // Store S to SMEM
        #pragma unroll
        for (int m = 0; m < 2; ++m) {
            #pragma unroll
            for (int n = 0; n < 2; ++n) {
                int n_tile = warp_n * 2 + n;
                wmma::store_matrix_sync(smem_S + (warp_m * 32 + m * 16) * BN + n_tile * 16,
                                        s_frag[m * 2 + n], BN, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // Scale and mask
        for (int i = tid; i < BM * BN; i += NUM_THREADS) {
            int col = i % BN;
            smem_S[i] = (col >= actual_kv) ? -INFINITY : smem_S[i] * scale;
        }
        __syncthreads();

        // Issue V load (overlaps with softmax)
        {
            int row = tid / 4;
            int col_chunk = tid % 4;
            int gm_row = kv_start + row;
            __nv_bfloat16* smem_ptr = smem_V + row * D + col_chunk * 32;
            if (gm_row < S) {
                const __nv_bfloat16* gmem_ptr = V_base + gm_row * D + col_chunk * 32;
                for (int i = 0; i < 4; i++) {
                    uint32_t s_addr = (uint32_t)__cvta_generic_to_shared(smem_ptr + i * 8);
                    cp_async_16B(s_addr, gmem_ptr + i * 8);
                }
            } else {
                for (int i = 0; i < 4; i++) {
                    ((int4*)smem_ptr)[i] = make_int4(0, 0, 0, 0);
                }
            }
            cp_async_commit();
        }

        // Softmax: 2 threads per row, each handles 32 cols
        {
            int row = tid / 2;
            int half = tid % 2;
            int col_start = half * 32;
            bool valid = (row < actual_bm);

            // Pass 1: find row max
            float old_max = valid ? smem_rm[row] : 0.0f;
            float local_max = -INFINITY;
            if (valid) {
                #pragma unroll
                for (int c = 0; c < 32; ++c)
                    local_max = fmaxf(local_max, smem_S[row * BN + col_start + c]);
            }
            local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, 1));

            float new_max = valid ? fmaxf(old_max, local_max) : 0.0f;
            float rescale = valid ? __expf(old_max - new_max) : 1.0f;

            // Pass 2: compute exp, sum, P
            float local_sum = 0.0f;
            if (valid) {
                #pragma unroll
                for (int c = 0; c < 32; ++c) {
                    float val = __expf(smem_S[row * BN + col_start + c] - new_max);
                    local_sum += val;
                    smem_P[row * BN + col_start + c] = __float2bfloat16(val);
                }
            }
            local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, 1);

            if (valid) {
                float old_sum = smem_rs[row];
                float new_sum = old_sum * rescale + local_sum;
                if (half == 0) {
                    smem_rm[row] = new_max;
                    smem_rs[row] = new_sum;
                    smem_rescale[row] = rescale;
                }
            }
        }
        __syncthreads();

        // Wait for V load
        cp_async_wait_all();
        __syncthreads();

        // Rescale O fragments
        #pragma unroll
        for (int m = 0; m < 2; ++m) {
            int frag_row0 = warp_m * 32 + m * 16 + lane_id / 4;
            int frag_row1 = frag_row0 + 8;
            float r0 = (frag_row0 < actual_bm) ? smem_rescale[frag_row0] : 1.0f;
            float r1 = (frag_row1 < actual_bm) ? smem_rescale[frag_row1] : 1.0f;

            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                int idx = m * 4 + n;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    if (j < 2 || (j >= 4 && j < 6))
                        o_frag[idx].x[j] *= r0;
                    else
                        o_frag[idx].x[j] *= r1;
                }
            }
        }

        // O += P @ V  (M=128, N=128, K=64)
        {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> p_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> v_frag;

            #pragma unroll
            for (int k_step = 0; k_step < BN; k_step += WMMA_K) {
                #pragma unroll
                for (int m = 0; m < 2; ++m) {
                    wmma::load_matrix_sync(p_frag, smem_P + (warp_m * 32 + m * 16) * BN + k_step, BN);
                    #pragma unroll
                    for (int n = 0; n < 4; ++n) {
                        int n_tile = warp_n * 4 + n;
                        wmma::load_matrix_sync(v_frag, smem_V + k_step * D + n_tile * 16, D);
                        wmma::mma_sync(o_frag[m * 4 + n], p_frag, v_frag, o_frag[m * 4 + n]);
                    }
                }
            }
        }
        __syncthreads();
    }

    // Final normalization O /= sum
    #pragma unroll
    for (int m = 0; m < 2; ++m) {
        int frag_row0 = warp_m * 32 + m * 16 + lane_id / 4;
        int frag_row1 = frag_row0 + 8;
        float is0 = (frag_row0 < actual_bm && smem_rs[frag_row0] > 0.0f) ? 1.0f / smem_rs[frag_row0] : 0.0f;
        float is1 = (frag_row1 < actual_bm && smem_rs[frag_row1] > 0.0f) ? 1.0f / smem_rs[frag_row1] : 0.0f;

        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            int idx = m * 4 + n;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                if (j < 2 || (j >= 4 && j < 6))
                    o_frag[idx].x[j] *= is0;
                else
                    o_frag[idx].x[j] *= is1;
            }
        }
    }

    // Store O to SMEM (reuse smem_Q area as FP32 output, 128*128*4 = 64KB needed)
    // smem_Q is 32KB, smem_S is 32KB -> together 64KB for FP32 O output
    float* smem_O_out;
    #pragma unroll
    for (int m = 0; m < 2; ++m) {
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            int n_tile = warp_n * 4 + n;
            int abs_row = warp_m * 32 + m * 16;
            if (abs_row < 64) {
                smem_O_out = (float*)smem_Q + abs_row * D + n_tile * 16;
            } else {
                smem_O_out = (float*)smem_S + (abs_row - 64) * D + n_tile * 16;
            }
            wmma::store_matrix_sync(smem_O_out, o_frag[m * 4 + n], D, wmma::mem_row_major);
        }
    }
    __syncthreads();

    // Convert FP32 -> BF16 and write to global O
    for (int i = tid; i < BM * D / 8; i += NUM_THREADS) {
        int row = i / (D / 8);
        int col8 = i % (D / 8);
        int gm_row = m_start + row;
        if (gm_row < S) {
            float* fp32_ptr;
            if (row < 64) {
                fp32_ptr = (float*)smem_Q + row * D + col8 * 8;
            } else {
                fp32_ptr = (float*)smem_S + (row - 64) * D + col8 * 8;
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

    int smem_size = BM * D * 2 + BN * D * 2 * 2 + BM * BN * 4 + BM * 4 * 3;

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