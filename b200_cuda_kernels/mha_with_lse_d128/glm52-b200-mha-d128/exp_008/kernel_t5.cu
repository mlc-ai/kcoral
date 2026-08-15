#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cmath>
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

namespace flash_attn {

constexpr int D = 128;
constexpr int BM = 128;
constexpr int BK = 64;
constexpr int THREADS = 128;

__global__ void attention_kernel(
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
    int q_idx = q_start + tid;
    int warp_id = tid / 32;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* Q_tile = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* K_tile = Q_tile + BM * D;
    __nv_bfloat16* V_tile = K_tile + BK * D;
    __nv_bfloat16* P_bf16_tile = V_tile + BK * D;
    float* P_tile = reinterpret_cast<float*>(P_bf16_tile + BM * BK);
    float* O_tile = P_tile + BM * BK;

    int bh = (b * H + h);
    const __nv_bfloat16* Q_base = Q + (uint64_t)bh * S * D;
    const __nv_bfloat16* K_base = K + (uint64_t)bh * S * D;
    const __nv_bfloat16* V_base = V + (uint64_t)bh * S * D;
    __nv_bfloat16* O_base = O + (uint64_t)bh * S * D;
    float* LSE_base = LSE + (uint64_t)bh * S;

    // Load Q_tile
    for (int i = tid; i < BM * D / 8; i += THREADS) {
        int row = i / (D / 8);
        int col = i % (D / 8);
        if (q_start + row < S) {
            reinterpret_cast<uint4*>(Q_tile)[i] = reinterpret_cast<const uint4*>(Q_base + (uint64_t)(q_start + row) * D)[col];
        } else {
            reinterpret_cast<uint4*>(Q_tile)[i] = make_uint4(0, 0, 0, 0);
        }
    }

    // Init O_tile
    for (int i = tid; i < BM * D; i += THREADS) {
        O_tile[i] = 0.0f;
    }
    __syncthreads();

    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    const float scale = 0.0883883476f;

    int num_k_blocks = (S + BK - 1) / BK;

    for (int kb = 0; kb < num_k_blocks; kb++) {
        int k_start = kb * BK;

        // Load K and V tiles
        for (int i = tid; i < BK * D / 8; i += THREADS) {
            int row = i / (D / 8);
            int col = i % (D / 8);
            if (k_start + row < S) {
                reinterpret_cast<uint4*>(K_tile)[i] = reinterpret_cast<const uint4*>(K_base + (uint64_t)(k_start + row) * D)[col];
                reinterpret_cast<uint4*>(V_tile)[i] = reinterpret_cast<const uint4*>(V_base + (uint64_t)(k_start + row) * D)[col];
            } else {
                reinterpret_cast<uint4*>(K_tile)[i] = make_uint4(0, 0, 0, 0);
                reinterpret_cast<uint4*>(V_tile)[i] = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        // QK^T using WMMA - each warp computes 32x64 tile of P
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[2][4];

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                wmma::fill_fragment(c_frag[i][j], 0.0f);
            }
        }

        #pragma unroll
        for (int k = 0; k < 8; k++) {
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                wmma::load_matrix_sync(a_frag, Q_tile + (warp_id * 32 + i * 16) * D + k * 16, D);
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    wmma::load_matrix_sync(b_frag, K_tile + (k * 16) * D + j * 16, D);
                    wmma::mma_sync(c_frag[i][j], a_frag, b_frag, c_frag[i][j]);
                }
            }
        }

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                wmma::store_matrix_sync(P_tile + (warp_id * 32 + i * 16) * BK + j * 16, c_frag[i][j], BK, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // Softmax per row
        float* P_row = P_tile + tid * BK;
        __nv_bfloat16* P_bf16_row = P_bf16_tile + tid * BK;
        float* O_row = O_tile + tid * D;

        float block_max = -INFINITY;
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            P_row[k] *= scale;
            block_max = fmaxf(block_max, P_row[k]);
        }

        float new_max = fmaxf(max_score, block_max);
        float correction = (max_score > -INFINITY) ? __expf(max_score - new_max) : 0.0f;

        float block_sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float p = __expf(P_row[k] - new_max);
            P_row[k] = p;
            block_sum += p;
            P_bf16_row[k] = __float2bfloat16(p);
        }

        sum_exp = sum_exp * correction + block_sum;

        #pragma unroll
        for (int d = 0; d < D; d++) {
            O_row[d] *= correction;
        }

        max_score = new_max;
        __syncthreads();

        // PV using scalar FMA
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float p = P_row[k];
            __nv_bfloat16* V_row = V_tile + k * D;
            #pragma unroll
            for (int d = 0; d < D; d++) {
                O_row[d] = fmaf(p, __bfloat162float(V_row[d]), O_row[d]);
            }
        }
        __syncthreads();
    }

    // Finalize
    if (q_idx < S) {
        float inv_sum = 1.0f / sum_exp;
        __nv_bfloat16* O_row = O_base + (uint64_t)q_idx * D;
        float* O_acc_row = O_tile + tid * D;
        
        #pragma unroll
        for (int d = 0; d < D; d += 4) {
            O_row[d]     = __float2bfloat16(O_acc_row[d]     * inv_sum);
            O_row[d + 1] = __float2bfloat16(O_acc_row[d + 1] * inv_sum);
            O_row[d + 2] = __float2bfloat16(O_acc_row[d + 2] * inv_sum);
            O_row[d + 3] = __float2bfloat16(O_acc_row[d + 3] * inv_sum);
        }

        LSE_base[q_idx] = max_score + __logf(sum_exp);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);

    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    int num_q_blocks = (S + BM - 1) / BM;
    dim3 grid((int)B, (int)H, num_q_blocks);
    dim3 block(THREADS);

    // Shared memory: Q(32K) + K(16K) + V(16K) + P_bf16(16K) + P(32K) + O(64K) = 176KB
    size_t smem_size = BM * D * sizeof(__nv_bfloat16) + 
                       2 * BK * D * sizeof(__nv_bfloat16) +
                       BM * BK * sizeof(__nv_bfloat16) +
                       BM * BK * sizeof(float) +
                       BM * D * sizeof(float);

    cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size);

    attention_kernel<<<grid, block, smem_size, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data, (int)B, (int)H, (int)S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flash_attn::run);

}  // namespace flash_attn