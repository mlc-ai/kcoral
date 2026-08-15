#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define BLOCK_M 64
#define BLOCK_N 64
#define THREADS 256
#define HEAD_DIM 128

namespace mha_cuda {

__global__ void flash_attn_fwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S,
    int q_stride_s, int q_stride_h, int q_stride_b,
    int k_stride_s, int k_stride_h, int k_stride_b,
    int v_stride_s, int v_stride_h, int v_stride_b,
    int o_stride_s, int o_stride_h, int o_stride_b,
    int lse_stride_h, int lse_stride_b
) {
    int bx = blockIdx.x;
    int num_tiles_m = (S + BLOCK_M - 1) / BLOCK_M;
    int total_tiles = H * num_tiles_m;
    
    int batch_idx = bx / total_tiles;
    int rem = bx % total_tiles;
    int head_idx = rem / num_tiles_m;
    int tile_m = rem % num_tiles_m;

    if (batch_idx >= B || head_idx >= H || tile_m >= num_tiles_m) return;

    int local_S = S - tile_m * BLOCK_M;
    if (local_S <= 0) return;

    const __nv_bfloat16* q_ptr = Q + batch_idx * q_stride_b + head_idx * q_stride_h + tile_m * BLOCK_M * q_stride_s;
    const __nv_bfloat16* k_base = K + batch_idx * k_stride_b + head_idx * k_stride_h;
    const __nv_bfloat16* v_base = V + batch_idx * v_stride_b + head_idx * v_stride_h;
    __nv_bfloat16* o_ptr = O + batch_idx * o_stride_b + head_idx * o_stride_h + tile_m * BLOCK_M * o_stride_s;
    float* lse_ptr = LSE + batch_idx * lse_stride_b + head_idx * lse_stride_h + tile_m * BLOCK_M;

    // Shared memory layout
    // sK: BLOCK_N x HEAD_DIM bf16 (16KB)
    // sV: BLOCK_N x HEAD_DIM bf16 (16KB)
    // sO: BLOCK_M x HEAD_DIM float   (32KB)
    extern __shared__ char smem[];
    __nv_bfloat16* sK = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sV = sK + BLOCK_N * HEAD_DIM;
    float* sO = reinterpret_cast<float*>(sV + BLOCK_N * HEAD_DIM);

    // Initialize sO to zero (only valid rows)
    for (int i = threadIdx.x; i < BLOCK_M * HEAD_DIM; i += THREADS) {
        sO[i] = 0.0f;
    }
    __syncthreads();

    // Per-thread state for online softmax
    float row_max[4] = {-CUDART_INF_F};
    float row_sum[4] = {0.0f};
    
    // Register staging for attention scores: 4 rows x BLOCK_N cols
    float s_tile[4][BLOCK_N];
    // Q registers: 4 rows x HEAD_DIM
    float q_regs[4][HEAD_DIM];

    // Load Q tile into registers
    int tid = threadIdx.x;
    int row_start = tid * 4;
    bool row_valid[4];
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        int row = row_start + r;
        row_valid[r] = (row < local_S);
        if (row_valid[r]) {
            const __nv_bfloat16* pq = q_ptr + row * q_stride_s;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d += 2) {
                q_regs[r][d]   = __bfloat162float(pq[d]);
                q_regs[r][d+1] = __bfloat162float(pq[d+1]);
            }
        } else {
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) q_regs[r][d] = 0.0f;
        }
    }

    float scale = rsqrtf(static_cast<float>(HEAD_DIM));

    // Iterate over KV sequence
    for (int n_start = 0; n_start < S; n_start += BLOCK_N) {
        // Cooperative load of K and V
        int local_BN = min(BLOCK_N, S - n_start);
        for (int i = tid; i < local_BN * HEAD_DIM; i += THREADS) {
            int rn = i / HEAD_DIM;
            int c = i % HEAD_DIM;
            sK[i] = k_base[(n_start + rn) * k_stride_s + c];
            sV[i] = v_base[(n_start + rn) * v_stride_s + c];
        }
        // Pad remaining rows if incomplete tile
        if (threadIdx.x == 0 && local_BN < BLOCK_N) {
            for (int rn = local_BN; rn < BLOCK_N; ++rn) {
                for (int d = 0; d < HEAD_DIM; ++d) {
                    sK[rn * HEAD_DIM + d] = __float2bfloat16(0.0f);
                    sV[rn * HEAD_DIM + d] = __float2bfloat16(0.0f);
                }
            }
        }
        __syncthreads();

        // Compute S = Q @ K^T for 4 rows
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            if (!row_valid[r]) {
                #pragma unroll
                for (int k_c = 0; k_c < BLOCK_N; ++k_c) s_tile[r][k_c] = -CUDART_INF_F;
                continue;
            }
            #pragma unroll
            for (int k_c = 0; k_c < BLOCK_N; ++k_c) {
                float ds = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d += 4) {
                    ds += q_regs[r][d]   * __bfloat162float(sK[k_c * HEAD_DIM + d]);
                    ds += q_regs[r][d+1] * __bfloat162float(sK[k_c * HEAD_DIM + d+1]);
                    ds += q_regs[r][d+2] * __bfloat162float(sK[k_c * HEAD_DIM + d+2]);
                    ds += q_regs[r][d+3] * __bfloat162float(sK[k_c * HEAD_DIM + d+3]);
                }
                s_tile[r][k_c] = (k_c < local_BN) ? (ds * scale) : (-CUDART_INF_F);
            }
        }

        // Online softmax + O accumulation
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            if (!row_valid[r]) continue;
            
            // Find max in current tile
            float new_max = -CUDART_INF_F;
            #pragma unroll
            for (int k_c = 0; k_c < BLOCK_N; ++k_c)
                if (s_tile[r][k_c] > new_max) new_max = s_tile[r][k_c];
            
            new_max = fmaxf(row_max[r], new_max);
            float alpha = expf(row_max[r] - new_max);
            row_sum[r] *= alpha;

            // Scale existing sO and accumulate new contributions
            float p_sum = 0.0f;
            int sO_offset = (row_start + r) * HEAD_DIM;
            #pragma unroll
            for (int k_c = 0; k_c < BLOCK_N; ++k_c) {
                float p = expf(s_tile[r][k_c] - new_max);
                p_sum += p;
                
                // Accumulate P * V into shared sO
                float pv = p;
                int v_off = k_c * HEAD_DIM;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d += 2) {
                    atomicAdd(&sO[sO_offset + d],   pv * __bfloat162float(sV[v_off + d]));
                    atomicAdd(&sO[sO_offset + d+1], pv * __bfloat162float(sV[v_off + d+1]));
                }
            }
            // Apply scaling factor to existing accumulated values in sO
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d += 2) {
                atomicAdd(&sO[sO_offset + d],   (alpha - 1.0f) * sO[sO_offset + d]);
                atomicAdd(&sO[sO_offset + d+1], (alpha - 1.0f) * sO[sO_offset + d+1]);
            }

            row_sum[r] += p_sum;
            row_max[r] = new_max;
        }
        __syncthreads();
    }

    // Final normalization, convert to BF16 and write out
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        int row = row_start + r;
        if (!row_valid[r]) continue;
        
        float norm = (row_sum[r] == 0.0f) ? 0.0f : (1.0f / row_sum[r]);
        int out_off = row * o_stride_s;
        int sO_off = row * HEAD_DIM;
        
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d += 2) {
            o_ptr[out_off + d]   = __float2bfloat16(sO[sO_off + d]   * norm);
            o_ptr[out_off + d+1] = __float2bfloat16(sO[sO_off + d+1] * norm);
        }
        lse_ptr[row] = row_max[r] + logf(row_sum[r]);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int B = 4, H = 48, D = 128;
    int S = static_cast<int>(Q.size(2));
    
    // Strides for [B, H, S, D] layout
    int q_stride_s = D, q_stride_h = S * D, q_stride_b = H * S * D;
    int k_stride_s = D, k_stride_h = S * D, k_stride_b = H * S * D;
    int v_stride_s = D, v_stride_h = S * D, v_stride_b = H * S * D;
    int o_stride_s = D, o_stride_h = S * D, o_stride_b = H * S * D;
    int lse_stride_h = S, lse_stride_b = H * S;

    int num_tiles_m = (S + BLOCK_M - 1) / BLOCK_M;
    dim3 grid(B * H * num_tiles_m);
    dim3 block(THREADS);
    
    // SMEM: sK(16KB) + sV(16KB) + sO(32KB) = 64KB
    size_t smem_size = BLOCK_N * D * 2 + BLOCK_N * D * 2 + BLOCK_M * D * 4;

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    flash_attn_fwd_kernel<<<grid, block, smem_size, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<__nv_bfloat16*>(O.data_ptr()),
        static_cast<float*>(LSE.data_ptr()),
        B, H, S,
        q_stride_s, q_stride_h, q_stride_b,
        k_stride_s, k_stride_h, k_stride_b,
        v_stride_s, v_stride_h, v_stride_b,
        o_stride_s, o_stride_h, o_stride_b,
        lse_stride_h, lse_stride_b
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_cuda::run);

} // namespace mha_cuda