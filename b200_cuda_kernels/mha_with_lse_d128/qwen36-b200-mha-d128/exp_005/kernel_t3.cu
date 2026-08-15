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

#define TILE_M 32
#define TILE_N 64
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
    // One CTA processes one batch x head x (one or more Q-tiles)
    int bx = blockIdx.x;
    int num_q_tiles = (S + TILE_M - 1) / TILE_M;
    int total_tiles = H * num_q_tiles;
    
    int batch_idx = bx / total_tiles;
    int rem = bx % total_tiles;
    int head_idx = rem / num_q_tiles;
    int tile_q = rem % num_q_tiles;

    if (batch_idx >= B || head_idx >= H || tile_q >= num_q_tiles) return;

    int valid_rows = min(TILE_M, S - tile_q * TILE_M);
    int tid = threadIdx.x;

    extern __shared__ char smem[];
    __nv_bfloat16* sK = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sV = sK + TILE_N * HEAD_DIM;

    // Each thread handles TILE_M rows (threadIdx.x must equal TILE_M... no, we use one thread per row)
    // Actually, let's use TILE_M threads, each owning 1 row of output
    
    int row_in_tile = tid;
    if (row_in_tile >= valid_rows) return;

    int q_row = tile_q * TILE_M + row_in_tile;
    const __nv_bfloat16* q_base = Q + batch_idx * q_stride_b + head_idx * q_stride_h + q_row * q_stride_s;
    const __nv_bfloat16* k_ptr = K + batch_idx * k_stride_b + head_idx * k_stride_h;
    const __nv_bfloat16* v_ptr = V + batch_idx * v_stride_b + head_idx * v_stride_h;
    __nv_bfloat16* o_dst = O + batch_idx * o_stride_b + head_idx * o_stride_h + q_row * o_stride_s;
    float* lse_dst = LSE + batch_idx * lse_stride_b + head_idx * lse_stride_h + q_row;

    // Load Q row into registers
    float q_reg[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        q_reg[d] = __bfloat162float(q_base[d]);
    }

    // Initialize online softmax state
    float row_max = -CUDART_INF_F;
    float row_sum = 0.0f;
    
    // Output accumulator (FP32 precision)
    float o_reg[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) o_reg[d] = 0.0f;

    float scale = rsqrtf(static_cast<float>(HEAD_DIM));

    // Iterate over K/V tiles
    for (int kv_start = 0; kv_start < S; kv_start += TILE_N) {
        // Cooperative load of K and V tiles to shared memory
        int valid_n = min(TILE_N, S - kv_start);
        
        for (int i = tid; i < valid_n * HEAD_DIM; i += TILE_M) {
            int rn = i / HEAD_DIM;
            int c = i % HEAD_DIM;
            sK[i] = k_ptr[(kv_start + rn) * k_stride_s + c];
            sV[i] = v_ptr[(kv_start + rn) * v_stride_s + c];
        }
        __syncthreads();

        // Compute scores S[q_row] = q_reg @ K^T[tile]
        float s_local[TILE_N];
        #pragma unroll
        for (int nc = 0; nc < TILE_N; ++nc) {
            if (nc >= valid_n) {
                s_local[nc] = -CUDART_INF_F;
            } else {
                float ds = 0.0f;
                const __nv_bfloat16* pk = &sK[nc * HEAD_DIM];
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d += 4) {
                    ds += q_reg[d]   * __bfloat162float(pk[d]);
                    ds += q_reg[d+1] * __bfloat162float(pk[d+1]);
                    ds += q_reg[d+2] * __bfloat162float(pk[d+2]);
                    ds += q_reg[d+3] * __bfloat162float(pk[d+3]);
                }
                s_local[nc] = ds * scale;
            }
        }

        // Find max in this tile
        float tile_max = -CUDART_INF_F;
        #pragma unroll
        for (int nc = 0; nc < TILE_N; ++nc) {
            if (s_local[nc] > tile_max) tile_max = s_local[nc];
        }

        // Update online softmax
        float new_max = fmaxf(row_max, tile_max);
        float alpha = expf(row_max - new_max);
        row_sum *= alpha;

        // Accumulate P * V and update sum
        float p_sum = 0.0f;
        #pragma unroll
        for (int nc = 0; nc < TILE_N; ++nc) {
            float p = expf(s_local[nc] - new_max);
            p_sum += p;
            
            const __nv_bfloat16* pv = &sV[nc * HEAD_DIM];
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d += 2) {
                o_reg[d]   = alpha * o_reg[d]   + p * __bfloat162float(pv[d]);
                o_reg[d+1] = alpha * o_reg[d+1] + p * __bfloat162float(pv[d+1]);
            }
        }
        row_sum += p_sum;
        row_max = new_max;

        __syncthreads();
    }

    // Final normalization and store
    float norm = (row_sum == 0.0f) ? 0.0f : (1.0f / row_sum);
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d += 2) {
        o_dst[d]   = __float2bfloat16(o_reg[d]   * norm);
        o_dst[d+1] = __float2bfloat16(o_reg[d+1] * norm);
    }
    *lse_dst = row_max + logf(row_sum);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int B = 4, H = 48, D = 128;
    int S = static_cast<int>(Q.size(2));
    
    int q_stride_s = D, q_stride_h = S * D, q_stride_b = H * S * D;
    int k_stride_s = D, k_stride_h = S * D, k_stride_b = H * S * D;
    int v_stride_s = D, v_stride_h = S * D, v_stride_b = H * S * D;
    int o_stride_s = D, o_stride_h = S * D, o_stride_b = H * S * D;
    int lse_stride_h = S, lse_stride_b = H * S;

    int num_tiles_m = (S + TILE_M - 1) / TILE_M;
    int grid_size = B * H * num_tiles_m;
    dim3 grid(grid_size);
    dim3 block(TILE_M);
    
    // SMEM: sK + sV = 2 * TILE_N * HEAD_DIM * 2 bytes = 2 * 64 * 128 * 2 = 32768 = 32KB
    size_t smem_size = 2ULL * TILE_N * HEAD_DIM * sizeof(__nv_bfloat16);

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