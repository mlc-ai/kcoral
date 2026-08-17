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

#define TILE_M 64
#define TILE_N 64
#define HEAD_DIM 128
#define BLOCK_SIZE 128

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
    int num_q_tiles = (S + TILE_M - 1) / TILE_M;
    int total_tiles = H * num_q_tiles;
    
    int batch_idx = bx / total_tiles;
    int rem = bx % total_tiles;
    int head_idx = rem / num_q_tiles;
    int tile_q = rem % num_q_tiles;

    if (batch_idx >= B || head_idx >= H || tile_q >= num_q_tiles) return;

    int valid_rows = min(TILE_M, S - tile_q * TILE_M);
    int tid = threadIdx.x;

    // Shared memory: sK[TILE_N][HEAD_DIM] + sV[TILE_N][HEAD_DIM], both bf16
    extern __shared__ char smem[];
    __nv_bfloat16* sK = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sV = sK + TILE_N * HEAD_DIM;

    bool is_row_worker = (tid < valid_rows);
    int my_row = tid;  // Local row index within the Q-tile

    // Online softmax state for row workers
    float row_m = -CUDART_INF_F;
    float row_l = 0.0f;
    float o_reg[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) o_reg[d] = 0.0f;

    // Load Q row into registers (only row workers)
    float q_reg[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) q_reg[d] = 0.0f;

    if (is_row_worker) {
        int global_q_row = tile_q * TILE_M + my_row;
        const __nv_bfloat16* q_ptr = Q + batch_idx * q_stride_b + head_idx * q_stride_h 
                                   + global_q_row * q_stride_s;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d += 2) {
            q_reg[d]   = __bfloat162float(q_ptr[d]);
            q_reg[d+1] = __bfloat162float(q_ptr[d+1]);
        }
    }

    float scale = rsqrtf(static_cast<float>(HEAD_DIM));

    // Iterate over KV tiles
    for (int kv_start = 0; kv_start < S; kv_start += TILE_N) {
        int valid_n = min(TILE_N, S - kv_start);
        
        // Cooperative load K and V tiles (all BLOCK_SIZE threads participate)
        int total_elems = valid_n * HEAD_DIM;
        
        #pragma unroll 8
        for (int i = tid; i < total_elems; i += BLOCK_SIZE) {
            int rn = i / HEAD_DIM;
            int c = i % HEAD_DIM;
            sK[i] = *(K + batch_idx * k_stride_b + head_idx * k_stride_h 
                    + (kv_start + rn) * k_stride_s + c);
            sV[i] = *(V + batch_idx * v_stride_b + head_idx * v_stride_h 
                    + (kv_start + rn) * v_stride_s + c);
        }
        __syncthreads();

        // Row workers compute attention scores and update online softmax
        if (is_row_worker) {
            // Compute dot products: s_local[nc] = q_reg . sK[nc,:]
            float s_local[TILE_N];
            #pragma unroll
            for (int nc = 0; nc < valid_n; ++nc) {
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
            for (int nc = valid_n; nc < TILE_N; ++nc) s_local[nc] = -CUDART_INF_F;

            // Find tile max
            float tile_max = -CUDART_INF_F;
            #pragma unroll
            for (int nc = 0; nc < TILE_N; ++nc)
                if (s_local[nc] > tile_max) tile_max = s_local[nc];

            // Online softmax update
            float new_m = fmaxf(row_m, tile_max);
            float alpha = expf(row_m - new_m);
            row_l *= alpha;

            float p_sum = 0.0f;
            #pragma unroll
            for (int nc = 0; nc < TILE_N; ++nc) {
                float p = expf(s_local[nc] - new_m);
                p_sum += p;

                const __nv_bfloat16* pv = &sV[nc * HEAD_DIM];
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d += 2) {
                    o_reg[d]   = alpha * o_reg[d]   + p * __bfloat162float(pv[d]);
                    o_reg[d+1] = alpha * o_reg[d+1] + p * __bfloat162float(pv[d+1]);
                }
            }
            row_l += p_sum;
            row_m = new_m;
        }
        __syncthreads();
    }

    // Final normalization and store output + LSE
    if (is_row_worker) {
        float norm = (row_l == 0.0f || row_l != row_l) ? 0.0f : (1.0f / row_l);
        
        int global_q_row = tile_q * TILE_M + my_row;
        __nv_bfloat16* o_dst = O + batch_idx * o_stride_b + head_idx * o_stride_h 
                             + global_q_row * o_stride_s;
        float* lse_dst = LSE + batch_idx * lse_stride_b + head_idx * lse_stride_h + global_q_row;

        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d += 2) {
            o_dst[d]   = __float2bfloat16(o_reg[d]   * norm);
            o_dst[d+1] = __float2bfloat16(o_reg[d+1] * norm);
        }
        *lse_dst = row_m + logf(row_l);
    }
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
    dim3 grid(B * H * num_tiles_m);
    dim3 block(BLOCK_SIZE);
    
    // SMEM: sK + sV = 2 * TILE_N * HEAD_DIM * sizeof(bf16) = 2 * 64 * 128 * 2 = 32KB
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