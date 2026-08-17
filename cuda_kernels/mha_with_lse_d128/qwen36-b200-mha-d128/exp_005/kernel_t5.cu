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
#define THREADS 256

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

    extern __shared__ char smem[];
    __nv_bfloat16* sK = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sV = sK + TILE_N * HEAD_DIM;

    // Each thread handles 4 Q-rows (THREADS=256, TILE_M=64 => 4 rows/thread... no, 256/64=4 threads per row)
    // Actually: 256 threads, 64 rows => each thread handles 4 rows
    // Threads 0..15 handle row 0..3, threads 16..31 handle row 4..7, etc.
    // row_base = (tid / 4) * 16 ... no, simpler: 
    // Each thread i owns rows: (i % 64)*4, wait that's complex.
    // Simpler: each thread owns 4 consecutive rows out of 64. Thread 0 owns rows 0,1,2,3.
    // But 256*4=1024 > 64. So: 256 threads, 64 rows => 4 threads per row.
    // Each thread processes all KV columns for its assigned rows.
    // Let's just do: each thread owns exactly 4 rows, cycling through:
    // thread t -> rows (t%64), (t%64)+16, (t%64)+32, (t%64)+48 -- too complex.
    
    // Simplest: 256 threads, but only first 64 do work, each owning 1 row.
    // But then K/V loading is inefficient.
    // Better: use cooperative loading with strided access, then each thread does its rows.
    
    // With 256 threads and 64 output rows, each thread handles 4 rows.
    // Thread tid owns rows: base_row = tid, with stride = TILE_M? No.
    // thread tid owns row_r where r = tid % TILE_M + (tid/TILE_M)*something? Messy.
    // 
    // Cleanest: Use round-robin assignment. Thread tid owns row = tid % TILE_M if tid < 4*TILE_M.
    // And each such thread owns 1 row. So only first 64 threads really matter? No, 256.
    // OK let me just do: every thread computes all TILE_M rows but that wastes 4x.
    // OR: thread owns rows [thread_idx] with pitch TILE_M across loop iterations.
    // Simplest working design: 64 active threads, each owns 1 row. Load KV cooperatively.
    if (tid >= valid_rows) return;

    int my_row = tid;
    int global_row = tile_q * TILE_M + my_row;

    const __nv_bfloat16* q_base = Q + batch_idx * q_stride_b + head_idx * q_stride_h 
                              + global_row * q_stride_s;
    __nv_bfloat16* o_dst = O + batch_idx * o_stride_b + head_idx * o_stride_h 
                        + global_row * o_stride_s;
    float* lse_dst = LSE + batch_idx * lse_stride_b + head_idx * lse_stride_h + global_row;

    // Load Q row into registers
    float q_reg[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        q_reg[d] = __bfloat162float(q_base[d]);
    }

    float row_max = -CUDART_INF_F;
    float row_sum = 0.0f;
    float o_reg[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) o_reg[d] = 0.0f;

    float scale = rsqrtf(static_cast<float>(HEAD_DIM));

    for (int kv_start = 0; kv_start < S; kv_start += TILE_N) {
        int valid_n = min(TILE_N, S - kv_start);
        
        // Cooperative load K and V tiles using strided access
        int k_v_len = valid_n * HEAD_DIM;
        #pragma unroll 4
        for (int i = tid; i < k_v_len; i += valid_rows) {
            int rn = i / HEAD_DIM;
            int c = i % HEAD_DIM;
            sK[i] = *(K + batch_idx * k_stride_b + head_idx * k_stride_h 
                    + (kv_start + rn) * k_stride_s + c);
            sV[i] = *(V + batch_idx * v_stride_b + head_idx * v_stride_h 
                    + (kv_start + rn) * v_stride_s + c);
        }
        __syncthreads();

        // Compute attention scores for my row against all KV rows in tile
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

        // Online softmax
        float tile_max = -CUDART_INF_F;
        #pragma unroll
        for (int nc = 0; nc < TILE_N; ++nc)
            if (s_local[nc] > tile_max) tile_max = s_local[nc];

        float new_max = fmaxf(row_max, tile_max);
        float alpha = expf(row_max - new_max);
        row_sum *= alpha;

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

    // Normalize and store
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
    dim3 grid(B * H * num_tiles_m);
    dim3 block(TILE_M);
    
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