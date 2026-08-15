#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_causal_d128 {

constexpr int BLOCK_M = 64;  // Q rows per tile
constexpr int BD      = 128; // Head dimension
constexpr int GROUP_D = 4;   // Output dims processed per thread per sweep

__global__ void mha_causal_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
          __nv_bfloat16* __restrict__ O,
    float*                 __restrict__ LSE_out,
    int B, int H, int S,
    float scale)
{
    // Each thread processes one (q_row, dim_group) across ALL key positions
    int b = blockIdx.x / H;
    int h = blockIdx.x % H;
    
    int64_t base = (int64_t)b * H * S * BD + (int64_t)h * S * BD;

    // Q-tile offset (grid.y indexes which set of 64 Q rows)
    int q_tile_start = blockIdx.y * BLOCK_M;
    if (q_tile_start >= S) return;

    // Thread processes 4 output dims: start_d = threadIdx.x * GROUP_D
    int tid = threadIdx.x;  // 0..63 (blockDim.x = 64)
    int d0 = tid * GROUP_D;
    if (d0 >= BD) return;

    // Query row within this tile (same for all threads in the block)
    // We use a single shared variable indexed differently - actually we process 
    // one query row PER thread in this tile for better register usage
    int q_local = tid;  // thread 0..63 -> q_local 0..63
    int q_global = q_tile_start + q_local;
    if (q_global >= S) return;

    // Load Q[q_local, d0:d0+4] into registers
    float q[GROUP_D];
    for (int di = 0; di < GROUP_D; ++di) {
        q[di] = __bfloat162float(Q[base + (int64_t)q_global * BD + d0 + di]);
    }

    // Online softmax state
    float row_max = -1e20f;
    float row_sum = 0.f;
    float acc[GROUP_D] = {0.f, 0.f, 0.f, 0.f};

    // Scan through all key positions
    for (int k = 0; k < S; ++k) {
        // Dot product Q[q_global, :] · K[k, :] for our 4 dims
        float val[GROUP_D] = {0.f, 0.f, 0.f, 0.f};
        int64_t k_base_dim = base + (int64_t)k * BD;
        int64_t q_base_dim = base + (int64_t)q_global * BD;
        
        #pragma unroll
        for (int di = 0; di < GROUP_D; ++di) {
            // Accumulate dot product for this output dim
            float s = 0.f;
            for (int d = 0; d < BD; d += 4) {
                float qa = __bfloat162float(Q[q_base_dim + d]);
                float qb = __bfloat162float(Q[q_base_dim + d + 1]);
                float qc = __bfloat162float(Q[q_base_dim + d + 2]);
                float qd = __bfloat162float(Q[q_base_dim + d + 3]);
                float ka = __bfloat162float(K[k_base_dim + d]);
                float kb = __bfloat162float(K[k_base_dim + d + 1]);
                float kc = __bfloat162float(K[k_base_dim + d + 2]);
                float kd = __bfloat162float(K[k_base_dim + d + 3]);
                s += qa*ka + qb*kb + qc*kc + qd*kd;
            }
            s *= scale;
            
            // Causal mask
            if (k > q_global) {
                s = -1e20f;
            }
            
            // Online softmax update for THIS output-dim group
            // Actually softmax is per-query ROW, not per-output-dim
            // So all 4 dims share the same attention weight
            val[di] = s;
        }
        
        // Use val[0] (they're all the same attention logit for this q,k pair)
        float s = val[0];
        
        // Online softmax
        float alpha;
        if (s > row_max) {
            float diff = row_max - s;
            float correction = expf(diff);
            for (int di = 0; di < GROUP_D; ++di) {
                acc[di] *= correction;
            }
            row_sum *= correction;
            alpha = 1.f;
            row_max = s;
        } else {
            alpha = expf(s - row_max);
        }
        row_sum += alpha;
        
        // Accumulate weighted V
        int64_t v_base_dim = base + (int64_t)k * BD;
        for (int di = 0; di < GROUP_D; ++di) {
            acc[di] += alpha * __bfloat162float(V[v_base_dim + d0 + di]);
        }
    }

    // Normalize and write output
    if (row_sum > 0.f && q_global < S) {
        float inv_sum = 1.f / row_sum;
        for (int di = 0; di < GROUP_D; ++di) {
            int idx = (int)(base + (int64_t)q_global * BD + d0 + di);
            O[idx] = __float2bfloat16(acc[di] * inv_sum);
        }
        
        // Write LSE (only once per q_global)
        if (tid == 0) {
            LSE_out[b * H * S + h * S + q_global] = row_max + logf(row_sum);
        }
    } else if (q_global < S) {
        // All masked - fill with zero output and sentinel LSE
        for (int di = 0; di < GROUP_D; ++di) {
            int idx = (int)(base + (int64_t)q_global * BD + d0 + di);
            O[idx] = __float2bfloat16(0.f);
        }
        if (tid == 0) {
            LSE_out[b * H * S + h * S + q_global] = -1e10f;
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE)
{
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);

    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    int64_t num_bh = B * H;
    int64_t num_m_tiles = (S + BLOCK_M - 1) / BLOCK_M;
    
    // grid.x = batch*head, grid.y = Q-tile, block.x = 64 threads (one per Q row in tile)
    dim3 grid(num_bh, num_m_tiles, 1);
    dim3 block(BLOCK_M, 1, 1);

    float scale = rsqrtf(static_cast<float>(D));

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    mha_causal_kernel<<<grid, block, 0, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data,
        static_cast<int>(B), static_cast<int>(H), static_cast<int>(S),
        scale);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_causal_d128::run);

}  // namespace mha_causal_d128