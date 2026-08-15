#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <cstdio>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define TILE_N 64
#define D 128
#define INF 1e20f

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_impl {

__device__ __forceinline__ int my_min(int a, int b) { return a < b ? a : b; }

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

// Each block.y processes 128 consecutive query rows.
// Within a block, threads cooperate: each thread handles 1 query row, 
// and for each K-V tile, threads compute dot products collaboratively.
__global__ void mha_causal_kernel(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    __nv_bfloat16* O,
    float* LSE_out,
    int S, int B, int H) {

    const int bh_idx = blockIdx.x;
    if (bh_idx >= B * H) return;

    const int tid = threadIdx.x;  // 0..127
    const int q_row = blockIdx.y * 128 + tid;  // Each thread gets one query row
    if (q_row >= S) return;

    const long long bh_stride = (long long)S * D;
    const __nv_bfloat16* Q_bh = Q + bh_idx * bh_stride;
    const __nv_bfloat16* K_bh = K + bh_idx * bh_stride;
    const __nv_bfloat16* V_bh = V + bh_idx * bh_stride;
    __nv_bfloat16* O_bh = O + bh_idx * bh_stride;
    float* LSE_bh = LSE_out + bh_idx * S;

    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* smem_K = smem;
    __nv_bfloat16* smem_V = smem_K + TILE_N * D;

    const float inv_sqrt_d = 1.0f / sqrtf((float)D);
    const __nv_bfloat16* q_row_ptr = Q_bh + (long long)q_row * D;

    // Cache Q row into local registers (only D=128 elements, manageable)
    __shared__ float q_cache[128][D];
    if (tid < 128) {
        for (int d = 0; d < D; ++d) {
            q_cache[tid][d] = __bfloat162float(q_row_ptr[d]);
        }
    }
    __syncthreads();

    float max_s = -INF;
    float sum_e = 1.0f;
    
    // Accumulators stored in shared memory for later reads
    // Each thread computes its row's accumulator in registers
    __shared__ float acc_o_share[128][D];
    if (tid < 128) {
        for (int d = 0; d < D; ++d) acc_o_share[tid][d] = 0.0f;
    }
    __syncthreads();

    for (int k_tile = 0; k_tile < S; k_tile += TILE_N) {
        int k_end = my_min(k_tile + TILE_N, S);
        int tile_len = k_end - k_tile;

        // Cooperative load of K and V tiles
        for (int i = tid; i < tile_len * D; i += blockDim.x) {
            int r = i / D;
            int c = i % D;
            smem_K[i] = K_bh[(long long)(k_tile + r) * D + c];
            smem_V[i] = V_bh[(long long)(k_tile + r) * D + c];
        }
        __syncthreads();

        // Compute scores for this query row against all keys in tile
        float tile_max_local = -INF;
        float scores[TILE_N];
        
        for (int k_off = 0; k_off < tile_len; ++k_off) {
            bool masked = (k_tile + k_off > q_row);
            if (!masked) {
                float s = 0.0f;
                for (int d = 0; d < D; ++d) {
                    s += q_cache[tid][d] * __bfloat162float(smem_K[k_off * D + d]);
                }
                s *= inv_sqrt_d;
                scores[k_off] = s;
                tile_max_local = fmaxf(tile_max_local, s);
            } else {
                scores[k_off] = -INF;
                tile_max_local = fmaxf(tile_max_local, -INF);
            }
        }

        // Warp reduce to find tile_max
        float tile_max = warp_reduce_max(tile_max_local);

        // Skip if entire tile is masked
        if (tile_max > -0.5f * INF) {
            float p_scale = expf(max_s - tile_max);
            for (int d = 0; d < D; ++d) acc_o_share[tid][d] *= p_scale;
            sum_e *= p_scale;

            // Apply softmax and accumulate
            float local_sum = 0.0f;
            for (int k_off = 0; k_off < tile_len; ++k_off) {
                float s = scores[k_off];
                if (k_tile + k_off > q_row) s = -INF;
                float p = expf(s - tile_max);
                local_sum += p;
                for (int d = 0; d < D; ++d) {
                    acc_o_share[tid][d] += p * __bfloat162float(smem_V[k_off * D + d]);
                }
            }
            // Add to global sum_e via atomics or sequential access
            // Since only one thread accesses sum_e per tile, use atomicAdd
            atomicAdd(&sum_e, local_sum);
            max_s = tile_max;
        }
        __syncthreads();
    }

    // Write results
    float final_inv = 1.0f / sum_e;
    for (int d = 0; d < D; ++d) {
        O_bh[(long long)q_row * D + d] = __float2bfloat16(acc_o_share[tid][d] * final_inv);
    }
    LSE_bh[q_row] = max_s + logf(sum_e);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    const int64_t B = Q.size(0);
    const int64_t H = Q.size(1);
    const int64_t S_val = Q.size(2);
    
    const __nv_bfloat16* d_Q = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* d_K = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* d_V = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* d_O = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* d_LSE = static_cast<float*>(LSE.data_ptr());
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int S_int = static_cast<int>(S_val);
    int B_int = static_cast<int>(B);
    int H_int = static_cast<int>(H);
    
    dim3 grid(B_int * H_int, (S_int + 127) / 128);
    dim3 blk(128);
    
    // Shared memory: K cache + V cache + Q cache + acc_o share
    size_t smem_KV = 2ULL * TILE_N * D * sizeof(__nv_bfloat16);
    size_t smem_qcache = 128ULL * D * sizeof(float);
    size_t smem_acc = 128ULL * D * sizeof(float);
    size_t smem_bytes = smem_KV + smem_qcache + smem_acc;
    
    mha_causal_kernel<<<grid, blk, smem_bytes, stream>>>(
        d_Q, d_K, d_V, d_O, d_LSE, S_int, B_int, H_int);
        
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace mha_impl

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_impl::run);