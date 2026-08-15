#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cmath>
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

// Tile configuration: balance shared memory vs occupancy
constexpr int T_S = 128;   // Key/Value tile size along sequence
constexpr int THREADS = 128; // Matches fixed head dimension D=128

__global__ void MhaCausalKernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S, int D
) {
    uint32_t bid = blockIdx.x;
    uint32_t total_queries = static_cast<uint32_t>(B) * static_cast<uint32_t>(H) * static_cast<uint32_t>(S);
    if (bid >= total_queries) return;

    uint32_t b = bid / (H * S);
    uint32_t h = (bid % (H * S)) / S;
    uint32_t q_pos = bid % S;

    uint32_t tid = threadIdx.x;
    if (tid >= D) return;

    // Shared memory layout: sQ[D] + sK[T_S*D] + sV[T_S*D]
    extern __shared__ char smem_raw[];
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sK = sQ + D;
    __nv_bfloat16* sV = sK + T_S * D;

    // Cooperative load Q row into shared memory (coalesced)
    for (uint32_t i = tid; i < D; i += blockDim.x) {
        sQ[i] = Q[(b * H * S + h * S + q_pos) * D + i];
    }
    __syncthreads();

    float q_val = __bfloat162float(sQ[tid]);
    float inv_sqrt_d = 1.0f / sqrtf(static_cast<float>(D));
    
    // Incremental softmax state (thread-local, no race conditions)
    constexpr float NEG_INF = -1e20f;
    float p_max = NEG_INF;
    float p_sum = 0.0f;
    float acc_o = 0.0f;

    uint32_t num_tiles = (S + T_S - 1) / T_S;
    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        int kv_start = static_cast<int>(tile_id) * T_S;

        // Cooperative load K and V tiles
        for (int i = tid; i < T_S * D; i += blockDim.x) {
            int r = i / D;
            int c = i % D;
            int k_abs = kv_start + r;
            if (k_abs < S) {
                uint32_t idx = (b * H * S + h * S + k_abs) * D + c;
                sK[r * D + c] = K[idx];
                sV[r * D + c] = V[idx];
            }
        }
        __syncthreads();

        // Dot product & incremental softmax accumulation
        for (int r = 0; r < T_S; ++r) {
            int k_abs = kv_start + r;
            // Causal mask: attend only to past/current positions
            if (k_abs >= S || k_abs > static_cast<int>(q_pos)) continue;
            
            float k_val = __bfloat162float(sK[r * D + tid]);
            float v_val = __bfloat162float(sV[r * D + tid]);
            float score = q_val * k_val * inv_sqrt_d;

            if (score > p_max) {
                float scale = expf(p_max - score);
                p_sum = p_sum * scale + 1.0f;
                acc_o = acc_o * scale + v_val;
                p_max = score;
            } else {
                float weight = expf(score - p_max);
                p_sum += weight;
                acc_o += weight * v_val;
            }
        }
        // Barrier stabilizes shared memory for next tile iteration
        __syncthreads();
    }

    // Final normalization
    float out_val = 0.0f;
    float lse_val = NEG_INF;
    if (p_sum > 0.0f) {
        out_val = acc_o / p_sum;
        lse_val = p_max + logf(p_sum);
    }

    // Write results back to global memory
    O[(b * H * S + h * S + q_pos) * D + tid] = __float2bfloat16(out_val);
    if (tid == 0) {
        LSE[b * H * S + h * S + q_pos] = lse_val;
    }
}

namespace mha_impl {
void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int B = static_cast<int>(Q.size(0));
    int H = static_cast<int>(Q.size(1));
    int S = static_cast<int>(Q.size(2));
    int D = static_cast<int>(Q.size(3));

    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE.data_ptr());

    // Dynamic shared memory: sQ(D) + sK(T_S*D) + sV(T_S*D)
    size_t smem_elements = static_cast<size_t>(D) + 2ULL * T_S * static_cast<size_t>(D);
    size_t smem_bytes = smem_elements * sizeof(__nv_bfloat16);
    
    int blocks = B * H * S;
    int threads = D; // 128 threads per block
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    MhaCausalKernel<<<blocks, threads, smem_bytes, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr, B, H, S, D);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}
} // namespace mha_impl

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_impl::run);