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

constexpr int T_S = 128;   // Sequence tile size for K/V staging
constexpr int D_HEAD = 128; // Fixed head dimension per task spec

__global__ void MhaCausalKernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S, int D
) {
    // One block per (batch, head, query_position)
    int bid = blockIdx.x;
    if (bid >= B * H * S) return;

    int b = bid / (H * S);
    int h = (bid % (H * S)) / S;
    int q_pos = bid % S;

    int tid = threadIdx.x;
    if (tid >= D) return;

    // Shared memory layout: sQ[D] + sK[T_S*D] + sV[T_S*D]
    extern __shared__ char smem_char[];
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem_char);
    __nv_bfloat16* sK = &sQ[D];
    __nv_bfloat16* sV = &sK[T_S * D];

    // Load query row into shared memory
    sQ[tid] = Q[(b * H * S + h * S + q_pos) * D + tid];
    __syncthreads();

    float q_val = __bfloat162float(sQ[tid]);
    float inv_sqrt_d = 1.0f / sqrtf(static_cast<float>(D));
    
    // Incremental softmax state per thread
    constexpr float NEG_INF = -1e20f;
    float local_max = NEG_INF;
    float local_sum = 0.0f;
    float acc_o = 0.0f;

    int num_tiles = (S + T_S - 1) / T_S;
    for (int tile_id = 0; tile_id < num_tiles; ++tile_id) {
        int kv_start = tile_id * T_S;

        // Cooperative load K and V tiles
        for (int i = tid; i < T_S * D; i += blockDim.x) {
            int k_idx = i / D;
            int d_idx = i % D;
            int abs_k = kv_start + k_idx;
            if (abs_k < S) {
                int global_idx = (b * H * S + h * S + abs_k) * D + d_idx;
                sK[k_idx * D + d_idx] = K[global_idx];
                sV[k_idx * D + d_idx] = V[global_idx];
            }
        }
        __syncthreads();

        // Dot product & incremental softmax accumulation
        for (int k_off = 0; k_off < T_S; ++k_off) {
            int k_abs = kv_start + k_off;
            // Causal mask: only attend to past or current tokens
            if (k_abs < S && k_abs <= q_pos) {
                float k_val = __bfloat162float(sK[k_off * D + tid]);
                float v_val = __bfloat162float(sV[k_off * D + tid]);
                float score = q_val * k_val * inv_sqrt_d;

                if (score > local_max) {
                    float ratio = expf(local_max - score);
                    local_sum = local_sum * ratio + 1.0f;
                    acc_o = acc_o * ratio + v_val;
                    local_max = score;
                } else {
                    float ratio = expf(score - local_max);
                    local_sum += ratio;
                    acc_o += ratio * v_val;
                }
            }
        }
        __syncthreads();
    }

    // Final normalization
    float out_val = 0.0f;
    float lse_val = NEG_INF;
    if (local_sum > 0.0f) {
        out_val = acc_o / local_sum;
        lse_val = local_max + logf(local_sum);
    }

    // Write results
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

    // Launch configuration: 1 block per query position, 1 thread per head dimension
    int threads = D; 
    int blocks = B * H * S;
    
    // Dynamic shared memory: sQ(D) + sK(T_S*D) + sV(T_S*D)
    size_t smem_bytes = sizeof(__nv_bfloat16) * (static_cast<size_t>(D) + 2ULL * T_S * static_cast<size_t>(D));
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    MhaCausalKernel<<<blocks, threads, smem_bytes, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr, B, H, S, D);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}
} // namespace mha_impl

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_impl::run);