#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>
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

namespace mha_impl {

template<int TILE_S>
__global__ void mha_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* LSE,
    int B, int H, int S, float scale) {
    
    int b = blockIdx.x;
    int h = blockIdx.y;
    int sq = blockIdx.z;
    if (b >= B || h >= H || sq >= S) return;

    int tid = threadIdx.x;
    constexpr int NUM_THREADS = 128;
    constexpr int D = 128;

    __shared__ __nv_bfloat16 sK[TILE_S][D];
    __shared__ __nv_bfloat16 sV[TILE_S][D];
    __shared__ float sDot[TILE_S];

    // Base pointer offset for this batch/head/query-pos
    uint64_t base_offset = (uint64_t)(b * H + h) * S * D + sq * D;
    float q_val = __bfloat162float(Q[base_offset + tid]);

    float o_acc = 0.0f;
    float m = -FLT_MAX;
    float d_sum = 0.0f;
    uint32_t full_mask = 0xFFFFFFFF;

    for (int base_sk = 0; base_sk < S; base_sk += TILE_S) {
        int tile_size = min(TILE_S, S - base_sk);

        // Zero shared memory accumulator for this tile
        for (int i = tid; i < TILE_S; i += NUM_THREADS) {
            sDot[i] = 0.0f;
        }
        __syncthreads();

        // Cooperative load of K and V tile into shared memory
        for (int r = 0; r < TILE_S; ++r) {
            int src_sk = base_sk + r;
            if (src_sk < S) {
                uint64_t src_off = (uint64_t)(b * H + h) * S * D + src_sk * D;
                sK[r][tid] = K[src_off + tid];
                sV[r][tid] = V[src_off + tid];
            }
        }
        __syncthreads();

        // Compute attention scores and apply online softmax
        for (int r = 0; r < tile_size; ++r) {
            float k_val = __bfloat162float(sK[r][tid]);
            float prod = q_val * k_val * scale;

            // Warp-level reduction of the dot product
            float warpSum = prod;
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                warpSum += __shfl_down_sync(full_mask, warpSum, offset);
            }
            // Atomically add warp sum to shared accumulator
            if ((tid & 31) == 0) {
                atomicAdd(&sDot[r], warpSum);
            }
            __syncthreads(); // Wait for all atomicAdds to complete

            float score = sDot[r];

            // Online softmax state update (numerically stable)
            float max_new = max(m, score);
            float m_old = m;
            m = max_new;
            float scale_factor = expf(m_old - m);
            d_sum = d_sum * scale_factor + expf(score - m);
            o_acc *= scale_factor;

            float v_val = __bfloat162float(sV[r][tid]);
            o_acc += v_val * expf(score - m);
        }
    }

    // Final normalization and store
    o_acc /= d_sum;
    O[base_offset + tid] = __float2bfloat16(o_acc);

    if (tid == 0) {
        uint64_t lse_off = (uint64_t)(b * H + h) * S + sq;
        LSE[lse_off] = m + logf(d_sum);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
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

    float scale = 1.0f / sqrtf(static_cast<float>(D));

    constexpr int TILE_S = 64;
    constexpr int NUM_THREADS = 128;

    dim3 grid(B, H, S);
    dim3 block(NUM_THREADS);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    mha_kernel<TILE_S><<<grid, block, 0, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data,
        static_cast<int>(B), static_cast<int>(H), static_cast<int>(S), scale);
        
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

} // namespace mha_impl

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_impl::run);