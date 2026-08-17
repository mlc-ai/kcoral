#include <cuda_runtime.h>
#include <cuda_bf16.h>
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

namespace mha_impl {

constexpr int D = 128;
constexpr int MQ = 128;
constexpr int BK = 64;
constexpr int THD = 64;
constexpr int COLS_PER_THD = 2;

template<int Dim_D, int Dim_MQ, int Dim_BK, int Dim_THD, int Dim_CPT>
__global__ void mha_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S, float scale) {

    __shared__ __nv_bfloat16 sK[Dim_BK][Dim_D];
    __shared__ __nv_bfloat16 sV[Dim_BK][Dim_D];

    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    int q_base = blockIdx.y * Dim_MQ;
    int tid = threadIdx.x;

    uint64_t base_off = (uint64_t)bh * S * Dim_D;

    float m_reg[Dim_MQ];
    float d_reg[Dim_MQ];
    float acc0[Dim_MQ];
    float acc1[Dim_MQ];
    __nv_bfloat16 q_frag[Dim_MQ][Dim_CPT];

    // Initialize registers and load Q tile
    #pragma unroll
    for (int i = 0; i < Dim_MQ; ++i) {
        m_reg[i] = -FLT_MAX;
        d_reg[i] = 0.0f;
        acc0[i] = 0.0f;
        acc1[i] = 0.0f;

        int q_idx = q_base + i;
        if (q_idx < S) {
            uint64_t q_off = base_off + (uint64_t)q_idx * Dim_D;
            q_frag[i][0] = Q[q_off + tid * Dim_CPT];
            q_frag[i][1] = Q[q_off + tid * Dim_CPT + 1];
        } else {
            q_frag[i][0] = __float2bfloat16(0.0f);
            q_frag[i][1] = __float2bfloat16(0.0f);
        }
    }

    // Iterate over KV tiles
    for (int kv_base = 0; kv_base < S; kv_base += Dim_BK) {
        // Load K and V tiles cooperatively
        for (int k = 0; k < Dim_BK; ++k) {
            int kv_idx = kv_base + k;
            uint64_t kv_off = base_off + (uint64_t)kv_idx * Dim_D;
            if (kv_idx < S) {
                sK[k][tid * Dim_CPT] = K[kv_off + tid * Dim_CPT];
                sK[k][tid * Dim_CPT + 1] = K[kv_off + tid * Dim_CPT + 1];
                sV[k][tid * Dim_CPT] = V[kv_off + tid * Dim_CPT];
                sV[k][tid * Dim_CPT + 1] = V[kv_off + tid * Dim_CPT + 1];
            }
        }
        __syncthreads();

        // Process each query row in the current Q tile
        for (int q = 0; q < Dim_MQ; ++q) {
            if (q_base + q >= S) continue;

            // Find maximum score within this KV tile
            float tile_max = -FLT_MAX;
            for (int k = 0; k < Dim_BK; ++k) {
                float s = __bfloat162float(q_frag[q][0]) * __bfloat162float(sK[k][tid * Dim_CPT])
                        + __bfloat162float(q_frag[q][1]) * __bfloat162float(sK[k][tid * Dim_CPT + 1]);
                s *= scale;
                if (s > tile_max) tile_max = s;
            }

            // Online softmax state update
            float old_m = m_reg[q];
            float new_m = fmaxf(old_m, tile_max);
            float alpha = expf(old_m - new_m);
            m_reg[q] = new_m;
            d_reg[q] *= alpha;
            acc0[q] *= alpha;
            acc1[q] *= alpha;

            // Accumulate weighted V and update denominator
            for (int k = 0; k < Dim_BK; ++k) {
                float s = __bfloat162float(q_frag[q][0]) * __bfloat162float(sK[k][tid * Dim_CPT])
                        + __bfloat162float(q_frag[q][1]) * __bfloat162float(sK[k][tid * Dim_CPT + 1]);
                s *= scale;
                
                float p = expf(s - new_m);
                d_reg[q] += p;
                acc0[q] += p * __bfloat162float(sV[k][tid * Dim_CPT]);
                acc1[q] += p * __bfloat162float(sV[k][tid * Dim_CPT + 1]);
            }
        }
        __syncthreads();
    }

    // Normalize, convert to bf16, and store to global memory
    for (int q = 0; q < Dim_MQ; ++q) {
        if (q_base + q >= S) continue;
        float inv_d = 1.0f / d_reg[q];
        uint64_t out_off = base_off + (uint64_t)(q_base + q) * Dim_D;
        O[out_off + tid * Dim_CPT] = __float2bfloat16(acc0[q] * inv_d);
        O[out_off + tid * Dim_CPT + 1] = __float2bfloat16(acc1[q] * inv_d);

        if (tid == 0) {
            uint64_t lse_off = (uint64_t)bh * S + (q_base + q);
            LSE[lse_off] = m_reg[q] + logf(d_reg[q]);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D_dim = Q.size(3); // Expected to be 128
    
    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    float scale = 1.0f / sqrtf(static_cast<float>(D_dim));

    dim3 grid(B * H, (S + MQ - 1) / MQ);
    dim3 block(THD);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    mha_kernel<D, MQ, BK, THD, COLS_PER_THD><<<grid, block, 0, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data,
        static_cast<int>(B), static_cast<int>(H), static_cast<int>(S), scale);
        
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

} // namespace mha_impl

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_impl::run);