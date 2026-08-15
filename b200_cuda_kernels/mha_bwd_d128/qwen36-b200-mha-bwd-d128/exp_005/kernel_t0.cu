#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_bwd {

/**
 * Multi-head attention backward kernel for BF16.
 * 
 * For each (batch, head) pair processed by one block:
 * - blockDim.x = d (head dimension)
 * - Each thread handles one feature position
 * - We iterate over query positions, computing attention scores and gradients
 * - dK and dV use atomicAdd since multiple query positions contribute to same target
 */
__global__ void mha_bwd_kernel(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    const __nv_bfloat16* O,
    const __nv_bfloat16* dO,
    const float* L,
    __nv_bfloat16* dQ,
    __nv_bfloat16* dK,
    __nv_bfloat16* dV,
    int BH,
    int S,
    int d,
    float inv_scale)
{
    int bh = blockIdx.x;
    int tid = threadIdx.x;
    int bsize = blockDim.x;  // equals d

    int64_t stride_bh = (int64_t)S * d;
    int64_t stride_seq = d;

    // Shared memory layout:
    // [0:d)         = Q[bh, q_s, :] for current query position
    // [d:d+S)       = S[q_s, ks] scores for current query  
    // [d+S:d+2S)    = dP[q_s, ks] for current query
    // Size = d + 2*S floats
    extern __shared__ char smem_raw[];
    float* Q_shared = reinterpret_cast<float*>(smem_raw);
    float* scores = Q_shared + d;
    float* dP_vals = scores + S;

    // Process each query position
    for (int q_s = 0; q_s < S; ++q_s) {
        // Load Q[bh, q_s, :] into shared memory
        Q_shared[tid] = static_cast<float>(__bfloat162float(Q[bh * stride_bh + q_s * stride_seq + tid]));
        __syncthreads();

        float lse = L[bh * S + q_s];

        // Compute attention scores S[q_s, ks] = dot(Q[q_s,:], K[ks,:]) / sqrt(d)
        // Thread tid computes one feature's contribution, then we reduce
        if (tid < S) {
            int ks = tid;
            float dot = 0.0f;
            for (int f = 0; f < d; ++f) {
                float qf = Q_shared[f];
                float kf = static_cast<float>(__bfloat162float(K[bh * stride_bh + ks * stride_seq + f]));
                dot += qf * kf;
            }
            scores[tid] = dot * inv_scale;
        }
        __syncthreads();

        // Compute dP[q_s, ks] = sum_f dO[q_s, f] * V[ks, f]
        if (tid < S) {
            int ks = tid;
            float dp = 0.0f;
            for (int f = 0; f < d; ++f) {
                float dof = static_cast<float>(__bfloat162float(dO[bh * stride_bh + q_s * stride_seq + f]));
                float vf = static_cast<float>(__bfloat162float(V[bh * stride_bh + ks * stride_seq + f]));
                dp += dof * vf;
            }
            dP_vals[tid] = dp;
        }
        __syncthreads();

        // Compute correction term: sum_{ks'} P[q_s, ks'] * dP[q_s, ks']
        float corr = 0.0f;
        for (int ks = 0; ks < S; ++ks) {
            float p = expf(scores[ks] - lse);
            corr += p * dP_vals[ks];
        }

        // Compute gradients
        // dS[q_s, ks] = P[q_s, ks] * (dP[q_s, ks] - corr)
        // dQ[q_s, tid] = sum_{ks} dS[q_s, ks] * K[ks, tid]  (local accumulation)
        // dK[ks, tid] += dS[q_s, ks] * Q[q_s, tid]           (atomic)
        // dV[ks, tid] += P[q_s, ks] * dO[q_s, tid]           (atomic)

        float dq_accum = 0.0f;
        float q_tid = Q_shared[tid];
        float dO_tid = static_cast<float>(__bfloat162float(dO[bh * stride_bh + q_s * stride_seq + tid]));

        for (int ks = 0; ks < S; ++ks) {
            float p = expf(scores[ks] - lse);
            float dp = dP_vals[ks];
            float ds = p * (dp - corr);

            float kf = static_cast<float>(__bfloat162float(K[bh * stride_bh + ks * stride_seq + tid]));
            dq_accum += ds * kf;

            // Atomic updates for dK and dV
            atomicAdd(reinterpret_cast<float*>(&dK[bh * stride_bh + ks * stride_seq + tid]), ds * q_tid);
            atomicAdd(reinterpret_cast<float*>(&dV[bh * stride_bh + ks * stride_seq + tid]), p * dO_tid);
        }

        // Write dQ (no atomic needed - each q_s is unique)
        dQ[bh * stride_bh + q_s * stride_seq + tid] = __float2bfloat16(dq_accum * inv_scale);
    }
}

void run(tvm::ffi::TensorView Q,
         tvm::ffi::TensorView K,
         tvm::ffi::TensorView V,
         tvm::ffi::TensorView O,
         tvm::ffi::TensorView dO,
         tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ,
         tvm::ffi::TensorView dK,
         tvm::ffi::TensorView dV)
{
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3);

    int BH = static_cast<int>(B * H);
    int S_int = static_cast<int>(S);
    int d_int = static_cast<int>(d);

    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_ptr = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr = static_cast<const float*>(L.data_ptr());

    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());

    float inv_scale = 1.0f / std::sqrt(static_cast<float>(d));

    int block_size = d_int;
    int grid_size = BH;

    // Shared memory: d floats for Q + 2*S floats for scores and dP
    int smem_bytes = d_int * sizeof(float) + 2 * S_int * sizeof(float);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    mha_bwd_kernel<<<grid_size, block_size, smem_bytes, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
        dQ_ptr, dK_ptr, dV_ptr,
        BH, S_int, d_int, inv_scale);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace mha_bwd

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd::run);