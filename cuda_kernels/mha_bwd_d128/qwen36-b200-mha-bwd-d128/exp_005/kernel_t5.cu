#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
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

namespace mha_bwd_opt {

/**
 * Correct multi-head attention backward kernel.
 * 
 * Layout: [B, H, S, d] flattened as BH × S × d
 * 
 * Architecture:
 * - Grid: B*H blocks (one per batch-head pair)
 * - Block: d=128 threads, one per feature dimension
 * - Shared memory stores Q row [d] + P[S] + dP[S]
 * 
 * Algorithm per query position qs:
 *   Phase A: Compute P[qs,ks] and dP[qs,ks] for all ks
 *   Phase B: Reduce to get correction = sum_{all ks} P * dP
 *   Phase C: Compute dQ (local), dK/dV (native bf16 atomicAdd)
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
    
    int64_t stride_bh = (int64_t)S * d;
    int64_t stride_seq = d;

    // Shared memory layout (all float32):
    // Q_smem[d]      = current Q row being processed
    // P_smem[S]      = P[qs, ks] for all key positions
    // dP_smem[S]     = dP[qs, ks] for all key positions
    extern __shared__ char smem[];
    float* Q_smem = reinterpret_cast<float*>(smem);
    float* P_smem = Q_smem + d;
    float* dP_smem = P_smem + S;

    // Process each query position (thread-striped across the block)
    for (int qs = tid; qs < S; qs += d) {
        float lse = L[bh * S + qs];
        int64_t qs_base = bh * stride_bh + qs * stride_seq;
        
        // === Load Q[qs, :] into shared memory ===
        Q_smem[tid] = static_cast<float>(__bfloat162float(Q[qs_base + tid]));
        __syncthreads();
        
        // === Phase A: Compute P[qs,ks] and dP[qs,ks] for all ks ===
        // Thread tid handles keys starting at tid with stride = d
        for (int ks = tid; ks < S; ks += d) {
            int64_t ks_base = bh * stride_bh + ks * stride_seq;
            
            float dot_s = 0.0f;
            float dot_dp = 0.0f;
            #pragma unroll 8
            for (int f = 0; f < d; ++f) {
                float qf = Q_smem[f];
                float kf = static_cast<float>(__bfloat162float(K[ks_base + f]));
                float vf = static_cast<float>(__bfloat162float(V[ks_base + f]));
                float dof = static_cast<float>(__bfloat162float(dO[qs_base + f]));
                dot_s  += qf * kf;
                dot_dp += dof * vf;
            }
            P_smem[ks]    = expf(dot_s * inv_scale - lse);
            dP_smem[ks]   = dot_dp;
        }
        __syncthreads();
        
        // === Phase B: Compute correction term ===
        // correction = sum_{ks=0}^{S-1} P[qs,ks] * dP[qs,ks]
        float local_corr = 0.0f;
        for (int ks = tid; ks < S; ks += d) {
            local_corr += P_smem[ks] * dP_smem[ks];
        }
        
        // Warp-level binary tree reduction (d=128 = 4 warps of 32 threads)
        float corr = local_corr;
        #pragma unroll
        for (int offset = 64; offset > 0; offset >>= 1) {
            float val = __shfl_down_sync(0xFFFFFFFF, corr, offset);
            corr += val;
        }
        // Broadcast thread 0's result to all threads
        corr = __shfl_sync(0xFFFFFFFF, corr, 0);
        
        // === Phase C: Compute gradients ===
        float dq_acc = 0.0f;
        float q_tid = Q_smem[tid];
        float dO_tid = static_cast<float>(__bfloat162float(dO[qs_base + tid]));
        
        for (int ks = tid; ks < S; ks += d) {
            float p  = P_smem[ks];
            float dp = dP_smem[ks];
            float ds = p * (dp - corr);
            
            // dQ[qs, tid] += ds * K[ks, tid]  (local accumulation, no atomic)
            float kf = static_cast<float>(__bfloat162float(K[bh * stride_bh + ks * stride_seq + tid]));
            dq_acc += ds * kf;
            
            // dK[ks, tid] += ds * Q[qs, tid]
            // Use native bf16 atomicAdd (supported on sm_90+/Blackwell)
            int64_t dk_idx = bh * stride_bh + ks * stride_seq + tid;
            atomicAdd(&dK[dk_idx], __float2bfloat16(ds * q_tid));
            
            // dV[ks, tid] += P[qs, ks] * dO[qs, tid]
            int64_t dv_idx = bh * stride_bh + ks * stride_seq + tid;
            atomicAdd(&dV[dv_idx], __float2bfloat16(p * dO_tid));
        }
        
        // Write dQ result (each (bh, qs, tid) is unique — no atomics needed)
        dQ[qs_base + tid] = __float2bfloat16(dq_acc * inv_scale);
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

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    // Zero-initialize dK and dV before launching kernel
    int64_t total_elements = B * H * S * d;
    void* ptrs[2] = {dK_ptr, dV_ptr};
    size_t bytes = static_cast<size_t>(total_elements) * sizeof(__nv_bfloat16);
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaMemsetAsync(ptrs[i], 0, bytes, stream));
    }

    int block_size = d_int;   // 128 threads per block
    int grid_size  = BH;      // B*H blocks total

    // Shared memory: d floats for Q + 2*S floats for P and dP
    int smem_bytes = d_int * sizeof(float) + 2 * S_int * sizeof(float);

    mha_bwd_kernel<<<grid_size, block_size, smem_bytes, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
        dQ_ptr, dK_ptr, dV_ptr,
        BH, S_int, d_int, inv_scale);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace mha_bwd_opt

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_opt::run);