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
 * Optimized MHA backward kernel with KV tiling.
 * 
 * Layout: [B, H, S, d] flattened as BH × S × d
 * 
 * Architecture:
 * - Grid: B*H blocks (one per batch-head pair)
 * - Block: d=128 threads, one per feature dimension
 * 
 * Key optimization: online accumulation
 * Instead of storing P[S] and dP[S] in shared memory (too much for S=4096),
 * we stream through KV tiles and accumulate:
 *   - corr_accum += P[ks] * dP[ks]     (running correction term)
 *   - dq_acc[tid] += ds[ks] * K[ks,tid]  (dQ accumulator per feature)
 *   - dK[ks,tid] += ds[ks] * Q[qs,tid]    (atomic: multiple qs contribute)
 *   - dV[ks,tid] += P[ks] * dO[qs,tid]    (atomic: multiple qs contribute)
 * 
 * After all KV tiles, apply correction retroactively:
 *   dQ_final[tid] = dq_acc_pre_corr[tid] - corr * P_K_weighted[tid]
 *                 = dq_with_zero_correction - corr * sum(P*K)
 * 
 * We maintain two parallel accumulators:
 *   - Without correction: ds_raw = P * dP (pretend corr=0)
 *   - Correction weight: P_only = P
 * Then: true_dq = sum(ds_raw * K) - corr * sum(P * K)
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
    int lane = tid % 32;
    int warp = tid / 32;
    int num_warps = d / 32;
    
    int64_t stride_bh = (int64_t)S * d;
    int64_t stride_seq = d;

    // Shared memory layout:
    // Q_smem[d]:          current Q row (loaded once per qs, reused for all kv tiles)
    // K_smem[d][KV_TILE]: K tile (transposed for coalesced load)  
    // V_smem[d][KV_TILE]: V tile (transposed)
    // dO_smem[d]:         dO row for current qs
    // warp_corr[num_warps]: cross-warp reduction scratch
    extern __shared__ char smem[];
    float* Q_smem = reinterpret_cast<float*>(smem);
    float* K_smem = Q_smem + d;
    float* V_smem = K_smem + d * 32;
    float* dO_smem = V_smem + d * 32;
    float* warp_corr = dO_smem + d;

    constexpr int KV_TILE = 32;
    int num_kv_tiles = (S + KV_TILE - 1) / KV_TILE;

    // Process each query position (thread-striped)
    for (int qs = tid; qs < S; qs += d) {
        float lse = L[bh * S + qs];
        int64_t qs_base = bh * stride_bh + qs * stride_seq;
        
        // === Load Q[qs, :] and dO[qs, :] into shared memory ===
        Q_smem[tid] = static_cast<float>(__bfloat162float(Q[qs_base + tid]));
        dO_smem[tid] = static_cast<float>(__bfloat162float(dO[qs_base + tid]));
        __syncthreads();
        
        // Online accumulators for dQ (per feature, local to thread)
        float dq_pdp_acc = 0.0f;      // sum(P[ks]*dP[ks] * K[ks,tid]) — dQ contribution WITHOUT correction
        float dq_p_acc = 0.0f;         // sum(P[ks] * K[ks,tid]) — correction weight for dQ
        float local_corr = 0.0f;       // sum(P[ks]*dP[ks]) — will become correction term after all tiles
        
        // Stream through KV tiles
        for (int kti = 0; kti < num_kv_tiles; ++kti) {
            int ks_start = kti * KV_TILE;
            int ks_end = min(ks_start + KV_TILE, S);
            int ks_valid = ks_end - ks_start;
            
            // Load K and V tiles into shared memory
            // Coalesced: each thread loads one element per key position
            for (int ki = tid; ki < d * ks_valid; ki += d) {
                int f = ki / ks_valid;           // feature 0..d-1
                int ki_local = ki % ks_valid;    // local key pos 0..ks_valid-1
                int ks_global = ks_start + ki_local;
                int64_t idx = bh * stride_bh + ks_global * stride_seq + f;
                K_smem[f * KV_TILE + ki_local] = static_cast<float>(__bfloat162float(K[idx]));
                V_smem[f * KV_TILE + ki_local] = static_cast<float>(__bfloat162float(V[idx]));
            }
            __syncthreads();
            
            // Process each key in this tile
            for (int ki_local = 0; ki_local < ks_valid; ++ki_local) {
                int ks_global = ks_start + ki_local;
                
                // Compute dot products for score and dP
                float dot_s = 0.0f;
                float dot_dp = 0.0f;
                #pragma unroll 8
                for (int f = 0; f < d; ++f) {
                    dot_s += Q_smem[f] * K_smem[f * KV_TILE + ki_local];
                    dot_dp += dO_smem[f] * V_smem[f * KV_TILE + ki_local];
                }
                
                float s_val = dot_s * inv_scale;
                float p = expf(s_val - lse);
                float dp = dot_dp;
                
                // Accumulate correction term components
                local_corr += p * dp;
                
                // Online dQ accumulator:
                // dq_contribution_without_correction = p * dp * K[ks, tid]
                float kt = K_smem[tid * KV_TILE + ki_local];
                dq_pdp_acc += (p * dp) * kt;
                
                // Correction weight for dQ: sum(P * K) — needed to subtract later
                dq_p_acc += p * kt;
                
                // Full dS for this ks: ds = p * (dp - corr)
                // But we don't know corr yet! So compute dK/dV with deferred correction.
                // 
                // dK[ks,tid] = ds * Q[qs,tid] = p*(dp-corr)*Q = p*dp*Q - p*corr*Q
                //   → atomic add: p*dp*Q now, subtract p*corr*Q later
                // Similarly for dV: P*dO now (unchanged by correction)
                
                float q_tid = Q_smem[tid];
                float do_tid = dO_smem[tid];
                
                // dK: immediate contribution (without correction)
                int64_t dk_idx = bh * stride_bh + ks_global * stride_seq + tid;
                atomicAdd(&dK[dk_idx], __float2bfloat16(p * dp * q_tid));
                
                // Save P*tid*q for later correction subtraction via shared mem + atomic
                // Actually we need to subtract p*corr*Q for EACH ks.
                // Store p*q per-feature in a small array? Too much for S=4096.
                // 
                // Alternative: after getting corr, do ANOTHER pass through KV tiles
                // to subtract corr*P*Q from dK.
                // For dV: no correction needed! P*dO is already complete.
                
                // dV: complete (no correction applies to dV)
                int64_t dv_idx = bh * stride_bh + ks_global * stride_seq + tid;
                atomicAdd(&dV[dv_idx], __float2bfloat16(p * do_tid));
            }
            __syncthreads();
        }
        
        // === Reduce local_corr across the block ===
        // Warp intra-reduction
        float warp_sum = local_corr;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float val = __shfl_down_sync(0xFFFFFFFF, warp_sum, offset);
            warp_sum += val;
        }
        if (lane == 0) warp_corr[warp] = warp_sum;
        __syncthreads();
        
        // Cross-warp reduction
        float corr = 0.0f;
        if (warp == 0 && lane < num_warps) corr = warp_corr[lane];
        #pragma unroll
        for (int offset = 2; offset > 0; offset >>= 1) {
            if (lane < num_warps) {
                float val = __shfl_down_sync(0xFFFFFFFF, corr, offset);
                corr += val;
            }
        }
        corr = __shfl_sync(0xFFFFFFFF, corr, 0);
        
        // === Apply correction retroactively ===
        // dQ[qs,tid] = dq_pdp_acc - corr * dq_p_acc  (then multiply by inv_scale)
        float dq_final = (dq_pdp_acc - corr * dq_p_acc) * inv_scale;
        dQ[qs_base + tid] = __float2bfloat16(dq_final);
        
        // Subtract correction from dK: dK[ks,tid] -= corr * P[ks] * Q[qs,tid]
        // Need to re-stream through KV tiles
        for (int kti = 0; kti < num_kv_tiles; ++kti) {
            int ks_start = kti * KV_TILE;
            int ks_end = min(ks_start + KV_TILE, S);
            int ks_valid = ks_end - ks_start;
            
            for (int ki = tid; ki < d * ks_valid; ki += d) {
                int f = ki / ks_valid;
                int ki_local = ki % ks_valid;
                int ks_global = ks_start + ki_local;
                int64_t idx = bh * stride_bh + ks_global * stride_seq + f;
                K_smem[f * KV_TILE + ki_local] = static_cast<float>(__bfloat162float(K[idx]));
            }
            __syncthreads();
            
            float q_tid = Q_smem[tid];
            for (int ki_local = 0; ki_local < ks_valid; ++ki_local) {
                int ks_global = ks_start + ki_local;
                float dot_s = 0.0f;
                #pragma unroll 8
                for (int f = 0; f < d; ++f) {
                    dot_s += Q_smem[f] * K_smem[f * KV_TILE + ki_local];
                }
                float p = expf(dot_s * inv_scale - lse);
                
                // Subtract correction: dK -= corr * P * Q
                int64_t dk_idx = bh * stride_bh + ks_global * stride_seq + tid;
                atomicAdd(&dK[dk_idx], __float2bfloat16(-corr * p * q_tid));
            }
            __syncthreads();
        }
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

    // Zero-initialize dK and dV
    int64_t total_elements = B * H * S * d;
    size_t bytes = static_cast<size_t>(total_elements) * sizeof(__nv_bfloat16);
    CUDA_CHECK(cudaMemsetAsync(dK_ptr, 0, bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(dV_ptr, 0, bytes, stream));

    int block_size = d_int;   // 128 threads per block
    int grid_size  = BH;      // B*H blocks

    // Shared memory:
    // Q_smem[d] + K_smem[d*32] + V_smem[d*32] + dO_smem[d] + warp_corr[num_warps]
    int smem_bytes = (d_int + d_int * 32 + d_int * 32 + d_int + 4) * sizeof(float);

    mha_bwd_kernel<<<grid_size, block_size, smem_bytes, stream>>>(
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
        dQ_ptr, dK_ptr, dV_ptr,
        BH, S_int, d_int, inv_scale);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace mha_bwd_opt

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_opt::run);