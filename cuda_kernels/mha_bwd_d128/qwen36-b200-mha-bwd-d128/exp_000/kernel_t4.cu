#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <algorithm>
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

namespace mha_bwd_impl {

// Flash Attention style tiled backward kernel
// TQ = query tile size (rows of S loaded at once)
// TK = key-value tile size (columns of S processed at once)
template <int TQ, int TK>
__global__ void mha_backward_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const float* __restrict__ L,
    const __nv_bfloat16* __restrict__ dO,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int S, int d) 
{
    const int bh = blockIdx.x;
    const int tid = threadIdx.x;
    
    const int bi = bh / H;
    const int hi = bh % H;
    
    const float scale = rsqrtf(static_cast<float>(d));
    
    const uint64_t bh_off = static_cast<uint64_t>(bh) * S * d;
    const uint64_t l_off = static_cast<uint64_t>(bi) * H * S + static_cast<uint64_t>(hi) * S;
    
    const __nv_bfloat16* Q_bh = Q + bh_off;
    const __nv_bfloat16* K_bh = K + bh_off;
    const __nv_bfloat16* V_bh = V + bh_off;
    const __nv_bfloat16* dO_bh = dO + bh_off;
    const float* L_bh = L + l_off;
    __nv_bfloat16* dQ_bh = dQ + bh_off;
    __nv_bfloat16* dK_bh = dK + bh_off;
    __nv_bfloat16* dV_bh = dV + bh_off;
    
    // Warp-based indexing: each thread within warp handles specific offsets
    // blockDim.x should be TQ * something or divide evenly
    
    // Shared memory for loading tiles
    // sm_Q: TQ * d floats? No, too large. Instead each thread loads its own d-slice.
    extern __shared__ char smem[];
    
    // Layout in shared memory:
    // [0 : TK*d/2]:    K tile as bf16 (row-major by k, strided by d per thread group)
    // [TK*d/2 : TK*d]: V tile as bf16
    // Actually, with bf16 and d=128, we need careful layout.
    // Let each thread contribute one element at a time rather than staging everything.
    
    // Per-thread register storage for Q and dO across the TQ tile
    // Each thread handles d/blockDim.x elements of the d-dimension
    const int n_threads_per_head = blockDim.x;
    const int d_stride = (d + n_threads_per_head - 1) / n_threads_per_head;
    const int d_start = tid * d_stride;
    const int d_end = min(d_start + d_stride, d);
    const int d_local = d_end - d_start;
    
    // Q and dO values for current query tile (FP32 accumulator format)
    // We need to iterate over d-dimension manually since each thread handles multiple d-elems
    // To simplify: each thread handles exactly one d-element when d == blockDim.x
    // For generality, let's handle any d <= blockDim.x
    
    // ---- Phase A: Process dQ ----
    // For each query tile, compute dQ[q] by iterating over all KV tiles
    // dQ[q][dd] = sum_k P[q][k]*(dS[q][k]-corr[q])*K[k][dd]
    //            = dq_a[q][dd] - corr[q]*dq_b[q][dd]
    // where dq_a[q][dd] = sum_k P*q*k * dS*q*k * K[k][dd]
    //       dq_b[q][dd] = sum_k P[q][k] * K[k][dd]
    //       corr[q]     = sum_k P*q*k * dS[q][k]
    // But these sums are over d as well! So each thread contributes partials.
    
    // For simplicity, assume d == blockDim.x so each thread owns exactly one d-element
    const int di = tid;
    
    // Registers for Q and dO values across query tile
    float Q_r[TQ];
    float dO_r[TQ];
    float L_r[TQ];
    
    // Iterators for q index within thread's range
    #pragma unroll
    for (int qi = d_start; qi < d_end; ++qi) {
        int di_iter = qi;
        
        #pragma unroll
        for (int tt = 0; tt < TQ; ++tt) {
            int qq = blockIdx.y * TQ * n_threads_per_head + tt * n_threads_per_head + tid;
            if (qq < S) {
                Q_r[tt] = __bfloat162float(Q_bh[qq * d + di_iter]);
                dO_r[tt] = __bfloat162float(dO_bh[qq * d + di_iter]);
                L_r[tt] = L_bh[qq];
            } else {
                Q_r[tt] = 0.f;
                dO_r[tt] = 0.f;
                L_r[tt] = 0.f;
            }
        }
        
        // Local accumulators for this thread's d-element
        float corr[TQ];
        float dq_a[TQ];
        float dq_b[TQ];
        #pragma unroll
        for (int tt = 0; tt < TQ; ++tt) {
            corr[tt] = 0.f;
            dq_a[tt] = 0.f;
            dq_b[tt] = 0.f;
        }
        
        // Iterate over KV tiles
        const int q_base = blockIdx.y * TQ * n_threads_per_head + tid;
        
        for (int kt = 0; kt < S; kt += TK) {
            const int k_end = min(kt + TK, S);
            const int tk = k_end - kt;
            
            // Load K and V for this KV tile into registers (each thread = one d-element)
            float K_r[TK];
            float V_r[TK];
            #pragma unroll
            for (int vv = 0; vv < tk; ++vv) {
                const int kk = kt + vv;
                K_r[vv] = __bfloat162float(K_bh[kk * d + di_iter]);
                V_r[vv] = __bfloat162float(V_bh[kk * d + di_iter]);
            }
            
            // Compute contributions
            #pragma unroll
            for (int tt = 0; tt < TQ; ++tt) {
                const int qq = q_base + tt * n_threads_per_head;
                if (qq >= S) continue;
                
                const float qv = Q_r[tt];
                const float dov = dO_r[tt];
                const float lv = L_r[tt];
                
                float c_add = 0.f;
                float da_add = 0.f;
                float db_add = 0.f;
                
                #pragma unroll
                for (int vv = 0; vv < tk; ++vv) {
                    const float kv = K_r[vv];
                    const float vv_v = V_r[vv];
                    const float score = qv * kv * scale;
                    const float p_val = expf(score - lv);
                    const float ds = dov * vv_v;
                    
                    c_add += p_val * ds;
                    da_add += p_val * ds * kv;
                    db_add += p_val * kv;
                }
                corr[tt] += c_add;
                dq_a[tt] += da_add;
                dq_b[tt] += db_add;
            }
        }
        
        // Write dQ
        #pragma unroll
        for (int tt = 0; tt < TQ; ++tt) {
            const int qq = q_base + tt * n_threads_per_head;
            if (qq < S) {
                const float dq = dq_a[tt] - corr[tt] * dq_b[tt];
                dQ_bh[qq * d + di_iter] = __float2bfloat16(dq);
            }
        }
    }
    
    // ---- Phase B: Process dV ----
    // dV[k][di] = sum_q P[q][k] * dO[q][di]
    // Straightforward: iterate over q tiles, accumulate into dV
    
    // Use shared memory to reduce per-k values across threads in the block
    float* dV_smem = reinterpret_cast<float*>(smem);
    
    #pragma unroll
    for (int ki = d_start; ki < d_end; ++ki) {
        int di_iter = ki;
        
        // Zero shared memory
        for (int k = tid; k < S; k += n_threads_per_head) {
            dV_smem[k] = 0.f;
        }
        __syncthreads();
        
        // Process Q tiles
        for (int qb = 0; qb < (S + TQ * n_threads_per_head - 1) / (TQ * n_threads_per_head); qb++) {
            #pragma unroll
            for (int tt = 0; tt < TQ; ++tt) {
                const int qq = qb * TQ * n_threads_per_head + tt * n_threads_per_head + tid;
                if (qq >= S) continue;
                
                const float qv = __bfloat162float(Q_bh[qq * d + di_iter]);
                const float dov = __bfloat162float(dO_bh[qq * d + di_iter]);
                const float lv = L_bh[qq];
                
                // Iterate over all KV tiles to accumulate dV
                for (int kt = 0; kt < S; kt += TK) {
                    const int k_end = min(kt + TK, S);
                    const int tk = k_end - kt;
                    
                    float dv_acc[TK];
                    #pragma unroll
                    for (int vv = 0; vv < tk; ++vv) dv_acc[vv] = 0.f;
                    
                    #pragma unroll
                    for (int vv = 0; vv < tk; ++vv) {
                        const int kk = kt + vv;
                        const float kv = __bfloat162float(K_bh[kk * d + di_iter]);
                        const float score = qv * kv * scale;
                        const float p_val = expf(score - lv);
                        dv_acc[vv] += p_val * dov;
                    }
                    
                    #pragma unroll
                    for (int vv = 0; vv < tk; ++vv) {
                        const int kk = kt + vv;
                        atomicAdd(&dV_smem[kk], dv_acc[vv]);
                    }
                }
            }
            __syncthreads();
        }
        
        // Reduce shared memory across threads (sum over all d-stride contributions)
        // Each block's threads collectively computed contributions to dV for d_start..d_end
        // We need to add our contribution to global memory
        
        // Since we're using the block for one (batch,head), and each thread handles different d-range,
        // we just write our accumulated shared memory to global output at the right d-offset
        
        for (int k = tid; k < S; k += n_threads_per_head) {
            dV_bh[k * d + di_iter] = __float2bfloat16(dV_smem[k]);
        }
        __syncthreads();
    }
    
    // ---- Phase C: Process dK ----
    // dK[k][di] = sum_q P[q][k]*(dS[q][k]-corr[q])*Q[q][di]
    // Need corr[q] which requires summing over all k... this needs two passes or caching.
    // Use shared memory to cache corr[q] for all q.
    
    float* dK_smem = reinterpret_cast<float*>(smem) + S;
    float* corr_smem = reinterpret_cast<float*>(smem) + 2 * S;
    
    #pragma unroll
    for (int ki = d_start; ki < d_end; ++ki) {
        int di_iter = ki;
        
        // Zero shared memory
        for (int idx = tid; idx < 2 * S; idx += n_threads_per_head) {
            dK_smem[idx - S] = 0.f;
            corr_smem[idx] = 0.f;
        }
        __syncthreads();
        
        // Pass 1: Compute corr[q] for all q
        for (int qb = 0; qb < (S + TQ * n_threads_per_head - 1) / (TQ * n_threads_per_head); qb++) {
            #pragma unroll
            for (int tt = 0; tt < TQ; ++tt) {
                const int qq = qb * TQ * n_threads_per_head + tt * n_threads_per_head + tid;
                if (qq >= S) continue;
                
                float c = 0.f;
                const float qv = __bfloat162float(Q_bh[qq * d + di_iter]);
                const float dov = __bfloat162float(dO_bh[qq * d + di_iter]);
                const float lv = L_bh[qq];
                
                for (int kt = 0; kt < S; kt += TK) {
                    const int k_end = min(kt + TK, S);
                    const int tk = k_end - kt;
                    for (int vv = 0; vv < tk; ++vv) {
                        const int kk = kt + vv;
                        const float kv = __bfloat162float(K_bh[kk * d + di_iter]);
                        const float vval = __bfloat162float(V_bh[kk * d + di_iter]);
                        const float score = qv * kv * scale;
                        const float p_val = expf(score - lv);
                        c += p_val * dov * vval;
                    }
                }
                atomicAdd(&corr_smem[qq], c);
            }
            __syncthreads();
        }
        
        // Pass 2: Compute dK using cached corr
        for (int qb = 0; qb < (S + TQ * n_threads_per_head - 1) / (TQ * n_threads_per_head); qb++) {
            #pragma unroll
            for (int tt = 0; tt < TQ; ++tt) {
                const int qq = qb * TQ * n_threads_per_head + tt * n_threads_per_head + tid;
                if (qq >= S) continue;
                
                const float qv = __bfloat162float(Q_bh[qq * d + di_iter]);
                const float dov = __bfloat162float(dO_bh[qq * d + di_iter]);
                const float lv = L_bh[qq];
                const float cc = corr_smem[qq];
                
                for (int kt = 0; kt < S; kt += TK) {
                    const int k_end = min(kt + TK, S);
                    const int tk = k_end - kt;
                    float dk_acc[TK];
                    #pragma unroll
                    for (int vv = 0; vv < tk; ++vv) dk_acc[vv] = 0.f;
                    
                    for (int vv = 0; vv < tk; ++vv) {
                        const int kk = kt + vv;
                        const float kv = __bfloat162float(K_bh[kk * d + di_iter]);
                        const float vval = __bfloat162float(V_bh[kk * d + di_iter]);
                        const float score = qv * kv * scale;
                        const float p_val = expf(score - lv);
                        const float dp = p_val * (dov * vval - cc);
                        dk_acc[vv] += dp * qv;
                    }
                    
                    #pragma unroll
                    for (int vv = 0; vv < tk; ++vv) {
                        const int kk = kt + vv;
                        atomicAdd(&dK_smem[kk], dk_acc[vv]);
                    }
                }
            }
            __syncthreads();
        }
        
        // Write dK to global memory
        for (int k = tid; k < S; k += n_threads_per_head) {
            dK_bh[k * d + di_iter] = __float2bfloat16(dK_smem[k]);
        }
        __syncthreads();
    }
}

}  // namespace mha_bwd_impl

extern "C" {

void run(tvm::ffi::TensorView Q,
         tvm::ffi::TensorView K,
         tvm::ffi::TensorView V,
         tvm::ffi::TensorView O,
         tvm::ffi::TensorView dO,
         tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ,
         tvm::ffi::TensorView dK,
         tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    const int64_t B = Q.size(0);
    const int64_t H = Q.size(1);
    const int64_t S = Q.size(2);
    const int64_t d = Q.size(3);
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    // Grid: x=batch-head, y=query-tile-blocks
    // Block: d threads
    constexpr int TQ = 8;
    constexpr int TK = 32;
    const int dthreads = static_cast<int>(d);
    const int n_query_tile_blocks = (static_cast<int>(S) + TQ * dthreads - 1) / (TQ * dthreads);
    
    dim3 grid(static_cast<int>(B * H), n_query_tile_blocks);
    dim3 block(dthreads);
    
    // Shared memory: 3 arrays of S floats
    const int shmem_bytes = 3 * static_cast<int>(S) * sizeof(float);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    mha_bwd_impl::mha_backward_kernel<TQ, TK><<<grid, block, shmem_bytes, stream>>>(
        Q_ptr, K_ptr, V_ptr, L_ptr, dO_ptr,
        dQ_ptr, dK_ptr, dV_ptr,
        static_cast<int>(B), static_cast<int>(H),
        static_cast<int>(S), static_cast<int>(d));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}  // extern "C"