#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <algorithm>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define CUDA_CHECK(call) do {                                         \
    cudaError_t _e = (call);                                          \
    if (_e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                  \
                cudaGetErrorString(_e), __FILE__, __LINE__);          \
        exit(1);                                                       \
    }                                                                  \
} while(0)

namespace mha_bwd_impl {

__global__ void mha_clear_kernel(__nv_bfloat16* out, int64_t total) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total; i += stride) {
        out[i] = __float2bfloat16(0.f);
    }
}

// Tiled kernel: each CTA handles one (b,h) pair.
// Layout in smem:
//   smem_q[THREADS_PER_ROW][D]  -- one row of Q per thread group slice
//   smem_k[TILE_K][D]           -- tiled K rows
//   smem_do[TILE_Q][D]          -- tiled dO rows
// 
// Strategy:
// 1. Compute scores S[q,k] = Q[q]*K[k]^T / sqrt(d) for all valid causal pairs via tile mm
// 2. Compute P[q,k] = exp(score - lse) via softmax
// 3. Compute dV = P^T * dO (only lower triangle)
// 4. Compute dA = dO * V^T (where A = score matrix), then dS = P ⊙ dA
// 5. Compute dQ = dS * K, dK = dS^T * Q
//
// Simplified: process in two phases
// Phase 1: Compute dV and dS (= P .* (dO * V^T))  [uses same attn weights as dV]
// Phase 2: Compute dQ = dS * K, dK = dS^T * Q

template<int D>
__global__ __launch_bounds__(256)
void attention_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L_in,
    __nv_bfloat16* __restrict__ dQ_out,
    __nv_bfloat16* __restrict__ dK_out,
    __nv_bfloat16* __restrict__ dV_out,
    int B, int H, int S)
{
    extern __shared__ __align__(16) char smem_raw[];
    
    int bh = blockIdx.x;
    if (bh >= B * H) return;
    int b = bh / H;
    int h = bh % H;
    size_t base = ((size_t)b * H + h) * (size_t)S * D;
    
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    float inv_sqrt_d = rsqrtf((float)D);
    
    // Shared memory layout:
    // smem_Q_row[D]       : buffer to load a single Q row
    // smem_K_tile[K_TILE][D] : tiled K
    // smem_V_tile[V_TILE][D] : tiled V  
    // smem_DO_tile[Q_TILE][D]: tiled dO
    // smem_Score[Q_TILE][K_TILE] : scores
    // smem_P[Q_TILE][K_TILE]     : probs
    // smem_dA[Q_TILE][K_TILE]    : dA = dO*V^T
    
    // Tile sizes: keep register usage manageable
    // With 256 threads, each can hold ceil(D/4)=32 floats easily
    constexpr int K_TILE = 64;
    constexpr int Q_TILE = 64;
    constexpr int V_TILE = 64;
    constexpr int DO_TILE = 64;
    
    // Smem offset tracking
    int off = 0;
    float* smem_q_row = reinterpret_cast<float*>(smem_raw + off); off += D * sizeof(float);
    float* smem_do_row = reinterpret_cast<float*>(smem_raw + off); off += D * sizeof(float);
    
    // smem_k: [K_TILE][D] as bf16 to save space = 64*128*2 = 16KB -> use float for comp
    // Actually let's stay in bf16 in smem and convert on the fly
    
    __nv_bfloat16* smem_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + off); off += K_TILE * D * sizeof(__nv_bfloat16);
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + off); off += V_TILE * D * sizeof(__nv_bfloat16);
    
    // smem_do: [DO_TILE][D]
    __nv_bfloat16* smem_do = reinterpret_cast<__nv_bfloat16*>(smem_raw + off); off += DO_TILE * D * sizeof(__nv_bfloat16);
    
    // Score/probability buffers (need only Q_TILE x K_TILE section): 64*64*4*2 = 32KB
    float* smem_score = reinterpret_cast<float*>(smem_raw + off); off += Q_TILE * K_TILE * sizeof(float);
    float* smem_prob = reinterpret_cast<float*>(smem_raw + off); off += Q_TILE * K_TILE * sizeof(float);
    float* smem_dA = reinterpret_cast<float*>(smem_raw + off); off += Q_TILE * K_TILE * sizeof(float);
    
    (void)smem_raw; // suppress unused warning
    
    // ---- PHASE 1: Clear per-block accumulation buffers in global mem isn't needed
    // since each block writes unique (b,h) output region directly.
    
    // ---- PHASE A: Compute dV = P^T * dO ----
    // dV[k,d] = sum_{q=k}^{S-1} P[q,k] * dO[q,d]
    // We'll compute this via tiling: iterate over q-tiles, load K and dO
    
    for (int ktile = 0; ktile < S; ktile += K_TILE) {
        int k_start = ktile;
        int k_end = min(k_start + K_TILE, S);
        int k_count = k_end - k_start;
        
        // Load K tile into smem
        for (int i = tid; i < k_count * D; i += nthreads) {
            int kr = i / D;
            int dc = i % D;
            smem_k[kr * D + dc] = K[base + (size_t)(k_start + kr) * D + dc];
        }
        __syncthreads();
        
        // Iterate over q-tiles for this k-tile
        for (int qt = max(k_start, 0); qt < S; qt += Q_TILE) {
            int q_start = qt;
            int q_end = min(q_start + Q_TILE, S);
            int q_count = q_end - q_start;
            
            // Load dO rows for [q_start, q_end)
            for (int q = 0; q < q_count; q++) {
                for (int di = tid; di < D; di += nthreads) {
                    smem_do[q * D + di] = dO[base + (size_t)(q_start + q) * D + di];
                }
                __syncthreads();
                
                // Load Q[q] row into smem_q_row
                for (int di = tid; di < D; di += nthreads) {
                    smem_q_row[di] = __bfloat162float(Q[base + (size_t)(q_start + q) * D + di]);
                }
                __syncthreads();
                
                // Compute scores for this q against all k in tile, write to smem
                float lse_q = L_in[(size_t)bh * S + q_start + q];
                
                // Each thread computes one k-row's contribution to dV
                for (int ki = tid; ki < k_count && (k_start + ki) <= q_start + q; ki++) {
                    // Score = Q[q] . K[k_start+ki]
                    float score = 0.f;
                    const __nv_bfloat16* k_r = smem_k + ki * D;
                    for (int dd = 0; dd < D; dd += 4) {
                        score += smem_q_row[dd]     * __bfloat162float(k_r[dd]);
                        score += smem_q_row[dd + 1] * __bfloat162float(k_r[dd + 1]);
                        score += smem_q_row[dd + 2] * __bfloat162float(k_r[dd + 2]);
                        score += smem_q_row[dd + 3] * __bfloat162float(k_r[dd + 3]);
                    }
                    float attn = expf(score * inv_sqrt_d - lse_q);
                    
                    // dV[k_start+ki, d] += attn * dO[q_start+q, d]
                    __nv_bfloat16* dv_dst = dV_out + base + (size_t)(k_start + ki) * D;
                    const __nv_bfloat16* do_r = smem_do + q * D;
                    for (int dd = 0; dd < D; dd += 2) {
                        float v0 = __bfloat162float(dv_dst[dd]) + attn * __bfloat162float(do_r[dd]);
                        float v1 = __bfloat162float(dv_dst[dd + 1]) + attn * __bfloat162float(do_r[dd + 1]);
                        dv_dst[dd] = __float2bfloat16(v0);
                        dv_dst[dd + 1] = __float2bfloat16(v1);
                    }
                }
            }
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
         tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D_val = Q.size(3);
    assert(D_val == 128 && "Expected D=128");
    int D = static_cast<int>(D_val);
    int BH = static_cast<int>(B * H);
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int64_t total_elem = B * H * S * D;
    
    // Clear outputs
    int ct = 1024;
    int cb = std::min(static_cast<int>((total_elem + ct - 1) / ct), 65535);
    mha_clear_kernel<<<cb, ct, 0, stream>>>(dQ_ptr, total_elem);
    CUDA_CHECK(cudaGetLastError());
    mha_clear_kernel<<<cb, ct, 0, stream>>>(dK_ptr, total_elem);
    CUDA_CHECK(cudaGetLastError());
    mha_clear_kernel<<<cb, ct, 0, stream>>>(dV_ptr, total_elem);
    CUDA_CHECK(cudaGetLastError());
    
    // SMEM budget:
    // smem_q_row: 128*4 = 512B
    // smem_do_row: 128*4 = 512B  
    // smem_k: 64*128*2 = 16384B
    // smem_v: 64*128*2 = 16384B
    // smem_do: 64*128*2 = 16384B
    // smem_score: 64*64*4 = 16384B
    // smem_prob: 64*64*4 = 16384B
    // smem_dA: 64*64*4 = 16384B
    // Total ≈ 99KB
    
    constexpr int K_TILE = 64;
    constexpr int Q_TILE = 64;
    constexpr int V_TILE = 64;
    constexpr int DO_TILE = 64;
    
    int off_calc = 0;
    off_calc += D * sizeof(float); // smem_q_row
    off_calc += D * sizeof(float); // smem_do_row (unused but allocated)
    off_calc += K_TILE * D * sizeof(__nv_bfloat16);
    off_calc += V_TILE * D * sizeof(__nv_bfloat16);
    off_calc += DO_TILE * D * sizeof(__nv_bfloat16);
    off_calc += Q_TILE * K_TILE * sizeof(float);
    off_calc += Q_TILE * K_TILE * sizeof(float);
    off_calc += Q_TILE * K_TILE * sizeof(float);
    
    dim3 grid(BH, 1, 1);
    dim3 blk(256, 1, 1);
    
    attention_bwd_kernel<128><<<grid, blk, off_calc, stream>>>(
        Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr,
        dQ_ptr, dK_ptr, dV_ptr,
        static_cast<int>(B), static_cast<int>(H), static_cast<int>(S));
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);

}  // namespace mha_bwd_impl