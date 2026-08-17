#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_bwd_d128_causal {

constexpr int HEAD_DIM = 128;
constexpr int TILE_M = 32;
constexpr int TILE_N = 32;
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = 4;
constexpr int BLOCK_THREADS = NUM_WARPS * WARP_SIZE;  // 128

constexpr float INV_SQRT_D = 0.08838834764831844f;
constexpr float LOG2E = 1.4426950408889634f;

__device__ __forceinline__ float bf162f(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

__device__ __forceinline__ __nv_bfloat16 f2bf16(float val) {
    return __float2bfloat16(val);
}

__device__ __forceinline__ float fast_exp2(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__global__ void mha_bwd_fused_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int S, int D
) {
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    
    int bh_idx = blockIdx.x;
    int q_tile_idx = blockIdx.y;
    
    if (bh_idx >= B * H || q_tile_idx * TILE_M >= S) return;
    
    int b = bh_idx / H;
    int h = bh_idx % H;
    
    // Base pointers for this (b, h)
    const __nv_bfloat16* Q_bh = Q + (uint64_t)(b * H + h) * S * D;
    const __nv_bfloat16* K_bh = K + (uint64_t)(b * H + h) * S * D;
    const __nv_bfloat16* V_bh = V + (uint64_t)(b * H + h) * S * D;
    const __nv_bfloat16* dO_bh = dO + (uint64_t)(b * H + h) * S * D;
    const float* L_bh = L + (b * H + h) * S;
    
    __nv_bfloat16* dQ_bh = dQ + (uint64_t)(b * H + h) * S * D;
    __nv_bfloat16* dK_bh = dK + (uint64_t)(b * H + h) * S * D;
    __nv_bfloat16* dV_bh = dV + (uint64_t)(b * H + h) * S * D;
    
    int qm = q_tile_idx * TILE_M;
    
    // Shared memory layout
    // Q tile: TILE_M x D = 32 x 128 bf16 = 8KB
    // K tile: TILE_N x D = 32 x 128 bf16 = 8KB
    // V tile: TILE_N x D = 32 x 128 bf16 = 8KB
    // Total: 24KB shared memory
    extern __shared__ char smem[];
    
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sK = sQ + TILE_M * D;
    __nv_bfloat16* sV = sK + TILE_N * D;
    
    // Each warp handles one query row in the tile
    int my_q = qm + warp_id;
    if (my_q >= S) return;
    
    float my_lse = L_bh[my_q];
    
    // Load dO[my_q, :] into registers
    float dO_reg[D];
    for (int di = lane_id; di < D; di += WARP_SIZE) {
        dO_reg[di] = bf162f(dO_bh[(uint64_t)my_q * D + di]);
    }
    // Broadcast within warp via shuffle
    #pragma unroll
    for (int sz = 1; sz < WARP_SIZE; sz <<= 1) {
        for (int di = 0; di < D; di++) {
            float val = __shfl_down_sync(0xFFFFFFFF, dO_reg[di], sz);
            if (lane_id < D - sz) {
                // Not quite right for non-power-of-2 D. Let me use a different approach.
            }
        }
    }
    
    // Simplified: each lane loads 4 elements and broadcasts are done implicitly
    // Re-load with proper broadcasting
    for (int di = 0; di < D; di++) {
        dO_reg[di] = bf162f(dO_bh[(uint64_t)my_q * D + di]);
    }
    
    // Initialize dQ accumulator for this query row
    float dQ_acc[D] = {};
    
    // Accumulator for dV (will use atomicAdd)
    float dV_temp[D] = {};
    
    // Loop over K/N tiles
    int num_n_tiles = (S + TILE_N - 1) / TILE_N;
    for (int nt = 0; nt < num_n_tiles; nt++) {
        int n_start = nt * TILE_N;
        if (n_start >= S) break;
        int n_end = min(n_start + TILE_N, S);
        
        // Load K tile into shared memory
        for (int ni = 0; ni < TILE_N; ni++) {
            int nk = n_start + ni;
            for (int di = lane_id; di < D; di += WARP_SIZE) {
                if (nk < S) {
                    sK[ni * D + di] = K_bh[(uint64_t)nk * D + di];
                } else {
                    sK[ni * D + di] = f2bf16(0.0f);
                }
            }
        }
        __syncthreads();
        
        // Load V tile into shared memory  
        for (int ni = 0; ni < TILE_N; ni++) {
            int nk = n_start + ni;
            for (int di = lane_id; di < D; di += WARP_SIZE) {
                if (nk < S) {
                    sV[ni * D + di] = V_bh[(uint64_t)nk * D + di];
                } else {
                    sV[ni * D + di] = f2bf16(0.0f);
                }
            }
        }
        __syncthreads();
        
        // Load Q tile into shared memory (all warps cooperatively)
        for (int mi = 0; mi < TILE_M; mi++) {
            int mk = qm + mi;
            for (int di = lane_id; di < D; di += WARP_SIZE) {
                if (mk < S) {
                    sQ[mi * D + di] = Q_bh[(uint64_t)mk * D + di];
                } else {
                    sQ[mi * D + di] = f2bf16(0.0f);
                }
            }
        }
        __syncthreads();
        
        // Compute attention scores for my_q vs this K tile
        float scores[TILE_N];
        float probs[TILE_N];
        
        for (int ni = 0; ni < TILE_N; ni++) {
            int nk = n_start + ni;
            float score = 0.0f;
            for (int di = 0; di < D; di++) {
                score += bf162f(sQ[warp_id * D + di]) * bf162f(sK[ni * D + di]);
            }
            score *= INV_SQRT_D;
            
            if (nk <= my_q) {
                probs[ni] = fast_exp2((score - my_lse) * LOG2E);
            } else {
                probs[ni] = 0.0f;
            }
        }
        
        // Compute dP_scalar[ni] = sum_d dO[my_q, d] * V[nk, d]
        float dP_scalar[TILE_N];
        float dP_dot_sum = 0.0f;  // sum_k P[q,k] * dP[q,k]
        
        for (int ni = 0; ni < TILE_N; ni++) {
            float dps = 0.0f;
            for (int di = 0; di < D; di++) {
                dps += dO_reg[di] * bf162f(sV[ni * D + di]);
            }
            dP_scalar[ni] = dps;
            if (probs[ni] > 0.0f) {
                dP_dot_sum += probs[ni] * dps;
            }
        }
        
        // dS[q, ni] = P[q, ni] * (dP_scalar[ni] - dP_dot_sum)
        // dQ[q, d] += sum_ni dS[q, ni] * K[ni, d]
        for (int di = 0; di < D; di++) {
            float dq_sum = 0.0f;
            for (int ni = 0; ni < TILE_N; ni++) {
                int nk = n_start + ni;
                if (nk <= my_q && probs[ni] > 0.0f) {
                    float ds = probs[ni] * (dP_scalar[ni] - dP_dot_sum);
                    dq_sum += ds * bf162f(sK[ni * D + di]);
                }
            }
            dQ_acc[di] += dq_sum;
        }
        
        // Accumulate dV: dV[nk, d] += P[my_q, nk] * dO[my_q, d]
        for (int ni = 0; ni < TILE_N; ni++) {
            int nk = n_start + ni;
            if (nk <= my_q && probs[ni] > 0.0f) {
                for (int di = lane_id; di < D; di += WARP_SIZE) {
                    float pval = probs[ni] * dO_reg[di];
                    atomicAdd((float*)((char*)dV_bh + (uint64_t)nk * D * 2 + di * 2), pval);
                }
            }
        }
        
        // Accumulate dK: dK[nk, d] += sum_q_in_tile dS[q, nk] * Q[q, d]
        // Each warp contributes for its query row
        for (int ni = 0; ni < TILE_N; ni++) {
            int nk = n_start + ni;
            if (nk <= my_q && probs[ni] > 0.0f) {
                float ds = probs[ni] * (dP_scalar[ni] - dP_dot_sum);
                for (int di = lane_id; di < D; di += WARP_SIZE) {
                    float contrib = ds * bf162f(sQ[warp_id * D + di]);
                    atomicAdd((float*)((char*)dK_bh + (uint64_t)nk * D * 2 + di * 2), contrib);
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write dQ output
    for (int di = lane_id; di < D; di += WARP_SIZE) {
        dQ_bh[(uint64_t)my_q * D + di] = f2bf16(dQ_acc[di]);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_ptr = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr = static_cast<const float*>(L.data_ptr());
    
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    int num_q_tiles = (int)((S + TILE_M - 1) / TILE_M);
    int total_bh = (int)(B * H);
    
    dim3 grid(total_bh, num_q_tiles, 1);
    dim3 block(BLOCK_THREADS);
    
    size_t smem_bytes = (TILE_M * HEAD_DIM + 2 * TILE_N * HEAD_DIM) * sizeof(__nv_bfloat16);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    mha_bwd_fused_kernel<<<grid, block, smem_bytes, stream>>>(
        Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr,
        dQ_ptr, dK_ptr, dV_ptr,
        (int)B, (int)H, (int)S, (int)D);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128_causal::run);

}  // namespace mha_bwd_d128_causal