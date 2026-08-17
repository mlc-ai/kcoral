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
constexpr int BLOCK_THREADS = NUM_WARPS * WARP_SIZE;

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

// Warp-level parallel reduce sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
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
    static_assert(HEAD_DIM == 128, "Only D=128 supported");
    
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    
    int bh_idx = blockIdx.x;
    int q_tile_idx = blockIdx.y;
    
    if (bh_idx >= B * H || q_tile_idx * TILE_M >= S) return;
    
    int b = bh_idx / H;
    int h = bh_idx % H;
    
    const __nv_bfloat16* Q_bh = Q + (uint64_t)(b * H + h) * S * HEAD_DIM;
    const __nv_bfloat16* K_bh = K + (uint64_t)(b * H + h) * S * HEAD_DIM;
    const __nv_bfloat16* V_bh = V + (uint64_t)(b * H + h) * S * HEAD_DIM;
    const __nv_bfloat16* dO_bh = dO + (uint64_t)(b * H + h) * S * HEAD_DIM;
    const float* L_bh = L + (b * H + h) * S;
    
    __nv_bfloat16* dQ_bh = dQ + (uint64_t)(b * H + h) * S * HEAD_DIM;
    __nv_bfloat16* dK_bh = dK + (uint64_t)(b * H + h) * S * HEAD_DIM;
    __nv_bfloat16* dV_bh = dV + (uint64_t)(b * H + h) * S * HEAD_DIM;
    
    int qm = q_tile_idx * TILE_M;
    
    // Shared memory layout:
    // sQ: TILE_M x HEAD_DIM bf16 = 32 * 128 = 4096 bf16 = 8KB
    // sK: TILE_N x HEAD_DIM bf16 = 32 * 128 = 4096 bf16 = 8KB  
    // sV: TILE_N x HEAD_DIM bf16 = 32 * 128 = 4096 bf16 = 8KB
    // smem_dK_sums: TILE_N x (HEAD_DIM/WARP_SIZE) fp32 for partial sums = 32 * 4 = 128 fp32 = 512B
    extern __shared__ char smem[];
    
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sK = sQ + TILE_M * HEAD_DIM;
    __nv_bfloat16* sV = sK + TILE_N * HEAD_DIM;
    float* smem_dK_reduce = reinterpret_cast<float*>(sV + TILE_N * HEAD_DIM);
    float* smem_dV_reduce = smem_dK_reduce + TILE_N * (HEAD_DIM / WARP_SIZE);
    
    int my_q = qm + warp_id;
    if (my_q >= S) return;
    
    float my_lse = L_bh[my_q];
    
    // Load dO[my_q, :] into registers
    float dO_reg[HEAD_DIM];
    for (int di = 0; di < HEAD_DIM; di++) {
        dO_reg[di] = bf162f(dO_bh[(uint64_t)my_q * HEAD_DIM + di]);
    }
    
    // dQ accumulator for this query row
    float dQ_acc[HEAD_DIM] = {};
    
    // Per-thread accumulators for dK and dV contributions (local before reduce)
    // We accumulate per-n-position within each n-tile
    
    int num_n_tiles = (S + TILE_N - 1) / TILE_N;
    for (int nt = 0; nt < num_n_tiles; nt++) {
        int n_start = nt * TILE_N;
        
        // Load K tile cooperatively
        for (int ni = 0; ni < TILE_N; ni++) {
            int nk = n_start + ni;
            for (int di = lane_id; di < HEAD_DIM; di += WARP_SIZE) {
                if (nk < S) {
                    sK[ni * HEAD_DIM + di] = K_bh[(uint64_t)nk * HEAD_DIM + di];
                } else {
                    sK[ni * HEAD_DIM + di] = f2bf16(0.0f);
                }
            }
        }
        __syncthreads();
        
        // Load V tile cooperatively
        for (int ni = 0; ni < TILE_N; ni++) {
            int nk = n_start + ni;
            for (int di = lane_id; di < HEAD_DIM; di += WARP_SIZE) {
                if (nk < S) {
                    sV[ni * HEAD_DIM + di] = V_bh[(uint64_t)nk * HEAD_DIM + di];
                } else {
                    sV[ni * HEAD_DIM + di] = f2bf16(0.0f);
                }
            }
        }
        __syncthreads();
        
        // Load Q tile cooperatively
        for (int mi = 0; mi < TILE_M; mi++) {
            int mk = qm + mi;
            for (int di = lane_id; di < HEAD_DIM; di += WARP_SIZE) {
                if (mk < S) {
                    sQ[mi * HEAD_DIM + di] = Q_bh[(uint64_t)mk * HEAD_DIM + di];
                } else {
                    sQ[mi * HEAD_DIM + di] = f2bf16(0.0f);
                }
            }
        }
        __syncthreads();
        
        // Compute probs for this query row vs K tile
        float probs[TILE_N];
        for (int ni = 0; ni < TILE_N; ni++) {
            int nk = n_start + ni;
            float score = 0.0f;
            #pragma unroll
            for (int di = 0; di < HEAD_DIM; di++) {
                score += bf162f(sQ[warp_id * HEAD_DIM + di]) * bf162f(sK[ni * HEAD_DIM + di]);
            }
            score *= INV_SQRT_D;
            probs[ni] = (nk <= my_q) ? fast_exp2((score - my_lse) * LOG2E) : 0.0f;
        }
        
        // Compute dP_scalar[ni] = sum_d dO[q,d] * V[nk,d]
        float dP_scalar[TILE_N];
        float dP_dot_sum = 0.0f;
        
        for (int ni = 0; ni < TILE_N; ni++) {
            float dps = 0.0f;
            #pragma unroll
            for (int di = 0; di < HEAD_DIM; di++) {
                dps += dO_reg[di] * bf162f(sV[ni * HEAD_DIM + di]);
            }
            dP_scalar[ni] = dps;
            if (probs[ni] > 0.0f) {
                dP_dot_sum += probs[ni] * dps;
            }
        }
        
        // dS[q,ni] = P[q,ni] * (dP_scalar[ni] - dP_dot_sum)
        // dQ[q,d] += sum_ni dS[q,ni] * K[ni,d]
        for (int di = 0; di < HEAD_DIM; di++) {
            float dq_sum = 0.0f;
            #pragma unroll
            for (int ni = 0; ni < TILE_N; ni++) {
                int nk = n_start + ni;
                if (nk <= my_q && probs[ni] > 0.0f) {
                    float ds = probs[ni] * (dP_scalar[ni] - dP_dot_sum);
                    dq_sum += ds * bf162f(sK[ni * HEAD_DIM + di]);
                }
            }
            dQ_acc[di] += dq_sum;
        }
        
        // Accumulate partial dK and dV contributions per thread
        // Each thread handles 4 d-dimensions (lane_id, lane_id+32, lane_id+64, lane_id+96)
        // For each n-position, compute dK[nk,d] contribution and dV[nk,d] contribution
        
        // Local per-n accumulators: sum over warp members will use shared memory
        // For dK: contrib = ds * Q[my_q, d]
        // For dV: contrib = prob * dO[my_q, d]
        
        // We need to reduce across all 4 warps (NUM_WARPS) for each (n_pos, d_sub)
        // Strategy: each thread computes its contribution, warp-reduce, then 
        // cooperative-group-add to shared memory per (ni, di_group)
        
        // Per-n local accum for this warp's contribution
        float dk_local[TILE_N][HEAD_DIM / WARP_SIZE] = {};
        float dv_local[TILE_N][HEAD_DIM / WARP_SIZE] = {};
        
        #pragma unroll
        for (int ni = 0; ni < TILE_N; ni++) {
            int nk = n_start + ni;
            if (nk <= my_q && probs[ni] > 0.0f) {
                float pval = probs[ni];
                float ds = pval * (dP_scalar[ni] - dP_dot_sum);
                
                #pragma unroll
                for (int dg = 0; dg < HEAD_DIM / WARP_SIZE; dg++) {
                    int di = dg * WARP_SIZE + lane_id;
                    dk_local[ni][dg] = ds * bf162f(sQ[warp_id * HEAD_DIM + di]);
                    dv_local[ni][dg] = pval * dO_reg[di];
                }
            }
        }
        
        // Warp-level reduction for each (ni, dg)
        #pragma unroll
        for (int ni = 0; ni < TILE_N; ni++) {
            #pragma unroll
            for (int dg = 0; dg < HEAD_DIM / WARP_SIZE; dg++) {
                dk_local[ni][dg] = warp_reduce_sum(dk_local[ni][dg]);
                dv_local[ni][dg] = warp_reduce_sum(dv_local[ni][dg]);
            }
        }
        
        // Write warp-reduced results to shared memory (only lane 0 of each warp)
        // Use a simple barrier then additive accumulation
        if (lane_id == 0) {
            #pragma unroll
            for (int ni = 0; ni < TILE_N; ni++) {
                #pragma unroll
                for (int dg = 0; dg < HEAD_DIM / WARP_SIZE; dg++) {
                    smem_dK_reduce[ni * (HEAD_DIM / WARP_SIZE) + dg] = dk_local[ni][dg];
                    smem_dV_reduce[ni * (HEAD_DIM / WARP_SIZE) + dg] = dv_local[ni][dg];
                }
            }
        }
        __syncthreads();
        
        // Now all warps read from shared memory and add up across warps
        // This requires accumulating NUM_WARPS times. We'll use a sequential approach.
        // First sync after writing, then another pass reads and sums.
        
        // Each thread adds its warp's result; subsequent warps will add on top
        // Actually we need all warps' contributions summed. Simple approach:
        // After __syncthreads(), every thread reads all NUM_WARPS warp results
        
        #pragma unroll
        for (int ni = 0; ni < TILE_N; ni++) {
            #pragma unroll
            for (int dg = 0; dg < HEAD_DIM / WARP_SIZE; dg++) {
                int di = dg * WARP_SIZE + lane_id;
                float dk_total = 0.0f, dv_total = 0.0f;
                #pragma unroll
                for (int w = 0; w < NUM_WARPS; w++) {
                    // smem was overwritten by last warp to arrive, so this won't work correctly
                    // Need a different strategy
                }
            }
        }
        
        // The above approach has issues - let me rewrite with a cleaner strategy
        // where we use shared memory properly with atomicAdds or a two-phase approach
        __syncthreads();
        
        // Cleaner approach: use atomicAdd in shared memory for cross-warp reduction
        // Reset shared mem first
        if (tid == 0) {
            for (int i = 0; i < TILE_N * (HEAD_DIM / WARP_SIZE); i++) {
                smem_dK_reduce[i] = 0.0f;
                smem_dV_reduce[i] = 0.0f;
            }
        }
        __syncthreads();
        
        // Each thread atomically adds its contribution
        #pragma unroll
        for (int ni = 0; ni < TILE_N; ni++) {
            int nk = n_start + ni;
            if (nk <= my_q && probs[ni] > 0.0f) {
                float pval = probs[ni];
                float ds = pval * (dP_scalar[ni] - dP_dot_sum);
                
                #pragma unroll
                for (int dg = 0; dg < HEAD_DIM / WARP_SIZE; dg++) {
                    int di = dg * WARP_SIZE + lane_id;
                    float dk_val = ds * bf162f(sQ[warp_id * HEAD_DIM + di]);
                    float dv_val = pval * dO_reg[di];
                    
                    atomicAdd(&smem_dK_reduce[ni * (HEAD_DIM / WARP_SIZE) + dg], dk_val);
                    atomicAdd(&smem_dV_reduce[ni * (HEAD_DIM / WARP_SIZE) + dg], dv_val);
                }
            }
        }
        __syncthreads();
        
        // Now shared memory contains the full reduced dK/dV contributions for this n-tile
        // Each thread writes to global memory (each thread handles 4 d-dims)
        #pragma unroll
        for (int ni = 0; ni < TILE_N; ni++) {
            int nk = n_start + ni;
            if (nk < S) {
                #pragma unroll
                for (int dg = 0; dg < HEAD_DIM / WARP_SIZE; dg++) {
                    int di = dg * WARP_SIZE + lane_id;
                    dK_bh[(uint64_t)nk * HEAD_DIM + di] = f2bf16(smem_dK_reduce[ni * (HEAD_DIM / WARP_SIZE) + dg]);
                    dV_bh[(uint64_t)nk * HEAD_DIM + di] = f2bf16(smem_dV_reduce[ni * (HEAD_DIM / WARP_SIZE) + dg]);
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write dQ
    for (int di = lane_id; di < HEAD_DIM; di += WARP_SIZE) {
        dQ_bh[(uint64_t)my_q * HEAD_DIM + di] = f2bf16(dQ_acc[di]);
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
    
    // Shared memory: Q(K,V tiles in bf16) + reduction buffers in fp32
    size_t smem_bytes = (TILE_M * HEAD_DIM + 2 * TILE_N * HEAD_DIM) * sizeof(__nv_bfloat16)
                      + 2 * TILE_N * (HEAD_DIM / WARP_SIZE) * sizeof(float);
    
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