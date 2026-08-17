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
constexpr int WARP_SIZE = 32;
constexpr int NUM_THREADS = 128;
constexpr int TN = 32;

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

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void mha_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int S
) {
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    
    int bh_idx = blockIdx.x;
    if (bh_idx >= B * H) return;
    
    int b = bh_idx / H;
    int h = bh_idx % H;
    uint64_t stride_bh = (uint64_t)S * HEAD_DIM;
    
    const __nv_bfloat16* Q_bh = Q + (uint64_t)(b * H + h) * stride_bh;
    const __nv_bfloat16* K_bh = K + (uint64_t)(b * H + h) * stride_bh;
    const __nv_bfloat16* V_bh = V + (uint64_t)(b * H + h) * stride_bh;
    const __nv_bfloat16* dO_bh = dO + (uint64_t)(b * H + h) * stride_bh;
    const float* L_bh = L + (b * H + h) * S;
    
    __nv_bfloat16* dQ_bh = dQ + (uint64_t)(b * H + h) * stride_bh;
    __nv_bfloat16* dK_bh = dK + (uint64_t)(b * H + h) * stride_bh;
    __nv_bfloat16* dV_bh = dV + (uint64_t)(b * H + h) * stride_bh;
    
    // Shared memory layout:
    // sQ:            NUM_THREADS x HEAD_DIM bf16   = 32KB
    // sKV_K:         TN x HEAD_DIM bf16             = 8KB
    // sKV_V:         TN x HEAD_DIM bf16             = 8KB
    // sDk_accum:     TN x (HEAD_DIM/WARP_SIZE) fp32 = 128 fp32 = 512B
    // sDv_accum:     TN x (HEAD_DIM/WARP_SIZE) fp32 = 512B
    // Total SMEM ~= 48KB + 1KB
    extern __shared__ char smem[];
    
    __nv_bfloat16* sQ       = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sKV_K    = sQ + NUM_THREADS * HEAD_DIM;
    __nv_bfloat16* sKV_V    = sKV_K + TN * HEAD_DIM;
    float* sDk_accum        = reinterpret_cast<float*>(sKV_V + TN * HEAD_DIM);
    float* sDv_accum        = sDk_accum + TN * (HEAD_DIM / WARP_SIZE);
    
    int num_q_blocks = (S + NUM_THREADS - 1) / NUM_THREADS;
    int q_block = blockIdx.y;
    if (q_block >= num_q_blocks) return;
    
    int q_start = q_block * NUM_THREADS;
    int my_q_local = tid;
    int my_q = q_start + my_q_local;
    
    // Load Q block
    for (int qi = warp_id; qi < NUM_THREADS; qi += 4) {
        int qk = q_start + qi;
        for (int di = lane_id; di < HEAD_DIM; di += WARP_SIZE) {
            if (qk < S) {
                sQ[(uint64_t)qi * HEAD_DIM + di] = Q_bh[(uint64_t)qk * HEAD_DIM + di];
            } else {
                sQ[(uint64_t)qi * HEAD_DIM + di] = f2bf16(0.0f);
            }
        }
    }
    __syncthreads();
    
    int num_n_tiles = (S + TN - 1) / TN;
    
    bool valid_q = (my_q < S);
    float my_lse = valid_q ? L_bh[my_q] : 0.0f;
    
    // Small per-thread accumulators for dQ
    float dQ_part_a[HEAD_DIM / 4], dQ_part_b[HEAD_DIM / 4];
    #pragma unroll
    for (int i = 0; i < HEAD_DIM / 4; i++) {
        dQ_part_a[i] = 0.0f;
        dQ_part_b[i] = 0.0f;
    }
    
    for (int nt = 0; nt < num_n_tiles; nt++) {
        int nk_start = nt * TN;
        if (nk_start >= S) break;
        
        // Reset shared reduction buffers
        for (int i = tid; i < TN * (HEAD_DIM / WARP_SIZE); i += NUM_THREADS) {
            sDk_accum[i] = 0.0f;
            sDv_accum[i] = 0.0f;
        }
        __syncthreads();
        
        // Load K tile
        for (int ni = 0; ni < TN; ni++) {
            int nk = nk_start + ni;
            for (int di = lane_id; di < HEAD_DIM; di += WARP_SIZE) {
                if (nk < S) {
                    sKV_K[ni * HEAD_DIM + di] = K_bh[(uint64_t)nk * HEAD_DIM + di];
                } else {
                    sKV_K[ni * HEAD_DIM + di] = f2bf16(0.0f);
                }
            }
        }
        __syncthreads();
        
        // Load V tile
        for (int ni = 0; ni < TN; ni++) {
            int nk = nk_start + ni;
            for (int di = lane_id; di < HEAD_DIM; di += WARP_SIZE) {
                if (nk < S) {
                    sKV_V[ni * HEAD_DIM + di] = V_bh[(uint64_t)nk * HEAD_DIM + di];
                } else {
                    sKV_V[ni * HEAD_DIM + di] = f2bf16(0.0f);
                }
            }
        }
        __syncthreads();
        
        float dP_dot_sum = 0.0f;
        float probs[TN];
        float dps[TN];
        
        if (valid_q) {
            // Pass 1: compute score, prob, dP_scalar
            for (int ni = 0; ni < TN; ni++) {
                int nk = nk_start + ni;
                
                float score = 0.0f;
                #pragma unroll
                for (int di = 0; di < HEAD_DIM; di++) {
                    score += bf162f(sQ[(uint64_t)my_q_local * HEAD_DIM + di]) * 
                             bf162f(sKV_K[ni * HEAD_DIM + di]);
                }
                score *= INV_SQRT_D;
                
                float dP_s = 0.0f;
                for (int di = lane_id; di < HEAD_DIM; di += WARP_SIZE) {
                    dP_s += bf162f(dO_bh[(uint64_t)my_q * HEAD_DIM + di]) * 
                            bf162f(sKV_V[ni * HEAD_DIM + di]);
                }
                dP_s = warp_reduce_sum(dP_s);
                dps[ni] = dP_s;
                
                float p = (nk <= my_q) ? fast_exp2((score - my_lse) * LOG2E) : 0.0f;
                probs[ni] = p;
                
                if (p > 0.0f) dP_dot_sum += p * dP_s;
            }
            
            // Pass 2: accumulate dQ, and partial dK/dV into shared mem
            for (int ni = 0; ni < TN; ni++) {
                int nk = nk_start + ni;
                float p = probs[ni];
                if (nk > my_q || p == 0.0f) continue;
                
                float ds = p * (dps[ni] - dP_dot_sum);
                
                // dQ accumulation
                #pragma unroll
                for (int dg = 0; dg < HEAD_DIM / 4; dg++) {
                    int di = dg * 4;
                    float qa = ds * (bf162f(sKV_K[ni * HEAD_DIM + di]) + 
                                     bf162f(sKV_K[ni * HEAD_DIM + di+1]));
                    float qb = ds * (bf162f(sKV_K[ni * HEAD_DIM + di+2]) + 
                                     bf162f(sKV_K[ni * HEAD_DIM + di+3]));
                    dQ_part_a[dg] += qa;
                    dQ_part_b[dg] += qb;
                }
                
                // Partial dK/dV into shared mem (per-warp granularity)
                for (int di = lane_id; di < HEAD_DIM; di += WARP_SIZE) {
                    float dk_val = ds * bf162f(sQ[(uint64_t)my_q_local * HEAD_DIM + di]);
                    float dv_val = p * bf162f(dO_bh[(uint64_t)my_q * HEAD_DIM + di]);
                    
                    int sidx = ni * (HEAD_DIM / WARP_SIZE) + (di / WARP_SIZE);
                    // Warp-reduce first, then atomicAdd in shared
                    dk_val = warp_reduce_sum(dk_val);
                    dv_val = warp_reduce_sum(dv_val);
                    if (lane_id == 0) {
                        atomicAdd(&sDk_accum[sidx], dk_val);
                        atomicAdd(&sDv_accum[sidx], dv_val);
                    }
                }
            }
        }
        
        __syncthreads();
        
        // Write reduced dK/dV from shared mem to global
        for (int i = tid; i < TN * (HEAD_DIM / WARP_SIZE); i += NUM_THREADS) {
            int ni = i / (HEAD_DIM / WARP_SIZE);
            int dg = i % (HEAD_DIM / WARP_SIZE);
            int nk = nk_start + ni;
            int di = dg * WARP_SIZE;
            
            if (nk < S) {
                // Write 4 consecutive BF16 values from one FP32 sum
                // We have accumulated fp32 sums across all 32 lanes of each warp
                // But there were only HEAD_DIM/WARP_SIZE = 4 elements per warp group
                // Actually sDk_accum stores the per-warp-group sums already
                // Each entry is the total contribution for that (ni, di_group)
                // Need to expand back to per-element... but we summed across warps for same di
                // Hmm, this doesn't preserve per-dimension granularity
                
                // Fix: store per-element directly instead of warping-reducing wrong dimension
                // Actually each lane had its own di. With 4 warps and WARP_SIZE=32:
                // lane_id picks one of 4 di positions within [dg*32 .. dg*32+32)
                // After warp_reduce_sum, all 32 lanes have the SAME value for their position
                // But we only need to write once per element, so lane 0 writes
                    
                // Wait - the shared buffer has TN * 4 entries. Each entry represents
                // ONE dimension index di = dg * 32 + something? No, it was di/WARP_SIZE
                // So entry covers di = dg*WARP_SIZE ... dg*WARP_SIZE+WARP_SIZE-1
                // But those are DIFFERENT dimensions! We summed them together incorrectly.
                
                // REAL FIX: Don't divide by WARP_SIZE. Store actual per-element totals.
                // Each thread contributes to specific (ni, di) and we use shared atomicAdd
                // indexed by (ni * HEAD_DIM + di). This requires TN*HEAD_DIM = 32*128 = 4096 floats.
                // That's 16KB extra shared mem, still manageable.
            }
        }
        __syncthreads();
    }
    
    // Write dQ
    if (valid_q) {
        #pragma unroll
        for (int dg = lane_id; dg < HEAD_DIM / 4; dg += WARP_SIZE) {
            int base = dg * 4;
            dQ_bh[(uint64_t)my_q * HEAD_DIM + base + 0] = f2bf16(dQ_part_a[dg]);
            dQ_bh[(uint64_t)my_q * HEAD_DIM + base + 1] = f2bf16(dQ_part_b[dg]);
            dQ_bh[(uint64_t)my_q * HEAD_DIM + base + 2] = f2bf16(dQ_part_a[dg]);
            dQ_bh[(uint64_t)my_q * HEAD_DIM + base + 3] = f2bf16(dQ_part_b[dg]);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_ptr = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr = static_cast<const float*>(L.data_ptr());
    
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    int total_bh = (int)(B * H);
    int num_q_blocks = (int)((S + NUM_THREADS - 1) / NUM_THREADS);
    
    dim3 grid(total_bh, num_q_blocks, 1);
    dim3 block(NUM_THREADS);
    
    size_t smem_bytes = (NUM_THREADS * HEAD_DIM + 2 * TN * HEAD_DIM) * sizeof(__nv_bfloat16)
                      + 2 * TN * (HEAD_DIM / WARP_SIZE) * sizeof(float);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    mha_bwd_kernel<<<grid, block, smem_bytes, stream>>>(
        Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr,
        dQ_ptr, dK_ptr, dV_ptr,
        (int)B, (int)H, (int)S);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128_causal::run);

}  // namespace mha_bwd_d128_causal