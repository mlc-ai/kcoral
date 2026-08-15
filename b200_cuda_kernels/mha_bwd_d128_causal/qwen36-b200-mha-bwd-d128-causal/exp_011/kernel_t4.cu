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
constexpr int TN = 32;  // N tile size

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

// Helper: atomic add float value to bf16 array at index idx
// Uses uint2 reinterpretation for proper 4-byte alignment
__device__ __forceinline__ void atomic_add_bf16(__nv_bfloat16* arr, int idx, float val) {
    // Cast to uint2*, atomically add to the pair containing idx
    uint2* ptr = reinterpret_cast<uint2*>(arr);
    int pair_idx = idx >> 1;
    int bit_offset = (idx & 1) ? 16 : 0;
    
    uint2 old_val = ptr[pair_idx];
    uint2 new_val;
    do {
        float existing = (bit_offset == 0) 
            ? __uint_as_float(old_val.x) 
            : __uint_as_float(old_val.y);
        float summed = existing + val;
        
        if (bit_offset == 0) {
            new_val.x = __float_as_uint(summed);
        } else {
            new_val.y = __float_as_uint(summed);
        }
        new_val = (bit_offset == 0) ? make_uint2(__float_as_uint(summed), old_val.y) 
                                    : make_uint2(old_val.x, __float_as_uint(summed));
    } while (old_val.x != atomicCAS(&ptr[pair_idx].x, old_val.x, new_val.x) || 
             (bit_offset == 0 && old_val.y != new_val.y));
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
    int B, int H, int S, int D
) {
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    
    int bh_idx = blockIdx.x;
    if (bh_idx >= B * H) return;
    
    int b = bh_idx / H;
    int h = bh_idx % H;
    
    const __nv_bfloat16* Q_bh = Q + (uint64_t)(b * H + h) * S * D;
    const __nv_bfloat16* K_bh = K + (uint64_t)(b * H + h) * S * D;
    const __nv_bfloat16* V_bh = V + (uint64_t)(b * H + h) * S * D;
    const __nv_bfloat16* dO_bh = dO + (uint64_t)(b * H + h) * S * D;
    const float* L_bh = L + (b * H + h) * S;
    
    __nv_bfloat16* dQ_bh = dQ + (uint64_t)(b * H + h) * S * D;
    __nv_bfloat16* dK_bh = dK + (uint64_t)(b * H + h) * S * D;
    __nv_bfloat16* dV_bh = dV + (uint64_t)(b * H + h) * S * D;
    
    // Shared memory
    extern __shared__ char smem[];
    
    // sQ: 128x128 bf16 = 32KB (all query rows in block)
    // sKV_K: TN x D bf16 = 8KB
    // sKV_V: TN x D bf16 = 8KB  
    // Total: 48KB
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* sKV_K = sQ + NUM_THREADS * D;
    __nv_bfloat16* sKV_V = sKV_K + TN * D;
    
    int num_q_blocks = (S + NUM_THREADS - 1) / NUM_THREADS;
    int q_block = blockIdx.y;
    if (q_block >= num_q_blocks) return;
    
    int q_start = q_block * NUM_THREADS;
    int my_q_local = tid;
    int my_q = q_start + my_q_local;
    
    // Load Q block into shared memory
    for (int qi = warp_id; qi < NUM_THREADS; qi += 4) {
        int qk = q_start + qi;
        for (int di = lane_id; di < D; di += WARP_SIZE) {
            if (qk < S) {
                sQ[(uint64_t)qi * D + di] = Q_bh[(uint64_t)qk * D + di];
            } else {
                sQ[(uint64_t)qi * D + di] = f2bf16(0.0f);
            }
        }
    }
    __syncthreads();
    
    int num_n_tiles = (S + TN - 1) / TN;
    
    // Initialize dQ accumulator per thread
    float dQ_acc[D];
    #pragma unroll
    for (int i = 0; i < D; i++) dQ_acc[i] = 0.0f;
    
    for (int nt = 0; nt < num_n_tiles; nt++) {
        int nk_start = nt * TN;
        if (nk_start >= S) break;
        
        // Load K tile cooperatively
        for (int ni = 0; ni < TN; ni++) {
            int nk = nk_start + ni;
            for (int di = lane_id; di < D; di += WARP_SIZE) {
                if (nk < S) {
                    sKV_K[ni * D + di] = K_bh[(uint64_t)nk * D + di];
                } else {
                    sKV_K[ni * D + di] = f2bf16(0.0f);
                }
            }
        }
        __syncthreads();
        
        // Load V tile cooperatively
        for (int ni = 0; ni < TN; ni++) {
            int nk = nk_start + ni;
            for (int di = lane_id; di < D; di += WARP_SIZE) {
                if (nk < S) {
                    sKV_V[ni * D + di] = V_bh[(uint64_t)nk * D + di];
                } else {
                    sKV_V[ni * D + di] = f2bf16(0.0f);
                }
            }
        }
        __syncthreads();
        
        if (my_q >= S) goto next_tile;
        
        float my_lse = L_bh[my_q];
        
        // Compute scores and dP_scalar for each n-position in this tile
        float stored_prob[TN];
        float stored_dP_scalar[TN];
        float dP_dot_sum = 0.0f;
        
        #pragma unroll
        for (int ni = 0; ni < TN; ni++) {
            int nk = nk_start + ni;
            
            // score = Q[my_q,:] . K[nk,:]
            float score = 0.0f;
            #pragma unroll
            for (int di = 0; di < D; di++) {
                score += bf162f(sQ[(uint64_t)my_q_local * D + di]) * bf162f(sKV_K[ni * D + di]);
            }
            score *= INV_SQRT_D;
            
            // dP_scalar = dO[my_q,:] . V[nk,:]
            float dP_s = 0.0f;
            for (int di = lane_id; di < D; di += WARP_SIZE) {
                dP_s += bf162f(dO_bh[(uint64_t)my_q * D + di]) * bf162f(sKV_V[ni * D + di]);
            }
            dP_s = warp_reduce_sum(dP_s);
            
            stored_dP_scalar[ni] = dP_s;
            
            float prob = (nk <= my_q) ? fast_exp2((score - my_lse) * LOG2E) : 0.0f;
            stored_prob[ni] = prob;
            
            if (prob > 0.0f) {
                dP_dot_sum += prob * dP_s;
            }
        }
        
        // Second pass: compute ds and accumulate outputs
        #pragma unroll
        for (int ni = 0; ni < TN; ni++) {
            int nk = nk_start + ni;
            float prob = stored_prob[ni];
            if (nk > my_q || prob == 0.0f) continue;
            
            float ds = prob * (stored_dP_scalar[ni] - dP_dot_sum);
            
            // Accumulate dQ
            #pragma unroll
            for (int di = 0; di < D; di++) {
                dQ_acc[di] += ds * bf162f(sKV_K[ni * D + di]);
            }
            
            // Accumulate dK and dV using atomics
            for (int di = lane_id; di < D; di += WARP_SIZE) {
                float dk_val = ds * bf162f(sQ[(uint64_t)my_q_local * D + di]);
                float dv_val = prob * bf162f(dO_bh[(uint64_t)my_q * D + di]);
                
                // Atomic add to bf16 arrays using uint2 technique
                atomic_add_bf16(dK_bh, (int)((uint64_t)nk * D + di), dk_val);
                atomic_add_bf16(dV_bh, (int)((uint64_t)nk * D + di), dv_val);
            }
        }
        
        next_tile:
        __syncthreads();
    }
    
    // Write dQ (each thread writes its own row - no conflict)
    if (my_q < S) {
        for (int di = lane_id; di < D; di += WARP_SIZE) {
            dQ_bh[(uint64_t)my_q * D + di] = f2bf16(dQ_acc[di]);
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
    
    int total_bh = (int)(B * H);
    int num_q_blocks = (int)((S + NUM_THREADS - 1) / NUM_THREADS);
    
    dim3 grid(total_bh, num_q_blocks, 1);
    dim3 block(NUM_THREADS);
    
    size_t smem_bytes = (NUM_THREADS * HEAD_DIM + 2 * TN * HEAD_DIM) * sizeof(__nv_bfloat16);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    mha_bwd_kernel<<<grid, block, smem_bytes, stream>>>(
        Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr,
        dQ_ptr, dK_ptr, dV_ptr,
        (int)B, (int)H, (int)S, (int)D);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128_causal::run);

}  // namespace mha_bwd_d128_causal