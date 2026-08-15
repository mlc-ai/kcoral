#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <cfloat>
#include <cmath>
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

namespace mha_blackwell {

constexpr int BLOCK_M = 64;   // Query rows per CTA
constexpr int BLOCK_N = 64;   // KV columns per tile
constexpr int BLOCK_D = 128;  // Head dimension
constexpr int TPB = 128;      // Threads per CTA
// 2 threads per query row, each handling 64 elements of D
constexpr int TPQ_ROW = TPB / BLOCK_M; // 2
constexpr int D_PER_THREAD = BLOCK_D / TPQ_ROW; // 64

__global__ void mha_causal_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
          __nv_bfloat16* __restrict__ O,
          float*         __restrict__ LSE,
    int B, int H, int S, int D,
    float inv_sqrt_d)
{
    extern __shared__ char smem_raw[];
    
    // Shared memory layout: K tile [BLOCK_N x BLOCK_D], V tile [BLOCK_N x BLOCK_D]
    // Both stored in row-major (n-major): smem[k*n + d]
    __nv_bfloat16* __restrict__ sK = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* __restrict__ sV = reinterpret_cast<__nv_bfloat16*>(smem_raw + BLOCK_N * BLOCK_D * sizeof(__nv_bfloat16));
    
    const int stride_SD = S * D;     // stride for (S,D) plane
    const int stride_BHD = B * H * S * D;
    
    // blockIdx.x encodes (batch, head), blockIdx.y encodes query tile
    const int bh = blockIdx.x;
    const int b = bh / H;
    const int h = bh % H;
    const int q_tile = blockIdx.y;
    
    const int tid = threadIdx.x;
    const int q_local = tid / TPQ_ROW;     // 0..63, query row within tile
    const int lane = tid % TPQ_ROW;         // 0 or 1, which half of D
    
    const int q_global = q_tile * BLOCK_M + q_local;
    const bool q_valid = q_global < S;
    
    // Base pointers for this batch/head
    const __nv_bfloat16* Q_bh = Q + b * stride_SD * H + h * stride_SD;
    const __nv_bfloat16* K_bh = K + b * stride_SD * H + h * stride_SD;
    const __nv_bfloat16* V_bh = V + b * stride_SD * H + h * stride_SD;
    __nv_bfloat16* O_bh = O + b * stride_SD * H + h * stride_SD;
    float* LSE_bh = LSE + b * S * H + h * S;
    
    // ----- Load Q row fragment (fp32, scaled) -----
    float q_frag[D_PER_THREAD];
    float o_acc[D_PER_THREAD];
    
    if (q_valid) {
        const __nv_bfloat16* qr = Q_bh + q_global * D;
        const int d0 = lane * D_PER_THREAD;
        #pragma unroll
        for (int i = 0; i < D_PER_THREAD; ++i) {
            q_frag[i] = __bfloat162float(qr[d0 + i]) * inv_sqrt_d;
        }
    } else {
        #pragma unroll
        for (int i = 0; i < D_PER_THREAD; ++i) {
            q_frag[i] = 0.0f;
        }
    }
    #pragma unroll
    for (int i = 0; i < D_PER_THREAD; ++i) {
        o_acc[i] = 0.0f;
    }
    
    // Online softmax accumulators
    float m_prev = -FLT_MAX;   // running max
    float l_prev = 0.0f;       // running sum (of exp(s - m))
    
    const int num_ntiles = (S + BLOCK_N - 1) / BLOCK_N;
    const unsigned full_mask = 0xFFFFFFFFU;
    
    for (int nt = 0; nt < num_ntiles; ++nt) {
        const int k_base = nt * BLOCK_N;
        
        // ===== Load K tile into shared memory (coalesced) =====
        // Each thread loads D_PER_THREAD contiguous elements from global memory,
        // distributed across BLOCK_N key rows.
        // Thread t loads: elements at flat offsets (t * D_PER_THREAD + i) repeated for each row
        // Global address: K_bh + (k_base + n_local) * D + d_col
        // To make adjacent threads access adjacent memory, each thread's first load
        // is at a unique contiguous position.
        // Flat index within tile: n_local * BLOCK_D + d_col
        // We assign each thread: n_local varies, d_col fixed for that thread
        // d_col = lane * D_PER_THREAD + (tid_within_team variation)
        
        // Simpler: thread t loads seq of D_PER_THREAD elements, then steps by BLOCK_D
        const int d0 = lane * D_PER_THREAD;
        for (int i = 0; i < D_PER_THREAD; ++i) {
            const int d = d0 + i;
            for (int n_local = q_local; n_local < BLOCK_N; n_local += BLOCK_M) {
                // Hmm, this doesn't cover all rows with 64 threads properly.
                // Let me use a different mapping.
            }
        }
        
        // ACTUAL SIMPLER COALESCED LOAD:
        // Total elements: BLOCK_N * BLOCK_D = 8192 bf16
        // Each thread loads 8192 / 128 = 64 elements
        // Flat sequential: thread t loads elements at flat[t*64 + i] for i in 0..63
        // Flat address f maps to (n=f/BLOCK_D, d=f%BLOCK_D) in shared memory
        // And to global address K_bh[(k_base + n) * D + d]
        for (int i = 0; i < D_PER_THREAD; ++i) {
            int flat = tid * D_PER_THREAD + i;
            if (flat < BLOCK_N * BLOCK_D) {
                int nl = flat / BLOCK_D;       // 0..63
                int dc = flat % BLOCK_D;       // 0..127
                int ng = k_base + nl;
                if (ng < S) {
                    sK[flat] = K_bh[ng * D + dc];
                } else {
                    sK[flat] = __float2bfloat16(0.0f);
                }
            }
        }
        
        // ===== Load V tile into shared memory (coalesced) =====
        for (int i = 0; i < D_PER_THREAD; ++i) {
            int flat = tid * D_PER_THREAD + i;
            if (flat < BLOCK_N * BLOCK_D) {
                int nl = flat / BLOCK_D;
                int dc = flat % BLOCK_D;
                int ng = k_base + nl;
                if (ng < S) {
                    sV[flat] = V_bh[ng * D + dc];
                } else {
                    sV[flat] = __float2bfloat16(0.0f);
                }
            }
        }
        __syncthreads();
        
        // ===== Compute Q @ K^T scores for this query row =====
        // q_frag has D_PER_THREAD floats
        // For each key column kc in 0..BLOCK_N-1:
        //   score = sum_{d=0}^{D-1} q[d] * K[kc][d]
        //   split between 2 lanes: each computes D_PER_THREAD partial sum, then reduce
        
        // Check causal validity: key must be before query
        const int kv_start = k_base;  // first key index in this tile
        const int kv_end = min(k_base + BLOCK_N, S);
        // Causal mask: key_pos < query_pos, so key must be in [kv_start, min(kv_end, q_global))
        const int last_valid_key = min(kv_end, q_global);
        const int n_valid_keys = q_valid ? max(0, last_valid_key - kv_start) : 0;
        
        if (n_valid_keys == 0) {
            __syncthreads();
            continue;
        }
        
        // For each valid key column, compute dot product
        float tile_max_score = -FLT_MAX;
        
        // Process all valid key columns
        for (int vc = 0; vc < n_valid_keys; ++vc) {
            // Compute partial dot product for this thread's segment
            float partial = 0.0f;
            const __nv_bfloat16* kcol = sK + vc * BLOCK_D;
            
            #pragma unroll
            for (int di = 0; di < D_PER_THREAD; di += 4) {
                partial += q_frag[di+0] * __bfloat162float(kcol[d0 + di+0]);
                partial += q_frag[di+1] * __bfloat162float(kcol[d0 + di+1]);
                partial += q_frag[di+2] * __bfloat162float(kcol[d0 + di+2]);
                partial += q_frag[di+3] * __bfloat162float(kcol[d0 + di+3]);
            }
            
            // Reduce across the two lanes sharing this query row using shfl
            float score = partial + __shfl_down_sync(full_mask, partial, 1);
            
            if (score > tile_max_score) tile_max_score = score;
        }
        
        // ===== Online softmax update =====
        // Find new running max
        float m_new = m_prev;
        if (tile_max_score > m_prev) m_new = tile_max_score;
        
        // Scale factor for previous accumulation
        float alpha = 1.0f;
        if (m_new != m_prev) {
            alpha = expf(m_prev - m_new);
        }
        
        // Update running sum
        float l_new = l_prev * alpha;
        
        // Recompute scores with updated max and accumulate output
        for (int vc = 0; vc < n_valid_keys; ++vc) {
            float partial = 0.0f;
            const __nv_bfloat16* kcol = sK + vc * BLOCK_D;
            
            #pragma unroll
            for (int di = 0; di < D_PER_THREAD; di += 4) {
                partial += q_frag[di+0] * __bfloat162float(kcol[d0 + di+0]);
                partial += q_frag[di+1] * __bfloat162float(kcol[d0 + di+1]);
                partial += q_frag[di+2] * __bfloat162float(kcol[d0 + di+2]);
                partial += q_frag[di+3] * __bfloat162float(kcol[d0 + di+3]);
            }
            
            float score = partial + __shfl_down_sync(full_mask, partial, 1);
            float p = expf(score - m_new);
            
            l_new += p;
            
            // Accumulate p * V[vc] into o_acc
            const __nv_bfloat16* vcol = sV + vc * BLOCK_D;
            #pragma unroll
            for (int di = 0; di < D_PER_THREAD; ++di) {
                o_acc[di] += p * __bfloat162float(vcol[d0 + di]);
            }
        }
        
        // Also need to scale old o_acc by alpha
        if (alpha != 1.0f) {
            #pragma unroll
            for (int i = 0; i < D_PER_THREAD; ++i) {
                o_acc[i] *= alpha;
            }
        }
        
        m_prev = m_new;
        l_prev = l_new;
        
        __syncthreads();
    }
    
    // ===== Write-back =====
    if (q_valid) {
        float* lse_ptr = LSE_bh + q_global;
        __nv_bfloat16* o_ptr = O_bh + q_global * D;
        const int d0 = lane * D_PER_THREAD;
        
        if (l_prev > 0.0f) {
            float norm = 1.0f / l_prev;
            *lse_ptr = m_prev + logf(l_prev);
            #pragma unroll
            for (int i = 0; i < D_PER_THREAD; ++i) {
                o_ptr[d0 + i] = __float2bfloat16(o_acc[i] * norm);
            }
        } else {
            *lse_ptr = -FLT_MAX;
            #pragma unroll
            for (int i = 0; i < D_PER_THREAD; ++i) {
                o_ptr[d0 + i] = __float2bfloat16(0.0f);
            }
        }
    } else {
        const int d0 = lane * D_PER_THREAD;
        float* lse_ptr = LSE_bh + q_global;
        __nv_bfloat16* o_ptr = O_bh + q_global * D;
        *lse_ptr = 0.0f;
        #pragma unroll
        for (int i = 0; i < D_PER_THREAD; ++i) {
            o_ptr[d0 + i] = __float2bfloat16(0.0f);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());
    
    float inv_sqrt_d = 1.0f / sqrtf(static_cast<float>(D));
    
    int64_t num_bh = B * H;
    int64_t num_qtiles = (S + BLOCK_M - 1) / BLOCK_M;
    
    dim3 grid(static_cast<unsigned>(num_bh), static_cast<unsigned>(num_qtiles));
    dim3 block(TPB);
    
    // Shared memory: K tile + V tile = 2 * BLOCK_N * BLOCK_D bf16
    size_t smem_size = 2ULL * BLOCK_N * BLOCK_D * sizeof(__nv_bfloat16);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    mha_causal_kernel<<<grid, block, smem_size, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data,
        static_cast<int>(B), static_cast<int>(H), static_cast<int>(S),
        static_cast<int>(D), inv_sqrt_d);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_blackwell::run);

}  // namespace mha_blackwell