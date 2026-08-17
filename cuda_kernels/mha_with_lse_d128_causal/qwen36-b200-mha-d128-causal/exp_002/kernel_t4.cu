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

// Tile configuration
constexpr int BM = 16;   // Query rows per block
constexpr int BN = 64;   // Key/Value columns per tile
constexpr int BD = 128;  // Head dimension
constexpr int TPB = 128; // Threads per block
// Each of 128 threads / 16 query rows = 8 threads per query row
// Each thread handles BD/8 = 16 elements along D
constexpr int TPQ_ROW = TPB / BM;      // 8
constexpr int D_PER_TH = BD / TPQ_ROW;  // 16

__global__ void mha_causal_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
          __nv_bfloat16* __restrict__ O,
          float*         __restrict__ LSE,
    int B, int H, int S, int D,
    float inv_sqrt_d)
{
    // Shared memory: K tile [BN x BD], V tile [BN x BD]
    // Layout: sK[n][d], sV[n][d] -- row-major, n varies fastest in flat storage
    extern __shared__ char smem_buf[];
    
    __nv_bfloat16* sK = reinterpret_cast<__nv_bfloat16*>(smem_buf);
    __nv_bfloat16* sV = reinterpret_cast<__nv_bfloat16*>(smem_buf + BN * BD * sizeof(__nv_bfloat16));
    
    int tid = threadIdx.x;
    int q_local = tid / TPQ_ROW;     // 0..15, which query row this thread contributes to
    int lane = tid % TPQ_ROW;         // 0..7, position within the row group
    int d_off = lane * D_PER_TH;      // Starting D index for this thread
    
    // Decode block assignment
    int bh_idx = blockIdx.x;
    int b = bh_idx / H;
    int h = bh_idx % H;
    int q_tile_start = blockIdx.y * BM;
    
    int q_row = q_tile_start + q_local;
    bool valid_q = q_row < S;
    
    // Strides for global memory (layout: [B][H][S][D])
    int stride_SD = S * D;
    int stride_BHD = H * stride_SD;
    int stride_LSE_BHS = H * S;
    
    // Base pointers for this (batch, head)
    const __nv_bfloat16* Q_base = Q + b * stride_BHD + h * stride_SD;
    const __nv_bfloat16* K_base = K + b * stride_BHD + h * stride_SD;
    const __nv_bfloat16* V_base = V + b * stride_BHD + h * stride_SD;
    __nv_bfloat16* O_base = O + b * stride_BHD + h * stride_SD;
    float* LSE_base = LSE + b * stride_LSE_BHS + h * S;
    
    // Load Q fragment: each thread loads D_PER_TH consecutive bf16 values for its row
    // Expanded to fp32 and scaled by inv_sqrt_d
    float q[D_PER_TH];
    float o_acc[D_PER_TH];
    
    if (valid_q) {
        const __nv_bfloat16* qr = Q_base + q_row * D + d_off;
        #pragma unroll
        for (int i = 0; i < D_PER_TH; ++i) {
            q[i] = __bfloat162float(qr[i]) * inv_sqrt_d;
        }
    } else {
        #pragma unroll
        for (int i = 0; i < D_PER_TH; ++i) q[i] = 0.0f;
    }
    #pragma unroll
    for (int i = 0; i < D_PER_TH; ++i) o_acc[i] = 0.0f;
    
    // Online softmax state per query row
    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    
    int num_k_tiles = (S + BN - 1) / BN;
    
    // Mask within the row group for shfl operations
    unsigned mask = ((1u << TPQ_ROW) - 1) << (q_local * TPQ_ROW);
    
    for (int tk = 0; tk < num_k_tiles; ++tk) {
        int k_start = tk * BN;
        
        // --- Cooperative K load ---
        // Map thread id to flat shared mem index: cover BN*BD elements
        int elems_per_thread = (BN * BD + TPB - 1) / TPB;  // ceil
        for (int i = 0; i < elems_per_thread; ++i) {
            int flat = tid * elems_per_thread + i;
            if (flat < BN * BD) {
                int n_local = flat / BD;   // 0..BN-1
                int d_col = flat % BD;     // 0..BD-1
                int k_glob = k_start + n_local;
                if (k_glob < S) {
                    sK[flat] = K_base[k_glob * D + d_col];
                } else {
                    sK[flat] = __float2bfloat16(0.0f);
                }
            }
        }
        
        // --- Cooperative V load ---
        for (int i = 0; i < elems_per_thread; ++i) {
            int flat = tid * elems_per_thread + i;
            if (flat < BN * BD) {
                int n_local = flat / BD;
                int d_col = flat % BD;
                int v_glob = k_start + n_local;
                if (v_glob < S) {
                    sV[flat] = V_base[v_glob * D + d_col];
                } else {
                    sV[flat] = __float2bfloat16(0.0f);
                }
            }
        }
        __syncthreads();
        
        // --- Compute Q @ K^T scores for this thread's query row ---
        // For each key column kc in 0..BN-1, compute dot product
        // Partial sum from this thread, then reduce with siblings in row group
        
        int k_end_global = min(k_start + BN, S);
        int last_valid_k = valid_q ? min(k_end_global, q_row) : 0;
        int first_valid_k = k_start;
        int n_valid_keys = max(0, last_valid_k - first_valid_k);
        
        // Find local tile range: keys in [first_valid_k, last_valid_k) map to
        // local indices [0, n_valid_keys) in the current tile
        // Local key index = k_global - k_start
        int first_lc = 0;
        int last_lc_exclusive = n_valid_keys;
        
        float tile_max_score = -FLT_MAX;
        
        // We need to compute scores for ALL keys (even masked ones use -FLT_MAX)
        // but we can skip accumulation for masked keys
        // For simplicity, compute all BN scores
        
        for (int kc = 0; kc < BN; ++kc) {
            int k_glob = k_start + kc;
            
            if (!valid_q || k_glob >= q_row || k_glob >= S) {
                // Masked: score = -inf
                // Still need partial for shfl, but result is irrelevant
                continue;
            }
            
            // Dot product: sum_{d=d_off..d_off+D_PER_TH-1} q[d]*K[kc][d]
            float partial = 0.0f;
            const __nv_bfloat16* kcol = sK + kc * BD + d_off;
            
            #pragma unroll
            for (int di = 0; di < D_PER_TH; ++di) {
                partial += q[di] * __bfloat162float(kcol[di]);
            }
            
            // Reduce across the 8 threads sharing this query row
            // Thread IDs for this row: q_local*TPQ_ROW .. q_local*TPQ_ROW+7
            // Use shfl_xor within the group
            float score = partial;
            for (int offset = 4; offset > 0; offset >>= 1) {
                float other = __shfl_down_sync(mask, partial, offset);
                if (lane < offset) {
                    partial += other;
                }
            }
            score = partial;
            
            // For offline softmax, store scores temporarily
            // Actually we need them again for the softmax step, so cache them
            if (score > tile_max_score) tile_max_score = score;
        }
        
        if (n_valid_keys == 0) {
            __syncthreads();
            continue;
        }
        
        // --- Online softmax update ---
        float old_max = row_max;
        if (tile_max_score > row_max) {
            row_max = tile_max_score;
        }
        
        float alpha = (old_max != row_max && old_max > -FLT_MAX * 0.5f) ? expf(old_max - row_max) : 1.0f;
        
        // Scale previous accumulations
        if (alpha != 1.0f) {
            row_sum *= alpha;
            #pragma unroll
            for (int i = 0; i < D_PER_TH; ++i) o_acc[i] *= alpha;
        }
        
        // Recompute scores and accumulate weighted V
        for (int kc = 0; kc < BN; ++kc) {
            int k_glob = k_start + kc;
            if (!valid_q || k_glob >= q_row || k_glob >= S) continue;
            
            // Recompute dot product
            float partial = 0.0f;
            const __nv_bfloat16* kcol = sK + kc * BD + d_off;
            #pragma unroll
            for (int di = 0; di < D_PER_TH; ++di) {
                partial += q[di] * __bfloat162float(kcol[di]);
            }
            
            float score = partial;
            for (int offset = 4; offset > 0; offset >>= 1) {
                float other = __shfl_down_sync(mask, partial, offset);
                if (lane < offset) partial += other;
            }
            score = partial;
            
            float p = expf(score - row_max);
            row_sum += p;
            
            // Accumulate p * V[kc][:]
            const __nv_bfloat16* vcol = sV + kc * BD + d_off;
            #pragma unroll
            for (int di = 0; di < D_PER_TH; ++di) {
                o_acc[di] += p * __bfloat162float(vcol[di]);
            }
        }
        
        __syncthreads();
    }
    
    // --- Write-back output and LSE ---
    if (valid_q) {
        float* lse_ptr = LSE_base + q_row;
        __nv_bfloat16* optr = O_base + q_row * D + d_off;
        
        if (row_sum > 0.0f) {
            float norm = 1.0f / row_sum;
            *lse_ptr = row_max + logf(row_sum);
            #pragma unroll
            for (int i = 0; i < D_PER_TH; ++i) {
                optr[i] = __float2bfloat16(o_acc[i] * norm);
            }
        } else {
            *lse_ptr = -FLT_MAX;
            #pragma unroll
            for (int i = 0; i < D_PER_TH; ++i) {
                optr[i] = __float2bfloat16(0.0f);
            }
        }
    } else {
        float* lse_ptr = LSE_base + q_row;
        __nv_bfloat16* optr = O_base + q_row * D + d_off;
        *lse_ptr = 0.0f;
        #pragma unroll
        for (int i = 0; i < D_PER_TH; ++i) {
            optr[i] = __float2bfloat16(0.0f);
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
    int64_t num_qtiles = (S + BM - 1) / BM;
    
    dim3 grid(static_cast<unsigned int>(num_bh), static_cast<unsigned int>(num_qtiles), 1);
    dim3 block(TPB, 1, 1);
    
    size_t smem_size = 2ULL * BN * BD * sizeof(__nv_bfloat16);
    
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