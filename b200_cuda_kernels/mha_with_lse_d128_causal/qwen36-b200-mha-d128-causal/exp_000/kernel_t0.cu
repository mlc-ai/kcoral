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

namespace mha_d128_causal {

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int NUM_THREADS = 128;
constexpr int TMA_TILE_ELTS = BLOCK_M * BLOCK_N; // 4096 elements = 8KB per bf16 tile
constexpr int RESCALE_THRESHOLD = 2.0f;

template <typename T>
struct alignas(64) TMA_Descriptor {};

__forceinline__ __device__ void init_barrier(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
                 :: "r"((unsigned)(__cvta_generic_to_shared(bar))), "r"(count));
}

__forceinline__ __device__ void fence_barrier_init() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__forceinline__ __device__ void tma_load_2d(const CUtensorMap* desc, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :
        : "r"((unsigned)(__cvta_generic_to_shared(smem))),
          "l"((unsigned long long)desc),
          "r"(c0), "r"(c1),
          "r"((unsigned)(__cvta_generic_to_shared(bar)))
        : "memory");
}

__forceinline__ __device__ uint2 ld_shared_128(unsigned addr) {
    uint2 v;
    asm volatile("ld.shared.v2.b32 {%0, %1}, [%2];"
                 : "=r"(v.x), "=r"(v.y)
                 : "r"(addr));
    return v;
}

__global__ void fa_fwd_kernel(
    const __grid_constant__ CUtensorMap tma_q,
    const __grid_constant__ CUtensorMap tma_k,
    const __grid_constant__ CUtensorMap tma_v,
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S, int B, int H, int D)
{
    extern __shared__ char smem_raw[];

    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sK = sQ + BLOCK_M * BLOCK_N;
    __nv_bfloat16* sV = sK + BLOCK_M * BLOCK_N;
    uint64_t* bar_Q = reinterpret_cast<uint64_t*>(sV + BLOCK_M * BLOCK_N);
    uint64_t* bar_KV = bar_Q + 2;

    // Initialize barriers - only thread 0 does this
    if (threadIdx.x == 0) {
        init_barrier(bar_Q, NUM_THREADS);
        init_barrier(bar_KV, NUM_THREADS);
        fence_barrier_init();
    }
    __syncthreads();

    int batch_idx = blockIdx.x / H;
    int head_idx = blockIdx.x % H;
    int stride_bh = (size_t)H * S * D;
    int stride_hs = (size_t)S * D;

    const __nv_bfloat16* base_Q = Q + (size_t)batch_idx * stride_bh + head_idx * stride_hs;
    const __nv_bfloat16* base_K = K + (size_t)batch_idx * stride_bh + head_idx * stride_hs;
    const __nv_bfloat16* base_V = V + (size_t)batch_idx * stride_bh + head_idx * stride_hs;
    __nv_bfloat16* base_O = O + (size_t)batch_idx * stride_bh + head_idx * stride_hs;
    float* base_LSE = LSE + (size_t)batch_idx * (size_t)H * S + head_idx * S;

    const float inv_sqrt_d = rsqrtf((float)D);
    
    // Per-thread row assignments: 128 threads cover 64 rows, 2 threads per row
    const int row_stride = BLOCK_M * BLOCK_M / NUM_THREADS; // = 32
    
    // Local accumulators for online softmax
    float row_max[BLOCK_M] = {};
    float row_sum[BLOCK_M] = {};
    float acc_val[BLOCK_M][BLOCK_N];
    
    #pragma unroll
    for (int i = 0; i < BLOCK_M; i++) {
        row_max[i] = -FLT_MAX;
        row_sum[i] = 0.0f;
        #pragma unroll
        for (int j = 0; j < BLOCK_N; j++) {
            acc_val[i][j] = 0.0f;
        }
    }

    // Number of key-blocks to process
    int num_steps = (S + BLOCK_N - 1) / BLOCK_N;
    int q_block_start = 0;
    
    // Pre-calculate TMA coordinates for this batch/head
    int tma_c1 = batch_idx * H + head_idx; // Flattened batch-head index

    int kv_barrier = 0;
    int q_barrier = 0;

    for (int step = 0; step < num_steps; step++) {
        int k_block_start = step * BLOCK_N;
        bool is_last_step = (step == num_steps - 1);
        
        // Issue TMA loads for Q, K, V with proper offsets
        if (threadIdx.x == 0) {
            // Load Q tile at current query block
            int q_offset = q_block_start * D; // Offset in elements along the sequence*dims
            tma_load_2d(&tma_q, bar_Q + q_barrier, sQ, q_offset, tma_c1);
            
            // Load K tile at current key block
            int k_offset = k_block_start * D;
            tma_load_2d(&tma_k, bar_KV + kv_barrier, sK, k_offset, tma_c1);
            
            // Load V tile at current key block  
            tma_load_2d(&tma_v, bar_KV + kv_barrier, sV, k_offset, tma_c1);
            
            // Switch barrier indices for next iteration
            q_barrier = 1 - q_barrier;
            kv_barrier = 1 - kv_barrier;
        }
        
        // Wait for all threads to arrive and for TMA to complete
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" 
                     : : "r"((unsigned)(__cvta_generic_to_shared(bar_Q))));
        asm volatile("mbarrier.try_wait.parity.shared.b64 _, [%0], %1;"
                     : : "r"((unsigned)(__cvta_generic_to_shared(bar_Q))), "r"(q_barrier & 1));
        
        // Wait for KV barrier too
        asm volatile("mbarrier.try_wait.parity.shared.b64 _, [%0], %1;"
                     : : "r"((unsigned)(__cvta_generic_to_shared(bar_KV))), "r"(kv_barrier & 1));
        
        __syncthreads();
        
        // Compute QK^T for assigned rows
        for (int row = threadIdx.x; row < BLOCK_M; row += NUM_THREADS) {
            int abs_q_pos = q_block_start + row;
            if (abs_q_pos >= S) continue;
            
            #pragma unroll
            for (int col = 0; col < BLOCK_N; col += 4) {
                if (col + 3 >= BLOCK_N && !is_last_step) break;
                
                float sum = 0.0f;
                int k_pos_base = k_block_start + col;
                
                // Vectorized load from shared memory
                unsigned q_addr = (unsigned)__cvta_generic_to_shared(sQ + row * BLOCK_N + col);
                unsigned k_addr = (unsigned)__cvta_generic_to_shared(sK + col * BLOCK_M + row);
                
                // Note: K needs transpose access pattern - sK[col*BLOCK_M + row] gives K[k_row, q_col]
                // which is actually K[col..col+3][row] in original layout
                
                for (int jj = 0; jj < 4 && (k_pos_base + jj) < S; jj++) {
                    unsigned qa = q_addr + jj * sizeof(__nv_bfloat16);
                    unsigned ka = k_addr + jj * BLOCK_N * sizeof(__nv_bfloat16);
                    
                    __nv_bfloat16 q_val = *reinterpret_cast<__nv_bfloat16*>(qa);
                    __nv_bfloat16 k_val = *reinterpret_cast<__nv_bfloat16*>(ka);
                    sum += (float)__bfloat162float(q_val) * (float)__bfloat162float(k_val);
                }
                
                // Apply causal mask
                int abs_k_pos = k_pos_base;
                if (abs_k_pos > abs_q_pos) {
                    acc_val[row][col] = -FLT_MAX;
                } else {
                    acc_val[row][col] = sum * inv_sqrt_d;
                }
            }
        }
        
        // Apply softmax update and accumulate
        for (int row = threadIdx.x; row < BLOCK_M; row += NUM_THREADS) {
            int abs_q_pos = q_block_start + row;
            if (abs_q_pos >= S) continue;
            
            float cur_max = row_max[row];
            float cur_sum = row_sum[row];
            
            #pragma unroll
            for (int col = 0; col < BLOCK_N; col++) {
                float val = acc_val[row][col];
                int abs_k_pos = k_block_start + col;
                
                if (abs_k_pos > abs_q_pos || abs_k_pos >= S) {
                    continue;
                }
                
                float scaled = val * inv_sqrt_d;
                float exp_val;
                
                if (scaled > cur_max) {
                    exp_val = expf(cur_max - scaled);
                    float prev_scaled = exp_val;
                    
                    // Rescale if needed
                    if (cur_max > -FLT_MAX) {
                        float alpha = expf(cur_max - scaled);
                        float beta = expf(scaled - scaled);
                        
                        #pragma unroll
                        for (int kk = 0; kk < BLOCK_N; kk++) {
                            acc_val[row][kk] *= alpha;
                            acc_val[row][kk] += beta * (float)__bfloat162float(sV[col * BLOCK_M + kk]);
                        }
                    } else {
                        #pragma unroll
                        for (int kk = 0; kk < BLOCK_N; kk++) {
                            acc_val[row][kk] += (float)__bfloat162float(sV[col * BLOCK_M + kk]);
                        }
                    }
                    
                    cur_sum = cur_sum * exp_val + 1.0f;
                    cur_max = scaled;
                } else {
                    float e = expf(scaled - cur_max);
                    cur_sum += e;
                    
                    #pragma unroll
                    for (int kk = 0; kk < BLOCK_N; kk++) {
                        acc_val[row][kk] += e * (float)__bfloat162float(sV[col * BLOCK_M + kk]);
                    }
                }
            }
            
            row_max[row] = cur_max;
            row_sum[row] = cur_sum;
        }
        
        q_block_start += BLOCK_M;
        if (q_block_start >= S) q_block_start = 0;
    }
    
    // Epilogue: normalize and store
    for (int row = threadIdx.x; row < BLOCK_M; row += NUM_THREADS) {
        int abs_q_pos = q_block_start - BLOCK_M + row; // Adjust for final position
        if (abs_q_pos < 0 || abs_q_pos >= S) continue;
        
        float denom = row_sum[row];
        float log_sumexp = row_max[row] + logf(denom);
        
        // Write LSE atomically
        atomicAdd(base_LSE + abs_q_pos, log_sumexp);
        
        // Normalize and store output
        for (int col = 0; col < BLOCK_N; col++) {
            int abs_k_pos = (num_steps - 1) * BLOCK_N + col;
            if (abs_k_pos >= S) break;
            
            float final_val = acc_val[row][col] / denom;
            base_O[(size_t)abs_q_pos * D + col] = __float2bfloat16(final_val);
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
    
    void* q_ptr = (void*)Q.data_ptr();
    void* k_ptr = (void*)K.data_ptr();
    void* v_ptr = (void*)V.data_ptr();
    void* o_ptr = (void*)O.data_ptr();
    void* lse_ptr = (void*)LSE.data_ptr();
    
    // Create TMA descriptors for Q, K, V
    // Layout: [B, H, S, D] flattened as (S*D) x (B*H)
    cuuint64_t global_dim[2] = {(cuuint64_t)S * D, (cuuint64_t)B * H};
    cuuint64_t global_stride[1] = {(cuuint64_t)S * D * sizeof(__nv_bfloat16)};
    cuuint32_t box_dim[2] = {BLOCK_N, BLOCK_M}; // inner=cols(D-side), outer=rows(S-side)
    cuuint32_t elem_stride[2] = {1, 1};
    
    CUtensorMap tma_q, tma_k, tma_v;
    
    auto encode_tma = [&](CUtensorMap* desc, void* base) -> CUresult {
        return cuTensorMapEncodeTiled(
            desc, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2, base,
            global_dim, global_stride,
            box_dim, elem_stride,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    };
    
    CUresult err_q = encode_tma(&tma_q, q_ptr);
    CUresult err_k = encode_tma(&tma_k, k_ptr);
    CUresult err_v = encode_tma(&tma_v, v_ptr);
    
    if (err_q != CUDA_SUCCESS || err_k != CUDA_SUCCESS || err_v != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to create TMA descriptor\n");
        return;
    }
    
    dim3 grid(B * H);
    dim3 block(NUM_THREADS);
    
    // Shared memory: 3 tiles of BLOCK_M*BLOCK_N bf16 + 2 mbarriers * 8 bytes
    size_t smem_size = 3 * BLOCK_M * BLOCK_N * sizeof(__nv_bfloat16) + 2 * sizeof(uint64_t);
    
    cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id);
    
    // Launch kernel with dynamic shared memory
    fa_fwd_kernel<<<grid, block, smem_size, stream>>>(
        tma_q, tma_k, tma_v,
        static_cast<const __nv_bfloat16*>(q_ptr),
        static_cast<const __nv_bfloat16*>(k_ptr),
        static_cast<const __nv_bfloat16*>(v_ptr),
        static_cast<__nv_bfloat16*>(o_ptr),
        static_cast<float*>(lse_ptr),
        (int)S, (int)B, (int)H, (int)D);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_d128_causal::run);

}  // namespace mha_d128_causal