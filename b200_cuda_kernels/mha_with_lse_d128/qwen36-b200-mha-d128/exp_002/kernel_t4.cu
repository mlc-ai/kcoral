#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>
#include <float.h>
#include <math.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace flash_attn_blackwell {

constexpr uint32_t BM = 64;    // Q tile height (rows processed per block)
constexpr uint32_t BN = 64;    // KV tile width (cols processed per iteration)
constexpr uint32_t BD = 128;   // Head dimension (contracted)
constexpr uint32_t NUM_THREADS = 256;
constexpr uint32_t WARP_SIZE = 32;

__device__ __forceinline__ float bf16_to_float_val(uint16_t v) {
    union { uint16_t u; __nv_bfloat16 h; };
    u = v;
    return __bfloat162float(h);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16_val(float v) {
    return __float2bfloat16(v);
}

__device__ __forceinline__ void write_bf16_zero(__nv_bfloat16& val) {
    val = __float2bfloat16(0.f);
}

__global__ void flash_attn_fwd_kernel(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    __nv_bfloat16* O,
    float* LSE,
    int32_t B, int32_t H, int32_t S, int32_t D,
    int64_t stride_qs, int64_t stride_qd,
    int64_t stride_ks, int64_t stride_kd,
    int64_t stride_vs, int64_t stride_vd,
    int64_t stride_os, int64_t stride_od,
    int64_t stride_lse_s
) {
    extern __shared__ char smem_raw[];
    
    // Partition shared memory
    __nv_bfloat16* smem_Q  = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* smem_K  = smem_Q  + BM * BD;
    __nv_bfloat16* smem_V  = smem_K  + BN * BD;
    float* smem_S          = reinterpret_cast<float*>(smem_V + BN * BD);
    float* smem_O_acc      = smem_S + BM * BN;
    float* smem_row_max_old = smem_O_acc + BM * BD;
    float* smem_row_max_new = smem_row_max_old + BM;
    float* smem_tile_sum   = smem_row_max_new + BM;
    alignas(8) uint64_t barrier_kv_load;
    
    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    const int bid = blockIdx.x;
    
    // Decode block assignment: (batch, head, q_tile)
    int num_q_tiles = (S + BM - 1) / BM;
    int q_tile = bid % num_q_tiles;
    int head_batch = bid / num_q_tiles;
    int batch = head_batch / H;
    int head = head_batch % H;
    
    // Bail out of bounds
    if (batch >= B || head >= H || q_tile * BM >= S) return;
    
    // Base coordinates for this block
    int32_t q_row_base = q_tile * BM;
    int32_t actual_BM = min(BM, S - q_row_base);
    
    // Initialize mbarrier
    if (tid == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], %1;"
            :: "r"(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&barrier_kv_load))), "r"(2));
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();
    
    // Calculate base pointers into global memory for this (batch, head)
    const __nv_bfloat16* Q_base = Q + static_cast<int64_t>(batch) * H * S * D 
                                  + static_cast<int64_t>(head) * S * D 
                                  + static_cast<int64_t>(q_row_base) * D;
    const __nv_bfloat16* K_base = K + static_cast<int64_t>(batch) * H * S * D 
                                  + static_cast<int64_t>(head) * S * D;
    const __nv_bfloat16* V_base = V + static_cast<int64_t>(batch) * H * S * D 
                                  + static_cast<int64_t>(head) * S * D;
    __nv_bfloat16* O_base = O + static_cast<int64_t>(batch) * H * S * D 
                            + static_cast<int64_t>(head) * S * D 
                            + static_cast<int64_t>(q_row_base) * D;
    float* LSE_base = LSE + static_cast<int64_t>(batch) * H * S 
                      + static_cast<int64_t>(head) * S 
                      + q_row_base;
    
    // ============================================
    // Load Q tile into shared memory
    // ============================================
    {
        // Each thread loads 4 bf16 values
        for (int idx = tid; idx < BM * BD; idx += NUM_THREADS) {
            int m = idx / BD;
            int d = idx % BD;
            if (m < actual_BM && d < D) {
                smem_Q[idx] = Q_base[m * D + d];
            } else {
                write_bf16_zero(smem_Q[idx]);
            }
        }
        __syncthreads();
    }
    
    // Initialize output accumulator in shared memory
    for (int idx = tid; idx < BM * BD; idx += NUM_THREADS) {
        smem_O_acc[idx] = 0.f;
    }
    __syncthreads();
    
    // Threads 0..actual_BM-1 own one row each with register state
    bool owns_row = (tid < actual_BM);
    float my_row_max = -FLT_MAX;
    float my_row_sum = 0.f;
    
    float inv_sqrt_D = rsqrtf(static_cast<float>(D));
    int num_kv_tiles = (S + BN - 1) / BN;
    
    // Main loop over KV tiles
    for (int tn = 0; tn < num_kv_tiles; ++tn) {
        int kv_col_base = tn * BN;
        int32_t actual_BN = min(BN, S - kv_col_base);
        
        // Reset barrier for KV loads
        if (tid == 0) {
            asm volatile("mbarrier.inval.shared.b64 [%0];" 
                :: "r"(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&barrier_kv_load))));
            asm volatile("mbarrier.init.shared.b64 [%0], %1;"
                :: "r"(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&barrier_kv_load))), "r"(2));
            asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
        }
        __syncthreads();
        
        // Warp 0 loads K, Warp 1 loads V concurrently
        if (warp == 0) {
            for (int idx = tid / 4; idx < BN * BD; idx += NUM_THREADS / 4) {
                int n = idx / BD;
                int d = idx % BD;
                if (n < actual_BN && d < D) {
                    smem_K[idx] = K_base[static_cast<int64_t>(kv_col_base) * D + static_cast<int64_t>(n) * D + d];
                } else {
                    write_bf16_zero(smem_K[idx]);
                }
            }
            if (tid == 0) {
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                    :: "r"(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&barrier_kv_load))), 
                       "r"(BN * BD * sizeof(__nv_bfloat16)));
            }
        }
        if (warp == 1) {
            for (int idx = tid / 4; idx < BN * BD; idx += NUM_THREADS / 4) {
                int n = idx / BD;
                int d = idx % BD;
                if (n < actual_BN && d < D) {
                    smem_V[idx] = V_base[static_cast<int64_t>(kv_col_base) * D + static_cast<int64_t>(n) * D + d];
                } else {
                    write_bf16_zero(smem_V[idx]);
                }
            }
            if (tid == 0) {
                asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                    :: "r"(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&barrier_kv_load))),
                       "r"(BN * BD * sizeof(__nv_bfloat16)));
            }
        }
        
        // All threads wait for both K and V loads
        {
            asm volatile(
                "{\n.reg .pred P;\n"
                ".WAIT_%=:\n"
                "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
                "@!P bra .WAIT_%=;\n"
                "}\n"
                :: "r"(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&barrier_kv_load))), "r"(tn & 1));
        }
        __syncthreads();
        
        // ============================================
        // Phase 1: Compute S = Q @ K^T for this tile
        // We compute BM x BN attention scores
        // ============================================
        
        int num_elements_per_thread = (BM * BN + NUM_THREADS - 1) / NUM_THREADS;
        
        for (int elem = 0; elem < num_elements_per_thread; ++elem) {
            int idx = tid + elem * NUM_THREADS;
            if (idx >= BM * BN) break;
            
            int m = idx / BN;
            int n = idx % BN;
            
            if (m >= actual_BM || n >= actual_BN) {
                smem_S[idx] = -FLT_MAX;
                continue;
            }
            
            float qk = 0.f;
            const __nv_bfloat16* q_ptr = smem_Q + m * BD;
            const __nv_bfloat16* k_ptr = smem_K + n * BD;
            
            // Vectorized dot product (4 bf16 at a time)
            for (int d = 0; d < BD; d += 4) {
                uint16_t qu[4];
                uint16_t ku[4];
                memcpy(qu, &q_ptr[d], 4 * sizeof(uint16_t));
                memcpy(ku, &k_ptr[d], 4 * sizeof(uint16_t));
                
                float q0 = bf16_to_float_val(qu[0]);
                float q1 = bf16_to_float_val(qu[1]);
                float q2 = bf16_to_float_val(qu[2]);
                float q3 = bf16_to_float_val(qu[3]);
                float k0 = bf16_to_float_val(ku[0]);
                float k1 = bf16_to_float_val(ku[1]);
                float k2 = bf16_to_float_val(ku[2]);
                float k3 = bf16_to_float_val(ku[3]);
                qk += q0*k0 + q1*k1 + q2*k2 + q3*k3;
            }
            
            qk *= inv_sqrt_D;
            smem_S[idx] = qk;
        }
        __syncthreads();
        
        // ============================================
        // Phase 2: Find tile max per row
        // ============================================
        {
            if (owns_row) {
                float max_val = -FLT_MAX;
                int row_start = tid * BN;
                for (int col = 0; col < actual_BN; ++col) {
                    float val = smem_S[row_start + col];
                    if (val > max_val) max_val = val;
                }
                smem_S[BM * BN + tid] = max_val;  // Store row max
            }
            __syncthreads();
        }
        
        // ============================================
        // Phase 3: Update running max and store old/new max
        // ============================================
        {
            if (owns_row) {
                float tile_max = smem_S[BM * BN + tid];
                float new_row_max = fmaxf(my_row_max, tile_max);
                smem_row_max_old[tid] = my_row_max;
                smem_row_max_new[tid] = new_row_max;
                my_row_max = new_row_max;
            }
            __syncthreads();
        }
        
        // ============================================
        // Phase 4: Scale accumulated output if max increased
        // ============================================
        {
            for (int idx = tid; idx < BM * BD; idx += NUM_THREADS) {
                int m = idx / BD;
                if (m < actual_BM) {
                    float row_old = smem_row_max_old[m];
                    float row_new = smem_row_max_new[m];
                    smem_O_acc[idx] *= expf(row_old - row_new);
                }
            }
            __syncthreads();
        }
        
        // ============================================
        // Phase 5: Compute P = exp(S - row_max) and accumulate row sums
        // ============================================
        {
            int num_elements_per_thread = (BM * BN + NUM_THREADS - 1) / NUM_THREADS;
            for (int elem = 0; elem < num_elements_per_thread; ++elem) {
                int idx = tid + elem * NUM_THREADS;
                if (idx >= BM * BN) break;
                
                int m = idx / BN;
                if (m >= actual_BM) continue;
                
                float row_max_val = smem_row_max_new[m];
                float s_val = smem_S[idx];
                
                float p_val = (s_val == -FLT_MAX) ? 0.f : expf(s_val - row_max_val);
                smem_S[idx] = p_val;  // Overwrite S with P
            }
            
            // Reduce sum per row
            if (owns_row) {
                float sum = 0.f;
                int row_start = tid * BN;
                for (int col = 0; col < actual_BN; ++col) {
                    sum += smem_S[row_start + col];
                }
                smem_tile_sum[tid] = sum;
            }
            __syncthreads();
        }
        
        // Update running row_sum
        {
            if (owns_row) {
                float old_max = smem_row_max_old[tid];
                float new_max = smem_row_max_new[tid];
                float tile_sum = smem_tile_sum[tid];
                
                // row_sum = old_sum * exp(old_max - new_max) + tile_sum
                float scaled_old = my_row_sum * expf(old_max - new_max);
                my_row_sum = scaled_old + tile_sum;
            }
            __syncthreads();
        }
        
        // ============================================
        // Phase 6: Accumulate P @ V into output
        // smem_O_acc[m*BD + d] += sum_n(P[m*BN+n] * V[n*BD+d])
        // ============================================
        {
            for (int idx = tid; idx < BM * BD; idx += NUM_THREADS) {
                int m = idx / BD;
                int d = idx % BD;
                
                if (m >= actual_BM) continue;
                
                float acc = 0.f;
                const float* p_row = smem_S + m * BN;
                const __nv_bfloat16* v_data = smem_V;
                
                for (int n = 0; n < actual_BN; ++n) {
                    float p_val = p_row[n];
                    if (p_val == 0.f) continue;
                    uint16_t vu;
                    memcpy(&vu, v_data + n * BD + d, sizeof(uint16_t));
                    float v_val = bf16_to_float_val(vu);
                    acc += p_val * v_val;
                }
                
                smem_O_acc[idx] += acc;
            }
            __syncthreads();
        }
    }
    
    // ============================================
    // Epilogue: Normalize output and write LSE
    // ============================================
    {
        if (owns_row) {
            float lse = my_row_sum;
            float final_max = my_row_max;
            
            if (lse > 0.f && final_max > -FLT_MAX/2.f) {
                float norm = 1.f / lse;
                lse = final_max + logf(lse);  // row_max + log(row_sum)
                
                // Normalize output row
                int base = tid * BD;
                for (int d = 0; d < BD; d += 4) {
                    smem_O_acc[base + d]     *= norm;
                    smem_O_acc[base + d + 1] *= norm;
                    smem_O_acc[base + d + 2] *= norm;
                    smem_O_acc[base + d + 3] *= norm;
                }
                
                // Write LSE
                LSE_base[tid] = lse;
            } else {
                // Zero attention
                LSE_base[tid] = -FLT_MAX;
            }
        }
        __syncthreads();
        
        // Write output O
        for (int idx = tid; idx < BM * BD; idx += NUM_THREADS) {
            int m = idx / BD;
            int d = idx % BD;
            
            if (m < actual_BM && d < D) {
                int64_t o_idx = static_cast<int64_t>(m) * D + d;
                O_base[o_idx] = float_to_bf16_val(smem_O_acc[idx]);
            }
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
    
    // Strides for contiguous row-major layout [B,H,S,D]
    int64_t stride_qd = 1;
    int64_t stride_qs = D;
    int64_t stride_lse_s = S;
    
    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());
    
    // Grid: one block per (batch, head, q_tile)
    int32_t num_q_tiles = static_cast<int32_t>((S + BM - 1) / BM);
    dim3 grid(static_cast<int32_t>(B) * static_cast<int32_t>(H) * num_q_tiles);
    dim3 block(NUM_THREADS);
    
    // Shared memory size calculation
    size_t smem_size = 0;
    smem_size += BM * BD * sizeof(__nv_bfloat16);  // smem_Q: 16KB
    smem_size += BN * BD * sizeof(__nv_bfloat16);  // smem_K: 8KB
    smem_size += BN * BD * sizeof(__nv_bfloat16);  // smem_V: 8KB
    smem_size += BM * BN * sizeof(float);          // smem_S: 16KB
    smem_size += BM * BD * sizeof(float);          // smem_O_acc: 32KB
    smem_size += BM * sizeof(float);               // smem_row_max_old: 256B
    smem_size += BM * sizeof(float);               // smem_row_max_new: 256B
    smem_size += BM * sizeof(float);               // smem_tile_sum: 256B
    smem_size += 16;                               // barrier padding
    smem_size += 128;                              // extra alignment
    // Total ≈ 81.5 KB
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    flash_attn_fwd_kernel<<<grid, block, smem_size, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data,
        static_cast<int32_t>(B), static_cast<int32_t>(H),
        static_cast<int32_t>(S), static_cast<int32_t>(D),
        stride_qs, stride_qd,
        stride_qs, stride_qd,
        stride_qs, stride_qd,
        stride_qs, stride_qd,
        stride_lse_s
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flash_attn_blackwell::run);

}  // namespace flash_attn_blackwell