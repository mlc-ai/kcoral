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

// ---- TMA Descriptor Helper ----
static inline CUresult create_tma_2d_descriptor_bf16(CUtensorMap* tmap, void* globalAddress, 
    uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, 
    uint32_t smem_inner_dim, uint32_t smem_outer_dim,
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * sizeof(__nv_bfloat16)};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        tmap,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2, // tensorRank
        globalAddress,
        globalDim,
        globalStrides,
        boxDim,
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        l2Promotion,
        oobFill
    );
}

// ---- Inline Device Functions ----
__device__ __forceinline__ uint32_t cvta_shared(uintptr_t p) {
    uint32_t r;
    asm volatile("cvta.to.shared.u32 %0, %1;" : "=r"(r) : "l"(p));
    return r;
}

__device__ __forceinline__ void init_mbarrier(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"(cvta_shared(reinterpret_cast<uintptr_t>(bar))), "r"(count));
}

__device__ __forceinline__ void fence_mbarrier_init() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"(cvta_shared(reinterpret_cast<uintptr_t>(bar))), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait(uint64_t* bar, uint32_t parity) {
    asm volatile(
        "{\n.reg .pred P;\n"
        ".reg .u64 phase;\n"
        "WAIT_LOOP_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_LOOP_%=;\n"
        "}\n"
        :: "r"(cvta_shared(reinterpret_cast<uintptr_t>(bar))), "r"(parity));
}

__device__ __forceinline__ void tma_load_2d(const CUtensorMap* desc, uint64_t* barrier, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4}], [%2];"
        :: "r"(cvta_shared(reinterpret_cast<uintptr_t>(smem))),
           "l"((uintptr_t)desc),
           "r"(cvta_shared(reinterpret_cast<uintptr_t>(barrier))),
           "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void fence_proxy_async() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ float bf16_to_float(uint16_t v) {
    __nv_bfloat16 h;
    memcpy(&h, &v, sizeof(h));
    return __bfloat162float(h);
}

__device__ __forceinline__ uint16_t float_to_bf16(float v) {
    __nv_bfloat16 h = __float2bfloat16(v);
    uint16_t r;
    memcpy(&r, &h, sizeof(r));
    return r;
}

namespace flash_attn_blackwell {

constexpr uint32_t BM = 128;   // Q tile height (rows processed per block)
constexpr uint32_t BN = 128;   // KV tile width (cols processed per iteration)
constexpr uint32_t BD = 128;   // Head dimension (contracted)
constexpr uint32_t NUM_THREADS = 256;
constexpr uint32_t WARP_SIZE = 32;
constexpr uint32_t NUM_WARPS = NUM_THREADS / WARP_SIZE;  // 8 warps
constexpr uint32_t BK_PER_UMMA = 16;  // UMMA contracts 16 cols of K per call
constexpr uint32_t NUM_UMMAS = BD / BK_PER_UMMA;  // 8 UMMA calls to cover BD=128

// Shared memory layouts:
// smem_Q: BM x BD = 128 x 128 bf16 = 32KB
// smem_K: BN x BD = 128 x 128 bf16 = 32KB  
// smem_V: BN x BD = 128 x 128 bf16 = 32KB
// smem_S: BM x BN float32 (attention scores) = 64KB
// smem_O_acc: BM x BD float32 (output accumulator) = 64KB
// Barrier and misc: ~1KB
// Total: ~191KB

__global__ void flash_attn_fwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O_store,
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    __nv_bfloat16* O,
    float* LSE,
    int32_t B, int32_t H, int32_t S, int32_t D,
    int32_t stride_qs, int32_t stride_qd,
    int32_t stride_ks, int32_t stride_kd,
    int32_t stride_vs, int32_t stride_vd,
    int32_t stride_os, int32_t stride_od,
    int32_t stride_lse_s
) {
    extern __shared__ char smem_raw[];
    
    // Partition shared memory
    __nv_bfloat16* smem_Q  = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* smem_K  = smem_Q  + BM * BD;
    __nv_bfloat16* smem_V  = smem_K  + BN * BD;
    float* smem_S          = reinterpret_cast<float*>(smem_V + BN * BD);
    float* smem_O_acc      = smem_S + BM * BN;
    alignas(8) uint64_t barrier_Q;
    alignas(8) uint64_t barrier_KV;
    
    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    const int bid = blockIdx.x;
    
    // Decode block assignment: (batch, head, q_tile)
    int q_tile = bid % ((S + BM - 1) / BM);
    int head_batch = bid / ((S + BM - 1) / BM);
    int batch = head_batch / H;
    int head = head_batch % H;
    
    // Bail out of bounds
    if (batch >= B || head >= H || q_tile * BM >= S) return;
    
    // Base coordinates for this block
    int32_t q_row_base = q_tile * BM;
    
    // Initialize mbarriers
    if (tid == 0) {
        init_mbarrier(&barrier_Q, 1);
        init_mbarrier(&barrier_KV, 2);  // Wait for both K and V loads
        fence_mbarrier_init();
    }
    __syncthreads();
    
    // Load Q tile once (first time)
    {
        prefetch_tensormap(&tma_Q);
        tma_load_2d(&tma_Q, &barrier_Q, smem_Q, q_row_base, 0);
        mbarrier_arrive_expect_tx(&barrier_Q, 0);
        mbarrier_wait(&barrier_Q, 0);
    }
    __syncthreads();
    
    // Initialize output accumulator and softmax state for each row
    float row_max[BM];
    float row_sum[BM];
    
    // Each thread initializes its portion of the output accumulator
    for (int idx = tid; idx < BM * BD; idx += NUM_THREADS) {
        smem_O_acc[idx] = 0.f;
    }
    for (int idx = tid; idx < BM; idx += NUM_THREADS) {
        row_max[idx] = -FLT_MAX;
        row_sum[idx] = 0.f;
    }
    __syncthreads();
    
    float inv_sqrt_D = rsqrtf(static_cast<float>(D));
    int num_kv_tiles = (S + BN - 1) / BN;
    
    // Main loop over KV tiles
    for (int tn = 0; tn < num_kv_tiles; ++tn) {
        int kv_col_base = tn * BN;
        
        // Reset barrier for KV loads
        if (tid == 0) {
            asm volatile("mbarrier.inval.shared.b64 [%0];" :: "r"(cvta_shared(reinterpret_cast<uintptr_t>(&barrier_KV))));
            init_mbarrier(&barrier_KV, 2);
            fence_mbarrier_init();
        }
        __syncthreads();
        
        // Warp 0 loads K, Warp 1 loads V concurrently
        if (warp == 0) {
            prefetch_tensormap(&tma_K);
            tma_load_2d(&tma_K, &barrier_KV, smem_K, kv_col_base, 0);
            fence_proxy_async();
        }
        if (warp == 1) {
            prefetch_tensormap(&tma_V);
            tma_load_2d(&tma_V, &barrier_KV, smem_V, kv_col_base, 0);
            fence_proxy_async();
        }
        
        // All threads wait for both K and V loads
        mbarrier_wait(&barrier_KV, (tn & 1));
        __syncthreads();
        
        // ============================================
        // Phase 1: Compute S = Q @ K^T for this tile
        // We compute BM x BN attention scores
        // Each thread computes some (m, n) elements
        // ============================================
        
        // Manual tiling of the contracted dimension (BD=128)
        // Each thread processes multiple (m,n) pairs, doing dot products over d
        int num_elements_per_thread = (BM * BN + NUM_THREADS - 1) / NUM_THREADS;
        
        float local_tile_max = -FLT_MAX;
        
        for (int elem = 0; elem < num_elements_per_thread; ++elem) {
            int idx = tid + elem * NUM_THREADS;
            if (idx >= BM * BN) break;
            
            int m = idx / BN;
            int n = idx % BN;
            
            if (q_row_base + m >= S || kv_col_base + n >= S) {
                smem_S[idx] = -FLT_MAX;
                continue;
            }
            
            float qk = 0.f;
            const __nv_bfloat16* q_ptr = smem_Q + m * BD;
            const __nv_bfloat16* k_ptr = smem_K + n * BD;
            
            // Vectorized dot product (4 bf16 at a time = 2 half2 or manual)
            for (int d = 0; d < BD; d += 4) {
                float q0 = bf16_to_float(*reinterpret_cast<uint16_t*>(&q_ptr[d]));
                float q1 = bf16_to_float(*reinterpret_cast<uint16_t*>(&q_ptr[d+1]));
                float q2 = bf16_to_float(*reinterpret_cast<uint16_t*>(&q_ptr[d+2]));
                float q3 = bf16_to_float(*reinterpret_cast<uint16_t*>(&q_ptr[d+3]));
                float k0 = bf16_to_float(*reinterpret_cast<uint16_t*>(&k_ptr[d]));
                float k1 = bf16_to_float(*reinterpret_cast<uint16_t*>(&k_ptr[d+1]));
                float k2 = bf16_to_float(*reinterpret_cast<uint16_t*>(&k_ptr[d+2]));
                float k3 = bf16_to_float(*reinterpret_cast<uint16_t*>(&k_ptr[d+3]));
                qk += q0*k0 + q1*k1 + q2*k2 + q3*k3;
            }
            
            qk *= inv_sqrt_D;
            smem_S[idx] = qk;
            
            if (qk > local_tile_max) local_tile_max = qk;
        }
        __syncthreads();
        
        // ============================================
        // Phase 2: Find tile max per row
        // ============================================
        // Use warp-level reduction within each row group
        // Warp handles 32 consecutive n values for a subset of rows
        {
            // Each thread contributes to reducing max for one or more rows
            // For simplicity: threads 0..127 handle rows 0..127
            // Use shared memory for row-wise reduction
            
            if (tid < BM) {
                // Collect max across BN elements for this row
                float max_val = -FLT_MAX;
                int row_start = tid * BN;
                for (int col = 0; col < BN; ++col) {
                    float val = smem_S[row_start + col];
                    if (val > max_val) max_val = val;
                }
                smem_S[BM * BN + tid] = max_val;  // Store row max at end of smem_S
            }
            __syncthreads();
        }
        
        // ============================================
        // Phase 3: Update running max and compute softmax
        // ============================================
        {
            float old_row_max = row_max[tid % BM];  // Each thread handles one row
            
            if (tid < BM) {
                float tile_max = smem_S[BM * BN + tid];
                float new_row_max = fmaxf(old_row_max, tile_max);
                row_max[tid] = new_row_max;
                
                // Save old_max and new_max for later
                smem_S[BM * BN * 2 + tid * 2 + 0] = old_row_max;
                smem_S[BM * BN * 2 + tid * 2 + 1] = new_row_max;
            }
            __syncthreads();
        }
        
        // ============================================
        // Phase 4: Scale accumulated output if max increased
        // ============================================
        {
            float old_max = smem_S[BM * BN * 2 + (tid % BM) * 2 + 0];
            float new_max = smem_S[BM * BN * 2 + (tid % BM) * 2 + 1];
            float scale_factor = expf(old_max - new_max);
            
            // Scale the output accumulator for rows handled by this thread
            for (int idx = tid; idx < BM * BD; idx += NUM_THREADS) {
                int m = idx / BD;
                // Get the correct scale for this row
                float row_old = smem_S[BM * BN * 2 + m * 2 + 0];
                float row_new = smem_S[BM * BN * 2 + m * 2 + 1];
                smem_O_acc[idx] *= expf(row_old - row_new);
            }
            __syncthreads();
        }
        
        // ============================================
        // Phase 5: Compute P = exp(S - row_max) and accumulate row sums
        // ============================================
        {
            float local_tile_sum = 0.f;
            
            int num_elements_per_thread = (BM * BN + NUM_THREADS - 1) / NUM_THREADS;
            for (int elem = 0; elem < num_elements_per_thread; ++elem) {
                int idx = tid + elem * NUM_THREADS;
                if (idx >= BM * BN) break;
                
                int m = idx / BN;
                float row_max_val = smem_S[BM * BN * 2 + m * 2 + 1];
                
                float s_val = smem_S[idx];
                float p_val = expf(s_val - row_max_val);
                smem_S[idx] = p_val;  // Overwrite S with P
                local_tile_sum += p_val;
            }
            
            // Reduce tile sum per row
            // Simple approach: each thread reduces its portion for its assigned rows
            // For simplicity, do a warp-level reduce
            float* row_sums = &smem_S[BM * BN * 3];
            if (tid < BM) {
                float sum = 0.f;
                int row_start = tid * BN;
                for (int col = 0; col < BN; ++col) {
                    sum += smem_S[row_start + col];
                }
                row_sums[tid] = sum;
            }
            __syncthreads();
        }
        
        // Update running row_sum
        {
            if (tid < BM) {
                float old_max = smem_S[BM * BN * 2 + tid * 2 + 0];
                float new_max = smem_S[BM * BN * 2 + tid * 2 + 1];
                float old_sum = row_sum[tid];
                float tile_sum = smem_S[BM * BN * 3 + tid];
                
                // row_sum = old_sum * exp(old_max - new_max) + tile_sum
                float scaled_old = old_sum * expf(old_max - new_max);
                row_sum[tid] = scaled_old + tile_sum;
            }
            __syncthreads();
        }
        
        // ============================================
        // Phase 6: Accumulate P @ V into output
        // smem_O_acc[m*BD + d] += sum_n(P[m*BN+n] * V[n*BD+d])
        // ============================================
        {
            // Each thread accumulates some (m, d) outputs
            for (int idx = tid; idx < BM * BD; idx += NUM_THREADS) {
                int m = idx / BD;
                int d = idx % BD;
                
                float acc = 0.f;
                const float* p_row = smem_S + m * BN;
                const __nv_bfloat16* v_data = smem_V;
                
                for (int n = 0; n < BN; ++n) {
                    if (kv_col_base + n >= S) break;
                    float p_val = p_row[n];
                    float v_val = bf16_to_float(*reinterpret_cast<uint16_t*>(v_data + n * BD + d));
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
        if (tid < BM) {
            float lse = row_sum[tid];
            if (lse > 0.f) {
                float norm = 1.f / lse;
                lse = smem_S[BM * BN * 2 + tid * 2 + 1] + logf(lse);  // row_max + log(row_sum)
                
                // Normalize output row
                for (int d = 0; d < BD; d += 4) {
                    int idx = tid * BD + d;
                    smem_O_acc[idx]    *= norm;
                    smem_O_acc[idx + 1] *= norm;
                    smem_O_acc[idx + 2] *= norm;
                    smem_O_acc[idx + 3] *= norm;
                }
                
                // Write LSE
                int out_row = q_row_base + tid;
                if (out_row < S) {
                    LSE[batch * H * S + head * S + out_row] = lse;
                }
            } else {
                // Zero attention
                float out_row = q_row_base + tid;
                if (out_row < S) {
                    LSE[batch * H * S + head * S + out_row] = -FLT_MAX;
                }
            }
        }
        __syncthreads();
        
        // Write output O
        for (int idx = tid; idx < BM * BD; idx += NUM_THREADS) {
            int m = idx / BD;
            int d = idx % BD;
            int out_row = q_row_base + m;
            int out_col = d;
            
            if (out_row < S && out_col < D) {
                int o_idx = batch * H * S * D + head * S * D + out_row * D + out_col;
                O[o_idx] = __float2bfloat16(smem_O_acc[idx]);
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
    
    // Strides for row-major layout
    int64_t stride_qd = 1;
    int64_t stride_qs = D;
    int64_t stride_hd = S * D;
    int64_t stride_bd = H * S * D;
    
    int32_t stride_qs_i = static_cast<int32_t>(stride_qs);
    int32_t stride_qd_i = static_cast<int32_t>(stride_qd);
    int32_t stride_ks_i = static_cast<int32_t>(stride_qs);
    int32_t stride_kd_i = static_cast<int32_t>(stride_qd);
    int32_t stride_vs_i = static_cast<int32_t>(stride_qs);
    int32_t stride_vd_i = static_cast<int32_t>(stride_qd);
    int32_t stride_os_i = static_cast<int32_t>(stride_qs);
    int32_t stride_od_i = static_cast<int32_t>(stride_qd);
    int32_t stride_lse_s = static_cast<int32_t>(S);
    
    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());
    
    // Create TMA descriptors
    CUtensorMap tma_Q, tma_K, tma_V, tma_O;
    
    // Q: (S, D) inner=D outer=S, box=(BD, BM)=(128, 128)
    CUDA_CHECK(cuInit(0));
    create_tma_2d_descriptor_bf16(&tma_Q, const_cast<__nv_bfloat16*>(Q_data),
        D, S, BD, BM,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    create_tma_2d_descriptor_bf16(&tma_K, const_cast<__nv_bfloat16*>(K_data),
        D, S, BD, BN,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    create_tma_2d_descriptor_bf16(&tma_V, const_cast<__nv_bfloat16*>(V_data),
        D, S, BD, BN,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    create_tma_2d_descriptor_bf16(&tma_O, O_data,
        D, S, BD, BM,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    // Grid: one block per (batch, head, q_tile)
    int32_t num_q_tiles = (S + BM - 1) / BM;
    dim3 grid(B * H * num_q_tiles);
    dim3 block(NUM_THREADS);
    
    // Shared memory size calculation
    size_t smem_size = 0;
    smem_size += BM * BD * sizeof(__nv_bfloat16);  // smem_Q
    smem_size += BN * BD * sizeof(__nv_bfloat16);  // smem_K
    smem_size += BN * BD * sizeof(__nv_bfloat16);  // smem_V
    smem_size += BM * BN * sizeof(float);          // smem_S (reuse regions)
    smem_size += BM * BD * sizeof(float);          // smem_O_acc
    smem_size += BM * sizeof(float);               // row max scratch area
    smem_size += BM * sizeof(float) * 2;           // old/new max storage
    smem_size += BM * sizeof(float);               // row sums
    smem_size += 64;                               // alignment padding
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    flash_attn_fwd_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, tma_O,
        Q_data, K_data, V_data, O_data, LSE_data,
        static_cast<int32_t>(B), static_cast<int32_t>(H),
        static_cast<int32_t>(S), static_cast<int32_t>(D),
        stride_qs_i, stride_qd_i,
        stride_ks_i, stride_kd_i,
        stride_vs_i, stride_vd_i,
        stride_os_i, stride_od_i,
        stride_lse_s
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flash_attn_blackwell::run);

}  // namespace flash_attn_blackwell