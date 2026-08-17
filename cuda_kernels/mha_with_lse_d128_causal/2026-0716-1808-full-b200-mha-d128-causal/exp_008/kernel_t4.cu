#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
#include <mma.h>
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

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        fprintf(stderr, "CU error %d at %s:%d\n",                 \
                (int)_e, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_causal_attention {

// -------------------------------------------------------------------------------------
// Helper Functions for SM100
// -------------------------------------------------------------------------------------

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase));
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void barrier_cta_sync() {
    asm volatile("barrier.sync.aligned 0, 128;" ::: "memory");
}

// -------------------------------------------------------------------------------------
// WGMMA Descriptors (Explicitly SWIZZLE_NONE)
// -------------------------------------------------------------------------------------

__device__ __forceinline__ uint64_t m168168_desc_row_major_swizzle_none(void* ptr) {
    uintptr_t base = (uintptr_t)ptr;
    uint64_t d = (base & 0x3FFFF) >> 2;
    d |= (1ULL << 48);
    return d | ((d & 0x3FFULL) << 16);
}

__device__ __forceinline__ uint64_t n168168_desc_col_major_swizzle_none(void* ptr) {
    uintptr_t base = (uintptr_t)ptr;
    uint64_t d = (base & 0x3FFFF) >> 2;
    return d | ((d & 0x3FFULL) << 16);
}

__device__ __forceinline__ uint64_t n168168_desc_row_major_swizzle_none(void* ptr) {
    uintptr_t base = (uintptr_t)ptr;
    uint64_t d = (base & 0x3FFFF) >> 2;
    d |= (1ULL << 48);
    return d | ((d & 0x3FFULL) << 16);
}

// -------------------------------------------------------------------------------------
// Causal Attention Kernel
// -------------------------------------------------------------------------------------

__global__ void causal_attention_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S_len, int B, int H)
{
    extern __shared__ char smem_raw[];
    size_t smem_addr = (size_t)smem_raw;
    smem_addr = (smem_addr + 1023) & ~1023;
    char* smem = (char*)smem_addr;

    // Memory Layout Planning & Allocation 
    // 1. s_S_and_Q: Union layout. Initially acts as Q (128x128 bf16 = 32KB). After QK GEMM completes, overwritten with S (128x128 fp32 = 64KB).
    // 2. s_K_flat: 2 phases x 32KB = 64KB
    // 3. s_V_flat: 2 phases x 32KB = 64KB
    // 4. s_P_bf16: 32KB
    // 5. s_O_fp32: 64KB
    // 6. Meta arrays (max, sum, prev): 1.5KB
    // Total: 263680 Bytes
    
    __nv_bfloat16* s_Q_flat = (__nv_bfloat16*)smem;                   // 32768 Bytes effectively utilized initially
    __nv_bfloat16* s_K_flat = (__nv_bfloat16*)(smem + 32768);         // 65536 Bytes
    __nv_bfloat16* s_V_flat = (__nv_bfloat16*)(smem + 98304);         // 65536 Bytes
    __nv_bfloat16* s_P_bf16 = (__nv_bfloat16*)(smem + 163840);        // 32768 Bytes
    float* s_O_fp32         = (float*)(smem + 196608);                // 65536 Bytes
    
    float* s_row_max = (float*)(smem + 262144);                       
    float* s_row_sum = (float*)(smem + 262656);                       
    float* s_prev_max = (float*)(smem + 263168);                      
    
    // Allocate remaining memory cleanly and completely for MBarriers strictly above the 263680 Byte mark.
    char* barrier_smem = smem + 263680;
    barrier_smem = (char*)(((size_t)barrier_smem + 127) & ~127);
    uint64_t* mbar_Q = (uint64_t*)barrier_smem;
    uint64_t* mbar_K = (uint64_t*)(barrier_smem + 8);
    uint64_t* mbar_V = (uint64_t*)(barrier_smem + 24);

    int bh = blockIdx.x * gridDim.y + blockIdx.y;
    int batch_idx = bh / H;
    int head_idx = bh % H;
    int bid = blockIdx.z;
    int q_blk = bid * 128;
    int lane_id = threadIdx.x % 32;
    
    float* LSE_ptr = LSE + batch_idx * H * S_len + head_idx * S_len;
    __nv_bfloat16* O_ptr = O + batch_idx * H * S_len * 128 + head_idx * S_len * 128;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(&mbar_K[0], 1);
        init_smem_barrier_fn(&mbar_K[1], 1);
        init_smem_barrier_fn(&mbar_V[0], 1);
        init_smem_barrier_fn(&mbar_V[1], 1);
        
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 32768); 
        
        tma_load_2d_fn(&tma_Q, mbar_Q, s_Q_flat, 0, bh * S_len + q_blk);
        tma_load_2d_fn(&tma_Q, mbar_Q, s_Q_flat + 8192, 64, bh * S_len + q_blk);
    }
    
    __syncthreads();
    mbarrier_wait_fn(mbar_Q, 0);
    fence_proxy_async_fn();
    
    float prev_max_val = -1e20f;
    float prev_sum = 0.0f;
    
    int num_blocks = (S_len + 127) / 128;
    int max_j = min(num_blocks, (q_blk / 128) + 1);
    
    // Initialize persistent Output accumulator in registers (O_permanent) mapped logically over full D=128 sequentially.
    wgmma::fragment<wgmma::accumulator, 128, 128, 16, float> acc_O[8];
    for(int i = 0; i < 8; ++i) wgmma::fill_fragment(acc_O[i], 0.0f);

    for (int j = 0; j < max_j; ++j) {
        int phase = j % 2;
        int next_phase = (j + 1) % 2;
        
        __nv_bfloat16* cur_K = &s_K_flat[phase * 16384];
        __nv_bfloat16* cur_V = &s_V_flat[phase * 16384];
        __nv_bfloat16* nxt_K = &s_K_flat[next_phase * 16384];
        __nv_bfloat16* nxt_V = &s_V_flat[next_phase * 16384];
        
        int k_blk = j * 128;
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_K[phase], 32768);
            tma_load_2d_fn(&tma_K, &mbar_K[phase], cur_K, 0, bh * S_len + k_blk);
            tma_load_2d_fn(&tma_K, &mbar_K[phase], cur_K + 8192, 64, bh * S_len + k_blk);
            
            mbarrier_arrive_and_expect_tx_fn(&mbar_V[phase], 32768);
            tma_load_2d_fn(&tma_V, &mbar_V[phase], cur_V, 0, bh * S_len + k_blk);
            tma_load_2d_fn(&tma_V, &mbar_V[phase], cur_V + 8192, 64, bh * S_len + k_blk);
            
            if (j + 1 < max_j) {
                int next_k_blk = (j + 1) * 128;
                mbarrier_arrive_and_expect_tx_fn(&mbar_K[next_phase], 32768);
                tma_load_2d_fn(&tma_K, &mbar_K[next_phase], nxt_K, 0, bh * S_len + next_k_blk);
                tma_load_2d_fn(&tma_K, &mbar_K[next_phase], nxt_K + 8192, 64, bh * S_len + next_k_blk);
                
                mbarrier_arrive_and_expect_tx_fn(&mbar_V[next_phase], 32768);
                tma_load_2d_fn(&tma_V, &mbar_V[next_phase], nxt_V, 0, bh * S_len + next_k_blk);
                tma_load_2d_fn(&tma_V, &mbar_V[next_phase], nxt_V + 8192, 64, bh * S_len + next_k_blk);
            }
        }
        
        __syncthreads(); 
        mbarrier_wait_fn(&mbar_K[phase], (j / 2) % 2);
        mbarrier_wait_fn(&mbar_V[phase], (j / 2) % 2);
        fence_proxy_async_fn();
        
        // Local accumulator for current block scoring 
        wgmma::fragment<wgmma::accumulator, 128, 128, 16, float> acc_S_local;
        wgmma::fill_fragment(acc_S_local, 0.0f);
        
        for (uint32_t k_step = 0; k_step < 128; k_step += 16) {
            uint64_t desc_A_Q = m168168_desc_row_major_swizzle_none(s_Q_flat + k_step * 2);
            uint64_t desc_B_K = n168168_desc_col_major_swizzle_none(cur_K + k_step * 2);
            uint32_t idesc_QK = (1u << 4) | (1u << 7) | (1u << 10) | ((128 / 8) << 17) | ((128 / 16) << 24);
            
            wgmma::mma_sync<uint16_t, 128, 128>(acc_S_local, desc_A_Q, desc_B_K, idesc_QK);
        }
        
        // Reuse precisely fully exhausted dead Q space dynamically for local peak extraction reduction temporarily.
        wgmma::store_matrix_sync(s_O_fp32, acc_S_local, 128, true);
        __syncthreads();
        
        float curr_max = -1e20f;
        float curr_sum = 0.0f;
        
        int row = lane_id;
        float max_val = -1e20f;
        
        // Extracted safely row-by-row leveraging fully unrolled vector math directly out of coalesced FMAs bounding limits safely (causal + oob masking)
        for (int col = 0; col < 128; col += 4) {
            int q_idx = q_blk + row;
            int k_idx = k_blk + col;
            
            float4 val = *(float4*)(&s_O_fp32[row * 128 + col]);
            if (k_idx >= S_len || k_idx > q_idx) {
                val.x = -1e20f; val.y = -1e20f; val.z = -1e20f; val.w = -1e20f;
            } else {
                val.x /= sqrtf(128.0f);
                val.y /= sqrtf(128.0f);
                val.z /= sqrtf(128.0f);
                val.w /= sqrtf(128.0f);
            }
            max_val = fmaxf(fmaxf(fmaxf(max_val, val.x), val.y), fmaxf(val.z, val.w));
            *(float4*)(&s_O_fp32[row * 128 + col]) = val; // Retain unscaled math temporaries concisely inline momentarily
        }
        
        max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, 16));
        max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, 8));
        max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, 4));
        max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, 2));
        max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, 1));
        
        curr_max = fmaxf(prev_max_val, max_val);
        float scale = expf(prev_max_val - curr_max);
        
        float sum_val = 0.0f;
        for (int col = 0; col < 128; col += 4) {
            int q_idx = q_blk + row;
            int k_idx = k_blk + col;
            
            float4 val = *(float4*)(&s_O_fp32[row * 128 + col]);
            if (k_idx < S_len && k_idx <= q_idx) {
                float exp_val = expf(val.x - curr_max);
                sum_val += exp_val;
                val.x = exp_val;
                
                exp_val = expf(val.y - curr_max);
                sum_val += exp_val;
                val.y = exp_val;
                
                exp_val = expf(val.z - curr_max);
                sum_val += exp_val;
                val.z = exp_val;
                
                exp_val = expf(val.w - curr_max);
                sum_val += exp_val;
                val.w = exp_val;
            } else {
                val.x = 0.0f;
                val.y = 0.0f;
                val.z = 0.0f;
                val.w = 0.0f;
            }
            *(float4*)(&s_O_fp32[row * 128 + col]) = val; // Retain pure post-softmax states perfectly aligned 
        }
        
        sum_val += __shfl_xor_sync(0xFFFFFFFF, sum_val, 16);
        sum_val += __shfl_xor_sync(0xFFFFFFFF, sum_val, 8);
        sum_val += __shfl_xor_sync(0xFFFFFFFF, sum_val, 4);
        sum_val += __shfl_xor_sync(0xFFFFFFFF, sum_val, 2);
        sum_val += __shfl_xor_sync(0xFFFFFFFF, sum_val, 1);
        
        curr_sum = prev_sum * scale + sum_val;
        
        if (threadIdx.x == 0) {
            s_prev_max[row] = curr_max;
            s_row_sum[row] = curr_sum;
        }
        
        // Critical final step mapping: Apply final uniform softmax division directly and pack cleanly into native 128B chunk swizzles 
        for (int col = 0; col < 128; col += 4) {
            float4 val = *(float4*)(&s_O_fp32[row * 128 + col]);
            float safe_sum = curr_sum < 1e-20f ? 1e-20f : curr_sum;
            
            val.x /= safe_sum;
            val.y /= safe_sum;
            val.z /= safe_sum;
            val.w /= safe_sum;
            
            int col_vec = col / 4;
            int swizzled_col_vec = (col_vec / 2) ^ (row % 8);
            int swizzled_col = swizzled_col_vec * 4 + (col % 4);
            int swizzled_idx = row * 128 + swizzled_col;
            
            __nv_bfloat16 bf_val[4];
            bf_val[0] = __float2bfloat16(val.x);
            bf_val[1] = __float2bfloat16(val.y);
            bf_val[2] = __float2bfloat16(val.z);
            bf_val[3] = __float2bfloat16(val.w);
            
            *(float4*)(&s_P_bf16[swizzled_idx]) = *(float4*)bf_val; 
        }
        
        __syncthreads(); 
        
        // Utilize highly specialized optimized WGMMA `pack` intrinsic modes for P * V multiplication. 
        // Ensures seamless hardware acceleration consumption mapping identically matching TMA parity layouts.
        for (uint32_t k_step = 0; k_step < 128; k_step += 16) {
            uint64_t desc_A_P = m168168_desc_row_major_swizzle_none(s_P_bf16 + k_step * 2);
            uint64_t desc_B_V = n168168_desc_row_major_swizzle_none(cur_V + k_step * 1024);
            uint32_t idesc_P_V = (1u << 4) | (1u << 7) | (1u << 10) | 
                                 (1u << 15) | (1u << 16) | 
                                 ((128 / 8) << 17) | ((128 / 16) << 24);
            
            for(int i = 0; i < 8; ++i) {
                wgmma::mma_sync<uint16_t, 128, 128>(acc_O[i], desc_A_P, desc_B_V, idesc_P_V);
            }
        }
        
        __syncthreads(); 
        
        prev_max_val = curr_max;
        prev_sum = curr_sum;
        float safe_final_sum = curr_sum < 1e-20f ? 1e-20f : curr_sum;
        float lse_val = curr_max + logf(safe_final_sum);
        
        if (q_blk + lane_id < S_len) {
            LSE_ptr[q_blk + lane_id] = lse_val;
        }
    }
    
    __syncthreads(); 
    
    // Store acc_O to s_O_fp32 utilizing identical, pristine 128x128 formatting structurally mapped cohesively.
    for(int i = 0; i < 8; ++i) {
        wgmma::store_matrix_sync(&s_O_fp32[i * 16384], acc_O[i], 128, true);
    }
    __syncthreads();
    
    // Rapid, fully-vectorized coalesced epilogue. 
    for (int i = 0; i < 128 * 128; i += 4) {
        int row = i / 128;
        int col = i % 128;
        
        if (q_blk + row < S_len) {
            __nv_bfloat16 out[4];
            out[0] = __float2bfloat16(s_O_fp32[i + 0]);
            out[1] = __float2bfloat16(s_O_fp32[i + 1]);
            out[2] = __float2bfloat16(s_O_fp32[i + 2]);
            out[3] = __float2bfloat16(s_O_fp32[i + 3]);
            
            *(float4*)(&O_ptr[(q_blk + row) * 128 + col]) = *(float4*)out;
        }
    }
}

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2, 
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

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  
  int64_t B = Q.size(0);
  int64_t H = Q.size(1);
  int64_t S_len = Q.size(2);
  int64_t D = Q.size(3);
  
  if (D != 128) {
    fprintf(stderr, "Error: D must be 128\n");
    exit(1);
  }
  
  const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
  
  __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
  float* LSE_data = static_cast<float*>(LSE.data_ptr());
  
  CUtensorMap tma_Q, tma_K, tma_V;
  
  CU_CHECK(create_tma_2d_descriptor_2B(
      &tma_Q, (void*)Q_data, 128, B * H * S_len, 64, 128,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  ));
  CU_CHECK(create_tma_2d_descriptor_2B(
      &tma_K, (void*)K_data, 128, B * H * S_len, 64, 128,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  ));
  CU_CHECK(create_tma_2d_descriptor_2B(
      &tma_V, (void*)V_data, 128, B * H * S_len, 64, 128,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  ));
  
  int threads = 128;
  dim3 grid(B, H, (S_len + 127) / 128);
  
  int smem_size = 263680; 
  cudaFuncSetAttribute(causal_attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  
  causal_attention_kernel<<<grid, threads, smem_size, static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id))>>>(
      tma_Q, tma_K, tma_V, O_data, LSE_data, S_len, B, H);
      
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id))));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_causal_attention::run);

} // namespace tvm_ffi_causal_attention