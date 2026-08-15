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

namespace tvm_ffi_example_cuda {

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
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

__device__ __forceinline__ bool include_in_softmax(uint32_t q_pos, uint32_t k_pos, uint32_t S_len) {
    return q_pos < S_len && q_pos >= k_pos;
}

__device__ __forceinline__ uint16_t swizzled_col_128B(uint16_t row, uint16_t col) {
    return (((col >> 3) ^ (row & 7)) << 3) + (col & 7);
}

__device__ __forceinline__ float2 read_tile_vec2(const uint16_t* tile, uint32_t row, uint32_t col) {
    uint32_t tile_idx = col / 64;
    uint32_t col_in_tile = col % 64;
    uint32_t sw_col = swizzled_col_128B(row, col_in_tile);
    return *(const float2*)&tile[tile_idx * 8192 + row * 64 + sw_col];
}

__device__ __forceinline__ void write_tile_vec2(uint16_t* tile, uint32_t row, uint32_t col, float2 val) {
    uint32_t tile_idx = col / 64;
    uint32_t col_in_tile = col % 64;
    uint32_t sw_col = swizzled_col_128B(row, col_in_tile);
    *(float2*)&tile[tile_idx * 8192 + row * 64 + sw_col] = val;
}

CUresult create_tma_4d_descriptor_2B(CUtensorMap* d, void* globalAddress, 
                                     uint64_t dim0, uint64_t dim1, uint64_t dim2, uint64_t dim3,
                                     uint32_t box0, uint32_t box1, uint32_t box2, uint32_t box3,
                                     CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[4] = {dim0, dim1, dim2, dim3};
    cuuint64_t globalStrides[3] = {dim0 * 2, dim0 * dim1 * 2, dim0 * dim1 * dim2 * 2};
    cuuint32_t boxDim[4] = {box0, box1, box2, box3};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, globalAddress, globalDim, globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

// -------------------------------------------------------------------------
// Attention Kernel
// -------------------------------------------------------------------------

__global__ void __launch_bounds__(128, 1) attention_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O, float* LSE,
    uint32_t S_len, uint32_t num_heads)
{
    uint32_t h = blockIdx.x;
    uint32_t b = blockIdx.y;
    uint32_t tid = threadIdx.x;

    extern __shared__ __align__(128) uint8_t smem_buf_raw[];
    
    uint16_t* smem_Q = (uint16_t*)smem_buf_raw;                     // [0 .. 32767] (32 KB)
    uint16_t* smem_K = (uint16_t*)(smem_buf_raw + 32768);            // [32768 .. 65535] (32 KB)
    uint16_t* smem_V = (uint16_t*)(smem_buf_raw + 65536);            // [65536 .. 98303] (32 KB)
    float* smem_S_fp32 = (float*)(smem_buf_raw + 98304);             // [98304 .. 163839] (64 KB)
    uint16_t* smem_P = (uint16_t*)(smem_buf_raw + 163840);           // [163840 .. 196607] (32 KB)
    uint16_t* smem_tmp = (uint16_t*)(smem_buf_raw + 196608);         // [196608 .. 229375] (32 KB)
    
    uint64_t* bar_Q = (uint64_t*)(smem_buf_raw + 229376);
    uint64_t* bar_KV = (uint64_t*)(smem_buf_raw + 229384);

    if (tid == 0) {
        init_smem_barrier_fn(bar_Q, 1);
        init_smem_barrier_fn(bar_KV, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    uint32_t phase_Q = 0;
    uint32_t phase_KV = 0;
    uint64_t offset_bh = (static_cast<uint64_t>(b) * num_heads + h) * S_len * 128;

    float denom = 1.0f / sqrtf(128.0f);

    for (uint32_t q_start = 0; q_start < S_len; q_start += 128) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_Q, 32768);
            tma_load_4d_fn(&tma_Q, bar_Q, smem_Q, 0, q_start, h, b);
            tma_load_4d_fn(&tma_Q, bar_Q, smem_Q + 8192, 64, q_start, h, b);
        }
        
        float m_val = -1e20f;
        float l_val = 0.0f;
        float out_reg_0[64] = {0};
        float out_reg_1[64] = {0};
        
        mbarrier_wait_fn(bar_Q, phase_Q);
        phase_Q ^= 1;
        
        uint32_t S_and_q_start = (q_start + 128 < S_len) ? (q_start + 128) : S_len;
        
        for (uint32_t k_start = 0; k_start < S_and_q_start; k_start += 128) {
            if (tid == 0) {
                mbarrier_arrive_and_expect_tx_fn(bar_KV, 65536);
                tma_load_4d_fn(&tma_K, bar_KV, smem_K, 0, k_start, h, b);
                tma_load_4d_fn(&tma_K, bar_KV, smem_K + 8192, 64, k_start, h, b);
                tma_load_4d_fn(&tma_V, bar_KV, smem_V, 0, k_start, h, b);
                tma_load_4d_fn(&tma_V, bar_KV, smem_V + 8192, 64, k_start, h, b);
            }
            mbarrier_wait_fn(bar_KV, phase_KV);
            phase_KV ^= 1;
            
            // Transpose K to K_T utilizing full warp to avoid uncoalesced SMEM operations
            for (uint32_t i = tid; i < 16384; i += 128) {
                uint32_t row = i >> 7;
                uint32_t col = i & 127;
                
                uint16_t val = read_tile_vec2(smem_K, row, col).x;
                
                uint32_t sw_col_T = swizzled_col_128B(col, row);
                uint32_t tile_idx_T = row / 64;
                uint32_t row_in_tile_T = row % 64;
                smem_tmp[tile_idx_T * 8192 + col * 64 + sw_col_T] = val;
            }
            __syncthreads();
            
            // Maintain an exclusively zero-initialized accumulator
            for (uint32_t i = tid; i < 16384; i += 128) {
                smem_S_fp32[i] = 0.0f;
            }
            __syncthreads();
            
            // Compute Q @ K_T
            for (int k = 0; k < 128; k += 2) {
                float2 q_vec = read_tile_vec2(smem_Q, tid, k);
                
                for (int n = 0; n < 128; n += 2) {
                    float2 k_vec = read_tile_vec2(smem_tmp, k, n);
                    
                    smem_S_fp32[tid * 128 + n] += q_vec.x * k_vec.x + q_vec.y * k_vec.y;
                }
            }
            __syncthreads();
            
            float row_max = -1e20f;
            uint32_t q_pos = q_start + tid;
            
            for (int n = 0; n < 128; ++n) {
                uint32_t k_pos = k_start + n;
                if (include_in_softmax(q_pos, k_pos, S_len)) {
                    float val = smem_S_fp32[tid * 128 + n] * denom;
                    smem_S_fp32[tid * 128 + n] = val;
                    row_max = fmaxf(row_max, val);
                } else {
                    smem_S_fp32[tid * 128 + n] = -1e20f;
                }
            }
            
            float m_new = fmaxf(m_val, row_max);
            float exp_old = (m_new > m_val) ? expf(m_val - m_new) : 1.0f;
            
            if (m_new > m_val) {
                for(int i = 0; i < 64; ++i) {
                    out_reg_0[i] *= exp_old;
                    out_reg_1[i] *= exp_old;
                }
            }
            
            float row_sum = 0.0f;
            for (int n = 0; n < 128; n += 2) {
                uint32_t k_pos = k_start + n;
                float p0 = 0.0f, p1 = 0.0f;
                if (include_in_softmax(q_pos, k_pos, S_len)) {
                    p0 = expf(smem_S_fp32[tid * 128 + n] - m_new);
                    row_sum += p0;
                }
                if (include_in_softmax(q_pos, k_pos + 1, S_len)) {
                    p1 = expf(smem_S_fp32[tid * 128 + n + 1] - m_new);
                    row_sum += p1;
                }
                
                float2 p_packed;
                p_packed.x = __float2bfloat16(p0);
                p_packed.y = __float2bfloat16(p1);
                
                write_tile_vec2(smem_P, tid, n, p_packed);
            }
            
            l_val = l_val * exp_old + row_sum;
            m_val = m_new;
            
            __syncthreads(); 
            
            // Compute P @ V
            for (int k = 0; k < 128; k += 2) {
                float2 p_vec = read_tile_vec2(smem_P, tid, k);
                
                for (int n = 0; n < 128; n += 2) {
                    float2 v_vec_0 = read_tile_vec2(smem_V, k, n);
                    float2 v_vec_1 = read_tile_vec2(smem_V, k + 1, n);
                    
                    out_reg_0[n/2] += p_vec.x * v_vec_0.x + p_vec.y * v_vec_1.x;
                    out_reg_1[n/2] += p_vec.x * v_vec_0.y + p_vec.y * v_vec_1.y;
                }
            }
        }
        
        if (l_val > 0.0f) {
            for(int i = 0; i < 64; ++i) {
                out_reg_0[i] /= l_val;
                out_reg_1[i] /= l_val;
            }
        }
        
        // Utilize the previously transposed temporary buffer space for linear output staging writes.
        uint16_t* smem_O = smem_tmp; 
        
        for (int i = 0; i < 64; ++i) {
            uint32_t r0 = ((uint32_t)__float_as_uint(__float2bfloat16(out_reg_1[i])) << 16) | __float_as_uint(__float2bfloat16(out_reg_0[i]));
            *(uint32_t*)&smem_O[tid * 128 + (i / 4) * 2] = r0;
        }
        __syncthreads();
        
        for (uint32_t row = 0; row < 128; ++row) {
            uint32_t q_idx = q_start + row;
            if (q_idx < S_len) {
                for (uint32_t col = tid; col < 128; col += 128) {
                    uint64_t b_idx = offset_bh + (uint64_t)q_idx * 128 + col;
                    *(uint16_t*)&O[b_idx] = smem_O[row * 128 + col];
                }
            }
        }
        
        if (tid < 128) {
            uint32_t q_idx = q_start + tid;
            if (q_idx < S_len) {
                uint32_t lse_idx = (b * num_heads + h) * S_len + q_idx;
                if (l_val > 0.0f) {
                    LSE[lse_idx] = m_val + logf(l_val);
                } else {
                    LSE[lse_idx] = -1e20f;
                }
            }
        }
        __syncthreads();
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  
  uint32_t B = Q.size(0);
  uint32_t H = Q.size(1);
  uint32_t S = Q.size(2);
  uint32_t D = Q.size(3); 
  
  CUtensorMap tma_Q, tma_K, tma_V;
  create_tma_4d_descriptor_2B(&tma_Q, Q.data_ptr(), D, S, H, B, 64, 128, 1, 1, CU_TENSOR_MAP_SWIZZLE_128B);
  create_tma_4d_descriptor_2B(&tma_K, K.data_ptr(), D, S, H, B, 64, 128, 1, 1, CU_TENSOR_MAP_SWIZZLE_128B);
  create_tma_4d_descriptor_2B(&tma_V, V.data_ptr(), D, S, H, B, 64, 128, 1, 1, CU_TENSOR_MAP_SWIZZLE_128B);

  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
  
  dim3 grid(H, B);
  dim3 block(128);
  
  CUDA_CHECK(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 229376));
  
  attention_kernel<<<grid, block, 229376, stream>>>(tma_Q, tma_K, tma_V, 
                                                     static_cast<__nv_bfloat16*>(O.data_ptr()), 
                                                     static_cast<float*>(LSE.data_ptr()), 
                                                     S, H);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

} // namespace tvm_ffi_example_cuda