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

// -------------------------------------------------------------------------
// TMA Hardware Wrappers
// -------------------------------------------------------------------------

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
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

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, 
                                     uint64_t dim0, uint64_t dim1, uint64_t dim2,
                                     uint32_t box0, uint32_t box1, uint32_t box2,
                                     CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[3] = {dim0, dim1, dim2};
    cuuint64_t globalStrides[2] = {dim0 * 2, dim0 * dim1 * 2};
    cuuint32_t boxDim[3] = {box0, box1, box2};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, globalAddress, globalDim, globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
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
    uint8_t* smem_buf = (uint8_t*)(((uintptr_t)smem_buf_raw + 1023) & ~1023); 
    
    uint16_t* smem_Q = (uint16_t*)smem_buf;                       // [0 .. 32767] (32 KB)
    uint16_t* smem_K = (uint16_t*)(smem_buf + 32768);             // [32768 .. 65535] (32 KB)
    uint16_t* smem_V = (uint16_t*)(smem_buf + 65536);             // [65536 .. 98303] (32 KB)
    float* smem_S = (float*)(smem_buf + 98304);                   // [98304 .. 163839] (64 KB)
    uint16_t* smem_P = (uint16_t*)(smem_buf + 163840);            // [163840 .. 196607] (32 KB)
    
    uint64_t* bar_Q = (uint64_t*)(smem_buf + 196608);
    uint64_t* bar_KV = (uint64_t*)(smem_buf + 196616);

    if (tid == 0) {
        init_smem_barrier_fn(bar_Q, 1);
        init_smem_barrier_fn(bar_KV, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    uint32_t phase_Q = 0;
    uint32_t phase_KV = 0;
    uint32_t c2 = h + b * num_heads;
    uint64_t offset_bh = (static_cast<uint64_t>(b) * num_heads + h) * S_len * 128;

    float denom = 1.0f / sqrtf(128.0f);

    for (uint32_t q_start = 0; q_start < S_len; q_start += 128) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_Q, 32768);
            tma_load_3d_fn(&tma_Q, bar_Q, smem_Q, 0, q_start, c2);
            tma_load_3d_fn(&tma_Q, bar_Q, smem_Q + 8192, 64, q_start, c2);
        }
        
        float m_val = -1e20f;
        float l_val = 0.0f;
        float out_regs_0[128] = {0};
        float out_regs_1[128] = {0};
        
        mbarrier_wait_fn(bar_Q, phase_Q);
        phase_Q ^= 1;
        
        uint32_t S_and_q_start = (q_start + 128 < S_len) ? (q_start + 128) : S_len;
        
        for (uint32_t k_start = 0; k_start < S_and_q_start; k_start += 128) {
            if (tid == 0) {
                mbarrier_arrive_and_expect_tx_fn(bar_KV, 65536);
                tma_load_3d_fn(&tma_K, bar_KV, smem_K, 0, k_start, c2);
                tma_load_3d_fn(&tma_K, bar_KV, smem_K + 8192, 64, k_start, c2);
                tma_load_3d_fn(&tma_V, bar_KV, smem_V, 0, k_start, c2);
                tma_load_3d_fn(&tma_V, bar_KV, smem_V + 8192, 64, k_start, c2);
            }
            mbarrier_wait_fn(bar_KV, phase_KV);
            phase_KV ^= 1;
            
            // Transpose K to K_T
            for (uint32_t i = tid; i < 16384; i += 128) {
                uint32_t row = i >> 7;
                uint32_t col = i & 127;
                uint32_t sw_col = (((col) >> 3) ^ (row & 7)) << 3 | (col & 7);
                uint16_t val = smem_K[row * 128 + sw_col];
                
                uint32_t sw_col_T = (((row) >> 3) ^ (col & 7)) << 3 | (row & 7);
                uint16_t* smem_K_T = (uint16_t*)(smem_buf + 163840); // Reuses smem_P initially
                smem_K_T[col * 128 + sw_col_T] = val;
            }
            __syncthreads();
            
            // Clear S accumulator
            for (uint32_t i = tid; i < 16384; i += 128) {
                smem_S[i] = 0.0f;
            }
            __syncthreads();
            
            // Compute Q @ K_T
            for (int k = 0; k < 128; k += 2) {
                float2 q_vec;
                uint32_t sw_col_q = (((k) >> 3) ^ (tid & 7)) << 3 | (k & 7);
                *(uint32_t*)&q_vec = *(const uint32_t*)&smem_Q[tid * 128 + sw_col_q];
                
                for (int n = 0; n < 128; n += 2) {
                    float2 k_vec;
                    uint32_t* smem_K_T_ptr = (uint32_t*)&smem_K_T[k * 128];
                    uint32_t sw_col_k = (((n) >> 3) ^ (k & 7)) << 3 | (n & 7);
                    *(uint32_t*)&k_vec = smem_K_T_ptr[sw_col_k / 2];
                    
                    smem_S[tid * 128 + n] += q_vec.x * k_vec.x + q_vec.y * k_vec.y;
                }
            }
            __syncthreads();
            
            float row_max = -1e20f;
            uint32_t q_pos = q_start + tid;
            
            for (int n = 0; n < 128; ++n) {
                uint32_t k_pos = k_start + n;
                if (include_in_softmax(q_pos, k_pos, S_len)) {
                    float val = smem_S[tid * 128 + n] * denom;
                    smem_S[tid * 128 + n] = val;
                    row_max = fmaxf(row_max, val);
                } else {
                    smem_S[tid * 128 + n] = -1e20f;
                }
            }
            
            float m_new = fmaxf(m_val, row_max);
            float exp_old = (m_new > m_val) ? expf(m_val - m_new) : 1.0f;
            
            if (m_new > m_val) {
                for(int i = 0; i < 128; ++i) {
                    out_regs_0[i] *= exp_old;
                    out_regs_1[i] *= exp_old;
                }
            }
            
            float row_sum = 0.0f;
            for (int n = 0; n < 128; n += 2) {
                uint32_t k_pos = k_start + n;
                float p0 = 0.0f, p1 = 0.0f;
                if (include_in_softmax(q_pos, k_pos, S_len)) {
                    p0 = expf(smem_S[tid * 128 + n] - m_new);
                    row_sum += p0;
                }
                if (include_in_softmax(q_pos, k_pos + 1, S_len)) {
                    p1 = expf(smem_S[tid * 128 + n + 1] - m_new);
                    row_sum += p1;
                }
                
                uint32_t sw_col = (((n) >> 3) ^ (tid & 7)) << 3 | (n & 7);
                uint32_t p_packed = ((uint32_t)__float_as_uint(__float2bfloat16(p1)) << 16) | __float_as_uint(__float2bfloat16(p0));
                *(uint32_t*)&smem_P[tid * 128 + sw_col] = p_packed;
            }
            
            l_val = l_val * exp_old + row_sum;
            m_val = m_new;
            
            __syncthreads(); 
            
            // Compute P @ V
            for (int k = 0; k < 128; k += 2) {
                float2 p_vec;
                uint32_t sw_col_p = (((k) >> 3) ^ (tid & 7)) << 3 | (k & 7);
                *(uint32_t*)&p_vec = *(const uint32_t*)&smem_P[tid * 128 + sw_col_p];
                
                for (int n = 0; n < 128; n += 2) {
                    float2 v_vec_0;
                    uint32_t sw_col_v0 = (((n) >> 3) ^ (k & 7)) << 3 | (n & 7);
                    *(uint32_t*)&v_vec_0 = *(const uint32_t*)&smem_V[k * 128 + sw_col_v0];
                    
                    float2 v_vec_1;
                    uint32_t sw_col_v1 = (((n) >> 3) ^ ((k+1) & 7)) << 3 | (n & 7);
                    *(uint32_t*)&v_vec_1 = *(const uint32_t*)&smem_V[(k + 1) * 128 + sw_col_v1];
                    
                    out_regs_0[n/2] += p_vec.x * v_vec_0.x + p_vec.y * v_vec_1.x;
                    out_regs_1[n/2] += p_vec.x * v_vec_0.y + p_vec.y * v_vec_1.y;
                }
            }
        }
        
        if (l_val > 0.0f) {
            for(int i = 0; i < 128; ++i) {
                out_regs_0[i] /= l_val;
                out_regs_1[i] /= l_val;
            }
        }
        
        // Use non-swizzled layout for coalesced output writes.
        uint16_t* smem_O = (uint16_t*)(smem_buf + 163840); 
        
        for (int i = 0; i < 64; ++i) {
            uint32_t col = i * 2;
            *(uint32_t*)&smem_O[tid * 128 + col] = ((uint32_t)__float_as_uint(__float2bfloat16(out_regs_1[i])) << 16) | __float_as_uint(__float2bfloat16(out_regs_0[i]));
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
  create_tma_3d_descriptor_2B(&tma_Q, Q.data_ptr(), D, S, B * H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B);
  create_tma_3d_descriptor_2B(&tma_K, K.data_ptr(), D, S, B * H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B);
  create_tma_3d_descriptor_2B(&tma_V, V.data_ptr(), D, S, B * H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B);

  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
  
  dim3 grid(H, B);
  dim3 block(128);
  
  CUDA_CHECK(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 200000));
  
  attention_kernel<<<grid, block, 200000, stream>>>(tma_Q, tma_K, tma_V, 
                                                     static_cast<__nv_bfloat16*>(O.data_ptr()), 
                                                     static_cast<float*>(LSE.data_ptr()), 
                                                     S, H);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

} // namespace tvm_ffi_example_cuda