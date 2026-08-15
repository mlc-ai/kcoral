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
// TMA & WGMMA Hardware Wrappers
// -------------------------------------------------------------------------

__device__ __forceinline__ uint64_t make_smem_desc_k_major_128B(void* ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    d |= (addr & 0x3FFFF) >> 4;
    d |= ((1ULL & 0x3FFFF) >> 4) << 16;
    d |= ((1024ULL & 0x3FFFF) >> 4) << 32;
    d |= (1ULL << 46);
    d |= (2ULL << 61);
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_mn_major_128B(void* ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    d |= (addr & 0x3FFFF) >> 4;
    d |= ((16384ULL & 0x3FFFF) >> 4) << 16; 
    d |= ((1024ULL & 0x3FFFF) >> 4) << 32;
    d |= (1ULL << 46);
    d |= (2ULL << 61);
    return d;
}

__device__ __forceinline__ uint64_t update_smem_addr(uint64_t desc, uint32_t addr) {
    return (desc & ~0x3FFFULL) | ((addr & 0x3FFFF) >> 4);
}

__device__ __forceinline__ void wgmma_16x16x16_fp16_cta(
    uint32_t row_A, uint32_t col_A, uint32_t row_B, uint32_t col_B,
    uint32_t row_D, float (&result)[2], uint32_t scale_D) {
    asm volatile("wgmma.m16n8k16.row_q0.col_q0.a16.b16.d16.f16.acc_f32\n"
                 "{ .shared::cluster %0, { %1, %2 }, { %3, %4 } }\n"
                 "sema.src=relaxed;\n"
                 :: "r"(row_D), "r"(row_A), "r"(col_A), "r"(row_B), "r"(col_B)
                 : : "memory");
}

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

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void store_128b(uint32_t addr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
}

__device__ __forceinline__ uint32_t pack_bf16(uint32_t a, uint32_t b) {
    __nv_bfloat162 res;
    res.x = __float2bfloat16(__uint_as_float(a));
    res.y = __float2bfloat16(__uint_as_float(b));
    return *reinterpret_cast<uint32_t*>(&res);
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

__global__ void __launch_bounds__(256, 2) attention_kernel(
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
    uint16_t* smem_P = (uint16_t*)(smem_buf + 98304);             // Reuses smem_S transiently (32 KB effectively overwritten)
    
    uint64_t* bar_Q = (uint64_t*)(smem_buf + 163840);
    uint64_t* bar_KV = (uint64_t*)(smem_buf + 163848);

    if (tid == 0) {
        init_smem_barrier_fn(bar_Q, 1);
        init_smem_barrier_fn(bar_KV, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    uint32_t phase_Q = 0;
    uint32_t phase_KV = 0;
    uint32_t c2 = h + b * num_heads;

    uint64_t desc_Q_0 = make_smem_desc_k_major_128B(smem_Q);
    uint64_t desc_Q_1 = make_smem_desc_k_major_128B(smem_Q + 8192);
    uint64_t desc_K_T_0 = make_smem_desc_k_major_128B(smem_K);
    uint64_t desc_K_T_1 = make_smem_desc_k_major_128B(smem_K + 8192);
    uint64_t desc_P_0 = make_smem_desc_k_major_128B(smem_P);
    uint64_t desc_P_1 = make_smem_desc_k_major_128B(smem_P + 8192);
    uint64_t desc_V_0 = make_smem_desc_mn_major_128B(smem_V);
    uint64_t desc_V_1 = make_smem_desc_mn_major_128B(smem_V + 8192);

    uint32_t saddr_Q = (uint32_t)__cvta_generic_to_shared(smem_Q);
    uint32_t saddr_K = (uint32_t)__cvta_generic_to_shared(smem_K);
    uint32_t saddr_V = (uint32_t)__cvta_generic_to_shared(smem_V);
    uint32_t saddr_P = (uint32_t)__cvta_generic_to_shared(smem_P);
    uint32_t saddr_S = (uint32_t)__cvta_generic_to_shared(smem_S);

    float denom = 1.0f / sqrtf(128.0f);

    // Thread mapping for conflict-free execution
    uint32_t row = (tid % 16) + ((tid / 16) % 4) * 4;
    uint32_t col = (tid % 16) / 2;
    uint32_t col_start = (tid / 64) * 8;

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
            
            fence_proxy_async_fn(); 
            
            float scale_D_0 = (k_start == 0 && q_start == 0) ? 0.0f : 1.0f;
            float scale_D_1 = (k_start == 0 && q_start == 0) ? 0.0f : 1.0f;

            for (uint32_t k = 0; k < 128; k += 16) {
                uint32_t k_byte = k * 2;
                uint64_t dQ0 = update_smem_addr(desc_Q_0, saddr_Q + k_byte);
                uint64_t dQT0 = update_smem_addr(desc_K_T_0, saddr_K + k_byte);
                
                for (uint32_t i = 0; i < 8; ++i) {
                    uint32_t row_i = i * 16;
                    float result_0[2], result_1[2];
                    
                    wgmma_16x16x16_fp16_cta((saddr_Q + k_byte) >> 1, (saddr_Q + k_byte) & 1, 
                                            (saddr_K + k_byte) >> 1, (saddr_K + k_byte) & 1, 
                                            saddr_S + (row_i + row) * 128 * 4, result_0, scale_D_0);
                    wgmma_16x16x16_fp16_cta((saddr_Q + k_byte) >> 1, (saddr_Q + k_byte) & 1, 
                                            (saddr_K + k_byte) >> 1, (saddr_K + k_byte) & 1, 
                                            saddr_S + (row_i + row) * 128 * 4 + 32, result_1, scale_D_1);
                }
            }
            
            float row_max_0 = -1e20f;
            float row_max_1 = -1e20f;
            float S_final_local_0[8] = {0};
            float S_final_local_1[8] = {0};

            for (uint32_t c = 0; c < 8; ++c) {
                uint32_t global_col = col_start + c;
                uint32_t idx_0 = (row * 128) + global_col;
                uint32_t idx_1 = (row * 128) + global_col + 4;
                
                float val_0 = smem_S[idx_0];
                float val_1 = smem_S[idx_1];
                
                if (include_in_softmax(q_start + row, k_start + global_col, S_len)) {
                    val_0 *= denom;
                    row_max_0 = fmaxf(row_max_0, val_0);
                }
                if (include_in_softmax(q_start + row, k_start + global_col + 4, S_len)) {
                    val_1 *= denom;
                    row_max_1 = fmaxf(row_max_1, val_1);
                }
                
                S_final_local_0[c] = val_0;
                S_final_local_1[c] = val_1;
            }
            
            float m_new_0 = fmaxf(m_val, row_max_0);
            float m_new_1 = fmaxf(m_val, row_max_1);
            float m_new = fmaxf(m_new_0, m_new_1);
            
            float exp_old = expf(m_val - m_new);
            if (m_new > m_val) {
                for(int i = 0; i < 128; ++i) {
                    out_regs_0[i] *= exp_old;
                    out_regs_1[i] *= exp_old;
                }
            }
            
            float row_sum = 0.0f;
            for (uint32_t c = 0; c < 8; ++c) {
                uint32_t global_col = col_start + c;
                
                float p_0 = 0.0f;
                if (include_in_softmax(q_start + row, k_start + global_col, S_len)) {
                    p_0 = expf(S_final_local_0[c] - m_new);
                    row_sum += p_0;
                }
                float p_1 = 0.0f;
                if (include_in_softmax(q_start + row, k_start + global_col + 4, S_len)) {
                    p_1 = expf(S_final_local_1[c] - m_new);
                    row_sum += p_1;
                }
                
                uint32_t p_col_0 = ((global_col >> 3) ^ (row & 7)) << 3 | (global_col & 7);
                uint32_t p_col_1 = (((global_col + 4) >> 3) ^ (row & 7)) << 3 | ((global_col + 4) & 7);
                
                uint32_t p_idx_0 = (row * 128) + p_col_0;
                uint32_t p_idx_1 = (row * 128) + p_col_1;
                
                *(uint16_t*)&smem_P[p_idx_0] = __float2bfloat16(p_0);
                *(uint16_t*)&smem_P[p_idx_1] = __float2bfloat16(p_1);
            }
            
            l_val = l_val * exp_old + row_sum;
            m_val = m_new;
            
            __syncthreads(); 
            
            for (uint32_t k = 0; k < 128; k += 16) {
                uint32_t k_byte = k * 2;
                uint64_t dP0 = update_smem_addr(desc_P_0, saddr_P + k_byte);
                uint64_t dV0 = update_smem_addr(desc_V_0, saddr_V + k_byte);
                
                for (uint32_t i = 0; i < 8; ++i) {
                    uint32_t row_i = i * 16;
                    wgmma_16x16x16_fp16_cta((saddr_P + k_byte) >> 1, (saddr_P + k_byte) & 1, 
                                            (saddr_V + k_byte) >> 1, (saddr_V + k_byte) & 1, 
                                            saddr_S + (row_i + row) * 128 * 4, out_regs_0, 1.0f);
                    wgmma_16x16x16_fp16_cta((saddr_P + k_byte) >> 1, (saddr_P + k_byte) & 1, 
                                            (saddr_V + k_byte) >> 1, (saddr_V + k_byte) & 1, 
                                            saddr_S + (row_i + row) * 128 * 4 + 32, out_regs_1, 1.0f);
                }
            }
        }
        
        if (l_val > 0.0f) {
            for(int i = 0; i < 128; ++i) {
                out_regs_0[i] /= l_val;
                out_regs_1[i] /= l_val;
            }
        }
        
        float* smem_O = (float*)smem_S; 
        
        for (int i = 0; i < 8; ++i) {
            uint32_t r0 = pack_bf16(__float_as_uint(out_regs_0[col_start + i]), __float_as_uint(out_regs_1[col_start + i]));
            uint32_t r1 = pack_bf16(__float_as_uint(out_regs_0[col_start + i + 4]), __float_as_uint(out_regs_1[col_start + i + 4]));
            *(uint32_t*)&smem_O[(row * 128) + col_start + i] = r0;
            *(uint32_t*)&smem_O[(row * 128) + col_start + i + 4] = r1;
        }
        
        __syncthreads();
        
        for (uint32_t i = tid; i < 16384; i += 256) {
            uint32_t row_o = i / 128;
            uint32_t col_o = i % 128;
            uint32_t q_idx = q_start + row_o;
            if (q_idx < S_len) {
                uint32_t b_idx = (b * num_heads + h) * S_len * 128 + q_idx * 128 + col_o;
                *(uint16_t*)&O[b_idx] = *(uint16_t*)&smem_O[i];
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
  dim3 block(256);
  
  // Request 192 KB dynamic shared memory space to safely hold our multi-stage pipeline elements.
  CUDA_CHECK(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 196608));
  
  attention_kernel<<<grid, block, 196608, stream>>>(tma_Q, tma_K, tma_V, 
                                                     static_cast<__nv_bfloat16*>(O.data_ptr()), 
                                                     static_cast<float*>(LSE.data_ptr()), 
                                                     S, H);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

} // namespace tvm_ffi_example_cuda