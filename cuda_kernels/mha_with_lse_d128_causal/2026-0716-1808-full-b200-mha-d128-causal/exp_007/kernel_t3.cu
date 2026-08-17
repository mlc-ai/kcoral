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
// TMA, MBarrier & Math Wrappers
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

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_k_major(void* ptr) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    uint64_t d = 0;
    d |= (addr & 0x3FFFF) >> 4;
    d |= ((1ULL & 0x3FFFF) >> 4) << 16;
    d |= ((1024ULL & 0x3FFFF) >> 4) << 32;
    d |= (1ULL << 46);
    uint32_t base_offset = (addr >> 7) & 0x7;
    d |= (base_offset << 49);
    d |= (2ULL << 61);
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_mn_major(void* ptr) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    uint64_t d = 0;
    d |= (addr & 0x3FFFF) >> 4;
    d |= ((8192ULL & 0x3FFFF) >> 4) << 16; 
    d |= ((1024ULL & 0x3FFFF) >> 4) << 32;
    d |= (1ULL << 46);
    uint32_t base_offset = (addr >> 7) & 0x7;
    d |= (base_offset << 49);
    d |= (2ULL << 61);
    return d;
}

__device__ __forceinline__ uint64_t update_smem_addr(uint64_t desc, uint32_t addr) {
    return (desc & ~0x3FFFULL) | ((addr & 0x3FFFF) >> 4);
}

__device__ __forceinline__ void umma_f16_cg1_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_1sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.shared::cta.b64"
        " [%0];"
        :: "r"(a));
}

__device__ __forceinline__ uint32_t make_instr_desc_fn_c1(uint32_t M, uint32_t N, uint32_t a_major, uint32_t b_major) {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= (a_major << 15);
    d |= (b_major << 16);
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
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

__global__ void __launch_bounds__(128, 2) attention_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O, float* LSE,
    uint32_t S_len, uint32_t num_heads)
{
    uint32_t h = blockIdx.x;
    uint32_t b = blockIdx.y;
    uint32_t tid = threadIdx.x;

    extern __shared__ __align__(128) uint8_t smem_pool[];
    
    uint16_t* smem_Q = (uint16_t*)smem_pool;                        // [0 .. 32767] (32 KB)
    uint16_t* smem_K = (uint16_t*)(smem_pool + 32768);              // [32768 .. 49151] (16 KB)
    uint16_t* smem_V = (uint16_t*)(smem_pool + 49152);              // [49152 .. 65535] (16 KB)
    uint16_t* smem_P = (uint16_t*)(smem_pool + 65536);              // [65536 .. 81919] (16 KB)
    
    uint64_t* bar_Q = (uint64_t*)(smem_pool + 81920);               // 8 Bytes
    uint64_t* bar_KV = (uint64_t*)(smem_pool + 81928);              // 8 Bytes
    uint32_t* tmem_ptr = (uint32_t*)(smem_pool + 81936);            // 20 Bytes

    if (tid == 0) {
        init_smem_barrier_fn(bar_Q, 1);
        init_smem_barrier_fn(bar_KV, 1);
        
        tmem_alloc_fn(&tmem_ptr[0], 128);
        tmem_alloc_fn(&tmem_ptr[1], 64);
        tmem_alloc_fn(&tmem_ptr[2], 64);
        tmem_alloc_fn(&tmem_ptr[3], 64);
        tmem_alloc_fn(&tmem_ptr[4], 64);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    uint32_t tmem_O = tmem_ptr[0];
    uint32_t tmem_Q = tmem_ptr[1];
    uint32_t tmem_K = tmem_ptr[2];
    uint32_t tmem_V = tmem_ptr[3];
    uint32_t tmem_S = tmem_ptr[4];
    uint32_t tmem_P = tmem_S; 
    
    uint32_t phase_Q = 0;
    uint32_t phase_KV = 0;
    uint32_t c2 = h + b * num_heads;
    uint64_t offset_bh = (static_cast<uint64_t>(b) * num_heads + h) * S_len * 128;

    float denom = 1.0f / sqrtf(128.0f);
    
    uint32_t idesc_QKT = make_instr_desc_fn_c1(128, 64, 0, 0);
    uint32_t idesc_PV = make_instr_desc_fn_c1(128, 64, 0, 1);
    
    for (uint32_t q_start = 0; q_start < S_len; q_start += 128) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_Q, 32768);
            tma_load_3d_fn(&tma_Q, bar_Q, smem_Q, 0, q_start, c2);
            tma_load_3d_fn(&tma_Q, bar_Q, smem_Q + 8192, 64, q_start, c2);
        }
        mbarrier_wait_fn(bar_Q, phase_Q);
        phase_Q ^= 1;
        
        float m_val = -1e20f;
        float l_val = 0.0f;
        uint32_t kv_idx = 0;
        
        uint32_t S_and_q_start = (q_start + 128 < S_len) ? (q_start + 128) : S_len;
        
        for (uint32_t k_start = 0; k_start < S_and_q_start; k_start += 64) {
            if (tid == 0) {
                mbarrier_arrive_and_expect_tx_fn(bar_KV, 49152);
                tma_load_3d_fn(&tma_K, bar_KV, smem_K, 0, k_start, c2);
                tma_load_3d_fn(&tma_K, bar_KV, smem_K + 4096, 64, k_start, c2);
                tma_load_3d_fn(&tma_V, bar_KV, smem_V, 0, k_start, c2);
                tma_load_3d_fn(&tma_V, bar_KV, smem_V + 4096, 64, k_start, c2);
            }
            mbarrier_wait_fn(bar_KV, phase_KV);
            phase_KV ^= 1;
            fence_proxy_async_fn();
            
            // Compute Q @ K_T
            if (tid == 0) {
                for (uint32_t k = 0; k < 64; k += 16) {
                    uint64_t dQ = update_smem_addr(make_smem_desc_k_major(smem_Q), (uint32_t)__cvta_generic_to_shared((uint8_t*)smem_Q + k * 2));
                    uint64_t dK = update_smem_addr(make_smem_desc_k_major(smem_K), (uint32_t)__cvta_generic_to_shared((uint8_t*)smem_K + k * 2));
                    umma_f16_cg1_fn(tmem_S, dQ, dK, idesc_QKT, (kv_idx == 0 && k == 0) ? 0 : 1);
                }
                for (uint32_t k = 0; k < 64; k += 16) {
                    uint64_t dQ = update_smem_addr(make_smem_desc_k_major(smem_Q + 8192), (uint32_t)__cvta_generic_to_shared((uint8_t*)smem_Q + 8192 + k * 2));
                    uint64_t dK = update_smem_addr(make_smem_desc_k_major(smem_K + 4096), (uint32_t)__cvta_generic_to_shared((uint8_t*)smem_K + 4096 + k * 2));
                    umma_f16_cg1_fn(tmem_S, dQ, dK, idesc_QKT, 1);
                }
            }
            umma_commit_1sm_fn(bar_KV);
            mbarrier_wait_fn(bar_KV, phase_KV);
            phase_KV ^= 1;
            
            float row_max = -1e20f;
            uint32_t q_pos_base = q_start + (tid / 64) * 64;
            
            for (int n = 0; n < 64; ++n) {
                uint32_t col = n % 8; 
                uint32_t row_offset = n / 8;
                uint32_t row = (tid % 64) + row_offset * 64; 
                
                if (row < 128) {
                    uint32_t reg_S_idx = row_offset * 8 + col;
                    float val = *reinterpret_cast<float*>(&__nv_bfloat162{*reinterpret_cast<__nv_bfloat162*>(&((uint32_t*)tmem_S)[row * 64 + (col << 1)])});
                    
                    uint32_t q_pos = q_pos_base + (row - (tid % 64));
                    uint32_t k_pos = k_start + col;
                    if (include_in_softmax(q_pos, k_pos, S_len)) {
                        val /= denom;
                        row_max = fmaxf(row_max, val);
                    }
                }
            }
            
            float m_new = fmaxf(m_val, row_max);
            float exp_old = (m_new > m_val) ? expf(m_val - m_new) : 1.0f;
            
            float row_sum = 0.0f;
            for (int n = 0; n < 64; ++n) {
                uint32_t col = n % 8;
                uint32_t row_offset = n / 8;
                uint32_t row = (tid % 64) + row_offset * 64;
                
                if (row < 128) {
                    uint32_t reg_S_idx = row_offset * 8 + col;
                    float val = *reinterpret_cast<float*>(&__nv_bfloat162{*reinterpret_cast<__nv_bfloat162*>(&((uint32_t*)tmem_S)[row * 64 + (col << 1)])});
                    
                    uint32_t q_pos = q_pos_base + (row - (tid % 64));
                    uint32_t k_pos = k_start + col;
                    
                    float p = 0.0f;
                    if (include_in_softmax(q_pos, k_pos, S_len)) {
                        p = expf(val - m_new);
                        row_sum += p;
                    }
                    
                    uint32_t sw_col = (((col) >> 3) ^ (row & 7)) << 3 | (col & 7);
                    *(uint16_t*)&smem_P[row * 128 + sw_col] = __float2bfloat16(p);
                }
            }
            
            l_val = l_val * exp_old + row_sum;
            m_val = m_new;
            
            __syncthreads(); 
            
            if (kv_idx > 0 && m_new > m_val) {
                float* smem_O = (float*)smem_S; 
                for (uint32_t i = tid; i < 16384; i += 128) {
                    smem_O[i] *= exp_old;
                }
            }
            
            __syncthreads();
            
            // Compute P @ V
            if (tid == 0) {
                for (uint32_t k = 0; k < 64; k += 16) {
                    uint64_t dP0 = update_smem_addr(make_smem_desc_k_major(smem_P), (uint32_t)__cvta_generic_to_shared((uint8_t*)smem_P + k * 2));
                    uint64_t dV0 = update_smem_addr(make_smem_desc_mn_major(smem_V), (uint32_t)__cvta_generic_to_shared((uint8_t*)smem_V + k * 128)); 
                    umma_f16_cg1_fn(tmem_O, dP0, dV0, idesc_PV, 1);
                    
                    uint64_t dV1 = update_smem_addr(make_smem_desc_mn_major(smem_V + 4096), (uint32_t)__cvta_generic_to_shared((uint8_t*)smem_V + 4096 + k * 128));
                    umma_f16_cg1_fn(tmem_O + 4096, dP0, dV1, idesc_PV, 1);
                }
            }
            umma_commit_1sm_fn(bar_KV);
            mbarrier_wait_fn(bar_KV, phase_KV);
            phase_KV ^= 1;
            
            kv_idx++;
        }
        
        if (l_val > 0.0f) {
            float* smem_O = (float*)smem_S; 
            for (uint32_t i = tid; i < 16384; i += 128) {
                smem_O[i] /= l_val;
            }
        }
        
        __syncthreads();
        
        for (uint32_t row = 0; row < 128; ++row) {
            uint32_t q_idx = q_start + row;
            if (q_idx < S_len) {
                for (uint32_t col = tid; col < 128; col += 128) {
                    uint64_t b_idx = offset_bh + (uint64_t)q_idx * 128 + col;
                    *(uint16_t*)&O[b_idx] = *(uint16_t*)&smem_O[row * 128 + col];
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
    
    if (tid == 0) {
        tmem_dealloc_fn(tmem_O, 128);
        tmem_dealloc_fn(tmem_Q, 64);
        tmem_dealloc_fn(tmem_K, 64);
        tmem_dealloc_fn(tmem_V, 64);
        tmem_dealloc_fn(tmem_S, 64);
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
  
  CUDA_CHECK(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 90112));
  
  attention_kernel<<<grid, block, 90112, stream>>>(tma_Q, tma_K, tma_V, 
                                                     static_cast<__nv_bfloat16*>(O.data_ptr()), 
                                                     static_cast<float*>(LSE.data_ptr()), 
                                                     S, H);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

} // namespace tvm_ffi_example_cuda