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

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_k_major_128B(void* ptr) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    uint64_t d = 0;
    d |= (addr & 0x3FFFF) >> 4;
    d |= ((1ULL & 0x3FFFF) >> 4) << 16;
    d |= ((1024ULL & 0x3FFFF) >> 4) << 32;
    d |= (1ULL << 46);
    uint32_t base_offset = (addr >> 7) & 0x7;
    d |= ((uint64_t)base_offset << 49);
    d |= (2ULL << 61);
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_mn_major_128B(void* ptr) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    uint64_t d = 0;
    d |= (addr & 0x3FFFF) >> 4;
    d |= ((8192ULL & 0x3FFFF) >> 4) << 16; 
    d |= ((1024ULL & 0x3FFFF) >> 4) << 32;
    d |= (1ULL << 46);
    uint32_t base_offset = (addr >> 7) & 0x7;
    d |= ((uint64_t)base_offset << 49);
    d |= (2ULL << 61);
    return d;
}

__device__ __forceinline__ uint64_t update_smem_addr(uint64_t desc, uint32_t addr) {
    return (desc & ~0x3FFFULL) | ((addr & 0x3FFFF) >> 4);
}

__device__ __forceinline__ void umma_f16_cg1(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc) {
    asm volatile(
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 0;\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc));
}

__device__ __forceinline__ void umma_f16_cg1_acc(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, float scale_d) {
    asm volatile(
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 0, 1, %4;\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "f"(scale_d));
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
    
    uint16_t* smem_Q = (uint16_t*)smem_buf;                   // [0 .. 32767] (32 KB)
    uint16_t* smem_K = (uint16_t*)(smem_buf + 32768);         // [32768 .. 49151] (16 KB)
    uint16_t* smem_V = (uint16_t*)(smem_buf + 49152);         // [49152 .. 65535] (16 KB)
    float* smem_S_fp32 = (float*)(smem_buf + 98304);          // [98304 .. 163839] (64 KB)
    uint16_t* smem_P = (uint16_t*)(smem_buf + 163840);        // [163840 .. 196607] (32 KB)
    
    uint64_t* bar_Q = (uint64_t*)(smem_buf + 196608);
    uint64_t* bar_KV = (uint64_t*)(smem_buf + 196616);
    uint32_t* tmem_ptr = (uint32_t*)(smem_buf + 196624);       // 16 Bytes (padded sizing)

    if (tid == 0) {
        init_smem_barrier_fn(bar_Q, 1);
        init_smem_barrier_fn(bar_KV, 1);
        
        tmem_alloc_fn(&tmem_ptr[0], 128);
        tmem_alloc_fn(&tmem_ptr[1], 64);
        tmem_alloc_fn(&tmem_ptr[2], 64);
        tmem_alloc_fn(&tmem_ptr[3], 64);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    uint32_t tmem_S = tmem_ptr[0];
    uint32_t tmem_O_0 = tmem_ptr[1];
    uint32_t tmem_O_1 = tmem_ptr[2];
    uint32_t tmem_S_base = tmem_S;
    uint32_t tmem_O_0_base = tmem_O_0;
    uint32_t tmem_O_1_base = tmem_O_1;
    
    uint32_t phase_Q = 0;
    uint32_t phase_KV = 0;
    uint64_t offset_bh = (static_cast<uint64_t>(b) * num_heads + h) * S_len * 128;

    float denom = 1.0f / sqrtf(128.0f);
    
    uint32_t idesc_QKT = make_instr_desc_fn_c1(128, 128, 0, 0);
    uint32_t idesc_PV_0 = make_instr_desc_fn_c1(128, 64, 0, 1);
    uint32_t idesc_PV_1 = make_instr_desc_fn_c1(128, 64, 0, 1);
    
    uint64_t desc_Q_0 = make_smem_desc_k_major_128B(smem_Q);
    uint64_t desc_Q_1 = make_smem_desc_k_major_128B(smem_Q + 8192);
    uint64_t desc_K_0 = make_smem_desc_k_major_128B(smem_K);
    uint64_t desc_K_1 = make_smem_desc_k_major_128B(smem_K + 8192);
    uint64_t desc_P_0 = make_smem_desc_k_major_128B(smem_P);
    uint64_t desc_P_1 = make_smem_desc_k_major_128B(smem_P + 8192);
    uint64_t desc_V_0 = make_smem_desc_mn_major_128B(smem_V);
    uint64_t desc_V_1 = make_smem_desc_mn_major_128B(smem_V + 8192);

    uint32_t saddr_Q = (uint32_t)__cvta_generic_to_shared(smem_Q);
    uint32_t saddr_K = (uint32_t)__cvta_generic_to_shared(smem_K);
    uint32_t saddr_V = (uint32_t)__cvta_generic_to_shared(smem_V);
    uint32_t saddr_P = (uint32_t)__cvta_generic_to_shared(smem_P);
    uint32_t saddr_P_1 = saddr_P + 16384;
    uint32_t saddr_V_1 = saddr_V + 16384;

    for (uint32_t q_start = 0; q_start < S_len; q_start += 128) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_Q, 32768);
            tma_load_4d_fn(&tma_Q, bar_Q, smem_Q, 0, q_start, h, b);
            tma_load_4d_fn(&tma_Q, bar_Q, smem_Q + 8192, 64, q_start, h, b);
        }
        mbarrier_wait_fn(bar_Q, phase_Q);
        phase_Q ^= 1;
        fence_proxy_async_fn();
        
        float m_val = -1e20f;
        float l_val = 0.0f;
        float exp_old = 1.0f;
        
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
            fence_proxy_async_fn();
            
            // Compute Q @ K_T utilizing highly optimized localized 16-step matrix multiplies
            if (tid == 0) {
                for (uint32_t k_half = 0; k_half < 2; ++k_half) {
                    uint64_t dQ = (k_half == 0) ? desc_Q_0 : desc_Q_1;
                    uint64_t dK = (k_half == 0) ? desc_K_0 : desc_K_1;
                    uint32_t base_Q = (k_half == 0) ? saddr_Q : (saddr_Q + 16384);
                    uint32_t base_K = (k_half == 0) ? saddr_K : (saddr_K + 16384);
                    
                    for (uint32_t k = 0; k < 4; ++k) {
                        uint64_t dQ_k = update_smem_addr(dQ, base_Q + k * 32);
                        uint64_t dK_k = update_smem_addr(dK, base_K + k * 32);
                        
                        if (k_start == 0 && k_half == 0 && k == 0) {
                            umma_f16_cg1(tmem_S_base + 0, dQ_k, dK_k, idesc_QKT);
                            umma_f16_cg1(tmem_S_base + 4096, dQ_k, dK_k, idesc_QKT);
                        } else {
                            umma_f16_cg1_acc(tmem_S_base + 0, dQ_k, dK_k, idesc_QKT, 1.0f);
                            umma_f16_cg1_acc(tmem_S_base + 4096, dQ_k, dK_k, idesc_QKT, 1.0f);
                        }
                    }
                }
            }
            umma_commit_1sm_fn(bar_KV);
            mbarrier_wait_fn(bar_KV, phase_KV);
            phase_KV ^= 1;
            
            if (threadIdx.x < 128) {
                uint32_t reg_S_idx_0 = (threadIdx.x * 2 + 0) * 64 + (k_start % 128);
                uint32_t reg_S_idx_1 = (threadIdx.x * 2 + 1) * 64 + (k_start % 128);
                
                float row_max = -1e20f;
                uint32_t q_pos_base = q_start + (threadIdx.x / 64) * 64;
                
                for (int n = 0; n < 64; ++n) {
                    float val_0 = *reinterpret_cast<float*>(&((uint32_t*)tmem_S_base)[reg_S_idx_0 + n]);
                    float val_1 = *reinterpret_cast<float*>(&((uint32_t*)tmem_S_base + 4096)[reg_S_idx_1 + n]);
                    
                    uint32_t q_pos_0 = q_pos_base + (threadIdx.x % 64) * 2 + 0;
                    uint32_t q_pos_1 = q_pos_base + (threadIdx.x % 64) * 2 + 1;
                    
                    if (include_in_softmax(q_pos_0, k_start + n, S_len)) {
                        val_0 *= denom;
                        row_max = fmaxf(row_max, val_0);
                    }
                    if (include_in_softmax(q_pos_1, k_start + n, S_len)) {
                        val_1 *= denom;
                        row_max = fmaxf(row_max, val_1);
                    }
                    
                    *reinterpret_cast<float*>(&((uint32_t*)tmem_S_base)[reg_S_idx_0 + n]) = val_0;
                    *reinterpret_cast<float*>(&((uint32_t*)tmem_S_base + 4096)[reg_S_idx_1 + n]) = val_1;
                }
                
                float m_new = fmaxf(m_val, row_max);
                exp_old = (m_new > m_val) ? expf(m_val - m_new) : 1.0f;
                
                float row_sum = 0.0f;
                for (int n = 0; n < 64; ++n) {
                    float val_0 = *reinterpret_cast<float*>(&((uint32_t*)tmem_S_base)[reg_S_idx_0 + n]);
                    float val_1 = *reinterpret_cast<float*>(&((uint32_t*)tmem_S_base + 4096)[reg_S_idx_1 + n]);
                    
                    uint32_t q_pos_0 = q_pos_base + (threadIdx.x % 64) * 2 + 0;
                    uint32_t q_pos_1 = q_pos_base + (threadIdx.x % 64) * 2 + 1;
                    
                    float p_0 = 0.0f;
                    if (include_in_softmax(q_pos_0, k_start + n, S_len)) {
                        p_0 = expf(val_0 - m_new);
                        row_sum += p_0;
                    }
                    float p_1 = 0.0f;
                    if (include_in_softmax(q_pos_1, k_start + n, S_len)) {
                        p_1 = expf(val_1 - m_new);
                        row_sum += p_1;
                    }
                    
                    // Linearize writes effectively resolving internal 128B chunk swizzling mappings natively 
                    uint32_t sw_col_0 = (((n) >> 3) ^ (((threadIdx.x % 64)*2 + 0) & 7)) << 3 | (n & 7);
                    uint32_t sw_col_1 = (((n) >> 3) ^ (((threadIdx.x % 64)*2 + 1) & 7)) << 3 | (n & 7);
                    
                    *(uint16_t*)&smem_P[((threadIdx.x % 64)*2 + 0) * 128 + sw_col_0] = __float2bfloat16(p_0);
                    *(uint16_t*)&smem_P[((threadIdx.x % 64)*2 + 1) * 128 + sw_col_1] = __float2bfloat16(p_1);
                }
                
                l_val = l_val * exp_old + row_sum;
                exp_old = expf(m_val - m_new);
                m_val = m_new;
            }
            __syncthreads(); 
            
            // Compute P @ V
            if (tid == 0) {
                for (uint32_t k_half = 0; k_half < 2; ++k_half) {
                    uint64_t dP = (k_half == 0) ? desc_P_0 : desc_P_1;
                    uint64_t dV = (k_half == 0) ? desc_V_0 : desc_V_1;
                    uint32_t base_P = (k_half == 0) ? saddr_P : saddr_P_1;
                    uint32_t base_V = (k_half == 0) ? saddr_V : saddr_V_1;
                    
                    for (uint32_t K_iter = 0; K_iter < 4; ++K_iter) {
                        uint64_t dP_k = update_smem_addr(dP, base_P + K_iter * 32);
                        uint64_t dV_k = update_smem_addr(dV, base_V + K_iter * 4096); 
                        
                        float scale_D_0 = (k_start == 0) ? 0.0f : ((m_val > -1e19f) ? expf((m_val - exp_old) - m_val) : 1.0f);

                        umma_f16_cg1_acc(tmem_O_0_base + 0, dP_k, dV_k, idesc_PV_0, scale_D_0);
                        umma_f16_cg1_acc(tmem_O_0_base + 4096, dP_k, dV_k, idesc_PV_0, scale_D_0);
                        
                        umma_f16_cg1_acc(tmem_O_1_base + 0, dP_k, dV_k, idesc_PV_1, scale_D_0);
                        umma_f16_cg1_acc(tmem_O_1_base + 4096, dP_k, dV_k, idesc_PV_1, scale_D_0);
                    }
                }
            }
            umma_commit_1sm_fn(bar_KV);
            mbarrier_wait_fn(bar_KV, phase_KV);
            phase_KV ^= 1;
        }
        
        // Utilize raw unswizzled linear formatting for globally coalesced vectorized output writes.
        float* smem_O = (float*)smem_K; 
        
        for (uint32_t i = tid; i < 16384; i += 256) {
            if (l_val > 0.0f) {
                smem_O[i] = *reinterpret_cast<float*>(&((uint32_t*)tmem_O_0_base)[i]) / l_val;
            } else {
                smem_O[i] = 0.0f;
            }
        }
        __syncthreads();
        
        for (uint32_t row = 0; row < 128; ++row) {
            uint32_t q_idx = q_start + row;
            if (q_idx < S_len) {
                for (uint32_t col = tid; col < 128; col += 128) {
                    uint64_t b_idx = offset_bh + (uint64_t)q_idx * 128 + col;
                    *(uint16_t*)&O[b_idx] = __float2bfloat16(smem_O[row * 128 + col]);
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
        tmem_dealloc_fn(tmem_S, 128);
        tmem_dealloc_fn(tmem_O_0, 64);
        tmem_dealloc_fn(tmem_O_1, 64);
        tmem_dealloc_fn(tmem_ptr[3], 64);
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
  dim3 block(256);
  
  // Request precisely bounded 200000 bytes of dynamic shared memory space to resolve OOB crashes.
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