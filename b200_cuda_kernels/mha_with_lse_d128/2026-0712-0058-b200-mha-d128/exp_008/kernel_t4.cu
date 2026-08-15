#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>
#include <algorithm>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

#define CU_CHECK(call) do {                                      \
    CUresult _e = (call);                                        \
    if (_e != CUDA_SUCCESS) {                                    \
        const char* err_str;                                     \
        cuGetErrorName(_e, &err_str);                            \
        fprintf(stderr, "CU error %s at %s:%d\n",                \
                err_str, __FILE__, __LINE__);                    \
        exit(1);                                                 \
    }                                                            \
} while(0)

__device__ __forceinline__ bool elect_one_sync_fn() {
    uint32_t pred;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "elect.sync _|p, 0xFFFFFFFF;\n"
        "selp.b32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(pred));
    return pred != 0;
}

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
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

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    uint32_t smem_mbar = (uint32_t)__cvta_generic_to_shared(bar);
    uint32_t smem_ptr  = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"(smem_ptr), "l"((uint64_t)d), "r"(smem_mbar), "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_cg2(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (0u << 15);   
    d |= (0u << 16);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ uint32_t pack_bf16_fn(uint32_t fp32_a, uint32_t fp32_b) {
    __nv_bfloat16 a = __float2bfloat16(__uint_as_float(fp32_a));
    __nv_bfloat16 b = __float2bfloat16(__uint_as_float(fp32_b));
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&a)),
          "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ uint32_t get_lane_cta0(uint32_t tid) {
    uint32_t warp_id = tid / 32;
    uint32_t cta_warp_id = cluster_rank_fn() / 32;
    uint32_t global_warp_id = cluster_rank_fn() % 32;
    uint32_t block_warp_id = cta_warp_id + global_warp_id * 4;
    return block_warp_id * 32 + (tid % 4) * 8 + (tid / 4) % 8;
}

__device__ __forceinline__ uint32_t get_swizzled_col(uint32_t row, uint32_t col) {
    uint32_t x_chunk = col / 8;
    uint32_t x_rem = col % 8;
    uint32_t swizzled_x_chunk = (row % 8) ^ x_chunk;
    return swizzled_x_chunk * 8 + x_rem;
}

__device__ __forceinline__ void umma_commit_2sm_fn(uint64_t* bar) {
    if (threadIdx.x == 0) {
        uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
        asm volatile(
            "tcgen05.commit.cta_group::2"
            ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
            " [%0], %1;"
            :: "r"(a), "h"((uint16_t)0x3));
    }
}

__device__ __forceinline__ void umma_wait_2sm_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase));
}

const int H = 48;

__global__ void attention_kernel(
    __grid_constant__ const CUtensorMap tma_Q,
    __grid_constant__ const CUtensorMap tma_K,
    __grid_constant__ const CUtensorMap tma_V,
    __nv_bfloat16* O, float* LSE, int S_len) 
{
    uint32_t batch_head = blockIdx.y;
    uint32_t b = batch_head / H;
    uint32_t h = batch_head % H;
    uint32_t q_start = blockIdx.x * 128;
    uint32_t my_q_start = q_start + (cluster_rank_fn() % 2) * 64;
    if (my_q_start >= S_len) return;

    uint32_t tmem_S, tmem_O_0, tmem_O_1;
    if (elect_one_sync_fn()) {
        tmem_alloc_fn(&tmem_S, 64);
        tmem_alloc_fn(&tmem_O_0, 64);
        tmem_alloc_fn(&tmem_O_1, 64);
    }
    __syncthreads();

    extern __shared__ __align__(1024) char smem_pool[];
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)smem_pool;                         
    __nv_bfloat16* smem_K = smem_Q + 128*64;                                  
    __nv_bfloat16* smem_V = smem_K + 128*64;                                  
    __nv_bfloat16* smem_P = smem_V + 128*64;                                   
    float* smem_S_fp32 = (float*)smem_P + 64*64;                               

    uint64_t* mbar_Q = (uint64_t*)((char*)smem_S_fp32 + 64*64*4);
    uint64_t* mbar_K_V = mbar_Q + 1;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_K_V, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t batch_head_offset = (b * H + h) * S_len;

    uint32_t base_S = tmem_S;
    uint32_t base_O_0 = tmem_S + 64 * 512;
    uint32_t base_O_1 = tmem_S + 128 * 512;

    uint32_t phase_Q = 0;
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 16384);
        tma_load_3d_fn(&tma_Q, mbar_Q, smem_Q, 0, my_q_start, batch_head);
        tma_load_3d_fn(&tma_Q, mbar_Q, smem_Q + 4096, 64, my_q_start, batch_head);
    }
    mbarrier_wait_fn(mbar_Q, phase_Q);
    phase_Q ^= 1;
    fence_proxy_async_fn();
    __syncthreads();

    float running_max_val[64];
    float running_sum_val[64];
    for (int i = threadIdx.x; i < 64; i += blockDim.x) {
        running_max_val[i] = -1e20f;
        running_sum_val[i] = 0.0f;
    }
    __syncthreads();

    uint32_t num_iters = (S_len + 63) / 64;
    uint32_t phase_K_V = 0;

    float log2e_f = 1.4426950408889634f;

    // Zero accumulator O_TMEM correctly utilizing 32-bit writes scaling over entire block quadrant ranges natively.
    for (uint32_t col = 0; col < 64; col+=4) {
        uint32_t r = get_lane_cta0(threadIdx.x);
        uint32_t packed = 0; 
        uint32_t addr = (r << 16) | col;
        asm volatile("tcgen05.st.sync.aligned.32x32b.b32 [%0], %1;" :: "r"(base_O_0 + addr), "r"(packed));
        asm volatile("tcgen05.st.sync.aligned.32x32b.b32 [%0], %1;" :: "r"(base_O_1 + addr), "r"(packed));
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    __syncthreads();

    for (uint32_t iter = 0; iter < num_iters; ++iter) {
        uint32_t kv_start = iter * 64;
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_K_V, 32768); 
            tma_load_3d_fn(&tma_K, mbar_K_V, smem_K, 0, kv_start, batch_head);
            tma_load_3d_fn(&tma_K, mbar_K_V, smem_K + 4096, 64, kv_start, batch_head);
            tma_load_3d_fn(&tma_V, mbar_K_V, smem_V, 0, kv_start, batch_head);
            tma_load_3d_fn(&tma_V, mbar_K_V, smem_V + 4096, 64, kv_start, batch_head);
        }
        mbarrier_wait_fn(mbar_K_V, phase_K_V);
        phase_K_V ^= 1;
        fence_proxy_async_fn();
        __syncthreads();
        
        // Zero accumulator S_TMEM natively leveraging 32-bit width fills scaling cohesively over quadrant blocks.
        for (uint32_t col = 0; col < 64; col+=4) {
            uint32_t r = get_lane_cta0(threadIdx.x);
            uint32_t packed = 0;
            uint32_t addr = (r << 16) | col;
            asm volatile("tcgen05.st.sync.aligned.32x32b.b32 [%0], %1;" :: "r"(base_S + addr), "r"(packed));
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        
        uint32_t idesc_S = make_instr_desc_cg2(64, 64);
        for (uint32_t chunk = 0; chunk < 2; ++chunk) {
            for (uint32_t k = 0; k < 64; k += 16) {
                uint32_t k_offset = chunk * 4096 + k;
                uint64_t desc_Q = make_smem_desc_sm100_fn(smem_Q + k_offset, 1, 1024);
                uint64_t desc_K = make_smem_desc_sm100_fn(smem_K + k_offset, 1, 1024);
                bool accum = (chunk > 0 || k > 0);
                uint32_t r = get_lane_cta0(threadIdx.x);
                uint32_t addr_s = (r << 16) | k;
                asm volatile(
                    "{\n.reg .pred p;\n"
                    "setp.ne.b32 p, %5, 0;\n"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                    :: "r"(base_S + addr_s), "l"(desc_Q), "l"(desc_K), "r"(idesc_S), "r"(accum));
            }
        }
        
        umma_commit_2sm_fn(mbar_K_V);
        umma_wait_2sm_fn(mbar_K_V, phase_K_V);
        phase_K_V ^= 1;
        
        // Extract fully computed S slice to linearly laid out Shared Memory 
        for (uint32_t col = 0; col < 64; col += 4) {
            uint32_t r = get_lane_cta0(threadIdx.x);
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(base_S + (r << 16) | col, &r0, &r1, &r2, &r3);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            if (threadIdx.x < 64) {
                uint32_t base = threadIdx.x * 64 + col;
                smem_S_fp32[base + 0] = __uint_as_float(r0);
                smem_S_fp32[base + 1] = __uint_as_float(r1);
                smem_S_fp32[base + 2] = __uint_as_float(r2);
                smem_S_fp32[base + 3] = __uint_as_float(r3);
            }
        }
        __syncthreads();
        
        if (threadIdx.x < 64) {
            uint32_t tid = threadIdx.x;
            uint32_t global_q = my_q_start + tid;
            
            float row_max = -1e20f;
            for (uint32_t c = 0; c < 64; c++) {
                uint32_t idx = tid * 64 + c;
                uint32_t global_kv = kv_start + c;
                if (global_kv < S_len) {
                    row_max = fmaxf(row_max, smem_S_fp32[idx]);
                }
            }
            
            float m_new = fmaxf(running_max_val[tid], row_max);
            float old_max = running_max_val[tid];
            float scale_o = fast_exp2f_fn((old_max - m_new) * log2e_f);
            float s_new = running_sum_val[tid] * scale_o;
            
            for (uint32_t c = 0; c < 64; c++) {
                uint32_t idx = tid * 64 + c;
                float v = smem_S_fp32[idx] * (1.0f / sqrtf(128.0f));
                uint32_t global_kv = kv_start + c;
                if (global_kv >= S_len || v > 1e19f) { 
                    v = -1e20f; 
                }
                float e = fast_exp2f_fn((v - m_new) * log2e_f);
                s_new += e;
                smem_P[idx] = __float2bfloat16(e);
            }
            
            running_max_val[tid] = m_new;
            running_sum_val[tid] = s_new;
            
            float final_scale_o = (m_new > -1e19f && running_sum_val[tid] > 0.0f) ? scale_o : 0.0f;
            for (uint32_t col = 0; col < 64; col += 2) {
                uint32_t r = get_lane_cta0(tid);
                uint32_t r0, r1, r2, r3;
                tmem_load_4x_fn(base_O_0 + (r << 16) | col, &r0, &r1, &r2, &r3);
                tmem_load_4x_fn(base_O_1 + (r << 16) | col, &r0, &r1, &r2, &r3);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                
                float f0 = __uint_as_float(r0) * final_scale_o;
                float f1 = __uint_as_float(r1) * final_scale_o;
                float f2 = __uint_as_float(r2) * final_scale_o;
                float f3 = __uint_as_float(r3) * final_scale_o;
                
                uint32_t packed0 = pack_bf16_fn(__float_as_uint(f0), __float_as_uint(f1));
                uint32_t packed1 = pack_bf16_fn(__float_as_uint(f2), __float_as_uint(f3));
                
                uint32_t swizzled_col0 = get_swizzled_col(tid, col);
                uint32_t swizzled_col1 = get_swizzled_col(tid, col + 2);
                
                uint32_t addr0a = (r << 16) | swizzled_col0;
                uint32_t addr1a = (r << 16) | swizzled_col1;
                asm volatile("tcgen05.st.sync.aligned.32x32b.b32 [%0], %1;" :: "r"(base_O_0 + addr0a), "r"(packed0));
                asm volatile("tcgen05.st.sync.aligned.32x32b.b32 [%0], %1;" :: "r"(base_O_1 + addr1a), "r"(packed1));
            }
        }
        __syncthreads();
        
        uint32_t idesc_P_V = make_instr_desc_cg2(64, 128);
        for (uint32_t chunk = 0; chunk < 2; ++chunk) {
            for (uint32_t k = 0; k < 64; k += 16) {
                uint32_t k_offset_P = k;
                uint32_t k_offset_V = chunk * 4096 + k;
                
                uint32_t swizzled_k_P = get_swizzled_col(chunk * 64 + (threadIdx.x % 64), k);
                uint32_t r = get_lane_cta0(threadIdx.x);
                uint32_t col_P = (r << 16) | swizzled_k_P;
                
                uint64_t desc_P = make_smem_desc_sm100_fn(smem_P + k_offset_P, 1, 1024);
                uint64_t desc_V = make_smem_desc_sm100_fn(smem_V + k_offset_V, 1024, 1024);
                
                bool accum_P = (chunk > 0 || k > 0);
                asm volatile(
                    "{\n.reg .pred p;\n"
                    "setp.ne.b32 p, %5, 0;\n"
                    "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                    :: "r"(chunk == 0 ? base_O_0 + col_P : base_O_1 + col_P), "l"(desc_P), "l"(desc_V), "r"(idesc_P_V), "r"(accum_P));
            }
        }
        
        umma_commit_2sm_fn(mbar_K_V);
        umma_wait_2sm_fn(mbar_K_V, phase_K_V);
        phase_K_V ^= 1;
        __syncthreads();
    }
    
    // Coalesced vectorized epilogue extractingOutputs directly leveraging native 32-bit packing bounds checks.
    for (uint32_t col = 0; col < 64; col += 4) {
        uint32_t r = get_lane_cta0(threadIdx.x);
        uint32_t r0, r1, r2, r3;
        uint32_t r0_b, r1_b, r2_b, r3_b;
        tmem_load_4x_fn(base_O_0 + (r << 16) | col, &r0, &r1, &r2, &r3);
        tmem_load_4x_fn(base_O_1 + (r << 16) | col, &r0_b, &r1_b, &r2_b, &r3_b);
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        if (threadIdx.x < 64) {
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            
            float f0_b = __uint_as_float(r0_b);
            float f1_b = __uint_as_float(r1_b);
            float f2_b = __uint_as_float(r2_b);
            float f3_b = __uint_as_float(r3_b);
            
            uint32_t q_idx = my_q_start + threadIdx.x;
            if (q_idx < S_len) {
                uint32_t d_base = ((b * H + h) * S_len + q_idx) * 128;
                if (col < 64) {
                    O[d_base + col] = __float2bfloat16(f0);
                    O[d_base + col + 1] = __float2bfloat16(f1);
                    O[d_base + col + 2] = __float2bfloat16(f2);
                    O[d_base + col + 3] = __float2bfloat16(f3);
                }
                if (col + 64 < 128) {
                    O[d_base + col + 64] = __float2bfloat16(f0_b);
                    O[d_base + col + 65] = __float2bfloat16(f1_b);
                    O[d_base + col + 66] = __float2bfloat16(f2_b);
                    O[d_base + col + 67] = __float2bfloat16(f3_b);
                }
            }
        }
    }
    
    // Limit indexing strictly ensuring Out of Bounds guardrails on scalar writes.
    if (threadIdx.x < 64) {
        uint32_t tid = threadIdx.x;
        uint32_t global_q = my_q_start + tid;
        if (global_q < S_len) {
            float lse_i = running_max_val[tid] + __logf(running_sum_val[tid]);
            LSE[(b * H + h) * S_len + global_q] = lse_i;
        }
    }
}

namespace tvm_ffi_mha_d128 {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  
  int64_t B = Q.size(0);
  int64_t H_val = Q.size(1);
  int64_t S = Q.size(2);
  int64_t D = Q.size(3);
  
  if (H_val != H) {
      fprintf(stderr, "Error: Expected heads=%d but got %ld\n", H, H_val);
      exit(1);
  }
  
  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
  
  CUtensorMap tma_Q, tma_K, tma_V;
  create_tma_3d_descriptor_2B(&tma_Q, Q.data_ptr(), D, S, B * H_val, 64, 64, 1, 
      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  create_tma_3d_descriptor_2B(&tma_K, K.data_ptr(), D, S, B * H_val, 64, 64, 1, 
      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  create_tma_3d_descriptor_2B(&tma_V, V.data_ptr(), D, S, B * H_val, 64, 64, 1, 
      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      
  dim3 grid((S + 127)/128, B * H_val);
  dim3 block(128);
  
  CUDA_CHECK(cudaFuncSetAttribute(
      attention_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      65536));
      
  cudaLaunchConfig_t config = {};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = 65536;
  config.stream = stream;
  
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = 2;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  config.attrs = attrs;
  config.numAttrs = 1;
  
  CU_CHECK(cudaLaunchKernelEx(&config, attention_kernel,
      tma_Q, tma_K, tma_V, 
      static_cast<__nv_bfloat16*>(O.data_ptr()), 
      static_cast<float*>(LSE.data_ptr()), 
      S));
      
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_mha_d128