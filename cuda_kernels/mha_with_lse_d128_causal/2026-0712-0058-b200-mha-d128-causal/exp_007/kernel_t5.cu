#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CU_CHECK(call) do {                                      \
    CUresult _e = (call);                                        \
    if (_e != CUDA_SUCCESS) {                                    \
        fprintf(stderr, "CUDA error %d at %s:%d\n", (int)_e,      \
                __FILE__, __LINE__);                               \
        exit(1);                                                   \
    }                                                              \
} while(0)

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)


namespace tvm_ffi_example_cuda {

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

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(phase));
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(bar)),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void named_barrier_sync_fn(int bar_id, int count) {
    asm volatile("barrier.sync.aligned %0, %1;" :: "r"(bar_id), "r"(count));
}

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
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

__device__ __forceinline__ uint64_t get_desc_k_major(void* ptr, int k_offset_elements) {
    return make_smem_desc_sm100_fn((char*)ptr + k_offset_elements * 2, 1, 1024);
}

__device__ __forceinline__ uint64_t get_desc_n_major(void* ptr, int k_offset_elements) {
    return make_smem_desc_sm100_fn((char*)ptr + k_offset_elements * 2, 8192, 1024);
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
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

__device__ __forceinline__ void umma_f16_cg1_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void write_s2s_swizzled_128B(__nv_bfloat16* smem, int row, int col, float val) {
    int x_chunk = col / 8;
    int x_rem = col % 8;
    int y_chunk = row % 8;
    int new_x_chunk = x_chunk ^ y_chunk;
    int new_col = new_x_chunk * 8 + x_rem;
    smem[row * 64 + new_col] = __float2bfloat16(val);
}

__global__ __launch_bounds__(128) void causal_attention_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* out_O, float* out_lse,
    uint64_t seq_len, uint64_t head_dim) 
{
    int block_idx = blockIdx.x; 
    int batch_head_idx = blockIdx.y; 
    
    extern __shared__ __align__(1024) char smem[];
    __nv_bfloat16* smem_Q_0 = (__nv_bfloat16*)smem;
    __nv_bfloat16* smem_Q_1 = smem_Q_0 + 8192;
    __nv_bfloat16* smem_K_0 = smem_Q_1 + 8192;
    __nv_bfloat16* smem_K_1 = smem_K_0 + 8192;
    __nv_bfloat16* smem_V_0 = smem_K_1 + 8192;
    __nv_bfloat16* smem_V_1 = smem_V_0 + 8192;
    __nv_bfloat16* smem_P_0 = smem_V_1 + 8192;
    __nv_bfloat16* smem_P_1 = smem_P_0 + 8192;
    
    uint64_t* mbar_Q = (uint64_t*)(smem_P_1 + 8192);
    uint64_t* mbar_KV = mbar_Q + 1;
    uint64_t* mbar_UMMA = mbar_KV + 1;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_KV, 1);
        init_smem_barrier_fn(mbar_UMMA, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t tmem_s, tmem_o;
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_s, 128);
        tmem_alloc_fn(&tmem_o, 128);
    }
    __syncthreads();

    uint64_t outer_idx = (uint64_t)batch_head_idx * seq_len + (uint64_t)block_idx * 128;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 32768);
        tma_load_2d_fn(&tma_Q, mbar_Q, smem_Q_0, 0, outer_idx);
        tma_load_2d_fn(&tma_Q, mbar_Q, smem_Q_1, 64, outer_idx);
    }
    mbarrier_wait_fn(mbar_Q, 0);

    int tid = threadIdx.x;
    float global_rowmax = -1e20f;
    float global_rowsum = 0.0f;

    int global_row = batch_head_idx * seq_len + block_idx * 128;

    int kv_phase = 0;
    int umma_phase = 0;

    for (int j = 0; j <= block_idx; ++j) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_KV, 32768 * 2);
            tma_load_2d_fn(&tma_K, mbar_KV, smem_K_0, 0, batch_head_idx * seq_len + j * 128);
            tma_load_2d_fn(&tma_K, mbar_KV, smem_K_1, 64, batch_head_idx * seq_len + j * 128);
            tma_load_2d_fn(&tma_V, mbar_KV, smem_V_0, 0, batch_head_idx * seq_len + j * 128);
            tma_load_2d_fn(&tma_V, mbar_KV, smem_V_1, 64, batch_head_idx * seq_len + j * 128);
        }
        mbarrier_wait_fn(mbar_KV, kv_phase);
        kv_phase ^= 1;

        // Perform Q @ K^T -> tmem_s
        uint32_t idesc_q_k = make_instr_desc_fn(128, 128);
        
        for (int k = 0; k < 64; k += 16) {
            uint64_t desc_a = get_desc_k_major(smem_Q_0, k);
            uint64_t desc_b = get_desc_k_major(smem_K_0, k);
            if (k == 0) umma_f16_cg1_fn(tmem_s, desc_a, desc_b, idesc_q_k, 0);
            else        umma_f16_cg1_fn(tmem_s, desc_a, desc_b, idesc_q_k, 1);
        }
        for (int k = 0; k < 64; k += 16) {
            uint64_t desc_a = get_desc_k_major(smem_Q_1, k);
            uint64_t desc_b = get_desc_k_major(smem_K_1, k);
            umma_f16_cg1_fn(tmem_s, desc_a, desc_b, idesc_q_k, 1);
        }

        if (threadIdx.x == 0) {
            uint32_t a = (uint32_t)__cvta_generic_to_shared(mbar_UMMA);
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared.b64 [%0];" :: "r"(a));
        }
        mbarrier_wait_fn(mbar_UMMA, umma_phase);
        umma_phase ^= 1;
        
        float thread_rowmax = -1e20f;
        for (int col = 0; col < 128; col += 4) {
            uint32_t v0, v1, v2, v3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(v0),"=r"(v1),"=r"(v2),"=r"(v3) : "r"(tmem_s + col));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float vals[4] = {__uint_as_float(v0), __uint_as_float(v1), __uint_as_float(v2), __uint_as_float(v3)};
            for (int i=0; i<4; ++i) {
                float s_val = vals[i] * (1.f / sqrtf(head_dim));
                int global_col = j * 128 + col + i;
                if (global_col <= global_row) {
                    if (s_val > thread_rowmax) thread_rowmax = s_val;
                }
            }
        }

        // Software feedback trick delaying scale until after reduction 
        float thread_rowsum_local = 0.0f;
        for (int col = 0; col < 128; col += 4) {
            uint32_t v0, v1, v2, v3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(v0),"=r"(v1),"=r"(v2),"=r"(v3) : "r"(tmem_s + col));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

            float vals[4] = {__uint_as_float(v0), __uint_as_float(v1), __uint_as_float(v2), __uint_as_float(v3)};
            for (int i=0; i<4; ++i) {
                float s_val = vals[i] * (1.f / sqrtf(head_dim));
                int global_col = j * 128 + col + i;
                
                if (global_col <= global_row) {
                    float val = s_val - thread_rowmax;
                    val = __expf(val);
                    thread_rowsum_local += val;
                    if (global_col < 64) {
                        write_s2s_swizzled_128B(smem_P_0, tid, global_col, val);
                    } else {
                        write_s2s_swizzled_128B(smem_P_1, tid, global_col - 64, val);
                    }
                } else {
                    if (global_col < 64) {
                        write_s2s_swizzled_128B(smem_P_0, tid, global_col, 0.0f);
                    } else {
                        write_s2s_swizzled_128B(smem_P_1, tid, global_col - 64, 0.0f);
                    }
                }
            }
        }

        float new_max_i = fmaxf(global_rowmax, thread_rowmax);
        float scale_o = (global_rowmax == -1e20f) ? 1.0f : __expf(global_rowmax - new_max_i);
        
        if (scale_o != 1.0f) {
            for (int col = 0; col < 128; col += 4) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_o + col));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                
                float f0 = __uint_as_float(r0) * scale_o;
                float f1 = __uint_as_float(r1) * scale_o;
                float f2 = __uint_as_float(r2) * scale_o;
                float f3 = __uint_as_float(r3) * scale_o;
                
                uint32_t new_r0 = __float_as_uint(f0);
                uint32_t new_r1 = __float_as_uint(f1);
                uint32_t new_r2 = __float_as_uint(f2);
                uint32_t new_r3 = __float_as_uint(f3);
                
                asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    :: "r"(new_r0), "r"(new_r1), "r"(new_r2), "r"(new_r3), "r"(tmem_o + col));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            }
        }
        
        named_barrier_sync_fn(1, 128);
        fence_async_shared_fn();

        uint64_t desc_p_0 = get_desc_k_major(smem_P_0, 0);
        uint64_t desc_v_0 = get_desc_n_major(smem_V_0, 0);
        
        uint32_t idesc_p_v = make_instr_desc_fn(128, 128);
        idesc_p_v |= (1u << 16);   // transpose B
        
        for (int k = 0; k < 64; k += 16) {
            uint64_t desc_a = get_desc_k_major(smem_P_0, k);
            uint64_t desc_b = get_desc_n_major(smem_V_0, k);
            if (k == 0) umma_f16_cg1_fn(tmem_o, desc_a, desc_b, idesc_p_v, 0);
            else        umma_f16_cg1_fn(tmem_o, desc_a, desc_b, idesc_p_v, 1);
        }
        for (int k = 0; k < 64; k += 16) {
            uint64_t desc_a = get_desc_k_major(smem_P_1, k);
            uint64_t desc_b = get_desc_n_major(smem_V_1, k);
            umma_f16_cg1_fn(tmem_o, desc_a, desc_b, idesc_p_v, 1);
        }
        
        if (threadIdx.x == 0) {
            uint32_t a = (uint32_t)__cvta_generic_to_shared(mbar_UMMA);
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared.b64 [%0];" :: "r"(a));
        }
        mbarrier_wait_fn(mbar_UMMA, umma_phase);
        umma_phase ^= 1;
        
        global_rowmax = new_max_i;
        global_rowsum = global_rowsum * scale_o + thread_rowsum_local;
    }

    for (int col = 0; col < 128; col += 4) {
        uint32_t v0, v1, v2, v3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(v0),"=r"(v1),"=r"(v2),"=r"(v3) : "r"(tmem_o + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float o_vals[4] = {__uint_as_float(v0), __uint_as_float(v1), __uint_as_float(v2), __uint_as_float(v3)};
        for(int i=0; i<4; ++i) {
            float out_val = o_vals[i] / global_rowsum;
            int global_col = col + i;
            if (global_row < seq_len && global_col < head_dim) {
                out_O[(uint64_t)global_row * head_dim + global_col] = __float2bfloat16(out_val);
            }
        }
    }

    if (tid < 128) {
        int global_row_idx = batch_head_idx * seq_len + block_idx * 128 + tid;
        if (global_row_idx < seq_len) {
            float rowmax_f32 = global_rowmax;
            float rowsum_f32 = global_rowsum;
            out_lse[global_row_idx] = rowmax_f32 + __logf(rowsum_f32) * 1.4426950;
        }
    }

    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_s, 128);
        tmem_dealloc_fn(tmem_o, 128);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
  CUDA_CHECK(cudaSetDevice(Q.device().device_id)); 
  uint64_t B = Q.size(0);
  uint64_t H = Q.size(1);
  uint64_t S = Q.size(2);
  uint64_t D = Q.size(3); 

  CUtensorMap tma_Q, tma_K, tma_V;
  __nv_bfloat16* q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
  __nv_bfloat16* k_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
  __nv_bfloat16* v_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());

  CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, q_ptr, D, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, k_ptr, D, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, v_ptr, D, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

  __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
  float* lse_ptr = static_cast<float*>(LSE.data_ptr());

  int64_t blocks_x = (S + 127) / 128;
  dim3 grid(blocks_x, B * H);
  dim3 block(128);

  size_t smem_size = 131072 + 64;
  CUDA_CHECK(cudaFuncSetAttribute(causal_attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
  
  causal_attention_kernel<<<grid, block, smem_size, stream>>>(
      tma_Q, tma_K, tma_V, o_ptr, lse_ptr, S, D);
      
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_example_cuda