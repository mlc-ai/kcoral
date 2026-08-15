#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
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

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void mbarrier_arrive_fn(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])) : "memory");
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

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void load_S_from_tmem(uint32_t tmem_S_base, float* temp_S, int tid) {
    uint32_t tmem_addr = tmem_S_base + tid * 4;
    for (int i = 0; i < 4; ++i) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
           : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_addr));
        temp_S[i*4 + 0] = __uint_as_float(r0);
        temp_S[i*4 + 1] = __uint_as_float(r1);
        temp_S[i*4 + 2] = __uint_as_float(r2);
        temp_S[i*4 + 3] = __uint_as_float(r3);
        tmem_addr += 4;
    }
}

__device__ __forceinline__ void cp_data_to_tmem(uint32_t smem_addr, uint32_t tmem_addr, int num_floats) {
    for (int i = 0; i < num_floats; ++i) {
        uint32_t val = *(uint32_t*)(smem_addr + i * 4);
        *(uint32_t*)__cvta_shared_to_generic(&((uint8_t*)tmem_addr)[i * 4]) = val;
    }
}

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
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
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];" :: "r"(a));
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

__device__ __forceinline__ uint32_t make_instr_desc_fn_pv(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (0u << 15);   
    d |= (1u << 16);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

extern __shared__ __align__(128) uint8_t smem_pool[];

__global__ void __launch_bounds__(128) attention_kernel(
    __grid_constant__ const CUtensorMap tma_Q,
    __grid_constant__ const CUtensorMap tma_K,
    __grid_constant__ const CUtensorMap tma_V,
    __nv_bfloat16* O, float* LSE, int S)
{
    setmaxnreg_inc_sync_fn<256>();
    
    int q_start = blockIdx.x * 128;
    int bh = blockIdx.y;
    
    if (q_start >= S) return;
    
    int tid = threadIdx.x; 
    
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)(smem_pool + 0);               
    __nv_bfloat16* smem_K = (__nv_bfloat16*)(smem_pool + 32768);              
    __nv_bfloat16* smem_V = (__nv_bfloat16*)(smem_pool + 65536);              
    __nv_bfloat16* smem_P = smem_K;                                  
    float* smem_O = (float*)(smem_pool + 98304);                            
    __nv_bfloat16* smem_out = (__nv_bfloat16*)(smem_pool + 163840);           
    
    uint64_t* mbar_Q = (uint64_t*)(smem_pool + 229376);                     
    uint64_t* mbar_KV = (uint64_t*)(smem_pool + 229384);                    
    uint64_t* mbar_QK = (uint64_t*)(smem_pool + 229392);                    

    float* smem_m_prev = (float*)(smem_pool + 229400);                       
    float* smem_l_prev = (float*)(smem_pool + 229912);                       
    
    if (tid == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_KV, 1);
        init_smem_barrier_fn(mbar_QK, 1);
    }
    __syncthreads();
    
    uint32_t tmem_S, tmem_O;
    if (tid == 0) {
        uint32_t ncols_S = 128; 
        tmem_alloc_fn(&tmem_S, ncols_S);
        uint32_t ncols_O = 128; 
        tmem_alloc_fn(&tmem_O, ncols_O);
    }
    __syncthreads();
    
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 32768);
        tma_load_3d_fn(&tma_Q, mbar_Q, smem_Q, 0, q_start, bh);
        tma_load_3d_fn(&tma_Q, mbar_Q, smem_Q + 8192, 64, q_start, bh);
    } else {
        mbarrier_arrive_fn(mbar_Q);
    }
    mbarrier_wait_fn(mbar_Q, 0);
    
    if (tid < 128) {
        smem_m_prev[tid] = -1e20f;
        smem_l_prev[tid] = 0.0f;
    }
    __syncthreads();
    
    float inv_sqrt_D = 1.0f / sqrtf(128.0f);
    
    uint64_t num_kv_blocks = (S + 127) / 128;
    uint32_t phase_KV = 0;
    uint32_t phase_QK = 0;
    uint32_t phase_PV = 0;
    
    for (uint64_t kv_start = 0; kv_start < num_kv_blocks * 128; kv_start += 128) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_KV, 65536);
            tma_load_3d_fn(&tma_K, mbar_KV, smem_K, 0, kv_start, bh);
            tma_load_3d_fn(&tma_K, mbar_KV, smem_K + 8192, 64, kv_start, bh);
            tma_load_3d_fn(&tma_V, mbar_KV, smem_V, 0, kv_start, bh);
            tma_load_3d_fn(&tma_V, mbar_KV, smem_V + 8192, 64, kv_start, bh);
        } else {
            mbarrier_arrive_fn(mbar_KV);
        }
        mbarrier_wait_fn(mbar_KV, phase_KV);
        phase_KV ^= 1;
        __syncthreads();
        
        cp_data_to_tmem((uint32_t)__cvta_generic_to_shared(smem_Q), tmem_S, 8);
        cp_data_to_tmem((uint32_t)__cvta_generic_to_shared(smem_K), tmem_S + 4096, 8); 
        
        for (int n_block = 0; n_block < 128; n_block += 8) {
            for (int k_chunk = 0; k_chunk < 128; k_chunk += 16) {
                uint32_t offset = k_chunk * 16; 
                uint64_t desc_Q = make_smem_desc_sm100_fn(smem_Q + offset, 16, 1024);
                uint64_t desc_K = make_smem_desc_sm100_fn(smem_K + offset, 16, 1024);
                
                uint32_t accum_S_0 = (n_block == 0 && k_chunk == 0) ? 0 : 1;
                uint32_t tmem_S0 = 0;
                umma_f16_cg1_fn(tmem_S0, desc_Q, desc_K, make_instr_desc_fn(64, 128), accum_S_0);
                
                uint32_t accum_S_1 = (n_block == 0 && k_chunk == 0) ? 0 : 1;
                uint32_t tmem_S1 = 64 * 128;
                umma_f16_cg1_fn(tmem_S1, desc_Q + offset + 64*128*2, desc_K, make_instr_desc_fn(64, 128), accum_S_1);
            }
        }
        
        umma_commit_1sm_fn(mbar_QK);
        mbarrier_wait_fn(mbar_QK, phase_QK);
        phase_QK ^= 1;
        
        float temp_S[128];
        load_S_from_tmem(tmem_S, temp_S, tid);
        tmem_load_fence_fn();
        
        float row_max[128];
        for (int j = tid; j < 128; j += blockDim.x) {
            float val = temp_S[j];
            val *= inv_sqrt_D;
            int kv_idx = kv_start + j;
            if (kv_idx >= S) {
                val = -1e20f;
            }
            temp_S[j] = val;
            row_max[j] = val;
        }
        
        float max_val = -1e20f;
        for (int j = tid; j < 128; j += blockDim.x) {
            max_val = fmaxf(max_val, row_max[j]);
        }
        
        for (int offset = 16; offset > 0; offset /= 2) {
            max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, offset));
        }
        
        float m_prev = smem_m_prev[tid];
        float m_new = fmaxf(m_prev, max_val);
        
        float rescale = 0.0f;
        if (m_new > -1e19f) {
            rescale = fast_exp2f_fn((m_prev - m_new) * 1.44269504f);
        }
        
        for (int col = 0; col < 128; col++) {
            smem_O[tid * 128 + col] *= rescale;
        }
        
        float sum_p = 0;
        for (int j = tid; j < 128; j += blockDim.x) {
            float val = temp_S[j];
            float p = 0;
            if (val > -1e19f) {
                p = fast_exp2f_fn((val - m_new) * 1.44269504f);
            }
            sum_p += p;
            row_max[j] = p;
        }
        
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_p += __shfl_xor_sync(0xFFFFFFFF, sum_p, offset);
        }
        
        float l_prev = smem_l_prev[tid];
        smem_l_prev[tid] = l_prev * rescale + sum_p;
        smem_m_prev[tid] = m_new;
        
        for (int j = tid; j < 128; j += blockDim.x) {
            smem_P[tid * 128 + j] = __float2bfloat16(row_max[j]);
        }
        
        fence_async_shared_fn();
        __syncthreads();
        
        cp_data_to_tmem((uint32_t)__cvta_generic_to_shared(smem_P), tmem_S, 8); 
        cp_data_to_tmem((uint32_t)__cvta_generic_to_shared(smem_V), tmem_S + 4096, 8); 
        
        for (int k_chunk = 0; k_chunk < 128; k_chunk += 16) {
            uint32_t offset = k_chunk * 16;
            uint64_t desc_P = make_smem_desc_sm100_fn(smem_P + offset, 16, 1024);
            uint64_t desc_V = make_smem_desc_sm100_fn(smem_V + offset, 16, 1024);
            
            uint32_t accum_PV_0 = (k_chunk == 0) ? 0 : 1;
            uint32_t accum_PV_1 = (k_chunk == 0) ? 0 : 1;
            
            uint32_t tmem_O0 = tmem_O;
            umma_f16_cg1_fn(tmem_O0, desc_P, desc_V, make_instr_desc_fn_pv(64, 128), accum_PV_0);
            
            uint32_t tmem_O1 = tmem_O + 64 * 128;
            umma_f16_cg1_fn(tmem_O1, desc_P + offset + 64*128*2, desc_V, make_instr_desc_fn_pv(64, 128), accum_PV_1);
        }
        
        umma_commit_1sm_fn(mbar_QK);
        mbarrier_wait_fn(mbar_QK, phase_PV);
        phase_PV ^= 1;
        
        for (int col = 0; col < 128; col++) {
            uint32_t r;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];" : "=r"(r) : "r"(tmem_O + tid * 128 + col));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            __nv_bfloat16 pv_val = __float2bfloat16(__uint_as_float(r));
            smem_O[tid * 128 + col] += __bfloat162float(pv_val) / smem_l_prev[tid];
        }
        
        __syncthreads();
    }
    
    for (int i = tid; i < 128 * 128; i += 128) {
        int r = i / 128;
        int c = i % 128;
        float val = smem_O[r * 128 + c];
        
        if (smem_l_prev[r] > 0) {
            val /= smem_l_prev[r];
        }
        
        smem_out[i] = __float2bfloat16(val);
    }
    __syncthreads();
    
    uint32_t warp_id = tid / 32;
    uint32_t lane_id = tid % 32;
    uint32_t num_steps = (128 + 3) / 4;
    
    uint64_t offset_O = (bh * S + q_start) * 128;
    
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        if (row >= 128) continue;
        
        uint32_t global_row = q_start + row;
        if (global_row >= S) continue;
        
        uint32_t col_start = lane_id * 4;
        uint32_t global_col = col_start;
        
        if (global_col + 3 < 128) {
            uint2 data = *(uint2*)&smem_out[row * 128 + col_start];
            *(uint2*)&O[offset_O + row * 128 + col_start] = data;
        }
    }
    
    if (q_start + tid < S) {
        LSE[bh * S + q_start + tid] = smem_m_prev[tid] + __logf(smem_l_prev[tid]);
    }
}

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, const void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_middle_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[3] = {gmem_inner_dim, gmem_middle_dim, gmem_outer_dim};
    cuuint64_t globalStrides[2] = {gmem_inner_dim * 2, gmem_inner_dim * gmem_middle_dim * 2};
    cuuint32_t boxDim[3] = {smem_inner_dim, smem_outer_dim, 1};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        3, 
        (void*)globalAddress,
        globalDim,
        globalStrides,
        boxDim, 
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
  CUDA_CHECK(cudaSetDevice(Q.device().device_id));
  
  int64_t B = Q.size(0);
  int64_t H = Q.size(1);
  int64_t S = Q.size(2);
  int64_t D = Q.size(3);
  
  if (D != 128) {
    fprintf(stderr, "Expected D=128, got D=%ld\n", D);
    exit(1);
  }
  
  const __nv_bfloat16* g_Q = static_cast<const __nv_bfloat16*>(Q.data_ptr());
  const __nv_bfloat16* g_K = static_cast<const __nv_bfloat16*>(K.data_ptr());
  const __nv_bfloat16* g_V = static_cast<const __nv_bfloat16*>(V.data_ptr());
  
  __nv_bfloat16* g_O = static_cast<__nv_bfloat16*>(O.data_ptr());
  float* g_LSE = static_cast<float*>(LSE.data_ptr());
  
  CUtensorMap tma_Q, tma_K, tma_V;
  
  CUresult res_Q = create_tma_3d_descriptor_2B(&tma_Q, g_Q, D, S, B * H, 128, 128, CU_TENSOR_MAP_SWIZZLE_128B);
  if (res_Q != CUDA_SUCCESS) { fprintf(stderr, "TMA Q failed\n"); exit(1); }
  
  CUresult res_K = create_tma_3d_descriptor_2B(&tma_K, g_K, D, S, B * H, 128, 128, CU_TENSOR_MAP_SWIZZLE_128B);
  if (res_K != CUDA_SUCCESS) { fprintf(stderr, "TMA K failed\n"); exit(1); }
  
  CUresult res_V = create_tma_3d_descriptor_2B(&tma_V, g_V, D, S, B * H, 128, 128, CU_TENSOR_MAP_SWIZZLE_128B);
  if (res_V != CUDA_SUCCESS) { fprintf(stderr, "TMA V failed\n"); exit(1); }
  
  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
  
  dim3 grid((S + 127) / 128, B * H);
  dim3 block(128);
  
  int smem_size = 231168;
  CUDA_CHECK(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  
  attention_kernel<<<grid, block, smem_size, stream>>>(tma_Q, tma_K, tma_V, g_O, g_LSE, S);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda