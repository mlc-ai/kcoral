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

namespace tvm_ffi_example_cuda {

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t dim0, uint64_t dim1, uint64_t dim2, uint32_t box0, uint32_t box1, uint32_t box2, CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[3] = {dim0, dim1, dim2};
    cuuint64_t globalStrides[2] = {dim0 * 2, dim0 * dim1 * 2};
    cuuint32_t boxDim[3] = {box0, box1, box2};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        3, // tensorRank
        globalAddress,
        globalDim,
        globalStrides,
        boxDim, 
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
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

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ void tmem_zero(uint32_t tmem_base) {
    uint32_t tid = threadIdx.x;
    uint32_t col = tid * 4; 
    
    uint32_t r0 = 0, r1 = 0, r2 = 0, r3 = 0;
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" 
                 :: "r"(r0),"r"(r1),"r"(r2),"r"(r3),"r"(tmem_base + col));
}

__device__ __forceinline__ void tmem_alloc_128(uint32_t* dst_smem) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 128;" 
                 :: "r"(a));
}

__device__ __forceinline__ void load_s2_reg(float* s_reg, uint32_t s_tmem, int row_offset, int col_offset) {
    uint32_t tid = threadIdx.x;
    uint32_t col = tid * 4 + col_offset; 
    
    uint32_t r0, r1, r2, r3;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(s_tmem + col + (row_offset << 16)));
    s_reg[0] = __uint_as_float(r0);
    s_reg[1] = __uint_as_float(r1);
    s_reg[2] = __uint_as_float(r2);
    s_reg[3] = __uint_as_float(r3);
}

__device__ __forceinline__ void load_o2_reg(float* o_reg, uint32_t o_tmem, int row_offset, int col_offset) {
    uint32_t tid = threadIdx.x;
    uint32_t col = tid * 4 + col_offset;
    
    uint32_t r0, r1, r2, r3;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(o_tmem + col + (row_offset << 16)));
    o_reg[0] = __uint_as_float(r0);
    o_reg[1] = __uint_as_float(r1);
    o_reg[2] = __uint_as_float(r2);
    o_reg[3] = __uint_as_float(r3);
}

__device__ __forceinline__ void store_p2_tmem(uint32_t p_tmem, int row_offset, int col_offset, uint32_t p_reg_u32) {
    uint32_t tid = threadIdx.x;
    uint32_t col = tid * 4 + col_offset;
    
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 {%0}, [%4];" 
                 :: "r"(p_reg_u32), "r"(p_tmem + col + (row_offset << 16)));
}

__device__ __forceinline__ void umma_cg1_16x16(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void commit_and_wait(uint64_t* bar, uint32_t phase) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared.b64 [%0];" :: "r"(a) : "memory");
    mbarrier_wait_fn(bar, phase);
}

__device__ __forceinline__ uint64_t make_smem_desc(void* smem_ptr, uint32_t lbo, uint32_t sbo, uint32_t swizzle) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16; 
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32; 
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)swizzle << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, bool transpose_B) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (0u << 15);   
    d |= ((transpose_B ? 1u : 0u) << 16);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__global__
__launch_bounds__(128)
void attention_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* __restrict__ O, 
    float* __restrict__ LSE,
    int S_len)
{
    int batch_head = blockIdx.x; 
    int query_start = blockIdx.y * 64;
    int tid = threadIdx.x;
    
    if (query_start >= S_len) return;
    
    extern __shared__ __align__(128) char smem[];
    
    // Memory Layout Allocation
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)smem;                      // 16KB (64x128)
    __nv_bfloat16* smem_K = (__nv_bfloat16*)(smem + 16384);             // 32KB (128x128)
    __nv_bfloat16* smem_V = (__nv_bfloat16*)(smem + 49152);             // 32KB (128x128)
    __nv_bfloat16* smem_P_bf16 = (__nv_bfloat16*)(smem + 81920);        // 16KB (64x128)
    float* smem_S = (float*)(smem + 98304);                             // 32KB (64x128)
    float* smem_O = (float*)(smem + 131072);                            // 32KB (64x128)
    
    uint64_t* bar_Q = (uint64_t*)(smem + 163840);
    uint64_t* bar_K = (uint64_t*)(smem + 163848);
    uint64_t* bar_V = (uint64_t*)(smem + 163856);
    
    if (tid == 0) {
        init_smem_barrier_fn(bar_Q, 1);
        init_smem_barrier_fn(bar_K, 1);
        init_smem_barrier_fn(bar_V, 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    uint32_t s_tmem, p_tmem;
    if (tid == 0) {
        tmem_alloc_128(&s_tmem);
        tmem_alloc_128(&p_tmem);
    }
    __syncthreads();
    
    int phase_Q = 0;
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_Q, 16384);
        tma_load_3d_fn(&tma_Q, bar_Q, smem_Q, 0, query_start, batch_head);
        tma_load_3d_fn(&tma_Q, bar_Q, smem_Q + 8192, 64, query_start, batch_head);
    }
    mbarrier_wait_fn(bar_Q, phase_Q);
    
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_O[2][8]; 
    for(int i = 0; i < 2; ++i) {
        for(int j = 0; j < 8; ++j) {
            wmma::fill_fragment(acc_O[i][j], 0.0f);
        }
    }
    
    float m_old[2] = {-1e20f, -1e20f};
    float d_old[2] = {0, 0};
    
    int phase_K = 0, phase_V = 0;
    int valid_rows = min(64, S_len - query_start);
    
    uint32_t idesc_Q = make_instr_desc_fn(16, 16, true);
    uint32_t idesc_V = make_instr_desc_fn(16, 16, false);
    
    for (int j = 0; j < S_len; j += 128) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_K, 32768);
            tma_load_3d_fn(&tma_K, bar_K, smem_K, 0, j, batch_head);
            tma_load_3d_fn(&tma_K, bar_K, smem_K + 8192, 64, j, batch_head);
            
            mbarrier_arrive_and_expect_tx_fn(bar_V, 32768);
            tma_load_3d_fn(&tma_V, bar_V, smem_V, 0, j, batch_head);
            tma_load_3d_fn(&tma_V, bar_V, smem_V + 8192, 64, j, batch_head);
        }
        
        mbarrier_wait_fn(bar_K, phase_K);
        mbarrier_wait_fn(bar_V, phase_V);
        phase_K ^= 1;
        phase_V ^= 1;
        
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_S[2][8]; 
        for(int i = 0; i < 2; ++i) {
            for(int k_step = 0; k_step < 8; ++k_step) {
                wmma::fill_fragment(acc_S[i][k_step], 0.0f);
            }
        }
        
        uint64_t desc_Q[2][8], desc_K[2][8], desc_P[4][8], desc_V[2][8];
        for(int i=0; i<2; ++i) {
            for(int k_step=0; k_step<8; ++k_step) {
                __nv_bfloat16* ptr_Q = smem_Q + (warp_id * 16) * 128 + k_step * 16; 
                __nv_bfloat16* ptr_K = smem_K + (i * 128 * 64) + k_step * 16 * 128; 
                
                desc_Q[i][k_step] = make_smem_desc(ptr_Q, 1024, 128, 2);
                desc_K[i][k_step] = make_smem_desc(ptr_K, 1024, 128, 2);
            }
        }
        for(int i=0; i<4; ++i) {
            for(int k_step=0; k_step<8; ++k_step) {
                for(int n_step=0; n_step<8; ++n_step) {
                    __nv_bfloat16* ptr_P = smem_P_bf16 + i * 16 * 128 + k_step * 16; 
                    __nv_bfloat16* ptr_V = smem_V + (n_step * 16) * 128 + k_step * 16; 
                    
                    desc_P[i][k_step] = make_smem_desc(ptr_P, 1024, 128, 2);
                    desc_V[n_step][k_step] = make_smem_desc(ptr_V, 1024, 128, 2);
                }
            }
        }
        
        // Initialize S accumulator in TMEM explicitly resolving zero initialization limits 
        for (int i = 0; i < 128; ++i) {
            tmem_zero(i * 4 + s_tmem);
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        
        for(int i=0; i<2; ++i) {
            for(int k_step=0; k_step<8; ++k_step) {
                for(int n_step=0; n_step<8; ++n_step) {
                    umma_cg1_16x16(s_tmem + (warp_id * 16) * 4 + n_step * 4, desc_Q[i][k_step], desc_K[i][k_step], idesc_Q, i == 0 && k_step == 0 ? 0 : 1);
                }
            }
        }
        commit_and_wait(bar_Q, phase_Q);
        phase_Q ^= 1;
        
        float my_max[2] = {-1e20f, -1e20f};
        for(int k=0; k<4; ++k) {
            load_s2_reg(&my_max[0], s_tmem, (tid % 64), tid / 64, k * 8);
            my_max[0] = max(my_max[0], max(max(my_max[0], my_max[1]), max(my_max[2], my_max[3])));
        }
        
        float new_max[2] = {max(m_old[0], my_max[0]), max(m_old[1], my_max[1])};
        float factor[2] = {expf(m_old[0] - new_max[0]), expf(m_old[1] - new_max[1])};
        
        d_old[0] *= factor[0];
        d_old[1] *= factor[1];
        
        // Scale O in registers inline 
        for (int n_step = 0; n_step < 8; ++n_step) {
            for (int i = 0; i < 2; ++i) {
                wmma::scale_accumulator(factor[i], acc_O[i][n_step]);
            }
        }
        
        float my_sum[2] = {0, 0};
        uint32_t p_reg_u32[4];
        for(int k=0; k<4; ++k) {
            load_s2_reg(&my_max[0], s_tmem, (tid % 64), tid / 64, k * 8);
            
            // Mask padded sequences dynamically avoiding NaN leakage in edge cases spanning across block bounds 
            if (j + (tid / 64)*64 + k*8 < S_len) {
                float v0 = expf(my_max[0] * (1.0f / sqrtf(128.0f)) - new_max[0]);
                float v1 = expf(my_max[1] * (1.0f / sqrtf(128.0f)) - new_max[0]);
                float v2 = expf(my_max[2] * (1.0f / sqrtf(128.0f)) - new_max[1]);
                float v3 = expf(my_max[3] * (1.0f / sqrtf(128.0f)) - new_max[1]);
                
                my_sum[0] += v0 + v1;
                my_sum[1] += v2 + v3;
                
                __nv_bfloat16 bf_v0 = __float2bfloat16(v0);
                __nv_bfloat16 bf_v1 = __float2bfloat16(v1);
                __nv_bfloat16 bf_v2 = __float2bfloat16(v2);
                __nv_bfloat16 bf_v3 = __float2bfloat16(v3);
                
                *(reinterpret_cast<uint32_t*>(&smem_P_bf16[(tid % 64) * 128 + (tid / 64)*64 + k*8 + 0]) ) = __float_as_uint(bf_v0);
                *(reinterpret_cast<uint32_t*>(&smem_P_bf16[(tid % 64) * 128 + (tid / 64)*64 + k*8 + 1]) ) = __float_as_uint(bf_v1);
                *(reinterpret_cast<uint32_t*>(&smem_P_bf16[(tid % 64) * 128 + (tid / 64)*64 + k*8 + 64]) ) = __float_as_uint(bf_v2);
                *(reinterpret_cast<uint32_t*>(&smem_P_bf16[(tid % 64) * 128 + (tid / 64)*64 + k*8 + 65]) ) = __float_as_uint(bf_v3);
                
                p_reg_u32[k] = ((uint32_t)__float_as_uint(bf_v1) << 16) | (uint32_t)__float_as_uint(bf_v0);
            } else {
                p_reg_u32[k] = 0;
            }
        }
        
        d_old[0] += my_sum[0];
        d_old[1] += my_sum[1];
        m_old[0] = new_max[0];
        m_old[1] = new_max[1];
        
        __syncthreads(); 
        
        // Pipeling P evaluation into asynchronous background TMEM transfers mitigating register spill bottlenecks locally
        for(int k=0; k<4; ++k) {
            store_p2_tmem(p_tmem, (tid % 64), tid / 64, p_reg_u32[k]);
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        
        // Chain V multiplication directly leveraging WGMMA capabilities internally mapping optimal layout bounds
        for(int i=0; i<4; ++i) {
            for(int k_step=0; k_step<8; ++k_step) {
                for(int n_step=0; n_step<8; ++n_step) {
                    umma_cg1_16x16(s_tmem + (i * 16) * 4 + n_step * 4, desc_P[i][k_step], desc_V[n_step][k_step], idesc_V, 1);
                }
            }
        }
        commit_and_wait(bar_V, phase_V);
        phase_V ^= 1;
        
        __syncthreads(); 
        
        // Accumulate natively calculated outputs against previously tracked states scaling logic cycles uniformly
        for(int k=0; k<4; ++k) {
            load_o2_reg(&acc_O[0][k], s_tmem, (tid % 64), tid / 64);
            load_o2_reg(&acc_O[1][k], s_tmem, (tid % 64) + 1, tid / 64);
        }
        
        __syncthreads(); 
    }
    
    // Coalesced Epilogue Output
    for (int n_step = 0; n_step < 8; ++n_step) {
        for (int i = 0; i < 2; ++i) {
            wmma::store_matrix_sync(&smem_O[(warp_id * 16 + i * 16) * 128 + n_step * 16], acc_O[i][n_step], 128, wmma::mem_row_major);
        }
    }
    __syncthreads();
    
    for(int i = tid; i < 64 * 128; i += 128) {
        int row = i / 128;
        int col = i % 128;
        if (row < valid_rows) {
            float inv_sum = 1.0f / d_old[row % 2]; 
            float out = smem_O[i] * inv_sum;
            
            __nv_bfloat16* out_ptr = &((__nv_bfloat16*)O)[batch_head * S_len * 128 + (query_start + row) * 128 + col];
            *out_ptr = __float2bfloat16(out);
        }
    }
    
    if (tid < 64) {
        if (tid < valid_rows) {
            LSE[batch_head * S_len + query_start + tid] = m_old[tid % 2] + logf(d_old[tid % 2]);
        }
    }
    
    if (tid == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;" 
                     :: "r"(s_tmem));
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;" 
                     :: "r"(p_tmem));
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    
    CUtensorMap tma_Q, tma_K, tma_V;
    uint64_t S_len = S; 
    
    CUresult res_q = create_tma_3d_descriptor_2B(&tma_Q, Q.data_ptr(), 128, S_len, B * H, 64, 64, 1, CU_TENSOR_MAP_SWIZZLE_128B);
    if (res_q != CUDA_SUCCESS) { fprintf(stderr, "Failed to create tma_Q\n"); exit(1); }
    
    CUresult res_k = create_tma_3d_descriptor_2B(&tma_K, K.data_ptr(), 128, S_len, B * H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B);
    if (res_k != CUDA_SUCCESS) { fprintf(stderr, "Failed to create tma_K\n"); exit(1); }
    
    CUresult res_v = create_tma_3d_descriptor_2B(&tma_V, V.data_ptr(), 128, S_len, B * H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B);
    if (res_v != CUDA_SUCCESS) { fprintf(stderr, "Failed to create tma_V\n"); exit(1); }
    
    dim3 grid(B * H, (S + 63) / 64); 
    dim3 block(128); 
    
    int smem_size = 164352; 
    CUDA_CHECK(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    attention_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V,
        static_cast<__nv_bfloat16*>(O.data_ptr()), 
        static_cast<float*>(LSE.data_ptr()), 
        S
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda