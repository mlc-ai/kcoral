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

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ void tma_load_2d(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_store_2d(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1) : "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void commit_cp_async_bulk(uint64_t* bar) {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
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

__device__ __forceinline__ void tcgen05_fence_before_fn() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void named_barrier_sync_fn(int bar_id, int count) {
    asm volatile("barrier.sync.aligned %0, %1;" :: "r"(bar_id), "r"(count));
}

__device__ __forceinline__ uint64_t make_smem_desc(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16; 
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32; 
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_k_major_128B(void* smem_ptr) {
    return make_smem_desc(smem_ptr, 1, 1024);
}

__device__ __forceinline__ uint64_t make_smem_desc_mn_major_128B(void* smem_ptr) {
    return make_smem_desc(smem_ptr, 8192, 1024);
}

__device__ __forceinline__ uint32_t make_instr_desc_cg2(uint32_t M, uint32_t N, int trans_b) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (0u << 15);   
    d |= ((trans_b & 1) << 16);   
    d |= ((N / 16) << 17);     
    d |= ((M / 32) << 24);    
    return d;
}

__device__ __forceinline__ void umma_f16_cg2_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__global__ void __launch_bounds__(128) attention_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    float* LSE, int S)
{
    extern __shared__ __align__(1024) char smem_pool[];
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)(smem_pool + 0);
    __nv_bfloat16* smem_K0 = (__nv_bfloat16*)(smem_pool + 32768);
    __nv_bfloat16* smem_K1 = (__nv_bfloat16*)(smem_pool + 49152);
    __nv_bfloat16* smem_V0 = (__nv_bfloat16*)(smem_pool + 65536);
    __nv_bfloat16* smem_V1 = (__nv_bfloat16*)(smem_pool + 81920);
    __nv_bfloat16* smem_P = (__nv_bfloat16*)(smem_pool + 98304);
    
    uint64_t* bar = (uint64_t*)(smem_pool + 131072);
    
    uint32_t tmem_addr;
    if (threadIdx.x == 0) {
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], 256;\n"
            :: "r"((uint32_t)__cvta_generic_to_shared(&tmem_addr)));
    }
    __syncthreads();
    
    uint32_t tmem_S_base = tmem_addr;
    uint32_t tmem_O_base = tmem_addr + 128;
    
    int bh = blockIdx.x;
    int m_start_cluster = blockIdx.y * 256;
    uint32_t cluster_rank = cluster_rank_fn();
    int m_start_cta = m_start_cluster + cluster_rank * 128;
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar, 1);
        init_smem_barrier_fn(bar + 1, 1);
        init_smem_barrier_fn(bar + 2, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar, 32768);
        tma_load_2d(&tma_Q, bar, smem_Q, 0, bh * S + m_start_cta);
        tma_load_2d(&tma_Q, bar, smem_Q + 16384, 64, bh * S + m_start_cta);
    }
    mbarrier_wait_fn(bar, 0);
    
    float* m_val = (float*)(smem_pool + 131104);
    float* l_val = (float*)(smem_pool + 131232);
    float* scale_ptr = (float*)(smem_pool + 131360);
    
    if (threadIdx.x < 128) {
        m_val[threadIdx.x] = -1e20f;
        l_val[threadIdx.x] = 0.0f;
    }
    if (threadIdx.x == 0) {
        *scale_ptr = 1.0f / sqrtf(128.0f);
    }
    
    if (threadIdx.x < 128) {
        for (int i = 0; i < 128; i += 2) {
            uint32_t tmem_o_addr = ((m_start_cta + threadIdx.x) << 16) | (128 + i);
            *(float*)(tmem_addr + tmem_o_addr) = 0.0f;
        }
    }
    __syncthreads();
    
    uint32_t scale_u32 = __float_as_uint(*scale_ptr);
    uint32_t idesc_QK = make_instr_desc_cg2(128, 128, 0);
    uint32_t idesc_PV = make_instr_desc_cg2(128, 128, 1);
    
    uint64_t desc_Q0 = make_smem_desc_k_major_128B(smem_Q);
    uint64_t desc_Q1 = make_smem_desc_k_major_128B(smem_Q + 16384);
    uint64_t desc_K0 = make_smem_desc_k_major_128B(smem_K0);
    uint64_t desc_K1 = make_smem_desc_k_major_128B(smem_K1);
    uint64_t desc_V0 = make_smem_desc_mn_major_128B(smem_V0);
    uint64_t desc_V1 = make_smem_desc_mn_major_128B(smem_V1);
    uint64_t desc_P0 = make_smem_desc_k_major_128B(smem_P);
    uint64_t desc_P1 = make_smem_desc_k_major_128B(smem_P + 16384);
    
    int phase_k = 0;
    for (int chunk = 0; chunk < S; chunk += 128) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar + 1, 65536);
            tma_load_2d(&tma_K, bar + 1, smem_K0, 0, bh * S + chunk);
            tma_load_2d(&tma_K, bar + 1, smem_K1, 64, bh * S + chunk);
            tma_load_2d(&tma_V, bar + 1, smem_V0, 0, bh * S + chunk);
            tma_load_2d(&tma_V, bar + 1, smem_V1, 64, bh * S + chunk);
        }
        commit_cp_async_bulk(bar + 1);
        mbarrier_wait_fn(bar + 1, phase_k);
        phase_k ^= 1;
        
        uint32_t tmem_s_addr = ((m_start_cta) << 16);
        tcgen05_fence_before_fn();
        
        if (threadIdx.x == 0) {
            for (int k = 0; k < 4; ++k) {
                uint64_t current_A = desc_Q0 + k * 2;
                uint64_t current_B = desc_K0 + k * 2;
                umma_f16_cg2_fn(tmem_s_addr, current_A, current_B, idesc_QK, 1);
            }
            for (int k = 0; k < 4; ++k) {
                uint64_t current_A = desc_Q1 + k * 2;
                uint64_t current_B = desc_K1 + k * 2;
                umma_f16_cg2_fn(tmem_s_addr, current_A, current_B, idesc_QK, 1);
            }
        }
        tcgen05_fence_after_fn();
        
        float my_max = -1e20f;
        float my_sum = 0.0f;
        int row_idx = threadIdx.x;
        
        uint32_t S_vals[32][4];
        for (uint32_t col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn((m_start_cta << 16) | col, &r0, &r1, &r2, &r3);
            
            uint32_t r0_s = __uint_as_float(r0) < -1e19f ? 0 : __float_as_uint(__uint_as_float(r0) * *scale_ptr);
            uint32_t r1_s = __uint_as_float(r1) < -1e19f ? 0 : __float_as_uint(__uint_as_float(r1) * *scale_ptr);
            uint32_t r2_s = __uint_as_float(r2) < -1e19f ? 0 : __float_as_uint(__uint_as_float(r2) * *scale_ptr);
            uint32_t r3_s = __uint_as_float(r3) < -1e19f ? 0 : __float_as_uint(__uint_as_float(r3) * *scale_ptr);
            
            float f0 = __uint_as_float(r0_s);
            float f1 = __uint_as_float(r1_s);
            float f2 = __uint_as_float(r2_s);
            float f3 = __uint_as_float(r3_s);
            
            my_max = fmaxf(fmaxf(fmaxf(fmaxf(my_max, f0), f1), f2), f3);
            
            S_vals[col/4][0] = r0_s;
            S_vals[col/4][1] = r1_s;
            S_vals[col/4][2] = r2_s;
            S_vals[col/4][3] = r3_s;
        }
        
        float new_m = fmaxf(m_val[row_idx], my_max);
        if (new_m != m_val[row_idx]) {
            for (int i = 0; i < 128; i += 2) {
                uint32_t tmem_o_addr = ((m_start_cta + row_idx) << 16) | (128 + i);
                float* o_ptr = (float*)(tmem_addr + tmem_o_addr);
                *o_ptr *= expf(m_val[row_idx] - new_m);
            }
            l_val[row_idx] *= expf(m_val[row_idx] - new_m);
            m_val[row_idx] = new_m;
        }
        
        for (int i = 0; i < 32; ++i) {
            float p0 = expf(__uint_as_float(S_vals[i][0]) - m_val[row_idx]);
            float p1 = expf(__uint_as_float(S_vals[i][1]) - m_val[row_idx]);
            float p2 = expf(__uint_as_float(S_vals[i][2]) - m_val[row_idx]);
            float p3 = expf(__uint_as_float(S_vals[i][3]) - m_val[row_idx]);
            my_sum += p0 + p1 + p2 + p3;
            
            __nv_bfloat16 bf_p0 = __float2bfloat16(p0);
            __nv_bfloat16 bf_p1 = __float2bfloat16(p1);
            __nv_bfloat16 bf_p2 = __float2bfloat16(p2);
            __nv_bfloat16 bf_p3 = __float2bfloat16(p3);
            
            uint32_t packed0 = ((uint32_t)*reinterpret_cast<uint16_t*>(&bf_p1) << 16) | *reinterpret_cast<uint16_t*>(&bf_p0);
            uint32_t packed1 = ((uint32_t)*reinterpret_cast<uint16_t*>(&bf_p3) << 16) | *reinterpret_cast<uint16_t*>(&bf_p2);
            
            S_vals[i][0] = packed0;
            S_vals[i][1] = packed1;
        }
        l_val[row_idx] += my_sum;
        
        for (int k = 0; k < 4; ++k) {
            for (int i = 0; i < 4; ++i) {
                int c = k * 8 + i;
                int sc = (((c >> 3) ^ (row_idx & 7)) << 3) | (c & 7);
                *(uint32_t*)&smem_P[row_idx * 64 + sc] = S_vals[k * 2 + i / 2][i % 2];
            }
        }
        named_barrier_sync_fn(2, 128);
        
        uint32_t tmem_o_addr = ((m_start_cta) << 16) | 128;
        tcgen05_fence_before_fn();
        if (threadIdx.x == 0) {
            for (int k = 0; k < 4; ++k) {
                uint64_t current_A = desc_P0 + k * 2;
                uint64_t current_B0 = desc_V0 + k * 64;
                umma_f16_cg2_fn(tmem_o_addr, current_A, current_B0, idesc_PV, 1);
                
                uint64_t current_B1 = desc_V1 + k * 64;
                umma_f16_cg2_fn(tmem_o_addr, current_A, current_B1, idesc_PV, 1);
            }
            for (int k = 0; k < 4; ++k) {
                uint64_t current_A = desc_P1 + k * 2;
                uint64_t current_B0 = desc_V0 + k * 64;
                umma_f16_cg2_fn(tmem_o_addr, current_A, current_B0, idesc_PV, 1);
                
                uint64_t current_B1 = desc_V1 + k * 64;
                umma_f16_cg2_fn(tmem_o_addr, current_A, current_B1, idesc_PV, 1);
            }
        }
        tcgen05_fence_after_fn();
    }
    
    if (threadIdx.x < 128) {
        int row_idx = threadIdx.x;
        for (int i = 0; i < 128; i += 2) {
            uint32_t tmem_o_addr = ((m_start_cta + row_idx) << 16) | (128 + i);
            float* o_ptr = (float*)(tmem_addr + tmem_o_addr);
            *o_ptr /= l_val[row_idx];
        }
    }
    __syncthreads();
    
    for (int i = threadIdx.x; i < 128 * 128 / 8; i += blockDim.x) {
        int row_idx = i / 16;
        int col_idx = (i % 16) * 8;
        
        uint32_t tmem_o_addr = ((m_start_cta + row_idx) << 16) | (128 + col_idx);
        uint32_t val = *(uint32_t*)(tmem_addr + tmem_o_addr);
        
        int sc = (((col_idx >> 3) ^ (row_idx & 7)) << 3) | (col_idx & 7);
        *(uint32_t*)&smem_Q[row_idx * 64 + sc] = val;
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        tma_store_2d(&tma_O, smem_Q, 0, bh * S + m_start_cta);
        tma_store_2d(&tma_O, smem_Q + 16384, 64, bh * S + m_start_cta);
        commit_cp_async_bulk(bar + 2);
        tma_store_wait<0>();
    }
    
    if (threadIdx.x < 128) {
        if (m_start_cta + threadIdx.x < S) {
            LSE[bh * S + m_start_cta + threadIdx.x] = m_val[threadIdx.x] + logf(l_val[threadIdx.x]);
        }
    }
    
    if (threadIdx.x == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 256;\n" :: "r"(tmem_addr));
    }
    __syncthreads();
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
    int64_t S = Q.size(2);
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE.data_ptr());
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O;
    if (create_tma_2d_descriptor_2B(&tma_Q, (void*)Q_ptr, 128, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != 0) exit(1);
    if (create_tma_2d_descriptor_2B(&tma_K, (void*)K_ptr, 128, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != 0) exit(1);
    if (create_tma_2d_descriptor_2B(&tma_V, (void*)V_ptr, 128, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != 0) exit(1);
    if (create_tma_2d_descriptor_2B(&tma_O, (void*)O_ptr, 128, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != 0) exit(1);
    
    dim3 grid(B * H, (S + 255) / 256, 1);
    dim3 block(128);
    int smem_size = 139264;
    
    CUDA_CHECK(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    CUDA_CHECK(cudaLaunchKernelEx(&config, attention_kernel, tma_Q, tma_K, tma_V, tma_O, LSE_ptr, S));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda