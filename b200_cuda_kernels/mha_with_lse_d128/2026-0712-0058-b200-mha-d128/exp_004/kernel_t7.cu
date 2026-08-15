#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
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

__device__ __forceinline__ uint32_t get_shmem_addr(void* ptr) {
    return (uint32_t)__cvta_generic_to_shared(ptr);
}

__device__ __forceinline__ uint32_t swizzle_128B(uint32_t row, uint32_t col_elements) {
    const uint32_t stride_elements = 64;
    return row * stride_elements + (((row & 7) ^ (col_elements >> 3)) << 3) + (col_elements & 7);
}

__device__ __forceinline__ uint64_t make_smem_desc(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = get_shmem_addr(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16; 
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32; 
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   // SWIZZLE_128B
    return d;
}

template<int A_MAJOR, int B_MAJOR, int M, int N>
__device__ __forceinline__ uint32_t make_instr_desc() {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= ((A_MAJOR & 1) << 15);   
    d |= ((B_MAJOR & 1) << 16);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ void umma_f16_cg1(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void init_mbar(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(bytes));
}

__device__ __forceinline__ void wait_mbar(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase));
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_store_2d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_store_fence_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
}

CUresult create_tma_2d_descriptor_BF16(CUtensorMap* d, void* globalAddress, 
                                uint64_t gmem_dim0, uint64_t gmem_dim1,
                                uint32_t box_dim0, uint32_t box_dim1, CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[2] = {gmem_dim0, gmem_dim1};
    cuuint64_t globalStrides[1] = {gmem_dim0 * 2};
    cuuint32_t boxDim[2] = {box_dim0, box_dim1};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_before_fn() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__global__ __launch_bounds__(128) void AttentionKernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    float* LSE_ptr,
    int S)
{
    int b_outer = blockIdx.y;
    int s_offset_q = blockIdx.x * 64;
    int tid = threadIdx.x;
    int cta_id = cluster_rank_fn();
    int s_offset_cta = s_offset_q + cta_id * 64;

    extern __shared__ __align__(1024) char smem[];
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)smem;                
    __nv_bfloat16* smem_K = smem_Q + 64 * 128;                   
    __nv_bfloat16* smem_V = smem_K + 64 * 128;                    
    __nv_bfloat16* smem_P = smem_V + 64 * 128;                      
    __nv_bfloat16* smem_O_bf = smem_P + 64 * 64;
    
    uint32_t tmem_S, tmem_O;
    
    if (tid == 0) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 64;" : : "r"(get_shmem_addr(&tmem_S)));
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 128;" : : "r"(get_shmem_addr(&tmem_O)));
        
        uint64_t* bar = (uint64_t*)(smem + 49152);
        init_mbar(bar, 1);
    }
    __syncthreads();

    uint64_t* bar = (uint64_t*)(smem + 49152);
    int phase = 0;
    
    if (tid == 0) {
        arrive_expect_tx(bar, 16384);
        tma_load_2d_fn(&tma_Q, bar, smem_Q, 0, b_outer * S + s_offset_cta);
        tma_load_2d_fn(&tma_Q, bar, smem_Q + 4096, 64, b_outer * S + s_offset_cta);
    }
    wait_mbar(bar, phase);
    phase ^= 1;
    __syncthreads();

    float global_max_val = -1e20f;
    float sum_val = 0.0f;
    float scale = 1.0f / sqrtf(128.0f);
    
    // Critical Fix: Explicitly define Q as A-major, and K as A-major (which effectively acts as B-major in Q @ K^T)
    uint32_t idesc_qkt = make_instr_desc<0, 0, 64, 64>();
    // P is A-major (row-major), V is B-major (column-major) -> standard matmul layout mapping
    uint32_t idesc_pv = make_instr_desc<0, 1, 64, 128>();

    int num_S_blocks = (S + 63) / 64;
    for (int j = 0; j < num_S_blocks; j++) {
        int s_kv = j * 64;
        
        if (tid == 0) {
            arrive_expect_tx(bar, 32768);
            tma_load_2d_fn(&tma_K, bar, smem_K, 0, b_outer * S + s_kv);
            tma_load_2d_fn(&tma_K, bar, smem_K + 4096, 64, b_outer * S + s_kv);
            tma_load_2d_fn(&tma_V, bar, smem_V, 0, b_outer * S + s_kv);
            tma_load_2d_fn(&tma_V, bar, smem_V + 4096, 64, b_outer * S + s_kv);
        }
        wait_mbar(bar, phase);
        phase ^= 1;
        __syncthreads();

        fence_proxy_async_fn();
        
        if (tid == 0) {
            uint64_t desc_Q = make_smem_desc(smem_Q, 1, 1024);
            uint64_t desc_K = make_smem_desc(smem_K, 1, 1024);
            
            uint32_t accum_local = 0;
            for(int k = 0; k < 8; k++) {
                uint64_t step_Q = desc_Q + k * 2;
                uint64_t step_K = desc_K + k * 2;
                umma_f16_cg1(tmem_S, step_Q, step_K, idesc_qkt, accum_local);
                accum_local = 1;
            }
            tcgen05_fence_before_fn();
        }
        __syncthreads();
        
        float local_max_val = -1e20f;
        
        if (tid < 128) {
            for(int i = 0; i < 64; i += 4) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_S + (tid << 16) + i));
                
                float s0 = __uint_as_float(r0) * scale;
                float s1 = __uint_as_float(r1) * scale;
                float s2 = __uint_as_float(r2) * scale;
                float s3 = __uint_as_float(r3) * scale;
                
                if (s_kv + i + 0 >= S) s0 = -1e20f;
                if (s_kv + i + 1 >= S) s1 = -1e20f;
                if (s_kv + i + 2 >= S) s2 = -1e20f;
                if (s_kv + i + 3 >= S) s3 = -1e20f;
                
                if (s0 > local_max_val) local_max_val = s0;
                if (s1 > local_max_val) local_max_val = s1;
                if (s2 > local_max_val) local_max_val = s2;
                if (s3 > local_max_val) local_max_val = s3;
            }
        }
        
        __shared__ float local_max_all[128];
        if (tid < 128) local_max_all[tid] = local_max_val;
        __syncthreads();
        local_max_val = local_max_all[tid];
        
        float local_sum_exp = 0.0f;
        if (tid < 128) {
            for(int i = 0; i < 64; i += 4) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_S + (tid << 16) + i));
                
                float s0 = __uint_as_float(r0) * scale;
                float s1 = __uint_as_float(r1) * scale;
                float s2 = __uint_as_float(r2) * scale;
                float s3 = __uint_as_float(r3) * scale;
                
                if (s_kv + i + 0 >= S) s0 = -1e20f;
                if (s_kv + i + 1 >= S) s1 = -1e20f;
                if (s_kv + i + 2 >= S) s2 = -1e20f;
                if (s_kv + i + 3 >= S) s3 = -1e20f;
                
                float p0 = fast_exp2f_fn((s0 - local_max_val) * 1.4426950408889634f);
                float p1 = fast_exp2f_fn((s1 - local_max_val) * 1.4426950408889634f);
                float p2 = fast_exp2f_fn((s2 - local_max_val) * 1.4426950408889634f);
                float p3 = fast_exp2f_fn((s3 - local_max_val) * 1.4426950408889634f);
                
                local_sum_exp += (p0 + p1 + p2 + p3);
            }
        }
        
        __shared__ float local_sum_all[128];
        if (tid < 128) local_sum_all[tid] = local_sum_exp;
        __syncthreads();
        local_sum_exp = local_sum_all[tid];
        
        float new_global_max = global_max_val;
        if (local_max_val > new_global_max) new_global_max = local_max_val;
        
        float alpha = fast_exp2f_fn((global_max_val - new_global_max) * 1.4426950408889634f);
        sum_val *= alpha;
        
        float lse_scale_factor = fast_exp2f_fn((local_max_val - new_global_max) * 1.4426950408889634f);
        sum_val += local_sum_exp * lse_scale_factor;
        
        global_max_val = new_global_max;
        
        if (tid < 128) {
            for(int i = 0; i < 64; i += 4) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_S + (tid << 16) + i));
                
                float s0 = __uint_as_float(r0) * scale;
                float s1 = __uint_as_float(r1) * scale;
                float s2 = __uint_as_float(r2) * scale;
                float s3 = __uint_as_float(r3) * scale;
                
                if (s_kv + i + 0 >= S) s0 = -1e20f;
                if (s_kv + i + 1 >= S) s1 = -1e20f;
                if (s_kv + i + 2 >= S) s2 = -1e20f;
                if (s_kv + i + 3 >= S) s3 = -1e20f;
                
                float p0 = fast_exp2f_fn((s0 - local_max_val) * 1.4426950408889634f) * lse_scale_factor;
                float p1 = fast_exp2f_fn((s1 - local_max_val) * 1.4426950408889634f) * lse_scale_factor;
                float p2 = fast_exp2f_fn((s2 - local_max_val) * 1.4426950408889634f) * lse_scale_factor;
                float p3 = fast_exp2f_fn((s3 - local_max_val) * 1.4426950408889634f) * lse_scale_factor;
                
                smem_P[swizzle_128B(tid, i)] = __float2bfloat16(p0);
                smem_P[swizzle_128B(tid, i+1)] = __float2bfloat16(p1);
                smem_P[swizzle_128B(tid, i+2)] = __float2bfloat16(p2);
                smem_P[swizzle_128B(tid, i+3)] = __float2bfloat16(p3);
            }
        }
        __syncthreads();
        
        fence_proxy_async_fn();
        
        if (tid == 0) {
            uint64_t desc_P = make_smem_desc(smem_P, 1, 1024);
            uint64_t desc_V = make_smem_desc(smem_V, 8192, 1024);
            
            uint32_t accum = 1;
            for(int k = 0; k < 4; k++) {
                uint64_t step_P = desc_P + k * 2;
                uint64_t step_V = desc_V + k * 128;
                umma_f16_cg1(tmem_O, step_P, step_V, idesc_pv, accum);
            }
            tcgen05_fence_before_fn();
        }
        __syncthreads();
    }
    
    __syncthreads();
    
    if (tid < 128) {
        for(int i = 0; i < 64; i += 4) {
            uint32_t r0, r1, r2, r3;
            if (tid < 64) {
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_O + (tid << 16) + i));
            } else {
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_O + ((tid - 64) << 16) + i + 64));
            }
            
            float f0 = __uint_as_float(r0) / sum_val;
            float f1 = __uint_as_float(r1) / sum_val;
            float f2 = __uint_as_float(r2) / sum_val;
            float f3 = __uint_as_float(r3) / sum_val;
            
            if (tid < 64) {
                smem_O_bf[swizzle_128B(tid, i)] = __float2bfloat16(f0);
                smem_O_bf[swizzle_128B(tid, i+1)] = __float2bfloat16(f1);
                smem_O_bf[swizzle_128B(tid, i+2)] = __float2bfloat16(f2);
                smem_O_bf[swizzle_128B(tid, i+3)] = __float2bfloat16(f3);
            } else {
                smem_O_bf[swizzle_128B(tid-64, i+64)] = __float2bfloat16(f0);
                smem_O_bf[swizzle_128B(tid-64, i+65)] = __float2bfloat16(f1);
                smem_O_bf[swizzle_128B(tid-64, i+66)] = __float2bfloat16(f2);
                smem_O_bf[swizzle_128B(tid-64, i+67)] = __float2bfloat16(f3);
            }
        }
    }
    __syncthreads();
    
    if (tid == 0) {
        tma_store_fence_fn();
        tma_store_2d_fn(&tma_O, smem_O_bf, 0, b_outer * S + s_offset_cta);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }
    
    if (tid < 64) {
        int real_s = s_offset_cta + tid;
        if (real_s < S) {
            LSE_ptr[b_outer * S + real_s] = global_max_val + logf(sum_val);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id)); 
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE.data_ptr());
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O;
    if (create_tma_2d_descriptor_BF16(&tma_Q, (void*)Q_ptr, D, B*H*S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B) != CUDA_SUCCESS ||
        create_tma_2d_descriptor_BF16(&tma_K, (void*)K_ptr, D, B*H*S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B) != CUDA_SUCCESS ||
        create_tma_2d_descriptor_BF16(&tma_V, (void*)V_ptr, D, B*H*S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B) != CUDA_SUCCESS ||
        create_tma_2d_descriptor_BF16(&tma_O, (void*)O_ptr, D, B*H*S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B) != CUDA_SUCCESS) {
        fprintf(stderr, "TMA descriptor creation failed\n");
        exit(1);
    }
    
    int num_S_blocks = (S + 63) / 64;
    dim3 grid(num_S_blocks, B * H);
    dim3 block(128);
    
    CUDA_CHECK(cudaFuncSetAttribute(AttentionKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 49152));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 49152;
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, AttentionKernel, tma_Q, tma_K, tma_V, tma_O, LSE_ptr, S));
    CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}  // namespace tvm_ffi_example_cuda