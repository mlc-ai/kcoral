#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>
#include <math.h>
#include <stdio.h>
#include <cstdint>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        fprintf(stderr, "CU error %d at %s:%d\n",                 \
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)

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

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ void tma_store_3d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [%0, {%2, %3, %4}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t taddr, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(taddr));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)0 << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, int a_major, int b_major) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= ((uint32_t)a_major << 15);
    d |= ((uint32_t)b_major << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

CUresult create_tma_3d_descriptor_2B(
    CUtensorMap* d, void* globalAddress, 
    uint64_t D, uint64_t S, uint64_t B_H, 
    uint32_t box_D, uint32_t box_S, uint32_t box_B_H
) {
    cuuint64_t globalDim[3] = {D, S, B_H};
    cuuint64_t globalStrides[2] = {D * 2, D * S * 2};
    cuuint32_t boxDim[3] = {box_D, box_S, box_B_H};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        3, 
        globalAddress,
        globalDim,
        globalStrides,
        boxDim, 
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

__global__ void mha_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    float* __restrict__ LSE,
    int B, int H, int S
) {
    extern __shared__ __align__(128) char dynamic_smem[];
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)dynamic_smem;           
    __nv_bfloat16* smem_K = smem_Q + 128 * 128;                     
    __nv_bfloat16* smem_V = smem_K + 128 * 128;                     
    __nv_bfloat16* smem_P = smem_V + 128 * 128;                     
    float* smem_O = (float*)(smem_P + 128 * 128);                   
    
    uint64_t* mbar_tma = (uint64_t*)(smem_O + 128 * 128);           
    uint64_t* mbar_umma = mbar_tma + 1;                             
    uint32_t* tmem_addr = (uint32_t*)(mbar_umma + 1);               

    int tid = threadIdx.x;
    int b = blockIdx.z;
    int h = blockIdx.y;
    int q_start = blockIdx.x * 128;

    if (tid < 32) {
        if (tid == 0) {
            init_smem_barrier_fn(mbar_tma, 1);
            init_smem_barrier_fn(mbar_umma, 1);
        }
        tmem_alloc_cg1_fn(tmem_addr, 128); 
    }
    __syncthreads();
    fence_smem_barrier_init_fn();
    
    for (int i = 0; i < 128; i++) {
        smem_O[tid * 128 + i] = 0.0f;
    }

    float my_m = -INFINITY;
    float my_l = 0.0f;

    int tma_phase = 0;

    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_tma, 32768);
        tma_load_3d_fn(&tma_Q, mbar_tma, smem_Q, 0, q_start, b * H + h);
    }
    mbarrier_wait_fn(mbar_tma, tma_phase);
    tma_phase ^= 1;

    uint32_t tmem = *tmem_addr;
    int umma_phase = 0;

    for (int kv_start = 0; kv_start < S; kv_start += 128) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_tma, 65536); 
            tma_load_3d_fn(&tma_K, mbar_tma, smem_K, 0, kv_start, b * H + h);
            tma_load_3d_fn(&tma_V, mbar_tma, smem_V, 0, kv_start, b * H + h);
        }
        mbarrier_wait_fn(mbar_tma, tma_phase);
        tma_phase ^= 1;
        
        // Q @ K^T
        for (int k = 0; k < 128; k += 16) {
            uint64_t desc_Q = make_smem_desc_sm100_fn(smem_Q + k, 16, 2048);
            uint64_t desc_K = make_smem_desc_sm100_fn(smem_K + k, 16, 2048);
            uint32_t idesc = make_instr_desc_fn(128, 128, 0, 0); // Q: K-Major, K: K-Major
            uint32_t accum = (k == 0) ? 0 : 1;
            
            if (tid == 0) {
                asm volatile(
                    "{\n.reg .pred p;\n"
                    "setp.ne.b32 p, %4, 0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                    :: "r"(tmem), "l"(desc_Q), "l"(desc_K), "r"(idesc), "r"(accum));
            }
        }
        
        if (tid == 0) {
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(mbar_umma)));
        }
        mbarrier_wait_fn(mbar_umma, umma_phase);
        umma_phase ^= 1;
        
        // Softmax & P compute
        uint32_t r_vals[128];
        for (int c = 0; c < 128; c += 4) {
            tmem_load_4x_fn(tmem + c, &r_vals[c], &r_vals[c+1], &r_vals[c+2], &r_vals[c+3]);
        }
        tmem_load_fence_fn();

        float row_max = -INFINITY;
        for (int c = 0; c < 128; c++) {
            float f = __uint_as_float(r_vals[c]) * 0.0883883476f;
            if (kv_start + c < S) row_max = fmaxf(row_max, f);
        }
        
        float m_new = (q_start + tid < S) ? fmaxf(my_m, row_max) : -INFINITY;
        float exp_diff = (q_start + tid < S) ? __expf(my_m - m_new) : 0.0f;
        my_m = m_new;
        
        float row_sum = 0.0f;
        for (int c = 0; c < 128; c++) {
            float p = 0.0f;
            if (q_start + tid < S && kv_start + c < S) {
                p = __expf(__uint_as_float(r_vals[c]) * 0.0883883476f - m_new);
            }
            row_sum += p;
            smem_P[tid * 128 + c] = __float2bfloat16(p);
        }
        my_l = my_l * exp_diff + row_sum;
        
        for (int c = 0; c < 128; c++) {
            smem_O[tid * 128 + c] *= exp_diff;
        }
        
        fence_async_shared_fn(); 
        __syncthreads();
        
        // P @ V
        for (int k = 0; k < 128; k += 16) {
            uint64_t desc_P = make_smem_desc_sm100_fn(smem_P + k, 16, 2048);
            uint64_t desc_V = make_smem_desc_sm100_fn(smem_V + k * 128, 2048, 16);
            uint32_t idesc = make_instr_desc_fn(128, 128, 0, 1); // P: K-Major, V: MN-Major
            uint32_t accum = (k == 0) ? 0 : 1;
            
            if (tid == 0) {
                asm volatile(
                    "{\n.reg .pred p;\n"
                    "setp.ne.b32 p, %4, 0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                    :: "r"(tmem), "l"(desc_P), "l"(desc_V), "r"(idesc), "r"(accum));
            }
        }
        
        if (tid == 0) {
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(mbar_umma)));
        }
        mbarrier_wait_fn(mbar_umma, umma_phase);
        umma_phase ^= 1;
        
        uint32_t r_vals2[128];
        for (int c = 0; c < 128; c += 4) {
            tmem_load_4x_fn(tmem + c, &r_vals2[c], &r_vals2[c+1], &r_vals2[c+2], &r_vals2[c+3]);
        }
        tmem_load_fence_fn();
        
        for (int c = 0; c < 128; c++) {
            smem_O[tid * 128 + c] += __uint_as_float(r_vals2[c]);
        }
    }
    
    if (q_start + tid < S) {
        float inv_l = (my_l > 0.0f) ? (1.0f / my_l) : 0.0f;
        for (int c = 0; c < 128; c++) {
            float out_val = smem_O[tid * 128 + c] * inv_l;
            smem_P[tid * 128 + c] = __float2bfloat16(out_val);
        }
        int64_t lse_offset = (int64_t)b * H * S + (int64_t)h * S + q_start + tid;
        LSE[lse_offset] = my_m + logf(my_l);
    }
    
    fence_async_shared_fn(); 
    __syncthreads();
    
    if (tid == 0) {
        tma_store_3d_fn(&tma_O, smem_P, 0, q_start, b * H + h);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }
    __syncthreads();

    if (tid < 32) {
        tmem_dealloc_cg1_fn(tmem, 128);
    }
}

namespace tvm_ffi_mha {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);

    const __nv_bfloat16* q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* k_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* v_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* o_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_data = static_cast<float*>(LSE.data_ptr());

    CUtensorMap tma_Q, tma_K, tma_V, tma_O;
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_Q, (void*)q_data, 128, S, B * H, 128, 128, 1));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_K, (void*)k_data, 128, S, B * H, 128, 128, 1));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_V, (void*)v_data, 128, S, B * H, 128, 128, 1));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_O, (void*)o_data, 128, S, B * H, 128, 128, 1));

    dim3 block(128);
    dim3 grid((S + 127) / 128, H, B);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    int dynamic_smem = 196736; 

    CUDA_CHECK(cudaFuncSetAttribute(
        mha_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        dynamic_smem
    ));

    mha_kernel<<<grid, block, dynamic_smem, stream>>>(
        tma_Q, tma_K, tma_V, tma_O, lse_data,
        B, H, S
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}  // namespace tvm_ffi_mha