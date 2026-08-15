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

namespace tvm_ffi_gemm {

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

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
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

__device__ __forceinline__ void umma_f16_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
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

__global__ void cta_gemm_kernel( 
    const __grid_constant__ CUtensorMap tma_A, 
    const __grid_constant__ CUtensorMap tma_B, 
    void* C_void,
    uint32_t M, uint32_t N) 
{
    extern __shared__ __align__(1024) char smem_pool[];

    __nv_bfloat16* A_smem_raw = (__nv_bfloat16*)smem_pool;
    __nv_bfloat16* B_smem_raw = (__nv_bfloat16*)(smem_pool + 16384);
    __nv_bfloat16* A_smem[2] = {A_smem_raw, A_smem_raw + 8192}; 
    __nv_bfloat16* B_smem[2] = {B_smem_raw, B_smem_raw + 16384}; 

    uint64_t* full_barriers = (uint64_t*)(smem_pool + 49152);
    uint64_t* empty_barriers = (uint64_t*)(smem_pool + 49168);

    uint32_t tmem_c_addr;
    if (threadIdx.x < 32) { 
        tmem_alloc_fn(&tmem_c_addr, 128);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&full_barriers[0], 1);
        init_smem_barrier_fn(&full_barriers[1], 1);
        init_smem_barrier_fn(&empty_barriers[0], 1);
        init_smem_barrier_fn(&empty_barriers[1], 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t cta_n = blockIdx.x; 
    uint32_t cta_m = blockIdx.y;
    uint32_t n_base = cta_n * 128;
    uint32_t m_base = cta_m * 128;

    if (threadIdx.x == 0) {
        uint32_t tx_bytes = 8192 + 16384;
        mbarrier_arrive_and_expect_tx_fn(&full_barriers[0], tx_bytes);
        tma_load_2d_fn(&tma_A, &full_barriers[0], A_smem[0], 0, m_base);
        tma_load_2d_fn(&tma_B, &full_barriers[0], B_smem[0], 0, n_base);
    }

    uint32_t idesc = make_instr_desc_fn(64, 128);

    for (uint32_t step_idx = 0; step_idx < 80; step_idx++) {
        uint32_t stage = step_idx % 2;
        uint32_t next_stage = (step_idx + 1) % 2;

        mbarrier_wait_fn(&full_barriers[stage], (step_idx / 2) & 1);
        fence_proxy_async_fn();
        __syncthreads();

        if (step_idx + 1 < 80) {
            if (threadIdx.x == 0) {
                uint32_t next_k = (step_idx + 1) * 64;
                uint32_t tx_bytes = 8192 + 16384;
                mbarrier_arrive_and_expect_tx_fn(&full_barriers[next_stage], tx_bytes);
                
                tma_load_2d_fn(&tma_A, &full_barriers[next_stage], A_smem[next_stage], next_k, m_base);
                tma_load_2d_fn(&tma_B, &full_barriers[next_stage], B_smem[next_stage], next_k, n_base);
            }
        }

        uint64_t desc_a = make_smem_desc_sm100_fn(A_smem[stage], 1, 1024);
        uint64_t desc_b = make_smem_desc_sm100_fn(B_smem[stage], 1, 1024);

        for (uint32_t k = 0; k < 4; ++k) {
            uint32_t accum = (step_idx == 0 && k == 0) ? 0 : 1;
            umma_f16_fn(tmem_c_addr, desc_a, desc_b, idesc, accum);
            desc_a += 2;
            desc_b += 2;
        }
        
        if (threadIdx.x == 0) {
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];"
                :: "r"((uint32_t)__cvta_generic_to_shared(&empty_barriers[stage])));
        }
        mbarrier_wait_fn(&empty_barriers[stage], ((step_idx - stage) / 2) & 1);
        __syncthreads();
    }

    __nv_bfloat16* C = (__nv_bfloat16*)C_void;
    for (uint32_t col = 0; col < 128; col += 4) {
        uint32_t taddr = tmem_c_addr + col;
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(taddr));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        uint32_t m_idx = m_base + threadIdx.x;
        if (m_idx < M) {
            uint32_t nc = n_base + col;
            if (nc + 3 < N) {
                __nv_bfloat16 vals[4];
                vals[0] = __float2bfloat16(__uint_as_float(r0));
                vals[1] = __float2bfloat16(__uint_as_float(r1));
                vals[2] = __float2bfloat16(__uint_as_float(r2));
                vals[3] = __float2bfloat16(__uint_as_float(r3));
                
                uint2 data = *reinterpret_cast<uint2*>(vals);
                *reinterpret_cast<uint2*>(C + (uint64_t)m_idx * N + nc) = data;
            }
        }
    }

    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_c_addr, 128);
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id)); 

    uint32_t M = A.size(0);
    constexpr uint32_t N = 7168;
    constexpr uint32_t K = 5120;

    CUtensorMap tma_A, tma_B;
    create_tma_2d_descriptor_2B(&tma_A, A.data_ptr(), K, M, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_NONE, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    create_tma_2d_descriptor_2B(&tma_B, B.data_ptr(), K, N, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_NONE, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    uint32_t grid_x = (N + 127) / 128;
    uint32_t grid_y = (M + 127) / 128;
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(128, 1, 1);
    
    uint32_t smem_size = 50 * 1024;
    CUDA_CHECK(cudaFuncSetAttribute(cta_gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_size;
    config.stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    cudaLaunchKernelEx(&config, cta_gemm_kernel, tma_A, tma_B, C.data_ptr(), M, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(config.stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_gemm