#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
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

namespace tvm_ffi_gemm {

using namespace nvcuda;

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
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

__device__ __forceinline__ uint64_t modify_smem_addr(uint64_t desc, uint32_t offset_bytes) {
    uint32_t addr = (desc >> 0) & 0x3FFFF;
    addr += (offset_bytes >> 4);
    uint64_t new_desc = (desc & ~((uint64_t)0x3FFFF)) | (addr & 0x3FFFF);
    return new_desc;
}

__global__ __launch_bounds__(128, 2)
void gemm_kernel(__grid_constant__ const CUtensorMap tma_A, 
                 __grid_constant__ const CUtensorMap tma_B,
                 __nv_bfloat16* C, int M, int N, int K) {
    
    int m_start = blockIdx.x * 128;
    int n_start = (blockIdx.y * 64) + (cluster_rank_fn() * 64);
    int m_local_start = m_start + cluster_rank_fn() * 64;
    
    extern __shared__ __align__(1024) uint8_t smem[];
    uint8_t* A_smem[2] = {smem, smem + 16384};
    uint8_t* B_smem_half0[2] = {smem + 32768, smem + 49152};
    uint8_t* B_smem_half1[2] = {smem + 40960, smem + 57344};
    uint64_t* bar = (uint64_t*)(smem + 65536);
    uint32_t* p_C_tmem_addr = (uint32_t*)(smem + 65560);
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&bar[0], 1);
        init_smem_barrier_fn(&bar[1], 1);
        init_smem_barrier_fn(&bar[2], 1);
        init_smem_barrier_fn(&bar[3], 1);
        init_smem_barrier_fn(&bar[4], 1);
        tmem_alloc_fn(p_C_tmem_addr, 128);
    }
    __syncthreads();
    
    uint32_t C_tmem_base = p_C_tmem_addr[0];
    uint32_t C_tmem_addr = C_tmem_base; 
    
    int phase_A[2] = {0, 0};
    int phase_B[2] = {0, 0};
    int phase_C = 0;
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&bar[0], 16384);
        tma_load_2d_fn(&tma_A, &bar[0], A_smem[0], 0, m_local_start);
        
        mbarrier_arrive_and_expect_tx_fn(&bar[1], 16384);
        tma_load_2d_fn(&tma_B, &bar[1], B_smem_half0[0], 0, n_start);
        tma_load_2d_fn(&tma_B, &bar[1], B_smem_half1[0], 0, n_start);
    }
    
    for (int k_iter = 0; k_iter < K; k_iter += 64) {
        int stage = (k_iter / 64) % 2;
        int next_stage = 1 - stage;
        
        mbarrier_wait_fn(&bar[stage == 0 ? 0 : 2], phase_A[stage] & 1);
        phase_A[stage]++;
        
        if (k_iter + 64 < K) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&bar[next_stage == 0 ? 0 : 2], 16384);
                tma_load_2d_fn(&tma_A, &bar[next_stage == 0 ? 0 : 2], A_smem[next_stage], k_iter + 64, m_local_start);
            }
        }
        
        mbarrier_wait_fn(&bar[stage == 0 ? 1 : 3], phase_B[stage] & 1);
        phase_B[stage]++;
        
        if (k_iter + 64 < K) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&bar[next_stage == 0 ? 1 : 3], 16384);
                tma_load_2d_fn(&tma_B, &bar[next_stage == 0 ? 1 : 3], B_smem_half0[next_stage], k_iter + 64, n_start);
                tma_load_2d_fn(&tma_B, &bar[next_stage == 0 ? 1 : 3], B_smem_half1[next_stage], k_iter + 64, n_start);
            }
        }
        
        __syncthreads();
        
        uint64_t desc_A = make_smem_desc_sm100_fn(A_smem[stage], 1, 1024);
        uint64_t desc_B_half0 = make_smem_desc_sm100_fn(B_smem_half0[stage], 1, 1024);
        uint64_t desc_B_half1 = make_smem_desc_sm100_fn(B_smem_half1[stage], 1, 1024);
        uint32_t idesc = make_instr_desc_fn(128, 64);
        
        for(int i = 0; i < 4; i++) {
            uint32_t row_base = cluster_rank_fn() * 64;
            uint32_t c_addr = ((row_base + threadIdx.x) << 16) | (i * 16);
            uint64_t a_desc = modify_smem_addr(desc_A, i * 32);
            uint64_t b_desc = modify_smem_addr(desc_B_half0, i * 32);
            uint32_t accum = (k_iter == 0 && i == 0) ? 0 : 1;
            
            umma_f16_cg1_fn(c_addr, a_desc, b_desc, idesc, accum);
        }
        
        for(int i = 0; i < 4; i++) {
            uint32_t row_base = cluster_rank_fn() * 64;
            uint32_t c_addr = ((row_base + threadIdx.x) << 16) | 64 + (i * 16);
            uint64_t a_desc = modify_smem_addr(desc_A, i * 32);
            uint64_t b_desc2 = modify_smem_addr(desc_B_half1, i * 32);
            uint32_t accum = (k_iter == 0 && i == 0) ? 0 : 1;
            
            umma_f16_cg1_fn(c_addr, a_desc, b_desc2, idesc, accum);
        }
        
        if (threadIdx.x == 0) {
            umma_commit_1sm_fn(&bar[4]);
        }
        mbarrier_wait_fn(&bar[4], phase_C & 1);
        phase_C++;
        __syncthreads();
    }
    
    asm volatile("fence.proxy.async;" ::: "memory");
    __syncthreads();
    
    for (uint32_t col = 0; col < 64; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(((threadIdx.x << 16) | col)));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        uint32_t global_m = m_local_start + threadIdx.x;
        uint32_t global_n = n_start + col;
        
        if (global_m < M && global_n + 3 < N) {
            C[(uint64_t)global_m * N + global_n + 0] = __float2bfloat16(__uint_as_float(r0));
            C[(uint64_t)global_m * N + global_n + 1] = __float2bfloat16(__uint_as_float(r1));
            C[(uint64_t)global_m * N + global_n + 2] = __float2bfloat16(__uint_as_float(r2));
            C[(uint64_t)global_m * N + global_n + 3] = __float2bfloat16(__uint_as_float(r3));
        }
    }
    
    for (uint32_t col = 64; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(((threadIdx.x << 16) | col)));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        uint32_t global_m = m_local_start + threadIdx.x;
        uint32_t global_n = n_start + col;
        
        if (global_m < M && global_n + 3 < N) {
            C[(uint64_t)global_m * N + global_n + 0] = __float2bfloat16(__uint_as_float(r0));
            C[(uint64_t)global_m * N + global_n + 1] = __float2bfloat16(__uint_as_float(r1));
            C[(uint64_t)global_m * N + global_n + 2] = __float2bfloat16(__uint_as_float(r2));
            C[(uint64_t)global_m * N + global_n + 3] = __float2bfloat16(__uint_as_float(r3));
        }
    }
    
    if (threadIdx.x == 0) {
        tmem_dealloc_fn(C_tmem_addr, 128);
    }
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

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    int64_t M_val = A.size(0);
    int64_t K_val = A.size(1);
    int64_t N_val = B.size(0);
    
    CUtensorMap tma_A, tma_B;
    __nv_bfloat16* a_ptr = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* b_ptr = static_cast<__nv_bfloat16*>(B.data_ptr());
    
    create_tma_2d_descriptor_2B(&tma_A, a_ptr, K_val, M_val, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_B, b_ptr, K_val, N_val, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    dim3 grid((M_val + 127) / 128, (N_val + 63) / 64);
    dim3 block(128);
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 65536;
    config.stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaFuncSetAttribute(tvm_ffi_gemm::gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536));

    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, tma_A, tma_B, static_cast<__nv_bfloat16*>(C.data_ptr()), M_val, N_val, K_val));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(config.stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_gemm::run);

}  // namespace tvm_ffi_gemm