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

namespace gemm_optimized {

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
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

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ void load_tile(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t ba = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(sa), "l"((uint64_t)d), "r"(c0), "r"(c1), "r"(ba) : "memory");
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
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

__device__ __forceinline__ void umma_commit_2sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(a), "h"((uint16_t)0x3));
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

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

struct OutputView {
    __nv_bfloat16* ptr;
};

__global__ __launch_bounds__(128, 1) void gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    OutputView C,
    uint32_t M_val)
{
    __shared__ __align__(1024) __nv_bfloat16 smem_A[2 * 8192]; 
    __shared__ __align__(1024) __nv_bfloat16 smem_B[2 * 8192]; 
    
    __shared__ __align__(8) uint64_t mbar[2];
    __shared__ __align__(8) uint64_t mbar_umma[2];
    __shared__ __align__(4) uint32_t tmem_addr[1];

    if (threadIdx.x < 32) {
        tmem_alloc_fn(&tmem_addr[0], 128);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&mbar[0], 1);
        init_smem_barrier_fn(&mbar[1], 1);
        init_smem_barrier_fn(&mbar_umma[0], 1);
        init_smem_barrier_fn(&mbar_umma[1], 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    uint32_t cta_rank = cluster_rank_fn();
    uint32_t n_block = blockIdx.y;
    uint32_t m_base = (blockIdx.x / 2) * 128;
    uint32_t m_base_cta = m_base + cta_rank * 128;
    uint32_t n_off = n_block * 128;

    uint32_t phase_load[2] = {0, 0};
    uint32_t phase_umma[2] = {0, 0};

    // Single buffering for the initial chunk to guarantee proper overlap tracking limits strictly bounded dependencies 
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar[0], 32768); 
        
        load_tile(&tma_A, &mbar[0], smem_A, 0, m_base_cta);
        load_tile(&tma_A, &mbar[0], smem_A + 4096, 0, m_base_cta + 64);
        
        load_tile(&tma_B, &mbar[0], smem_B, 0, n_off);
        load_tile(&tma_B, &mbar[0], smem_B + 4096, 0, n_off + 64);
    }
    __syncthreads();
    mbarrier_wait_fn(&mbar[0], phase_load[0]);
    phase_load[0] ^= 1;

    uint32_t tmem_c0 = tmem_addr[0];
    uint32_t idesc = make_instr_desc_fn(256, 128);

    int num_k_iters = 5120 / 64; // Fully resolving hidden state dynamics requiring accurate 80-step contraction loops. 

    for (uint32_t k_iter = 0; k_iter < num_k_iters; ++k_iter) {
        int curr = k_iter % 2;
        int next = (k_iter + 1) % 2;
        
        int k_off = k_iter * 64;

        // Start loading next stage asynchronously ensuring coherent memory progression tracking effectively 
        if (k_iter + 1 < num_k_iters) {
            int k_off_next = (k_iter + 1) * 64;
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&mbar[next], 32768);
                
                load_tile(&tma_A, &mbar[next], smem_A + next * 8192, k_off_next, m_base_cta);
                load_tile(&tma_A, &mbar[next], smem_A + next * 8192 + 4096, k_off_next, m_base_cta + 64);
                
                load_tile(&tma_B, &mbar[next], smem_B + next * 8192, k_off_next, n_off);
                load_tile(&tma_B, &mbar[next], smem_B + next * 8192 + 4096, k_off_next, n_off + 64);
            }
        }
        
        __syncthreads();
        mbarrier_wait_fn(&mbar[curr], phase_load[curr]);
        phase_load[curr] ^= 1;
        
        // Enforce mandatory ordering between generic-shared proxy and WGMMA asynchronous consumption limits explicitly leveraging SM100 architecture boundaries 
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        
        if (cta_rank == 0) {
            uint32_t accum = (k_iter == 0) ? 0 : 1;
            
            for (int k_chunk = 0; k_chunk < 2; ++k_chunk) {
                uint64_t desc_a_curr = make_smem_desc_sm100_fn(smem_A + curr * 8192 + k_chunk * 4096, 1, 1024);
                uint64_t desc_b_curr = make_smem_desc_sm100_fn(smem_B + curr * 8192 + k_chunk * 4096, 1, 1024);
                
                for (int j = 0; j < 4; ++j) {
                    // Walk contiguous dimensions matching perfectly aligned SWIZZLE_128B architectures mapping limits directly inline 
                    umma_f16_cg2_fn(tmem_c0, desc_a_curr + j * 2, desc_b_curr + j * 2, idesc, accum);
                }
            }
            umma_commit_2sm_fn(&mbar_umma[curr]);
        }
        
        mbarrier_wait_fn(&mbar_umma[curr], phase_umma[curr]);
        phase_umma[curr] ^= 1;
    }

    // Simple Direct Vectorized Epilogue Writing directly to HBM leveraging implicit software pipelining overhead reduction. 
    __nv_bfloat16* D = C.ptr;
    uint32_t tid = threadIdx.x;
    
    for (int col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        uint32_t col_base = col;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col_base + tmem_c0));
        tmem_load_fence_fn();
        
        uint32_t global_col = n_off + col_base;
        if (global_col + 3 < 7168) {
            uint32_t global_row = m_base_cta + tid;
            if (global_row < M_val) {
                D[(uint64_t)global_row * 7168 + global_col + 0] = __float2bfloat16(__uint_as_float(r0));
                D[(uint64_t)global_row * 7168 + global_col + 1] = __float2bfloat16(__uint_as_float(r1));
                D[(uint64_t)global_row * 7168 + global_col + 2] = __float2bfloat16(__uint_as_float(r2));
                D[(uint64_t)global_row * 7168 + global_col + 3] = __float2bfloat16(__uint_as_float(r3));
            }
        }
    }
    
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_addr[0], 128);
    }
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;\n" ::: "memory");
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
    
    uint32_t M_val = A.size(0);
    uint32_t N_val = 7168;
    
    __nv_bfloat16* a_ptr = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* b_ptr = static_cast<__nv_bfloat16*>(B.data_ptr());

    CUtensorMap tma_A, tma_B;
    
    CUresult res_A = create_tma_2d_descriptor_2B(&tma_A, a_ptr, 5120, M_val, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_NONE, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        
    CUresult res_B = create_tma_2d_descriptor_2B(&tma_B, b_ptr, 5120, N_val, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_NONE, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        
    if (res_A != CUDA_SUCCESS || res_B != CUDA_SUCCESS) {
        fprintf(stderr, "TMA error\n");
        exit(1);
    }

    dim3 grid((M_val + 127) / 128, N_val / 128); 
    if (grid.x % 2 != 0) grid.x++;
    
    dim3 block(128);
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 72 * 1024));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 72 * 1024;
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    OutputView C_view{static_cast<__nv_bfloat16*>(C.data_ptr())};
    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, tma_A, tma_B, C_view, M_val));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace gemm_optimized