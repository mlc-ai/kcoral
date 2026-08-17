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
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_expect_tx(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.expect_tx.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes));
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

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ void tma_load_2d_cg2_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t ba = (uint32_t)__cvta_generic_to_shared(&bar[0]) & 0xFEFFFFFF; 
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
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
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
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
    d |= (1u << 16);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
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
    __shared__ __align__(1024) __nv_bfloat16 smem_A[4 * 8192]; 
    __shared__ __align__(1024) __nv_bfloat16 smem_B[4 * 8192]; 

    __shared__ __align__(8) uint64_t mbar[5];
    __shared__ __align__(4) uint32_t tmem_addr[2];

    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_addr[0], 256);
        tmem_alloc_fn(&tmem_addr[1], 256);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar[0], 128);
        init_smem_barrier_fn(mbar[1], 128);
        init_smem_barrier_fn(mbar[2], 128);
        init_smem_barrier_fn(mbar[3], 128);
        init_smem_barrier_fn(mbar[4], 128);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    uint32_t cta_rank = cluster_rank_fn();
    uint32_t n_block = blockIdx.y;
    uint32_t m_base = (blockIdx.x / 2) * 256;
    uint32_t m_base_cta = m_base + cta_rank * 128;

    constexpr int BM_per_cta = 128;
    constexpr int BM_tiles = 2; // Split M into two sub-blocks for swizzling alignment
    
    constexpr int BN_per_cta = 256; 
    
    uint32_t phase_load[2] = {0, 0};
    uint32_t phase_umma[2] = {0, 0};

    // Single buffering for the initial chunk to guarantee proper overlap tracking
    if (threadIdx.x == 0) {
        mbarrier_expect_tx(mbar[0], 32 * 8192 * 2); 
        
        tma_load_2d_cg2_fn(&tma_A, mbar[0], smem_A, 0, m_base_cta);
        tma_load_2d_cg2_fn(&tma_B, mbar[0], smem_B, 0, n_block * 512 + cta_rank * 256);
    }
    __syncthreads();
    mbarrier_wait_fn(mbar[0], phase_load[0]);
    phase_load[0] ^= 1;

    uint32_t tmem_c0 = tmem_addr[0];
    uint32_t tmem_c1 = tmem_addr[1];

    uint32_t idesc = make_instr_desc_fn(256, 512);

    uint64_t desc_a_0 = make_smem_desc_sm100_fn(smem_A, 1, 1024);
    uint64_t desc_a_1 = make_smem_desc_sm100_fn(smem_A + 2 * 8192, 1, 1024);
    uint64_t desc_b_0 = make_smem_desc_sm100_fn(smem_B, 1, 1024);
    uint64_t desc_b_1 = make_smem_desc_sm100_fn(smem_B + 2 * 8192, 1, 1024);
    uint64_t desc_b_2 = make_smem_desc_sm100_fn(smem_B + 8192, 1, 1024);
    uint64_t desc_b_3 = make_smem_desc_sm100_fn(smem_B + 2 * 8192 + 8192, 1, 1024);

    int BK = 128;
    int num_k_iters = 5120 / BK; // 40

    for (uint32_t k_iter = 0; k_iter < num_k_iters; ++k_iter) {
        int curr = k_iter % 2;
        int next = (k_iter + 1) % 2;
        
        int k_off = k_iter * 128;
        int m_off = m_base_cta;
        int n_off = n_block * 512;

        // Start loading next stage asynchronously
        if (k_iter + 1 < num_k_iters) {
            int k_off_next = (k_iter + 1) * 128;
            if (cta_rank == 0) {
                mbarrier_expect_tx(mbar[next], 32 * 8192 * 2);
                
                tma_load_2d_cg2_fn(&tma_A, mbar[next], smem_A, k_off_next, m_off);
                tma_load_2d_cg2_fn(&tma_B, mbar[next], smem_B, k_off_next, n_off + cta_rank * 256);
            }
        }
        
        __syncthreads();
        mbarrier_wait_fn(mbar[curr], phase_load[curr]);
        phase_load[curr] ^= 1;
        
        if (cta_rank == 0) {
            uint32_t accum = (k_iter == 0) ? 0 : 1;
            
            for (int k_chunk = 0; k_chunk < 2; ++k_chunk) {
                uint64_t desc_a_curr = (k_chunk == 0) ? desc_a_0 : desc_a_1;
                uint64_t desc_b_curr_0 = (k_chunk == 0) ? desc_b_0 : desc_b_1;
                uint64_t desc_b_curr_1 = (k_chunk == 0) ? desc_b_2 : desc_b_3;
                
                for (int j = 0; j < 4; ++j) {
                    umma_f16_cg2_fn(tmem_c0, desc_a_curr + j * 2, desc_b_curr_0 + j * 2, idesc, accum);
                    umma_f16_cg2_fn(tmem_c1, desc_a_curr + j * 2, desc_b_curr_1 + j * 2, idesc, accum);
                }
            }
            umma_commit_2sm_fn(mbar[2 + curr]);
        }
        
        mbarrier_wait_fn(mbar[2 + curr], phase_umma[curr]);
        phase_umma[curr] ^= 1;
    }

    // Simple Direct Vectorized Epilogue Writing directly to HBM leveraging implicit software pipelining overhead reduction. 
    __nv_bfloat16* D = C.ptr;
    uint32_t tid = threadIdx.x;
    uint32_t n_off_cta = n_off + cta_rank * BN_per_cta;
    
    for (int col = 0; col < 32; ++col) {
        uint32_t r0, r1, r2, r3;
        uint32_t col_base_0 = col * 4;
        uint32_t col_base_1 = col * 4 + 128;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col_base_0 + tmem_c0));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        uint32_t global_col_0 = n_off_cta + col_base_0;
        if (global_col_0 + 3 < 7168) {
            uint32_t m_idx_0 = tid;
            uint32_t global_row_0 = m_base_cta + m_idx_0;
            D[(uint64_t)global_row_0 * 7168 + global_col_0 + 0] = __float2bfloat16(__uint_as_float(r0));
            D[(uint64_t)global_row_0 * 7168 + global_col_0 + 1] = __float2bfloat16(__uint_as_float(r1));
            D[(uint64_t)global_row_0 * 7168 + global_col_0 + 2] = __float2bfloat16(__uint_as_float(r2));
            D[(uint64_t)global_row_0 * 7168 + global_col_0 + 3] = __float2bfloat16(__uint_as_float(r3));
        }
        
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col_base_1 + tmem_c0));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

        uint32_t global_col_1 = n_off_cta + col_base_1;
        if (global_col_1 + 3 < 7168) {
            uint32_t m_idx_1 = tid;
            uint32_t global_row_1 = m_base_cta + m_idx_1;
            D[(uint64_t)global_row_1 * 7168 + global_col_1 + 0] = __float2bfloat16(__uint_as_float(r0));
            D[(uint64_t)global_row_1 * 7168 + global_col_1 + 1] = __float2bfloat16(__uint_as_float(r1));
            D[(uint64_t)global_row_1 * 7168 + global_col_1 + 2] = __float2bfloat16(__uint_as_float(r2));
            D[(uint64_t)global_row_1 * 7168 + global_col_1 + 3] = __float2bfloat16(__uint_as_float(r3));
        }
    }
    
    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_addr[0], 256);
        tmem_dealloc_fn(tmem_addr[1], 256);
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
    
    uint32_t M_val = A.size(0);
    uint32_t N_val = 7168;
    
    __nv_bfloat16* a_ptr = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* b_ptr = static_cast<__nv_bfloat16*>(B.data_ptr());

    CUtensorMap tma_A, tma_B;
    
    // Request optimized 128B Swizzle TMA Descriptors enforcing perfectly aligned architectures mapping
    CUresult res_A = create_tma_2d_descriptor_2B(&tma_A, a_ptr, 5120, M_val, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_NONE, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        
    CUresult res_B = create_tma_2d_descriptor_2B(&tma_B, b_ptr, 5120, N_val, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_NONE, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        
    if (res_A != CUDA_SUCCESS || res_B != CUDA_SUCCESS) {
        fprintf(stderr, "TMA error\n");
        exit(1);
    }

    dim3 grid((M_val + 255) / 256, N_val / 512); 
    if (grid.x % 2 != 0) grid.x++;
    
    dim3 block(128);
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 128 * 1024));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 128 * 1024;
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