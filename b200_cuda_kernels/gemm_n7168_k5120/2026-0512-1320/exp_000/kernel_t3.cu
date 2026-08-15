#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
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

// ----------------------------------------------------------------------------------
// Helper Functions
// ----------------------------------------------------------------------------------

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ void cluster_sync_fn() {
    asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n" ::: "memory");
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

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t sbo, uint32_t lbo, uint32_t swizzle) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr) >> 4;
    d |= (uint64_t)(addr & 0x3FFF);
    d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32; 
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)swizzle << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);           // c_format = FP32
    d |= (1u << 7);           // a_format = BF16
    d |= (1u << 10);          // b_format = BF16
    d |= (0u << 16);          // Transpose B = 0 (K-Major)
    d |= ((N / 8) << 17);     // n_dim
    d |= ((M / 16) << 24);    // m_dim
    return d;
}

__device__ __forceinline__ void tmem_epilogue_coalesced_4w_fn(
    uint32_t tmem_c,
    __nv_bfloat16* D, __nv_bfloat16* smem_out,
    uint32_t M, uint32_t N, uint32_t m_block, uint32_t n_block,
    uint32_t BM, uint32_t BN) {
    // Phase 1: TMEM -> SMEM
    for (uint32_t col = 0; col < BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        uint32_t addr = tmem_c + col;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(addr));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t base = threadIdx.x * BN + col;
        smem_out[base + 0] = __float2bfloat16(__uint_as_float(r0));
        smem_out[base + 1] = __float2bfloat16(__uint_as_float(r1));
        smem_out[base + 2] = __float2bfloat16(__uint_as_float(r2));
        smem_out[base + 3] = __float2bfloat16(__uint_as_float(r3));
    }
    __syncthreads();
    
    // Phase 2: SMEM -> Global
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = (BM + 3) / 4;
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        if (row >= BM) continue;
        uint32_t global_row = m_block * BM + row;
        uint32_t col_start = lane_id * 4;
        uint32_t global_col = n_block * BN + col_start;
        if (global_row < M && global_col + 3 < N) {
            uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * BN + col_start]);
            *reinterpret_cast<uint2*>(D + (uint64_t)global_row * N + global_col) = data;
        }
    }
}

__device__ __forceinline__ void prefetch_tma_descriptor_fn(const CUtensorMap* d) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(d) : "memory");
}

CUresult create_tma_2d_descriptor_2B(
    CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, 
    uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, 
    CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
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

// ----------------------------------------------------------------------------------
// SMEM Layout
// ----------------------------------------------------------------------------------

struct SharedStorage {
    alignas(1024) __nv_bfloat16 A[2][128][64]; 
    alignas(1024) __nv_bfloat16 B[2][128][64]; 
    alignas(8) uint64_t tma_bar[2];
    alignas(8) uint64_t mma_bar[2];
    alignas(4) uint32_t tmem_addr;
};

// ----------------------------------------------------------------------------------
// Kernel
// ----------------------------------------------------------------------------------

__global__ void gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* __restrict__ C,
    uint32_t M, uint32_t N, uint32_t K) 
{
    extern __shared__ char smem_buf[];
    SharedStorage& shared = *reinterpret_cast<SharedStorage*>(smem_buf);

    if (threadIdx.x < 32) {
        tmem_alloc_fn(&shared.tmem_addr, 128); 
    }
    __syncthreads();
    uint32_t tmem_c = shared.tmem_addr;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&shared.tma_bar[0], 1);
        init_smem_barrier_fn(&shared.tma_bar[1], 1);
        init_smem_barrier_fn(&shared.mma_bar[0], 1);
        init_smem_barrier_fn(&shared.mma_bar[1], 1);
        fence_smem_barrier_init_fn();
        prefetch_tma_descriptor_fn(&tma_A);
        prefetch_tma_descriptor_fn(&tma_B);
    }
    cluster_sync_fn();

    uint32_t logical_m_block = blockIdx.y * 2 + cluster_rank_fn();
    uint32_t logical_n_block = blockIdx.z;

    uint32_t m_base = logical_m_block * 128;
    uint32_t n_base = logical_n_block * 128;

    uint32_t tx_bytes = 128 * 64 * 2 * 2; 
    uint32_t idesc = make_instr_desc_fn(256, 128); 

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&shared.tma_bar[0], tx_bytes);
        tma_load_2d_fn(&tma_A, &shared.tma_bar[0], (void*)&shared.A[0][0][0], 0, m_base);
        tma_load_2d_fn(&tma_B, &shared.tma_bar[0], (void*)&shared.B[0][0][0], 0, n_base);
    }

    uint32_t BK = 64;
    for (uint32_t k = 0; k < K; k += BK) {
        uint32_t iter = k / BK;
        int stage = iter % 2;
        int next_stage = (stage + 1) % 2;
        
        mbarrier_wait_fn(&shared.tma_bar[stage], (iter / 2) % 2);
        
        cluster_sync_fn();
        
        if (k + BK < K) {
            if (iter > 0) {
                mbarrier_wait_fn(&shared.mma_bar[next_stage], ((iter - 1) / 2) % 2);
                cluster_sync_fn();
            }
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&shared.tma_bar[next_stage], tx_bytes);
                tma_load_2d_fn(&tma_A, &shared.tma_bar[next_stage], (void*)&shared.A[next_stage][0][0], k + BK, m_base);
                tma_load_2d_fn(&tma_B, &shared.tma_bar[next_stage], (void*)&shared.B[next_stage][0][0], k + BK, n_base);
            }
        }
        
        if (cluster_rank_fn() == 0 && threadIdx.x == 0) {
            for (uint32_t step = 0; step < 4; ++step) {
                // A is M x K -> K-Major (No Transpose). SBO=1024, LBO=16
                uint64_t desc_a = make_smem_desc_sm100_fn((void*)&shared.A[stage][0][step * 16], 1024, 16, 2);
                // B is N x K, target logic K x N -> K-Major (No Transpose). SBO=1024, LBO=16
                uint64_t desc_b = make_smem_desc_sm100_fn((void*)&shared.B[stage][0][step * 16], 1024, 16, 2);
                uint32_t accum = (iter == 0 && step == 0) ? 0 : 1;
                umma_f16_cg2_fn(tmem_c, desc_a, desc_b, idesc, accum);
            }
            umma_commit_2sm_fn(&shared.mma_bar[stage]);
        }
    }

    uint32_t last_iter = (K / BK) - 1;
    mbarrier_wait_fn(&shared.mma_bar[last_iter % 2], (last_iter / 2) % 2);

    cluster_sync_fn();

    tmem_epilogue_coalesced_4w_fn(
        tmem_c,
        C, (__nv_bfloat16*)&shared, 
        M, N, logical_m_block, logical_n_block, 
        128, 128);

    cluster_sync_fn();
    
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_c, 128);
    }
}

// ----------------------------------------------------------------------------------
// Host Runtime
// ----------------------------------------------------------------------------------

namespace tvm_ffi_gemm {

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);
    
    __nv_bfloat16* A_ptr = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* B_ptr = static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());

    CUtensorMap tma_A, tma_B;
    // A: M x K
    CUresult res_A = create_tma_2d_descriptor_2B(&tma_A, A_ptr, K, M, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res_A != CUDA_SUCCESS) { fprintf(stderr, "TMA A failed\n"); exit(1); }
    
    // B: N x K
    CUresult res_B = create_tma_2d_descriptor_2B(&tma_B, B_ptr, K, N, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res_B != CUDA_SUCCESS) { fprintf(stderr, "TMA B failed\n"); exit(1); }

    int64_t m_blocks = (M + 255) / 256;
    int64_t n_blocks = N / 128;
    
    dim3 grid(2, m_blocks, n_blocks);
    dim3 block(128);
    
    int smem_bytes = sizeof(SharedStorage);
    CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_bytes;
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, tma_A, tma_B, C_ptr, M, N, K));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_gemm::run);

} // namespace tvm_ffi_gemm