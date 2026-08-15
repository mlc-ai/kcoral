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

// --- Helper Functions ---

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

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

__device__ __forceinline__ void tma_load_multicast_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, uint16_t mask) {
    uint64_t cache_hint = 0;
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%0], [%1, {%4, %5}], [%2], %3, %6;"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "h"(mask), "r"(c0), "r"(c1), "l"(cache_hint) : "memory");
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

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
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
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= (0u << 15);   // a_major = 0 (K-Major)
    d |= (0u << 16);   // b_major = 0 (K-Major)
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ void tmem_epilogue_coalesced_4w_fn_fixed(
    __nv_bfloat16* D, __nv_bfloat16* smem_out,
    uint32_t M, uint32_t N, uint32_t m_block, uint32_t n_block,
    uint32_t BM, uint32_t BN, uint32_t tmem_base) {
    
    // Phase 1: TMEM -> SMEM
    for (uint32_t col = 0; col < BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        uint32_t taddr = tmem_base + col;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(taddr));
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
        
        for (uint32_t c = col_start; c < BN; c += 128) {
            uint32_t global_col = n_block * BN + c;
            if (global_row < M) {
                if (global_col + 3 < N) {
                    uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * BN + c]);
                    *reinterpret_cast<uint2*>(D + (uint64_t)global_row * N + global_col) = data;
                } else {
                    for (int i = 0; i < 4; ++i) {
                        if (global_col + i < N) {
                            D[(uint64_t)global_row * N + global_col + i] = smem_out[row * BN + c + i];
                        }
                    }
                }
            }
        }
    }
}

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, const void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2, 
        (void*)globalAddress,
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

// --- Kernel and Host Code ---

struct ComputeStorage {
    __align__(128) __nv_bfloat16 A[3][128 * 64];
    __align__(128) __nv_bfloat16 B[3][256 * 64];
    __align__(8) uint64_t full_barriers[3];
    __align__(8) uint64_t empty_barriers[3];
};

union SharedStorage {
    ComputeStorage compute;
    __align__(128) __nv_bfloat16 C[128 * 256];
};

extern __shared__ __align__(128) uint8_t smem[];

__global__ void gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K) 
{
    setmaxnreg_inc_sync_fn<256>();
    
    SharedStorage& shared = *reinterpret_cast<SharedStorage*>(smem);
    
    int cluster_m = blockIdx.x / 2;
    int cta_rank = cluster_rank_fn();
    int cluster_n = blockIdx.y;
    
    int m_start = cluster_m * 256 + cta_rank * 128;
    int n_start = cluster_n * 256;
    int b_half_off = cta_rank * 128;
    
    int K_STEPS = (K + 63) / 64;
    
    int phase_full_arr[3] = {0, 0, 0};
    int phase_empty_arr[3] = {0, 0, 0};
    
    __shared__ uint32_t tmem_c_smem;
    if (threadIdx.x < 32) {
        tmem_alloc_fn(&tmem_c_smem, 256);
    }
    __syncthreads();
    uint32_t tmem_c = tmem_c_smem;
    
    if (threadIdx.x == 0) {
        for (int i = 0; i < 3; ++i) {
            init_smem_barrier_fn(&shared.compute.full_barriers[i], 1);
            init_smem_barrier_fn(&shared.compute.empty_barriers[i], 1);
            mbarrier_arrive_fn(&shared.compute.empty_barriers[i]); 
        }
    }
    __syncthreads();
    
    // Prologue
    for (int stage = 0; stage < 3 && stage < K_STEPS; ++stage) {
        if (threadIdx.x == 0) {
            mbarrier_wait_fn(&shared.compute.empty_barriers[stage], phase_empty_arr[stage]);
            phase_empty_arr[stage] ^= 1;
            
            mbarrier_arrive_and_expect_tx_fn(&shared.compute.full_barriers[stage], 49152);
            
            tma_load_2d_fn(&tma_A, &shared.compute.full_barriers[stage], shared.compute.A[stage], stage * 64, m_start);
            tma_load_multicast_2d_fn(&tma_B, &shared.compute.full_barriers[stage], 
                                     (void*)(shared.compute.B[stage] + b_half_off * 64), 
                                     stage * 64, n_start + b_half_off, 0x3);
        }
    }
    
    // Main Loop
    for (int k_iter = 0; k_iter < K_STEPS; ++k_iter) {
        int stage = k_iter % 3;
        
        if (threadIdx.x == 0) {
            mbarrier_wait_fn(&shared.compute.full_barriers[stage], phase_full_arr[stage]);
        }
        __syncthreads();
        phase_full_arr[stage] ^= 1;
        
        fence_proxy_async_fn();
        
        if (cta_rank == 0 && threadIdx.x == 0) {
            uint64_t desc_a = make_smem_desc_sm100_fn(shared.compute.A[stage], 0, 1024);
            uint64_t desc_b = make_smem_desc_sm100_fn(shared.compute.B[stage], 0, 1024);
            uint32_t idesc = make_instr_desc_fn(256, 256);
            uint32_t accum = (k_iter == 0) ? 0 : 1;
            
            umma_f16_cg2_fn(tmem_c, desc_a, desc_b, idesc, accum);
            umma_commit_2sm_fn(&shared.compute.empty_barriers[stage]);
        }
        
        // Issue Next TMA
        int next_k_iter = k_iter + 3;
        if (next_k_iter < K_STEPS) {
            int next_stage = next_k_iter % 3;
            int next_k = next_k_iter * 64;
            if (threadIdx.x == 0) {
                mbarrier_wait_fn(&shared.compute.empty_barriers[next_stage], phase_empty_arr[next_stage]);
                phase_empty_arr[next_stage] ^= 1;
                
                mbarrier_arrive_and_expect_tx_fn(&shared.compute.full_barriers[next_stage], 49152);
                tma_load_2d_fn(&tma_A, &shared.compute.full_barriers[next_stage], shared.compute.A[next_stage], next_k, m_start);
                tma_load_multicast_2d_fn(&tma_B, &shared.compute.full_barriers[next_stage], 
                                         (void*)(shared.compute.B[next_stage] + b_half_off * 64), 
                                         next_k, n_start + b_half_off, 0x3);
            }
        }
    }
    
    // Epilogue Sync
    __syncthreads();
    if (threadIdx.x == 0 && K_STEPS > 0) {
        int last_stage = (K_STEPS - 1) % 3;
        mbarrier_wait_fn(&shared.compute.empty_barriers[last_stage], phase_empty_arr[last_stage]);
    }
    __syncthreads();
    tcgen05_fence_after_fn();
    
    // Store C
    tmem_epilogue_coalesced_4w_fn_fixed(
        C, shared.C,
        M, N,
        cluster_m * 2 + cta_rank, cluster_n,
        128, 256, tmem_c
    );
    
    __syncthreads();
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_c, 256);
    }
}

namespace tvm_ffi_example_cuda {

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);
    
    const __nv_bfloat16* A_ptr = static_cast<const __nv_bfloat16*>(A.data_ptr());
    const __nv_bfloat16* B_ptr = static_cast<const __nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());
    
    CUtensorMap tma_A, tma_B;
    
    CUresult res_A = create_tma_2d_descriptor_2B(
        &tma_A, A_ptr, K, M, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    if (res_A != CUDA_SUCCESS) {
        fprintf(stderr, "TMA A failed\n");
        exit(1);
    }
    
    CUresult res_B = create_tma_2d_descriptor_2B(
        &tma_B, B_ptr, K, N, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    if (res_B != CUDA_SUCCESS) {
        fprintf(stderr, "TMA B failed\n");
        exit(1);
    }
    
    int cluster_size = 2;
    int M_BLOCK = 256; 
    int N_BLOCK = 256; 
    
    int grid_x = ((M + M_BLOCK - 1) / M_BLOCK) * cluster_size;
    int grid_y = (N + N_BLOCK - 1) / N_BLOCK;
    int grid_z = 1;
    
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(128, 1, 1);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = sizeof(SharedStorage);
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = cluster_size;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaFuncSetAttribute((void*)gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedStorage)));
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, tma_A, tma_B, A_ptr, B_ptr, C_ptr, M, N, K));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda