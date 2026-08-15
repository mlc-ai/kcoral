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

// Helper functions for SM100
__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ void cluster_sync_fn() {
    asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n" ::: "memory");
}

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx_cluster_fn(uint64_t* bar, uint32_t tx, uint32_t target_cta) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    uint32_t remote_a;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(remote_a) : "r"(a), "r"(target_cta));
    asm volatile("mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%0], %1;"
                 :: "r"(remote_a), "r"(tx));
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

__device__ __forceinline__ void tmem_load_8x_fn(uint32_t col,
    uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
    uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),
                   "=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(col));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
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

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr) >> 4;
    d |= (uint64_t)(addr & 0x3FFF);
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);           
    d |= (1u << 7);           
    d |= (1u << 10);          
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
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, globalAddress, globalDim,
        globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle, l2Promotion, oobFill
    );
}

// ---------------------------------------------------------------------------
// SM100 Compute Kernel 
// ---------------------------------------------------------------------------
constexpr int STAGE = 2;
extern __shared__ __align__(1024) uint8_t smem_pool[];

__global__ void __launch_bounds__(128) cta_gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* __restrict__ C,
    uint32_t M, uint32_t N, uint32_t K) 
{
    setmaxnreg_inc_sync_fn<256>();
    
    uint32_t rank = cluster_rank_fn();
    
    uint32_t tile_n = (blockIdx.x / 2) * 256;
    uint32_t tile_m = blockIdx.y * 256;
    uint32_t tile_m_local = tile_m + (rank == 0 ? 0 : 128);
    uint32_t tile_n_local = tile_n + (rank == 0 ? 0 : 128);
    
    __nv_bfloat16* smem_A_ptr = (__nv_bfloat16*)smem_pool;
    __nv_bfloat16* smem_B_ptr = (__nv_bfloat16*)(smem_pool + 16384 * STAGE);
    
    uint64_t* mbar_load = (uint64_t*)(smem_pool + 32768 * STAGE);
    uint64_t* mbar_commit = (uint64_t*)(smem_pool + 32768 * STAGE + STAGE * 8);

    if (threadIdx.x == 0) {
        for (int s = 0; s < STAGE; ++s) {
            if (rank == 0) {
                init_smem_barrier_fn(&mbar_load[s], 2);
            }
            init_smem_barrier_fn(&mbar_commit[s], 1);
        }
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t tmem_addr;
    __shared__ uint32_t smem_tmem_addr;
    if (threadIdx.x < 32) {
        tmem_alloc_fn(&smem_tmem_addr, 256);
    }
    __syncthreads();
    tmem_addr = smem_tmem_addr;

    // Prologue (TMA Pipeline Prefill)
    for (int s = 0; s < STAGE; ++s) {
        if (threadIdx.x == 0) {
            uint32_t tx_bytes = 32768; // 16KB A + 16KB B (Summing both CTAs' expected TX is 65536)
            mbarrier_arrive_expect_tx_cluster_fn(&mbar_load[s], tx_bytes, 0);
            
            uint32_t k_idx = s * 64;
            if (k_idx < K) {
                tma_load_2d_cg2_fn(&tma_A, &mbar_load[s], smem_A_ptr + s * 8192, k_idx, tile_m_local);
                tma_load_2d_cg2_fn(&tma_B, &mbar_load[s], smem_B_ptr + s * 8192, k_idx, tile_n_local);
            }
        }
    }

    int smem_pipe_read = 0;
    int smem_pipe_write = STAGE;
    uint32_t idesc = make_instr_desc_fn(256, 256);

    // K-Dimension Compute Iteration
    for (uint32_t k = 0; k < K; k += 64) {
        int s_read = smem_pipe_read % STAGE;
        
        if (rank == 0) {
            uint32_t phase_load = (smem_pipe_read / STAGE) & 1;
            mbarrier_wait_fn(&mbar_load[s_read], phase_load);
            
            uint64_t desc_a = make_smem_desc_sm100_fn(smem_A_ptr + s_read * 8192, 1024);
            uint64_t desc_b = make_smem_desc_sm100_fn(smem_B_ptr + s_read * 8192, 1024);
            
            for (int i = 0; i < 4; ++i) {
                uint32_t accum = (k == 0 && i == 0) ? 0 : 1;
                // Add 2 to the descriptor to push the address pointer forward by 32 bytes
                uint64_t d_a = desc_a + (i * 2);
                uint64_t d_b = desc_b + (i * 2);
                umma_f16_cg2_fn(tmem_addr, d_a, d_b, idesc, accum);
            }
            
            umma_commit_2sm_fn(&mbar_commit[s_read]);
        }
        
        uint32_t phase_commit = (smem_pipe_read / STAGE) & 1;
        mbarrier_wait_fn(&mbar_commit[s_read], phase_commit);
        
        if (k + STAGE * 64 < K) {
            int s_write = smem_pipe_write % STAGE;
            if (threadIdx.x == 0) {
                uint32_t tx_bytes = 32768;
                mbarrier_arrive_expect_tx_cluster_fn(&mbar_load[s_write], tx_bytes, 0);
                
                uint32_t k_idx = k + STAGE * 64;
                tma_load_2d_cg2_fn(&tma_A, &mbar_load[s_write], smem_A_ptr + s_write * 8192, k_idx, tile_m_local);
                tma_load_2d_cg2_fn(&tma_B, &mbar_load[s_write], smem_B_ptr + s_write * 8192, k_idx, tile_n_local);
            }
            smem_pipe_write++;
        }
        smem_pipe_read++;
    }

    // Epilogue
    __syncthreads();
    __nv_bfloat16* smem_out = (__nv_bfloat16*)smem_pool;

    // Phase 1: TMEM -> SMEM
    for (uint32_t col = 0; col < 256; col += 8) {
        uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
        uint32_t addr = tmem_addr + col; 
        tmem_load_8x_fn(addr, &r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
        tmem_load_fence_fn(); 
        
        uint32_t smem_base = threadIdx.x * 256 + col;
        smem_out[smem_base + 0] = __float2bfloat16(__uint_as_float(r0));
        smem_out[smem_base + 1] = __float2bfloat16(__uint_as_float(r1));
        smem_out[smem_base + 2] = __float2bfloat16(__float2bfloat16(__uint_as_float(r2)));
        smem_out[smem_base + 3] = __float2bfloat16(__uint_as_float(r3));
        smem_out[smem_base + 4] = __float2bfloat16(__uint_as_float(r4));
        smem_out[smem_base + 5] = __float2bfloat16(__uint_as_float(r5));
        smem_out[smem_base + 6] = __float2bfloat16(__uint_as_float(r6));
        smem_out[smem_base + 7] = __float2bfloat16(__uint_as_float(r7));
    }
    __syncthreads();

    // Phase 2: SMEM -> Global (Coalesced 16-byte vectorized writes)
    for (int i = threadIdx.x; i < 128 * 256 / 8; i += 128) {
        int row = i / 32;
        int col = (i % 32) * 8;
        
        int global_m = tile_m_local + row;
        int global_n = tile_n_local + col;
        
        if (global_m < M && global_n < N) {
            uint4 data = *reinterpret_cast<uint4*>(&smem_out[row * 256 + col]);
            *reinterpret_cast<uint4*>(&C[global_m * N + global_n]) = data;
        }
    }

    cluster_sync_fn();
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_addr, 256);
    }
}

namespace tvm_ffi_gemm {

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);
    
    if (M == 0 || N == 0 || K == 0) return;
    
    CUtensorMap tma_A, tma_B;
    
    if (create_tma_2d_descriptor_2B(&tma_A, A.data_ptr(), K, M, 64, 128, 
                                    CU_TENSOR_MAP_SWIZZLE_128B, 
                                    CU_TENSOR_MAP_L2_PROMOTION_L2_128B, 
                                    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) {
        fprintf(stderr, "TMA A failed\n"); exit(1);
    }
    
    if (create_tma_2d_descriptor_2B(&tma_B, B.data_ptr(), K, N, 64, 128, 
                                    CU_TENSOR_MAP_SWIZZLE_128B, 
                                    CU_TENSOR_MAP_L2_PROMOTION_L2_128B, 
                                    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) {
        fprintf(stderr, "TMA B failed\n"); exit(1);
    }
    
    int grid_x = 2 * (N / 256);
    int grid_y = (M + 255) / 256;
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(128, 1, 1);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 65568; 
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, cta_gemm_kernel, tma_A, tma_B, static_cast<__nv_bfloat16*>(C.data_ptr()), (uint32_t)M, (uint32_t)N, (uint32_t)K));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_gemm::run);

} // namespace tvm_ffi_gemm