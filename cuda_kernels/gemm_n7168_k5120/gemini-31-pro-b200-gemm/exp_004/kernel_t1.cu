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

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        fprintf(stderr, "CU error %d at %s:%d\n",                 \
                (int)_e, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

// -------------------------------------------------------------------------
// SM100 Helper Functions
// -------------------------------------------------------------------------
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

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
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

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
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

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16; 
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32; 
    d |= (uint64_t)1 << 46;  
    
    // Compute base offset for 128B swizzle
    uint32_t base_offset = (addr >> 7) & 0x7;
    d |= (uint64_t)(base_offset) << 49;
    
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

__device__ __forceinline__ void umma_f16_cg2_fn(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
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

__device__ __forceinline__ void tmem_epilogue_coalesced_4w_fixed_fn(
    __nv_bfloat16* D, __nv_bfloat16* smem_out,
    uint32_t M, uint32_t N, uint32_t m_block, uint32_t n_block,
    uint32_t BM, uint32_t BN) {
    
    // Phase 1: TMEM -> SMEM
    for (uint32_t col = 0; col < BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col));
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
        
        for (uint32_t c = 0; c < BN; c += 128) {
            uint32_t col_start = c + lane_id * 4;
            if (col_start < BN) {
                uint32_t global_col = n_block * BN + col_start;
                if (global_row < M && global_col + 3 < N) {
                    uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * BN + col_start]);
                    *reinterpret_cast<uint2*>(D + (uint64_t)global_row * N + global_col) = data;
                }
            }
        }
    }
}

// -------------------------------------------------------------------------
// Kernel and Host Setup
// -------------------------------------------------------------------------

struct SharedStorage {
    __align__(1024) uint8_t A[4][16384]; 
    __align__(1024) uint8_t B[4][16384];
    __align__(128)  __nv_bfloat16 epilogue[128][256];
    __align__(8)    uint64_t bar_produce[4];
    __align__(8)    uint64_t bar_consume[4];
    __align__(16)   uint32_t tmem_addr;
};

__global__ void gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* C,
    uint32_t M, uint32_t N, uint32_t K
) {
    setmaxnreg_inc_sync_fn<256>();

    extern __shared__ __align__(1024) uint8_t smem_raw[];
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_raw);

    uint32_t cluster_id_x = blockIdx.x / 2;
    uint32_t cluster_id_y = blockIdx.y;
    uint32_t rank = cluster_rank_fn();

    uint32_t c_m = cluster_id_y * 256;
    uint32_t c_n = cluster_id_x * 256;

    if (c_m >= M) return;

    if (threadIdx.x < 32) {
        tmem_alloc_fn(&smem.tmem_addr, 256);
    }
    __syncthreads();
    uint32_t tmem_addr = smem.tmem_addr;

    uint64_t desc_A[4], desc_B[4];
    if (rank == 0) {
        for(int i = 0; i < 4; i++) {
            desc_A[i] = make_smem_desc_sm100_fn(smem.A[i], 1, 1024);
            desc_B[i] = make_smem_desc_sm100_fn(smem.B[i], 1, 1024);
        }
    }
    uint32_t instr_desc = make_instr_desc_fn(256, 256);

    if (threadIdx.x == 0) {
        for (int i = 0; i < 4; ++i) {
            init_smem_barrier_fn(&smem.bar_produce[i], 2);
            init_smem_barrier_fn(&smem.bar_consume[i], 1);
        }
    }
    fence_smem_barrier_init_fn();
    __syncthreads();
    cluster_sync_fn();

    uint32_t k_iters = K / 64;
    uint32_t stages = 4;
    uint32_t stage_produce = stages;
    uint32_t stage_consume = 0;
    uint32_t phase_produce_wait[4] = {0, 0, 0, 0};
    uint32_t phase_consume_wait[4] = {0, 0, 0, 0};

    // Prologue
    for (uint32_t i = 0; i < stages; ++i) {
        if (i < k_iters) {
            if (threadIdx.x == 0) {
                if (rank == 0) {
                    mbarrier_arrive_and_expect_tx_fn(&smem.bar_produce[i], 32768);
                } else {
                    mbarrier_arrive_expect_tx_cluster_fn(&smem.bar_produce[i], 32768, 0);
                }
                tma_load_2d_cg2_fn(&tma_A, &smem.bar_produce[i], smem.A[i], i * 64, c_m + rank * 128);
                tma_load_2d_cg2_fn(&tma_B, &smem.bar_produce[i], smem.B[i], i * 64, c_n + rank * 128);
            }
        }
    }

    // Main loop
    for (uint32_t k = 0; k < k_iters; ++k) {
        int sc = stage_consume % stages;
        
        if (rank == 0 && threadIdx.x == 0) {
            mbarrier_wait_fn(&smem.bar_produce[sc], phase_produce_wait[sc]);
            phase_produce_wait[sc] ^= 1;
            
            uint32_t accum = (k > 0) ? 1 : 0;
            umma_f16_cg2_fn(tmem_addr, desc_A[sc], desc_B[sc], instr_desc, accum);
            umma_commit_2sm_fn(&smem.bar_consume[sc]);
        }
        
        if (stage_produce < k_iters) {
            int sp = stage_produce % stages;
            if (threadIdx.x == 0) {
                mbarrier_wait_fn(&smem.bar_consume[sp], phase_consume_wait[sp]);
                phase_consume_wait[sp] ^= 1;
                
                if (rank == 0) {
                    mbarrier_arrive_and_expect_tx_fn(&smem.bar_produce[sp], 32768);
                } else {
                    mbarrier_arrive_expect_tx_cluster_fn(&smem.bar_produce[sp], 32768, 0);
                }
                tma_load_2d_cg2_fn(&tma_A, &smem.bar_produce[sp], smem.A[sp], stage_produce * 64, c_m + rank * 128);
                tma_load_2d_cg2_fn(&tma_B, &smem.bar_produce[sp], smem.B[sp], stage_produce * 64, c_n + rank * 128);
            }
            stage_produce++;
        }
        stage_consume++;
    }

    if (threadIdx.x == 0) {
        int last_sc = (k_iters - 1) % stages;
        mbarrier_wait_fn(&smem.bar_consume[last_sc], phase_consume_wait[last_sc]);
    }
    __syncthreads();
    tcgen05_fence_after_fn();

    tmem_epilogue_coalesced_4w_fixed_fn(
        C, &smem.epilogue[0][0],
        M, N, 
        cluster_id_y * 2 + rank, 
        cluster_id_x,            
        128, 256                 
    );

    cluster_sync_fn();
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_addr, 256);
    }
}

inline CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
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

namespace gemm_n7168_k5120 {

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0); 

    CUtensorMap tma_A, tma_B;
    
    CU_CHECK(create_tma_2d_descriptor_2B(
        &tma_A, A.data_ptr(), K, M, 64, 128,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
    ));

    CU_CHECK(create_tma_2d_descriptor_2B(
        &tma_B, B.data_ptr(), K, N, 64, 128,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
    ));

    int num_n = (N + 255) / 256;
    int num_m = (M + 255) / 256;
    dim3 grid(num_n * 2, num_m, 1);
    dim3 block(128, 1, 1);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = sizeof(SharedStorage);
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, tma_A, tma_B, static_cast<__nv_bfloat16*>(C.data_ptr()), M, N, K));
    CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace gemm_n7168_k5120