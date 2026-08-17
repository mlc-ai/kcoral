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

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        const char* err_str;                                       \
        cuGetErrorString(_e, &err_str);                            \
        fprintf(stderr, "CU error %s at %s:%d\n",                 \
                err_str, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

// -------------------------------------------------------------------------
// Architecture / Hardware Utility Functions
// -------------------------------------------------------------------------

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

__device__ __forceinline__ uint32_t pack_bf16_fn(uint32_t fp32_a, uint32_t fp32_b) {
    __nv_bfloat162 res = __floats2bfloat162_rn(__uint_as_float(fp32_a), __uint_as_float(fp32_b));
    return *reinterpret_cast<uint32_t*>(&res);
}

// -------------------------------------------------------------------------
// High Performance Vectorized Epilogue
// -------------------------------------------------------------------------

__device__ __forceinline__ void tmem_epilogue_coalesced_4w_fn(
    __nv_bfloat16* D, __nv_bfloat16* smem_out,
    uint32_t M, uint32_t N, uint32_t m_base, uint32_t n_base,
    uint32_t tmem_c, uint32_t BM, uint32_t BN) {
    
    uint32_t BN_pad = BN + 8; // Prevent shared memory bank conflicts

    // Phase 1: TMEM -> SMEM (Bank conflict-free dump)
    for (uint32_t col = 0; col < BN; col += 8) {
        uint32_t r[8];
        uint32_t taddr = tmem_c + col;
        // x8 load shape extracts 8 lanes from TMEM instantly
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                     : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),
                       "=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]) : "r"(taddr));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        uint32_t base_words = (threadIdx.x * BN_pad + col) / 2;
        ((uint32_t*)smem_out)[base_words + 0] = pack_bf16_fn(r[0], r[1]);
        ((uint32_t*)smem_out)[base_words + 1] = pack_bf16_fn(r[2], r[3]);
        ((uint32_t*)smem_out)[base_words + 2] = pack_bf16_fn(r[4], r[5]);
        ((uint32_t*)smem_out)[base_words + 3] = pack_bf16_fn(r[6], r[7]);
    }
    __syncthreads();
    
    // Phase 2: SMEM -> GMEM (Coalesced and vectorized float4 stores)
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    
    for (uint32_t row = warp_id; row < BM; row += 4) {
        uint32_t global_row = m_base + row;
        if (global_row >= M) continue;
        
        uint32_t col = lane_id * 8; // Lane id dictates the 16 bytes chunk
        uint32_t global_col = n_base + col;
        
        if (global_col + 7 < N) {
            float4 data = *reinterpret_cast<float4*>(&smem_out[row * BN_pad + col]);
            *reinterpret_cast<float4*>(D + (uint64_t)global_row * N + global_col) = data;
        } else {
            // Guarded unrolled element-wise tail loop (If N is unaligned)
            for(int i = 0; i < 8; ++i) {
                if (global_col + i < N) {
                    D[(uint64_t)global_row * N + global_col + i] = smem_out[row * BN_pad + col + i];
                }
            }
        }
    }
}

// -------------------------------------------------------------------------
// Main Asynchronous GEMM Kernel
// -------------------------------------------------------------------------

constexpr int STAGE = 4;
constexpr int BM = 128;
constexpr int BN_half = 128;
constexpr int BK = 64;

__global__ void __launch_bounds__(128) gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* C,
    int M, int N, int K) {

    // Expand register pool mapping immediately to bypass standard register limitations for latency hiding
    asm volatile("setmaxnreg.inc.sync.aligned.u32 248;" ::: "memory");

    extern __shared__ uint8_t smem_raw[];
    uintptr_t raw_ptr = (uintptr_t)smem_raw;
    uintptr_t base_ptr = (raw_ptr + 1023) & ~1023; // 1024-byte alignment mandatory for SM100 TMA
    uint8_t* smem_base = smem_raw + (base_ptr - raw_ptr);

    __nv_bfloat16* A_smem[STAGE];
    __nv_bfloat16* B_smem[STAGE];
    for (int i = 0; i < STAGE; ++i) {
        A_smem[i] = (__nv_bfloat16*)(smem_base + i * 16384);
        B_smem[i] = (__nv_bfloat16*)(smem_base + STAGE * 16384 + i * 16384);
    }
    
    uint64_t* mbar_tma = (uint64_t*)(smem_base + STAGE * 32768);
    uint64_t* mbar_umma = mbar_tma + STAGE;

    int cluster_rank = cluster_rank_fn();
    int cluster_x = blockIdx.x / 2;
    int cluster_y = blockIdx.y;

    int m_base = cluster_x * 256 + cluster_rank * 128;
    int n_base_half = cluster_y * 256 + cluster_rank * 128;
    int n_base = cluster_y * 256;

    __shared__ uint32_t tmem_c_ptr;
    if (threadIdx.x < 32) {
        tmem_alloc_fn(&tmem_c_ptr, 256); // Yields 128x256 TMEM block
    }
    
    if (threadIdx.x == 0) {
        for(int i = 0; i < STAGE; ++i) {
            init_smem_barrier_fn(&mbar_tma[i], 1);
            init_smem_barrier_fn(&mbar_umma[i], 1);
        }
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    uint32_t tmem_c = tmem_c_ptr;
    uint32_t idesc = make_instr_desc_fn(256, 256);

    int tma_phase[STAGE] = {0};
    int umma_phase[STAGE] = {0};

    // Prologue Pipeline (Fills STAGE - 1)
    for (int i = 0; i < STAGE - 1; ++i) {
        int k = i * BK;
        if (k < K) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&mbar_tma[i], 32768);
                tma_load_2d_fn(&tma_A, &mbar_tma[i], A_smem[i], k, m_base);
                tma_load_2d_fn(&tma_B, &mbar_tma[i], B_smem[i], k, n_base_half);
            }
        }
    }

    // Main Compute Loop
    for (int k = 0; k < K; k += BK) {
        int stage = (k / BK) % STAGE;
        int next_k = k + (STAGE - 1) * BK;
        int next_stage = (stage + STAGE - 1) % STAGE;

        mbarrier_wait_fn(&mbar_tma[stage], tma_phase[stage]);
        cluster_sync_fn(); // Barrier ensures safe cross-cluster SMEM B data usage

        if (next_k < K) {
            if (k >= BK) {
                // Stall next TMA copy into buffer until Math releases it
                mbarrier_wait_fn(&mbar_umma[next_stage], umma_phase[next_stage]);
                umma_phase[next_stage] ^= 1;
            }
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&mbar_tma[next_stage], 32768);
                tma_load_2d_fn(&tma_A, &mbar_tma[next_stage], A_smem[next_stage], next_k, m_base);
                tma_load_2d_fn(&tma_B, &mbar_tma[next_stage], B_smem[next_stage], next_k, n_base_half);
            }
        }

        // Only Leader CTA commits UMMA instructions 
        if (cluster_rank == 0) {
            if (threadIdx.x == 0) {
                for (int k_step = 0; k_step < BK / 16; ++k_step) {
                    uint64_t desc_A = make_smem_desc_sm100_fn(A_smem[stage], 1024);
                    uint64_t desc_B = make_smem_desc_sm100_fn(B_smem[stage], 1024);
                    
                    // Hardware naturally supports implicit matrix walking via +32 address offset tracking
                    desc_A += k_step * 2;
                    desc_B += k_step * 2;
                    
                    int accum = (k == 0 && k_step == 0) ? 0 : 1;
                    umma_f16_cg2_fn(tmem_c, desc_A, desc_B, idesc, accum);
                }
                umma_commit_2sm_fn(&mbar_umma[stage]);
            }
        }

        tma_phase[stage] ^= 1;
    }

    // Cleanly deplete the final executing pipeline tracks
    for (int step = 1; step < STAGE; ++step) {
        int k_issued = K - step * BK;
        if (k_issued >= 0) {
            int stage = (k_issued / BK) % STAGE;
            mbarrier_wait_fn(&mbar_umma[stage], umma_phase[stage]);
            umma_phase[stage] ^= 1;
        }
    }

    cluster_sync_fn();
    
    // Dump resultant TMEM 128x256 to C Global Memory
    tmem_epilogue_coalesced_4w_fn(C, (__nv_bfloat16*)smem_base, M, N, m_base, n_base, tmem_c, 128, 256);

    cluster_sync_fn();
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_c, 256);
    }
}

namespace tvm_ffi_gemm_cuda {

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0); 

    CUtensorMap tma_A, tma_B;
    
    CU_CHECK(create_tma_2d_descriptor_2B(
        &tma_A, A.data_ptr(), K, M, BK, BM,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));

    CU_CHECK(create_tma_2d_descriptor_2B(
        &tma_B, B.data_ptr(), K, N, BK, BN_half,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));

    int cluster_x_count = (M + 255) / 256;
    int grid_y = (N + 255) / 256;
    
    dim3 grid(cluster_x_count * 2, grid_y, 1); // Group into Cluster Pairs
    dim3 block(128, 1, 1);

    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 132 * 1024; // Expanded Shared Memory configuration covering 4-Stages
    
    CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes));

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    config.stream = stream;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2; 
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, tma_A, tma_B, static_cast<__nv_bfloat16*>(C.data_ptr()), M, N, K));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}