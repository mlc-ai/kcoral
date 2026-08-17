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
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
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
    // Even for cta_group::1, we use cta_group::1 alloc variant
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(addr), "r"(ncols));
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

__device__ __forceinline__ uint32_t make_instr_desc_cg1_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);           // c_format = FP32
    d |= (1u << 7);           // a_format = BF16
    d |= (1u << 10);          // b_format = BF16
    d |= ((N / 8) << 17);     // n_dim
    d |= ((M / 16) << 24);    // m_dim
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
    #pragma unroll 2
    for (uint32_t col = 0; col < BN; col += 16) {
        uint32_t r0[8], r1[8];
        uint32_t taddr0 = tmem_c + col;
        uint32_t taddr1 = tmem_c + col + 8;
        
        // x8 load shape extracts 8 lanes from TMEM instantly per instruction
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                     : "=r"(r0[0]),"=r"(r0[1]),"=r"(r0[2]),"=r"(r0[3]),
                       "=r"(r0[4]),"=r"(r0[5]),"=r"(r0[6]),"=r"(r0[7]) : "r"(taddr0));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                     : "=r"(r1[0]),"=r"(r1[1]),"=r"(r1[2]),"=r"(r1[3]),
                       "=r"(r1[4]),"=r"(r1[5]),"=r"(r1[6]),"=r"(r1[7]) : "r"(taddr1));
                       
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        uint4 p0, p1;
        p0.x = pack_bf16_fn(r0[0], r0[1]);
        p0.y = pack_bf16_fn(r0[2], r0[3]);
        p0.z = pack_bf16_fn(r0[4], r0[5]);
        p0.w = pack_bf16_fn(r0[6], r0[7]);
        
        p1.x = pack_bf16_fn(r1[0], r1[1]);
        p1.y = pack_bf16_fn(r1[2], r1[3]);
        p1.z = pack_bf16_fn(r1[4], r1[5]);
        p1.w = pack_bf16_fn(r1[6], r1[7]);

        uint32_t base_words0 = (threadIdx.x * BN_pad + col) / 2;
        *(uint4*)(&((uint32_t*)smem_out)[base_words0]) = p0;

        uint32_t base_words1 = (threadIdx.x * BN_pad + col + 8) / 2;
        *(uint4*)(&((uint32_t*)smem_out)[base_words1]) = p1;
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
constexpr int BN = 256;
constexpr int BK = 64;
constexpr int CLUSTER_SIZE = 4;

__global__ void __launch_bounds__(128) gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* C,
    int M, int N, int K) {

    extern __shared__ uint8_t smem_raw[];
    uintptr_t raw_ptr = (uintptr_t)smem_raw;
    uintptr_t base_ptr = (raw_ptr + 1023) & ~1023; // 1024-byte alignment mandatory for SM100 TMA
    uint8_t* smem_base = smem_raw + (base_ptr - raw_ptr);

    __nv_bfloat16* A_smem[STAGE];
    __nv_bfloat16* B_smem[STAGE];
    for (int i = 0; i < STAGE; ++i) {
        A_smem[i] = (__nv_bfloat16*)(smem_base + i * 16384);
        B_smem[i] = (__nv_bfloat16*)(smem_base + STAGE * 16384 + i * 32768);
    }
    
    uint64_t* mbar_tma = (uint64_t*)(smem_base + STAGE * 49152);
    uint64_t* mbar_umma = mbar_tma + STAGE;

    int cluster_rank = cluster_rank_fn();
    int m_base = blockIdx.x * BM;
    int n_base = blockIdx.y * BN;

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
    uint32_t idesc = make_instr_desc_cg1_fn(BM, BN);
    uint16_t mask = (1 << CLUSTER_SIZE) - 1; 

    int tma_phase[STAGE] = {0};
    int umma_phase[STAGE] = {0};

    // Prologue Pipeline (Fills all STAGE tracks)
    #pragma unroll
    for (int i = 0; i < STAGE - 1; ++i) { 
        int k = i * BK;
        if (k < K) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&mbar_tma[i], 49152); // 16KB + 32KB
                tma_load_2d_fn(&tma_A, &mbar_tma[i], A_smem[i], k, m_base);
                if (cluster_rank == 0) {
                    tma_load_multicast_2d_fn(&tma_B, &mbar_tma[i], B_smem[i], k, n_base, mask);
                }
            }
        }
    }

    // Main Compute Loop -> Perfect pipeline with strict overlap
    int K_iters = (K + BK - 1) / BK;
    
    #pragma unroll 2
    for (int i = 0; i < K_iters; ++i) {
        int compute_stage = i % STAGE;
        int issue_stage = (i + STAGE - 1) % STAGE;
        int issue_k = (i + STAGE - 1) * BK;

        if (issue_k < K) {
            if (i >= 1) {
                // Safely wait until hardware releases the target stage buffer back to us
                mbarrier_wait_fn(&mbar_umma[issue_stage], umma_phase[issue_stage]);
                umma_phase[issue_stage] ^= 1;
            }
            // Ensure no CTA begins loading and corrupts B before previous users are done
            cluster_sync_fn(); 
            
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&mbar_tma[issue_stage], 49152);
                tma_load_2d_fn(&tma_A, &mbar_tma[issue_stage], A_smem[issue_stage], issue_k, m_base);
                if (cluster_rank == 0) {
                    tma_load_multicast_2d_fn(&tma_B, &mbar_tma[issue_stage], B_smem[issue_stage], issue_k, n_base, mask);
                }
            }
        }

        mbarrier_wait_fn(&mbar_tma[compute_stage], tma_phase[compute_stage]);
        tma_phase[compute_stage] ^= 1;
        
        if (threadIdx.x == 0) {
            uint64_t base_desc_A = make_smem_desc_sm100_fn(A_smem[compute_stage], 1024);
            uint64_t base_desc_B = make_smem_desc_sm100_fn(B_smem[compute_stage], 1024);
            
            #pragma unroll
            for (int k_step = 0; k_step < 4; ++k_step) {
                int accum = (i == 0 && k_step == 0) ? 0 : 1;
                asm volatile(
                    "{\n.reg .pred p;\n"
                    "setp.ne.b32 p, %4, 0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                    :: "r"(tmem_c), "l"(base_desc_A + k_step * 2), "l"(base_desc_B + k_step * 2), "r"(idesc), "r"(accum));
            }
            uint32_t a = (uint32_t)__cvta_generic_to_shared(&mbar_umma[compute_stage]);
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                :: "r"(a));
        }
    }

    // Epilogue pipeline depletion
    for (int i = K_iters - STAGE + 1; i < K_iters; ++i) {
        if (i >= 0) {
            int stage = i % STAGE;
            mbarrier_wait_fn(&mbar_umma[stage], umma_phase[stage]);
            umma_phase[stage] ^= 1;
        }
    }
    
    // Dump resultant TMEM 128x256 to C Global Memory
    tmem_epilogue_coalesced_4w_fn(C, (__nv_bfloat16*)smem_base, M, N, m_base, n_base, tmem_c, BM, BN);

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
        &tma_B, B.data_ptr(), K, N, BK, BN,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));

    int grid_x = (M + 127) / 128;
    int grid_y = (N + 255) / 256;
    
    // Pad grid_x to ensure complete clusters
    grid_x = ((grid_x + CLUSTER_SIZE - 1) / CLUSTER_SIZE) * CLUSTER_SIZE;
    
    dim3 grid(grid_x, grid_y, 1); 
    dim3 block(128, 1, 1);

    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 196608 + 1024; // ~193 KB 
    
    CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes));

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    config.stream = stream;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = CLUSTER_SIZE; 
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, tma_A, tma_B, static_cast<__nv_bfloat16*>(C.data_ptr()), M, N, K));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}