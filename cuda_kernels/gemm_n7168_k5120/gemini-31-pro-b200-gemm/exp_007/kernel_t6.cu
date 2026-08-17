#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        const char* err_name;                                      \
        cuGetErrorName(_e, &err_name);                             \
        fprintf(stderr, "CU error %s at %s:%d\n",                  \
                err_name, __FILE__, __LINE__);                     \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_example_cuda {

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_fn(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])) : "memory");
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

__device__ __forceinline__ void tma_load_2d_cg2_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t ba = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    
    uint32_t cta_rank = cluster_rank_fn();
    uint32_t cluster_sa, cluster_ba;
    
    // Explicitly target data writes into executing CTA SMEM but funnel completion signals safely into peer 0 tracking pipeline
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(cluster_sa) : "r"(sa), "r"(cta_rank));
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(cluster_ba) : "r"(ba), "r"(0)); 
    
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(cluster_sa), "l"((uint64_t)d), "r"(c0), "r"(c1), "r"(cluster_ba) : "memory");
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

__device__ __forceinline__ void umma_f16_cg2_fn(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_2sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    uint32_t cluster_a;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(cluster_a) : "r"(a), "r"(0)); 
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(cluster_a), "h"((uint16_t)0x3)); 
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16; 
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32; 
    d |= (uint64_t)1 << 46;   
    
    uint32_t base_offset = (addr >> 7) & 0x7;
    d |= (uint64_t)(base_offset) << 49;
    
    d |= (uint64_t)2 << 61;   // CU_TENSOR_MAP_SWIZZLE_128B Equivalent flag
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

__device__ __forceinline__ void tmem_epilogue_coalesced_4w_fn(
    __nv_bfloat16* D, __nv_bfloat16* smem_out, uint32_t tmem_c,
    uint32_t M, uint32_t N, uint32_t m_block, uint32_t n_block,
    uint32_t BM, uint32_t BN) {
    
    for (uint32_t col = 0; col < BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        uint32_t taddr = tmem_c + col;
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
    __nv_bfloat16* C,
    uint32_t M, uint32_t N, uint32_t K)
{
    uint32_t cta_rank = cluster_rank_fn();

    // SMEM size fits up to 34816. We just need 24576 (A: 16384, B: 8192) or 32768 globally for epilogue safety
    __shared__ __align__(1024) int8_t smem_pool[3][32768];
    int8_t* smem_A_ptr[3];
    int8_t* smem_B_ptr[3];

    for (int i = 0; i < 3; ++i) {
        smem_A_ptr[i] = smem_pool[i];
        smem_B_ptr[i] = smem_pool[i] + 16384; 
    }

    __shared__ uint64_t full_barriers[3];
    __shared__ uint64_t empty_barriers[3];

    __shared__ uint32_t tmem_ptr_smem;

    if (threadIdx.x == 0) {
        for (int i = 0; i < 3; ++i) {
            if (cta_rank == 0) {
                init_smem_barrier_fn(&full_barriers[i], 2);
            }
            init_smem_barrier_fn(&empty_barriers[i], 1);
        }
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 0; i < 3; ++i) {
            mbarrier_arrive_fn(&empty_barriers[i]);
        }
    }
    
    // Allocate exactly 128 cols per CTA matching 128x128 FP32 (256x128 global footprint across 2 CTAs)
    if (threadIdx.x < 32) {
        tmem_alloc_fn(&tmem_ptr_smem, 128);
    }
    __syncthreads();
    uint32_t tmem_c = tmem_ptr_smem;

    uint32_t full_phase_arr[3] = {0, 0, 0};
    uint32_t empty_phase_arr[3] = {0, 0, 0};

    int num_k_iters = K / 64;
    int cluster_id_x = blockIdx.x / 2;
    int m_start = cluster_id_x * 256;
    int n_start = blockIdx.y * 128;

    uint32_t idesc = make_instr_desc_fn(256, 128);

    for (int k_iter = 0; k_iter < num_k_iters; ++k_iter) {
        int s = k_iter % 3;

        mbarrier_wait_fn(&empty_barriers[s], empty_phase_arr[s]);

        int k_coord = k_iter * 64;

        if (threadIdx.x == 0) {
            if (cta_rank == 0) {
                // CTA 0 pulls its distinct half of both A and B locally mapping signals uniformly
                tma_load_2d_cg2_fn(&tma_A, &full_barriers[s], smem_A_ptr[s], k_coord, m_start);
                tma_load_2d_cg2_fn(&tma_B, &full_barriers[s], smem_B_ptr[s], k_coord, n_start);
                
                mbarrier_arrive_and_expect_tx_fn(&full_barriers[s], 24576); 
            } else {
                // CTA 1 pulls the complimentary half resolving to CTA 0's central barrier counter
                tma_load_2d_cg2_fn(&tma_A, &full_barriers[s], smem_A_ptr[s], k_coord, m_start + 128);
                tma_load_2d_cg2_fn(&tma_B, &full_barriers[s], smem_B_ptr[s], k_coord, n_start + 64);
                
                mbarrier_arrive_expect_tx_cluster_fn(&full_barriers[s], 24576, 0);
            }
        }

        if (cta_rank == 0) {
            mbarrier_wait_fn(&full_barriers[s], full_phase_arr[s]);

            if (threadIdx.x == 0) {
                // SBO = 1024 safely advances through unified 128B pattern banks 
                for (int k_step = 0; k_step < 4; ++k_step) {
                    uint32_t a_desc = make_smem_desc_sm100_fn((void*)(smem_A_ptr[s] + k_step * 32), 16, 1024);
                    uint32_t b_desc = make_smem_desc_sm100_fn((void*)(smem_B_ptr[s] + k_step * 32), 16, 1024);
                    uint32_t accum = (k_iter == 0 && k_step == 0) ? 0 : 1;
                    umma_f16_cg2_fn(tmem_c, a_desc, b_desc, idesc, accum);
                }
                umma_commit_2sm_fn(&empty_barriers[s]);
            }
            full_phase_arr[s] ^= 1;
        }
        empty_phase_arr[s] ^= 1;
    }

    int last_s = (num_k_iters - 1) % 3;
    mbarrier_wait_fn(&empty_barriers[last_s], empty_phase_arr[last_s]);

    __syncthreads();
    
    // Extrapolates out symmetrically mapped CTA halves resolving naturally to exact output boundaries
    uint32_t m_block = cluster_id_x * 2 + cta_rank;
    uint32_t n_block = blockIdx.y;
    tmem_epilogue_coalesced_4w_fn(C, (__nv_bfloat16*)smem_pool[0], tmem_c, M, N, m_block, n_block, 128, 128);

    __syncthreads();
    
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_c, 128);
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    int32_t M = A.size(0);
    int32_t K = A.size(1);
    int32_t N = B.size(0);

    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    CUtensorMap tma_A, tma_B;
    // Partitioning dynamically loads 128x64 pieces natively over independent CTA blocks 
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_A, A.data_ptr(), K, M, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    // B naturally truncates its height per-CTA loading to complimentary 64x64 chunks independently mapping
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_B, B.data_ptr(), K, N, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    int blocks_m = (M + 255) / 256;
    int blocks_n = (N + 127) / 128;
    
    dim3 grid(blocks_m * 2, blocks_n, 1);
    dim3 block(128, 1, 1);

    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    
    config.attrs = attrs;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(&config, cta_gemm_kernel, tma_A, tma_B, static_cast<__nv_bfloat16*>(C.data_ptr()), M, N, K));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda