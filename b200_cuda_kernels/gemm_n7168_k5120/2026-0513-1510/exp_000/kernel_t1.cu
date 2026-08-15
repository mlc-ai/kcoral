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
    __nv_bfloat16 a = __float2bfloat16(__uint_as_float(fp32_a));
    __nv_bfloat16 b = __float2bfloat16(__uint_as_float(fp32_b));
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&a)),
          "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}

__device__ __forceinline__ void tmem_epilogue_coalesced_4w_fn_v2(
    __nv_bfloat16* D, __nv_bfloat16* smem_out,
    uint32_t M, uint32_t N, uint32_t m_base, uint32_t n_base,
    uint32_t tmem_c, uint32_t BM, uint32_t BN) {
    
    for (uint32_t col = 0; col < BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        uint32_t taddr = tmem_c + col;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(taddr));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        uint32_t base_words = (threadIdx.x * BN + col) / 2;
        uint32_t p0 = pack_bf16_fn(r0, r1);
        uint32_t p1 = pack_bf16_fn(r2, r3);
        ((uint32_t*)smem_out)[base_words + 0] = p0;
        ((uint32_t*)smem_out)[base_words + 1] = p1;
    }
    __syncthreads();
    
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = (BM + 3) / 4;
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        if (row >= BM) continue;
        uint32_t global_row = m_base + row;
        
        for(uint32_t c_iter = 0; c_iter < BN; c_iter += 128) {
            uint32_t col_start = c_iter + lane_id * 4;
            uint32_t global_col = n_base + col_start;
            if (global_row < M && global_col + 3 < N) {
                uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * BN + col_start]);
                *reinterpret_cast<uint2*>(D + (uint64_t)global_row * N + global_col) = data;
            } else if (global_row < M) {
                for(int i = 0; i < 4; ++i) {
                    if (global_col + i < N) {
                        D[(uint64_t)global_row * N + global_col + i] = smem_out[row * BN + col_start + i];
                    }
                }
            }
        }
    }
}


__global__ void __launch_bounds__(128) gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* C,
    int M, int N, int K) {

    extern __shared__ uint8_t smem_raw[];
    uintptr_t raw_ptr = (uintptr_t)smem_raw;
    uintptr_t base_ptr = (raw_ptr + 1023) & ~1023;
    uint8_t* smem_base = smem_raw + (base_ptr - raw_ptr);

    __nv_bfloat16* A_smem[2] = {
        (__nv_bfloat16*)(smem_base),
        (__nv_bfloat16*)(smem_base + 128 * 64 * 2)
    };
    __nv_bfloat16* B_smem[2] = {
        (__nv_bfloat16*)(smem_base + 32768),
        (__nv_bfloat16*)(smem_base + 49152)
    };
    uint64_t* mbar_tma = (uint64_t*)(smem_base + 65536);
    uint64_t* mbar_umma = mbar_tma + 2;

    int cluster_rank = cluster_rank_fn();
    int cluster_x = blockIdx.x / 2;
    int cluster_y = blockIdx.y;

    int m_base = cluster_x * 256 + cluster_rank * 128;
    int n_base_half = cluster_y * 256 + cluster_rank * 128;
    int n_base = cluster_y * 256;

    __shared__ uint32_t tmem_c_ptr;
    if (threadIdx.x < 32) {
        tmem_alloc_fn(&tmem_c_ptr, 256);
    }
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&mbar_tma[0], 1);
        init_smem_barrier_fn(&mbar_tma[1], 1);
        init_smem_barrier_fn(&mbar_umma[0], 1);
        init_smem_barrier_fn(&mbar_umma[1], 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    uint32_t tmem_c = tmem_c_ptr;
    uint32_t idesc = make_instr_desc_fn(256, 256);

    int tma_phase[2] = {0, 0};
    int umma_phase[2] = {0, 0};
    int buf = 0;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar_tma[0], 32768);
        tma_load_2d_fn(&tma_A, &mbar_tma[0], A_smem[0], 0, m_base);
        tma_load_2d_fn(&tma_B, &mbar_tma[0], B_smem[0], 0, n_base_half);
    }

    for (int k = 0; k < K; k += 64) {
        mbarrier_wait_fn(&mbar_tma[buf], tma_phase[buf]);
        cluster_sync_fn();

        int next_k = k + 64;
        int next_buf = buf ^ 1;

        if (next_k < K) {
            if (k > 0) {
                mbarrier_wait_fn(&mbar_umma[next_buf], umma_phase[next_buf]);
                umma_phase[next_buf] ^= 1;
            }
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&mbar_tma[next_buf], 32768);
                tma_load_2d_fn(&tma_A, &mbar_tma[next_buf], A_smem[next_buf], next_k, m_base);
                tma_load_2d_fn(&tma_B, &mbar_tma[next_buf], B_smem[next_buf], next_k, n_base_half);
            }
        }

        if (cluster_rank == 0) {
            if (threadIdx.x == 0) {
                for (int k_step = 0; k_step < 4; ++k_step) {
                    uint64_t desc_A = make_smem_desc_sm100_fn(A_smem[buf], 1024);
                    uint64_t desc_B = make_smem_desc_sm100_fn(B_smem[buf], 1024);
                    desc_A += k_step * 2;
                    desc_B += k_step * 2;
                    int accum = (k == 0 && k_step == 0) ? 0 : 1;
                    umma_f16_cg2_fn(tmem_c, desc_A, desc_B, idesc, accum);
                }
                umma_commit_2sm_fn(&mbar_umma[buf]);
            }
        }

        tma_phase[buf] ^= 1;
        buf = next_buf;
    }

    mbarrier_wait_fn(&mbar_umma[buf ^ 1], umma_phase[buf ^ 1]);
    if (K > 64) {
        mbarrier_wait_fn(&mbar_umma[buf], umma_phase[buf]);
    }

    cluster_sync_fn();
    
    tmem_epilogue_coalesced_4w_fn_v2(C, (__nv_bfloat16*)smem_base, M, N, m_base, n_base, tmem_c, 128, 256);

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

    int BM = 128;
    int BN_half = 128;
    int BK = 64;

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
    
    dim3 grid(cluster_x_count * 2, grid_y, 1);
    dim3 block(128, 1, 1);

    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 68 * 1024; 
    
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