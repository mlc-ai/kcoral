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

namespace tvm_ffi_example_cuda {

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

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

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase_parity) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase_parity));
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void prefetch_tma_descriptor_fn(const CUtensorMap* d) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(d) : "memory");
}

__device__ __forceinline__ void tma_load_2d_cg2_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t ba = (uint32_t)__cvta_generic_to_shared(&bar[0]) & 0xFEFFFFFF; 
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(sa), "l"((uint64_t)d), "r"(c0), "r"(c1), "r"(ba) : "memory");
}

__device__ __forceinline__ void tmem_alloc_cg2_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_cg2_fn(uint32_t addr, int ncols) {
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

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo, uint32_t pattern_start) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    
    uint32_t base_offset = (pattern_start >> 7) & 0x7;
    d |= (uint64_t)base_offset << 49;
    
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

__device__ __forceinline__ void tmem_epilogue_manual(
    uint32_t tmem_c,
    __nv_bfloat16* smem_out,
    __nv_bfloat16* D,
    uint32_t M_total, uint32_t N_total,
    uint32_t m_start, uint32_t n_start,
    uint32_t BM, uint32_t BN) {
    
    uint32_t BN_pad = 258; 
    
    #pragma unroll 1
    for (uint32_t col = 0; col < BN; col += 32) {
        uint32_t r0[8], r1[8], r2[8], r3[8];
        
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r0[0]),"=r"(r0[1]),"=r"(r0[2]),"=r"(r0[3]),"=r"(r0[4]),"=r"(r0[5]),"=r"(r0[6]),"=r"(r0[7]) : "r"(tmem_c + col));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r1[0]),"=r"(r1[1]),"=r"(r1[2]),"=r"(r1[3]),"=r"(r1[4]),"=r"(r1[5]),"=r"(r1[6]),"=r"(r1[7]) : "r"(tmem_c + col + 8));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r2[0]),"=r"(r2[1]),"=r"(r2[2]),"=r"(r2[3]),"=r"(r2[4]),"=r"(r2[5]),"=r"(r2[6]),"=r"(r2[7]) : "r"(tmem_c + col + 16));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r3[0]),"=r"(r3[1]),"=r"(r3[2]),"=r"(r3[3]),"=r"(r3[4]),"=r"(r3[5]),"=r"(r3[6]),"=r"(r3[7]) : "r"(tmem_c + col + 24));
        
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        uint32_t base = threadIdx.x * BN_pad + col;
        
        *reinterpret_cast<uint4*>(&smem_out[base]) = make_uint4(
            pack_bf16_fn(r0[0], r0[1]), pack_bf16_fn(r0[2], r0[3]), pack_bf16_fn(r0[4], r0[5]), pack_bf16_fn(r0[6], r0[7]));
            
        *reinterpret_cast<uint4*>(&smem_out[base + 8]) = make_uint4(
            pack_bf16_fn(r1[0], r1[1]), pack_bf16_fn(r1[2], r1[3]), pack_bf16_fn(r1[4], r1[5]), pack_bf16_fn(r1[6], r1[7]));
            
        *reinterpret_cast<uint4*>(&smem_out[base + 16]) = make_uint4(
            pack_bf16_fn(r2[0], r2[1]), pack_bf16_fn(r2[2], r2[3]), pack_bf16_fn(r2[4], r2[5]), pack_bf16_fn(r2[6], r2[7]));
            
        *reinterpret_cast<uint4*>(&smem_out[base + 24]) = make_uint4(
            pack_bf16_fn(r3[0], r3[1]), pack_bf16_fn(r3[2], r3[3]), pack_bf16_fn(r3[4], r3[5]), pack_bf16_fn(r3[6], r3[7]));
    }
    
    __syncthreads();

    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = (BM + 3) / 4;
    
    #pragma unroll 1
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        if (row >= BM) continue;
        
        uint32_t global_row = m_start + row;
        uint32_t col_start = lane_id * 8; 
        uint32_t global_col = n_start + col_start;
        
        if (global_row < M_total && global_col < N_total) {
            uint4 data = *reinterpret_cast<uint4*>(&smem_out[row * BN_pad + col_start]);
            *reinterpret_cast<uint4*>(D + (uint64_t)global_row * N_total + global_col) = data;
        }
    }
}

extern __shared__ __align__(1024) uint8_t smem_pool[];

__global__ void gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* C_ptr,
    int M, int N, int K
) {
    setmaxnreg_inc_sync_fn<256>();
    
    int k_steps = K / 64;

    int cluster_n = blockIdx.x / 2;
    int cluster_m = blockIdx.y;
    int rank = cluster_rank_fn();

    int M_load = cluster_m * 256 + rank * 128;
    int N_load = cluster_n * 256 + rank * 128;
    int M_store = cluster_m * 256 + rank * 128;
    int N_store = cluster_n * 256;

    uint8_t* smem_A = smem_pool;
    uint8_t* smem_B = smem_pool + 49152;
    __nv_bfloat16* smem_C = (__nv_bfloat16*)smem_pool;

    __shared__ uint64_t bar[3];
    __shared__ uint64_t umma_bar[3];
    __shared__ uint32_t tmem_c_smem;

    if (threadIdx.x == 0) {
        for (int i = 0; i < 3; i++) {
            init_smem_barrier_fn(&umma_bar[i], 1);
        }
    }
    
    if (rank == 0) {
        if (threadIdx.x == 0) {
            for (int i = 0; i < 3; i++) {
                init_smem_barrier_fn(&bar[i], 1);
            }
        }
    }
    
    cluster_sync_fn();

    if (threadIdx.x == 0) {
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        tmem_alloc_cg2_fn(&tmem_c_smem, 256);
    }
    __syncthreads();
    uint32_t tmem_c = tmem_c_smem;

    if (threadIdx.x == 0) {
        prefetch_tma_descriptor_fn(&tma_A);
        prefetch_tma_descriptor_fn(&tma_B);

        for (int i = 0; i < 3 && i < k_steps; i++) {
            if (rank == 0) {
                mbarrier_arrive_and_expect_tx_fn(&bar[i], 65536);
            }
            tma_load_2d_cg2_fn(&tma_A, &bar[i], smem_A + i * 16384, i * 64, M_load);
            tma_load_2d_cg2_fn(&tma_B, &bar[i], smem_B + i * 16384, i * 64, N_load);
        }
    }

    uint32_t idesc = make_instr_desc_fn(256, 256);
    int accum = 0;

    uint64_t desc_A_stage[3];
    uint64_t desc_B_stage[3];
    
    if (rank == 0 && threadIdx.x == 0) {
        for (int i = 0; i < 3; i++) {
            uint32_t pattern_A = (uint32_t)__cvta_generic_to_shared(smem_A + i * 16384);
            uint32_t pattern_B = (uint32_t)__cvta_generic_to_shared(smem_B + i * 16384);
            desc_A_stage[i] = make_smem_desc_sm100_fn(smem_A + i * 16384, 1, 1024, pattern_A);
            desc_B_stage[i] = make_smem_desc_sm100_fn(smem_B + i * 16384, 1, 1024, pattern_B);
        }
    }
    
    int next_k_offset = 3 * 64;

    #pragma unroll 1
    for (int k = 0; k < k_steps; k++) {
        int stage = k % 3;

        if (rank == 0 && threadIdx.x == 0) {
            mbarrier_wait_fn(&bar[stage], (k / 3) & 1);
            fence_proxy_async_fn();

            uint64_t dA = desc_A_stage[stage];
            uint64_t dB = desc_B_stage[stage];

            umma_f16_cg2_fn(tmem_c, dA + 0, dB + 0, idesc, accum);
            accum = 1;
            umma_f16_cg2_fn(tmem_c, dA + 2, dB + 2, idesc, 1);
            umma_f16_cg2_fn(tmem_c, dA + 4, dB + 4, idesc, 1);
            umma_f16_cg2_fn(tmem_c, dA + 6, dB + 6, idesc, 1);

            umma_commit_2sm_fn(&umma_bar[stage]);
        }

        if (k >= 1 && (k + 2) < k_steps) {
            int prev_stage = (k - 1) % 3;
            
            if (threadIdx.x == 0) {
                mbarrier_wait_fn(&umma_bar[prev_stage], ((k - 1) / 3) & 1);
            }
            
            if (rank == 0 && threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&bar[prev_stage], 65536);
            }
            
            if (threadIdx.x == 0) {
                tma_load_2d_cg2_fn(&tma_A, &bar[prev_stage], smem_A + prev_stage * 16384, next_k_offset, M_load);
                tma_load_2d_cg2_fn(&tma_B, &bar[prev_stage], smem_B + prev_stage * 16384, next_k_offset, N_load);
                next_k_offset += 64;
            }
        }
    }

    if (k_steps > 0) {
        int last_k = k_steps - 1;
        if (threadIdx.x == 0) {
            mbarrier_wait_fn(&umma_bar[last_k % 3], (last_k / 3) & 1);
        }
    }

    __syncthreads();
    tcgen05_fence_after_fn();

    tmem_epilogue_manual(tmem_c, smem_C, C_ptr, M, N, M_store, N_store, 128, 256);

    cluster_sync_fn();
    if (threadIdx.x < 32) {
        tmem_dealloc_cg2_fn(tmem_c, 256);
    }
}

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    uint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    uint64_t globalStrides[1] = {gmem_inner_dim * 2};
    uint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    uint32_t elementStrides[2] = {1, 1};
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
    
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);
    
    if (M == 0 || N == 0 || K == 0) return;
    
    __nv_bfloat16* a_ptr = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* b_ptr = static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* c_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());
    
    CUtensorMap tma_A, tma_B;
    
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_A, a_ptr, K, M, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_B, b_ptr, K, N, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        
    dim3 block(128);
    dim3 grid((N / 256) * 2, M / 256);
    
    int smem_size = 98304;
    CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, tma_A, tma_B, c_ptr, M, N, K));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}