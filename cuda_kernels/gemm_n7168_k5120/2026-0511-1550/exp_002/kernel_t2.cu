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

namespace tvm_ffi_gemm_n7168_k5120_cuda {

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

__device__ __forceinline__ void mbarrier_arrive_fn(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])) : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    uint32_t phase_parity = phase & 1;
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

__device__ __forceinline__ void tmem_epilogue_coalesced_4w_fn(
    uint32_t tmem_addr,
    __nv_bfloat16* D, __nv_bfloat16* smem_out,
    uint32_t M, uint32_t N, uint32_t m_block, uint32_t n_block,
    uint32_t BM, uint32_t BN) {
    
    for (uint32_t col = 0; col < BN; col += 4) {
        uint32_t addr = tmem_addr + col;
        uint32_t r0, r1, r2, r3;
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
    
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = (BM + 3) / 4;
    
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        if (row >= BM) continue;
        uint32_t global_row = m_block * BM + row;
        uint32_t col_start = lane_id * 4;
        uint32_t global_col = n_block * BN + col_start;
        
        if (global_row < M) {
            if (global_col + 3 < N) {
                uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * BN + col_start]);
                *reinterpret_cast<uint2*>(D + (uint64_t)global_row * N + global_col) = data;
            } else {
                for (int c = 0; c < 4; ++c) {
                    if (global_col + c < N) {
                        D[(uint64_t)global_row * N + global_col + c] = smem_out[row * BN + col_start + c];
                    }
                }
            }
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

struct SharedStorage {
    union {
        struct {
            alignas(1024) __nv_bfloat16 A[2][128 * 64];
            alignas(1024) __nv_bfloat16 B[2][128 * 64];
        };
        alignas(1024) __nv_bfloat16 out[128 * 128];
    };
    alignas(8) uint64_t bar_A[2];
    alignas(8) uint64_t bar_B[2];
    alignas(8) uint64_t bar_umma[2];
    alignas(128) uint32_t tmem_addr;
};

__global__ void __launch_bounds__(128, 2) gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* C,
    uint32_t M, uint32_t N, uint32_t K) 
{
    extern __shared__ char smem_buf[];
    SharedStorage* smem = reinterpret_cast<SharedStorage*>(smem_buf);

    uint32_t cluster_rank = cluster_rank_fn();

    if (threadIdx.x / 32 == 0) {
        tmem_alloc_fn(&smem->tmem_addr, 128);
    }
    __syncthreads();
    uint32_t my_tmem = smem->tmem_addr;

    for (int buf = 0; buf < 2; ++buf) {
        if (threadIdx.x == 0) {
            init_smem_barrier_fn(&smem->bar_A[buf], 128);
            init_smem_barrier_fn(&smem->bar_B[buf], 128);
            init_smem_barrier_fn(&smem->bar_umma[buf], 1);
        }
    }
    __syncthreads();
    fence_smem_barrier_init_fn();
    // CRITICAL FIX: Make sure all CTAs in the cluster have initialized their mbarriers
    // before any cross-CTA async operations (multicast) trigger completion signaling.
    cluster_sync_fn(); 

    int num_k_steps = (K + 64 - 1) / 64;

    if (num_k_steps > 0) {
        int buf = 0;
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&smem->bar_A[buf], 128 * 64 * 2);
            tma_load_2d_fn(&tma_A, &smem->bar_A[buf], smem->A[buf], 0, blockIdx.x * 128);
        } else {
            mbarrier_arrive_fn(&smem->bar_A[buf]);
        }
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(&smem->bar_B[buf], 128 * 64 * 2);
            if (cluster_rank == 0) {
                tma_load_multicast_2d_fn(&tma_B, &smem->bar_B[buf], smem->B[buf], 0, blockIdx.y * 128, 0x3);
            }
        } else {
            mbarrier_arrive_fn(&smem->bar_B[buf]);
        }
    }

    for (int k_step = 0; k_step < num_k_steps; ++k_step) {
        int buf = k_step % 2;
        int next_buf = (k_step + 1) % 2;
        
        if (k_step + 1 < num_k_steps) {
            if (k_step > 0) {
                mbarrier_wait_fn(&smem->bar_umma[next_buf], (k_step - 1) / 2);
            }
            
            int next_k = (k_step + 1) * 64;
            
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&smem->bar_A[next_buf], 128 * 64 * 2);
                tma_load_2d_fn(&tma_A, &smem->bar_A[next_buf], smem->A[next_buf], next_k, blockIdx.x * 128);
            } else {
                mbarrier_arrive_fn(&smem->bar_A[next_buf]);
            }
            
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&smem->bar_B[next_buf], 128 * 64 * 2);
                if (cluster_rank == 0) {
                    tma_load_multicast_2d_fn(&tma_B, &smem->bar_B[next_buf], smem->B[next_buf], next_k, blockIdx.y * 128, 0x3);
                }
            } else {
                mbarrier_arrive_fn(&smem->bar_B[next_buf]);
            }
        }
        
        mbarrier_wait_fn(&smem->bar_A[buf], k_step / 2);
        mbarrier_wait_fn(&smem->bar_B[buf], k_step / 2);
        
        fence_proxy_async_fn();
        __syncthreads();
        
        uint32_t accum = (k_step == 0) ? 0 : 1;
        uint64_t desc_a = make_smem_desc_sm100_fn(smem->A[buf], 1024);
        uint64_t desc_b = make_smem_desc_sm100_fn(smem->B[buf], 1024);
        uint32_t idesc = make_instr_desc_fn(256, 128); 
        
        if (cluster_rank == 0 && threadIdx.x == 0) {
            umma_f16_cg2_fn(my_tmem, desc_a, desc_b, idesc, accum);
            umma_commit_2sm_fn(&smem->bar_umma[buf]);
        }
    }
    
    if (num_k_steps > 0) {
        int last_buf = (num_k_steps - 1) % 2;
        mbarrier_wait_fn(&smem->bar_umma[last_buf], (num_k_steps - 1) / 2);
    }
    __syncthreads();
    
    __nv_bfloat16* smem_out = &smem->out[0];
    tmem_epilogue_coalesced_4w_fn(my_tmem, C, smem_out, M, N, blockIdx.x, blockIdx.y, 128, 128);
    
    cluster_sync_fn();
    if (threadIdx.x / 32 == 0) {
        tmem_dealloc_fn(my_tmem, 128);
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    int64_t m = A.size(0);
    int64_t k = A.size(1);
    int64_t n = B.size(0);

    __nv_bfloat16* A_data = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* B_data = static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_data = static_cast<__nv_bfloat16*>(C.data_ptr());

    CUtensorMap tma_A, tma_B;
    CUresult res_A = create_tma_2d_descriptor_2B(
        &tma_A, A_data,
        k, m,
        64, 128,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    if (res_A != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to create TMA A\n");
        exit(1);
    }

    CUresult res_B = create_tma_2d_descriptor_2B(
        &tma_B, B_data,
        k, n,
        64, 128,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    if (res_B != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to create TMA B\n");
        exit(1);
    }

    int grid_x = (m + 255) / 256 * 2;
    int grid_y = (n + 127) / 128;

    if (grid_x == 0 || grid_y == 0) return;

    dim3 grid(grid_x, grid_y, 1);
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

    CUDA_CHECK(cudaFuncSetAttribute(
        (const void*)gemm_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        sizeof(SharedStorage)
    ));

    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, tma_A, tma_B, C_data, m, n, k));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_gemm_n7168_k5120_cuda::run);

}  // namespace tvm_ffi_gemm_n7168_k5120_cuda