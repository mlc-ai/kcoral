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
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        fprintf(stderr, "CU error %d at %s:%d\n",                 \
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_gemm {

CUresult create_tma_2d_descriptor_2B(
    CUtensorMap* d, 
    void* globalAddress, 
    uint64_t gmem_inner_dim, 
    uint64_t gmem_outer_dim, 
    uint32_t smem_inner_dim, 
    uint32_t smem_outer_dim, 
    CUtensorMapDataType dataType,
    CUtensorMapSwizzle swizzle, 
    CUtensorMapL2promotion l2Promotion, 
    CUtensorMapFloatOOBfill oobFill) 
{
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d,
        dataType,
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

__device__ __forceinline__ void prefetch_tma_descriptor_fn(const CUtensorMap* d) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(d) : "memory");
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

__device__ __forceinline__ void fence_mbarrier_init_fn() {
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
                 :: "r"(remote_a), "r"(tx) : "memory");
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

__device__ __forceinline__ void tma_load_2d_cg2_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t remote_sa;
    uint32_t rank = cluster_rank_fn();
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote_sa) : "r"(sa), "r"(rank));

    uint32_t ba = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    uint32_t remote_ba;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote_ba) : "r"(ba), "r"(0)); 

    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(remote_sa), "l"((uint64_t)d), "r"(c0), "r"(c1), "r"(remote_ba) : "memory");
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

__device__ __forceinline__ uint64_t advance_desc_k(uint64_t desc, uint32_t k_offset_bytes) {
    uint32_t old_addr_16 = desc & 0x3FFF;
    uint32_t new_addr_16 = old_addr_16 + (k_offset_bytes >> 4);
    uint32_t new_addr = new_addr_16 << 4;
    uint64_t base_offset = (new_addr >> 7) & 0x7;
    
    uint64_t new_desc = desc & ~(0x3FFFull); 
    new_desc |= (new_addr_16 & 0x3FFF);
    
    new_desc &= ~(0x7ull << 49); 
    new_desc |= (base_offset << 49);
    
    return new_desc;
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
    uint32_t remote_a;
    uint32_t rank = cluster_rank_fn();
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote_a) : "r"(a), "r"(rank));

    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(remote_a), "h"((uint16_t)0x3)); 
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

__device__ __forceinline__ void st_shared_128_fn(uint32_t addr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
}

union SharedMem {
    struct {
        __align__(1024) uint8_t A[3][128 * 64 * 2];
        __align__(1024) uint8_t B[3][2][128 * 64 * 2];
        uint64_t mbar_load[3];
        uint64_t mbar_umma[3];
        uint32_t tmem_c;
    } compute;
    struct {
        __align__(16) __nv_bfloat16 C[128 * 520];
    } epilogue;
};

__global__ __launch_bounds__(256, 1)
void gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* C_ptr,
    int M, int N, int K) 
{
    extern __shared__ SharedMem smem;

    if (threadIdx.x == 0) {
        prefetch_tma_descriptor_fn(&tma_A);
        prefetch_tma_descriptor_fn(&tma_B);
    }
    
    uint32_t rank = cluster_rank_fn();
    int32_t cluster_idx_x = blockIdx.x / 2;
    int32_t cluster_idx_y = blockIdx.y;
    
    int32_t m_coord = cluster_idx_x * 256 + rank * 128;
    int32_t n_coord_combined = cluster_idx_y * 512;
    
    int STAGES = 3;

    if (threadIdx.x == 0) {
        for (int s = 0; s < STAGES; ++s) {
            init_smem_barrier_fn(&smem.compute.mbar_load[s], 2);
            init_smem_barrier_fn(&smem.compute.mbar_umma[s], 1);
        }
        fence_mbarrier_init_fn();
    }
    cluster_sync_fn();

    int K_tiles = K / 64;
    int initial_loads = (K_tiles < STAGES - 1) ? K_tiles : STAGES - 1;

    for (int s = 0; s < initial_loads; ++s) {
        if (threadIdx.x == 0) {
            if (rank == 0) {
                mbarrier_arrive_and_expect_tx_fn(&smem.compute.mbar_load[s], 48 * 1024);
            } else {
                mbarrier_arrive_expect_tx_cluster_fn(&smem.compute.mbar_load[s], 48 * 1024, 0);
            }
            tma_load_2d_cg2_fn(&tma_A, &smem.compute.mbar_load[s], smem.compute.A[s], s * 64, m_coord);
            tma_load_2d_cg2_fn(&tma_B, &smem.compute.mbar_load[s], smem.compute.B[s][0], s * 64, n_coord_combined + rank * 128);
            tma_load_2d_cg2_fn(&tma_B, &smem.compute.mbar_load[s], smem.compute.B[s][1], s * 64, n_coord_combined + 256 + rank * 128);
        }
    }
    
    if (threadIdx.x < 32) {
        tmem_alloc_fn(&smem.compute.tmem_c, 512);
    }
    __syncthreads();
    
    uint32_t tmem_c = smem.compute.tmem_c;
    int phase_load_s[3] = {0, 0, 0};
    int phase_umma_s[3] = {0, 0, 0};

    for (int k = 0; k < K_tiles; ++k) {
        int s = k % STAGES;
        
        if (rank == 0 && threadIdx.x == 0) {
            mbarrier_wait_fn(&smem.compute.mbar_load[s], phase_load_s[s]);
            phase_load_s[s] ^= 1;
            fence_proxy_async_fn();
            
            uint64_t desc_a = make_smem_desc_sm100_fn(smem.compute.A[s], 1, 1024); 
            uint64_t desc_b0 = make_smem_desc_sm100_fn(smem.compute.B[s][0], 1, 1024); 
            uint64_t desc_b1 = make_smem_desc_sm100_fn(smem.compute.B[s][1], 1, 1024); 
            uint32_t idesc = make_instr_desc_fn(256, 256);
            
            #pragma unroll
            for (int k_step = 0; k_step < 64; k_step += 16) {
                uint64_t desc_a_k = advance_desc_k(desc_a, k_step * 2);
                uint64_t desc_b0_k = advance_desc_k(desc_b0, k_step * 2);
                uint64_t desc_b1_k = advance_desc_k(desc_b1, k_step * 2);
                uint32_t accum = (k == 0 && k_step == 0) ? 0 : 1;
                
                umma_f16_cg2_fn(tmem_c, desc_a_k, desc_b0_k, idesc, accum);
                umma_f16_cg2_fn(tmem_c + 256, desc_a_k, desc_b1_k, idesc, accum);
            }
            
            umma_commit_2sm_fn(&smem.compute.mbar_umma[s]); 
        }
        
        int next_k = k + STAGES - 1;
        if (next_k < K_tiles) {
            int s_load = next_k % STAGES;
            if (k > 0) {
                if (threadIdx.x == 0) {
                    mbarrier_wait_fn(&smem.compute.mbar_umma[s_load], phase_umma_s[s_load]);
                    phase_umma_s[s_load] ^= 1;
                }
            }
            if (threadIdx.x == 0) {
                if (rank == 0) {
                    mbarrier_arrive_and_expect_tx_fn(&smem.compute.mbar_load[s_load], 48 * 1024);
                } else {
                    mbarrier_arrive_expect_tx_cluster_fn(&smem.compute.mbar_load[s_load], 48 * 1024, 0);
                }
                tma_load_2d_cg2_fn(&tma_A, &smem.compute.mbar_load[s_load], smem.compute.A[s_load], next_k * 64, m_coord);
                tma_load_2d_cg2_fn(&tma_B, &smem.compute.mbar_load[s_load], smem.compute.B[s_load][0], next_k * 64, n_coord_combined + rank * 128);
                tma_load_2d_cg2_fn(&tma_B, &smem.compute.mbar_load[s_load], smem.compute.B[s_load][1], next_k * 64, n_coord_combined + 256 + rank * 128);
            }
        }
    }
    
    if (K_tiles > 0) {
        int last_s = (K_tiles - 1) % STAGES;
        if (threadIdx.x == 0) {
            mbarrier_wait_fn(&smem.compute.mbar_umma[last_s], phase_umma_s[last_s]);
        }
        __syncthreads();
    }
    
    uint32_t wg_id = threadIdx.x / 128; 
    uint32_t wg_lane = threadIdx.x % 128; 
    uint32_t start_col = wg_id * 256;
    uint32_t end_col = start_col + 256;
    
    for (uint32_t col = start_col; col < end_col; col += 8) {
        uint32_t r[8];
        uint32_t col_addr = tmem_c + col;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                     : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),
                       "=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]) : "r"(col_addr));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        uint32_t p0 = pack_bf16_fn(r[0], r[1]);
        uint32_t p1 = pack_bf16_fn(r[2], r[3]);
        uint32_t p2 = pack_bf16_fn(r[4], r[5]);
        uint32_t p3 = pack_bf16_fn(r[6], r[7]);
        
        uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(&smem.epilogue.C[wg_lane * 520 + col]);
        st_shared_128_fn(smem_addr, p0, p1, p2, p3);
    }
    __syncthreads();
    
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    for (uint32_t row = warp_id; row < 128; row += 8) {
        uint32_t gm = m_coord + row;
        if (gm < M) {
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                uint32_t col = lane_id * 8 + i * 256; 
                uint32_t gn = n_coord_combined + col;
                if (gn < N) {
                    uint4* src = (uint4*)&smem.epilogue.C[row * 520 + col];
                    uint4* dst = (uint4*)&C_ptr[gm * N + gn];
                    *dst = *src;
                }
            }
        }
    }
    
    __syncthreads();
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_c, 512);
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0); 
    
    if (M == 0 || N == 0 || K == 0) return;
    
    const __nv_bfloat16* a_ptr = static_cast<const __nv_bfloat16*>(A.data_ptr());
    const __nv_bfloat16* b_ptr = static_cast<const __nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* c_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());
    
    CUtensorMap tma_A, tma_B;
    
    CU_CHECK(create_tma_2d_descriptor_2B(
        &tma_A, (void*)a_ptr, K, M, 64, 128, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));
    
    CU_CHECK(create_tma_2d_descriptor_2B(
        &tma_B, (void*)b_ptr, K, N, 64, 128, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));
    
    int grid_x = ((M + 255) / 256) * 2;
    int grid_y = (N + 511) / 512;
    
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(grid_x, grid_y, 1);
    config.blockDim = dim3(256, 1, 1);
    config.dynamicSmemBytes = sizeof(SharedMem);
    config.stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, tma_A, tma_B, c_ptr, M, N, K));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(config.stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_gemm::run);

}  // namespace tvm_ffi_gemm