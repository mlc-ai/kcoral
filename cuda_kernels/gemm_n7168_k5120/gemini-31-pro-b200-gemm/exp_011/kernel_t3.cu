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

constexpr int NUM_STAGES = 6;
constexpr int NUM_PREFETCH = 5;

struct SharedStorage {
    alignas(1024) __nv_bfloat16 A[NUM_STAGES][128][64]; 
    alignas(1024) __nv_bfloat16 B[NUM_STAGES][128][64]; 
    alignas(16) uint64_t bar_load[NUM_STAGES];
    alignas(16) uint64_t bar_umma[NUM_STAGES];
};

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

__device__ __forceinline__ void tma_store_2d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_store_fence_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<int N_GRP>
__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N_GRP) : "memory");
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
    d |= (1u << 4);    // dtype = F32
    d |= (1u << 7);    // atype = BF16
    d |= (1u << 10);   // btype = BF16
    d |= (0u << 15);   // a transpose = 0 (K-Major)
    d |= (0u << 16);   // b transpose = 0 (K-Major)
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

__device__ __forceinline__ void st_shared_128_fn(uint32_t addr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
}

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

CUresult create_tma_2d_descriptor_2B(
    CUtensorMap* d, void* globalAddress, 
    uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, 
    uint32_t smem_inner_dim, uint32_t smem_outer_dim, 
    CUtensorMapDataType dataType,
    CUtensorMapSwizzle swizzle, 
    CUtensorMapL2promotion l2Promotion, 
    CUtensorMapFloatOOBfill oobFill) {
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

__global__ void __launch_bounds__(128) gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    const __grid_constant__ CUtensorMap tma_C,
    uint32_t K
) {
    setmaxnreg_inc_sync_fn<256>();
    
    extern __shared__ char smem_raw[];
    uintptr_t raw_addr = reinterpret_cast<uintptr_t>(smem_raw);
    uintptr_t aligned_addr = (raw_addr + 1023) & ~1023ULL;
    SharedStorage* smem = reinterpret_cast<SharedStorage*>(aligned_addr);
    
    uint32_t bx = blockIdx.x / 2;
    uint32_t by = blockIdx.y;
    uint32_t rank = cluster_rank_fn();
    
    uint32_t m_start = by * 256;
    uint32_t n_start = bx * 256;
    
    // Allocate TMEM
    __shared__ uint32_t smem_tmem_ptr;
    if (threadIdx.x < 32) {
        tmem_alloc_fn(&smem_tmem_ptr, 256); // 256 cols
    }
    __syncthreads();
    uint32_t tmem_c = smem_tmem_ptr;
    
    // Initialize mbarriers
    if (rank == 0 && threadIdx.x == 0) {
        for (int i = 0; i < NUM_STAGES; ++i) {
            init_smem_barrier_fn(&smem->bar_load[i], 2);
        }
        fence_smem_barrier_init_fn();
    }
    if (threadIdx.x == 0) {
        for (int i = 0; i < NUM_STAGES; ++i) {
            init_smem_barrier_fn(&smem->bar_umma[i], 1);
        }
        fence_smem_barrier_init_fn();
    }
    cluster_sync_fn();
    
    uint32_t num_k_blocks = (K + 63) / 64;
    int phase_load[NUM_STAGES] = {0};
    int phase_umma[NUM_STAGES] = {0};
    
    // Prologue TMA loads
    for (int i = 0; i < NUM_PREFETCH; ++i) {
        if (i < num_k_blocks) {
            if (threadIdx.x == 0) {
                if (rank == 0) {
                    mbarrier_arrive_and_expect_tx_fn(&smem->bar_load[i], 32768);
                } else {
                    mbarrier_arrive_expect_tx_cluster_fn(&smem->bar_load[i], 32768, 0);
                }
                int32_t c0 = i * 64;
                tma_load_2d_cg2_fn(&tma_A, &smem->bar_load[i], smem->A[i], c0, m_start + rank * 128);
                tma_load_2d_cg2_fn(&tma_B, &smem->bar_load[i], smem->B[i], c0, n_start + rank * 128);
            }
        }
    }
    
    uint32_t idesc = make_instr_desc_fn(256, 256);
    
    // Main loop: Compute and issue future loads
    for (int k = 0; k < num_k_blocks; ++k) {
        int s = k % NUM_STAGES;
        
        // Compute step k
        if (rank == 0 && threadIdx.x == 0) {
            mbarrier_wait_fn(&smem->bar_load[s], phase_load[s]);
            phase_load[s] ^= 1;
            tcgen05_fence_after_fn();
            
            #pragma unroll
            for (int step = 0; step < 4; ++step) {
                uint64_t desc_A = make_smem_desc_sm100_fn(&smem->A[s][0][step * 16], 1, 1024);
                uint64_t desc_B = make_smem_desc_sm100_fn(&smem->B[s][0][step * 16], 1, 1024);
                uint32_t accum = (k == 0 && step == 0) ? 0 : 1;
                umma_f16_cg2_fn(tmem_c, desc_A, desc_B, idesc, accum);
            }
            
            umma_commit_2sm_fn(&smem->bar_umma[s]);
        }
        
        // Issue TMA load for future block (k + NUM_PREFETCH)
        int load_k = k + NUM_PREFETCH;
        if (load_k < num_k_blocks) {
            int load_s = load_k % NUM_STAGES;
            if (threadIdx.x == 0) {
                // Ensure the UMMA that last used `load_s` is finished
                int last_umma_k = load_k - NUM_STAGES;
                if (last_umma_k >= 0) {
                    mbarrier_wait_fn(&smem->bar_umma[load_s], phase_umma[load_s]);
                    phase_umma[load_s] ^= 1;
                }
                
                if (rank == 0) {
                    mbarrier_arrive_and_expect_tx_fn(&smem->bar_load[load_s], 32768);
                } else {
                    mbarrier_arrive_expect_tx_cluster_fn(&smem->bar_load[load_s], 32768, 0);
                }
                int32_t c0 = load_k * 64;
                tma_load_2d_cg2_fn(&tma_A, &smem->bar_load[load_s], smem->A[load_s], c0, m_start + rank * 128);
                tma_load_2d_cg2_fn(&tma_B, &smem->bar_load[load_s], smem->B[load_s], c0, n_start + rank * 128);
            }
        }
    }
    
    // Wait for the trailing UMMA operations to complete
    if (threadIdx.x == 0) {
        int start_wait = num_k_blocks > NUM_STAGES ? num_k_blocks - NUM_STAGES : 0;
        for (int k = start_wait; k < num_k_blocks; ++k) {
            int s = k % NUM_STAGES;
            mbarrier_wait_fn(&smem->bar_umma[s], phase_umma[s]);
            phase_umma[s] ^= 1;
        }
    }
    
    // Ensure all threads wait before reusing SMEM for Epilogue
    __syncthreads(); 
    
    __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(&smem->A[0][0][0]);
    
    // Epilogue Phase 1: TMEM -> SMEM
    // Stride is 256, corresponding to block inner dimension size
    uint32_t stride = 256;
    #pragma unroll
    for (uint32_t col = 0; col < 256; col += 8) {
        uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
        uint32_t addr = tmem_c + col;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),
              "=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(addr));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        uint32_t base = threadIdx.x * stride + col;
        uint32_t p0 = pack_bf16_fn(r0, r1);
        uint32_t p1 = pack_bf16_fn(r2, r3);
        uint32_t p2 = pack_bf16_fn(r4, r5);
        uint32_t p3 = pack_bf16_fn(r6, r7);
        
        uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(&smem_out[base]);
        st_shared_128_fn(smem_addr, p0, p1, p2, p3);
    }
    __syncthreads();
    
    // Epilogue Phase 2: SMEM -> Global (TMA store)
    tma_store_fence_fn();
    
    if (threadIdx.x == 0) {
        tma_store_2d_fn(&tma_C, smem_out, n_start, m_start + rank * 128);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }
    
    // Wait for the TMA store to complete before deallocating TMEM (though TMEM is independent, it marks end of kernel cleanly)
    __syncthreads(); 
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_c, 256);
    }
}

namespace tvm_ffi_gemm_sm100 {

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0); 
    
    __nv_bfloat16* A_ptr = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* B_ptr = static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());
    
    CUtensorMap tma_A, tma_B, tma_C;
    
    CUresult res;
    res = create_tma_2d_descriptor_2B(&tma_A, A_ptr, K, M, 64, 128, 
                                      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                      CU_TENSOR_MAP_SWIZZLE_128B, 
                                      CU_TENSOR_MAP_L2_PROMOTION_NONE, 
                                      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "TMA A creation failed\n");
        exit(1);
    }
    
    res = create_tma_2d_descriptor_2B(&tma_B, B_ptr, K, N, 64, 128, 
                                      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                      CU_TENSOR_MAP_SWIZZLE_128B, 
                                      CU_TENSOR_MAP_L2_PROMOTION_NONE, 
                                      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "TMA B creation failed\n");
        exit(1);
    }

    res = create_tma_2d_descriptor_2B(&tma_C, C_ptr, N, M, 256, 128, 
                                      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                      CU_TENSOR_MAP_SWIZZLE_NONE, 
                                      CU_TENSOR_MAP_L2_PROMOTION_NONE, 
                                      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "TMA C creation failed\n");
        exit(1);
    }
    
    int num_m_blocks = (M + 255) / 256;
    int num_n_blocks = (N + 255) / 256;
    
    dim3 grid(num_n_blocks * 2, num_m_blocks, 1);
    dim3 block(128, 1, 1);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    // Add 1024 bytes padding to ensure we can perfectly align to 1024-byte boundary
    config.dynamicSmemBytes = sizeof(SharedStorage) + 1024;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedStorage) + 1024));
    
    cudaLaunchKernelEx(&config, gemm_kernel, tma_A, tma_B, tma_C, (uint32_t)K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_gemm_sm100::run);

} // namespace tvm_ffi_gemm_sm100