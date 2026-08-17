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

namespace tvm_ffi_gemm {

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 128;
constexpr int NUM_STAGES = 2;

__device__ __forceinline__ bool elect_one_sync_fn() {
    uint32_t pred;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "elect.sync _|p, 0xFFFFFFFF;\n"
        "selp.b32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(pred));
    return pred != 0;
}

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

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

__device__ __forceinline__ void named_barrier_sync_fn(int bar_id, int count) {
    asm volatile("barrier.sync.aligned %0, %1;" :: "r"(bar_id), "r"(count));
}

__device__ __forceinline__ void named_barrier_arrive_fn(int bar_id, int count) {
    asm volatile("barrier.arrive.aligned %0, %1;" :: "r"(bar_id), "r"(count));
}

__device__ __forceinline__ void named_barrier_wait_fn(int bar_id) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "barrier.try_wait.aligned P, %0;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"(bar_id));
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
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
    uint32_t base_offset = (addr >> 7) & 0x7;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    d |= (uint64_t)base_offset << 49;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (0u << 15);   
    d |= (1u << 16);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
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

__device__ __forceinline__ void tmem_store_bf16_row_fn(
    __nv_bfloat16* D, uint32_t tid, uint32_t M, uint32_t N,
    uint32_t m_base, uint32_t n_base, uint32_t BN) {
    uint32_t m_idx = m_base + tid;
    if (m_idx >= M) return;
    for (uint32_t col = 0; col < BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        float f0 = __uint_as_float(r0);
        float f1 = __uint_as_float(r1);
        float f2 = __uint_as_float(r2);
        float f3 = __uint_as_float(r3);
        uint32_t nc = n_base + col;
        __nv_bfloat16* out = D + (uint64_t)m_idx * N + nc;
        if (nc     < N) out[0] = __float2bfloat16(f0);
        if (nc + 1 < N) out[1] = __float2bfloat16(f1);
        if (nc + 2 < N) out[2] = __float2bfloat16(f2);
        if (nc + 3 < N) out[3] = __float2bfloat16(f3);
    }
}

__global__ void __launch_bounds__(256) gemm_kernel( 
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* D,
    uint32_t M
) {
    setmaxnreg_inc_sync_fn<256>();
    
    extern __shared__ char smem_buf[];
    char* smem_A_raw = (char*)(((uintptr_t)smem_buf + 1023) & ~1023);
    char* smem_B_raw = smem_A_raw + BLOCK_M * 64 * sizeof(__nv_bfloat16) * NUM_STAGES * 2;
    
    char* smem_A[NUM_STAGES];
    char* smem_B[NUM_STAGES];
    for (int i = 0; i < NUM_STAGES; ++i) {
        smem_A[i] = smem_A_raw + i * (BLOCK_M * 64 * sizeof(__nv_bfloat16) * 2);
        smem_B[i] = smem_B_raw + i * (BLOCK_N * 64 * sizeof(__nv_bfloat16) * 2);
    }
    
    __shared__ __align__(16) uint64_t full_barriers[NUM_STAGES];
    __shared__ __align__(16) uint64_t empty_barriers[NUM_STAGES];
    __shared__ __align__(4) uint32_t tmem_addr;
    __shared__ __align__(16) uint32_t named_bar[2];

    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_addr, 128);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&full_barriers[0], 1);
        init_smem_barrier_fn(&full_barriers[1], 1);
        init_smem_barrier_fn(&empty_barriers[0], 128);
        init_smem_barrier_fn(&empty_barriers[1], 128);
        named_bar[0] = 1; // Use Bar 1 for Stage 0
        named_bar[1] = 2; // Use Bar 2 for Stage 1
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    // Ensure consumer is initially signaled and ready
    if (threadIdx.x < 128) {
        mbarrier_arrive_fn(&empty_barriers[0]);
        mbarrier_arrive_fn(&empty_barriers[1]);
    }

    uint32_t m_base = blockIdx.y * 256;
    uint32_t n_base = blockIdx.x * 128;
    uint32_t cluster_rank = cluster_rank_fn();

    uint64_t desc_a_base_first_half[NUM_STAGES];
    uint64_t desc_a_base_second_half[NUM_STAGES];
    uint64_t desc_b_base_first_half[NUM_STAGES];
    uint64_t desc_b_base_second_half[NUM_STAGES];
    
    for (int stage = 0; stage < NUM_STAGES; ++stage) {
        desc_a_base_first_half[stage] = make_smem_desc_sm100_fn(smem_A[stage], 1, 1024);
        desc_a_base_second_half[stage] = make_smem_desc_sm100_fn(smem_A[stage] + 16384, 1, 1024);
        
        desc_b_base_first_half[stage] = make_smem_desc_sm100_fn(smem_B[stage], 8192, 1024);
        desc_b_base_second_half[stage] = make_smem_desc_sm100_fn(smem_B[stage] + 16384, 8192, 1024);
    }
    
    uint32_t idesc = make_instr_desc_fn(256, 256); 
    
    uint32_t phase[NUM_STAGES] = {0};
    uint32_t empty_phase[NUM_STAGES] = {0};

    int num_k_tiles = 5120 / BLOCK_K; // 40
    
    // Prime the pump (Stage 0)
    if (threadIdx.x == 0) {
        tma_load_2d_fn(&tma_A, &full_barriers[0], smem_A[0], 0, m_base + cluster_rank * 128);
        tma_load_2d_fn(&tma_A, &full_barriers[0], smem_A[0] + 16384, 64, m_base + cluster_rank * 128);
        tma_load_2d_fn(&tma_B, &full_barriers[0], smem_B[0], 0, n_base);
        tma_load_2d_fn(&tma_B, &full_barriers[0], smem_B[0] + 16384, 64, n_base);
        mbarrier_arrive_and_expect_tx_fn(&full_barriers[0], 65536);
    }

    for (int k = 0; k < num_k_tiles; ++k) {
        int consume_stage = k & 1;
        int produce_stage = (k + 1) & 1;

        // Consume current stage
        if (threadIdx.x < 128) {
            named_barrier_arrive_fn(named_bar[consume_stage], 128);
        }

        // Issue async TMA loads for the next stage
        if (k < num_k_tiles - 1) {
            if (threadIdx.x == 0) {
                named_barrier_wait_fn(named_bar[produce_stage]);
                
                mbarrier_wait_fn(&empty_barriers[produce_stage], empty_phase[produce_stage]);
                empty_phase[produce_stage] ^= 1;
                
                int next_k = (k + 1) * BLOCK_K;
                tma_load_2d_fn(&tma_A, &full_barriers[produce_stage], smem_A[produce_stage], next_k, m_base + cluster_rank * 128);
                tma_load_2d_fn(&tma_A, &full_barriers[produce_stage], smem_A[produce_stage] + 16384, next_k + 64, m_base + cluster_rank * 128);
                tma_load_2d_fn(&tma_B, &full_barriers[produce_stage], smem_B[produce_stage], next_k, n_base);
                tma_load_2d_fn(&tma_B, &full_barriers[produce_stage], smem_B[produce_stage] + 16384, next_k + 64, n_base);
                mbarrier_arrive_and_expect_tx_fn(&full_barriers[produce_stage], 65536);
            }
        }

        mbarrier_wait_fn(&full_barriers[consume_stage], phase[consume_stage]);
        phase[consume_stage] ^= 1;

        fence_proxy_async_fn();

        uint32_t accum = (k == 0) ? 0 : 1;
        
        // 1st Hopper native MMA issuing pattern (4 steps) leveraging rapid K advancement efficiency within 1st 128B pattern
        if (threadIdx.x == 0) {
            for (int i = 0; i < 4; ++i) {
                uint64_t desc_a = desc_a_base_first_half[consume_stage];
                uint64_t desc_b = desc_b_base_first_half[consume_stage];
                
                desc_a += (i * 16) * 2;
                desc_b += (i * 16) * 128;
                
                uint32_t tmem_c_addr = tmem_addr;
                umma_f16_cg2_fn(tmem_c_addr, desc_a, desc_b, idesc, accum);
                accum = 1;
            }
            
            // 2nd native MMA issuing pattern (4 steps) leveraging rapid K advancement efficiency within 2nd 128B pattern
            for (int i = 0; i < 4; ++i) {
                uint64_t desc_a = desc_a_base_second_half[consume_stage];
                uint64_t desc_b = desc_b_base_second_half[consume_stage];
                
                desc_a += (i * 16) * 2;
                desc_b += (i * 16) * 128;
                
                uint32_t tmem_c_addr = tmem_addr;
                umma_f16_cg2_fn(tmem_c_addr, desc_a, desc_b, idesc, accum);
                accum = 1;
            }
            umma_commit_2sm_fn(&full_barriers[consume_stage]);
        }
        
        mbarrier_wait_fn(&full_barriers[consume_stage], phase[consume_stage]);
        phase[consume_stage] ^= 1;

        // Hand off buffer ownership back to Producer scope implicitly via named barriers and __syncthreads
        if (threadIdx.x < 128) {
            mbarrier_arrive_fn(&empty_barriers[consume_stage]);
        }
        __syncthreads();
    }

    // Ensure final cta_group::2 commit fully resolves before writing epilogue outputs to TMEM
    mbarrier_wait_fn(&full_barriers[1], phase[1]);

    tmem_store_bf16_row_fn(D, threadIdx.x, M, 7168, m_base + cluster_rank * 128, n_base, 128);

    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_addr, 128);
    }
    __syncthreads();
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
  CUDA_CHECK(cudaSetDevice(A.device().device_id)); 
  int64_t M = A.size(0); 
  
  const __nv_bfloat16* a_ptr = static_cast<const __nv_bfloat16*>(A.data_ptr());
  const __nv_bfloat16* b_ptr = static_cast<const __nv_bfloat16*>(B.data_ptr());
  __nv_bfloat16* c_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());
  
  CUtensorMap tma_A, tma_B;
  CUresult res_a = create_tma_2d_descriptor_2B(&tma_A, (void*)a_ptr, 5120, M, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  CUresult res_b = create_tma_2d_descriptor_2B(&tma_B, (void*)b_ptr, 5120, 7168, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  
  if (res_a != CUDA_SUCCESS || res_b != CUDA_SUCCESS) {
      fprintf(stderr, "TMA descriptor failed.\n");
      exit(1);
  }

  uint32_t smem_size_A = BLOCK_M * 64 * sizeof(__nv_bfloat16) * NUM_STAGES * 2 + 1024;
  uint32_t smem_size_B = BLOCK_N * 64 * sizeof(__nv_bfloat16) * NUM_STAGES * 2 + 1024;
  uint32_t smem_size = smem_size_A + smem_size_B;

  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
  dim3 grid((7168 + 127) / 128, (M + 255) / 256);
  dim3 block(256);

  cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  
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
  
  CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, tma_A, tma_B, c_ptr, M));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill
    );
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_gemm::run);

} // namespace tvm_ffi_gemm