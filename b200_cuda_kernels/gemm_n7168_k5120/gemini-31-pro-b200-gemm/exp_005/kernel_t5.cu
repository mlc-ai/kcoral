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
        const char* err_name;                                      \
        cuGetErrorName(_e, &err_name);                             \
        fprintf(stderr, "CUDA Driver error %s at %s:%d\n",         \
                err_name, __FILE__, __LINE__);                     \
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

__device__ __forceinline__ void tma_load_2d_cg2_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t ba = (uint32_t)__cvta_generic_to_shared(&bar[0]) & 0xFEFFFFFF;
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(sa), "l"((uint64_t)d), "r"(c0), "r"(c1), "r"(ba) : "memory");
}

__device__ __forceinline__ void prefetch_tma_descriptor_fn(const CUtensorMap* d) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(d) : "memory");
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
    float2 f = make_float2(__uint_as_float(fp32_a), __uint_as_float(fp32_b));
    __nv_bfloat162 b = __float22bfloat162_rn(f);
    return *reinterpret_cast<uint32_t*>(&b);
}

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

constexpr int NUM_STAGES = 6;
struct __align__(1024) SharedStorage {
    __align__(1024) __nv_bfloat16 A[NUM_STAGES][128][64]; 
    __align__(1024) __nv_bfloat16 B[NUM_STAGES][128][64]; 
    uint64_t full_bar[NUM_STAGES];
    uint64_t umma_bar[NUM_STAGES];
    uint32_t tmem_addr;
};

__global__ void __launch_bounds__(128) gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A, 
    const __grid_constant__ CUtensorMap tma_B, 
    __nv_bfloat16* C, int M, int N, int K) {
    
    extern __shared__ __align__(1024) char smem_raw[];
    SharedStorage* smem = (SharedStorage*)smem_raw;
    
    int cta_rank = cluster_rank_fn();
    int M_blk = blockIdx.x / 2;
    int N_blk = blockIdx.y;
    
    int m_base = M_blk * 256 + cta_rank * 128;
    int n_base = N_blk * 256 + cta_rank * 128;
    int n_base_total = N_blk * 256;
    
    for (int i = 0; i < NUM_STAGES; ++i) {
        if (cta_rank == 0) {
            init_smem_barrier_fn(&smem->full_bar[i], 1);
        }
        init_smem_barrier_fn(&smem->umma_bar[i], 1);
    }
    __syncthreads();
    
    if (cta_rank == 0) {
        fence_smem_barrier_init_fn();
    }
    cluster_sync_fn();
    
    uint32_t tmem_c;
    if (threadIdx.x < 32) {
        tmem_alloc_fn(&smem->tmem_addr, 256);
    }
    __syncthreads();
    tmem_c = smem->tmem_addr;
    
    if (threadIdx.x == 0) {
        prefetch_tma_descriptor_fn(&tma_A);
        prefetch_tma_descriptor_fn(&tma_B);
    }
    
    int num_steps = K / 64;
    
    for (int i = 0; i < NUM_STAGES; ++i) {
        if (i < num_steps) {
            if (threadIdx.x == 0) {
                if (cta_rank == 0) {
                    mbarrier_arrive_and_expect_tx_fn(&smem->full_bar[i], 65536);
                }
                int k_coord = i * 64;
                tma_load_2d_cg2_fn(&tma_A, &smem->full_bar[i], smem->A[i], k_coord, m_base);
                tma_load_2d_cg2_fn(&tma_B, &smem->full_bar[i], smem->B[i], k_coord, n_base);
            }
        }
    }
    
    uint64_t desc_a[NUM_STAGES];
    uint64_t desc_b[NUM_STAGES];
    if (threadIdx.x == 0 && cta_rank == 0) {
        for (int i = 0; i < NUM_STAGES; ++i) {
            desc_a[i] = make_smem_desc_sm100_fn(smem->A[i], 1, 1024);
            desc_b[i] = make_smem_desc_sm100_fn(smem->B[i], 1, 1024);
        }
    }
    uint32_t idesc = make_instr_desc_fn(256, 256);
    
    for (int k_step = 0; k_step < num_steps; ++k_step) {
        int stage = k_step % NUM_STAGES;
        
        if (cta_rank == 0 && threadIdx.x == 0) {
            mbarrier_wait_fn(&smem->full_bar[stage], k_step / NUM_STAGES);
            
            uint64_t cur_desc_a = desc_a[stage];
            uint64_t cur_desc_b = desc_b[stage];
            
            for (int i = 0; i < 4; ++i) {
                bool is_first = (k_step == 0 && i == 0);
                uint32_t accum = is_first ? 0 : 1;
                umma_f16_cg2_fn(tmem_c, cur_desc_a, cur_desc_b, idesc, accum);
                cur_desc_a += 2;
                cur_desc_b += 2;
            }
            
            umma_commit_2sm_fn(&smem->umma_bar[stage]);
        }
        
        int load_step = k_step - (NUM_STAGES - 2);
        if (load_step >= 0) {
            int load_stage = load_step % NUM_STAGES;
            if (threadIdx.x == 0) {
                mbarrier_wait_fn(&smem->umma_bar[load_stage], load_step / NUM_STAGES);
                
                int next_k_step = load_step + NUM_STAGES;
                if (next_k_step < num_steps) {
                    if (cta_rank == 0) {
                        mbarrier_arrive_and_expect_tx_fn(&smem->full_bar[load_stage], 65536);
                    }
                    int next_k_coord = next_k_step * 64;
                    tma_load_2d_cg2_fn(&tma_A, &smem->full_bar[load_stage], smem->A[load_stage], next_k_coord, m_base);
                    tma_load_2d_cg2_fn(&tma_B, &smem->full_bar[load_stage], smem->B[load_stage], next_k_coord, n_base);
                }
            }
        }
    }
    
    if (threadIdx.x == 0) {
        int start_wait = num_steps - (NUM_STAGES - 2);
        if (start_wait < 0) start_wait = 0;
        for (int i = start_wait; i < num_steps; ++i) {
            mbarrier_wait_fn(&smem->umma_bar[i % NUM_STAGES], i / NUM_STAGES);
        }
    }
    
    __syncthreads();
    tcgen05_fence_after_fn();
    
    uint32_t tid = threadIdx.x;
    uint32_t* smem_out_u32 = (uint32_t*)smem_raw;
    const uint32_t SMEM_PITCH_U32 = 132;
    
    for (uint32_t col = 0; col < 256; col += 32) {
        uint32_t addr0 = tmem_c + col;
        uint32_t addr1 = tmem_c + col + 8;
        uint32_t addr2 = tmem_c + col + 16;
        uint32_t addr3 = tmem_c + col + 24;
        
        uint32_t r0_0, r1_0, r2_0, r3_0, r4_0, r5_0, r6_0, r7_0;
        uint32_t r0_1, r1_1, r2_1, r3_1, r4_1, r5_1, r6_1, r7_1;
        uint32_t r0_2, r1_2, r2_2, r3_2, r4_2, r5_2, r6_2, r7_2;
        uint32_t r0_3, r1_3, r2_3, r3_3, r4_3, r5_3, r6_3, r7_3;

        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r0_0),"=r"(r1_0),"=r"(r2_0),"=r"(r3_0),"=r"(r4_0),"=r"(r5_0),"=r"(r6_0),"=r"(r7_0) : "r"(addr0));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r0_1),"=r"(r1_1),"=r"(r2_1),"=r"(r3_1),"=r"(r4_1),"=r"(r5_1),"=r"(r6_1),"=r"(r7_1) : "r"(addr1));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r0_2),"=r"(r1_2),"=r"(r2_2),"=r"(r3_2),"=r"(r4_2),"=r"(r5_2),"=r"(r6_2),"=r"(r7_2) : "r"(addr2));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r0_3),"=r"(r1_3),"=r"(r2_3),"=r"(r3_3),"=r"(r4_3),"=r"(r5_3),"=r"(r6_3),"=r"(r7_3) : "r"(addr3));

        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

        uint32_t base = tid * SMEM_PITCH_U32 + col / 2;
        smem_out_u32[base + 0] = pack_bf16_fn(r0_0, r1_0);
        smem_out_u32[base + 1] = pack_bf16_fn(r2_0, r3_0);
        smem_out_u32[base + 2] = pack_bf16_fn(r4_0, r5_0);
        smem_out_u32[base + 3] = pack_bf16_fn(r6_0, r7_0);
        
        smem_out_u32[base + 4] = pack_bf16_fn(r0_1, r1_1);
        smem_out_u32[base + 5] = pack_bf16_fn(r2_1, r3_1);
        smem_out_u32[base + 6] = pack_bf16_fn(r4_1, r5_1);
        smem_out_u32[base + 7] = pack_bf16_fn(r6_1, r7_1);
        
        smem_out_u32[base + 8] = pack_bf16_fn(r0_2, r1_2);
        smem_out_u32[base + 9] = pack_bf16_fn(r2_2, r3_2);
        smem_out_u32[base + 10] = pack_bf16_fn(r4_2, r5_2);
        smem_out_u32[base + 11] = pack_bf16_fn(r6_2, r7_2);
        
        smem_out_u32[base + 12] = pack_bf16_fn(r0_3, r1_3);
        smem_out_u32[base + 13] = pack_bf16_fn(r2_3, r3_3);
        smem_out_u32[base + 14] = pack_bf16_fn(r4_3, r5_3);
        smem_out_u32[base + 15] = pack_bf16_fn(r6_3, r7_3);
    }
    
    __syncthreads();
    
    for (uint32_t k = 0; k < 32; ++k) {
        uint32_t i = k * 128 + tid;
        uint32_t row = i / 32;       
        uint32_t col_u4 = i % 32; 
        
        uint32_t g_row = m_base + row;
        uint32_t g_col = n_base_total + col_u4 * 8; 
        
        if (g_row < M && g_col < N) {
            uint4 data = *reinterpret_cast<uint4*>(&smem_out_u32[row * SMEM_PITCH_U32 + col_u4 * 4]);
            *reinterpret_cast<uint4*>(C + (uint64_t)g_row * N + g_col) = data;
        }
    }
    
    __syncthreads();
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_c, 256);
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

namespace tvm_ffi_example_cuda {

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);
    
    __nv_bfloat16* a_ptr = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* b_ptr = static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* c_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());
    
    CUtensorMap tma_A, tma_B;
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_A, a_ptr, K, M, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_B, b_ptr, K, N, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        
    int blocks_m = (M + 255) / 256;
    if (blocks_m % 2 != 0) blocks_m += 1;
    int blocks_n = (N + 255) / 256;
    
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(blocks_m, blocks_n, 1);
    config.blockDim = dim3(128, 1, 1);
    config.dynamicSmemBytes = sizeof(SharedStorage);
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedStorage)));
    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, tma_A, tma_B, c_ptr, (int)M, (int)N, (int)K));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}