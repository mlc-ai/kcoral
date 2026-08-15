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

namespace tvm_ffi_example_cuda {

#define BM_CTA 128
#define BN_CTA 128
#define BK 128
#define STAGES 3

struct SharedStorage {
    __align__(1024) __nv_bfloat16 A[STAGES][BM_CTA * BK];
    __align__(1024) __nv_bfloat16 B[STAGES][BN_CTA * BK];
    __align__(16) uint64_t full_mbar[STAGES];
    __align__(16) uint64_t empty_mbar[STAGES];
    __align__(16) uint32_t tmem_addr;
};

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
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

__device__ __forceinline__ void tma_load_2d_cg2_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t ba = (uint32_t)__cvta_generic_to_shared(&bar[0]) & 0xFEFFFFFF;
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
        :: "r"(sa), "l"((uint64_t)d), "r"(c0), "r"(c1), "r"(ba) : "memory");
}

__device__ __forceinline__ void tma_store_2d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols) : "memory");
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols) : "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo, uint32_t k_offset_bytes = 0) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr) + k_offset_bytes;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    uint32_t base_offset = (addr >> 7) & 0x7;
    d |= (uint64_t)base_offset << 49;
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
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
        :: "r"(a), "h"((uint16_t)0x3) : "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_before_fn() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ void cluster_sync_fn() {
    asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n" ::: "memory");
}

__device__ __forceinline__ uint32_t pack_bf16x2(uint32_t f0, uint32_t f1) {
    uint32_t res;
    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(res) : "f"(__uint_as_float(f1)), "f"(__uint_as_float(f0)));
    return res;
}

__device__ __forceinline__ void store_chunk_swizzled_4tiles(uint32_t* smem, int row, int col_elem, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    int tile_idx = col_elem / 64;
    int chunk_in_tile = (col_elem % 64) / 8;
    
    int swizzled_chunk = (chunk_in_tile & ~7) | ((chunk_in_tile & 7) ^ (row & 7));
    int offset_in_tile = row * 8 + swizzled_chunk;
    
    uint4* smem_v4 = reinterpret_cast<uint4*>(smem);
    int total_offset = tile_idx * 1024 + offset_in_tile;
    smem_v4[total_offset] = make_uint4(v0, v1, v2, v3);
}

__device__ __forceinline__ void my_tmem_epilogue_tma_fn(
    uint32_t tmem_base,
    const CUtensorMap* tma_C, __nv_bfloat16* smem_out,
    uint32_t global_m_start, uint32_t global_n_start,
    uint32_t BM_local, uint32_t BN_local) {
    
    for (uint32_t col = 0; col < BN_local; col += 32) {
        uint32_t r0[8], r1[8], r2[8], r3[8];
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" : "=r"(r0[0]),"=r"(r0[1]),"=r"(r0[2]),"=r"(r0[3]),"=r"(r0[4]),"=r"(r0[5]),"=r"(r0[6]),"=r"(r0[7]) : "r"(tmem_base + col));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" : "=r"(r1[0]),"=r"(r1[1]),"=r"(r1[2]),"=r"(r1[3]),"=r"(r1[4]),"=r"(r1[5]),"=r"(r1[6]),"=r"(r1[7]) : "r"(tmem_base + col + 8));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" : "=r"(r2[0]),"=r"(r2[1]),"=r"(r2[2]),"=r"(r2[3]),"=r"(r2[4]),"=r"(r2[5]),"=r"(r2[6]),"=r"(r2[7]) : "r"(tmem_base + col + 16));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" : "=r"(r3[0]),"=r"(r3[1]),"=r"(r3[2]),"=r"(r3[3]),"=r"(r3[4]),"=r"(r3[5]),"=r"(r3[6]),"=r"(r3[7]) : "r"(tmem_base + col + 24));
        
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        int row = threadIdx.x;
        uint32_t* smem_out_u32 = reinterpret_cast<uint32_t*>(smem_out);
        
        store_chunk_swizzled_4tiles(smem_out_u32, row, col + 0, 
            pack_bf16x2(r0[0], r0[1]), pack_bf16x2(r0[2], r0[3]), pack_bf16x2(r0[4], r0[5]), pack_bf16x2(r0[6], r0[7]));
        store_chunk_swizzled_4tiles(smem_out_u32, row, col + 8, 
            pack_bf16x2(r1[0], r1[1]), pack_bf16x2(r1[2], r1[3]), pack_bf16x2(r1[4], r1[5]), pack_bf16x2(r1[6], r1[7]));
        store_chunk_swizzled_4tiles(smem_out_u32, row, col + 16, 
            pack_bf16x2(r2[0], r2[1]), pack_bf16x2(r2[2], r2[3]), pack_bf16x2(r2[4], r2[5]), pack_bf16x2(r2[6], r2[7]));
        store_chunk_swizzled_4tiles(smem_out_u32, row, col + 24, 
            pack_bf16x2(r3[0], r3[1]), pack_bf16x2(r3[2], r3[3]), pack_bf16x2(r3[4], r3[5]), pack_bf16x2(r3[6], r3[7]));
    }
    
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();
    
    if (threadIdx.x == 0) {
        tma_store_2d_fn(tma_C, smem_out, global_n_start, global_m_start);
        tma_store_2d_fn(tma_C, (uint8_t*)smem_out + 16384, global_n_start + 64, global_m_start);
        tma_store_2d_fn(tma_C, (uint8_t*)smem_out + 32768, global_n_start + 128, global_m_start);
        tma_store_2d_fn(tma_C, (uint8_t*)smem_out + 49152, global_n_start + 192, global_m_start);
        
        asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
        asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
    }
}

__global__ void __launch_bounds__(128, 1) gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    const __grid_constant__ CUtensorMap tma_C,
    uint32_t M, uint32_t N, uint32_t K) {

    extern __shared__ __align__(1024) uint8_t raw_smem[];
    SharedStorage* smem = reinterpret_cast<SharedStorage*>(raw_smem);
    
    int tx = threadIdx.x;
    uint32_t cta_rank = cluster_rank_fn();
    
    uint32_t n_block = blockIdx.x / 2;
    uint32_t m_block = blockIdx.y;
    
    uint32_t my_m_start = m_block * 256 + cta_rank * 128;
    uint32_t my_b_n_start = n_block * 256 + cta_rank * 128;
    uint32_t my_d_n_start = n_block * 256; 
    
    int num_k_tiles = (K + BK - 1) / BK;
    
    setmaxnreg_inc_sync_fn<240>();
    
    if (tx < 32) {
        tmem_alloc_fn(&smem->tmem_addr, 256);
    }
    __syncthreads();
    
    uint32_t tmem_c = smem->tmem_addr;
    
    for (int i = 0; i < STAGES; ++i) {
        if (tx == 0) {
            init_smem_barrier_fn(&smem->empty_mbar[i], 1);
            if (cta_rank == 0) {
                init_smem_barrier_fn(&smem->full_mbar[i], 2);
            }
        }
    }
    __syncthreads();
    fence_smem_barrier_init_fn();
    __syncthreads();
    cluster_sync_fn();
    
    if (tx == 0) {
        for (int i = 0; i < STAGES - 1 && i < num_k_tiles; ++i) {
            uint32_t my_tx_bytes = 4 * (BM_CTA * 64 * 2); 
            if (cta_rank == 0) {
                mbarrier_arrive_and_expect_tx_fn(&smem->full_mbar[i], my_tx_bytes);
            } else {
                mbarrier_arrive_expect_tx_cluster_fn(&smem->full_mbar[i], my_tx_bytes, 0);
            }
            tma_load_2d_cg2_fn(&tma_A, &smem->full_mbar[i], smem->A[i], i * 128, my_m_start);
            tma_load_2d_cg2_fn(&tma_A, &smem->full_mbar[i], (uint8_t*)smem->A[i] + 16384, i * 128 + 64, my_m_start);
            tma_load_2d_cg2_fn(&tma_B, &smem->full_mbar[i], smem->B[i], i * 128, my_b_n_start);
            tma_load_2d_cg2_fn(&tma_B, &smem->full_mbar[i], (uint8_t*)smem->B[i] + 16384, i * 128 + 64, my_b_n_start);
        }
    }
    
    uint32_t idesc = make_instr_desc_fn(256, 256);
    uint32_t lbo = 1;
    uint32_t sbo = 1024;
    
    for (int k = 0; k < num_k_tiles; ++k) {
        int stage = k % STAGES;
        int next_k = k + STAGES - 1;
        int load_stage = next_k % STAGES;
        
        if (next_k < num_k_tiles) {
            if (tx == 0) {
                if (next_k >= STAGES) {
                    int prev_k = next_k - STAGES;
                    mbarrier_wait_fn(&smem->empty_mbar[load_stage], (prev_k / STAGES) & 1);
                }
                
                uint32_t my_tx_bytes = 4 * (BM_CTA * 64 * 2);
                if (cta_rank == 0) {
                    mbarrier_arrive_and_expect_tx_fn(&smem->full_mbar[load_stage], my_tx_bytes);
                } else {
                    mbarrier_arrive_expect_tx_cluster_fn(&smem->full_mbar[load_stage], my_tx_bytes, 0);
                }
                
                tma_load_2d_cg2_fn(&tma_A, &smem->full_mbar[load_stage], smem->A[load_stage], next_k * 128, my_m_start);
                tma_load_2d_cg2_fn(&tma_A, &smem->full_mbar[load_stage], (uint8_t*)smem->A[load_stage] + 16384, next_k * 128 + 64, my_m_start);
                tma_load_2d_cg2_fn(&tma_B, &smem->full_mbar[load_stage], smem->B[load_stage], next_k * 128, my_b_n_start);
                tma_load_2d_cg2_fn(&tma_B, &smem->full_mbar[load_stage], (uint8_t*)smem->B[load_stage] + 16384, next_k * 128 + 64, my_b_n_start);
            }
        }
        
        if (cta_rank == 0 && tx == 0) {
            mbarrier_wait_fn(&smem->full_mbar[stage], (k / STAGES) & 1);
            fence_async_shared_fn();
            
            #pragma unroll
            for (int step = 0; step < 8; ++step) {
                int block_idx = step / 4;
                int step_in_block = step % 4;
                
                uint64_t desc_a = make_smem_desc_sm100_fn(
                    (uint8_t*)smem->A[stage] + block_idx * 16384, 
                    lbo, sbo, step_in_block * 32);
                uint64_t desc_b = make_smem_desc_sm100_fn(
                    (uint8_t*)smem->B[stage] + block_idx * 16384, 
                    lbo, sbo, step_in_block * 32);
                    
                uint32_t accum = (k == 0 && step == 0) ? 0 : 1;
                umma_f16_cg2_fn(tmem_c, desc_a, desc_b, idesc, accum);
            }
            umma_commit_2sm_fn(&smem->empty_mbar[stage]);
        }
    }
    
    if (tx == 0) {
        int last_k = num_k_tiles - 1;
        int last_stage = last_k % STAGES;
        mbarrier_wait_fn(&smem->empty_mbar[last_stage], (last_k / STAGES) & 1);
        tcgen05_fence_before_fn();
    }
    __syncthreads();
    tcgen05_fence_after_fn();
    
    __nv_bfloat16* smem_out = (__nv_bfloat16*)smem;
    my_tmem_epilogue_tma_fn(tmem_c, &tma_C, smem_out, my_m_start, my_d_n_start, BM_CTA, 256);
    
    __syncthreads();
    cluster_sync_fn();
    
    if (tx < 32) {
        tmem_dealloc_fn(tmem_c, 256);
    }
}

CUresult create_tma_2d_descriptor_2B(
    CUtensorMap* d, void* globalAddress, 
    uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, 
    uint32_t smem_inner_dim, uint32_t smem_outer_dim, 
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
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

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);
    
    const __nv_bfloat16* a_ptr = static_cast<const __nv_bfloat16*>(A.data_ptr());
    const __nv_bfloat16* b_ptr = static_cast<const __nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* c_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());
    
    CUtensorMap tma_A, tma_B, tma_C;
    CUresult res_A = create_tma_2d_descriptor_2B(&tma_A, (void*)a_ptr, K, M, 64, BM_CTA, 
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res_A != 0) {
        fprintf(stderr, "TMA A descriptor creation failed with %d\n", res_A);
        exit(1);
    }
        
    CUresult res_B = create_tma_2d_descriptor_2B(&tma_B, (void*)b_ptr, K, N, 64, BN_CTA, 
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res_B != 0) {
        fprintf(stderr, "TMA B descriptor creation failed with %d\n", res_B);
        exit(1);
    }
    
    CUresult res_C = create_tma_2d_descriptor_2B(&tma_C, (void*)c_ptr, N, M, 64, BM_CTA, 
        CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res_C != 0) {
        fprintf(stderr, "TMA C descriptor creation failed with %d\n", res_C);
        exit(1);
    }
        
    dim3 grid(((N + 255) / 256) * 2, (M + 255) / 256);
    dim3 block(128);
    size_t smem_bytes = sizeof(SharedStorage);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_bytes;
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, tma_A, tma_B, tma_C, (uint32_t)M, (uint32_t)N, (uint32_t)K));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda