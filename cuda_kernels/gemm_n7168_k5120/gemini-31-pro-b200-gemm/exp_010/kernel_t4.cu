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

__device__ __forceinline__ void mbarrier_arrive_expect_tx_cg2_fn(uint64_t* bar, uint32_t tx) {
    uint32_t local_a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    uint32_t remote_a;
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;"
                 : "=r"(remote_a) : "r"(local_a));
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
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase & 1));
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
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

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
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

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_before_fn() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
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
    __nv_bfloat16 a = __float2bfloat16(__uint_as_float(fp32_a));
    __nv_bfloat16 b = __float2bfloat16(__uint_as_float(fp32_b));
    uint16_t a16 = *reinterpret_cast<uint16_t*>(&a);
    uint16_t b16 = *reinterpret_cast<uint16_t*>(&b);
    return ((uint32_t)b16 << 16) | (uint32_t)a16;
}

union SharedStorage {
    struct {
        alignas(1024) uint8_t A[4][16384]; // 4 stages, 128 rows, 64 cols (bf16)
        alignas(1024) uint8_t B[4][16384]; // 4 stages, 128 rows, 64 cols (bf16)
        alignas(16) uint64_t tma_bar[4];
        alignas(16) uint64_t umma_bar[4];
        alignas(16) uint32_t tmem_addr;
    };
    alignas(1024) uint8_t C_out[65536]; // Output buffer, 128x256
};

extern __shared__ uint8_t smem_pool[];

__global__ __launch_bounds__(128) void gemm_n7168_k5120_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    const __grid_constant__ CUtensorMap tma_C,
    uint32_t K) 
{
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_pool);

    uint32_t rank = cluster_rank_fn();
    int n_pair = blockIdx.x / 2;
    int m_pair = blockIdx.y;
    int num_k_tiles = (K + 63) / 64;

    if (threadIdx.x == 0) {
        if (rank == 0) {
            for (int i = 0; i < 4; ++i) init_smem_barrier_fn(&smem.tma_bar[i], 2);
        }
        for (int i = 0; i < 4; ++i) init_smem_barrier_fn(&smem.umma_bar[i], 1);
    }
    fence_smem_barrier_init_fn();
    cluster_sync_fn();

    uint32_t tx_bytes = 32768; // 16KB A + 16KB B per CTA
    int m_offset = m_pair * 256 + rank * 128;
    int n_offset = n_pair * 256 + rank * 128;

    if (threadIdx.x == 0) {
        for (int i = 0; i < 4; ++i) mbarrier_arrive_fn(&smem.umma_bar[i]); 
        
        for (int i = 0; i < 4 && i < num_k_tiles; ++i) {
            mbarrier_arrive_expect_tx_cg2_fn(&smem.tma_bar[i], tx_bytes);
            tma_load_2d_cg2_fn(&tma_A, &smem.tma_bar[i], smem.A[i], i * 64, m_offset);
            tma_load_2d_cg2_fn(&tma_B, &smem.tma_bar[i], smem.B[i], i * 64, n_offset);
        }
    }

    uint32_t tmem_c;
    if (threadIdx.x < 32) {
        tmem_alloc_fn(&smem.tmem_addr, 128);
    }
    __syncthreads();
    tmem_c = smem.tmem_addr;

    uint32_t idesc = make_instr_desc_fn(256, 256);

    for (int k = 0; k < num_k_tiles; ++k) {
        int stage = k % 4;
        
        if (rank == 0 && threadIdx.x == 0) {
            mbarrier_wait_fn(&smem.tma_bar[stage], k / 4);
            fence_proxy_async_fn();
            
            uint64_t desc_a = make_smem_desc_sm100_fn(smem.A[stage], 1, 1024);
            uint64_t desc_b = make_smem_desc_sm100_fn(smem.B[stage], 1, 1024);
            
            for (int step = 0; step < 4; ++step) {
                uint64_t desc_a_step = desc_a + (step * 32 >> 4);
                uint64_t desc_b_step = desc_b + (step * 32 >> 4);
                int accum = (k == 0 && step == 0) ? 0 : 1;
                umma_f16_cg2_fn(tmem_c, desc_a_step, desc_b_step, idesc, accum);
            }
            umma_commit_2sm_fn(&smem.umma_bar[stage]);
        }
        
        if (threadIdx.x == 0) {
            int next_k = k + 4;
            if (next_k < num_k_tiles) {
                int next_stage = next_k % 4;
                mbarrier_wait_fn(&smem.umma_bar[next_stage], (k / 4) + 1);
                
                mbarrier_arrive_expect_tx_cg2_fn(&smem.tma_bar[next_stage], tx_bytes);
                tma_load_2d_cg2_fn(&tma_A, &smem.tma_bar[next_stage], smem.A[next_stage], next_k * 64, m_offset);
                tma_load_2d_cg2_fn(&tma_B, &smem.tma_bar[next_stage], smem.B[next_stage], next_k * 64, n_offset);
            }
        }
    }

    int last_stage = (num_k_tiles - 1) % 4;
    if (threadIdx.x == 0) {
        mbarrier_wait_fn(&smem.umma_bar[last_stage], ((num_k_tiles - 1) / 4) + 1);
        tcgen05_fence_before_fn();
    }
    
    __syncthreads();
    tcgen05_fence_after_fn();

    uint32_t y = threadIdx.x;
    for (uint32_t col = 0; col < 128; col += 16) {
        uint32_t r[16];
        uint32_t taddr0 = tmem_c + col;
        uint32_t taddr1 = tmem_c + col + 8;
        
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                     : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),
                       "=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]) : "r"(taddr0));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                     : "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),
                       "=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]) : "r"(taddr1));
        
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        for (int i = 0; i < 2; ++i) {
            int current_col = col + i * 8;
            uint32_t pack0 = pack_bf16_fn(r[i*8+0], r[i*8+1]);
            uint32_t pack1 = pack_bf16_fn(r[i*8+2], r[i*8+3]);
            uint32_t pack2 = pack_bf16_fn(r[i*8+4], r[i*8+5]);
            uint32_t pack3 = pack_bf16_fn(r[i*8+6], r[i*8+7]);
            
            int chunk_idx = current_col / 32;
            int chunk_16b = (current_col % 32) / 4; 
            int swizzled_chunk_16b = (y % 8) ^ chunk_16b;
            
            int offset = y * 512 + chunk_idx * 128 + swizzled_chunk_16b * 16;
            *reinterpret_cast<uint4*>(&smem.C_out[offset]) = make_uint4(pack0, pack1, pack2, pack3);
        }
    }
    
    __syncthreads();

    if (threadIdx.x == 0) {
        tma_store_fence_fn();
        tma_store_2d_fn(&tma_C, smem.C_out, n_pair * 256, m_offset);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }
    
    __syncthreads();
    
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_c, 128);
    }
}

namespace tvm_ffi_gemm_n7168_k5120 {

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id)); 

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);

    if (M == 0 || N == 0 || K == 0) return;

    __nv_bfloat16* a_ptr = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* b_ptr = static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* c_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());

    CUtensorMap tma_A, tma_B, tma_C;

    cuuint64_t gDim_A[2] = {(cuuint64_t)K, (cuuint64_t)M};
    cuuint64_t gStrides_A[1] = {(cuuint64_t)K * 2};
    cuuint32_t bDim_A[2] = {64, 128};
    cuuint32_t eStrides[2] = {1, 1};

    CU_CHECK(cuTensorMapEncodeTiled(
        &tma_A, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, a_ptr,
        gDim_A, gStrides_A, bDim_A, eStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));

    cuuint64_t gDim_B[2] = {(cuuint64_t)K, (cuuint64_t)N};
    cuuint64_t gStrides_B[1] = {(cuuint64_t)K * 2};
    cuuint32_t bDim_B[2] = {64, 128};

    CU_CHECK(cuTensorMapEncodeTiled(
        &tma_B, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, b_ptr,
        gDim_B, gStrides_B, bDim_B, eStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));

    cuuint64_t gDim_C[2] = {(cuuint64_t)N, (cuuint64_t)M};
    cuuint64_t gStrides_C[1] = {(cuuint64_t)N * 2};
    cuuint32_t bDim_C[2] = {256, 128};

    CU_CHECK(cuTensorMapEncodeTiled(
        &tma_C, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, c_ptr,
        gDim_C, gStrides_C, bDim_C, eStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));

    int grid_x = (N + 255) / 256;
    int grid_y = (M + 255) / 256;
    
    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    int smem_size = sizeof(SharedStorage);
    CUDA_CHECK(cudaFuncSetAttribute(gemm_n7168_k5120_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(grid_x * 2, grid_y);
    config.blockDim = dim3(128);
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_n7168_k5120_kernel, tma_A, tma_B, tma_C, K));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_gemm_n7168_k5120::run);

}  // namespace tvm_ffi_gemm_n7168_k5120