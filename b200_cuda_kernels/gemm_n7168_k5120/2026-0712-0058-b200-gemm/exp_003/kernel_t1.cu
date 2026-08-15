#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>
#include <driver_types.h>

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
        fprintf(stderr, "CUDA Driver error %d at %s:%d\n",         \
                (int)_e, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_cuda_gemm {

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

__device__ __forceinline__ void tma_load_2d_cta_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tmem_alloc_fn_cg2(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn_cg2(uint32_t addr, int ncols) {
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
        "tcgen05.commit.cta_group::2.shared::cluster.mbarrier::arrive::one.b64"
        " [%0];"
        :: "r"(a));
}

__device__ __forceinline__ uint32_t make_swizzled_addr(void* base, int row, int col_bytes) {
    uint32_t base_addr = (uint32_t)__cvta_generic_to_shared(base);
    int x_chunk = col_bytes / 16;
    int swizzled_x_chunk = (row % 8) ^ x_chunk;
    int final_col = (swizzled_x_chunk * 16) + (col_bytes % 16);
    return base_addr + final_col;
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(uint32_t addr) {
    uint64_t d = 0;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
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
    d |= (1u << 16);    
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ void cluster_sync_fn() {
    asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n" ::: "memory");
}

__device__ __forceinline__ void named_barrier_sync_fn(int bar_id, int count) {
    asm volatile("barrier.sync.aligned %0, %1;" :: "r"(bar_id), "r"(count));
}

__global__ void __launch_bounds__(128) gemm_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* C, uint32_t M, uint32_t N) 
{
    extern __shared__ __align__(128) uint8_t smem[];
    uint64_t* mbar_smem = (uint64_t*)smem;
    uint64_t* mbar_cta1 = (uint64_t*)(smem + 16);
    uint32_t* tmem_addr = (uint32_t*)(smem + 32);
    
    uint32_t smem_base = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t align_offset = (1024 - (smem_base & 1023)) & 1023;
    uint64_t offset_A = 1024 + align_offset;
    __nv_bfloat16* smem_A = (__nv_bfloat16*)(smem + offset_A);
    uint64_t offset_B = offset_A + 8192;
    if (offset_B & 1023) {
        offset_B = (offset_B + 1023) & ~1023;
    }
    __nv_bfloat16* smem_B = (__nv_bfloat16*)(smem + offset_B);
    
    uint32_t cta_id = cluster_rank_fn();
    uint32_t m_block = blockIdx.x * 128;
    uint32_t n_block = blockIdx.y * 256;
    uint32_t tid = threadIdx.x;
    
    uint32_t m_base = m_block + cta_id * 64;
    uint32_t N_block = n_block + cta_id * 128;
    
    if (tid == 0) {
        tmem_alloc_fn_cg2(&tmem_addr[0], 128);
    }
    __syncthreads();
    
    if (tid == 0) {
        init_smem_barrier_fn(mbar_smem, 1);
        init_smem_barrier_fn(mbar_cta1, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    uint32_t idesc = make_instr_desc_fn(128, 256);
    uint32_t phase = 0;
    
    for (uint32_t k_chunk = 0; k_chunk < 5120; k_chunk += 64) {
        if (tid == 0) {
            uint32_t tx_bytes = 24576; 
            mbarrier_arrive_and_expect_tx_fn(mbar_smem, tx_bytes);
            
            tma_load_2d_cta_fn(&tma_A, mbar_smem, smem_A, k_chunk, m_base);
            tma_load_2d_cta_fn(&tma_B, mbar_smem, smem_B, k_chunk, N_block);
        }
        mbarrier_wait_fn(mbar_smem, phase);
        phase ^= 1;
        
        if (tid == 0) {
            uint32_t tmem_c = tmem_addr[0];
            
            for (uint32_t k = 0; k < 4; k++) {
                int row = k * 1024;
                int col = k * 32;
                
                uint32_t swizzled_addr_A = make_swizzled_addr(smem_A, row, col);
                uint64_t desc_A = ((uint64_t)(swizzled_addr_A & 0x3FFFF) >> 4) | ((uint64_t)1 << 46) | ((uint64_t)2 << 61);
                
                uint32_t swizzled_addr_B = make_swizzled_addr(smem_B, row, col);
                uint64_t desc_B = ((uint64_t)(swizzled_addr_B & 0x3FFFF) >> 4) | ((uint64_t)1 << 46) | ((uint64_t)2 << 61);
                
                if (k_chunk == 0 && k == 0) {
                    umma_f16_cg2_fn(tmem_c, desc_A, desc_B, idesc, 0);
                } else {
                    umma_f16_cg2_fn(tmem_c, desc_A, desc_B, idesc, 1);
                }
            }
            
            umma_commit_2sm_fn(mbar_cta1);
            mbarrier_wait_fn(mbar_cta1, phase);
            phase ^= 1;
        }
        cluster_sync_fn();
        __syncthreads();
    }
    
    named_barrier_sync_fn(1, 128);
    
    uint32_t warp_id = tid / 32;
    uint32_t warp_lane_base = warp_id * 32;
    
    for (uint32_t col = 0; col < 64; col += 4) {
        uint32_t taddr = (warp_lane_base << 16) | col;
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(taddr));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        uint32_t m_idx = m_base + warp_lane_base + (tid % 32);
        uint32_t global_col = N_block + col;
        
        if (m_idx < M) {
            __nv_bfloat16* out = C + (uint64_t)m_idx * N + global_col;
            if (global_col < N) out[0] = __float2bfloat16(__uint_as_float(r0));
            if (global_col + 1 < N) out[1] = __float2bfloat16(__uint_as_float(r1));
            if (global_col + 2 < N) out[2] = __float2bfloat16(__uint_as_float(r2));
            if (global_col + 3 < N) out[3] = __float2bfloat16(__uint_as_float(r3));
        }
    }
    
    if (tid == 0) {
        tmem_dealloc_fn_cg2(tmem_addr[0], 128);
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

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    uint32_t M = A.size(0);
    uint32_t N = 7168;
    uint32_t K = 5120;

    CUtensorMap tma_A, tma_B;
    CU_CHECK(create_tma_2d_descriptor_2B(
        &tma_A, A.data_ptr(), K, M, 64, 64, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(
        &tma_B, B.data_ptr(), K, N, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    dim3 grid((M + 127) / 128, N / 256);
    dim3 block(128);
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 49152;
    config.stream = stream;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(&config, gemm_kernel, tma_A, tma_B, static_cast<__nv_bfloat16*>(C.data_ptr()), M, N));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_cuda_gemm::run);

} // namespace tvm_ffi_cuda_gemm