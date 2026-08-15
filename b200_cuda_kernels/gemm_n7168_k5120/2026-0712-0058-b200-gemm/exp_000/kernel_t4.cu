#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
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

#define CU_CHECK_DRIVER(call) do {                                 \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        const char* err_name = nullptr;                            \
        cuGetErrorName(_e, &err_name);                             \
        fprintf(stderr, "CUDA Driver error %s at %s:%d\n",         \
                err_name ? err_name : "unknown", __FILE__, __LINE__); \
        exit(1);                                                   \
    }                                                              \
} while(0)

// ---------------- Device Helper Functions ----------------

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

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
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void umma_f16_cg1_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_1sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.shared::cta.b64"
        " [%0];"
        :: "r"(a));
}

__device__ __forceinline__ uint64_t make_smem_desc_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16; 
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32; 
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

// Helper to easily construct correctly-aligned descriptors accounting for logical-to-physical offset mapping.
__device__ __forceinline__ uint64_t make_smem_desc_k_major(void* smem_ptr, int block_m, int block_k) {
    (void)block_k;
    uint32_t sbo = (block_m > 8) ? (8 * 128) : 128;
    return make_smem_desc_fn(smem_ptr, 1, sbo);
}

__device__ __forceinline__ uint64_t make_smem_desc_n_major(void* smem_ptr, int block_m, int block_k) {
    uint32_t sbo = (block_k > 8) ? (8 * 128) : 128;
    uint32_t lbo = (block_k > 8) ? (sbo * (block_k / 8)) : 128;
    return make_smem_desc_fn(smem_ptr, lbo, sbo);
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, bool transposed) {
    uint32_t d = 0;
    d |= (1u << 4);     
    d |= (1u << 7);     
    d |= (1u << 10);    
    d |= (0u << 15);    
    d |= ((transposed ? 1 : 0) << 16);    
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapDataType dataType, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, dataType, 2, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill
    );
}


// ---------------- CUTLASS-style Kernel ----------------

__global__ void __launch_bounds__(128, 1) gemm_n7168_k5120_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* C,
    uint32_t M, uint32_t N) 
{
    setmaxnreg_inc_sync_fn<248>();

    extern __shared__ uint8_t smem_raw[];
    uintptr_t raw_addr = reinterpret_cast<uintptr_t>(smem_raw);
    // Dynamic shared memory alignment patch
    uintptr_t aligned_addr = (raw_addr + 1023) & ~1023;
    uint8_t* smem = (uint8_t*)aligned_addr;

    uint8_t* smem_A_0 = smem;                 // 8 KB
    uint8_t* smem_A_1 = smem + 8192;          // 8 KB
    uint8_t* smem_B_0 = smem + 16384;         // 8 KB
    uint8_t* smem_B_1 = smem + 24576;         // 8 KB
    
    uint64_t* bar_A = (uint64_t*)(smem + 32768);
    uint64_t* bar_B = (uint64_t*)(smem + 32776);
    uint32_t* tmem_D_ptr = (uint32_t*)(smem + 32784);

    uint32_t c_rank = cluster_rank_fn();
    uint32_t m_offset = blockIdx.x * 128;
    uint32_t n_block = blockIdx.y * 64;

    uint32_t m_base = m_offset + c_rank * 64;

    if (m_base >= M) return; 

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar_A, 1);
        init_smem_barrier_fn(bar_B, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads(); 
    
    uint32_t tmem_D = 0;
    if (threadIdx.x == 0) {
        tmem_alloc_fn(tmem_D_ptr, 64);
        tmem_D = *tmem_D_ptr;
    }
    __syncthreads();
    tmem_D = *tmem_D_ptr;

    // Fixed: Correctly flags B as transposed, utilizing N-major 128B swizzle for B, resolving silent incorrect outputs
    uint32_t idesc = make_instr_desc_fn(64, 64, true);
    
    uint32_t phase_A = 0;
    uint32_t phase_B = 0;
    uint32_t accum = 0;

    for (uint32_t k_tile = 0; k_tile < 40; ++k_tile) {
        uint32_t k_offset = k_tile * 128;

        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_A, 16384);
            tma_load_2d_fn(&tma_A, bar_A, smem_A_0, k_offset, m_base);
            tma_load_2d_fn(&tma_A, bar_A, smem_A_1, k_offset + 64, m_base);
        }

        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_B, 16384);
            tma_load_2d_fn(&tma_B, bar_B, smem_B_0, k_offset, n_block);
            tma_load_2d_fn(&tma_B, bar_B, smem_B_1, k_offset + 64, n_block);
        }

        mbarrier_wait_fn(bar_A, phase_A);
        mbarrier_wait_fn(bar_B, phase_B);
        phase_A ^= 1;
        phase_B ^= 1;
        
        fence_proxy_async_fn();

        if (threadIdx.x == 0) {
            uint64_t desc_a0 = make_smem_desc_k_major(smem_A_0, 64, 64);
            uint64_t desc_b0 = make_smem_desc_n_major(smem_B_0, 64, 64);
            
            // 1st TMA load processing (K = 0 .. 63)
            for (uint32_t k = 0; k < 4; ++k) {
                umma_f16_cg1_fn(tmem_D, desc_a0, desc_b0, idesc, accum);
                accum = 1;
                desc_a0 += 2;
                desc_b0 += 128;
            }
            
            uint64_t desc_a1 = make_smem_desc_k_major(smem_A_1, 64, 64);
            uint64_t desc_b1 = make_smem_desc_n_major(smem_B_1, 64, 64);
            
            // 2nd TMA load processing (K = 64 .. 127)
            for (uint32_t k = 0; k < 4; ++k) {
                umma_f16_cg1_fn(tmem_D, desc_a1, desc_b1, idesc, accum);
                desc_a1 += 2;
                desc_b1 += 128;
            }
            
            umma_commit_1sm_fn(bar_A);
        }

        mbarrier_wait_fn(bar_A, phase_A);
        phase_A ^= 1;
    }
    
    // Fixed: Simplified highly efficient epilogue writing. Removed problematic intermediate SMEM staging and massive local memory usage.
    uint32_t tid = threadIdx.x;
    uint32_t m_idx = tid;

    // Out of bounds guard
    if (m_idx < 64) {
        uint32_t global_row = m_base + m_idx;
        if (global_row < M) {
            for (uint32_t col = 0; col < 64; col += 4) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                
                uint32_t global_col = n_block + col;
                
                // Out of bounds guard
                if (global_col + 3 < N) {
                    float f0 = __uint_as_float(r0);
                    float f1 = __uint_as_float(r1);
                    float f2 = __uint_as_float(r2);
                    float f3 = __uint_as_float(r3);
                    
                    __nv_bfloat16 out_vec[4];
                    out_vec[0] = __float2bfloat16(f0);
                    out_vec[1] = __float2bfloat16(f1);
                    out_vec[2] = __float2bfloat16(f2);
                    out_vec[3] = __float2bfloat16(f3);
                    
                    // Perform direct vectorized store avoiding uncoalesced scalar writes
                    uint2 data = *reinterpret_cast<uint2*>(out_vec);
                    *reinterpret_cast<uint2*>(C + (uint64_t)global_row * N + global_col) = data;
                }
            }
        }
    }
    
    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_D, 64);
    }
}

namespace tvm_ffi {
void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    uint32_t M = A.size(0);
    uint32_t N = 7168;
    uint32_t K = 5120;
    
    CUtensorMap tma_A, tma_B;
    CU_CHECK_DRIVER(create_tma_2d_descriptor_2B(&tma_A, A.data_ptr(), K, M, 64, 64, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK_DRIVER(create_tma_2d_descriptor_2B(&tma_B, B.data_ptr(), K, N, 64, 64, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        
    dim3 grid((M + 127) / 128, N / 64, 1);
    dim3 block(128);
    
    uint32_t smem_size = 32784 + 128;
    CUDA_CHECK(cudaFuncSetAttribute(gemm_n7168_k5120_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_size;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    config.stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    cudaLaunchKernelEx(&config, gemm_n7168_k5120_kernel, tma_A, tma_B, static_cast<__nv_bfloat16*>(C.data_ptr()), M, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(config.stream));
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
} // namespace tvm_ffi