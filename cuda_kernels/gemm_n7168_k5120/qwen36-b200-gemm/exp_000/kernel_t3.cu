#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_RT_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                          \
    if (_e != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                  \
                cudaGetErrorString(_e), __FILE__, __LINE__);          \
        exit(1);                                                      \
    }                                                                 \
} while(0)

#define CUDA_DRV_CHECK(call) do {                                     \
    CUresult _r = (call);                                             \
    if (_r != CUDA_SUCCESS) {                                         \
        const char *_msg;                                             \
        cuGetErrorName(_r, &_msg);                                    \
        fprintf(stderr, "CUDA Driver error %s at %s:%d\n",           \
                _msg, __FILE__, __LINE__);                            \
        exit(1);                                                      \
    }                                                                 \
} while(0)

namespace tvm_ffi_gemm_blackwell {

// ============================================================
// Host-side helpers
// ============================================================

CUresult create_tma_2d_descriptor(
    CUtensorMap* d, void* globalAddress,
    uint64_t gmem_inner_dim, uint64_t gmem_outer_dim,
    uint32_t smem_inner_dim, uint32_t smem_outer_dim,
    CUtensorMapDataType dataType,
    CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2}; // 2 bytes per BF16
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, dataType, 2, globalAddress,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill);
}

// ============================================================
// Kernel constants
// ============================================================
constexpr uint32_t BLOCK_M = 64;
constexpr uint32_t BLOCK_N = 128;
constexpr uint32_t BLOCK_K = 16;
constexpr uint32_t NUM_THREADS = 128;  // 4 warps

// Shared memory layout:
//   [mbarrier: 8 bytes aligned]
//   [smem_A: BM*BK*2 = 64*16*2 = 2048 bytes]  
//   [smem_B: BN*BK*2 = 128*16*2 = 4096 bytes]
constexpr uint32_t SMEM_A_BYTES = BLOCK_M * BLOCK_K * 2;
constexpr uint32_t SMEM_B_BYTES = BLOCK_N * BLOCK_K * 2;
constexpr uint32_t TOTAL_SMEM_BYTES = 64 + SMEM_A_BYTES + SMEM_B_BYTES;

// ============================================================
// Device-side utility functions
// ============================================================

__device__ __forceinline__ void init_mbarrier_fn(uint32_t bar_smem_addr, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"(bar_smem_addr), "r"(count));
}

__device__ __forceinline__ void fence_mbarrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_wait_fn(uint32_t bar_smem_addr, uint32_t parity) {
    asm volatile(
        "LOOP_WAIT_%=: {"
        ".reg .pred P_W_%=;"
        "mbarrier.try_wait.parity.shared.b64 P_W_%=, [%0], %1;"
        "@!P_W_%= bra LOOP_WAIT_%=;"
        "}\n"
        : : "r"(bar_smem_addr), "r"(parity) : "memory");
}

__device__ __forceinline__ void tma_load_2d_fn(
    const CUtensorMap* desc, uint32_t mbar_smem_addr, uint32_t smem_dst_smem_addr,
    int32_t coord0, int32_t coord1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%3, %4}], [%2];"
        : : "r"(smem_dst_smem_addr),
            "l"(desc),
            "r"(mbar_smem_addr),
            "r"(coord0), "r"(coord1)
        : "memory");
}

__device__ __forceinline__ uint64_t make_smem_descriptor(uint32_t smem_addr, uint32_t lbo_bytes, uint32_t sbo_bytes) {
    uint64_t desc = 0;
    desc |= (uint64_t)((smem_addr & 0x3FFFF) >> 4);                         
    desc |= (uint64_t)((lbo_bytes & 0x3FFFF) >> 4) << 16;                   
    desc |= (uint64_t)((sbo_bytes & 0x3FFFF) >> 4) << 32;                   
    desc |= (uint64_t)1ULL << 46;                                           
    desc |= (uint64_t)2ULL << 61;                                          
    return desc;
}

__device__ __forceinline__ uint32_t make_umma_idesc(uint32_t M, uint32_t N) {
    uint32_t idesc = 0;
    idesc |= (1u << 4);     // dtype = FP32 (output accumulator)
    idesc |= (1u << 7);     // atype = BF16
    idesc |= (1u << 10);    // btype = BF16
    // Bits 15-16: transpose both 0 = K-major
    idesc |= ((N >> 3) & 0x3Fu) << 17;   
    idesc |= ((M >> 4) & 0x1Fu) << 24;   
    return idesc;
}

// ============================================================
// Kernel
// ============================================================

__global__ void gemm_kernel(const __nv_bfloat16* __restrict__ A,
                             const __nv_bfloat16* __restrict__ B,
                             __nv_bfloat16* __restrict__ C,
                             const __grid_constant__ CUtensorMap tma_A,
                             const __grid_constant__ CUtensorMap tma_B,
                             uint32_t M, uint32_t N, uint32_t K) {

    extern __shared__ unsigned char smem_buf[];

    uint32_t base_addr = (uint32_t)__cvta_generic_to_shared(smem_buf);
    uint32_t barrier_addr = (base_addr + 15) & ~15u;
    
    constexpr uint32_t BARRIER_OFFSET = 16;
    uint32_t smem_A_offset = BARRIER_OFFSET + 8;  
    uint32_t smem_B_offset = smem_A_offset + SMEM_A_BYTES;
    
    uint32_t smem_A_addr = barrier_addr + smem_A_offset;
    uint32_t smem_B_addr = barrier_addr + smem_B_offset;

    // Thread 0 initializes mbarrier (expects 2 arrivals: A load + B load)
    if (threadIdx.x == 0) {
        init_mbarrier_fn(barrier_addr, 2);
        fence_mbarrier_init_fn();
    }
    __syncthreads();

    // Compute block-level offsets
    uint32_t grid_x = static_cast<uint32_t>(blockIdx.x);
    uint32_t grid_n_blocks = (N + BLOCK_N - 1) / BLOCK_N;
    uint32_t n_block = grid_x % grid_n_blocks;
    uint32_t m_block = grid_x / grid_n_blocks;
    
    uint32_t a_row_base = m_block * BLOCK_M;
    uint32_t b_col_base = n_block * BLOCK_N;
    uint32_t k_iter_max = K / BLOCK_K;

    // Precompute shared memory descriptors (K-major, 128B swizzle)
    uint32_t SBO = 1024;
    uint32_t LBO_A = (BLOCK_M / 8) * SBO;  
    uint32_t LBO_B = (BLOCK_N / 8) * SBO;  
    
    uint64_t smem_desc_A = make_smem_descriptor(smem_A_addr, LBO_A, SBO);
    uint64_t smem_desc_B = make_smem_descriptor(smem_B_addr, LBO_B, SBO);

    // Instruction descriptor: BF16 x BF16 -> FP32, M=64, N=128, K=16
    uint32_t idesc = make_umma_idesc(BLOCK_M, BLOCK_N);

    // Allocate Tensor Memory for C accumulator (BLOCK_N columns of FP32)
    uint32_t tmem_c_addr;
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 : "=r"(tmem_c_addr)
                 : "r"((uint32_t)BLOCK_N));

    // Clear accumulator via UMMA (enable-input-d = false)
    asm volatile("{\n\t"
        ".reg .pred clear_p;\n\t"
        "setp.eq.b32 clear_p, 0, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, clear_p;\n\t"
        "}"
        : : "r"(tmem_c_addr), "l"(smem_desc_A), "l"(smem_desc_B), "r"(idesc));
    
    // Commit clear operation
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                 : : "r"(barrier_addr));
    if (threadIdx.x == 0) {
        mbarrier_wait_fn(barrier_addr, 0);
    }
    __syncthreads();

    // Prefetch TMA descriptors
    if (threadIdx.x == 0) {
        asm volatile("prefetch.tensormap [%0];" : : "l"(&tma_A));
        asm volatile("prefetch.tensormap [%0];" : : "l"(&tma_B));
    }

    // Main K-loop with pipelined loads and computation
    for (uint32_t k_step = 0; k_step < k_iter_max; ++k_step) {
        uint32_t k_off = k_step * BLOCK_K;

        // Issue TMA loads
        tma_load_2d_fn(&tma_A, barrier_addr, smem_A_addr, 
                        static_cast<int32_t>(k_off), static_cast<int32_t>(a_row_base));
        tma_load_2d_fn(&tma_B, barrier_addr, smem_B_addr, 
                        static_cast<int32_t>(k_off), static_cast<int32_t>(b_col_base));

        // Wait for TMA completion
        if (threadIdx.x == 0) {
            mbarrier_wait_fn(barrier_addr, 0);
        }
        __syncthreads();

        // Issue UMMA: D += A @ B
        asm volatile("{\n\t"
            ".reg .pred accum_p;\n\t"
            "setp.ne.b32 accum_p, 1, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, accum_p;\n\t"
            "}"
            : : "r"(tmem_c_addr), "l"(smem_desc_A), "l"(smem_desc_B), "r"(idesc));

        // Commit UMMA to mbarrier
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                     : : "r"(barrier_addr));

        // Wait for UMMA before next iteration
        if (threadIdx.x == 0) {
            mbarrier_wait_fn(barrier_addr, 0);
        }
        __syncthreads();
    }

    // Epilogue: TMEM FP32 -> BF16 -> Global C
    uint32_t tid = threadIdx.x;
    uint32_t local_row = tid;

    if (local_row < BLOCK_M) {
        uint32_t global_row = a_row_base + local_row;
        if (global_row < M) {
            for (uint32_t col = 0; col < BLOCK_N; col += 4) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                             : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                             : "r"(static_cast<uint32_t>(col)));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

                float f0 = __uint_as_float(r0);
                float f1 = __uint_as_float(r1);
                float f2 = __uint_as_float(r2);
                float f3 = __uint_as_float(r3);

                uint32_t nc = b_col_base + col;
                __nv_bfloat16* out_ptr = C + (uint64_t)global_row * N + nc;

                if (nc + 3 < N) {
                    out_ptr[0] = __float2bfloat16(f0);
                    out_ptr[1] = __float2bfloat16(f1);
                    out_ptr[2] = __float2bfloat16(f2);
                    out_ptr[3] = __float2bfloat16(f3);
                } else {
                    if (nc < N) out_ptr[0] = __float2bfloat16(f0);
                    if (nc + 1 < N) out_ptr[1] = __float2bfloat16(f1);
                    if (nc + 2 < N) out_ptr[2] = __float2bfloat16(f2);
                }
            }
        }
    }

    // Deallocate Tensor Memory
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 : : "r"(tmem_c_addr), "r"((uint32_t)BLOCK_N));
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_RT_CHECK(cudaSetDevice(A.device().device_id));

    uint32_t M = static_cast<uint32_t>(A.size(0));
    uint32_t N = 7168u;
    uint32_t K = 5120u;

    const __nv_bfloat16* A_data = static_cast<const __nv_bfloat16*>(A.data_ptr());
    const __nv_bfloat16* B_data = static_cast<const __nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_data = static_cast<__nv_bfloat16*>(C.data_ptr());

    // Create TMA descriptors on host
    CUtensorMap h_tma_A, h_tma_B;
    
    // A: [M, K] layout, loads tile of shape [BK(inner), BM(outer)]
    CUDA_DRV_CHECK(create_tma_2d_descriptor(
        &h_tma_A, const_cast<void*>(static_cast<const void*>(A_data)),
        K, M,
        BLOCK_K, BLOCK_M,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    // B: [N, K] layout, loads tile of shape [BK(inner), BN(outer)]
    CUDA_DRV_CHECK(create_tma_2d_descriptor(
        &h_tma_B, const_cast<void*>(static_cast<const void*>(B_data)),
        K, N,
        BLOCK_K, BLOCK_N,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    // Copy TMA descriptors to device
    CUtensorMap *d_tma_A = nullptr, *d_tma_B = nullptr;
    CUDA_DRV_CHECK(cuMemAlloc(reinterpret_cast<CUdeviceptr*>(&d_tma_A), sizeof(CUtensorMap)));
    CUDA_DRV_CHECK(cuMemAlloc(reinterpret_cast<CUdeviceptr*>(&d_tma_B), sizeof(CUtensorMap)));
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    CUDA_DRV_CHECK(cuMemcpyHtoDAsync(
        reinterpret_cast<CUdeviceptr>(d_tma_A), &h_tma_A, sizeof(CUtensorMap), stream));
    CUDA_DRV_CHECK(cuMemcpyHtoDAsync(
        reinterpret_cast<CUdeviceptr>(d_tma_B), &h_tma_B, sizeof(CUtensorMap), stream));

    // Launch configuration
    dim3 blockDim(NUM_THREADS);
    uint32_t grid_m_blocks = (M + BLOCK_M - 1) / BLOCK_M;
    uint32_t grid_n_blocks = (N + BLOCK_N - 1) / BLOCK_N;
    dim3 gridDim(grid_m_blocks * grid_n_blocks);
    
    size_t smem_bytes = TOTAL_SMEM_BYTES;
    
    gemm_kernel<<<gridDim, blockDim, smem_bytes, stream>>>(
        A_data, B_data, C_data, *d_tma_A, *d_tma_B, M, N, K);
    
    CUDA_RT_CHECK(cudaGetLastError());
    CUDA_RT_CHECK(cudaStreamSynchronize(stream));

    // Cleanup
    CUDA_DRV_CHECK(cuMemFree(reinterpret_cast<CUdeviceptr>(d_tma_A)));
    CUDA_DRV_CHECK(cuMemFree(reinterpret_cast<CUdeviceptr>(d_tma_B)));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_gemm_blackwell::run);

}  // namespace tvm_ffi_gemm_blackwell