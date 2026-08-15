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
//   [mbarrier: 8 bytes]
//   [smem_A: BM*BK*2 = 64*16*2 = 2048 bytes]  
//   [smem_B: BN*BK*2 = 128*16*2 = 4096 bytes]
constexpr uint32_t SMEM_A_BYTES = BLOCK_M * BLOCK_K * 2;
constexpr uint32_t SMEM_B_BYTES = BLOCK_N * BLOCK_K * 2;
constexpr uint32_t TOTAL_SMEM_BYTES = 8 + SMEM_A_BYTES + SMEM_B_BYTES;

// ============================================================
// Device-side utility functions
// ============================================================

__device__ __forceinline__ void init_mbarrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}

__device__ __forceinline__ void fence_mbarrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t parity) {
    asm volatile(
        ".reg .pred P_%;\n"
        "WAIT_LOOP_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P_%_, [%0], %1;\n"
        "@!P_%_ bra WAIT_LOOP_%=;\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(parity));
}

__device__ __forceinline__ void tma_load_2d_fn(
    const CUtensorMap* desc, uint64_t* mbar, void* smem_dst,
    int32_t coord0, int32_t coord1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem_dst)),
           "l"(desc),
           "r"((uint32_t)__cvta_generic_to_shared(mbar)),
           "r"(coord0), "r"(coord1) : "memory");
}

__device__ __forceinline__ uint64_t make_smem_descriptor(void* smem_ptr, uint32_t lbo_bytes, uint32_t sbo_bytes) {
    uint64_t desc = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    desc |= (uint64_t)((addr & 0x3FFFF) >> 4);                         // bits 0-13: start addr
    desc |= (uint64_t)((lbo_bytes & 0x3FFFF) >> 4) << 16;             // bits 16-29: LBO
    desc |= (uint64_t)((sbo_bytes & 0x3FFFF) >> 4) << 32;             // bits 32-45: SBO
    desc |= (uint64_t)1ULL << 46;                                     // version=1
    desc |= (uint64_t)2ULL << 61;                                     // swizzle=128B
    return desc;
}

__device__ __forceinline__ uint32_t make_umma_idesc(uint32_t M, uint32_t N) {
    uint32_t idesc = 0;
    idesc |= (1u << 4);     // dtype = FP32 (output accumulator)
    idesc |= (1u << 7);     // atype = BF16
    idesc |= (1u << 10);    // btype = BF16
    // Bits 15-16: transpose both 0 = K-major
    idesc |= ((N >> 3) & 0x3Fu) << 17;   // N dimension
    idesc |= ((M >> 4) & 0x1Fu) << 24;   // M dimension
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

    static __shared__ uint64_t sm_barrier[1];
    __shared__ unsigned char smem_buf[TOTAL_SMEM_BYTES];

    uint64_t* sm_barrier_ptr = reinterpret_cast<uint64_t*>(smem_buf);
    __nv_bfloat16* smem_A = reinterpret_cast<__nv_bfloat16*>(smem_buf + 8);
    __nv_bfloat16* smem_B = reinterpret_cast<__nv_bfloat16*>(smem_buf + 8 + SMEM_A_BYTES);

    // Thread 0 initializes mbarrier (expects 2 arrivals: A load + B load)
    if (threadIdx.x == 0) {
        init_mbarrier_fn(sm_barrier_ptr, 2);
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

    // Precompute shared memory descriptors
    // For K-major with 128B swizzle:
    // A: BM rows × BK cols, SBO = 8*128 = 1024 (for 128B swizzle spanning 8 spans along MN-mode)
    //    Actually for K-major with 128B swizzle: ATOM_MMODE_DIM=BM=64, ATOM_KMODE_DIM=BK=16
    //    SBO = 8 * 128 = 1024, LBO is not used (assumed 1) but we encode as (BM/8)*SBO = 8*1024=8192
    uint32_t SBO_A = 1024;
    uint32_t LBO_A = (BLOCK_M / 8) * SBO_A;  // 8 core matrices along M-mode
    
    uint32_t SBO_B = 1024;
    uint32_t LBO_B = (BLOCK_N / 8) * SBO_B;  // 16 core matrices along N-mode
    
    uint64_t smem_desc_A = make_smem_descriptor(smem_A, LBO_A, SBO_A);
    uint64_t smem_desc_B = make_smem_descriptor(smem_B, LBO_B, SBO_B);

    // Instruction descriptor
    uint32_t idesc = make_umma_idesc(BLOCK_M, BLOCK_N);

    // Allocate Tensor Memory for C accumulator (BLOCK_N columns of FP32)
    uint32_t tmem_c_addr;
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 : "=r"(tmem_c_addr)
                 : "r"((uint32_t)BLOCK_N));

    // Clear accumulator via UMMA with D disabled
    uint32_t accum_dis_flag = 0;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.eq.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n"
        "}\n"
        :: "r"(tmem_c_addr), "l"(smem_desc_A), "l"(smem_desc_B), "r"(idesc), "r"(accum_dis_flag));
    
    // Commit and wait for clear
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                 :: "r"((uint32_t)__cvta_generic_to_shared(sm_barrier_ptr)));
    if (threadIdx.x == 0) {
        mbarrier_wait_fn(sm_barrier_ptr, 0);
    }
    __syncthreads();

    // Prefetch TMA descriptors
    if (threadIdx.x == 0) {
        asm volatile("prefetch.tensormap [%0];" :: "l"(&tma_A) : "memory");
        asm volatile("prefetch.tensormap [%0];" :: "l"(&tma_B) : "memory");
    }

    // Main K-loop
    for (uint32_t k_step = 0; k_step < k_iter_max; ++k_step) {
        uint32_t k_off = k_step * BLOCK_K;

        // Issue TMA loads
        // A coordinates: inner=k_off, outer=a_row_base → loads A[a_row_base:a_row_base+BM, k_off:k_off+BK]
        // B coordinates: inner=k_off, outer=b_col_base → loads B[b_col_base:b_col_base+BN, k_off:k_off+BK]
        tma_load_2d_fn(&tma_A, sm_barrier_ptr, smem_A, 
                        static_cast<int32_t>(k_off), static_cast<int32_t>(a_row_base));
        tma_load_2d_fn(&tma_B, sm_barrier_ptr, smem_B, 
                        static_cast<int32_t>(k_off), static_cast<int32_t>(b_col_base));

        // Wait for TMA completion
        if (threadIdx.x == 0) {
            mbarrier_wait_fn(sm_barrier_ptr, 0);
        }
        __syncthreads();

        // Rebuild descriptors with current base addresses
        uint64_t desc_a = smem_desc_A;
        uint64_t desc_b = smem_desc_B;
        
        uint32_t addr_a = (uint32_t)__cvta_generic_to_shared(smem_A);
        uint32_t addr_b = (uint32_t)__cvta_generic_to_shared(smem_B);
        desc_a &= ~UINT64_C(0x3FFF);
        desc_b &= ~UINT64_C(0x3FFF);
        desc_a |= (uint64_t)(addr_a >> 4);
        desc_b |= (uint64_t)(addr_b >> 4);

        // Issue UMMA: D += A @ B
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "setp.ne.b32 p, 1, 0;\n"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n"
            "}\n"
            :: "r"(tmem_c_addr), "l"(desc_a), "l"(desc_b), "r"(idesc));

        // Commit UMMA
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                     :: "r"((uint32_t)__cvta_generic_to_shared(sm_barrier_ptr)));

        // Wait before next iteration
        if (threadIdx.x == 0) {
            mbarrier_wait_fn(sm_barrier_ptr, 0);
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
                 :: "r"(tmem_c_addr), "r"((uint32_t)BLOCK_N));
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

    // Copy TMA descriptors to device constant-memory-accessible location
    CUtensorMap *d_tma_A, *d_tma_B;
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