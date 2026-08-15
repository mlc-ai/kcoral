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
           "l"((const CUtensorMap*)desc),
           "r"((uint32_t)__cvta_generic_to_shared(mbar)),
           "r"(coord0), "r"(coord1) : "memory");
}

__device__ __forceinline__ uint64_t make_smem_descriptor_128b_swizzle(
    void* smem_ptr, uint32_t lbo_bytes, uint32_t sbo_bytes) {
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
    // Bits 15-16: transpose (both 0 = K-major for both A and B)
    idesc |= ((N >> 3) & 0x3Fu) << 17;   // N dimension (lower 3 bits implied)
    idesc |= ((M >> 4) & 0x1Fu) << 24;   // M dimension (lower 4 bits implied)
    return idesc;
}

// ============================================================
// Kernel constants
// ============================================================
constexpr uint32_t BLOCK_M = 64;
constexpr uint32_t BLOCK_N = 128;
constexpr uint32_t BLOCK_K = 16;
constexpr uint32_t WARP_SIZE = 32;
constexpr uint32_t NUM_THREADS = 128;  // 4 warps
constexpr uint32_t TMEM_COLS = BLOCK_N;  // 128 columns for C accumulator

// Shared memory layout:
//   [mbarrier: 8 bytes]
//   [smem_A: BM*BK*2 = 64*16*2 = 2048 bytes]
//   [smem_B: BN*BK*2 = 128*16*2 = 4096 bytes]
//   [smem_out: BM*BN*2 = 64*128*2 = 16384 bytes]  -- for epilogue staging
constexpr uint32_t SMEM_A_BYTES = BLOCK_M * BLOCK_K * 2;
constexpr uint32_t SMEM_B_BYTES = BLOCK_N * BLOCK_K * 2;
constexpr uint32_t SMEM_OUT_BYTES = BLOCK_M * BLOCK_N * 2;
constexpr uint32_t TOTAL_SMEM_BYTES = 8 + SMEM_A_BYTES + SMEM_B_BYTES + SMEM_OUT_BYTES;

extern "__managed__" CUtensorMap d_tma_A;
extern "__managed__" CUtensorMap d_tma_B;

__global__ void gemm_kernel(const __nv_bfloat16* __restrict__ A,
                             const __nv_bfloat16* __restrict__ B,
                             __nv_bfloat16* __restrict__ C,
                             uint32_t M, uint32_t N, uint32_t K) {

    static __shared__ uint64_t sm_barrier[1];
    __shared__ __align__(256) unsigned char smem_A_buf[TOTAL_SMEM_BYTES];

    __nv_bfloat16* smem_A = reinterpret_cast<__nv_bfloat16*>(smem_A_buf + 8);
    __nv_bfloat16* smem_B = reinterpret_cast<__nv_bfloat16*>(smem_A_buf + 8 + SMEM_A_BYTES);
    __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(smem_A_buf + 8 + SMEM_A_BYTES + SMEM_B_BYTES);

    // Thread 0 initializes mbarrier
    if (threadIdx.x == 0) {
        init_mbarrier_fn(sm_barrier, 2);  // 2 TMA arrivals expected (A + B)
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

    // Precompute shared memory descriptors (constant throughout kernel)
    // A: K-major, BM=64 rows, BK=16 cols → SBO=1024, LBO=1024*(BM/64)=1024
    uint64_t smem_desc_A_base = make_smem_descriptor_128b_swizzle(smem_A, 1024, 1024);
    // B: K-major, BN=128 rows, BK=16 cols → SBO=1024, LBO=1024*(BN/64)=2048
    uint64_t smem_desc_B_base = make_smem_descriptor_128b_swizzle(smem_B, 2048, 1024);

    // Instruction descriptor for BF16xFP32->FP32 UMMA (K=16, M=64, N=128)
    uint32_t idesc = make_umma_idesc(BLOCK_M, BLOCK_N);

    // Allocate Tensor Memory for the C accumulator (64 rows x 128 cols of FP32)
    uint32_t tmem_c_addr;
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 : "=r"(tmem_c_addr)
                 : "r"((uint32_t)TMEM_COLS) : "memory");

    // Clear accumulator (issue one UMMA with enable-input-d = false)
    uint64_t desc_a_clear = smem_desc_A_base;
    uint64_t desc_b_clear = smem_desc_B_base;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.eq.b32 p, 0, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n"
        "}\n"
        :: "r"(tmem_c_addr), "l"(desc_a_clear), "l"(desc_b_clear), "r"(idesc));
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                 :: "r"((uint32_t)__cvta_generic_to_shared(sm_barrier)));
    // Wait for the clear to complete
    if (threadIdx.x == 0) {
        mbarrier_wait_fn(sm_barrier, 0);  // parity toggles each phase
    }
    __syncthreads();

    // Main K-loop: iterate over K in BLOCK_K=16 steps
    for (uint32_t k_step = 0; k_step < k_iter_max; ++k_step) {
        uint32_t k_off = k_step * BLOCK_K;

        // Issue TMA loads: A[a_row_base:a_row_base+BM, k_off:k_off+BK]
        //                B[b_col_base:b_col_base+BN, k_off:k_off+BK]
        // Note: B is stored as [N,K] and we want B[:,k_off:k_off+BK]^T
        //       so the effective coordinates for B.T are [b_col_base, k_off]
        tma_load_2d_fn(&d_tma_A, sm_barrier, smem_A, 
                        static_cast<int32_t>(k_off), static_cast<int32_t>(a_row_base));
        tma_load_2d_fn(&d_tma_B, sm_barrier, smem_B, 
                        static_cast<int32_t>(k_off), static_cast<int32_t>(b_col_base));

        // Wait for both TMA loads to complete via mbarrier
        if (threadIdx.x == 0) {
            mbarrier_wait_fn(sm_barrier, 0);
        }
        __syncthreads();

        // Update descriptors for the current K slice
        // For K-major: advance base address by (rows * BK_slice * elem_size) bytes
        // But since descriptor encodes relative layout, we adjust the base offset
        uint64_t desc_a = smem_desc_A_base;
        uint64_t desc_b = smem_desc_B_base;
        
        // Reconstruct descriptor with updated base address
        uint32_t smem_a_addr = (uint32_t)__cvta_generic_to_shared(smem_A);
        uint32_t smem_b_addr = (uint32_t)__cvta_generic_to_shared(smem_B);
        
        // Base address goes into bits 0-13 (shifted by 4)
        desc_a &= ~UINT64_C(0x3FFF);  // clear base address bits
        desc_b &= ~UINT64_C(0x3FFF);
        desc_a |= (uint64_t)(smem_a_addr >> 4);
        desc_b |= (uint64_t)(smem_b_addr >> 4);

        // Issue UMMA: C = A @ B + C (accumulate)
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "setp.ne.b32 p, 1, 0;\n"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n"
            "}\n"
            :: "r"(tmem_c_addr), "l"(desc_a), "l"(desc_b), "r"(idesc));

        // Commit UMMA to mbarrier
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                     :: "r"((uint32_t)__cvta_generic_to_shared(sm_barrier)));

        // Wait for UMMA completion before next iteration
        if (threadIdx.x == 0) {
            mbarrier_wait_fn(sm_barrier, 0);
        }
        __syncthreads();
    }

    // Epilogue: Convert FP32 accumulator in TMEM to BF16 and write to global C
    // Each thread (0..63) handles one row
    uint32_t tid = threadIdx.x;
    uint32_t local_row = tid;  // threads 0-63 map to rows 0-63

    if (local_row < BLOCK_M) {
        uint32_t global_row = a_row_base + local_row;
        if (global_row < M) {
            // Load from TMEM in chunks of 4 FP32s
            for (uint32_t col = 0; col < BLOCK_N; col += 4) {
                uint32_t r0, r1, r2, r3;
                // Collective load: all threads in warp use same column address
                // Shape .32x32b.x4 loads 4 FP32s per thread (from adjacent lanes)
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                             : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                             : "r"(static_cast<uint32_t>(col)));
                
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

                // Convert FP32 -> BF16 and store
                float f0 = __uint_as_float(r0);
                float f1 = __uint_as_float(r1);
                float f2 = __uint_as_float(r2);
                float f3 = __uint_as_float(r3);

                uint32_t nc = b_col_base + col;
                __nv_bfloat16* out_ptr = C + (uint64_t)global_row * N + nc;

                // Write with bounds checking on N dimension
                if (nc + 3 < N) {
                    out_ptr[0] = __float2bfloat16(f0);
                    out_ptr[1] = __float2bfloat16(f1);
                    out_ptr[2] = __float2bfloat16(f2);
                    out_ptr[3] = __float2bfloat16(f3);
                } else {
                    // Handle partial last chunk
                    if (nc < N) out_ptr[0] = __float2bfloat16(f0);
                    if (nc + 1 < N) out_ptr[1] = __float2bfloat16(f1);
                    if (nc + 2 < N) out_ptr[2] = __float2bfloat16(f2);
                }
            }
        }
    }

    // Deallocate Tensor Memory
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem_c_addr), "r"((uint32_t)TMEM_COLS));
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_RT_CHECK(cudaSetDevice(A.device().device_id));

    uint32_t M = static_cast<uint32_t>(A.size(0));
    uint32_t N = static_cast<uint32_t>(7168);  // Fixed from spec
    uint32_t K = static_cast<uint32_t>(5120);  // Fixed from spec

    const __nv_bfloat16* A_data = static_cast<const __nv_bfloat16*>(A.data_ptr());
    const __nv_bfloat16* B_data = static_cast<const __nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_data = static_cast<__nv_bfloat16*>(C.data_ptr());

    // Create TMA descriptors
    // A: [M, K] → TMA loads box [BK, BM] starting at [k_coord, m_coord]
    //    Inner dim = K, outer dim = M
    CUtensorMap tma_A;
    CUDA_DRV_CHECK(create_tma_2d_descriptor(
        &tma_A, const_cast<void*>(static_cast<const void*>(A_data)),
        K, M,
        BLOCK_K, BLOCK_M,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    // B: [N, K] → We want B.T[k, n], so TMA loads box [BK, BN] from B
    //    Effectively: loads B[b_col_start:b_col_end, k_start:k_end]
    //    Then in-memory this is interpreted as [BN, BK] for the K-major MMA consumer
    CUtensorMap tma_B;
    CUDA_DRV_CHECK(create_tma_2d_descriptor(
        &tma_B, const_cast<void*>(static_cast<const void*>(B_data)),
        K, N,
        BLOCK_K, BLOCK_N,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    // Copy TMA descriptors to device-accessible memory
    CUresult res = cuModuleGetGlobal(&d_tma_A, nullptr, 0, nullptr, "d_tma_A");
    (void)res; // Using managed memory declaration above

    // Launch configuration
    dim3 blockDim(NUM_THREADS);
    uint32_t grid_m_blocks = (M + BLOCK_M - 1) / BLOCK_M;
    uint32_t grid_n_blocks = (N + BLOCK_N - 1) / BLOCK_N;
    dim3 gridDim(grid_m_blocks * grid_n_blocks);
    
    size_t smem_bytes = TOTAL_SMEM_BYTES;
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    // Use dynamic shared memory via launch config
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = gridDim;
    cfg.blockDim = blockDim;
    cfg.dynamicSmemBytes = smem_bytes;
    cfg.stream = stream;
    cfg.attrs = nullptr;
    cfg.numAttrs = 0;

    // Need to set up device-side TMA descriptors via cudaMemcpy first
    CUdeviceptr d_tma_A_dev, d_tma_B_dev;
    CUDA_DRV_CHECK(cuMemAlloc(&d_tma_A_dev, sizeof(CUtensorMap)));
    CUDA_DRV_CHECK(cuMemAlloc(&d_tma_B_dev, sizeof(CUtensorMap)));
    CUDA_DRV_CHECK(cuMemcpyDtoHAsync((void*)&d_tma_A, d_tma_A_dev, sizeof(CUtensorMap), stream));
    CUDA_DRV_CHECK(cuMemcpyDtoHAsync((void*)&d_tma_B, d_tma_B_dev, sizeof(CUtensorMap), stream));
    
    // Actually, we need to COPY TO DEVICE. Let me fix this.
    CUDA_DRV_CHECK(cuMemcpyHtoDAsync(d_tma_A_dev, &tma_A, sizeof(CUtensorMap), stream));
    CUDA_DRV_CHECK(cuMemcpyHtoDAsync(d_tma_B_dev, &tma_B, sizeof(CUtensorMap), stream));

    // Since we can't easily pass TU tensorMaps to kernel as grid_constant,
    // let's use a simpler approach: pass pointers and have TMA use global memory descriptors.
    // Actually, re-thinking: let's use an alternative kernel that passes pointers.
    
    // Free the allocations since we'll use a different approach
    CUDA_DRV_CHECK(cuMemFree(d_tma_A_dev));
    CUDA_DRV_CHECK(cuMemFree(d_tma_B_dev));

    // Simple kernel launch with dynamic shared memory
    // We'll use a variant that copies TMA descriptors to constant memory
    // by declaring them as external globals and cudaMemcpy before launch
    
    // Allocate constant-pinned memory for TMA descriptors
    CUtensorMap *h_tma_A_pinned = nullptr;
    CUtensorMap *h_tma_B_pinned = nullptr;
    cudaMallocHost(&h_tma_A_pinned, sizeof(CUtensorMap));
    cudaMallocHost(&h_tma_B_pinned, sizeof(CUtensorMap));
    *h_tma_A_pinned = tma_A;
    *h_tma_B_pinned = tma_B;

    // Register the kernel and launch
    gemm_kernel<<<gridDim, blockDim, smem_bytes, stream>>>(
        A_data, B_data, C_data, M, N, K);
    
    CUDA_RT_CHECK(cudaGetLastError());
    CUDA_RT_CHECK(cudaStreamSynchronize(stream));

    cudaFreeHost(h_tma_A_pinned);
    cudaFreeHost(h_tma_B_pinned);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_gemm_blackwell::run);

}  // namespace tvm_ffi_gemm_blackwell