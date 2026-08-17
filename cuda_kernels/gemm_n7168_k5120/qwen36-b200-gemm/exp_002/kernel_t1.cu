#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <cstdio>
#include <cassert>
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
    CUresult _r = (call);                                          \
    if (_r != CUDA_SUCCESS) {                                      \
        const char *err_str;                                       \
        cuGetErrorString(_r, &err_str);                            \
        fprintf(stderr, "cu error %s at %s:%d\n",                 \
                err_str, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_gemm_blackwell {

// ---- Device helper functions ----

__device__ __forceinline__ void init_smem_barrier(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}

__device__ __forceinline__ void fence_mbarrier_init() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait(uint64_t* bar, uint32_t parity) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "1:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra 1;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(parity));
}

__device__ __forceinline__ void fence_proxy_async() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

// TMA load: cp.async.bulk.tensor.2d to shared::cta
__device__ __forceinline__ void tma_load_2d(const CUtensorMap* desc, uint64_t* barrier,
                                            void* smem_dst, int32_t coord0, int32_t coord1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem_dst)),
           "l"((uint64_t)desc),
           "r"((uint32_t)__cvta_generic_to_shared(barrier)),
           "r"(coord0), "r"(coord1) : "memory");
}

// Host-side TMA descriptor creation
CUresult create_tma_2d_descriptor_BF16(CUtensorMap* d, void* globalAddress,
    uint64_t gmem_dim0, uint64_t gmem_dim1, uint64_t gmem_stride_bytes,
    uint32_t box_dim0, uint32_t box_dim1,
    CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[2] = {gmem_dim0, gmem_dim1};
    cuuint64_t globalStrides[1] = {gmem_stride_bytes};
    cuuint32_t boxDim[2] = {box_dim0, box_dim1};
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
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 64;
constexpr int WARPS = 4;
constexpr int TWP = 32;
constexpr int BT = WARPS * TWP; // 128

extern __shared__ char sdata[];

__global__ void gemm_sm100_kernel(
    const __grid_constant__ CUtensorMap tma_A_desc,
    const __grid_constant__ CUtensorMap tma_B_desc,
    __nv_bfloat16* __restrict__ C_out,
    uint64_t M, uint64_t N, uint64_t K) {

    const uint32_t tid = threadIdx.x;
    const uint32_t warp_id = tid / TWP;
    const uint32_t lane = tid % TWP;

    const uint64_t out_m = blockIdx.y * BLOCK_M;
    const uint64_t out_n = blockIdx.x * BLOCK_N;

    // Shared memory layout:
    // [0 .. BM*BK*2-1]          : A_smem[BK][BM]     (K-major after TMA: inner=K, outer=M)
    // [BM*BK*2 .. BM*BK*2+BN*BK*2-1] : B_smem[BK][BN] (K-major after TMA: inner=K, outer=N)
    // After that: mbarrier + pad
    
    constexpr size_t A_SIZE = BLOCK_M * BLOCK_K * sizeof(__nv_bfloat16);
    __nv_bfloat16* smem_A = reinterpret_cast<__nv_bfloat16*>(sdata);
    __nv_bfloat16* smem_B = reinterpret_cast<__nv_bfloat16*>(sdata + A_SIZE);
    uint64_t* smem_bar = reinterpret_cast<uint64_t*>(sdata + A_SIZE + BLOCK_N * BLOCK_K * sizeof(__nv_bfloat16));
    
    // Initialize barrier (thread 0 only)
    if (tid == 0) {
        init_smem_barrier(smem_bar, 1);
        fence_mbarrier_init();
        // Signal initial arrival so other threads can wait
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" 
            :: "r"((uint32_t)__cvta_generic_to_shared(smem_bar)) : "memory");
    }
    __syncthreads();

    // All threads wait for barrier init to complete
    mbarrier_wait(smem_bar, 0);

    // Allocate Tensor Memory for accumulator D (BLOCK_M x BLOCK_N fp32)
    // Allocate from one warp only (warp 0)
    // BM*BN*4 bytes = 64*64*4 = 16KB → ceil(log2(16KB/4)) cols = ... 
    // TMEM is addressed as lanes × cols. Each cell is 32-bit.
    // 64 lanes × 64 cols = 4096 cells × 4 bytes = 16KB ✓
    // Alloc unit: 32 columns, power-of-2 multiples
    uint32_t num_cols = 64; // 64 columns × 128 lanes = enough
    // Actually, alloc happens from ONE warp of the warpgroup
    // We'll use tcgen05.alloc from thread 0
    
    uint32_t tmem_addr;
    if (tid == 0 && warp_id == 0) {
        // Use a portion of shared mem as placeholder (required by instruction form)
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
            : "=r"(tmem_addr)
            : "r"((uint32_t)__cvta_generic_to_shared(sdata + A_SIZE + BLOCK_N * BLOCK_K * sizeof(__nv_bfloat16) + 64)),
              "r"(num_cols)
            : "memory");
    }
    __syncwarp();
    __syncthreads();

    // Broadcast tmem_addr
    uint32_t base_tmaddr = __shfl_sync(0xFFFFFFFF, tmem_addr, 0);

    // Number of K tiles
    uint64_t nk_tiles = K / BLOCK_K;

    // Instruction descriptor for UMMA: BF16×BF16→FP32, K-major both
    // idesc bits: dtype=F32(1), atype=BF16(1), btype=BF16(1), 
    //             transA=0, transB=0, n_dim>>3=N/8, m_dim>>4=M/16
    uint32_t make_instr(uint32_t mn_m, uint32_t mn_n) {
        uint32_t id = 0;
        id |= (1u << 4);      // dtype F32
        id |= (1u << 7);      // atype BF16
        id |= (1u << 10);     // btype BF16
        id |= (0u << 15);     // no trans A
        id |= (0u << 16);     // no trans B
        id |= ((mn_n >> 3) & 0x3Fu) << 17;   // n_dim >> 3
        id |= ((mn_m >> 4) & 0x1Fu) << 24;   // m_dim >> 4
        return id;
    }

    // Build SMEM descriptor for K-major 128B-swizzled layout
    uint64_t make_kmaj_sdesc(void* base_ptr, uint32_t m_dim, uint32_t k_stride_bytes) {
        uint64_t d = 0;
        uint32_t addr = (uint32_t)__cvta_generic_to_shared(base_ptr);
        // bits 0-13: encoded start address
        d |= (uint64_t)((addr >> 4) & 0x3FFF);
        // bits 16-29: LBO (not used for swizzle, assume 1)
        d |= (uint64_t)1 << 16;
        // bits 32-45: SBO = 8 * 128 = 1024 bytes → encoded = 1024 >> 4 = 64
        d |= (uint64_t)64 << 32;
        // bits 46-48: version = 1
        d |= (uint64_t)1 << 46;
        // bits 49-51: base_offset = 0
        // bit 52: LDM = 0 (byte offset relative)
        // bits 61-63: swizzle = 2 (128B)
        d |= (uint64_t)2 << 61;
        return d;
    }

    // Main K-loop with pipelined TMA loads + UMMA compute
    // Pipeline depth 2: load tile k+1 while computing on tile k
    int pipe_idx = 0;
    
    // Load first two K tiles asynchronously
    for (uint64_t kt = 0; kt < std::min((uint64_t)2, nk_tiles); ++kt) {
        uint64_t k_off = kt * BLOCK_K;
        
        // Arrive at barrier expecting total bytes for both A and B loads
        uint32_t expect_bytes = 2 * (BLOCK_M * BLOCK_K + BLOCK_N * BLOCK_K) * sizeof(__nv_bfloat16);
        mbarrier_arrive_expect_tx(smem_bar, expect_bytes);
        
        // TMA load A: coords(inner=k_off, outer=out_m)
        tma_load_2d(&tma_A_desc, smem_bar, smem_A, (int32_t)k_off, (int32_t)out_m);
        // TMA load B: coords(inner=k_off, outer=out_n) for B.T virtual view
        tma_load_2d(&tma_B_desc, smem_bar, smem_B, (int32_t)k_off, (int32_t)out_n);
        
        // Wait for completion of this phase
        mbarrier_wait(smem_bar, kt % 2);
    }

    // Now pipeline: iterate over remaining K tiles
    for (uint64_t kt = 0; kt < nk_tiles; ++kt) {
        uint64_t k_off = kt * BLOCK_K;
        
        // Fence: shared memory visible to async proxy before UMMA reads it
        fence_proxy_async();
        
        // Issue UMMA instructions in chunks of K=16
        // Accumulate flag: first K-tile → clear (0), else accumulate (1)
        uint32_t accum_flag = (kt == 0) ? 0 : 1;
        uint32_t instr = make_instr(BLOCK_M, BLOCK_N);
        
        // Adjusted SMEM descriptors for current K subtile
        // For K-major layout, advancing K index by ki*16 shifts the base address
        // by ki*16 * sizeof(bf16) = ki*32 bytes
        // Since we allocated full BK=BK, we walk through 4 sub-batches (64/16=4)
        for (uint32_t ki = 0; ki < BLOCK_K / 16; ++ki) {
            size_t byte_shift = ki * 16 * sizeof(__nv_bfloat16);
            __nv_bfloat16* a_base = smem_A + ki * 16 * BLOCK_M;
            __nv_bfloat16* b_base = smem_B + ki * 16 * BLOCK_N;
            
            uint64_t sd_a = make_kmaj_sdesc(a_base, BLOCK_M, BLOCK_K * sizeof(__nv_bfloat16));
            uint64_t sd_b = make_kmaj_sdesc(b_base, BLOCK_N, BLOCK_K * sizeof(__nv_bfloat16));
            
            // D output lane: for 64×64 result, start at appropriate lane
            // D occupies lanes 0..63 in TMEM for rows
            // Column stride matters: BN=64 columns needed
            uint32_t d_addr = base_tmaddr; // Start of our allocation
            
            // Single thread issues the MMA (cta_group::1, single-thread semantics)
            if (tid == 0) {
                umma_f16_cg1(d_addr, sd_a, sd_b, instr, accum_flag);
            }
        }
        
        // Commit UMMA ops → mbarrier
        if (tid == 0) {
            umma_commit(smem_bar);
        }
        
        // Load next K tile while committing (if not last)
        uint64_t kt_next = kt + 2;
        if (kt_next < nk_tiles) {
            uint64_t k_off_next = kt_next * BLOCK_K;
            uint32_t expect_bytes = 2 * (BLOCK_M * BLOCK_K + BLOCK_N * BLOCK_K) * sizeof(__nv_bfloat16);
            mbarrier_arrive_expect_tx(smem_bar, expect_bytes);
            tma_load_2d(&tma_A_desc, smem_bar, smem_A, (int32_t)k_off_next, (int32_t)out_m);
            tma_load_2d(&tma_B_desc, smem_bar, smem_B, (int32_t)k_off_next, (int32_t)out_n);
        }
        
        // Wait for UMMA to finish
        mbarrier_wait(smem_bar, (kt + 1) % 2);
    }

    // Final proxy fence after all UMMA completes
    fence_proxy_async();

    // Epilogue: Copy D[TMEM BLOCK_M×BLOCK_N FP32] → C_global BF16
    // Read from TMEM collectively via tcgen05.ld, convert, write to global
    
    // TMEM layout: 128 lanes × N_cols
    // Our allocation gave us contiguous columns starting at base_tmaddr
    // Result: D[lane][col] where lane=0..63 covers M-dimension, col=0..63 covers N-dimension
    
    uint32_t my_lane = tid % 64;
    if (my_lane < 64 && out_m + my_lane < M) {
        uint32_t row_global = out_m + my_lane;
        float* C_row = (float*)(C_out + (uint64_t)row_global * N);
        
        for (uint32_t c = 0; c < BLOCK_N; c += 4) {
            uint32_t r0, r1, r2, r3;
            // Load 4 FP32 values from TMEM lane=my_lane, columns=c..c+3
            uint32_t addr = (my_lane << 16) | c;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            uint32_t col_start = out_n + c;
            if (col_start + 3 < N) {
                __nv_bfloat16 bf0 = __float2bfloat16(__uint_as_float(r0));
                __nv_bfloat16 bf1 = __float2bfloat16(__uint_as_float(r1));
                __nv_bfloat16 bf2 = __float2bfloat16(__uint_as_float(r2));
                __nv_bfloat16 bf3 = __float2bfloat16(__uint_as_float(r3));
                C_row[col_start + 0] = bf0;
                C_row[col_start + 1] = bf1;
                C_row[col_start + 2] = bf2;
                C_row[col_start + 3] = bf3;
            } else if (col_start < N) {
                __nv_bfloat16 bf0 = __float2bfloat16(__uint_as_float(r0));
                C_row[col_start + 0] = bf0;
            }
        }
    }
    
    // Deallocate TMEM from same warp
    if (tid < TWP) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
            :: "r"(base_tmaddr), "r"(num_cols) : "memory");
    }
}

// Minimal ASM wrappers used inside kernel
__device__ __forceinline__ void umma_f16_cg1(uint32_t tmem_d, uint64_t sdesc_a, 
                                              uint64_t sdesc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_d), "l"(sdesc_a), "l"(sdesc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit(uint64_t* bar) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)) : "memory");
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    uint64_t M = A.size(0);
    uint64_t N = 7168;
    uint64_t K = 5120;
    
    assert((size_t)B.size(0) == N);
    assert((size_t)B.size(1) == K);
    assert((size_t)C.size(0) == M);
    assert((size_t)C.size(1) == N);
    
    const __nv_bfloat16* A_ptr = static_cast<const __nv_bfloat16*>(A.data_ptr());
    const __nv_bfloat16* B_ptr = static_cast<const __nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());
    
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M, 1);
    dim3 block(BT);
    
    // Shared memory: A_smem + B_smem + barrier + padding
    size_t smem_size = (BLOCK_M * BLOCK_K + BLOCK_N * BLOCK_K) * sizeof(__nv_bfloat16) + 256;
    
    // TMA descriptor for A: A[M,K] row-major
    // TMA coords: (inner=K, outer=M), box=(BK, BM)
    // Output in smem: [BK][BM] K-major
    CUtensorMap tma_A;
    CU_CHECK(create_tma_2d_descriptor_BF16(&tma_A, const_cast<__nv_bfloat16*>(A_ptr),
        K, M,                       // gmem dims: inner=K, outer=M
        K * sizeof(__nv_bfloat16),   // stride between rows of outer dim (= M rows, stride K*2)
        BLOCK_K, BLOCK_M,           // box: BK elements in inner(K), BM in outer(M)
        CU_TENSOR_MAP_SWIZZLE_128B));
    
    // TMA descriptor for B: B[N,K] row-major, virtually B.T[K,N]
    // B.T[k,n] = B[n,k], stored at base + n*K*2 + k*2
    // TMA coords: (inner=K, outer=N), box=(BK, BN)
    // For default TMA stride calculation with dims {K,N}: stride would be N*2
    // But we need stride K*2 to get correct mapping
    CUtensorMap tma_B;
    cuuint64_t globalDim_B[2] = {K, N};
    cuuint64_t globalStrides_B[1] = {K * sizeof(__nv_bfloat16)};  // Custom stride for B.T view
    cuuint32_t boxDim_B[2] = {BLOCK_K, BLOCK_N};
    cuuint32_t elementStrides_B[2] = {1, 1};
    CU_CHECK(cuTensorMapEncodeTiled(&tma_B, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        const_cast<__nv_bfloat16*>(B_ptr), globalDim_B, globalStrides_B,
        boxDim_B, elementStrides_B,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    gemm_sm100_kernel<<<grid, block, smem_size, stream>>>(
        tma_A, tma_B, C_ptr, M, N, K);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_gemm_blackwell::run);

}  // namespace tvm_ffi_gemm_blackwell