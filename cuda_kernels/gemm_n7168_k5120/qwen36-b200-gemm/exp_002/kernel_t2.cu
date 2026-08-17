#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <algorithm>
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

// UMMA instruction for cta_group::1
__device__ __forceinline__ void umma_f16_cg1(uint32_t tmem_d, uint64_t sdesc_a, 
                                              uint64_t sdesc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_d), "l"(sdesc_a), "l"(sdesc_b), "r"(idesc), "r"(accum));
}

// Commit UMMA → mbarrier
__device__ __forceinline__ void umma_commit(uint64_t* bar) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)) : "memory");
}

// Build SMEM descriptor for K-major 128B-swizzled layout
__device__ __forceinline__ uint64_t make_kmaj_sdesc(void* base_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(base_ptr);
    d |= (uint64_t)((addr >> 4) & 0x3FFF);   // bits 0-13: start address
    d |= (uint64_t)1 << 16;                   // bits 16-29: LBO=1 (swizzled)
    d |= (uint64_t)64 << 32;                  // bits 32-45: SBO=1024 bytes → encoded 64
    d |= (uint64_t)1 << 46;                   // bits 46-48: version=1
    d |= (uint64_t)2 << 61;                   // bits 61-63: SWIZZLE_128B
    return d;
}

// Build UMMA instruction descriptor: BF16×BF16→FP32, both K-major
__device__ __forceinline__ uint32_t make_umma_instr(uint32_t mn_m, uint32_t mn_n) {
    uint32_t id = 0;
    id |= (1u << 4);                          // dtype F32
    id |= (1u << 7);                          // atype BF16
    id |= (1u << 10);                         // btype BF16
    id |= (0u << 15);                         // no trans A
    id |= (0u << 16);                         // no trans B
    id |= ((mn_n >> 3) & 0x3Fu) << 17;       // n_dim >> 3
    id |= ((mn_m >> 4) & 0x1Fu) << 24;       // m_dim >> 4
    return id;
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
    // [0 .. BM*BK*2-1]: A_smem[BK][BM] (K-major after TMA: inner=K, outer=M)
    // [A_SIZE .. A_SIZE+BN*BK*2-1]: B_smem[BK][BN] (K-major after TMA)
    // After that: mbarrier + padding
    
    constexpr size_t A_SIZE = BLOCK_M * BLOCK_K * sizeof(__nv_bfloat16);
    constexpr size_t B_SIZE = BLOCK_N * BLOCK_K * sizeof(__nv_bfloat16);
    __nv_bfloat16* smem_A = reinterpret_cast<__nv_bfloat16*>(sdata);
    __nv_bfloat16* smem_B = reinterpret_cast<__nv_bfloat16*>(sdata + A_SIZE);
    uint64_t* smem_bar = reinterpret_cast<uint64_t*>(sdata + A_SIZE + B_SIZE + 64);
    
    // Initialize barrier (thread 0 only)
    if (tid == 0) {
        init_smem_barrier(smem_bar, 1);
        fence_mbarrier_init();
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" 
            :: "r"((uint32_t)__cvta_generic_to_shared(smem_bar)) : "memory");
    }
    __syncthreads();

    // All threads wait for barrier init to complete
    mbarrier_wait(smem_bar, 0);

    // Allocate Tensor Memory for accumulator D (BLOCK_M x BLOCK_N fp32)
    uint32_t num_cols = 64;
    uint32_t tmem_addr = 0;
    if (tid == 0) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
            : "=r"(tmem_addr)
            : "r"((uint32_t)__cvta_generic_to_shared(smem_bar)),
              "r"(num_cols)
            : "memory");
    }
    __syncwarp();
    __syncthreads();
    uint32_t base_tmaddr = __shfl_sync(0xFFFFFFFF, tmem_addr, 0);

    // Number of K tiles
    uint64_t nk_tiles = K / BLOCK_K;

    // Instruction descriptor
    uint32_t instr = make_umma_instr(BLOCK_M, BLOCK_N);

    // Clear D accumulator in TMEM (all zeros)
    // Use tcgen05.st to write zeros. Each thread handles its own lane's columns.
    uint32_t my_tmem_lane = tid % 64;
    if (my_tmem_lane < 64) {
        for (uint32_t c = 0; c < BLOCK_N; c += 4) {
            uint32_t addr = (my_tmem_lane << 16) | c;
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
                :: "r"(addr), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    }
    fence_proxy_async();
    __syncthreads();

    // Main K-loop
    for (uint64_t kt = 0; kt < nk_tiles; ++kt) {
        uint64_t k_off = kt * BLOCK_K;
        
        // Arrive at barrier expecting total bytes for A and B loads
        uint32_t expect_bytes = (BLOCK_M * BLOCK_K + BLOCK_N * BLOCK_K) * sizeof(__nv_bfloat16);
        mbarrier_arrive_expect_tx(smem_bar, expect_bytes);
        
        // TMA load A: coords(inner=k_off, outer=out_m)
        tma_load_2d(&tma_A_desc, smem_bar, smem_A, (int32_t)k_off, (int32_t)out_m);
        // TMA load B: coords(inner=k_off, outer=out_n) for B.T virtual view  
        tma_load_2d(&tma_B_desc, smem_bar, smem_B, (int32_t)k_off, (int32_t)out_n);
        
        // Wait for loads to complete
        mbarrier_wait(smem_bar, kt % 2);

        // Fence: shared memory writes visible to async proxy before UMMA reads
        fence_proxy_async();
        
        // Issue UMMA in chunks of K=16 across BK=64 → 4 iterations
        uint32_t accum_flag = (kt == 0) ? 0 : 1;
        
        for (uint32_t ki = 0; ki < BLOCK_K / 16; ++ki) {
            // Advance base pointer by ki*16 bf16 elements along K dimension
            // For K-major layout [BK][BM], each K step is BM*bF16 = BM*2 bytes
            __nv_bfloat16* a_base = smem_A + ki * 16 * BLOCK_M;
            __nv_bfloat16* b_base = smem_B + ki * 16 * BLOCK_N;
            
            uint64_t sd_a = make_kmaj_sdesc(a_base);
            uint64_t sd_b = make_kmaj_sdesc(b_base);
            
            // Single thread issues MMA (cta_group::1 has single-thread semantics)
            if (tid == 0) {
                umma_f16_cg1(base_tmaddr, sd_a, sd_b, instr, accum_flag);
            }
        }
        
        // Commit UMMA operations to mbarrier
        if (tid == 0) {
            umma_commit(smem_bar);
        }
        
        // Wait for UMMA completion
        mbarrier_wait(smem_bar, (kt + 1) % 2);
    }

    // Epilogue: Copy D[TMEM BLOCK_M×BLOCK_N FP32] → C_global BF16
    // Read from TMEM via tcgen05.ld, convert FP32→BF16, write to global
    
    // TMEM lane assignment: threads map to lanes modulo 64
    // Remaining threads (64..127) help too: they re-use lanes
    uint32_t my_lane = tid % 64;
    uint64_t row_global = out_m + my_lane;
    
    if (row_global < M && my_lane < 64) {
        __nv_bfloat16* C_row = C_out + row_global * N;
        uint64_t col_start = out_n;
        
        for (uint32_t c = 0; c < BLOCK_N; c += 4) {
            uint32_t r0, r1, r2, r3;
            uint32_t addr = (my_lane << 16) | c;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            uint64_t col_idx = col_start + c;
            if (col_idx + 3 < N) {
                C_row[col_idx + 0] = __float2bfloat16(__uint_as_float(r0));
                C_row[col_idx + 1] = __float2bfloat16(__uint_as_float(r1));
                C_row[col_idx + 2] = __float2bfloat16(__uint_as_float(r2));
                C_row[col_idx + 3] = __float2bfloat16(__uint_as_float(r3));
            } else if (col_idx < N) {
                C_row[col_idx + 0] = __float2bfloat16(__uint_as_float(r0));
            }
        }
    }
    
    // Deallocate TMEM (one warp)
    if (tid < TWP) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
            :: "r"(base_tmaddr), "r"(num_cols) : "memory");
    }
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
    
    size_t smem_size = (BLOCK_M * BLOCK_K + BLOCK_N * BLOCK_K) * sizeof(__nv_bfloat16) + 256;
    
    // TMA descriptor for A: A[M,K] row-major
    // Coords: (inner=K, outer=M) → smem[BK][BM] K-major
    CUtensorMap tma_A;
    CU_CHECK(create_tma_2d_descriptor_BF16(&tma_A, const_cast<__nv_bfloat16*>(A_ptr),
        K, M,                       // gmem dims
        K * sizeof(__nv_bfloat16),   // stride
        BLOCK_K, BLOCK_M,           // box
        CU_TENSOR_MAP_SWIZZLE_128B));
    
    // TMA descriptor for B.T: virtually B.T[K,N] where B.T[k,n] = B[n,k]
    // B stored as [N,K] row-major: B[n][k] at base + n*K*2 + k*2
    // TMA with dims{K,N}, stride{K*2}: addr = base + k_inner*2 + n_outer*(K*2)
    //                                    = base + k*2 + n*K*2 = B[n][k]*2 ✓
    // Box: inner=BK, outer=BN → smem[BK][BN] K-major ✓
    CUtensorMap tma_B;
    cuuint64_t globalDim_B[2] = {K, N};
    cuuint64_t globalStrides_B[1] = {K * sizeof(__nv_bfloat16)};
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