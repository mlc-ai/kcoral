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

__device__ __forceinline__ void mbarrier_arrive(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_parity(uint64_t* bar, uint32_t parity) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_P_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_P_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(parity));
}

__device__ __forceinline__ void fence_proxy_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void fence_proxy_async() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

// TMA 2D load into shared::cta, tracked by mbarrier tx-count
__device__ __forceinline__ void tma_load_2d(const CUtensorMap* desc, uint64_t* barrier,
                                            void* smem_dst, int32_t coord0, int32_t coord1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem_dst)),
           "l"((uint64_t)desc),
           "r"((uint32_t)__cvta_generic_to_shared(barrier)),
           "r"(coord0), "r"(coord1) : "memory");
}

// UMMA cta_group::1: BF16 x BF16 -> FP32 in TMEM
__device__ __forceinline__ void umma_f16_cg1(uint32_t tmem_d, uint64_t sdesc_a, 
                                              uint64_t sdesc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_d), "l"(sdesc_a), "l"(sdesc_b), "r"(idesc), "r"(accum));
}

// Wait for tcgen05 async ops to complete
__device__ __forceinline__ void umma_sync_all() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

// Build K-major shared-mem descriptor for tcgen05 UMMA (128B swizzle)
__device__ __forceinline__ uint64_t make_kmaj_sdesc(void* base_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(base_ptr);
    d |= (uint64_t)((addr >> 4) & 0x3FFF);   // bits 0-13: start address encoded
    d |= (uint64_t)1 << 16;                   // bits 16-29: LBO=1 (swizzled)
    d |= (uint64_t)64 << 32;                  // bits 32-45: SBO=1024>>4=64
    d |= (uint64_t)1 << 46;                   // bits 46-48: version=1
    d |= (uint64_t)2 << 61;                   // bits 61-63: SWIZZLE_128B
    return d;
}

// Instruction descriptor: BF16×BF16→FP32, both K-major (no transpose)
__device__ __forceinline__ uint32_t make_umma_instr(uint32_t mn_m, uint32_t mn_n) {
    uint32_t id = 0;
    id |= (1u << 4);                          // dtype F32
    id |= (1u << 7);                          // atype BF16
    id |= (1u << 10);                         // btype BF16
    id |= (0u << 15);                         // A: no transpose (K-major)
    id |= (0u << 16);                         // B: no transpose (K-major)
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
constexpr int BT = 128;  // 4 warps = 1 warpgroup

extern __shared__ char sdata[];

__global__ void gemm_sm100_kernel(
    const __grid_constant__ CUtensorMap tma_A_desc,
    const __grid_constant__ CUtensorMap tma_B_desc,
    __nv_bfloat16* __restrict__ C_out,
    uint64_t M, uint64_t N, uint64_t K) {

    const uint32_t tid = threadIdx.x;

    const uint64_t out_m = blockIdx.y * BLOCK_M;
    const uint64_t out_n = blockIdx.x * BLOCK_N;

    // Shared memory layout:
    // [0 .. A_BYTES-1]:              A_smem[BK][BM]
    // [A_BYTES .. A_BYTES+B_BYTES-1]: B_smem[BK][BN]  
    // After that: pad + mbarrier
    
    constexpr size_t A_BYTES = BLOCK_M * BLOCK_K * sizeof(__nv_bfloat16); // 8192
    constexpr size_t B_BYTES = BLOCK_N * BLOCK_K * sizeof(__nv_bfloat16); // 8192
    constexpr size_t PAD = 64;
    
    __nv_bfloat16* smem_A = reinterpret_cast<__nv_bfloat16*>(sdata);
    __nv_bfloat16* smem_B = reinterpret_cast<__nv_bfloat16*>(sdata + A_BYTES);
    uint64_t* bar_tma = reinterpret_cast<uint64_t*>(sdata + A_BYTES + B_BYTES + PAD);
    
    // --- Initialize barrier (thread 0 only) ---
    if (tid == 0) {
        init_smem_barrier(bar_tma, 1);
        fence_mbarrier_init();
        mbarrier_arrive(bar_tma);   // complete initial phase
    }
    __syncthreads();
    mbarrier_wait_parity(bar_tma, 0);

    // --- Allocate Tensor Memory for D accumulator ---
    // MUST BE EXECUTED BY ALL THREADS IN THE WARGROUP (all 128 threads)
    uint32_t num_cols = 64;
    uint32_t tmem_addr = 0;
    {
        // Every thread executes alloc; only tid==0 captures the returned address
        uint32_t my_addr;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
            : "=r"(my_addr)
            : "r"((uint32_t)__cvta_generic_to_shared(bar_tma)),
              "r"(num_cols));
        // Broadcast the address from tid 0
        tmem_addr = __shfl_sync(0xFFFFFFFF, my_addr, 0);
    }
    __syncthreads();
    uint32_t base_tmaddr = tmem_addr;

    // --- Clear D accumulator in TMEM ---
    // Threads 0..63 write zeros to their respective lanes
    if (tid < 64) {
        for (uint32_t c = 0; c < BLOCK_N; c += 4) {
            uint32_t addr = (tid << 16) | c;
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
                :: "r"(addr), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    }
    fence_proxy_async();
    __syncthreads();

    // --- Main K-tile loop ---
    uint64_t nk_tiles = K / BLOCK_K;
    uint32_t instr = make_umma_instr(BLOCK_M, BLOCK_N);
    uint32_t load_bytes = (BLOCK_M * BLOCK_K + BLOCK_N * BLOCK_K) * sizeof(__nv_bfloat16);

    for (uint64_t kt = 0; kt < nk_tiles; ++kt) {
        uint64_t k_off = kt * BLOCK_K;
        
        // === Load A and B tiles via TMA (only thread 0 initiates) ===
        if (tid == 0) {
            mbarrier_arrive_expect_tx(bar_tma, load_bytes);
            tma_load_2d(&tma_A_desc, bar_tma, smem_A, (int32_t)k_off, (int32_t)out_m);
            tma_load_2d(&tma_B_desc, bar_tma, smem_B, (int32_t)k_off, (int32_t)out_n);
        }
        // All threads wait for TMA completion (tx-count reaches 0)
        mbarrier_wait_parity(bar_tma, kt & 1);
        
        // Make SMEM visible to async proxy for UMMA consumption
        fence_proxy_async_shared();
        
        // === Compute: UMMA in K=16 chunks (only thread 0 issues) ===
        if (tid == 0) {
            uint32_t accum_flag = (kt == 0) ? 0 : 1;
            for (uint32_t ki = 0; ki < BLOCK_K / 16; ++ki) {
                __nv_bfloat16* a_base = smem_A + ki * 16 * BLOCK_M;
                __nv_bfloat16* b_base = smem_B + ki * 16 * BLOCK_N;
                
                uint64_t sd_a = make_kmaj_sdesc(a_base);
                uint64_t sd_b = make_kmaj_sdesc(b_base);
                
                umma_f16_cg1(base_tmaddr, sd_a, sd_b, instr, accum_flag);
            }
            // Wait for UMMA completion
            umma_sync_all();
        }
        // Synchronize so all threads know compute is done before next iteration's TMA
        __syncthreads();
    }

    // Final fence after all compute
    fence_proxy_async();

    // --- Epilogue: TMEM → Global output ---
    // Each of threads 0..63 reads its own lane row and stores to global
    if (tid < 64 && out_m + tid < M) {
        uint64_t row_global = out_m + tid;
        __nv_bfloat16* C_row = C_out + row_global * N;
        uint64_t col_start = out_n;
        
        for (uint32_t c = 0; c < BLOCK_N; c += 4) {
            uint32_t r0, r1, r2, r3;
            uint32_t addr = (tid << 16) | c;
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
    
    // Deallocate TMEM — MUST BE EXECUTED BY ALL THREADS IN THE WARGROUP
    {
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
    
    // TMA for A[M,K]: coords(inner=K, outer=M), box={BK,BM}
    CUtensorMap tma_A;
    CU_CHECK(create_tma_2d_descriptor_BF16(&tma_A, const_cast<__nv_bfloat16*>(A_ptr),
        K, M, K * sizeof(__nv_bfloat16), BLOCK_K, BLOCK_M,
        CU_TENSOR_MAP_SWIZZLE_128B));
    
    // TMA for B virtually viewed as B.T[K,N]:
    // B[N,K] row-major. B.T[k,n] = B[n,k]. With dims{K,N} stride{K*2}:
    // addr = base + k_inner*2 + n_outer*(K*2) = B[n][k]*2 ✓
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