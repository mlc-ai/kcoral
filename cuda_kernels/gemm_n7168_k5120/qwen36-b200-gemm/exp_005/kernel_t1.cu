#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <cstdio>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace gemm_blackwell {

// Tile configuration
static constexpr int BM = 128;   // M dimension per CTA
static constexpr int BN = 256;   // N dimension per CTA  
static constexpr int BK = 16;    // K dimension per micro-step
static constexpr int NUM_THREADS = 256;
static constexpr int WARP_SIZE = 32;
static constexpr int PRODUCER_THREADS = 1;     // Just thread 0
static constexpr int CONSUMER_THREADS = 128;   // Threads 0..127
static constexpr int TOTAL_CONSUMERS = 128;    // Consumer warps arrive
static constexpr int PIPE_STAGES = 2;          // Double buffer
static constexpr int TMEM_COLS = 256;          // Columns in TMEM (power of 2)
static constexpr int EXPECTED_TX_PER_LOAD = 1; // Producer arrives on barrier

// =============================================================================
// Device-side helpers
// =============================================================================

__device__ __forceinline__ void mbarrier_init(uint64_t* bar, unsigned count) {
    unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(addr), "r"(count));
}

__device__ __forceinline__ void mbarrier_fence_init() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive(uint64_t* bar) {
    unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(addr) : "memory");
}

__device__ __forceinline__ void mbarrier_expect_tx(uint64_t* bar, unsigned tx_bytes) {
    unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(addr), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_parity(uint64_t* bar, unsigned parity) {
    unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{\n.reg .pred p;\n.Lwait_mbar%=: "
        "mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
        "@!p bra .Lwait_mbar%=;\n}\n"
        :: "r"(addr), "r"(parity));
}

__device__ __forceinline__ void tma_load_2d_mbar(
    const CUtensorMap* desc, uint64_t* mbar, void* smem_dst,
    int32_t coord0, int32_t coord1)
{
    unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
    unsigned mbar_addr = static_cast<unsigned>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4}], [%2];"
        :: "r"(smem_addr), "l"((unsigned long long)desc),
           "r"(mbar_addr), "r"(coord0), "r"(coord1) : "memory");
}

__device__ __forceinline__ void prefetch_tensormap(const CUtensorMap* desc) {
    asm volatile("prefetch.tensormap [%0];" :: "l"((unsigned long long)desc));
}

__device__ __forceinline__ uint64_t make_smem_desc(void* ptr, unsigned lbo_bytes, unsigned sbo_bytes) {
    uint64_t d = 0;
    unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(ptr));
    d  = (uint64_t)((addr & 0x3FFFF) >> 4);                    // bits 0-13: base address >> 4
    d |= (uint64_t)(((lbo_bytes & 0x3FFFF) >> 4)) << 16;      // bits 16-29: LBO >> 4
    d |= (uint64_t)(((sbo_bytes & 0x3FFFF) >> 4)) << 32;      // bits 32-45: SBO >> 4
    d |= (uint64_t)1ULL << 46;                                // bits 46-48: version = 1
    // bits 61-63: swizzle = 0 (no swizzle)
    return d;
}

__device__ __forceinline__ unsigned make_instr_desc(unsigned M_dim, unsigned N_dim) {
    unsigned d = 0;
    d  = (1u << 4);                                         // dtype = FP32
    d |= (1u << 7);                                         // atype = BF16
    d |= (1u << 10);                                        // btype = BF16
    // No transpose for either A or B
    d |= ((N_dim >> 3) & 0x3F) << 17;                      // bits 17-22: N >> 3
    d |= ((M_dim >> 4) & 0x1F) << 24;                      // bits 24-28: M >> 4
    return d;
}

__device__ __forceinline__ void tmem_alloc_warp(uint32_t* smem_dst, unsigned ncols) {
    unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_warp(unsigned ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 0, %0;" :: "r"(ncols));
}

__device__ __forceinline__ void issue_umma(uint32_t d_offset, uint64_t desc_a, uint64_t desc_b,
                                            unsigned idesc, bool accumulate) {
    asm volatile(
        "{\n.reg .pred p_accum;\n"
        "setp.ne.b32 p_accum, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p_accum;\n}\n"
        :: "r"(d_offset), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accumulate ? 1u : 0u));
}

// =============================================================================
// Kernel
// =============================================================================

template<typename TA, typename TB, typename TC>
__global__ void __launch_bounds__(NUM_THREADS)
gemm_kernel(const __grid_constant__ CUtensorMap tma_desc_A,
            const __grid_constant__ CUtensorMap tma_desc_B,
            TA const* __restrict__ A_gmem,
            TB const* __restrict__ B_gmem,
            TC*         __restrict__ C_gmem,
            int M, int N, int K)
{
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    
    // External shared memory
    extern __shared__ char smem_raw[];
    
    // Shared memory layout (all 16-byte aligned):
    // [barriers: 4 x 8B = 32B] + [padding] + [A_tile] + [B_tile]
    uint64_t* __align__(16) barriers = reinterpret_cast<uint64_t*>(smem_raw);
    // Stage 0: load_bar[0], compute_bar[0]
    // Stage 1: load_bar[1], compute_bar[1]
    char* smem_data = smem_raw + 4 * sizeof(uint64_t);
    __nv_bfloat16* __align__(16) A_tile = reinterpret_cast<__nv_bfloat16*>(smem_data);
    __nv_bfloat16* __align__(16) B_tile = A_tile + BM * BK;
    
    // Block tile coordinates
    int m_block = blockIdx.y;
    int n_block = blockIdx.x;
    int m_start = m_block * BM;
    int n_start = n_block * BN;
    int num_k_steps = (K + BK - 1) / BK;
    
    // Byte sizes
    unsigned A_bytes = BM * BK * sizeof(__nv_bfloat16);  // 128*16*2 = 4096
    unsigned B_bytes = BN * BK * sizeof(__nv_bfloat16);  // 256*16*2 = 8192
    unsigned total_bytes = A_bytes + B_bytes;             // 12288
    
    // ===================== INITIALIZATION =====================
    // Thread 0: Initialize barriers and prefetched tensors
    if (tid == 0) {
        // Each load barrier expects: 1 producer arrival + TOTAL_CONSUMERS consumer arrivals
        unsigned barrier_count = 1 + TOTAL_CONSUMERS;
        mbarrier_init(&barriers[0], barrier_count);  // load_bar[0]
        mbarrier_init(&barriers[1], barrier_count);  // load_bar[1]
        mbarrier_init(&barriers[2], TOTAL_CONSUMERS);  // compute_bar[0]
        mbarrier_init(&barriers[3], TOTAL_CONSUMERS);  // compute_bar[1]
        mbarrier_fence_init();
        
        prefetch_tensormap(&tma_desc_A);
        prefetch_tensormap(&tma_desc_B);
    }
    __syncthreads();
    
    // ===================== TMEM ALLOCATION =====================
    // Warp 0 allocates TMEM space
    if (tid < WARP_SIZE) {
        // We need TMEM_COLS columns for the accumulator
        // Result is BM x BN = 128 x 256 FP32 values
        // Each lane stores one row, each column stores one output N-index
        tmem_alloc_warp(reinterpret_cast<uint32_t*>(&tmem_addr_reg), TMEM_COLS);
    }
    __syncthreads();
    
    // Register to hold TMEM base address (written by alloc)
    uint32_t tmem_addr_reg = 0;
    
    // Build descriptors
    // A: K-major [BK][BM], stride between columns (M-axis) = BM*2 = 256 bytes
    // Reference formula: SBO = 8*16 = 128, LBO = (BM/8)*SBO = 16*128 = 2048
    unsigned sbo_a = 128;
    unsigned lbo_a = (BM / 8) * sbo_a;  // 2048
    uint64_t desc_a = make_smem_desc(A_tile, lbo_a, sbo_a);
    
    // B: K-major [BK][BN], stride between columns (N-axis) = BN*2 = 512 bytes
    // SBO = 128, LBO = (BN/8)*SBO = 32*128 = 4096
    unsigned sbo_b = 128;
    unsigned lbo_b = (BN / 8) * sbo_b;  // 4096
    uint64_t desc_b = make_smem_desc(B_tile, lbo_b, sbo_b);
    
    // UMMA instruction descriptor: BM x BN, BF16 x BF16 -> FP32
    unsigned idesc = make_instr_desc(BM, BN);
    
    // ============================================================
    // MAIN PIPELINE LOOP
    // ============================================================
    int stage = 0;
    
    // ---- Initial load for stage 0 ----
    if (tid == 0) {
        // Load first K-tile
        tma_load_2d_mbar(&tma_desc_A, &barriers[stage*2], A_tile, 0, m_start);
        tma_load_2d_mbar(&tma_desc_B, &barriers[stage*2], B_tile, 0, n_start);
        mbarrier_expect_tx(&barriers[stage*2], total_bytes);
    }
    __syncthreads();
    
    for (int k_step = 0; k_step < num_k_steps; k_step++) {
        int next_stage = stage ^ 1;
        int next_k = (k_step + 1) * BK;
        
        // ---- Producer: preload next stage ----
        if (tid == 0 && k_step + 1 < num_k_steps) {
            tma_load_2d_mbar(&tma_desc_A, &barriers[next_stage*2], A_tile, next_k, m_start);
            tma_load_2d_mbar(&tma_desc_B, &barriers[next_stage*2], B_tile, next_k, n_start);
            mbarrier_expect_tx(&barriers[next_stage*2], total_bytes);
        }
        __syncthreads();
        
        // ---- Consumer: wait for current stage to be loaded ----
        if (tid < CONSUMER_THREADS) {
            mbarrier_arrive(&barriers[stage*2]);  // Arrive at load barrier
        }
        __syncthreads();
        // All threads wait for load completion
        mbarrier_wait_parity(&barriers[stage*2], 0);
        
        // ---- Compute: issue UMMA ----
        // Single thread issues UMMA (single-thread semantics for tcgen05.mma)
        if (tid == CONSUMER_THREADS) {  // Thread 128 issues
            issue_umma(0, desc_a, desc_b, idesc, k_step > 0);
        }
        
        // Commit UMMA completion to compute barrier
        // All consumer threads signal completion at compute barrier
        if (tid < CONSUMER_THREADS) {
            mbarrier_arrive(&barriers[stage*2 + 1]);  // Arrive at compute barrier
        }
        __syncthreads();
        
        // Advance
        stage = next_stage;
    }
    
    // ---- Final drain: wait for last UMMA ----
    mbarrier_wait_parity(&barriers[(num_k_steps % 2) * 2 + 1], 0);
    
    // ===================== EPILOGUE: TMEM -> Global =====================
    // Each thread tid < BM reads its row from TMEM and stores to C
    // TMEM layout: lane = row (m), column = output index (n)
    // We load in batches of 4 columns per iteration
    
    if (tid < BM) {
        int m_local = tid;
        int m_global = m_start + m_local;
        if (m_global >= M) return;
        
        TC* c_row = C_gmem + (uint64_t)m_global * N + n_start;
        
        for (int nc = 0; nc < BN; nc += 4) {
            // Load 4 FP32 values from TMEM (columns nc..nc+3, lane=m_local)
            float f0, f1, f2, f3;
            
            // Collective TMEM load: all threads participate, each reads at their lane
            // Shape: 32 rows (mapped by lane within warp) x 32b x 4 reps
            // With 256 threads in 8 warps, all warps issue the same instruction
            // Each thread's lane index within its 128-lane subset picks the row
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=f"(f0), "=f"(f1), "=f"(f2), "=f"(f3)
                         : "r"(nc));
            
            // Wait for TMEM load completion
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            // Convert to BF16 and store to global
            int n_global = n_start + nc;
            if (n_global < N)          c_row[nc]   = __float2bfloat16(f0);
            if (n_global + 1 < N)      c_row[nc+1] = __float2bfloat16(f1);
            if (n_global + 2 < N)      c_row[nc+2] = __float2bfloat16(f2);
            if (n_global + 3 < N)      c_row[nc+3] = __float2bfloat16(f3);
        }
    }
    
    // ===================== DEALLOCATE TMEM =====================
    if (tid < WARP_SIZE) {
        tmem_dealloc_warp(TMEM_COLS);
    }
}

// =============================================================================
// Host-side TMA descriptor builder
// =============================================================================

CUresult create_tma_2d_bf16(
    CUtensorMap* out, void* gmem,
    uint64_t inner_dim, uint64_t outer_dim,
    uint32_t box_inner, uint32_t box_outer,
    CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion l2_promo,
    CUtensorMapFloatOOBfill oob_fill)
{
    cuuint64_t globalDim[2] = {inner_dim, outer_dim};
    cuuint64_t globalStrides[1] = {inner_dim * sizeof(__nv_bfloat16)};
    cuuint32_t boxDim[2] = {box_inner, box_outer};
    cuuint32_t elemStrides[2] = {1, 1};
    
    return cuTensorMapEncodeTiled(out,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        gmem,
        globalDim,
        globalStrides,
        boxDim,
        elemStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        l2_promo,
        oob_fill);
}

// =============================================================================
// TVM-FFI entry point
// =============================================================================

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);  // B is [N, K], so N = 7168, K = 5120
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    // Build TMA descriptors
    // A: [M, K] -> inner=K, outer=M, box=[BK, BM]=[16, 128]
    CUtensorMap tma_A;
    CUresult err = create_tma_2d_bf16(&tma_A, 
        const_cast<void*>(A.data_ptr()),
        K, M, BK, BM,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "create_tma_A failed: %d\n", (int)err);
        exit(1);
    }
    
    // B: [N, K] -> inner=K, outer=N, box=[BK, BN]=[16, 256]
    CUtensorMap tma_B;
    err = create_tma_2d_bf16(&tma_B,
        const_cast<void*>(B.data_ptr()),
        K, N, BK, BN,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "create_tma_B failed: %d\n", (int)err);
        exit(1);
    }
    
    // Launch kernel
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(NUM_THREADS);
    
    // Shared memory: 4 barriers (32B) + padding + A_tile (4096B) + B_tile (8192B)
    size_t smem_size = 4 * sizeof(uint64_t) + BM * BK * 2 + BN * BK * 2;
    
    gemm_kernel<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16><<<grid, block, smem_size, stream>>>(
        tma_A, tma_B,
        static_cast<const __nv_bfloat16*>(A.data_ptr()),
        static_cast<const __nv_bfloat16*>(B.data_ptr()),
        static_cast<__nv_bfloat16*>(C.data_ptr()),
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_blackwell::run);

}  // namespace gemm_blackwell