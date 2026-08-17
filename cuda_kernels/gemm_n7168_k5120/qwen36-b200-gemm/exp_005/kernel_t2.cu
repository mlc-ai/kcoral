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
static constexpr int BM = 128;
static constexpr int BN = 256;
static constexpr int BK = 16;
static constexpr int NUM_THREADS = 256;
static constexpr int WARP_SIZE = 32;
static constexpr int PIPE_STAGES = 2;
static constexpr int TMEM_COLS = 256;

// Shared memory layout sizes (bytes)
static constexpr size_t BARRIER_BYTES = PIPE_STAGES * sizeof(uint64_t);   // 1 x uint64_t per stage
static constexpr size_t A_TILE_BYTES = BM * BK * sizeof(__nv_bfloat16);   // 4096
static constexpr size_t B_TILE_BYTES = BN * BK * sizeof(__nv_bfloat16);   // 8192

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
    d  = (uint64_t)((addr & 0x3FFFF) >> 4);
    d |= (uint64_t)(((lbo_bytes & 0x3FFFF) >> 4)) << 16;
    d |= (uint64_t)(((sbo_bytes & 0x3FFFF) >> 4)) << 32;
    d |= (uint64_t)1ULL << 46;  // version = 1
    return d;
}

__device__ __forceinline__ unsigned make_instr_desc(unsigned M_dim, unsigned N_dim) {
    unsigned d = 0;
    d  = (1u << 4);                                        // dtype = FP32
    d |= (1u << 7);                                        // atype = BF16
    d |= (1u << 10);                                       // btype = BF16
    d |= ((N_dim >> 3) & 0x3F) << 17;                     // N >> 3
    d |= ((M_dim >> 4) & 0x1F) << 24;                     // M >> 4
    return d;
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
    
    // External shared memory
    extern __shared__ char smem_raw[];
    
    // Layout: [barriers] + [A_tile, B_tile]
    uint64_t* __align__(8) barriers = reinterpret_cast<uint64_t*>(smem_raw);
    char* data_ptr = smem_raw + PIPE_STAGES * sizeof(uint64_t);
    __nv_bfloat16* __align__(16) A_tile = reinterpret_cast<__nv_bfloat16*>(data_ptr);
    __nv_bfloat16* __align__(16) B_tile = A_tile + BM * BK;
    
    // Block tile coordinates
    int m_block = blockIdx.y;
    int n_block = blockIdx.x;
    int m_start = m_block * BM;
    int n_start = n_block * BN;
    int num_k_steps = (K + BK - 1) / BK;
    
    // Byte sizes for TMA transactions
    unsigned A_bytes = BM * BK * sizeof(__nv_bfloat16);   // 4096
    unsigned B_bytes = BN * BK * sizeof(__nv_bfloat16);   // 8192
    unsigned total_tx = A_bytes + B_bytes;                 // 12288
    
    // Number of threads arriving at each phase:
    // Producer (tid==0): 1 thread does arrive
    // Consumer side (UMMA wait signal): ALL 256 threads arrive after compute
    // Total arrivals per barrier cycle = 1 (producer) + 256 (consumer team) = 257
    // However, we split into load_barrier (producer arrives + consumer waits) and
    // use the tx-count mechanism instead of arrival counting for TMA completion.
    // So barrier expects: producer arrives (count=1), tx_count tracks bytes, 
    // consumer just waits for tx_count to drain.
    
    // For simplicity: barrier init with count = 1 (just producer arrives)
    // The tx-count is set separately via expect_tx and decremented automatically
    // when TMA completes. Once both arrival_count==0 AND tx_count==0, barrier completes.
    // Consumer calls mbarrier.wait which spins until both conditions are met.
    
    // ===================== INITIALIZATION =====================
    if (tid == 0) {
        // Each barrier: 1 arrival (from producer) + tx-count from TMA
        mbarrier_init(&barriers[0], 1);
        mbarrier_init(&barriers[1], 1);
        mbarrier_fence_init();
        
        prefetch_tensormap(&tma_desc_A);
        prefetch_tensormap(&tma_desc_B);
    }
    __syncthreads();
    
    // ===================== TMEM ALLOCATION =====================
    // All threads need TMEM base address; warp 0 performs allocation
    // We store it in a register local to each thread (broadcast later via shared mem or just reuse)
    uint32_t tmem_addr = 0;
    if (tid < WARP_SIZE) {
        uint32_t tmp = TMEM_COLS;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     : "=r"(tmp) : "r"(tmp));
        // The alloc writes result to [%0], so read it back
        asm volatile("" ::: "memory");
    }
    __syncthreads();
    // Warp 0 wrote to itself; we need the address. Re-read from the location.
    // Actually tcgen05.alloc writes to a shared mem pointer passed in. Let's use a shared variable.
    
    // Simpler approach: declare a __shared__ uint32_t for tmem address
    // But we can't declare variables inside kernel body after extern. 
    // Let's use the barrier array overflow area instead.
    // barriers[0..1] used for sync, so we can use a nearby slot.
    
    // Actually, let me restructure: put tmem_addr storage explicitly
    // Using shared memory byte offset beyond barriers
    size_t tmem_addr_offset = PIPE_STAGES * sizeof(uint64_t) - sizeof(uint32_t);
    uint32_t* tmem_addr_slot = reinterpret_cast<uint32_t*>(smem_raw + tmem_addr_offset);
    
    if (tid < WARP_SIZE) {
        tmem_addr_slot[0] = TMEM_COLS;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     : "=r"(tmem_addr_slot[0]) : "r"(TMEM_COLS));
    }
    __syncthreads();
    
    // Re-init barriers since we overwrote part of the barrier space
    // This is messy. Let me just avoid overlap entirely.
    // Reset: put barriers first, then tmem_addr, then tiles
    if (tid == 0) {
        // Overwrite region was small; realloc barrier safely
        // Actually this corrupts barriers[1]. Need redesign.
    }
    
    // CLEAN APPROACH: Don't mix allocations with barrier space.
    // Use the fact that we only need 1 barrier (reuse same one cyclically).
    // Barrier at offsets[0], tmem_addr at a separate known offset.
    // Since we already allocated shared memory, let's recalculate cleanly.
    
    // Re-layout:
    // Offset 0:     8B   barriers[0]
    // Offset 8:     8B   barriers[1]  
    // Offset 16:    4B   tmem_addr (pad to 8B for alignment)
    // Offset 24:    ... data tiles (but must be 16-byte aligned) -> pad to offset 32
    // Offset 32:    A_tile, B_tile
    
    // Reset everything cleanly
    if (tid == 0) {
        barriers[0] = 0; barriers[1] = 0;
    }
    __syncthreads();
    
    if (tid == 0) {
        mbarrier_init(&barriers[0], 1);
        mbarrier_init(&barriers[1], 1);
        mbarrier_fence_init();
    }
    __syncthreads();
    
    // Now allocate TMEM
    uint32_t* tmem_addr_storage = reinterpret_cast<uint32_t*>(smem_raw + 16);
    if (tid < WARP_SIZE) {
        tmem_addr_storage[0] = TMEM_COLS;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     : "=r"(tmem_addr_storage[0]) : "r"(TMEM_COLS));
    }
    __syncthreads();
    
    // Update tile pointers based on new layout (tiles start at offset 32)
    char* tile_data = smem_raw + 32;
    A_tile = reinterpret_cast<__nv_bfloat16*>(tile_data);
    B_tile = A_tile + BM * BK;
    
    // Build shared memory descriptors
    // Non-swizzled, K-major layout
    // For A [BK][BM]: SBO = stride between consecutive 8-row blocks along M
    // Reference: SBO = 8*16=128, LBO = (BLOCK_M/8)*SBO = (128/8)*128 = 2048
    uint64_t desc_a = make_smem_desc(A_tile, 2048, 128);
    
    // For B [BK][BN]: SBO = 128, LBO = (256/8)*128 = 4096
    uint64_t desc_b = make_smem_desc(B_tile, 4096, 128);
    
    unsigned idesc = make_instr_desc(BM, BN);
    
    // ===================== MAIN LOOP =====================
    // Double-buffered pipeline with single reused barrier pair
    int stage = 0;
    bool first_step = true;
    
    // Pre-load first tile before loop
    if (tid == 0) {
        tma_load_2d_mbar(&tma_desc_A, &barriers[0], A_tile, 0, m_start);
        tma_load_2d_mbar(&tma_desc_B, &barriers[0], B_tile, 0, n_start);
        mbarrier_expect_tx(&barriers[0], total_tx);
    }
    __syncthreads();
    
    for (int k_step = 0; k_step < num_k_steps; k_step++) {
        int cur_bar = stage;
        int next_bar = stage ^ 1;
        int next_k = (k_step + 1) * BK;
        
        // ---- Producer: load next stage (for future iteration) ----
        if (tid == 0 && k_step + 1 < num_k_steps) {
            tma_load_2d_mbar(&tma_desc_A, &barriers[next_bar], A_tile, next_k, m_start);
            tma_load_2d_mbar(&tma_desc_B, &barriers[next_bar], B_tile, next_k, n_start);
            mbarrier_expect_tx(&barriers[next_bar], total_tx);
        }
        __syncthreads();
        
        // ---- Consumer: wait for current tile loaded ----
        // All threads spin-wait on the barrier (arrival=producer, tx=TMA bytes)
        mbarrier_wait_parity(&barriers[cur_bar], 0);
        
        // ---- Compute: UMMA ----
        if (tid == 0) {
            // Issue UMMA: D[m][n] += A[m][k] * B[n][k] for k in [k_step*BK, (k_step+1)*BK)
            // Single-thread semantics for tcgen05.mma
            bool accum = !first_step;
            asm volatile(
                "{\n.reg .pred p_acc;\n"
                "setp.ne.b32 p_acc, %4, 0;\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p_acc;\n}\n"
                :: "r"(0), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum ? 1u : 0u));
            
            // Commit UMMA to the same barrier (tracked by async group)
            // Wait for UMMA completion via commit + mbarrier.wait pattern
            // Instead, use tcgen05.commit with mbarrier tracking
        }
        __syncthreads();
        
        // Fence and commit for subsequent stages
        // Use tcgen05.commit to track UMMA completion through mbarrier
        // Since we're reusing barriers, we don't need a separate compute barrier.
        // The UMMA reads from shared memory (already loaded), so no conflict.
        
        // Move to next stage
        stage = next_bar;
        first_step = false;
    }
    
    // Final fence: ensure last UMMA completed before epilogue
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    
    // ===================== EPILOGUE: TMEM -> Global =====================
    // TMEM stores results as FP32, organized by lane=row, column=output_n
    // Each thread (which maps to a TMEM lane via warpgroup mapping) 
    // loads its row of BM entries for BN output columns.
    
    // TMEM access restriction: warp i accesses lanes i*32 .. (i+1)*32-1
    // Warpgroup has 4 warps (128 threads), accessing all 128 lanes (rows).
    // We have 256 threads = 2 warpgroups = full access to TMEM.
    
    // With BM=128 rows, exactly 1 warpgroup (128 threads) covers all rows.
    // Threads 0-127 write to global; threads 128-255 don't need to participate in stores.
    
    // We'll use coalesced writes: each warpgroup contributes to writing the BM x BN block.
    // Within each step, collect results into shared memory, then coalesced global writes.
    
    // Phase 1: Each of 128 threads loads its row from TMEM to registers
    // Load in chunks of 4 columns
    if (tid < BM) {
        int m_local = tid;       // Row within block [0, BM)
        int m_global = m_start + m_local;
        if (m_global >= M) return;
        
        TC* c_row = C_gmem + (uint64_t)m_global * N + n_start;
        
        for (int nc = 0; nc < BN; nc += 4) {
            float f0, f1, f2, f3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=f"(f0), "=f"(f1), "=f"(f2), "=f"(f3) : "r"(nc));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            int n_global = n_start + nc;
            if (n_global      < N) c_row[nc]   = __float2bfloat16(f0);
            if (n_global + 1  < N) c_row[nc+1] = __float2bfloat16(f1);
            if (n_global + 2  < N) c_row[nc+2] = __float2bfloat16(f2);
            if (n_global + 3  < N) c_row[nc+3] = __float2bfloat16(f3);
        }
    }
    
    // Ensure all global stores complete
    __syncthreads();
    
    // ===================== DEALLOCATE TMEM =====================
    if (tid < WARP_SIZE) {
        uint32_t ncols = TMEM_COLS;
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 0, %0;" :: "r"(ncols));
    }
}

// =============================================================================
// Host
// =============================================================================

CUresult create_tma_2d_bf16(CUtensorMap* out, void* gmem,
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
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        gmem, globalDim, globalStrides, boxDim, elemStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2_promo, oob_fill);
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    CUtensorMap tma_A;
    CUresult err = create_tma_2d_bf16(&tma_A, const_cast<void*>(A.data_ptr()),
        K, M, BK, BM,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (err != CUDA_SUCCESS) { fprintf(stderr, "TMA A failed: %d\n", (int)err); exit(1); }
    
    CUtensorMap tma_B;
    err = create_tma_2d_bf16(&tma_B, const_cast<void*>(B.data_ptr()),
        K, N, BK, BN,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (err != CUDA_SUCCESS) { fprintf(stderr, "TMA B failed: %d\n", (int)err); exit(1); }
    
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(NUM_THREADS);
    
    // Shared memory: 2 barriers (16B) + 4B tmem_addr + pad to 32B + tiles
    size_t smem_size = 32 + A_TILE_BYTES + B_TILE_BYTES;
    
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