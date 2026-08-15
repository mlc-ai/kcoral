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
    // Use unique stringification-safe local label
    asm volatile(
        "0:\n"
        "mbarrier.try_wait.parity.shared.b64 %0, [%1], %2;\n"
        "@%0 bra 0b;\n"
        : "=p"(__null_reg())
        : "r"(addr), "r"(parity)
        : "memory");
}

// Helper to get null predicate register
__device__ __forceinline__ void __null_reg() {}

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
    
    // Layout (all offsets from smem_raw):
    //   [0..7]     barriers[0]  (stage 0 load barrier)
    //   [8..15]    barriers[1]  (stage 1 load barrier)
    //   [16..19]   tmem_addr_storage
    //   [32..]     tile data (A_tile, B_tile) — 16-byte aligned
    
    uint64_t* barriers = reinterpret_cast<uint64_t*>(smem_raw);
    uint32_t* tmem_addr_storage = reinterpret_cast<uint32_t*>(smem_raw + 16);
    char* tile_data = smem_raw + 32;
    __nv_bfloat16* A_tile = reinterpret_cast<__nv_bfloat16*>(tile_data);
    __nv_bfloat16* B_tile = A_tile + BM * BK;
    
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
    
    // ===================== INITIALIZATION =====================
    if (tid == 0) {
        mbarrier_init(&barriers[0], 1);  // expect 1 arrival + tx-count
        mbarrier_init(&barriers[1], 1);
        mbarrier_fence_init();
        
        prefetch_tensormap(&tma_desc_A);
        prefetch_tensormap(&tma_desc_B);
    }
    __syncthreads();
    
    // ===================== TMEM ALLOCATION =====================
    // Warp 0 allocates TMEM (single warp issue granularity)
    if (tid < WARP_SIZE) {
        tmem_addr_storage[0] = TMEM_COLS;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     : "+r"(tmem_addr_storage[0]) : "r"(TMEM_COLS));
    }
    __syncthreads();
    
    // Build shared memory descriptors (non-swizzled, K-major)
    // A [BK][BM]: SBO=128, LBO=(BM/8)*SBO=(128/8)*128=2048
    uint64_t desc_a = make_smem_desc(A_tile, 2048, 128);
    // B [BK][BN]: SBO=128, LBO=(BN/8)*SBO=(256/8)*128=4096
    uint64_t desc_b = make_smem_desc(B_tile, 4096, 128);
    
    unsigned idesc = make_instr_desc(BM, BN);
    
    // ===================== MAIN LOOP =====================
    int stage = 0;
    bool first_step = true;
    
    // Pre-load stage 0 before loop
    if (tid == 0) {
        tma_load_2d_mbar(&tma_desc_A, &barriers[0], A_tile, 0, m_start);
        tma_load_2d_mbar(&tma_desc_B, &barriers[0], B_tile, 0, n_start);
        mbarrier_expect_tx(&barriers[0], total_tx);
    }
    __syncthreads();
    
    for (int k_step = 0; k_step < num_k_steps; k_step++) {
        int cur_bar_idx = stage;
        int next_bar_idx = stage ^ 1;
        int next_k = (k_step + 1) * BK;
        
        // ---- Producer: preload next stage ----
        if (tid == 0 && k_step + 1 < num_k_steps) {
            tma_load_2d_mbar(&tma_desc_A, &barriers[next_bar_idx], A_tile, next_k, m_start);
            tma_load_2d_mbar(&tma_desc_B, &barriers[next_bar_idx], B_tile, next_k, n_start);
            mbarrier_expect_tx(&barriers[next_bar_idx], total_tx);
        }
        __syncthreads();
        
        // ---- Consumer: wait for current tile loaded ----
        mbarrier_wait_parity(&barriers[cur_bar_idx], 0);
        
        // ---- Compute: UMMA ----
        // Single-thread semantics for tcgen05.mma
        if (tid == 0) {
            bool accum = !first_step;
            unsigned accum_val = accum ? 1u : 0u;
            
            // Issue UMMA with accumulator predicate
            asm volatile(
                ".reg .pred p_acc;\n"
                "setp.ne.b32 p_acc, %4, 0;\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p_acc;\n"
                :: "r"(0), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum_val));
        }
        __syncthreads();
        
        // Advance stage
        stage = next_bar_idx;
        first_step = false;
    }
    
    // Ensure all async ops completed before epilogue
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    
    // ===================== EPILOGUE: TMEM -> Global =====================
    // Each thread tid < BM loads its row from TMEM (FP32), converts to BF16, writes to C
    if (tid < BM) {
        int m_global = m_start + tid;
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
    
    __syncthreads();
    
    // ===================== DEALLOCATE TMEM =====================
    if (tid < WARP_SIZE) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 0, %0;" :: "r"(TMEM_COLS));
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
    
    // Shared memory: 32B header + A_tile (4096) + B_tile (8192)
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