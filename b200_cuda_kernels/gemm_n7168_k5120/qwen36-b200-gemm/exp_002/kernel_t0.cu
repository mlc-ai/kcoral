#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
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

// ---- Helper device functions ----

__device__ __forceinline__ uint32_t get_smid() {
    uint32_t smid;
    asm ("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
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

__device__ __forceinline__ void tma_load_2d_mbarrier_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_store_2d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
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
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];"
        :: "r"(a) : "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_k_major_fn(void* smem_ptr, uint32_t m_dim, uint32_t k_dim_bytes_per_row) {
    // K-major swizzled descriptor: atom m-mode = m_dim, atom k-mode spans 128B / sizeof(bf16) = 64
    // LBO not used (assumed 1), SBO = 8 * 128 = 1024 bytes
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)((addr >> 4) & 0x3FFF);              // bits 0-13: start address
    d |= (uint64_t)1 << 16;                              // bits 16-29: LBO = 1 (swizzled)
    d |= (uint64_t)(1024 >> 4) << 32;                   // bits 32-45: SBO = 1024
    d |= (uint64_t)1 << 46;                              // bits 46-48: version = 1
    d |= (uint64_t)0 << 49;                              // bits 49-51: base_offset = 0
    d |= (uint64_t)0 << 52;                              // bit 52: LDM = 0 (byte offset relative)
    d |= (uint64_t)2 << 61;                              // bits 61-63: SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_kk_fn(uint32_t M, uint32_t N) {
    // BF16 x BF16 -> FP32, A: K-Major (no transpose), B: K-Major (no transpose)
    uint32_t d = 0;
    d |= (1u << 4);     // dtype = F32
    d |= (1u << 7);     // atype = BF16
    d |= (1u << 10);    // btype = BF16
    d |= (0u << 15);    // A: No Transpose (K-Major)
    d |= (0u << 16);    // B: No Transpose (K-Major)
    d |= ((N / 8) << 17);   // n_dim >> 3
    d |= ((M / 16) << 24);  // m_dim >> 4
    return d;
}

__device__ __forceinline__ void tmem_ld_x16_b32_fn(uint32_t col,
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
    uint32_t& r4, uint32_t& r5, uint32_t& r6, uint32_t& r7,
    uint32_t& r8, uint32_t& r9, uint32_t& r10, uint32_t& r11,
    uint32_t& r12, uint32_t& r13, uint32_t& r14, uint32_t& r15) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
        : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),
          "=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7),
          "=r"(r8),"=r"(r9),"=r"(r10),"=r"(r11),
          "=r"(r12),"=r"(r13),"=r"(r14),"=r"(r15) : "r"(col));
}

__device__ __forceinline__ void tmem_st_x16_bf16_fn(uint32_t lane_idx, uint32_t base_col,
    const uint32_t* vals) {
    // Store 16 FP32 values converted to BF16 starting at lane_idx, base_col
    for (int i = 0; i < 16; i += 4) {
        float f0 = __uint_as_float(vals[i + 0]);
        float f1 = __uint_as_float(vals[i + 1]);
        float f2 = __uint_as_float(vals[i + 2]);
        float f3 = __uint_as_float(vals[i + 3]);
        uint32_t bf0 = __float2bfloat16_raw(f0) | ((__float2bfloat16_raw(f1)) << 16);
        uint32_t bf1 = __float2bfloat16_raw(f2) | ((__float2bfloat16_raw(f3)) << 16);
        uint32_t col = base_col + i;
        asm volatile("tcgen05.st.sync.aligned.16x128b.x4.b32 [%0], {%1,%2,%3,%4};"
            :: "r"(col), "r"(bf0), "r"(bf1), "r"(0), "r"(0) : "memory");
    }
}

// ---- Host-side TMA descriptor creation ----

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, 
    uint64_t gmem_inner_dim, uint64_t gmem_outer_dim,
    uint64_t gmem_stride,  // stride in bytes between rows of outer dimension
    uint32_t smem_inner_dim, uint32_t smem_outer_dim,
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_stride};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
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
        l2Promotion,
        oobFill
    );
}

namespace tvm_ffi_gemm_blackwell {

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 64;
constexpr int WARPS = 4;
constexpr int THREADS_PER_WARP = 32;
constexpr int BLOCK_THREADS = WARPS * THREADS_PER_WARP; // 128

// Shared memory layout:
// We allocate shared memory for A and B tiles plus mbarriers
// A_smem: BLOCK_M x BLOCK_K in K-major (column-major-like): each column has BLOCK_M bf16 elements
// Actually for TMA loading, we specify the box and let TMA handle the layout
// A_smem layout after TMA: row-major [BLOCK_M][BLOCK_K] for simplicity
// But UMMA needs K-major, so we either relayout or use swizzled UMMA descriptors

// For simplicity, let's use cp.async.bulk (non-tensor) for loading with manual conversion
// Or better: use TMA with proper coordinate mapping

// Let's try a simpler approach: standard async copy + mbarrier + WGMMA

extern __shared__ char shared_mem[];

__global__ void gemm_blackwell_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    const __grid_constant__ CUtensorMap tma_C,
    __nv_bfloat16* A_global,
    __nv_bfloat16* B_global,
    __nv_bfloat16* C_global,
    uint64_t M, uint64_t N, uint64_t K) {

    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;
    const uint32_t warp_id = tid / THREADS_PER_WARP;
    const uint32_t lane_id = tid % THREADS_PER_WARP;

    // Allocate Tensor Memory for D accumulator: BM x BN fp32 = 64 x 64 x 4 bytes = 16KB
    // In TMEM: lanes x cols, we need 64 lanes x 64 cols minimum
    extern __shared__ char shared_mem[];
    
    uint32_t tmem_addr;
    uint32_t tmem_cols = 64; // alloc 64 columns
    if (tid == 0) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
            : "=r"(tmem_addr)
            : "r"((uint32_t)__cvta_generic_to_shared(shared_mem)), "r"(tmem_cols));
    }
    __syncthreads();
    
    // Only one thread issues alloc - actually all warps in the block should participate
    // Per spec: single warp in CTA for cta_group::1
    // Let's have warp 0 handle allocation
    if (warp_id == 0) {
        // Already done above if tid==0 within warp 0, but we need the whole warp
    }
    __syncthreads();

    // Compute output tile position
    const uint64_t out_row = blockIdx.y * BLOCK_M;
    const uint64_t out_col = blockIdx.x * BLOCK_N;

    // Initialize mbarrier for loads (thread 0 only)
    uint64_t* barrier = reinterpret_cast<uint64_t*>(shared_mem + 4096);
    if (tid == 0) {
        init_smem_barrier_fn(barrier, BLOCK_THREADS);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    // Wait for barrier init
    if (tid == 0) {
        mbarrier_wait_fn(barrier, 0);
    }
    __syncthreads();

    // Clear TMEM accumulator D = 0
    // Each thread clears a portion of the 64x64 fp32 accumulator
    // TMEM: lane_index (upper 16 bits) x col_index (lower 16 bits)
    // Total 64 lanes x 64 cols
    uint32_t my_lane = tid; // Map thread to TMEM lane
    if (my_lane < 64) {
        for (uint32_t c = 0; c < 64; c++) {
            uint32_t tmem_d_addr = (my_lane << 16) | c;
            asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], %1;"
                :: "r"(tmem_d_addr), "r"(0) : "memory");
        }
    }
    fence_proxy_async_fn();
    __syncthreads();

    // Pipeline: unroll over K
    const uint64_t num_k_tiles = K / BLOCK_K;

    for (uint64_t k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const uint64_t k_base = k_tile * BLOCK_K;

        // --- Load A tile: [out_row:out_row+BM, k_base:k_base+BK] ---
        // A is M x K row-major. TMA coordinates: (inner=k, outer=m)
        // coord0 = k_base, coord1 = out_row
        
        // Signal expectation for A load (BM * BK * 2 bytes) + B load
        uint32_t expected_bytes = BLOCK_M * BLOCK_K * 2 + BLOCK_N * BLOCK_K * 2;
        mbarrier_arrive_and_expect_tx_fn(barrier, expected_bytes);

        // Load A via TMA
        tma_load_2d_mbarrier_fn(&tma_A, barrier, shared_mem, static_cast<int32_t>(k_base), static_cast<int32_t>(out_row));
        
        // Load B tile: B is N x K, we need B.T[k_base:k_base+BK, out_col:out_col+BN]
        // B.T is K x N, so B.T[K, N] layout...
        // In memory B[N, K] row-major. B.T access pattern: B[j][i] maps to j*K+i
        // We want B.T[k:k+BK, col:col+BN] = B[col:col+BN, k:k+BK]^T
        // Coordinate mapping: inner=col, outer=k
        // coord0 = out_col, coord1 = k_base
        
        tma_load_2d_mbarrier_fn(&tma_B, barrier, shared_mem + BLOCK_M * BLOCK_K * 2, static_cast<int32_t>(out_col), static_cast<int32_t>(k_base));

        // Wait for loads to complete
        mbarrier_wait_fn(barrier, k_tile % 2);

        // Make data visible to async proxy
        fence_proxy_async_fn();

        // Copy A_smem [BM][BK] to TMEM 
        // UMMA expects K-major: A is [K, M] conceptually for K-major interpretation
        // Our shared memory stores A in TMA layout (we defined gmem_inner_dim=K, gmem_outer_dim=M)
        // So A_smem is [BK][BM] - K-major naturally
        // Similarly B_smem from TMA with inner=N, outer=K gives [BN][BK] 
        // But we need B.T in K-major = [BK, BN]
        
        // Actually let's trace through more carefully:
        // TMA for A: globalDim={K, M}, so tensorCoords(c0=k_base, c1=out_row) gives box [k_base..k_base+BK][out_row..out_row+BM]
        // A_smem layout: [BK][BM] in K-major ✓
        // TMA for B: globalDim={N, K}, coords(c0=out_col, c1=k_base) gives box [out_col..out_col+BN][k_base..k_base+BK]  
        // B_smem layout: [BN][BK] - this is B's natural layout [N_subset][K_subset]
        // But we need B.T which is [K_subset][N_subset] = [BK][BN]
        // So B_smem needs to be transposed into [BK][BN] for K-major UMMA consumption
        
        // This is tricky - let me reconsider the TMA setup
        // For B, instead of loading B[N,K] directly, I should load B.T[K,N]
        // B.T has shape [K, N]. If we define TMA on B.T virtually:
        // B.T[TMEM_k][TMEM_n] = B[TMEM_n][TMEM_k]
        // So loading B.T[k_base:k_base+BK, out_col:out_col+BN] means reading from B[out_col:out_col+BN, k_base:k_base+BK]
        // This is a strided read - TMA can handle this if we set up strides correctly
        
        // Simpler approach: just swap the dimensions in TMA descriptor for B
        // Treat B as having shape [K, N] for TMA purposes with appropriate stride
        // stride of B between N indices = K*2 bytes
        // stride between K indices = 2 bytes (contiguous within row)
        // Wait, B[N][K] means B[i] = &base[i*K], so B.T[i][j] = B[j][i]
        // Loading B.T[k:k+BK, col:col+BN]:
        //   TMA coords in virtual B.T space: (inner=n_offset, outer=k_offset)
        //   Maps to B[col+n_offset][k+k_offset]
        //   Global address = base + (k+k_offset)*stride_K + (col+n_offset)*stride_N
        //   where stride_K = K*2 bytes, stride_N = 2 bytes (within row)
        // Hmm, this doesn't map cleanly to TMA because B.T isn't contiguous in memory
        
        // Alternative: Use non-swizzled TMA to load B chunk [BN][BK], then transpose in registers/shared mem
        // Or: Create TMA descriptor that reads B with swapped logical dimensions
        
        // For now, let me use a simpler method: regular async bulk copy with manual tiling
        // Reset approach: use cp.async.bulk instead of TMA for simplicity
        
        // Actually, I realize the issue. Let me redo the TMA for B properly.
        // B[N, K] stored row-major. B.T[k, n] = B[n][k].
        // To load B.T[k_base:k_base+BK, out_col:out_col+BN]:
        // Virtual 2D array VT of shape [K, N] where VT[k][n] = B[n][k]
        // VT is NOT contiguous. B[n][k+1] = B[n][k] + 2 bytes, but VT[k+1][n] = B[n][k+1] 
        // VT[k][n+1] = B[n+1][k] which is K*2 bytes away.
        // So VT has stride_in_inner = K*2 bytes, stride_in_outer = 2 bytes (wait, that's wrong too)
        
        // VT[k][n] is at byte offset (n*k_elem + k)*2 = n*K*2 + k*2
        // If inner dim is n (size BN), outer dim is k (size BK):
        // VT[k][n] = base + k*2 + n*(K*2)  -- hmm non-standard
        // TMA expects: base + inner_coord * inner_stride + outer_coord * outer_stride  
        // With inner=n, outer=k:
        // offset = n * 2 + k * (K*2) ✓ matches!
        // So: gmem_inner_dim=N, inner_stride=2, gmem_outer_dim=K, outer_stride=K*2
        
        // BUT wait - cuTensorMap uses globalStrides[i] as stride between elements in dimension i
        // For 2D tensor with dims (dim0, dim1), globalStrides[0] = dim1 * elem_size
        // If I set globalDim = {N, K}, strides = {K*2}, then address(inner, outer) = base + inner*elem_size + outer*(K*2)
        // That gives B[outer][inner] mapping - exactly what we want for B.T!
        
        // So for B's TMA: globalDim={N, K}, but coords = (n_offset, k_offset)
        // Address = base + n_offset*2 + k_offset*(K*2) = B[k_offset][n_offset]*2? No wait...
        // B[row][col] = base + row*K*2 + col*2
        // We want VT(k, n) = B(n, k) = base + n*K*2 + k*2
        // Setting coords=(n, k) with inner=n, outer=k: address = base + n*2 + k*(K*2)
        // Hmm, that's base + 2*n + 2*K*k ≠ base + 2*K*n + 2*k unless...
        
        // Let me just set the strides explicitly. For TMA with globalDim={N, K}:
        // address(coord0, coord1) = base + coord0 * elem_size + coord1 * globalStrides[0]
        //                           = base + coord0 * 2 + coord1 * (K*2)
        // We want this to equal B[coord1][coord0] = base + coord1*K*2 + coord0*2 ✓✓✓
        
        // YES! So loading B.T[k_base:k_base+BK, out_col:out_col+BN] = VT[k_base.., out_col..]
        // With VT interpreted as 2D array with inner=columns_of_VT=N_direction, outer=rows_of_VT=K_direction:
        // Set coord0 = n_within_box (mapping to B's column index), coord1 = k_within_box (mapping to B's row index)
        // TMA coords: c0 = out_col + n_local, c1 = k_base + k_local
        // This puts data at smem[BK+local][BN+local] = [k_local][n_local] which IS B.T layout ✓
        
        // Hmm wait, I already have the TMA call: tma_load_2d(tma_B, ..., out_col, k_base)
        // Where tma_B has globalDim={N, K}. So coord0=out_col means inner-N direction, coord1=k_base means outer-K direction.
        // Box size {smem_inner_dim=BN, smem_outer_dim=BK}
        // Loaded shape in smem: [BN][BK] with smem[n_local][k_local] = B[k_base+k_local][out_col+n_local]
        // But we want B.T layout [BK][BN] for UMMA...
        
        // The issue is TMA always produces [inner][outer] layout = [BN][BK] 
        // But UMMA needs K-major [BK][BN] for the B matrix factor
        // Solution: Either transpose after load, OR redefine what "inner" and "outer" mean for B's TMA
        
        // Redefine: set B's TMA with globalDim={K, N} but use custom stride trick
        // Nope, that changes semantics too much. Let me just swap box dimensions.
        // If I set boxDim={BK, BN} and coords={k_base, out_col}:
        // TMA reads B.T[k_base+k_local][out_col+n_local] → mapped to smem[BK][BN] ✓
        // But wait, B.T is virtual. Actual access = B[out_col+n_local][k_base+k_local]
        // With globalDim={K, N}, stride={N*2}, address = base + k_local*2 + n_local*(N*2)
        // But B[row][col] = base + row*K*2 + col*2
        // These don't match: k_local*2 + n_local*N*2 ≠ n_local*K*2 + k_local*2 (off by N vs K)
        
        // OK I think the cleanest solution is to accept the [BN][BK] layout and adjust UMMA expectations.
        // Or better yet: use MN-major descriptors with swizzle where applicable.
        
        // Actually for UMMA with non-transpose, both A and B should be K-major.
        // A[K, M] K-major: A stored as [BK][BM] ✓ from TMA(inner=K-dir)
        // B[K, N] K-major: B stored as [BK][BN] ✗ we have [BN][BK]
        
        // Final solution: Transpose B tile in shared memory after loading.
        // Or even simpler: Define B's TMA such that it outputs [BK][BN] layout directly.
        // By swapping the TMA boxDim to {BK, BN} and coords to {k_base, out_col},
        // AND changing globalDim to {K, N} with stride computed for B's row-major layout.
        
        // For B[N, K] row-major, B.T virtual layout:
        // We want to load B.T[k_base:k_base+BK, out_col:out_col+BN]
        // Interpret as 2D with dims (K_part, N_part), reading B[n_from_N][k_from_K]
        // globalDim = {K, N}, globalStrides = {N * 2} (but B doesn't have stride N!)
        // B's real stride between rows is K*2. For virtual B.T, stride between "rows" (K-index) is K*2.
        // If inner=N_part, outer=K_part: addr = base + inner*2 + outer*K*2
        // Which gives B[outer][inner] ✓
        // But if I want boxDim = {BK, BN} meaning inner runs BK then outer runs BN...
        // That means inner is K-direction, outer is N-direction
        // addr = base + inner_k * 2 + outer_n * K*2 = B[outer_n][inner_k] = B.T[inner_k][outer_n] ✓✓✓
        
        // Wait, I think I was overcomplicating this. Let me restart.
        
        // For B's TMA to produce [BK][BN] (= B.T[BK][BN]):
        // globalDim = {K, N}, so coords = {k_coord, n_coord}
        // globalStrides[0] = ??? 
        // Default stride = next_dim * elem_size = N * 2
        // addr = base + k_coord * 2 + n_coord * (N*2)
        // But actual B address for B.T[k][n] = B[n][k] = base + n*K*2 + k*2
        // For these to match: k*2 + n*N*2 = n*K*2 + k*2 → N*2 = K*2 → N = K ✗ generally false
        
        // When N≠K, we CAN'T use default stride. We need globalStrides[0] = K*2.
        // Then addr = base + k*2 + n*(K*2) = base + k*2 + n*K*2 = B[n][k]*2 ✓✓✓ PERFECT!
        
        // So for B's TMA:
        // globalDim = {K, N}, globalStrides = {K*2}, boxDim = {BK, BN}, coords = {k_base, out_col}
        // Output layout: [BK][BN] with smem[k][n] = B.T[k][n] ✓
        
        // Great! Now I need to update the host code to set up tma_B with:
        // globalDim = {K, N}, globalStrides = {K*2}, boxDim = {BK, BN}
        // And tma_A with:
        // globalDim = {K, M}, globalStrides = {K*2}, boxDim = {BK, BM}
        // coords for A: {k_base, out_row} → smem[A][BK][BM]
        // coords for B: {k_base, out_col} → smem[B][BK][BN]
        
        // For UMMA K-major: A is [K, M], B is [K, N]
        // Descriptor A points to smem_A, LBO/SBO set for [BK][BM] K-major
        // Descriptor B points to smem_B, LBO/SBO set for [BK][BN] K-major
        
        // After loads complete, issue UMMA:
        // UMMA: D(BM, BN) += A_smem(K-major BK×BM) × B_smem(K-major BK×BN)
        // idesc: M=BM, N=BN, atype=BF16, btype=BF16, dtype=F32
        // K is fixed at 16 per UMMA issue, so loop BK/16 times
        
        // Copy A_smem to TMEM and issue UMMA
        // tmemory addresses: (lane << 16) | col
        
        // First, move A from shared mem to TMEM
        // A_smem is at shared_mem, layout [BK][BM] = [64][64]
        // Each 128-byte swizzle line spans 8 K-major atoms
        // Copy row by row using cp.async.bulk.tensor to TMEM? No, TMEM uses tcgen05.cp
        
        // Actually, let me take a step back. The simplest working approach:
        // Use tcgen05.cp to move from shared to TMEM, then tcgen05.mma
        // tcgen05.cp has limited shape options
        
        // Or even simpler: Issue UMMA directly from shared memory descriptors (no TMEM copy needed!)
        // tcgen05.mma.cta_group::1.kind::f16 [d-tmem], a-desc, b-desc, idesc, pred
        // a-desc and b-desc point to SHARED memory! TMEM is only for D accumulation.
        
        // Perfect! Just build sdescriptors for A_smem[BK][BM] and B_smem[BK][BN] in K-major
        // Then call umma_cg1 for each K-chunk of 16.
        
        uint32_t a_smem_offset = 0;
        uint32_t b_smem_offset = BLOCK_M * BLOCK_K * 2; // A takes BM*BK*2 bytes
        
        // Build descriptors for current K tile
        uint64_t sdesc_a = make_smem_desc_k_major_fn(
            shared_mem + a_smem_offset, BLOCK_M, BLOCK_K * 2);
        uint64_t sdesc_b = make_smem_desc_k_major_fn(
            shared_mem + b_smem_offset, BLOCK_N, BLOCK_K * 2);
        
        // UMMA instruction descriptor: M=BLOCK_M, N=BLOCK_N
        uint32_t instr = make_instr_desc_kk_fn(BLOCK_M, BLOCK_N);
        
        // D accumulator in TMEM starts at tmem_addr
        // Accumulate over K in chunks of 16
        uint32_t accum_flag = (k_tile == 0) ? 0 : 1;
        
        for (uint32_t ki = 0; ki < BLOCK_K / 16; ++ki) {
            // Adjust descriptors for this K-subtile
            // A subtile: start at ki*16 in K dimension, still BM in M
            // Shift a-desc base by ki*16*bias and update M dim
            // For K-major, advancing K-startpoint shifts the base address
            // Each 16-element K advancement = 16 * 2 bytes = 32 bytes = 8 doublewords
            
            uint32_t ki_bytes = ki * 16 * 2; // bytes to advance in K dimension
            uint32_t base_shift_dw = ki_bytes / 4; // shift in words for sdescriptor encoding
            
            // Recreate descriptors with shifted base addresses
            uint64_t sdesc_a_sub = make_smem_desc_k_major_fn(
                (char*)(shared_mem + a_smem_offset) + ki_bytes, BLOCK_M, BLOCK_K * 2);
            uint64_t sdesc_b_sub = make_smem_desc_k_major_fn(
                (char*)(shared_mem + b_smem_offset) + ki_bytes, BLOCK_N, BLOCK_K * 2);
            
            // Issue UMMA for M=BLOCK_M, N=BLOCK_N, K=16
            umma_f16_cg1_fn(tmem_addr, sdesc_a_sub, sdesc_b_sub, instr, accum_flag);
        }
        
        // Commit UMMA operations
        umma_commit_1sm_fn(barrier);
        
        // Wait for UMMA completion
        mbarrier_wait_fn(barrier, (k_tile + 1) % 2);
    }

    // Epilogue: Store D[TMEM 64x64 FP32] -> C[shared mem BF16] -> Global C
    // Each thread stores part of the result
    
    // Read TMEM and convert FP32 -> BF16, write to global memory
    uint32_t lane = tid; // Thread 0-127, map to TMEM lane 0-63 (4 threads per lane)
    uint32_t tmem_lane = lane % 64;
    uint32_t thread_in_lane = lane / 64; // 0, 1, or 2
    
    if (tmem_lane < 64) {
        // Load row tmem_lane from TMEM and convert/store
        for (uint32_t col = 0; col < 64; col += 2) {
            uint32_t r0, r1;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x2.b32 {%0,%1}, [%2];"
                : "=r"(r0), "=r"(r1) : "r"((tmem_lane << 16) | col));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            // Convert to BF16
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            __nv_bfloat16 bf0 = __float2bfloat16(f0);
            __nv_bfloat16 bf1 = __float2bfloat16(f1);
            
            // Store to global memory
            if (out_row + tmem_lane < M && out_col + col < N) {
                C_global[(out_row + tmem_lane) * N + (out_col + col)] = bf0;
                if (out_col + col + 1 < N) {
                    C_global[(out_row + tmem_lane) * N + (out_col + col + 1)] = bf1;
                }
            }
        }
    }

    // Deallocate Tensor Memory
    if (tid < 32) { // One warp deallocates
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
            :: "r"(tmem_addr), "r"(tmem_cols));
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    uint64_t M = A.size(0);
    uint64_t N = 7168;
    uint64_t K = 5120;
    
    const __nv_bfloat16* A_data = static_cast<const __nv_bfloat16*>(A.data_ptr());
    const __nv_bfloat16* B_data = static_cast<const __nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_data = static_cast<__nv_bfloat16*>(C.data_ptr());
    
    // Verify dimensions
    assert(B.size(0) == N);
    assert(B.size(1) == K);
    assert(C.size(0) == M);
    assert(C.size(1) == N);
    
    // Grid dimensions: x=N/BLOCK_N, y=M/BLOCK_M
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    dim3 block(BLOCK_THREADS);
    
    // Shared memory: A_smem + B_smem + barrier + extra for TMEM pointer
    // A: BM * BK * 2 = 64 * 64 * 2 = 8192
    // B: BN * BK * 2 = 64 * 64 * 2 = 8192  
    // Barrier: 8 bytes, alignment padding
    // Total ≈ 16KB + small overhead
    size_t smem_size = BLOCK_M * BLOCK_K * 2 + BLOCK_N * BLOCK_K * 2 + 1024;
    
    // Create TMA descriptor for A: globalDim={K, M}, load [BK][BM] K-major
    CUtensorMap tma_A;
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_A, const_cast<__nv_bfloat16*>(A_data),
        K, M, K * 2,  // gmem_inner=K, gmem_outer=M, stride=K*2
        BLOCK_K, BLOCK_M,  // boxDim: inner=BK, outer=BM
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    // Create TMA descriptor for B: globalDim={K, N}, load [BK][BN] K-major
    // B is stored as [N, K] row-major. B.T[k, n] = B[n, k]
    // To load B.T as [BK][BN]: treat as virtual [K, N] array
    // With coords(k, n), default stride N*2 wouldn't work, need stride=K*2
    CUtensorMap tma_B;
    cuuint64_t globalDim_B[2] = {K, N};
    cuuint64_t globalStrides_B[1] = {K * 2};  // Custom stride for B.T virtual view
    cuuint32_t boxDim_B[2] = {BLOCK_K, BLOCK_N};
    cuuint32_t elementStrides_B[2] = {1, 1};
    CU_CHECK(cuTensorMapEncodeTiled(&tma_B, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        const_cast<__nv_bfloat16*>(B_data), globalDim_B, globalStrides_B,
        boxDim_B, elementStrides_B,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    // No TMA store needed - we write directly to global memory in epilogue
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    // Launch kernel
    gemm_blackwell_kernel<<<grid, block, smem_size, stream>>>(
        tma_A, tma_B, *(CUtensorMap*)nullptr, A_data, B_data, C_data, M, N, K);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_gemm_blackwell::run);

}  // namespace tvm_ffi_gemm_blackwell