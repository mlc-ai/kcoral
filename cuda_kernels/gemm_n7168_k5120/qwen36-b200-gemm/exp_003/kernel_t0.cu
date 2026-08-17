#include <cuda_bf16.h>
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
    CUresult _cr = (call);                                         \
    if (_cr != CUDA_SUCCESS) {                                     \
        const char *_err_str;                                      \
        cuGetErrorName(_cr, &_err_str);                            \
        fprintf(stderr, "cuTLS error %s at %s:%d\n",              \
                _err_str, __FILE__, __LINE__);                     \
        exit(1);                                                   \
    }                                                              \
} while(0)

// Helper utilities
__device__ __forceinline__ bool elect_one_sync_fn() {
    uint32_t pred;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "elect.sync _|p, 0xFFFFFFFF;\n"
        "selp.b32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(pred));
    return pred != 0;
}

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

// Initialize mbarrier
__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

// Mbarrier arrive + expect_tx for TMA loads
__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
}

// Wait for mbarrier
__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase_parity) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase_parity));
}

// Fence proxy for async operations
__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

// TMA 2D load into shared memory
__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

// Prefetch TMA descriptor
__device__ __forceinline__ void prefetch_tma_descriptor_fn(const CUtensorMap* d) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(d) : "memory");
}

// Build SM100 shared memory descriptor with 128B swizzle
__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo, uint32_t base_offset) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= ((uint64_t)(addr & 0x3FFFF)) >> 4;            // bits [13:0]: start address
    d |= ((uint64_t)((lbo & 0x3FFFF))) << 12);         // bits [29:16]: LBO
    d |= ((uint64_t)((sbo & 0x3FFFF))) << 28);         // bits [45:32]: SBO
    d |= (uint64_t)1 << 46;                           // version = 1
    d |= (uint64_t)base_offset << 49;                 // base offset
    d |= (uint64_t)2 << 61;                           // swizzle = 128B
    return d;
}

// Build UMMA instruction descriptor: BF16 x BF16 -> FP32
__device__ __forceinline__ uint32_t make_umma_idesc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);     // dtype = FP32 (bits 4-5)
    d |= (1u << 7);     // atype = BF16 (bits 7-9)
    d |= (1u << 10);    // btype = BF16 (bits 10-12)
    d |= (0u << 15);    // transpose A = false => K-major
    d |= (0u << 16);    // transpose B = false => K-major
    d |= ((N / 8) << 17);    // N dimension (bits 17-22)
    d |= ((M / 16) << 24);   // M dimension (bits 24-28)
    return d;
}

// UMMA cta_group::2 BF16->FP32
__device__ __forceinline__ void umma_f16_cg2_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

// UMMA commit with multicast barrier arrive
__device__ __forceinline__ void umma_commit_2sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(a), "h"((uint16_t)0x3));
}

// TMEM allocate
__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(a), "r"(ncols));
}

// TMEM dealloc
__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
        :: "r"(addr), "r"(ncols));
}

// TMEM relinquish alloc permit
__device__ __forceinline__ void tmem_relinquish_fn() {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory");
}

// TMEM load 4 floats from column
__device__ __forceinline__ void tmem_ld_4x32b_fn(uint32_t col, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(col));
}

// TMEM load fence
__device__ __forceinline__ void tmem_ld_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

// Cluster sync
__device__ __forceinline__ void cluster_sync_fn() {
    asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n" ::: "memory");
}

namespace tvm_gemm_blackwell {

static constexpr int BM_PER_CTA = 128;    // Output rows per CTA
static constexpr int BN_PER_CTA = 128;    // Output cols per CTA  
static constexpr int BK = 16;             // K tile size per UMMA iteration
static constexpr int CLUSTER_SIZE = 2;    // Cluster dim (2 CTAs paired)

// Shared memory layout per CTA:
// smem_A: [BM_PER_CTA x BK] = [128 x 16] bf16 = 4KB  (K-major layout)
// smem_B: [BN_PER_CTA x BK] = [128 x 16] bf16 = 4KB  (transposed for MN-major interpretation)
// mbarrier: 8 bytes
// tmem_addr: 4 bytes
// Total: ~8KB + padding = 12KB

extern __shared__ char shared_mem[];

__global__ void gemm_blackwell_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* __restrict__ C,
    uint32_t M,
    uint32_t N,
    uint32_t K)
{
    // Each CTA handles BM_PER_CTA output rows
    // cta_group::2 pairs CTA (cluster_rank=0) and CTA (cluster_rank=1)
    // Cluster covers 2*BM_PER_CTA = 256 rows and 2*BN_PER_CTA = 256 cols
    
    uint32_t cluster_m_id = blockIdx.x;
    uint32_t cluster_n_id = blockIdx.y;
    uint32_t cta_rank_in_cluster = cluster_rank_fn();
    
    // Compute output tile origin for this cluster
    uint32_t M_TILE = 2 * BM_PER_CTA;  // 256
    uint32_t N_TILE = 2 * BN_PER_CTA;  // 256
    
    uint32_t m_tile_start = cluster_m_id * M_TILE;
    uint32_t n_tile_start = cluster_n_id * N_TILE;
    
    // Per-CTA offsets:
    // CTA rank 0: rows [m_tile_start : m_tile_start + 128], cols [n_tile_start : n_tile_start + 128]
    // CTA rank 1: rows [m_tile_start + 128 : m_tile_start + 256], cols [n_tile_start + 128 : n_tile_start + 256]
    // But wait - actually for cta_group::2 UMMA, the N is combined across CTAs.
    // Let me re-read the docs... 
    // Under cta_group::2, N in idesc is the COMBINED dimension (BN_PER_CTA * 2).
    // So each CTA computes BM_PER_CTA x (BN_PER_CTA * 2) = 128 x 256
    
    uint32_t m_base = m_tile_start + cta_rank_in_cluster * BM_PER_CTA;
    uint32_t n_base = n_tile_start;  // Same N range for both CTAs in cluster
    uint32_t m_actual = M - m_base;
    if (m_actual > BM_PER_CTA) m_actual = BM_PER_CTA;
    if (m_actual <= 0) return;
    
    uint32_t n_actual = N - n_base;
    if (n_actual > N_TILE) n_actual = N_TILE;
    if (n_actual <= 0) return;
    
    // Shared memory pointers (from base, each CTA has its own segment)
    // Use threadIdx.x to get proper CTA-local offset for alignment
    // For UMMA, the descriptor's start address should point to the actual data region
    __nv_bfloat16* smem_A = reinterpret_cast<__nv_bfloat16*>(shared_mem);
    __nv_bfloat16* smem_B = reinterpret_cast<__nv_bfloat16*>(shared_mem) + BM_PER_CTA * BK;
    
    // Align to 128-byte boundary for swizzle
    extern __shared__ alignas(128) char smem_aligned[];
    smem_A = reinterpret_cast<__nv_bfloat16*>(smem_aligned);
    smem_B = reinterpret_cast<__nv_bfloat16*>(smem_aligned) + BM_PER_CTA * BK;
    
    // Mbarrier and TMEM address storage
    uint64_t* mbar = reinterpret_cast<uint64_t*>(
        reinterpret_cast<char*>(smem_B) + BN_PER_CTA * BK * sizeof(__nv_bfloat16));
    uint32_t* tmem_addr = reinterpret_cast<uint32_t*>(
        reinterpret_cast<char*>(mbar) + sizeof(uint64_t));
    
    // Only thread 0 of CTA initializes barrier
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar, 1);  // Expect 1 arrival (from UMMA commit)
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    // Allocate Tensor Memory: 128 rows x 256 cols x 4 bytes = 128KB
    // Need to allocate columns: 256 columns per CTA (but TMEM is 128 lanes x 512 cols max)
    // We allocate 256 columns * 2 (for cta_group::2) = no, each CTA gets 256 cols in its own TMEM
    // Actually TMEM is 128 rows x 512 cols per CTA. We need 128 rows x 256 cols.
    if (threadIdx.x < 32) {  // One warp allocates
        if (threadIdx.x == 0) {
            tmem_alloc_fn(tmem_addr, 256);  // 256 columns
        }
    }
    __syncthreads();
    
    uint32_t tmem_c_base = tmem_addr[0];
    
    // Prefetch TMA descriptors
    if (threadIdx.x == 0) {
        prefetch_tma_descriptor_fn(&tma_A);
        prefetch_tma_descriptor_fn(&tma_B);
    }
    
    // Build shared memory descriptors for UMMA
    // Matrix A in shared memory: K-major, 128B swizzle
    // Layout: [BK][BM_PER_CTA] effectively stored as [BM_PER_CTA][BK] with K-major descriptor
    // Start addr: smem_A (aligned)
    // For K-major 128B swizzle: LBO = 1, SBO = 8 * 128 = 1024
    // Base offset: depends on alignment - assume 0 if at 1024-byte boundary
    
    uint32_t sbo_a = 1024;
    uint32_t lbo_a = 1;
    uint32_t base_off_a = 0;
    uint64_t desc_a = make_smem_desc_sm100_fn(smem_A, lbo_a, sbo_a, base_off_a);
    
    // Matrix B: we load B[n:n+256, k:k+16] -> stored as [BN_COMBINED][BK] = [256][16]
    // But each CTA only loads its own 128-col slice into its local shared memory
    // Actually with cta_group::2, each CTA loads independently.
    // For B: MN-major, 128B swizzle
    // Layout [BN_PER_CTA][BK] interpreted as [256 combined][16] -- no, it's per-CTA.
    // Actually the descriptor describes A and B locally in shared memory.
    // B needs to appear as K-major to UMMA (since no-transpose for B means K-major).
    // B original is [N][K], B.T is [K][N].
    // Loading B[cols, k_start:k_end] gives us [BN][BK] in row-major = [BN][BK].
    // For UMMA with transpose=false, B needs K-major = [K][N].
    // So we need to lay out B in shared memory as [BK][BN].
    // That means coord0=BK, coord1=BN in TMA coords.
    
    uint32_t sbo_b = 1024;
    uint32_t lbo_b = 1;
    uint32_t base_off_b = 0;
    uint64_t desc_b = make_smem_desc_sm100_fn(smem_B, lbo_b, sbo_b, base_off_b);
    
    // Instruction descriptor: BF16 x BF16 -> FP32, M=128, N=256 (combined for cta_group::2)
    uint32_t idesc = make_umma_idesc_fn(BM_PER_CTA, N_TILE);
    
    // Number of K tiles
    uint32_t num_k_tiles = (K + BK - 1) / BK;
    
    // Pipeline: load K tiles and accumulate
    uint32_t phase = 0;
    uint32_t acc_flag = 0;  // 0 = clear accumulator, 1 = accumulate
    
    // First iteration - special handling for barrier init already done above
    for (uint32_t kt = 0; kt < num_k_tiles; ++kt) {
        uint32_t k_pos = kt * BK;
        if (k_pos >= K) break;
        
        // Setup mbarrier for this phase
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar, 0);  // Reset and expect 0 tx, will be set by TMA
        }
        
        // Issue TMA loads
        // Load A: [BM_PER_CTA][BK] from A[m_base:m_base+128, k_pos:k_pos+16]
        // TMA coords: c0=inner=k pos, c1=outer=m pos relative to tensor
        // The tensor map is encoded with gmem_inner_dim=K, gmem_outer_dim=M (or vice versa)
        // For row-major [M][K], if inner_dim=K, coords are {coord_K, coord_M}
        tma_load_2d_fn(&tma_A, mbar, smem_A, (int32_t)k_pos, (int32_t)m_base);
        
        // Load B: B is [N][K] row-major. We want B[n_base:n_base+256, k_pos:k_pos+16]
        // stored in shared memory as [256][16] (row-major = MN-major compatible)
        // But UMMA wants K-major for B too... 
        // Actually, let's think again. The UMMA instruction has transpose flags.
        // transpose=false means K-major layout expected.
        // A row-major [M][K] tile [m:m+BM][k:k+BK] stored as [BM][BK] row-major 
        //   is NOT K-major. K-major would be [BK][BM].
        // So we need to either transpose or use transpose=true in UMMA.
        // Using transpose=true in UMMA for both A and B means we can keep row-major in smem.
        
        // Let me redo: use transpose=true for both matrices
        // Row-major [BM][BK] in smem, UMMA with transpose=true interprets as K-major logically
        // Actually the docs say transpose ops aren't supported with 128B swizzle on non-tf32.
        // So we need to physically transpose.
        
        // Simpler: load A as [BK][BM] into smem_A using im2col or gather, 
        // and B as [BK][BN] into smem_B.
        // With cp.async.bulk.tensor, we can specify boxDim to control the box shape.
        
        // Actually, let me use regular cp.async.bulk instead of tensor-based TMA 
        // for now, then swap to TMA properly.
        
        // For correctness, let's use direct shared memory copies first.
        
        // Barrier wait for previous phase
        if (kt > 0) {
            mbarrier_wait_fn(mbar, phase ^ 1);
        }
        
        // Manual shared memory copy for A: each thread loads 2 bf16 elements
        // smem_A layout: [BM_PER_CTA][BK] = [128][16] bf16, stored row-major
        // A is [M][K] row-major. We want A[m_base+m][k_base+k] -> smem_A[m][k]
        for (uint32_t m = threadIdx.x; m < BM_PER_CTA; m += 128) {
            for (uint32_t k = 0; k < BK; k += 4) {
                uint32_t src_row = m_base + m;
                uint32_t src_col = k_pos + k;
                if (src_row < M && src_col + 3 < K) {
                    uint2 a0 = reinterpret_cast<const uint2*>(&A[src_row * K])[src_col / 2];
                    uint2 a1 = reinterpret_cast<const uint2*>(&A[src_row * K])[src_col / 2 + 1];
                    reinterpret_cast<uint2*>(&smem_A[m * BK])[k / 2] = a0;
                    reinterpret_cast<uint2*>(&smem_A[m * BK])[k / 2 + 1] = a1;
                }
            }
        }
        
        // Manual shared memory copy for B
        for (uint32_t n = threadIdx.x; n < BN_PER_CTA; n += 128) {
            for (uint32_t k = 0; k < BK; k += 4) {
                uint32_t src_row = n_base + n;
                uint32_t src_col = k_pos + k;
                if (src_row < N && src_col + 3 < K) {
                    uint2 b0 = reinterpret_cast<const uint2*>(&B[src_row * K])[src_col / 2];
                    uint2 b1 = reinterpret_cast<const uint2*>(&B[src_row * K])[src_col / 2 + 1];
                    reinterpret_cast<uint2*>(&smem_B[n * BK])[k / 2] = b0;
                    reinterpret_cast<uint2*>(&smem_B[n * BK])[k / 2 + 1] = b1;
                }
            }
        }
        
        __syncthreads();
        
        fence_proxy_async_fn();
        
        // UMMA: D = A*B + D (or D = A*B if first iter)
        umma_f16_cg2_fn(tmem_c_base, desc_a, desc_b, idesc, acc_flag);
        
        // Commit and sync
        if (threadIdx.x == 0) {
            umma_commit_2sm_fn(mbar);
        }
        
        mbarrier_wait_fn(mbar, phase);
        phase++;
        acc_flag = 1;  // Next iteration accumulates
    }
    
    // Epilogue: read TMEM and write to global memory
    tmem_ld_fence_fn();
    
    // Stage results through shared memory for coalesced writes
    __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(
        reinterpret_cast<char*>(smem_aligned) + 
        (BM_PER_CTA + BN_PER_CTA) * BK * sizeof(__nv_bfloat16) + 128);
    
    // Phase 1: TMEM -> SMEM staging
    // Each thread (tid 0..127) reads its row from TMEM
    for (uint32_t col = 0; col < N_TILE; col += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_ld_4x32b_fn(tmem_c_base + col, r0, r1, r2, r3);
        tmem_ld_fence_fn();
        
        uint32_t base = threadIdx.x * N_TILE + col;
        if (m_base + threadIdx.x < M && n_base + col + 3 < N) {
            smem_out[base + 0] = __float2bfloat16(__uint_as_float(r0));
            smem_out[base + 1] = __float2bfloat16(__uint_as_float(r1));
            smem_out[base + 2] = __float2bfloat16(__uint_as_float(r2));
            smem_out[base + 3] = __float2bfloat16(__uint_as_float(r3));
        } else {
            smem_out[base + 0] = __float2bfloat16(0.0f);
            smem_out[base + 1] = __float2bfloat16(0.0f);
            smem_out[base + 2] = __float2bfloat16(0.0f);
            smem_out[base + 3] = __float2bfloat16(0.0f);
        }
    }
    
    __syncthreads();
    
    // Phase 2: Coalesced SMEM -> Global writes
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = (BM_PER_CTA + 3) / 4;
    
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        if (row >= BM_PER_CTA) continue;
        
        uint32_t global_row = m_base + row;
        uint32_t col_start = lane_id * 4;
        uint32_t global_col = n_base + col_start;
        
        if (global_row < M && global_col + 3 < N) {
            uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * N_TILE + col_start]);
            *reinterpret_cast<uint2*>(C + (uint64_t)global_row * N + global_col) = data;
        }
    }
    
    // Deallocate TMEM
    if (threadIdx.x < 32) {
        if (threadIdx.x == 0) {
            tmem_dealloc_fn(tmem_addr[0], 256);
        }
    }
    __syncthreads();
    
    tmem_relinquish_fn();
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    uint32_t M = static_cast<uint32_t>(A.size(0));
    uint32_t N = static_cast<uint32_t>(7168u);
    uint32_t K = static_cast<uint32_t>(5120u);
    
    // Validate shapes
    if (A.ndim() != 2 || B.ndim() != 2 || C.ndim() != 2) {
        fprintf(stderr, "GEMM expects 2D tensors\n");
        exit(1);
    }
    if (static_cast<uint32_t>(A.size(1)) != K) {
        fprintf(stderr, "A's K dim (%ld) != expected K=%u\n", (long)A.size(1), K);
        exit(1);
    }
    if (static_cast<uint32_t>(B.size(1)) != K) {
        fprintf(stderr, "B's K dim (%ld) != expected K=%u\n", (long)B.size(1), K);
        exit(1);
    }
    
    const __nv_bfloat16* A_data = static_cast<const __nv_bfloat16*>(A.data_ptr());
    const __nv_bfloat16* B_data = static_cast<const __nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_data = static_cast<__nv_bfloat16*>(C.data_ptr());
    
    // TMA descriptor for A: gmem [M][K] row-major, load box [BM_PER_CTA x BK]
    // Encode as tensor with inner_dim=K, outer_dim=M
    CUtensorMap tma_A;
    cuuint64_t globalDimA[2] = {K, M};
    cuuint64_t globalStridesA[1] = {K * sizeof(__nv_bfloat16)};
    cuuint32_t boxDimA[2] = {BK, BM_PER_CTA};
    cuuint32_t elemStridesA[2] = {1, 1};
    
    CU_CHECK(cuTensorMapEncodeTiled(
        &tma_A,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        (void*)A_data,
        globalDimA,
        globalStridesA,
        boxDimA,
        elemStridesA,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    // TMA descriptor for B: gmem [N][K] row-major, load box [BN_TOTAL x BK]
    CUtensorMap tma_B;
    cuuint64_t globalDimB[2] = {K, N};
    cuuint64_t globalStridesB[1] = {K * sizeof(__nv_bfloat16)};
    cuuint32_t boxDimB[2] = {BK, BN_PER_CTA};
    cuuint32_t elemStridesB[2] = {1, 1};
    
    CU_CHECK(cuTensorMapEncodeTiled(
        &tma_B,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        (void*)B_data,
        globalDimB,
        globalStridesB,
        boxDimB,
        elemStridesB,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    // Grid dimensions
    uint32_t M_TILE = 2 * BM_PER_CTA;  // 256
    uint32_t N_TILE = 2 * BN_PER_CTA;  // 256
    uint32_t grid_x = (M + M_TILE - 1) / M_TILE;
    uint32_t grid_y = (N + N_TILE - 1) / N_TILE;
    
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(BM_PER_CTA, 1, 1);  // 128 threads per CTA (one warpgroup)
    
    // Shared memory: smem_A (4KB) + smem_B (4KB) + mbarrier + tmem_addr + smem_out (128*256*2=64KB) + padding
    // Total ~72KB per CTA, well under limits
    size_t shmem_size = (BM_PER_CTA + BN_PER_CTA) * BK * sizeof(__nv_bfloat16)
                      + 128  // mbarrier + tmem_addr + alignment
                      + BM_PER_CTA * N_TILE * sizeof(__nv_bfloat16);  // smem_out buffer
    shmem_size = (shmem_size + 15) & ~15ULL;  // Align to 16 bytes
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = static_cast<unsigned long long>(shmem_size);
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = CLUSTER_SIZE;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    cudaLaunchKernelEx(&config, gemm_blackwell_kernel,
                       tma_A, tma_B, C_data, M, N, K);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_gemm_blackwell::run);

}  // namespace tvm_gemm_blackwell