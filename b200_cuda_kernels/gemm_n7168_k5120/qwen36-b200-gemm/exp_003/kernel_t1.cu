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

// ---- Helper device functions ----

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

// Initialize mbarrier in shared memory
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

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void tma_store_fence_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
}

// Build SM100 shared memory descriptor
__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo, uint32_t base_offset, uint32_t swizzle_mode) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    // bits [13:0]: matrix start address encoded
    d |= ((uint64_t)(addr & 0x3FFFF)) >> 4;
    // bits [29:16]: LBO encoded
    d |= ((uint64_t)((lbo & 0x3FFFF))) << 12;
    // bits [45:32]: SBO encoded
    d |= ((uint64_t)((sbo & 0x3FFFF))) << 28;
    // bits [46:48]: fixed version = 1
    d |= (uint64_t)1 << 46;
    // bits [49:51]: base offset
    d |= (uint64_t)base_offset << 49;
    // bits [61:63]: swizzle mode
    d |= (uint64_t)swizzle_mode << 61;
    return d;
}

// Build UMMA instruction descriptor: BF16 x BF16 -> FP32
__device__ __forceinline__ uint32_t make_umma_idesc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);     // dtype = FP32 (bits 4-5)
    d |= (1u << 7);     // atype = BF16 (bits 7-9)
    d |= (1u << 10);    // btype = BF16 (bits 10-12)
    d |= (1u << 15);    // transpose A = true (row-major smem layout)
    d |= (1u << 16);    // transpose B = true (row-major smem layout)
    d |= ((N / 8) << 17);   // N dimension (bits 17-22)
    d |= ((M / 16) << 24);  // M dimension (bits 24-28)
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
__device__ __forceinline__ void umma_commit_2sm_fn(uint64_t* bar, uint16_t mask) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(a), "h"(mask));
}

// TMEM allocate/deallocate
__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
        :: "r"(addr), "r"(ncols));
}

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

namespace tvm_gemm_blackwell {

// Tile configuration
static constexpr int BM_PER_CTA = 128;   // Rows per CTA
static constexpr int BN_PER_CTA = 128;   // Cols per CTA  
static constexpr int BK = 16;            // K-tile size
static constexpr int CLUSTER_SIZE = 2;   // Cluster dim for cta_group::2
static constexpr int BLOCK_THREADS = 128;
static constexpr int M_TILE = BM_PER_CTA * CLUSTER_SIZE;  // 256
static constexpr int N_TILE = BN_PER_CTA * CLUSTER_SIZE;  // 256

// Shared memory layout per CTA (each CTA within cluster gets same layout):
// Total shared mem = ~72KB, well under limit
// Alignment: 256 bytes for safe swizzle access

__global__ void gemm_blackwell_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    uint32_t M,
    uint32_t N,
    uint32_t K)
{
    // Thread/block parameters
    uint32_t tid = threadIdx.x;
    uint32_t cta_rank = cluster_rank_fn();
    
    // Output tile coordinates for this CTA
    uint32_t m_block_start = blockIdx.x * M_TILE + cta_rank * BM_PER_CTA;
    uint32_t n_block_start = blockIdx.y * N_TILE;  // Both CTAs cover same N range
    
    // Clamp actual tile sizes
    uint32_t BM_actual = (m_block_start + BM_PER_CTA <= M) ? BM_PER_CTA : (M - m_block_start);
    if (BM_actual == 0) return;
    
    uint32_t BN_combined = (n_block_start + N_TILE <= N) ? N_TILE : (N - n_block_start);
    if (BN_combined == 0) return;
    
    // Shared memory partition (128-byte aligned)
    extern __shared__ char shmem_base[];
    
    // Align shared memory to 256-byte boundary for swizzle safety
    uintptr_t ptr = reinterpret_cast<uintptr_t>(shmem_base);
    ptr = (ptr + 255) & ~255ULL;
    char* aligned = reinterpret_cast<char*>(ptr);
    
    __nv_bfloat16* smem_A = reinterpret_cast<__nv_bfloat16*>(aligned);
    __nv_bfloat16* smem_B = smem_A + BM_PER_CTA * BK;
    __nv_bfloat16* smem_out = smem_B + N_TILE * BK;  // Staging buffer for epilogue: [BM][N_TILE] bf16
    uint64_t* mbar = reinterpret_cast<uint64_t*>(
        reinterpret_cast<char*>(smem_out) + BM_PER_CTA * N_TILE * sizeof(__nv_bfloat16));
    uint32_t* tmem_addr = reinterpret_cast<uint32_t*>(mbar + 1);
    
    // Phase 0: Initialize mbarrier (only thread 0)
    if (tid == 0) {
        init_smem_barrier_fn(mbar, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    // Allocate Tensor Memory: 128 lanes x 256 cols
    if (tid < 32 && elect_one_sync_fn()) {
        tmem_alloc_fn(tmem_addr, 256);
    }
    __syncthreads();
    
    uint32_t tmem_c_base = tmem_addr[0];
    
    // Build shared memory descriptors (no swizzle since transpose needs non-swizzled)
    // Layout A: [BM_PER_CTA][BK] row-major, stored as [BM][BK] bf16
    // For UMMA with transpose=true: expects [MN][K] i.e. [BM][BK] => correct
    // LBO = stride along minor (K) in 128-bit units = 16/16 = 1 normalized? 
    // Actually for non-swizzled:
    //   K-major (transposed view): ATOM_MMODE = BM, ATOM_KMODE = BK/sz = 8
    //   SBO = 8 * 16 = 128 (bytes), LBO = (ATOM_MMODE/8)*SBO = 16*128... 
    // Simpler: just use no-transpose descriptor and lay out physically transposed.
    // 
    // New plan: use transpose=false, lay out A as [BK][BM] and B as [BK][N_TILE].
    // This requires physical transpose during load.
    
    // Descriptor with no swizzle, K-major
    // For K-major no swizzle:
    //   span = 16B, ATOM_MMODE_DIM=BM, ATOM_KMODE_DIM=8
    //   SBO = 8*16 = 128, LBO = (BM/8)*SBO = 16*128 = 2048
    // But wait - we need the right formula. Let me just set LBO=SBO=something reasonable.
    // For simple case with no swizzle, both equal to leading_dim_in_bytes normalized.
    // 
    // Actually the simplest approach: swizzle_mode = 0, LBO = stride_minor/16, SBO = stride_major/16
    // For [K][M] K-major: stride_major(M) = 1*bfp16 = 1 unit, stride_minor(K) = BM*bfp16 = BM/8 units in 128-bit
    // For BF16: each element is 2 bytes, 128-bit = 64 bytes = 32 elements.
    // Leading dim = BM elements = BM*2 bytes = BM*2/16 normalized units = BM/8
    // Stride dim (minor/K) = 128-bit chunks = 1
    uint32_t lbo_val = 1;       // Minor stride (within K-dim) in 16B units
    uint32_t sbo_val = BK / 2;  // Major stride in 16B units (for BF16: BK/2 16-byte words)
    
    uint64_t desc_a = make_smem_desc_sm100_fn(smem_A, lbo_val, sbo_val, 0, 0);
    uint64_t desc_b = make_smem_desc_sm100_fn(smem_B, lbo_val, sbo_val, 0, 0);
    
    // Instruction descriptor: M=128, N=256 combined, transpose=true for row-major smem
    uint32_t idesc = make_umma_idesc_fn(BM_PER_CTA, N_TILE);
    
    // --- Main K-loop with pipelined G2S ---
    uint32_t num_k_tiles = (K + BK - 1) / BK;
    uint32_t phase = 0;
    uint32_t acc_flag = 0;
    
    // Prefetch first TMA setup is not needed here since we do direct loads
    
    for (uint32_t kt = 0; kt < num_k_tiles; ++kt) {
        uint32_t k_pos = kt * BK;
        uint32_t remaining_k = min(K - k_pos, static_cast<uint32_t>(BK));
        
        // Setup mbarrier for async completion tracking
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar, 0);
        }
        
        // --- Load A tile: [BM_actual][remaining_k] from A[m_block_start + :, k_pos + :] ---
        // Physical layout in smem: [remaining_k padded to BK][BM_actual] for K-major UMMA
        // Each thread pair handles 4 elements
        for (uint32_t m = tid; m < BM_actual; m += BLOCK_THREADS) {
            uint32_t src_row = m_block_start + m;
            for (uint32_t k_idx = 0; k_idx < remaining_k; k_idx += 4) {
                uint32_t src_col = k_pos + k_idx;
                // Store as [k][m] in smem_A: index = (k_idx + k_off) * BM + m
                // Smaller loop unroll
                for (uint32_t ko = 0; ko < 4; ++ko) {
                    if (src_col + ko < K) {
                        smem_A[(k_idx + ko) * BM_PER_CTA + m] = 
                            A[src_row * K + src_col + ko];
                    } else {
                        smem_A[(k_idx + ko) * BM_PER_CTA + m] = __float2bfloat16(0.0f);
                    }
                }
            }
        }
        
        // --- Load B tile: [N_TILE][remaining_k] from B[n_block_start + :, k_pos + :] ---
        // Physical layout in smem: [remaining_k padded to BK][N_TILE] for K-major UMMA
        for (uint32_t n = tid; n < N_TILE; n += BLOCK_THREADS) {
            uint32_t src_row = n_block_start + n;
            for (uint32_t k_idx = 0; k_idx < remaining_k; k_idx += 4) {
                uint32_t src_col = k_pos + k_idx;
                for (uint32_t ko = 0; ko < 4; ++ko) {
                    if (src_row < N && src_col + ko < K) {
                        smem_B[(k_idx + ko) * N_TILE + n] = 
                            B[src_row * K + src_col + ko];
                    } else {
                        smem_B[(k_idx + ko) * N_TILE + n] = __float2bfloat16(0.0f);
                    }
                }
            }
        }
        
        __syncthreads();
        
        fence_proxy_async_fn();
        
        // Issue UMMA: D[tmem] = A[smem_A] * B[smem_B] (+D)
        // cta_group::2: M=128 per CTA, N=256 combined
        umma_f16_cg2_fn(tmem_c_base, desc_a, desc_b, idesc, acc_flag);
        
        // Commit UMMA to mbarrier
        if (tid == 0) {
            umma_commit_2sm_fn(mbar, 0x3);  // Mask for both CTAs in cluster
        }
        
        // Wait for completion
        mbarrier_wait_fn(mbar, phase);
        phase++;
        acc_flag = 1;  // Subsequent iterations accumulate
    }
    
    // --- Epilogue: TMEM -> Global ---
    tmem_ld_fence_fn();
    
    // Stage FP32 results to shared memory as BF16
    // Each thread reads one row from TMEM (thread x -> lane)
    uint32_t tm_lane = tid;  // Thread maps directly to TMEM lane
    
    if (tm_lane < BM_actual) {
        // Load all N_TILE columns for this row
        for (uint32_t col = 0; col < N_TILE; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_ld_4x32b_fn(tmem_c_base + col, r0, r1, r2, r3);
            
            uint32_t base = tm_lane * N_TILE + col;
            if (m_block_start + tm_lane < M && n_block_start + col < N) {
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
    }
    
    tmem_ld_fence_fn();
    __syncthreads();
    
    // Coalesced global writes: warp-strided pattern
    uint32_t warp_id = tid / 32;
    uint32_t lane_id = tid % 32;
    
    for (uint32_t step = 0; step < (BM_actual + 3) / 4; ++step) {
        uint32_t local_row = step * 4 + warp_id;
        if (local_row >= BM_actual) continue;
        
        uint32_t global_row = m_block_start + local_row;
        uint32_t col_start = lane_id * 4;
        uint32_t global_col = n_block_start + col_start;
        
        if (global_row < M && global_col + 3 < N) {
            uint2 data;
            data.x = reinterpret_cast<uint2*>(&smem_out[local_row * N_TILE])[col_start / 2].x;
            data.y = reinterpret_cast<uint2*>(&smem_out[local_row * N_TILE])[col_start / 2].y;
            *(reinterpret_cast<uint2*>(&C[global_row * N + global_col])) = data;
        }
    }
    
    // Deallocate Tensor Memory
    if (tid < 32 && elect_one_sync_fn()) {
        tmem_dealloc_fn(tmem_addr[0], 256);
    }
    __syncthreads();
    
    tmem_relinquish_fn();
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    uint32_t M = static_cast<uint32_t>(A.size(0));
    uint32_t N = static_cast<uint32_t>(B.size(0));  // Reference computes A @ B.T, so B is [N][K]
    uint32_t K = static_cast<uint32_t>(A.size(1));
    
    const __nv_bfloat16* A_data = static_cast<const __nv_bfloat16*>(A.data_ptr());
    const __nv_bfloat16* B_data = static_cast<const __nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_data = static_cast<__nv_bfloat16*>(C.data_ptr());
    
    // Grid: blocks cover M_TILE x N_TILE regions
    uint32_t grid_x = (M + M_TILE - 1) / M_TILE;
    uint32_t grid_y = (N + N_TILE - 1) / N_TILE;
    
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(BLOCK_THREADS, 1, 1);
    
    // Shared memory calculation:
    // smem_A: [BK][BM] bf16 = 16*128*2 = 4KB
    // smem_B: [BK][N_TILE] bf16 = 16*256*2 = 8KB  
    // smem_out: [BM][N_TILE] bf16 = 128*256*2 = 64KB
    // mbarrier + tmem_addr: 12 bytes + pad
    // Total per CTA: ~76KB
    size_t shmem_size = 
        BK * BM_PER_CTA * sizeof(__nv_bfloat16) +      // smem_A
        BK * N_TILE * sizeof(__nv_bfloat16) +          // smem_B
        BM_PER_CTA * N_TILE * sizeof(__nv_bfloat16) +  // smem_out
        256;  // mbarrier + padding
    shmem_size = (shmem_size + 255) & ~255ULL;
    
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
                       A_data, B_data, C_data, M, N, K);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_gemm_blackwell::run);

}  // namespace tvm_gemm_blackwell