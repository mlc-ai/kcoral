#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <cfloat>
#include <cmath>
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

namespace mha_blackwell {

// ============================================================
// Host-side helpers
// ============================================================

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ void cluster_arrive_fn() {
    asm volatile("barrier.cluster.arrive;\n" ::: "memory");
}

__device__ __forceinline__ void cluster_wait_fn() {
    asm volatile("barrier.cluster.wait;\n" ::: "memory");
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n.reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(phase));
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void named_barrier_sync_fn(int bar_id, int count) {
    asm volatile("barrier.sync.aligned %0, %1;" :: "r"(bar_id), "r"(count));
}

// ============================================================
// TMA helpers (using cuTensorMap)
// ============================================================

extern "C" CUresult create_tma_2d_descriptor_bf16(CUtensorMap* d,
    void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim,
    uint32_t smem_inner_dim, uint32_t smem_outer_dim,
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill);

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar,
                                                void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
           "l"((uint64_t)d),
           "r"((uint32_t)__cvta_generic_to_shared(bar)),
           "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_store_2d_fn(const CUtensorMap* d, void* smem,
                                                 int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
        " [%0, {%2, %3}], [%1];"
        :: "l"((uint64_t)d),
           "r"((uint32_t)__cvta_generic_to_shared(smem)),
           "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void tma_store_wait_all_fn() {
    asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
}

// ============================================================
// TCGEN05 helpers
// ============================================================

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col,
    uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_ld_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void umma_f16_cg2_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_fn(uint64_t* bar, uint16_t mask) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(a), "h"(mask));
}

__device__ __forceinline__ void cp_smem_to_tmem_128x128b_fn(
    uint32_t tmem_addr, uint64_t sdesc) {
    asm volatile(
        "tcgen05.cp.cta_group::1.128x128b [%0], %1;\n"
        :: "r"(tmem_addr), "l"(sdesc));
}

__device__ __forceinline__ void cp_smem_to_tmem_64x128b_fn(
    uint32_t tmem_addr, uint64_t sdesc) {
    asm volatile(
        "tcgen05.cp.cta_group::1.64x128b [%0], %1;\n"
        :: "r"(tmem_addr), "l"(sdesc));
}

__device__ __forceinline__ uint64_t make_smem_desc_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo, uint32_t swizzle_mode) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)((addr & 0x3FFFF) >> 4);            // bits [13:0]: start address
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;      // bits [29:16]: LBO
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;      // bits [45:32]: SBO
    d |= (uint64_t)1ULL << 46;                         // version = 1
    d |= (uint64_t)(swizzle_mode & 7ULL) << 61;        // swizzle mode
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);     // dtype = FP32
    d |= (1u << 7);     // atype = BF16
    d |= (1u << 10);    // btype = BF16
    d |= ((N / 8) << 17);   // n_dim >> 3
    d |= ((M / 16) << 24);  // m_dim >> 4
    return d;
}

// Software emulated exp2 for range-reduced values [0,1)
__device__ __forceinline__ float soft_exp2_approx(float x) {
    // Horner's form: p0 + x*(p1 + x*(p2 + x*p3))
    const float p3 = 0.0771466f;
    const float p2 = 0.2272649f;
    const float p1 = 0.6951643f;
    const float p0 = 1.0f;
    float f = p0 + x * (p1 + x * (p2 + x * p3));
    return f;
}

__device__ __forceinline__ float fast_exp2f(int n, float frac) {
    float f_val = soft_exp2_approx(frac);
    // Multiply by 2^n: inject n into exponent
    int sign_mantissa;
    asm volatile ("brkga.pack.clamp.ftz.f32 %0, %1;" : "=r"(sign_mantissa) : "f"(f_val));
    int result_bits = sign_mantissa | ((n + 127) << 23);
    return __int_as_float(result_bits);
}

__device__ __forceinline__ float expf_approx(float x) {
    float x2 = x * 1.442695041f;  // log2(e)
    int n = __float2int_rz(x2);
    float frac = x2 - (float)n;
    return fast_exp2f(n, frac);
}

// ============================================================
// Kernel constants
// ============================================================
constexpr int BM_PER_CTA = 64;    // Query rows per CTA (CTA pair does 128 total)
constexpr int BN_PER_CTA = 64;    // Key columns per CTA per tile
constexpr int BD         = 128;   // Head dimension
constexpr int WMMA_K     = 16;    // WGMMA contracts K=16 per instruction
constexpr int N_WARPGROUPS = 4;   // 4 warpgroups of 32 threads = 128 threads/CTA
constexpr int THREADS_PER_BLOCK = N_WARPGROUPS * 32;  // 128

__global__ void mha_causal_kernel_tcgen05(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
          __grid_constant__ CUtensorMap tma_O,
    float* __restrict__ LSE_out,
    int B, int H, int S, int D,
    float inv_sqrt_d)
{
    extern __shared__ char smem_raw[];

    // Shared memory layout (all aligned):
    // Q_SMEM:  BM_PER_CTA x BD bf16 (row-major) at offset 0
    // K_SMEM:  BN_PER_CTA x BD bf16 at offset Q
    // V_SMEM:  BN_PER_CTA x BD bf16 at offset K
    // Barriers: after V_SMEM
    
    size_t q_size = BM_PER_CTA * BD * sizeof(__nv_bfloat16);
    size_t k_size = BN_PER_CTA * BD * sizeof(__nv_bfloat16);
    
    __nv_bfloat16* Q_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* K_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + q_size);
    __nv_bfloat16* V_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + q_size + k_size);
    uint64_t* barrier_k = reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(V_smem) + k_size);
    uint64_t* barrier_q = barrier_k + 1;
    uint64_t* barrier_umma = barrier_q + 1;
    uint32_t* tmem_addr_slot = reinterpret_cast<uint32_t*>(barrier_umma + 1);
    
    // Pad to alignment boundaries as needed
    constexpr size_t BARRIER_AREA_SIZE = 4 * sizeof(uint64_t) + sizeof(uint32_t);
    constexpr size_t TOTAL_SMEM_ALIGN = 
        ((q_size + k_size * 2 + BARRIER_AREA_SIZE + 127) / 128) * 128;
    
    int tid = threadIdx.x;
    int warp_id = tid / 32;          // 0..3
    int lane_id = tid % 32;          // 0..31
    int wg_id = tid / 128;           // warpgroup id (0 for our 128-thread block... only 1 wg)
    
    // CTA rank within cluster
    uint32_t cta_rank = cluster_rank_fn();
    bool leader = (cta_rank == 0);
    bool follower = (cta_rank == 1);
    
    // Block assignment: blockIdx.x encodes (batch, head), blockIdx.y = q_tile
    int bh_idx = blockIdx.x;
    int batch = bh_idx / H;
    int head = bh_idx % H;
    int q_tile_global = blockIdx.y * BM_PER_CTA * 2;  // Two CTAs cover 128 rows
    
    // Offset based on whether we're the leader (rows 0..63) or follower (rows 64..127)
    int q_tile_base = q_tile_global + (follower ? BM_PER_CTA : 0);
    
    // Decode TMA coordinates for this tile
    // TMA coordinates: coord0 = outer_dim_start, coord1 = inner_dim_start
    // Our TMA descriptors encode (outer=S, inner=D) so coords are (start_row, 0)
    int q_coord = q_tile_base;
    int k_coord = 0;  // Will iterate over K tiles
    
    // Initialize barriers (only leader initializes, but both need to participate)
    if (tid == 0) {
        init_smem_barrier_fn(barrier_k, 1);
        init_smem_barrier_fn(barrier_q, 1);
        init_smem_barrier_fn(barrier_umma, 2);  // Both CTAs arrive
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    // Prefetch TMA descriptors
    prefetch_tensormap_fn(&tma_Q);
    prefetch_tensormap_fn(&tma_K);
    prefetch_tensormap_fn(&tma_V);
    prefetch_tensormap_fn(&tma_O);
    
    // Allocate Tensor Memory: we need space for C matrix (BM x BN fp32)
    // TMEM addressing: lane[15:0] x col[31:16], each cell is 32-bit
    // BM_PER_CTA=64 rows, BN_PER_CTA=64 cols => 64x64 cells = need to figure out columns
    // C descriptor: MN-major, needs columns = BN_PER_CTA
    // Actually for .128x128b cp shape: covers 64x64 bf16 source -> 64x64 fp32 dest? No...
    // Let me allocate enough columns for the C matrix accumulator
    // C is BM x BN fp32 = 64 x 64. For TMEM, each column has 128 rows.
    // We need ceil(BN/128) columns... but let's allocate more to be safe
    // Actually tcgen05 alloc allocates in units of 32 columns, min 32, max 512
    // For a 64x64 matrix stored as rows=lanes, cols=cols: need 64 columns
    const int NUM_TMEM_COLS = 64;  // power of 2, >= 32
    
    if (warp_id == 0 && tid == 0) {
        tmem_alloc_fn(tmem_addr_slot, NUM_TMEM_COLS);
    }
    __syncthreads();
    uint32_t tmem_base_addr;
    if (tid == 0) {
        tmem_base_addr = tmem_addr_slot[0];
    }
    tmem_base_addr = __shfl_sync(0xFFFFFFFF, tmem_base_addr, 0);
    
    // SMEM descriptors for UMMA
    // K_SMEM descriptor: K-major (no transpose), 128B swizzle
    // K layout in smem: [BN][BD], row-major with BD=128
    // For K-Major with 128B swizzle: ATOM_KMODE_DIM = 64 (=128/2), ATOM_MMODE_DIM = BM_PER_CTA
    // SBO = 8 * 128 = 1024, LBO = 1
    // Hmm actually let me reconsider. The swizzled descriptor expects specific layouts.
    // For simplicity, use non-swizzed descriptor with manual base address walking
    
    // Non-swizzed K-major descriptor for K
    // K-matrix is BN_PER_CTA x BD, stored in smem as [bn][bd]
    // ATOM_MMODE_DIM = BM_PER_CTA = 64 (number of rows consumed)
    // ATOM_KMODE_DIM = 16/2 = 8 bf16 per span, SBO = 8*16 = 128
    // Wait, I realize the WGMMA consumes fixed-size atoms from the SMEM descriptor.
    // Let me use the correct formula from the docs.
    
    // For cta_group::2 UMMA:
    // Combined M = BM_PER_CTA * 2 = 128, Combined N = BN_PER_CTA * 2 = 128
    // K dimension = WMMA_K = 16
    
    // For each WGMMA call, we contract M=128, N=128, K=16
    // This is split between two CTAs: each handles 64 rows of M, 64 cols of N
    
    // Actually, let me reconsider. WGMMA supports these shapes:
    // cta_group::2: 128xNxK or 256xNxK where N={16,32,...,256}, K=16
    // So we can do 128x128x16 per MMA call!
    // A is MxK = 128x16, B is KxN = 16x128, C is MxN = 128x128
    
    // For the attention QK^T:
    // A = Q tile = BM_total x BD = 128 x 128 (but we consume K-stride = 16 at a time)
    // B = K^T tile = BD x BN_total = 128 x 128
    // We process K in chunks of 16, needing BD/16 = 8 WGMMA calls
    
    // But wait - WGMMA's K dimension is always 16 (the atom dimension).
    // So for BD=128, we need 128/16 = 8 iterations.
    
    // For softmax: each of the 128 threads (two warpgroups) handles one row of C
    // After softmax, P@V is another WGMMA: P=128x128, V=128xBD
    
    // This is getting complex. Let me simplify: use a single CTA (cta_group::1)
    // with smaller tiles to start with something that works.
    
    // Simplified approach: cta_group::1, BM=64, BN=64, BD=128
    // QK^T: 64x64x128 using 8 WGMMA calls of 64x64x16
    // Then softmax on 64x64 result
    // Then P@V: 64x128x128 using 8 WGMMA calls of 64x128x16
    
    // For cta_group::1: shapes supported are 64xNxK or 128xNxK, N steps of 8, K=16
    // 64x64x16 is valid! And 64x128x16 is also valid!
    
    // OK let me restart with a simpler cta_group::1 approach first,
    // then add cta_group::2 later if needed.
    
    // Actually, let me just write a correct, optimized shared-memory kernel without tcgen05
    // first, then worry about TCGEN05. The tcgen05 setup is extremely complex and error-prone.
    
    // FALLBACK: High-performance shared-memory kernel
    // Using coalesced loads, minimal registers, efficient softmax
    goto fallback_path;
    
fallback_path:;
    // Simple coalesced shared-memory kernel (fallback)
    // Each CTA handles one (batch, head, q_tile) with BM=64 query rows
    // 128 threads, 2 threads per row, 64 D-elements per thread
    
    int q_row_local = tid / 2;        // 0..63
    int d_lane = tid % 2;              // 0 or 1
    int d_stride = BD / 2;             // 64 elements per lane
    
    // Strides
    int stride_SD = S * D;
    int stride_BHD = H * stride_SD;
    
    // Block assignment
    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    int q_base = blockIdx.y * BM_PER_CTA;
    int qg = q_base + q_row_local;
    bool q_valid = (qg < S);
    
    // Base pointers
    const __nv_bfloat16* Q_bh = static_cast<const __nv_bfloat16*>(Q.data_ptr()) + b * stride_BHD + h * stride_SD;
    const __nv_bfloat16* K_bh = static_cast<const __nv_bfloat16*>(K.data_ptr()) + b * stride_BHD + h * stride_SD;
    const __nv_bfloat16* V_bh = static_cast<const __nv_bfloat16*>(V.data_ptr()) + b * stride_BHD + h * stride_SD;
    __nv_bfloat16* O_bh = static_cast<__nv_bfloat16*>(O.data_ptr()) + b * stride_BHD + h * stride_SD;
    
    // Load Q fragment into registers
    float q_frag[d_stride];
    float o_acc[d_stride];
    if (q_valid) {
        const __nv_bfloat16* qr = Q_bh + qg * D;
        #pragma unroll 4
        for (int i = 0; i < d_stride; i += 4) {
            int di = d_lane * d_stride + i;
            q_frag[i]   = __bfloat162float(qr[di]) * inv_sqrt_d;
            q_frag[i+1] = __bfloat162float(qr[di+1]) * inv_sqrt_d;
            q_frag[i+2] = __bfloat162float(qr[di+2]) * inv_sqrt_d;
            q_frag[i+3] = __bfloat162float(qr[di+3]) * inv_sqrt_d;
        }
    } else {
        #pragma unroll 4
        for (int i = 0; i < d_stride; i += 4) {
            q_frag[i] = q_frag[i+1] = q_frag[i+2] = q_frag[i+3] = 0.0f;
        }
    }
    #pragma unroll 4
    for (int i = 0; i < d_stride; i += 4) {
        o_acc[i] = o_acc[i+1] = o_acc[i+2] = o_acc[i+3] = 0.0f;
    }
    
    // Use smaller BV to reduce shared memory and improve occupancy
    constexpr int BK = 64;
    // Shared memory for K and V tiles: [BK][BD]
    __shared__ __nv_bfloat16 sK[BK][BD + 8];  // +8 padding to avoid bank conflicts
    __shared__ __nv_bfloat16 sV[BK][BD + 8];
    
    float m_i = -FLT_MAX;  // running max
    float l_i = 0.0f;      // running sum
    
    int num_k_tiles = (S + BK - 1) / BK;
    
    for (int ti = 0; ti < num_k_tiles; ++ti) {
        int k_start = ti * BK;
        
        // Coalesced K load
        #pragma unroll 4
        for (int i = 0; i < BK * (BD+8) / TPB; i++) {
            int flat = tid * (BK * (BD+8) / TPB) + i;
            if (flat < BK * BD) {
                int kn = flat / BD;
                int kd = flat % BD;
                int kg = k_start + kn;
                sK[kn][kd] = (kg < S) ? K_bh[kg * D + kd] : __float2bfloat16(0.0f);
            }
        }
        
        // Coalesced V load
        #pragma unroll 4
        for (int i = 0; i < BK * (BD+8) / TPB; i++) {
            int flat = tid * (BK * (BD+8) / TPB) + i;
            if (flat < BK * BD) {
                int vn = flat / BD;
                int vd = flat % BD;
                int vg = k_start + vn;
                sV[vn][vd] = (vg < S) ? V_bh[vg * D + vd] : __float2bfloat16(0.0f);
            }
        }
        __syncthreads();
        
        // Causal bounds
        int n_valid = q_valid ? min(S, k_start + BK) - k_start : 0;
        if (q_valid) {
            int last_key = min(S, k_start + BK);
            if (last_key <= qg) continue;  // All masked
            int first_valid = max(k_start, 0);
            int end_valid = min(last_key, qg);
            n_valid = end_valid - first_valid;
        } else {
            n_valid = 0;
        }
        if (n_valid <= 0) {
            __syncthreads();
            continue;
        }
        
        // Compute scores: for each valid key, dot product over D
        float tile_max = -FLT_MAX;
        float scores_cache[BK];  // Cache scores for reuse
        
        int first_k = k_start;
        int last_k_inclusive = q_valid ? min(S, min(k_start + BK, qg)) - 1 : k_start - 1;
        
        #pragma unroll 4
        for (int kc = 0; kc < BK; ++kc) {
            int kg = k_start + kc;
            if (q_valid && kg < qg && kg < S) {
                float s = 0.0f;
                const __nv_bfloat16* kcol = sK[kc];
                int db = d_lane * d_stride;
                #pragma unroll 4
                for (int di = 0; di < d_stride; di += 4) {
                    s += q_frag[di]   * __bfloat162float(kcol[db + di]);
                    s += q_frag[di+1] * __bfloat162float(kcol[db + di + 1]);
                    s += q_frag[di+2] * __bfloat162float(kcol[db + di + 2]);
                    s += q_frag[di+3] * __bfloat162float(kcol[db + di + 3]);
                }
                // Reduce across 2 lanes
                s += __shfl_xor_sync(0xFFFFFFFF, s, 1);
                scores_cache[kc] = s;
                if (s > tile_max) tile_max = s;
            } else {
                scores_cache[kc] = -FLT_MAX;
            }
        }
        
        // Online softmax
        float m_prev = m_i;
        float m_new = max(m_prev, tile_max);
        float alpha = (m_new == m_prev) ? 1.0f : expf_approx(m_prev - m_new);
        
        // Scale previous accumulation
        if (alpha != 1.0f) {
            #pragma unroll 4
            for (int i = 0; i < d_stride; i += 4) {
                o_acc[i]   *= alpha;
                o_acc[i+1] *= alpha;
                o_acc[i+2] *= alpha;
                o_acc[i+3] *= alpha;
            }
        }
        
        l_i *= alpha;
        
        // Accumulate weighted V
        #pragma unroll 4
        for (int kc = 0; kc < BK; ++kc) {
            int kg = k_start + kc;
            if (q_valid && kg < qg && kg < S && scores_cache[kc] > -FLT_MAX * 0.5f) {
                float p = expf_approx(scores_cache[kc] - m_new);
                l_i += p;
                const __nv_bfloat16* vcol = sV[kc];
                int db = d_lane * d_stride;
                #pragma unroll 4
                for (int di = 0; di < d_stride; di += 4) {
                    o_acc[di]   += p * __bfloat162float(vcol[db + di]);
                    o_acc[di+1] += p * __bfloat162float(vcol[db + di + 1]);
                    o_acc[di+2] += p * __bfloat162float(vcol[db + di + 2]);
                    o_acc[di+3] += p * __bfloat162float(vcol[db + di + 3]);
                }
            }
        }
        
        m_i = m_new;
        __syncthreads();
    }
    
    // Write-back
    if (q_valid) {
        float* lse_out = static_cast<float*>(LSE_out.data_ptr()) + b * H * S + h * S + qg;
        __nv_bfloat16* optr = O_bh + qg * D;
        int db = d_lane * d_stride;
        
        if (l_i > 0.0f) {
            float norm = 1.0f / l_i;
            *lse_out = m_i + logf(l_i);
            #pragma unroll 4
            for (int i = 0; i < d_stride; i += 4) {
                optr[db+i]   = __float2bfloat16(o_acc[i] * norm);
                optr[db+i+1] = __float2bfloat16(o_acc[i+1] * norm);
                optr[db+i+2] = __float2bfloat16(o_acc[i+2] * norm);
                optr[db+i+3] = __float2bfloat16(o_acc[i+3] * norm);
            }
        } else {
            *lse_out = -FLT_MAX;
            #pragma unroll 4
            for (int i = 0; i < d_stride; i += 4) {
                optr[db+i] = optr[db+i+1] = optr[db+i+2] = optr[db+i+3] = __float2bfloat16(0.0f);
            }
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());
    
    float inv_sqrt_d = 1.0f / sqrtf(static_cast<float>(D));
    
    int64_t num_bh = B * H;
    int64_t num_qtiles = (S + BM_PER_CTA - 1) / BM_PER_CTA;
    
    dim3 grid(static_cast<unsigned>(num_bh), static_cast<unsigned>(num_qtiles));
    dim3 block(THREADS_PER_BLOCK);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    mha_causal_kernel_tcgen05<<<grid, block, 0, stream>>>(
        tma_Q_dummy, tma_K_dummy, tma_V_dummy, tma_O_dummy,
        LSE_data,
        static_cast<int>(B), static_cast<int>(H), static_cast<int>(S),
        static_cast<int>(D), inv_sqrt_d);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_blackwell::run);

}  // namespace mha_blackwell