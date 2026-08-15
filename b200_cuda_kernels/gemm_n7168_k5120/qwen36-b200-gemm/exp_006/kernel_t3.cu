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
        const char *_err_str;                                      \
        cuGetErrorString(_r, &_err_str);                           \
        fprintf(stderr, "CU error %s at %s:%d\n",                 \
                _err_str, __FILE__, __LINE__);                     \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace gemm_blackwell {

// ==================== Device Helpers ====================

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

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t parity) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=: \n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(parity));
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
           "l"((uint64_t)d),
           "r"((uint32_t)__cvta_generic_to_shared(bar)),
           "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void umma_commit_1sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared.b64 [%0];"
        :: "r"(a));
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
        :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void tmem_wait_ld_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_fn(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)) : "memory");
}

// Build SMEM descriptor: non-swizzled, K-major, v1
__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;       // version=1
    d |= (uint64_t)0 << 61;       // no swizzle
    return d;
}

// Instruction descriptor: BF16x16 -> FP32, K-major A+B
__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);      // dtype = F32
    d |= (1u << 7);      // atype = BF16
    d |= (1u << 10);     // btype = BF16
    d |= ((N / 8) << 17);   // n_dim
    d |= ((M / 16) << 24);  // m_dim
    return d;
}

// Issue UMMA cta_group::1
__device__ __forceinline__ void umma_f16_fn(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
                                             uint32_t idesc, uint32_t do_accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(do_accum));
}

// Host-side TMA descriptor
CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, CUtensorMapDataType dataType,
    void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim,
    uint32_t smem_inner_dim, uint32_t smem_outer_dim,
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, dataType, 2, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill);
}

// ==================== Tile Parameters ====================
constexpr uint32_t BM_TILE = 128;     // M-per-CTA
constexpr uint32_t BN_TILE = 128;     // N-per-CTA
constexpr uint32_t BK_TILE = 16;      // K-per-step

constexpr uint32_t N_CONST = 7168;
constexpr uint32_t K_CONST = 5120;
constexpr uint32_t NUM_K = K_CONST / BK_TILE;  // 320

constexpr uint32_t NUM_STAGES = 2;
constexpr uint32_t A_TILE_BYTES = BM_TILE * BK_TILE * sizeof(__nv_bfloat16);  // 4096
constexpr uint32_t B_TILE_BYTES = BN_TILE * BK_TILE * sizeof(__nv_bfloat16);  // 4096
constexpr uint32_t TOTAL_TX_BYTES = A_TILE_BYTES + B_TILE_BYTES;  // 8192

constexpr uint32_t SHARED_MEM_PER_CTA = 
    A_TILE_BYTES * NUM_STAGES + B_TILE_BYTES * NUM_STAGES + 128;  // ~16KB + padding

// ================================================================ Kernel
extern __shared__ uint8_t smem_dynamic[];

__global__ void gemm_kernel_sm100(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* __restrict__ C_out,
    uint32_t M,
    uint32_t num_n_blocks)
{
    uint32_t bx = blockIdx.x;  // n-block index
    uint32_t by = blockIdx.y;  // m-block index
    
    uint32_t m_start = by * BM_TILE;
    uint32_t n_start = bx * BN_TILE;
    
    if (m_start >= M || n_start >= N_CONST) return;
    
    // ---- Shared-memory layout ----
    __nv_bfloat16* smem_A[NUM_STAGES];
    __nv_bfloat16* smem_B[NUM_STAGES];
    uint64_t* prod_bar[NUM_STAGES];
    uint64_t* cons_bar[NUM_STAGES];
    
    uint8_t* base = smem_dynamic;
    uint32_t off = 0;
    for (int s = 0; s < NUM_STAGES; ++s) {
        smem_A[s] = reinterpret_cast<__nv_bfloat16*>(base + off); off += A_TILE_BYTES;
    }
    for (int s = 0; s < NUM_STAGES; ++s) {
        smem_B[s] = reinterpret_cast<__nv_bfloat16*>(base + off); off += B_TILE_BYTES;
    }
    for (int s = 0; s < NUM_STAGES; ++s) {
        prod_bar[s] = reinterpret_cast<uint64_t*>(base + off); off += 64;
    }
    for (int s = 0; s < NUM_STAGES; ++s) {
        cons_bar[s] = reinterpret_cast<uint64_t*>(base + off); off += 64;
    }
    
    uint32_t* tmem_addr = reinterpret_cast<uint32_t*>(base + off);
    
    // ---- Init barriers ----
    constexpr uint32_t PROD_ARRIVALS = TOTAL_TX_BYTES;  // tx-count
    constexpr uint32_t CONS_ARRIVALS = 1;              // one arrive per consume
    
    if (threadIdx.x == 0) {
        for (int s = 0; s < NUM_STAGES; ++s) {
            init_smem_barrier_fn(prod_bar[s], PROD_ARRIVALS);
            init_smem_barrier_fn(cons_bar[s], CONS_ARRIVALS);
        }
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    // ---- Allocate TMEM: 512 cols => 512*128*4 = 256 KB ----
    // Holds the 128-row x 128-col x FP32 accumulator
    if (threadIdx.x == 0) {
        tmem_alloc_fn(tmem_addr, 512);
    }
    __syncthreads();
    
    uint32_t c_tmem_base = tmem_addr[0];  // base TMEM address for 128x128 fp32
    
    // ---- Descriptors (non-swizzle, K-major) ----
    // SMEM K-major, no-swizzle:
    //   ATOM_KMODE_DIM = 16/2 = 8  spans; SBO = 8*16 = 128
    //   LBO = (ATOM_MMODE_DIM/8)*SBO = (128/8)*128 = 2048
    uint32_t smem_lbo = 2048;
    uint32_t smem_sbo = 128;
    uint32_t idesc = make_instr_desc_fn(BM_TILE, BN_TILE);
    
    // ---- Pipeline ----
    uint32_t prod_par = 0;
    uint32_t cons_par = 0;
    
    // --- k_iter 0: load + compute ---
    {
        int stg = 0;
        int32_t kc = 0;
        int32_t mc = static_cast<int32_t>(m_start);
        int32_t nc = static_cast<int32_t>(n_start);
        
        if (threadIdx.x == 0) {
            tma_load_2d_fn(&tma_A, prod_bar[stg], smem_A[stg], kc, mc);
            tma_load_2d_fn(&tma_B, prod_bar[stg], smem_B[stg], kc, nc);
            mbarrier_arrive_and_expect_tx_fn(prod_bar[stg], TOTAL_TX_BYTES);
            prod_par ^= 1;
        }
        
        if (threadIdx.x < 128) {
            mbarrier_wait_fn(prod_bar[0], prod_par ^ 1);
        }
        __syncthreads();
        
        if (threadIdx.x < 128) {
            uint64_t da = make_smem_desc_sm100_fn(smem_A[0], smem_lbo, smem_sbo);
            uint64_t db = make_smem_desc_sm100_fn(smem_B[0], smem_lbo, smem_sbo);
            fence_proxy_async_fn();
            umma_f16_fn(c_tmem_base, da, db, idesc, 0);  // clear acc
            umma_commit_1sm_fn(cons_bar[0]);
            cons_par ^= 1;
        }
    }
    
    // --- k_iter 1 .. NUM_K-1 ---
    for (uint32_t ki = 1; ki < NUM_K; ++ki) {
        int stg  = ki & 1;
        int prev = (ki - 1) & 1;
        
        // Producer: start TMA for stage `stg`
        if (threadIdx.x == 0) {
            int32_t kc = static_cast<int32_t>(ki * BK_TILE);
            int32_t mc = static_cast<int32_t>(m_start);
            int32_t nc = static_cast<int32_t>(n_start);
            tma_load_2d_fn(&tma_A, prod_bar[stg], smem_A[stg], kc, mc);
            tma_load_2d_fn(&tma_B, prod_bar[stg], smem_B[stg], kc, nc);
            mbarrier_arrive_and_expect_tx_fn(prod_bar[stg], TOTAL_TX_BYTES);
            prod_par ^= 1;
        }
        
        // Consumer: wait for prev stage consumed
        if (threadIdx.x < 128) {
            mbarrier_wait_fn(cons_bar[prev], cons_par);
        }
        __syncthreads();
        
        // Wait for current stage produced
        if (threadIdx.x < 128) {
            mbarrier_wait_fn(prod_bar[stg], prod_par ^ 1);
        }
        __syncthreads();
        
        // Compute current stage
        if (threadIdx.x < 128) {
            uint64_t da = make_smem_desc_sm100_fn(smem_A[stg], smem_lbo, smem_sbo);
            uint64_t db = make_smem_desc_sm100_fn(smem_B[stg], smem_lbo, smem_sbo);
            fence_proxy_async_fn();
            umma_f16_fn(c_tmem_base, da, db, idesc, 1);  // accumulate
            umma_commit_1sm_fn(cons_bar[stg]);
            cons_par ^= 1;
        }
    }
    
    // Wait for final consumer to complete
    int last_stg = (NUM_K - 1) & 1;
    if (threadIdx.x < 128) {
        mbarrier_wait_fn(cons_bar[last_stg], cons_par);
    }
    __syncthreads();
    
    // Deallocate TMEM
    if (threadIdx.x == 0) {
        tmem_dealloc_fn(c_tmem_base, 512);
    }
    
    // ======================== Epilogue ========================
    // Each thread tid owns one row of the 128x128 accumulator
    uint32_t tid = threadIdx.x;
    if (tid >= 128) return;
    
    uint32_t global_row = m_start + tid;
    if (global_row >= M) return;
    
    int64_t row_offset = static_cast<int64_t>(global_row) * N_CONST;
    
    // Load 128 FP32 values from TMEM row tid, convert to BF16, store to global
    // TMEM address = column_index (rows implicit from collective lanes)
    for (uint32_t col = 0; col < BN_TILE; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col));
        tmem_wait_ld_fn();
        
        uint32_t nc = n_start + col;
        __nv_bfloat16* ptr = C_out + row_offset + nc;
        
        if (nc     < N_CONST) ptr[0] = __float2bfloat16(__uint_as_float(r0));
        if (nc + 1  < N_CONST) ptr[1] = __float2bfloat16(__uint_as_float(r1));
        if (nc + 2  < N_CONST) ptr[2] = __float2bfloat16(__uint_as_float(r2));
        if (nc + 3  < N_CONST) ptr[3] = __float2bfloat16(__uint_as_float(r3));
    }
}

// ======================== Host ========================

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    int64_t M_dyn = A.size(0);
    __nv_bfloat16* A_ptr = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* B_ptr = static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());
    
    CUtensorMap tma_A, tma_B;
    
    // A: global [M, K], TMA view [K_inner, M_outer], box=[BK, BM]
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_A, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        A_ptr, K_CONST, static_cast<uint64_t>(M_dyn),
        BK_TILE, BM_TILE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    // B: global [N, K], TMA view [K_inner, N_outer], box=[BK, BN]
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_B, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        B_ptr, K_CONST, N_CONST,
        BK_TILE, BN_TILE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    uint32_t gx = (N_CONST + BN_TILE - 1) / BN_TILE;
    uint32_t gy = (static_cast<uint32_t>(M_dyn) + BM_TILE - 1) / BM_TILE;
    
    dim3 grid(gx, gy);
    dim3 block(128, 1, 1);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = SHARED_MEM_PER_CTA;
    cfg.stream = stream;
    
    // Cluster dim=1 (single CTA clusters, no cross-CTA sync needed)
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = 1;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    
    cudaLaunchKernelEx(&cfg, gemm_kernel_sm100,
        tma_A, tma_B, C_ptr, static_cast<uint32_t>(M_dyn), gx);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_blackwell::run);

}  // namespace gemm_blackwell