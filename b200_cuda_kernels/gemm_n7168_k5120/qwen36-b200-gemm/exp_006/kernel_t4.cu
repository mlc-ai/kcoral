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

// ================================================================ Kernel
// TMA-load + cp.async + wgmma.mma_async pipeline GEMM
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
    
    constexpr uint32_t BM = 128;
    constexpr uint32_t BN = 128;
    constexpr uint32_t BK = 64;
    constexpr uint32_t N_CONST = 7168;
    constexpr uint32_t K_CONST = 5120;
    constexpr uint32_t NUM_K = K_CONST / BK;  // 80
    
    constexpr uint32_t NUM_STAGES = 2;
    constexpr uint32_t A_TILE_BYTES = BM * BK * sizeof(__nv_bfloat16);   // 16384
    constexpr uint32_t B_TILE_BYTES = BN * BK * sizeof(__nv_bfloat16);   // 16384
    
    uint32_t m_start = by * BM;
    uint32_t n_start = bx * BN;
    
    if (m_start >= M || n_start >= N_CONST) return;
    
    // ---- Shared-memory layout ----
    __nv_bfloat16* smem_A[NUM_STAGES];
    __nv_bfloat16* smem_B[NUM_STAGES];
    uint64_t* prod_bar[NUM_STAGES];
    
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
    
    // ---- Init barriers ----
    constexpr uint32_t TOTAL_TX = A_TILE_BYTES + B_TILE_BYTES;
    if (threadIdx.x == 0) {
        for (int s = 0; s < NUM_STAGES; ++s) {
            init_smem_barrier_fn(prod_bar[s], TOTAL_TX);
        }
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    // ---- WGMMA accumulator registers ----
    // Use 64 registers per thread for D matrix (staged in reg file)
    // We'll accumulate directly in the wgmma flow
    
    uint32_t tid = threadIdx.x;
    uint32_t warp_id = tid >> 5;
    uint32_t lane_id = tid & 31;
    
    // Each warpgroup (4 warps = 128 threads) processes one BM x BN tile
    // Within a warpgroup, wgmma uses collective operations
    
    // Accumulator storage: FP32 array shared among warps
    // Layout: each thread holds 4 rows × BN cols = small slices
    // Actually let's just use register-based accumulators per-warpgroup
    // BM=128, BN=128 => 128*128 fp32 = 65536 values
    // 128 threads × ~512 values/thread = too many regs, use shared mem staging
    
    // For simplicity: process in smaller sub-tiles that fit in registers
    constexpr uint32_t SUB_M = 16;   // 16 rows per sub-tile
    constexpr uint32_t SUB_N = 16;   // 16 cols per sub-tile
    constexpr uint32_t SUB_K = 16;   // wgmma native K
    
    // Load entire tiles into shared memory first, then compute with wgmma
    // Pipeline: 2 stages of TMA load
    
    uint32_t prod_par = 0;
    
    // --- Start stage 0 load ---
    {
        int stg = 0;
        int32_t kc = 0;
        int32_t mc = static_cast<int32_t>(m_start);
        int32_t nc = static_cast<int32_t>(n_start);
        
        if (threadIdx.x == 0) {
            tma_load_2d_fn(&tma_A, prod_bar[stg], smem_A[stg], kc, mc);
            tma_load_2d_fn(&tma_B, prod_bar[stg], smem_B[stg], kc, nc);
            mbarrier_arrive_and_expect_tx_fn(prod_bar[stg], TOTAL_TX);
            prod_par ^= 1;
        }
        
        __syncwarp();
        if (tid < 128) {
            mbarrier_wait_fn(prod_bar[0], prod_par ^ 1);
        }
        __syncthreads();
        
        // Compute all K chunks for stage 0 in a loop reading from smem
        // We'll process BK=64 in 4 steps of SUB_K=16
        for (uint32_t ki = 0; ki < NUM_K; ++ki) {
            int next_stg = (ki + 1) & 1;
            
            // Launch next TMA if available
            if (ki + 1 < NUM_K && threadIdx.x == 0) {
                int32_t kcoord = static_cast<int32_t>((ki + 1) * BK);
                int32_t mcoord = static_cast<int32_t>(m_start);
                int32_t ncoord = static_cast<int32_t>(n_start);
                tma_load_2d_fn(&tma_A, prod_bar[next_stg], smem_A[next_stg], kcoord, mcoord);
                tma_load_2d_fn(&tma_B, prod_bar[next_stg], smem_B[next_stg], kcoord, ncoord);
                mbarrier_arrive_and_expect_tx_fn(prod_bar[next_stg], TOTAL_TX);
                prod_par ^= 1;
            }
            
            // Current stage smem ready
            __nv_bfloat16* curA = smem_A[ki & 1];
            __nv_bfloat16* curB = smem_B[ki & 1];
            
            // Process this BK chunk in SUB_K steps using wgmma
            // wgmma mma_async.cg.a16b16.f32.m16n16k16 
            // shape: 16x16x16 per issue
            
            // Initialize accumulators (first K iteration only)
            float acc_reg[64];  // 16x16 = 256 floats? No, 16/4*16 = 64 x 4 = 256
            // Each warp computes 16x16, storing 256 floats = 64x32-bit regs
            if (ki == 0) {
                for (int i = 0; i < 64; ++i) acc_reg[i] = 0.0f;
            }
            
            // SMEM pointer walk: A[k_step][row], B[k_step][col]
            // With SUB_K=16, each wgmma consumes 16 K elements
            for (uint32_t ksub = 0; ksub < BK / SUB_K; ++ksub) {
                uint32_t k_offset = ki * BK + ksub * SUB_K;
                
                // WGMMATILE descriptor assembly
                // A: BM x SUB_K = 128x16 bf16, row-major => K-walk inner
                // B: BN x SUB_K = 128x16 bf16, row-major
                // wgmma.ldmatrix.sync.aligned.m16n8k32.trans.shared.b16 ...
                
                // Simpler approach: load into registers then use mma.sync
                // Each warp loads its 16x16 subtile
                
                uint32_t row_base = warp_id * 32;  // each warp owns 32 rows within BM=128
                uint32_t col_base = 0;
                
                if (row_base >= BM) break;
                
                // Manual load: each lane reads 1 bf16 pair from A and B
                // This gets complex fast. Let me use a cleaner approach.
            }
            
            // If next stage, wait for it before consuming current
            if (ki + 1 < NUM_K) {
                if (tid < 128) {
                    mbarrier_wait_fn(prod_bar[next_stg], prod_par ^ 1);
                }
                __syncthreads();
            }
        }
    }
    
    // This approach is getting too complicated with hand-crafted wgmma.
    // Let me switch to a clean synchronous kernel using direct smem access.
    
    // ---- FALLBACK: Simple synchronous kernel ----
    // Re-process everything cleanly
    __syncthreads();
    
    // Clear local output storage
    float local_C[BM / 4][BN / 4];  // Won't fit on stack
    // Instead, compute and store directly to global
    
    // Each thread processes one element of output
    // Block has 128 threads, tile is 128x128 = 16384 outputs
    // Need more threads or multiple passes
    
    uint32_t my_tid = threadIdx.x;
    if (my_tid < BM) {
        uint32_t out_row = m_start + my_tid;
        if (out_row >= M) return;
        
        int64_t row_off = static_cast<int64_t>(out_row) * N_CONST;
        
        // Accumulate in registers for this row
        // Row length BN=128, each element needs K=5120 multiply-add
        // Can't hold all 128 accumulators + tile data easily
        
        // Better: compute sub-row chunks
        for (uint32_t nc = 0; nc < BN; nc += 4) {
            float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
            
            uint32_t out_col0 = n_start + nc;
            uint32_t out_col1 = n_start + nc + 1;
            uint32_t out_col2 = n_start + nc + 2;
            uint32_t out_col3 = n_start + nc + 3;
            
            for (uint32_t kc = 0; kc < K_CONST; kc += 4) {
                // Load A[out_row][kc..kc+3]
                __nv_bfloat16 a_vals = reinterpret_cast<const __nv_bfloat16*>(&C_out)[0]; // dummy, will use proper ptr
                
                // Load B columns
                // This naive loop is extremely slow. Let me use a proper tiled approach.
                break;
            }
            
            // Store results
        }
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
    
    constexpr uint32_t BK_TILE = 64;
    constexpr uint32_t BM_TILE = 128;
    constexpr uint32_t BN_TILE = 128;
    constexpr uint32_t K_CONST = 5120;
    constexpr uint32_t N_CONST = 7168;
    
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_A, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        A_ptr, K_CONST, static_cast<uint64_t>(M_dyn),
        BK_TILE, BM_TILE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
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
    
    constexpr uint32_t A_BYTES = BM_TILE * BK_TILE * 2;
    constexpr uint32_t B_BYTES = BN_TILE * BK_TILE * 2;
    uint32_t smem_per_cta = A_BYTES * 2 + B_BYTES * 2 + 128;
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem_per_cta;
    cfg.stream = stream;
    
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

// Declare helper on host side
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

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_blackwell::run);

}  // namespace gemm_blackwell