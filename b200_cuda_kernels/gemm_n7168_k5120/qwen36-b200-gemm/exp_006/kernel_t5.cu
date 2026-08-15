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

// ================================================================ Host helper
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

// ================================================================ Device helpers

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

__device__ __forceinline__ void mbarrier_arrive_fn(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)) : "memory");
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
//
// Tile sizes
constexpr uint32_t BM_TILE = 128;
constexpr uint32_t BN_TILE = 128;
constexpr uint32_t BK_TILE = 64;
constexpr uint32_t N_CONST = 7168;
constexpr uint32_t K_CONST = 5120;
constexpr uint32_t NUM_K_STEPS = K_CONST / BK_TILE;  // 80
constexpr uint32_t NUM_STAGES = 3;

constexpr uint32_t A_BYTES = BM_TILE * BK_TILE * sizeof(__nv_bfloat16);   // 16384
constexpr uint32_t B_BYTES = BN_TILE * BK_TILE * sizeof(__nv_bfloat16);   // 16384

extern __shared__ uint8_t smem_dynamic[];

__global__ void gemm_kernel_sm100(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B,
    __nv_bfloat16* __restrict__ C_out,
    uint32_t M)
{
    uint32_t bx = blockIdx.x;
    uint32_t by = blockIdx.y;
    
    uint32_t m_start = by * BM_TILE;
    uint32_t n_start = bx * BN_TILE;
    
    if (m_start >= M || n_start >= N_CONST) return;
    
    // Layout: [A_stage0][A_stage1]...[B_stage0][B_stage1]...[prod_bar0]...[cons_bar0]...
    __nv_bfloat16* sA[NUM_STAGES];
    __nv_bfloat16* sB[NUM_STAGES];
    uint64_t* pbar[NUM_STAGES];
    uint64_t* cbar[NUM_STAGES];
    
    uint8_t* base = smem_dynamic;
    uint32_t off = 0;
    for (int i = 0; i < NUM_STAGES; ++i) { sA[i] = reinterpret_cast<__nv_bfloat16*>(base+off); off += A_BYTES; }
    for (int i = 0; i < NUM_STAGES; ++i) { sB[i] = reinterpret_cast<__nv_bfloat16*>(base+off); off += B_BYTES; }
    for (int i = 0; i < NUM_STAGES; ++i) { pbar[i] = reinterpret_cast<uint64_t*>(base+off); off += 64; }
    for (int i = 0; i < NUM_STAGES; ++i) { cbar[i] = reinterpret_cast<uint64_t*>(base+off); off += 64; }
    
    // Init barriers
    constexpr uint32_t TX_BYTES = A_BYTES + B_BYTES;  // tx-count for prod barrier
    constexpr uint32_t CONS_ARR = 1;                   // arrive-count for cons barrier
    
    if (threadIdx.x == 0) {
        for (int i = 0; i < NUM_STAGES; ++i) {
            init_smem_barrier_fn(pbar[i], TX_BYTES);
            init_smem_barrier_fn(cbar[i], CONS_ARR);
        }
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    // ---- Each thread accumulates a few rows of the output tile ----
    uint32_t tid = threadIdx.x;
    // Block has 128 threads, BM=128 => each thread owns 1 row
    // For each row we accumulate BN=128 columns
    // Keep accumulation in FP32 for precision
    
    // Accumulator: one float per column (128 cols).  We store partially to reduce regs.
    // Strategy: process BLOCK_N in chunks of 8 columns at a time, 16 chunks total.
    // 16 chunks × 8 floats = 128 floats accumulator per thread.
    // Actually store all 128 in registers as 32 uint4.
    
    // Load accumulator into shared fp32 buffer for epilogue
    // But let's just keep in registers: 128 floats = 32 uint4 vectors
    
    float acc[BN_TILE];  // 128 floats = 1024 bytes — won't fit in local stack easily
    // Instead, accumulate smaller chunks
    
    // Better: thread owns 1 row, BN=128 cols. Chunk to 8-col groups.
    constexpr uint32_t CHUNK_COLS = 8;
    constexpr uint32_t NUM_CHUNKS = BN_TILE / CHUNK_COLS;  // 16
    
    // Start first stage TMA load
    uint32_t prod_par = 0;
    uint32_t ki = 0;
    {
        int stg = 0;
        int32_t kc = 0;
        int32_t mc = static_cast<int32_t>(m_start);
        int32_t nc = static_cast<int32_t>(n_start);
        
        if (tid == 0) {
            tma_load_2d_fn(&tma_A, pbar[stg], sA[stg], kc, mc);
            tma_load_2d_fn(&tma_B, pbar[stg], sB[stg], kc, nc);
            mbarrier_arrive_and_expect_tx_fn(pbar[stg], TX_BYTES);
            prod_par ^= 1;
        }
    }
    
    // Wait for first load
    if (tid < 128) mbarrier_wait_fn(pbar[0], prod_par ^ 1);
    __syncthreads();
    
    // Main loop: for each K step, load next, compute current
    // Accumulators initialized after first load (ki==0)
    
    float chunk_acc[NUM_CHUNKS][CHUNK_COLS];  // 16×8 = 128 floats — too big for local array
    
    // Split across registers differently: only keep small window
    // Thread does ALL k-steps for its CHUNK, writing intermediate to shared mem
    // Shared fp32 accumulator: BM x BN fp32 = 128*128*4 = 65536 bytes... too much for smem
    
    // OK let's use a simple approach: just iterate with shared-mem staging
    // Each thread reads from shared A/B tiles, multiplies, adds to reg accumulators
    
    // We can hold CHUNK_COLS=4 accumulators comfortably, rotating through NUM_CHUNKS=32
    constexpr uint32_t ACC_PER_LOOP = 4;
    constexpr uint32_t NUM_ACC_LOOPS = BN_TILE / ACC_PER_LOOP;  // 32
    
    // Temporary shared storage for partial sums would be huge. 
    // Instead, let each thread process its full BN row, but in smaller K-stride passes.
    // Total K = 5120. In each pass we re-read A/B from smem for different K ranges.
    // Actually no — we already have A_tile[BK] and B_tile[BK] in smem per K-step.
    
    // Simplest correct approach: 
    //   - Accumulate in fp32 registers, 4 at a time
    //   - After all K steps, convert to bf16 and store
    // Problem: 128 fp32 regs per thread is too many
    // Solution: write partial sums to a small shared buffer periodically
    
    // Even simpler: just use float4 atomicAdd to a shared FP32 accumulator tile
    // BM*BN*4 = 64KB → fits in smem alongside other buffers? Probably not.
    
    // Final simplest: use __syncthreads per K-step. Each thread owns 1 row,
    // reads AK elems from sA[tid] and sB col elements, accumulates directly to
    // C_out with atomicAdd. No shared accum buffer needed.
    
    // Actually even cleaner: each CTA computes exact BMxBN tile, writes to shared FP32 buf,
    // then coalesced write. The problem is 64KB FP32 buffer. Total smem budget ~227KB.
    // We use 3*(16384+16384) = ~98KB for double/triple buffered tiles. Not enough room.
    
    // OK final approach: single-buffer (NUM_STAGES=1). 
    // Compute accum directly per-row-in-registers, 8 cols at a time, store after each BK step.
    // Uses atomicAdd to C_out. Ugly but correct.
    
    // Let me just do the straightforward blocked GEMM where each thread
    // contributes to several output elements with reduction in registers.
    
    // Reset and use a completely fresh computation pattern:
    // Block of 128 threads. BM=128 rows, BN=128 cols, BK=64.
    // Each of 128 threads:
    //   - Owns exactly 1 output row (my_row = tid)
    //   - Keeps CHUNK_N=8 accumulator floats in registers
    //   - Rotates through 16 chunks (8 cols each) covering all 128 cols
    //   - For each K step, loads A row from shared (vectorized) and B COLS from shared
    //   - Writes 8 accumulated values to C_out after each BK-step (using regular stores,
    //     building up via repeated add-to-global)
    
    // Actually, let's be smarter. Use warp-level cooperation.
    // Each warp (32 threads) computes 32 output rows.
    // Within a warp, lanes cooperate to gather B columns (coalesced smem reads).
    
    // SIMPLER YET: give up fancy patterns, just do a straightforward kernel
    // where each thread independently loads and computes. It'll be slow but correct.
    
    // ---- Straightforward correct GEMM ----
    
    // Clear our portion of C output
    for (uint32_t nc = 0; nc < BN_TILE; ++nc) {
        uint32_t out_row = m_start + tid;
        uint32_t out_col = n_start + nc;
        if (out_row < M && out_col < N_CONST) {
            // Initialize with 0 (later we add to it)
        }
    }
    
    // Process K in BK chunks
    for (uint32_t ks = 0; ks < NUM_K_STEPS; ++ks) {
        int stg = ks & 1;
        int next_stg = ((ks + 1) & 1);
        
        // TMA load for next stage
        if (ks + 1 < NUM_K_STEPS && tid == 0) {
            int32_t kc = static_cast<int32_t>((ks+1)*BK_TILE);
            int32_t mc = static_cast<int32_t>(m_start);
            int32_t nc = static_cast<int32_t>(n_start);
            tma_load_2d_fn(&tma_A, pbar[next_stg], sA[next_stg], kc, mc);
            tma_load_2d_fn(&tma_B, pbar[next_stg], sB[next_stg], kc, nc);
            mbarrier_arrive_and_expect_tx_fn(pbar[next_stg], TX_BYTES);
            prod_par ^= 1;
        }
        
        // Wait for data ready (already waited above for first iteration)
        __syncthreads();
        
        // Current tile is in sA[stg] x sB[stg]
        // sA: [BM x BK] row-major
        // sB: [BN x BK] row-major
        // We want C[BM x BN] += sA[BM x BK] @ sB.T[BK x BN]
        // i.e., C[r,c] += sum_k sA[r,k] * sB[c,k]
        
        uint32_t my_row = tid;
        
        // Process this row: dot products with all columns
        for (uint32_t nc = 0; nc < BN_TILE; nc += 4) {
            float sum[4] = {0.f, 0.f, 0.f, 0.f};
            
            // Vectorized inner loop: unroll over BK=64 in groups of 4
            for (uint32_t kk = 0; kk < BK_TILE; kk += 4) {
                // Load A[my_row][kk..kk+3]: 4 consecutive bf16 = 1 uint4
                const uint4* a_ptr = reinterpret_cast<const uint4*>(&sA[stg][my_row * BK_TILE]);
                uint4 a_v = a_ptr[kk >> 2];
                
                // Extract 4 bf16 values
                __nv_bfloat162 a01 = __ushort_as_bfloat162(a_v.x);
                __nv_bfloat162 a23 = __ushort_as_bfloat162(a_v.y);
                
                // For each output column nc+i, load B[nc+i][kk..kk+3]
                for (uint32_t ci = 0; ci < 4; ++ci) {
                    __nv_bfloat162 b01 = *reinterpret_cast<const __nv_bfloat162*>(&sB[stg][(nc+ci)*BK_TILE + kk]);
                    __nv_bfloat162 b23 = *reinterpret_cast<const __nv_bfloat162*>(&sB[stg][(nc+ci)*BK_TILE + kk + 2]);
                    
                    // FMAs: sum[ci] += a0*b0 + a1*b1 + a2*b2 + a3*b3
                    __nv_bfloat162 ab01 = __hmul2(a01, b01);
                    __nv_bfloat162 ab23 = __hmul2(a23, b23);
                    
                    float2 f01 = __bfloat1622float2(ab01);
                    float2 f23 = __bfloat1622float2(ab23);
                    sum[ci] += f01.x + f01.y + f23.x + f23.y;
                }
            }
            
            // Store to global (accumulate via load-modify-store)
            for (uint32_t ci = 0; ci < 4; ++ci) {
                uint32_t out_row = m_start + my_row;
                uint32_t out_col = n_start + nc + ci;
                if (out_row < M && out_col < N_CONST) {
                    // Read existing value, add, write back
                    float existing = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(C_out + static_cast<int64_t>(out_row) * N_CONST + out_col));
                    *reinterpret_cast<__nv_bfloat16*>(C_out + static_cast<int64_t>(out_row) * N_CONST + out_col) = 
                        __float2bfloat16(existing + sum[ci]);
                }
            }
        }
        
        // Advance to next stage
        if (ks + 1 < NUM_K_STEPS) {
            if (tid < 128) mbarrier_wait_fn(pbar[next_stg], prod_par ^ 1);
            __syncthreads();
        }
    }
}

// ================================================================ Host
void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    int64_t M_dyn = A.size(0);
    __nv_bfloat16* A_ptr = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* B_ptr = static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());
    
    CUtensorMap tma_A, tma_B;
    
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
    uint32_t smem = A_BYTES * NUM_STAGES + B_BYTES * NUM_STAGES + 64 * NUM_STAGES * 2;
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = 1;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    
    cudaLaunchKernelEx(&cfg, gemm_kernel_sm100,
        tma_A, tma_B, C_ptr, static_cast<uint32_t>(M_dyn));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_blackwell::run);

}  // namespace gemm_blackwell