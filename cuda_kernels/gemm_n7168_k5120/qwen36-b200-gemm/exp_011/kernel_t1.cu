#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace gemm_blackwell {

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int THREADS_PER_BLOCK = 256;

// Thread tile dimensions
constexpr int WM = 8;   // each thread computes 8 rows
constexpr int WN = 1;   // each thread computes 1 column (distributed across 32 threads per warp group)

extern "C" __global__ void gemm_kernel(
    const __grid_constant__ CUtensorMap tma_a_desc,
    const __grid_constant__ CUtensorMap tma_b_desc,
    __nv_bfloat16* dA,
    __nv_bfloat16* dB,
    __nv_bfloat16* dC,
    int M, int N, int K)
{
    extern __shared__ char smem_raw[];
    
    // Align shared memory
    uint64_t* mbar_a = reinterpret_cast<uint64_t*>(
        reinterpret_cast<char*>(smem_raw));
    uint64_t* mbar_b = reinterpret_cast<uint64_t*>(
        reinterpret_cast<char*>(smem_raw) + 64);
    
    // A_smem: BM x BK bf16 = 128 * 32 * 2 = 8KB
    // B_smem: BN x BK bf16 = 128 * 32 * 2 = 8KB  
    // Use offset from end of smem to align properly
    int smem_offset = 128; // Leave space for 2 mbarriers
    
    __nv_bfloat16* As = reinterpret_cast<__nv_bfloat16*>(
        reinterpret_cast<char*>(smem_raw) + smem_offset);
    __nv_bfloat16* Bs = reinterpret_cast<__nv_bfloat16*>(
        As + BM * BK);
    
    int tid = threadIdx.x;
    
    // Initialize mbarriers (one barrier for A load completion, one for B)
    if (tid == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], 1;" 
                     :: "r"(static_cast<unsigned>(__cvta_generic_to_shared(mbar_a))));
        asm volatile("mbarrier.init.shared.b64 [%0], 1;" 
                     :: "r"(static_cast<unsigned>(__cvta_generic_to_shared(mbar_b))));
    }
    __syncthreads();
    
    int m_block = static_cast<int>(blockIdx.y) * BM;
    int n_block = static_cast<int>(blockIdx.x) * BN;
    
    // Which rows this thread is responsible for (BM / (THREADS_PER_BLOCK / WN) )
    // 256 threads, each owns 8 rows -> 256*8 = 2048 > 128, so we need different mapping
    // Actually: 128 rows / 256 threads... only half threads are active per row range
    // Better: 128 rows, 256 threads. Each thread handles multiple column tiles.
    // Simplest: each thread handles WM=8 rows, but we have only 128/8=16 groups of rows
    // So 256/16 = 16 threads per row group
    
    // Layout: tid -> row_base = (tid / 32) * 8, col_offset = tid % 32
    int warp_idx = tid / 32;          // 0..7
    int lane_id = tid % 32;           // 0..31
    int row_group = warp_idx;         // which group of 8 rows: 0..7, covering rows 0..63
                                       // We need all 128 rows -> need more coverage
                               // Actually blockDim.x=256, we process BN=128 columns
    // Each thread does WM=8 rows * WN columns. Total capacity = 256*8 = 2048 row-slices
    // We need 128 rows * 128 cols = 16384 outputs per block.
    // So each thread does 16384/256 = 64 outputs = e.g. 8 rows * 8 cols
    // But that's too complex. Let's use a simpler scheme.
    
    // Simpler: Each thread owns 8 consecutive rows. But we have 256 threads and 128 rows.
    // So pairs of threads share each row's work? No. Let's do it differently:
    // tid computes output rows [row_start, row_start+WM) where row_start = (tid % 16) * 8
    // That gives 16 unique row starts, with 16 threads per row-start.
    // Each such thread pair covers some number of columns.
    
    // Even simpler: just use 2D-like indexing within the block
    // row_idx in [0, BM): each handled by multiple threads
    // col_idx in [0, BN): each handled by multiple threads
    // With 256 threads:
    // thread_col_id = tid / 16  -> 0..15 (columns groups)
    // thread_row_id = tid % 16  -> 0..15 (row groups)
    // Each row group = 8 rows (15*8=120, close to 128... need adjustment)
    
    // Let me use yet another scheme that clearly maps:
    // 256 threads cover 128 rows with 8 rows per thread-group-of-2
    // Actually simplest: just loop over all rows this thread contributes to
    
    // FINAL SIMPLE APPROACH:
    // For each GEMM step, thread `tid` accumulates contributions into partial result for:
    // - specific rows in [0, BM): determined by tid
    // - specific cols in [0, BN): determined by tid
    // Since BN=128 and we want good utilization: assign each thread 8 columns
    // 128 cols / 8 = 16 thread slots for columns, repeated 16 times across 256 threads
    // Row assignment: remaining 16 factor -> 128/16 = 8 rows per thread
    
    // So: col_slot = tid / 16 (0..15), row_slot = tid % 16 (0..15)
    // This thread handles rows [row_slot*8, row_slot*8+8) and cols [col_slot*8, col_slot*8+8)
    // Wait that's 64 outputs per thread which seems right: 256*64 = 16384
    
    int col_slot = tid / 16;
    int row_slot = tid % 16;
    int my_row_start = row_slot * 8;  // rows 0..127 covered by slots 0..15
    int my_col_start = col_slot * 8;  // cols 0..127 covered by slots 0..15
    
    float reg_acc[8][8] = {};
    
    int num_k_tiles = K / BK;
    
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        int k_off = k_tile * BK;
        
        // Issue TMA load for A tile: [m_block, m_block+BM) x [k_off, k_off+BK)
        if (tid == 0 && m_block < M) {
            unsigned bar_addr = static_cast<unsigned>(__cvta_generic_to_shared(mbar_a));
            unsigned asmem_addr = static_cast<unsigned>(__cvta_generic_to_shared(As));
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%2, %3}], [%4];"
                :: "r"(asmem_addr),
                   "l"(reinterpret_cast<uint64_t>(&tma_a_desc)),
                   "r"(0), "r"(k_off),
                   "r"(bar_addr) : "memory");
        }
        
        // Issue TMA load for B tile: [n_block, n_block+BN) x [k_off, k_off+BK)
        if (tid == 0 && n_block < N) {
            unsigned bar_addr = static_cast<unsigned>(__cvta_generic_to_shared(mbar_b));
            unsigned bsmem_addr = static_cast<unsigned>(__cvta_generic_to_shared(Bs));
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%2, %3}], [%4];"
                :: "r"(bsmem_addr),
                   "l"(reinterpret_cast<uint64_t>(&tma_b_desc)),
                   "r"(0), "r"(k_off),
                   "r"(bar_addr) : "memory");
        }
        
        __threadfence();
        
        // Wait for both barriers
        if (m_block < M || n_block < N) {
            if (tid == 0) {
                unsigned bar_a_addr = static_cast<unsigned>(__cvta_generic_to_shared(mbar_a));
                unsigned bar_b_addr = static_cast<unsigned>(__cvta_generic_to_shared(mbar_b));
                
                // Wait for A
                if (m_block < M) {
                    asm volatile(
                        "{\n.reg .pred p;\n"
                        "wait_a%=: \n"
                        "mbarrier.try_wait.parity.shared.b64 p, [%0], 0;\n"
                        "@!p bra wait_a%=;\n}\n"
                        :: "r"(bar_a_addr));
                }
                // Wait for B
                if (n_block < N) {
                    asm volatile(
                        "{\n.reg .pred p;\n"
                        "wait_b%=: \n"
                        "mbarrier.try_wait.parity.shared.b64 p, [%0], 0;\n"
                        "@!p bra wait_b%=;\n}\n"
                        :: "r"(bar_b_addr));
                }
            }
        }
        __syncthreads();
        
        // Compute local GEMM fragments
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            int row = my_row_start + r;
            if (row >= BM || (m_block + row) >= M) continue;
            
            #pragma unroll
            for (int c = 0; c < 8; ++c) {
                int col = my_col_start + c;
                if (col >= BN || (n_block + col) >= N) continue;
                
                #pragma unroll
                for (int k = 0; k < BK; ++k) {
                    float a_val = __bfloat162float(As[row * BK + k]);
                    float b_val = __bfloat162float(Bs[col * BK + k]);
                    reg_acc[r][c] += a_val * b_val;
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store results
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            int row = my_row_start + r;
            int col = my_col_start + c;
            if (row >= BM || col >= BN) continue;
            int mr = m_block + row;
            int nc = n_block + col;
            if (mr < M && nc < N) {
                dC[static_cast<long long>(mr) * N + nc] = 
                    __float2bfloat16(reg_acc[r][c]);
            }
        }
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    int M = static_cast<int>(A.size(0));
    int N = static_cast<int>(B.size(0));
    int K = static_cast<int>(A.size(1));
    
    __nv_bfloat16* d_A = static_cast<__nv_bfloat16*>(A.data_ptr());
    __nv_bfloat16* d_B = static_cast<__nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* d_C = static_cast<__nv_bfloat16*>(C.data_ptr());
    
    // Grid: one block per output tile
    int grid_x = (N + BN - 1) / BN;
    int grid_y = (M + BM - 1) / BM;
    dim3 grid(grid_x, grid_y);
    dim3 block(THREADS_PER_BLOCK);
    
    // Shared memory: 2 mbarrier(8B) + padding + As + Bs
    int smem_size = 128 + BM * BK * sizeof(__nv_bfloat16) + BN * BK * sizeof(__nv_bfloat16);
    smem_size = (smem_size + 255) & ~255;
    
    // Create TMA descriptor for A: [M x K] loading [BM x BK]
    CUtensorMap tma_a;
    cuuint64_t globalDimA[2] = {static_cast<cuuint64_t>(K), static_cast<cuuint64_t>(M)};
    cuuint64_t globalStrideA[1] = {static_cast<cuuint64_t>(K) * sizeof(__nv_bfloat16)};
    cuuint32_t boxDimA[2] = {BK, BM};
    cuuint32_t elemStrideA[2] = {1, 1};
    
    CUresult ret = cuTensorMapEncodeTiled(
        &tma_a,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        d_A,
        globalDimA, globalStrideA,
        boxDimA, elemStrideA,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (ret != CUDA_SUCCESS) {
        const char* err_str = nullptr;
        cuGetErrorName(ret, &err_str);
        fprintf(stderr, "TMA A encode failed: %s\n", err_str ? err_str : "unknown");
    }
    
    // Create TMA descriptor for B: [N x K] loading [BN x BK]
    CUtensorMap tma_b;
    cuuint64_t globalDimB[2] = {static_cast<cuuint64_t>(K), static_cast<cuuint64_t>(N)};
    cuuint64_t globalStrideB[1] = {static_cast<cuuint64_t>(K) * sizeof(__nv_bfloat16)};
    cuuint32_t boxDimB[2] = {BK, BN};
    cuuint32_t elemStrideB[2] = {1, 1};
    
    ret = cuTensorMapEncodeTiled(
        &tma_b,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        d_B,
        globalDimB, globalStrideB,
        boxDimB, elemStrideB,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (ret != CUDA_SUCCESS) {
        const char* err_str = nullptr;
        cuGetErrorName(ret, &err_str);
        fprintf(stderr, "TMA B encode failed: %s\n", err_str ? err_str : "unknown");
    }
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    // Launch kernel
    gemm_kernel<<<grid, block, smem_size, stream>>>(
        tma_a, tma_b, d_A, d_B, d_C, M, N, K);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace gemm_blackwell

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_blackwell::run);