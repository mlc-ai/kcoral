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
constexpr int BLOCK_THREADS = 256;
// Each thread computes WM rows × WN cols per iteration
constexpr int WM = 8;   // rows per thread
constexpr int WN = 1;   // columns per thread (spread across 32 lanes)

extern "C" __global__ void gemm_kernel(
    const __grid_constant__ CUtensorMap tma_a_desc,
    const __grid_constant__ CUtensorMap tma_b_desc,
    __nv_bfloat16* dA,
    __nv_bfloat16* dB,
    __nv_bfloat16* dC,
    int M, int N, int K)
{
    extern __shared__ char smem_raw[];

    // mbarrier at offset 0 (8 bytes), aligned
    uint64_t* smem_mbar_a = reinterpret_cast<uint64_t*>(smem_raw);
    // pad to next 128-byte boundary for A_smem
    char* after_mbar = smem_raw + 128;
    // mbarrier for B at +128
    uint64_t* smem_mbar_b = reinterpret_cast<uint64_t*>(after_mbar);
    // pad to next 128-byte boundary for B_smem
    char* after_mbar_b = after_mbar + 128;

    __nv_bfloat16* As = reinterpret_cast<__nv_bfloat16*>(after_mbar_b);
    // As takes BM*BK*2 bytes = 8192
    // Bs starts after As, aligned to 128
    int as_bytes = BM * BK * sizeof(__nv_bfloat16); // 8192
    char* bs_ptr = reinterpret_cast<char*>(As + BM * BK);
    // Round up to 128-byte alignment
    int bs_aligned_offset = (as_bytes + 127) & ~127;
    __nv_bfloat16* Bs = reinterpret_cast<__nv_bfloat16*>(after_mbar_b + bs_aligned_offset);

    int tid = threadIdx.x;

    // Initialize barriers: thread 0 only
    if (tid == 0) {
        // We use expect_tx: no thread arrivals, just transaction completion
        asm volatile("mbarrier.init.shared.b64 [%0], 0;" 
                     :: "r"(static_cast<unsigned>(__cvta_generic_to_shared(smem_mbar_a))));
        asm volatile("mbarrier.init.shared.b64 [%0], 0;" 
                     :: "r"(static_cast<unsigned>(__cvta_generic_to_shared(smem_mbar_b))));
    }
    __syncthreads();

    int m_block = static_cast<int>(blockIdx.y) * BM;
    int n_block = static_cast<int>(blockIdx.x) * BN;

    // Thread tile assignment:
    // row_group_idx in [0, 16): covers rows [row_group_idx*8, row_group_idx*8+8)
    // col_slot in [0, 16): covers cols [col_slot*1, col_slot*1+1) within BN
    // 16*16 = 256 threads total
    int row_grp = tid / 16;          // 0..15 -> rows 0..127
    int col_slot = tid % 16;         // 0..15 -> cols 0..15 in BN space, covering 128 cols by repeating
    
    // Actually let's redo: tid goes 0..255, we want 128 rows * 128 cols = 16384 outputs
    // Each thread produces 16384/256 = 64 outputs
    // 64 = 8 rows * 8 cols
    // row_grp = tid / 16 -> 0..15 (each group handles 8 rows)
    // Intra-group: tid % 16 -> 0..15 maps to 8 column slots (cols 0,8,16,...,120)
    // No wait, that's only 8 cols * 8 rows = 64 per thread from 16 slot values.
    // Let's use: tid % 16 = col_slot, and each col_slot handles 8 consecutive columns.
    // col_slot 0 -> cols 0..7, col_slot 1 -> cols 8..15, ..., col_slot 15 -> cols 120..127

    int my_row_base = row_grp * WM;     // 0,8,16,...,120
    int my_col_base = col_slot * WN;    // Actually let's make it 8 cols per thread
    // With 16 col_slots: 16*8=128 cols covered, good.
    // Wait, then row_grp*tid/16 gives 16 values for 128 rows. Each thread gets 8 rows.
    // And tid%16 gives 16 values for 128 cols. Need 8 cols each = 128/16=8. OK.
    int wn_per_thread = 8;
    my_col_base = col_slot * wn_per_thread;

    // Accumulators: 8 rows * 8 cols
    float acc[WM][wn_per_thread];
    #pragma unroll
    for (int r = 0; r < WM; ++r)
        #pragma unroll
        for (int c = 0; c < wn_per_thread; ++c)
            acc[r][c] = 0.0f;

    int num_k_tiles = (K + BK - 1) / BK;

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        int k_off = kt * BK;
        int bk_eff = min(BK, K - k_off);

        unsigned smem_bar_a = static_cast<unsigned>(__cvta_generic_to_shared(smem_mbar_a));
        unsigned smem_bar_b = static_cast<unsigned>(__cvta_generic_to_shared(smem_mbar_b));
        unsigned smem_as = static_cast<unsigned>(__cvta_generic_to_shared(As));
        unsigned smem_bs = static_cast<unsigned>(__cvta_generic_to_shared(Bs));

        // Issue TMA for A tile at global coords (row=0, col=k_off) relative to tensor map origin
        // Tensor map has globalDim = {K, M} with strides {K*2, ...}
        // boxDim = {BK, BM}
        if (tid == 0 && m_block < M) {
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%2, %3}], [%4];"
                : 
                : "r"(smem_as),
                  "l"(reinterpret_cast<uint64_t>(&tma_a_desc)),
                  "r"(0),       // coord in dim 0 (inner=K): starts at 0 within box
                  "r"(k_off),   // coord in dim 1 (outer=M) — actually this is global offset
                  "r"(smem_bar_a)
                : "memory");
        }

        // Issue TMA for B tile
        if (tid == 0 && n_block < N) {
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%2, %3}], [%4];"
                :
                : "r"(smem_bs),
                  "l"(reinterpret_cast<uint64_t>(&tma_b_desc)),
                  "r"(0),
                  "r"(k_off),
                  "r"(smem_bar_b)
                : "memory");
        }

        // Barrier wait
        if (tid == 0) {
            if (m_block < M) {
                asm volatile(
                    "{\n.reg .pred p;\n.L_wait_a%=: "
                    "mbarrier.try_wait.parity.shared.b64 p, [%0], 0;\n"
                    "@!p bra .L_wait_a%=;\n}\n"
                    : : "r"(smem_bar_a));
            }
            if (n_block < N) {
                asm volatile(
                    "{\n.reg .pred p;\n.L_wait_b%=: "
                    "mbarrier.try_wait.parity.shared.b64 p, [%0], 0;\n"
                    "@!p bra .L_wait_b%=;\n}\n"
                    : : "r"(smem_bar_b));
            }
        }
        __syncthreads();

        // Compute GEMM fragment
        #pragma unroll
        for (int r = 0; r < WM; ++r) {
            int ar = my_row_base + r;
            bool row_valid = (ar < BM) && (m_block + ar < M);
            
            if (!row_valid) continue;

            #pragma unroll
            for (int c = 0; c < wn_per_thread; ++c) {
                int ac = my_col_base + c;
                bool col_valid = (ac < BN) && (n_block + ac < N);
                
                if (!col_valid) continue;

                float sum = 0.0f;
                #pragma unroll
                for (int k = 0; k < bk_eff; ++k) {
                    float va = __bfloat162float(As[ar * BK + k]);
                    float vb = __bfloat162float(Bs[ac * BK + k]);
                    sum += va * vb;
                }
                acc[r][c] += sum;
            }
        }

        __syncthreads();
    }

    // Store results
    #pragma unroll
    for (int r = 0; r < WM; ++r) {
        int mr = my_row_base + r;
        if (mr >= BM || m_block + mr >= M) continue;
        int m_global = m_block + mr;

        #pragma unroll
        for (int c = 0; c < wn_per_thread; ++c) {
            int nc = my_col_base + c;
            if (nc >= BN || n_block + nc >= N) continue;
            int n_global = n_block + nc;
            dC[static_cast<long long>(m_global) * N + n_global] = 
                __float2bfloat16(acc[r][c]);
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

    int grid_x = (N + BN - 1) / BN;
    int grid_y = (M + BM - 1) / BM;
    dim3 grid(grid_x, grid_y);
    dim3 block(BLOCK_THREADS);

    // SM layout: [128B mbar_a | 128B mbar_b | As(BM*BK*2) | Bs(BN*BK*2)]
    int as_bytes = BM * BK * sizeof(__nv_bfloat16);
    int bs_aligned = (as_bytes + 127) & ~127;
    int smem_size = 128 + 128 + bs_aligned + BN * BK * sizeof(__nv_bfloat16);
    smem_size = (smem_size + 127) & ~127;

    // Create TMA descriptors
    // A: [M x K], Row-major. Load [BM x BK] tile.
    // Tensor rank 2, dims: {dim0=K(inner), dim1=M(outer)}
    // Inner dim = K, outer dim = M
    // boxDim: {BK, BM}
    CUtensorMap tma_a;
    cuuint64_t globalDimA[2] = {static_cast<cuuint64_t>(K), static_cast<cuuint64_t>(M)};
    cuuint64_t globalStrideA[1] = {static_cast<cuuint64_t>(K) * sizeof(__nv_bfloat16)};
    cuuint32_t boxDimA[2] = {static_cast<cuuint32_t>(BK), static_cast<cuuint32_t>(BM)};
    cuuint32_t elemStrideA[2] = {1, 1};

    // Use 64B swizzle for safer alignment (requires 512B aligned dest, stride divisible by 64B)
    // With BM=128, BK=32 bf16: inner slab = 32*2=64B ✓
    CUresult ret = cuTensorMapEncodeTiled(
        &tma_a,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        d_A,
        globalDimA, globalStrideA,
        boxDimA, elemStrideA,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_64B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_64B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (ret != CUDA_SUCCESS) {
        const char* msg = nullptr;
        cuGetErrorName(ret, &msg);
        fprintf(stderr, "TMA A failed: %s\n", msg ? msg : "?");
    }

    // B: [N x K], Row-major. Load [BN x BK] tile.
    CUtensorMap tma_b;
    cuuint64_t globalDimB[2] = {static_cast<cuuint64_t>(K), static_cast<cuuint64_t>(N)};
    cuuint64_t globalStrideB[1] = {static_cast<cuuint64_t>(K) * sizeof(__nv_bfloat16)};
    cuuint32_t boxDimB[2] = {static_cast<cuuint32_t>(BK), static_cast<cuuint32_t>(BN)};
    cuuint32_t elemStrideB[2] = {1, 1};

    ret = cuTensorMapEncodeTiled(
        &tma_b,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        d_B,
        globalDimB, globalStrideB,
        boxDimB, elemStrideB,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_64B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_64B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (ret != CUDA_SUCCESS) {
        const char* msg = nullptr;
        cuGetErrorName(ret, &msg);
        fprintf(stderr, "TMA B failed: %s\n", msg ? msg : "?");
    }

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    gemm_kernel<<<grid, block, smem_size, stream>>>(
        tma_a, tma_b, d_A, d_B, d_C, M, N, K);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace gemm_blackwell

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_blackwell::run);