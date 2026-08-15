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
constexpr int WM = 8;   
constexpr int WN = 8;   

__device__ __forceinline__ void mbarrier_wait(uint64_t* bar) {
    unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{\n\t"
        ".reg .pred P;\n\t"
        "WAIT_%=:\n\t"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], 0;\n\t"
        "@!P bra WAIT_%=;\n\t"
        "}\n"
        : : "r"(addr) : "memory");
}

extern "C" __global__ void gemm_kernel(
    const __grid_constant__ CUtensorMap tma_a_desc,
    const __grid_constant__ CUtensorMap tma_b_desc,
    __nv_bfloat16* dA,
    __nv_bfloat16* dB,
    __nv_bfloat16* dC,
    int M, int N, int K)
{
    extern __shared__ char smem_raw[];

    // Layout: [mbarrier_8B | pad_504B | A_smem | B_smem]
    // Total barriers section: 512 bytes for alignment
    uint64_t* smem_mbar_a = reinterpret_cast<uint64_t*>(smem_raw);
    // Skip 512 bytes to get proper alignment for next section  
    __nv_bfloat16* As = reinterpret_cast<__nv_bfloat16*>(smem_raw + 512);
    __nv_bfloat16* Bs = reinterpret_cast<__nv_bfloat16*>(As + BM * BK);

    int tid = threadIdx.x;

    // Initialize barrier with expect_count=1
    if (tid == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], 1;" 
                     :: "r"(static_cast<unsigned>(__cvta_generic_to_shared(smem_mbar_a))));
    }
    __syncthreads();

    int m_block = static_cast<int>(blockIdx.y) * BM;
    int n_block = static_cast<int>(blockIdx.x) * BN;

    // Only valid blocks proceed with GEMM
    bool block_valid = (m_block < M) && (n_block < N);

    int row_grp = tid / 16;          
    int col_slot = tid % 16;
    
    int my_row_base = row_grp * WM;     
    int my_col_base = col_slot * WN;

    float acc[WM][WN];
    #pragma unroll
    for (int r = 0; r < WM; ++r)
        #pragma unroll
        for (int c = 0; c < WN; ++c)
            acc[r][c] = 0.0f;

    int num_k_tiles = (K + BK - 1) / BK;

    unsigned smem_bar_a = static_cast<unsigned>(__cvta_generic_to_shared(smem_mbar_a));
    unsigned smem_as = static_cast<unsigned>(__cvta_generic_to_shared(As));
    unsigned smem_bs = static_cast<unsigned>(__cvta_generic_to_shared(Bs));

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        int k_off = kt * BK;
        int bk_eff = min(BK, K - k_off);

        // Issue TMA for A tile (shared mbarrier for both A and B)
        if (tid == 0 && block_valid) {
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%2, %3}], [%4];"
                : 
                : "r"(smem_as),
                  "l"(reinterpret_cast<uint64_t>(&tma_a_desc)),
                  "r"(0),       
                  "r"(k_off),   
                  "r"(smem_bar_a)
                : "memory");
            
            // Issue TMA for B tile
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%2, %3}], [%4];"
                :
                : "r"(smem_bs),
                  "l"(reinterpret_cast<uint64_t>(&tma_b_desc)),
                  "r"(0),
                  "r"(k_off),
                  "r"(smem_bar_a)
                : "memory");
        }

        // tid==0 arrives to signal completion of thread-side duties
        if (tid == 0 && block_valid) {
            asm volatile("mbarrier.arrive.shared.b64 _, [%0];" 
                         : : "r"(smem_bar_a) : "memory");
        }

        // All threads in valid blocks wait
        if (block_valid) {
            mbarrier_wait(smem_mbar_a);
        }
        __syncthreads();

        // Compute GEMM fragment
        if (block_valid) {
            #pragma unroll
            for (int r = 0; r < WM; ++r) {
                int ar = my_row_base + r;
                
                #pragma unroll
                for (int c = 0; c < WN; ++c) {
                    int ac = my_col_base + c;
                    bool col_valid = (ac < BN) && ((n_block + ac) < N);
                    
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
        }

        __syncthreads();
    }

    // Store results
    if (block_valid) {
        #pragma unroll
        for (int r = 0; r < WM; ++r) {
            int mr = my_row_base + r;
            if (mr >= BM) continue;
            int m_global = m_block + mr;

            #pragma unroll
            for (int c = 0; c < WN; ++c) {
                int nc = my_col_base + c;
                if (nc >= BN || (n_block + nc) >= N) continue;
                int n_global = n_block + nc;
                dC[static_cast<long long>(m_global) * N + n_global] = 
                    __float2bfloat16(acc[r][c]);
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

    int grid_x = (N + BN - 1) / BN;
    int grid_y = (M + BM - 1) / BM;
    dim3 grid(grid_x, grid_y);
    dim3 block(BLOCK_THREADS);

    // SM: [512B mbarrier_section | As(8192B) | Bs(8192B)]
    int smem_size = 512 + BM * BK * sizeof(__nv_bfloat16) + BN * BK * sizeof(__nv_bfloat16);
    smem_size = (smem_size + 127) & ~127;

    // Create TMA descriptors - NO SWIZZLE for simplicity
    CUtensorMap tma_a;
    cuuint64_t globalDimA[2] = {static_cast<cuuint64_t>(K), static_cast<cuuint64_t>(M)};
    cuuint64_t globalStrideA[1] = {static_cast<cuuint64_t>(K) * sizeof(__nv_bfloat16)};
    cuuint32_t boxDimA[2] = {static_cast<cuuint32_t>(BK), static_cast<cuuint32_t>(BM)};
    cuuint32_t elemStrideA[2] = {1, 1};

    CUresult ret = cuTensorMapEncodeTiled(
        &tma_a,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        d_A,
        globalDimA, globalStrideA,
        boxDimA, elemStrideA,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_64B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (ret != CUDA_SUCCESS) {
        const char* msg = nullptr;
        cuGetErrorName(ret, &msg);
        fprintf(stderr, "TMA A failed: %s\n", msg ? msg : "?");
    }

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
        CU_TENSOR_MAP_SWIZZLE_NONE,
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