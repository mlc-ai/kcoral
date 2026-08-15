#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t e = (call); \
        if (e != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while(0)

namespace tvm_ffi_gemm_bf16 {

static constexpr int BLOCK_M = 64;
static constexpr int BLOCK_N = 64;
static constexpr int TM = 8;
static constexpr int TN = 8;
static constexpr int K_STRIDE = 16;

__global__ void gemm_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K) 
{
    extern __shared__ char smem_buf[];
    
    // shared_A: BLOCK_M x K_STRIDE, shared_BT: BLOCK_N x K_STRIDE
    __nv_bfloat16* shared_A   = reinterpret_cast<__nv_bfloat16*>(smem_buf);
    __nv_bfloat16* shared_BT  = shared_A + BLOCK_M * K_STRIDE;
    
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    
    int block_m_start = blockIdx.y * BLOCK_M;
    int block_n_start = blockIdx.x * BLOCK_N;
    
    // Accumulators: each thread maintains TM x TN FP32 values
    float acc[TM][TN] = {};
    
    for (int kb = 0; kb < K; kb += K_STRIDE) {
        int ks = min(K_STRIDE, K - kb);
        
        // === Load A tile: block_rows x K_STRIDE ===
        // Each thread loads one row of the A k-block
        for (int i = tid_y; i < TM; i += blockDim.y) {
            int global_i = block_m_start + i;
            __nv_bfloat16* dst = &shared_A[i * K_STRIDE];
            if (global_i < M) {
                const __nv_bfloat16* src = A + (size_t)global_i * K + kb;
                for (int j = 0; j < ks; ++j) {
                    dst[j] = src[j];
                }
                for (int j = ks; j < K_STRIDE; ++j) {
                    dst[j] = __float2bfloat16(0.0f);
                }
            } else {
                for (int j = 0; j < K_STRIDE; ++j) {
                    dst[j] = __float2bfloat16(0.0f);
                }
            }
        }
        
        // === Load B^T tile: block_cols x K_STRIDE ===
        // B is [N, K] row-major; we need B^T[k,n] = B[n,k]
        // Each thread loads one row of the BT k-block (one column of B block)
        for (int i = tid_x; i < TN; i += blockDim.x) {
            int global_j = block_n_start + i;
            __nv_bfloat16* dst = &shared_BT[i * K_STRIDE];
            if (global_j < N) {
                const __nv_bfloat16* src = B + (size_t)global_j * K + kb;
                for (int j = 0; j < ks; ++j) {
                    dst[j] = src[j];
                }
                for (int j = ks; j < K_STRIDE; ++j) {
                    dst[j] = __float2bfloat16(0.0f);
                }
            } else {
                for (int j = 0; j < K_STRIDE; ++j) {
                    dst[j] = __float2bfloat16(0.0f);
                }
            }
        }
        
        __syncthreads();
        
        // Read into registers for compute (avoids repeated SMEM banking)
        __nv_bfloat16 a_reg[TM][K_STRIDE];
        __nv_bfloat16 b_reg[TN][K_STRIDE];
        
        for (int i = 0; i < TM; ++i) {
            for (int j = 0; j < K_STRIDE; ++j) {
                a_reg[i][j] = shared_A[i * K_STRIDE + j];
            }
        }
        for (int i = 0; i < TN; ++i) {
            for (int j = 0; j < K_STRIDE; ++j) {
                b_reg[i][j] = shared_BT[i * K_STRIDE + j];
            }
        }
        
        // Multiply-accumulate: C += A[ktile] * BT[ktile]^T
        for (int kk = 0; kk < K_STRIDE; ++kk) {
            for (int i = 0; i < TM; ++i) {
                float fa = static_cast<float>(a_reg[i][kk]);
                for (int j = 0; j < TN; ++j) {
                    float fb = static_cast<float>(b_reg[j][kk]);
                    acc[i][j] = fma(fa, fb, acc[i][j]);
                }
            }
        }
        
        __syncthreads();
    }
    
    // === Store results back to global memory ===
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            int global_i = block_m_start + i;
            int global_j = block_n_start + j;
            if (global_i < M && global_j < N) {
                C[(size_t)global_i * N + global_j] = __float2bfloat16(acc[i][j]);
            }
        }
    }
}

void run(TensorView A, TensorView B, TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    int M = static_cast<int>(A.size(0));
    int N = 7168;
    int K = 5120;
    
    const __nv_bfloat16* A_ptr = static_cast<const __nv_bfloat16*>(A.data_ptr());
    const __nv_bfloat16* B_ptr = static_cast<const __nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16*       C_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());
    
    dim3 block(TM, TN, 1);  // 8x8 = 64 threads
    
    int grid_x = (N + BLOCK_N - 1) / BLOCK_N;
    int grid_y = (M + BLOCK_M - 1) / BLOCK_M;
    dim3 grid(grid_x, grid_y, 1);
    
    // Shared memory: 2 * (BLOCK_M + BLOCK_N) * K_STRIDE bf16 elements
    size_t smem_bytes = 2ULL * (BLOCK_M + BLOCK_N) * K_STRIDE * sizeof(__nv_bfloat16);
    
    cudaStream_t stream = nullptr;
    TVMFFIDataTypeCode dtype_code;
    int device_id;
    {
        TVMPODValue_ dv = TVMDLDevice2VDevice(A.device());
        dtype_code = dv.v_int64 & 0xFFFF;
        device_id = dv.v_int64 >> 32;
        stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(dtype_code, device_id));
    }
    
    gemm_kernel<<<grid, block, static_cast<size_t>(smem_bytes), stream>>>(
        A_ptr, B_ptr, C_ptr, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_gemm_bf16::run);

}  // namespace tvm_ffi_gemm_bf16