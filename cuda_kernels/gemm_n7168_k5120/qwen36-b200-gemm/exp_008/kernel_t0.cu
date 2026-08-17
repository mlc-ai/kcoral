#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t e = (call);                                        \
    if (e != cudaSuccess) {                                        \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(e), __FILE__, __LINE__);        \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace gemm_sm100 {

static constexpr int BM = 64;    // Block size in M dimension
static constexpr int BN = 64;    // Block size in N dimension
static constexpr int BK = 32;    // Block size in K dimension
static constexpr int B_PAD = 16; // Padding for B shared memory to avoid bank conflicts
static constexpr int THREADS_X = BN / 4; // 16 threads in X, each handles 4 columns
static constexpr int THREADS_Y = BM;     // 64 threads in Y, each handles 1 row

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 val) {
    float f;
    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(f) : "h"(*reinterpret_cast<__half*>(&val)));
    return f;
}

__global__ void gemm_blocked_bf16_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    uint64_t M, uint64_t N, uint64_t K)
{
    // Shared memory: As[BM][BK] + Bs[BN][BK+B_PAD]
    extern __shared__ __nv_bfloat16 smem[];
    
    __nv_bfloat16* __restrict__ As = smem;
    __nv_bfloat16* __restrict__ Bs = smem + BM * BK;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Global row for this thread
    uint64_t g_row = blockIdx.y * BM + ty;
    // Base global column for this block
    uint64_t g_col_base = blockIdx.x * BN;
    
    // Each thread computes 4 dot products (for 4 consecutive columns)
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint64_t cols[4] = {g_col_base + 4*tx + 0,
                        g_col_base + 4*tx + 1,
                        g_col_base + 4*tx + 2,
                        g_col_base + 4*tx + 3};
    
    uint64_t num_k_tiles = (K + BK - 1) / BK;
    
    for (uint64_t t = 0; t < num_k_tiles; ++t) {
        uint64_t k_start = t * BK;
        
        // ===================== Load A tile =====================
        // As[ty][local_k] = A[g_row][k_start + local_k]
        // Stride 4: each thread loads 4 consecutive K elements
        bool a_valid_row = (g_row < M);
        uint64_t a_base_ptr = a_valid_row ? g_row * K : 0;
        
        #pragma unroll
        for (int v = 0; v < 4; v++) {
            int lk = 4 * tx + v;
            if (lk < BK) {
                uint64_t a_idx = k_start + lk;
                if (a_valid_row && a_idx < K) {
                    As[ty * BK + lk] = A[a_base_ptr + a_idx];
                } else {
                    As[ty * BK + lk] = __float2bfloat16_rn(0.0f);
                }
            }
        }
        
        // ===================== Load B tile =====================
        // Bs[local_n][local_k] = B[cols[local_n]][k_start + local_k]
        // Each of 4 columns needs BK elements
        #pragma unroll
        for (int cn = 0; cn < 4; cn++) {
            if (cols[cn] < N) {
                uint64_t b_row_offset = cols[cn] * K;
                for (int vk = 0; vk < BK; vk += 4) {
                    int lk = vk;
                    uint64_t b_idx = k_start + lk;
                    if (b_idx + 3 < K) {
                        Bs[(cn * BK + lk) * (BK + B_PAD) + 0] = B[b_row_offset + b_idx + 0];
                        Bs[(cn * BK + lk) * (BK + B_PAD) + 1] = B[b_row_offset + b_idx + 1];
                        Bs[(cn * BK + lk) * (BK + B_PAD) + 2] = B[b_row_offset + b_idx + 2];
                        Bs[(cn * BK + lk) * (BK + B_PAD) + 3] = B[b_row_offset + b_idx + 3];
                    } else {
                        Bs[(cn * BK + lk) * (BK + B_PAD) + 0] = __float2bfloat16_rn(0.0f);
                        Bs[(cn * BK + lk) * (BK + B_PAD) + 1] = __float2bfloat16_rn(0.0f);
                        Bs[(cn * BK + lk) * (BK + B_PAD) + 2] = __float2bfloat16_rn(0.0f);
                        Bs[(cn * BK + lk) * (BK + B_PAD) + 3] = __float2bfloat16_rn(0.0f);
                    }
                }
            }
        }
        
        __syncthreads();
        
        // ===================== Compute =====================
        // C[g_row][cols[cn]] += sum_{k=0}^{BK-1} A[g_row][k_start+k] * B[cols[cn]][k_start+k]
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            __nv_bfloat16 a_val = As[ty * BK + k];
            #pragma unroll
            for (int cn = 0; cn < 4; cn++) {
                __nv_bfloat16 b_val = Bs[(cn * BK + k) * (BK + B_PAD)];
                acc[cn] += bf16_to_f32(a_val) * bf16_to_f32(b_val);
            }
        }
        
        __syncthreads();
    }
    
    // ===================== Store =====================
    // Only store if within bounds
    #pragma unroll
    for (int cn = 0; cn < 4; cn++) {
        if (g_row < M && cols[cn] < N) {
            C[g_row * N + cols[cn]] = __float2bfloat16_rn(acc[cn]);
        }
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
  CUDA_CHECK(cudaSetDevice(A.device().device_id));
  
  uint64_t M = A.size(0);
  uint64_t N = 7168;
  uint64_t K = 5120;
  
  dim3 block(THREADS_X, THREADS_Y); // 16 × 64 = 1024 threads
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  
  // Shared memory: As[BM][BK] + Bs[BN][BK + B_PAD]
  size_t sm_bytes = (BM * BK + BN * (BK + B_PAD)) * sizeof(__nv_bfloat16);
  
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
  
  gemm_blocked_bf16_kernel<<<grid, block, sm_bytes, stream>>>(
      static_cast<const __nv_bfloat16*>(A.data_ptr()),
      static_cast<const __nv_bfloat16*>(B.data_ptr()),
      static_cast<__nv_bfloat16*>(C.data_ptr()),
      M, N, K);
      
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_sm100::run);

}  // namespace gemm_sm100