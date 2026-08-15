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
static constexpr int B_PAD = 8;  // Padding for B shared memory
static constexpr int TX = BN / 4; // 16 threads in X, each handles 4 columns
static constexpr int TY = BM;     // 64 threads in Y, each handles 1 row

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

__global__ void gemm_blocked_bf16_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    uint64_t M, uint64_t N, uint64_t K)
{
    extern __shared__ __nv_bfloat16 smem[];
    
    __nv_bfloat16* __restrict__ As = smem;
    __nv_bfloat16* __restrict__ Bs = smem + BM * BK;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    uint64_t g_row = blockIdx.y * BM + ty;
    uint64_t g_col_base = blockIdx.x * BN;
    
    bool row_valid = (g_row < M);
    
    // Each thread computes 4 dot products
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint64_t cols[4] = {g_col_base + 4*tx + 0,
                        g_col_base + 4*tx + 1,
                        g_col_base + 4*tx + 2,
                        g_col_base + 4*tx + 3};
    
    uint64_t num_k_tiles = (K + BK - 1) / BK;
    
    for (uint64_t t = 0; t < num_k_tiles; ++t) {
        uint64_t k_start = t * BK;
        
        // ===================== Load A tile into As[BM][BK] =====================
        // As layout: As[ty][lk] -> index = ty * BK + lk
        #pragma unroll
        for (int v = 0; v < 4; v++) {
            int lk = 4 * tx + v;
            if (lk < BK) {
                uint64_t a_k = k_start + lk;
                if (row_valid && a_k < K) {
                    As[ty * BK + lk] = A[g_row * K + a_k];
                } else {
                    As[ty * BK + lk] = __float2bfloat16_rn(0.0f);
                }
            }
        }
        
        // ===================== Load B tile into Bs[BK][BN+B_PAD] =====================
        // Bs layout: Bs[lk][ln] -> index = lk * (BN + B_PAD) + ln
        // Each thread loads 4 elements along N dimension for each K index (4x4 window)
        #pragma unroll
        for (int cn = 0; cn < 4; cn++) {
            uint64_t col = cols[cn];
            if (col < N) {
                #pragma unroll
                for (int vk = 0; vk < BK; vk += 4) {
                    int ln = 4 * tx + cn;  // fixed N offset for this thread+column
                    int lk = vk;           // K loop index
                    
                    uint64_t k_off = k_start + lk;
                    if (k_off < K) {
                        uint64_t base = col * K + k_off;
                        Bs[lk * (BN + B_PAD) + ln + 0] = B[base + 0];
                        Bs[lk * (BN + B_PAD) + ln + 1] = B[base + 1];
                        Bs[lk * (BN + B_PAD) + ln + 2] = B[base + 2];
                        Bs[lk * (BN + B_PAD) + ln + 3] = B[base + 3];
                    } else {
                        Bs[lk * (BN + B_PAD) + ln + 0] = __float2bfloat16_rn(0.0f);
                        Bs[lk * (BN + B_PAD) + ln + 1] = __float2bfloat16_rn(0.0f);
                        Bs[lk * (BN + B_PAD) + ln + 2] = __float2bfloat16_rn(0.0f);
                        Bs[lk * (BN + B_PAD) + ln + 3] = __float2bfloat16_rn(0.0f);
                    }
                }
            }
        }
        
        __syncthreads();
        
        // ===================== Compute WMMA-style dot product =====================
        // acc[cn] += sum_k As[ty][k] * Bs[k][4*tx+cn]
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float a_val = bf16_to_f32(As[ty * BK + k]);
            #pragma unroll
            for (int cn = 0; cn < 4; cn++) {
                float b_val = bf16_to_f32(Bs[k * (BN + B_PAD) + (4 * tx + cn)]);
                acc[cn] += a_val * b_val;
            }
        }
        
        __syncthreads();
    }
    
    // ===================== Store results =====================
    #pragma unroll
    for (int cn = 0; cn < 4; cn++) {
        uint64_t col = cols[cn];
        if (row_valid && col < N) {
            C[g_row * N + col] = __float2bfloat16_rn(acc[cn]);
        }
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
  CUDA_CHECK(cudaSetDevice(A.device().device_id));
  
  uint64_t M = A.size(0);
  uint64_t N = 7168;
  uint64_t K = 5120;
  
  dim3 block(TX, TY); // 16 x 64 = 1024 threads
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  
  // Shared memory: As[BM][BK] + Bs[BK][BN + B_PAD]
  size_t sm_bytes = (BM * BK + BK * (BN + B_PAD)) * sizeof(__nv_bfloat16);
  
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