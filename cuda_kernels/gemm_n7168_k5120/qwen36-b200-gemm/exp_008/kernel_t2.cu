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

static constexpr int BM = 64;
static constexpr int BN = 64;
static constexpr int BK = 32;
static constexpr int B_PAD = 4;
static constexpr int TX = 16;
static constexpr int TY = 64;

__global__ void gemm_blocked_bf16_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    uint64_t M, uint64_t N, uint64_t K)
{
    extern __shared__ __nv_bfloat16 smem[];
    
    __nv_bfloat16* __restrict__ As = smem;           // As[BM][BK]
    __nv_bfloat16* __restrict__ Bs = smem + BM * BK; // Bs[BK][BN + B_PAD]
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    uint64_t g_row = blockIdx.y * BM + ty;
    uint64_t g_col_base = blockIdx.x * BN;
    
    bool row_valid = (g_row < M);
    
    // Each thread computes for 2 output columns
    int2 local_n_offset = make_int2(2 * tx, 2 * tx + 1);
    uint64_t cols[2] = {g_col_base + local_n_offset.x, g_col_base + local_n_offset.y};
    bool col_valid[2] = {(cols[0] < N), (cols[1] < N)};
    
    float acc[2] = {0.0f, 0.0f};
    
    uint64_t num_k_tiles = (K + BK - 1) / BK;
    
    for (uint64_t t = 0; t < num_k_tiles; ++t) {
        uint64_t k_start = t * BK;
        
        // ===================== Load A tile =====================
        // As layout: As[ty][kl] = As[ty*BK + kl]
        // Each thread loads 2 consecutive K values
        #pragma unroll
        for (int idx = 0; idx < 2; idx++) {
            int kl = 2 * idx;
            uint64_t k_off = k_start + kl;
            if (row_valid && k_off < K) {
                As[(ty * BK + kl)       ] = A[g_row * K + k_off];
                As[(ty * BK + kl + 1)   ] = A[g_row * K + k_off + 1];
            } else {
                As[(ty * BK + kl)       ] = __float2bfloat16_rn(0.0f);
                As[(ty * BK + kl + 1)   ] = __float2bfloat16_rn(0.0f);
            }
        }
        
        // ===================== Load B tile =====================
        // Bs layout: Bs[kl][ln] = Bs[kl * (BN + B_PAD) + ln]
        // Each thread loads 2 consecutive N values for each K row
        #pragma unroll
        for (int kl = 0; kl < BK; kl += 2) {
            uint64_t k_off = k_start + kl;
            
            // Load for local_n_offset.x
            if (col_valid[0] && k_off < K) {
                Bs[kl * (BN + B_PAD) + local_n_offset.x]       = B[cols[0] * K + k_off];
                Bs[(kl + 1) * (BN + B_PAD) + local_n_offset.x] = B[cols[0] * K + k_off + 1];
            } else {
                Bs[kl * (BN + B_PAD) + local_n_offset.x]       = __float2bfloat16_rn(0.0f);
                Bs[(kl + 1) * (BN + B_PAD) + local_n_offset.x] = __float2bfloat16_rn(0.0f);
            }
            
            // Load for local_n_offset.y
            if (col_valid[1] && k_off < K) {
                Bs[kl * (BN + B_PAD) + local_n_offset.y]       = B[cols[1] * K + k_off];
                Bs[(kl + 1) * (BN + B_PAD) + local_n_offset.y] = B[cols[1] * K + k_off + 1];
            } else {
                Bs[kl * (BN + B_PAD) + local_n_offset.y]       = __float2bfloat16_rn(0.0f);
                Bs[(kl + 1) * (BN + B_PAD) + local_n_offset.y] = __float2bfloat16_rn(0.0f);
            }
        }
        
        __syncthreads();
        
        // ===================== Compute =====================
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float a_val = __bfloat162float(As[ty * BK + k]);
            
            if (col_valid[0]) {
                acc[0] += a_val * __bfloat162float(Bs[k * (BN + B_PAD) + local_n_offset.x]);
            }
            if (col_valid[1]) {
                acc[1] += a_val * __bfloat162float(Bs[k * (BN + B_PAD) + local_n_offset.y]);
            }
        }
        
        __syncthreads();
    }
    
    // ===================== Store =====================
    #pragma unroll
    for (int c = 0; c < 2; c++) {
        if (row_valid && col_valid[c]) {
            C[g_row * N + cols[c]] = __float2bfloat16_rn(acc[c]);
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