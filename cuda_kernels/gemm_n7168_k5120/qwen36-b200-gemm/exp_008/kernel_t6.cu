#include <cuda_bf16.h>
#include <cuda_runtime.h>
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

// 1024 threads per block: TX=32, TY=32
static constexpr int TX = 32;
static constexpr int TY = 32;

__global__ void gemm_blocked_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    uint64_t M, uint64_t N, uint64_t K)
{
    extern __shared__ __nv_bfloat16 smem[];
    
    // As[BM][BK]: As[ty_BM][lk] => index = ty_BM * BK + lk
    __nv_bfloat16* __restrict__ As = smem;
    // Bs[BK][BN]: Bs[lk][ln] => index = lk * BN + ln  
    __nv_bfloat16* __restrict__ Bs = smem + BM * BK;
    
    int tx = threadIdx.x;  // 0..31
    int ty = threadIdx.y;  // 0..31
    
    // Linear thread id within block: 0..1023
    int tid = ty * TX + tx;
    
    uint64_t g_row = blockIdx.y * BM + (tid % BM);
    bool row_valid = (g_row < M);
    
    // Each thread computes 1 output element
    uint64_t g_col_in_block = tid / BM;  // 0..15 since 1024/BM = 16
    uint64_t g_col = blockIdx.x * BN + g_col_in_block;
    bool col_valid = (g_col < N);
    
    float acc = 0.f;
    
    uint64_t num_k_tiles = (K + BK - 1) / BK;
    
    for (uint64_t t = 0; t < num_k_tiles; t++) {
        uint64_t k_start = t * BK;
        
        // ---- Load A tile cooperatively ----
        // As[BM*BK] elements, 1024 threads => ~2 per thread
        // Thread tid handles elements around As[tid / 2 * something]
        // Simpler: each thread (ty,tid) maps to specific row & col in As
        
        // Load for As: row = tid % BM = g_row % BM, col cycles through
        // Each thread loads ceil(BM*BK / 1024) = ceil(2048/1024) = 2 elements
        for (int v = 0; v < 2; v++) {
            int flat_idx = tid * 2 + v;  // 0..2047
            int r = flat_idx / BK;       // 0..63
            int c = flat_idx % BK;       // 0..31
            
            uint64_t global_r = blockIdx.y * BM + r;
            uint64_t k_off = k_start + c;
            
            if (global_r < M && k_off < K) {
                As[r * BK + c] = A[global_r * K + k_off];
            } else {
                As[r * BK + c] = __float2bfloat16_rn(0.f);
            }
        }
        
        // ---- Load B tile cooperatively ----
        // Bs[BK*BN] elements = 32*64 = 2048 elements, 1024 threads => 2 per thread
        for (int v = 0; v < 2; v++) {
            int flat_idx = tid * 2 + v;  // 0..2047
            int r = flat_idx / BN;       // 0..31 (=BK)
            int c = flat_idx % BN;       // 0..63 (=BN)
            
            uint64_t global_c = blockIdx.x * BN + c;
            uint64_t k_off = k_start + r;
            
            if (global_c < N && k_off < K) {
                Bs[r * BN + c] = B[global_c * K + k_off];
            } else {
                Bs[r * BN + c] = __float2bfloat16_rn(0.f);
            }
        }
        
        __syncthreads();
        
        // ---- Compute dot product for this thread's output element ----
        // acc += sum_{k=0}^{BK-1} As[g_row_local][k] * Bs[k][g_col_in_block]
        int row_local = g_row - blockIdx.y * BM;
        
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            acc += __bfloat162float(As[row_local * BK + k]) *
                   __bfloat162float(Bs[k * BN + g_col_in_block]);
        }
        
        __syncthreads();
    }
    
    // ---- Store result ----
    if (row_valid && col_valid) {
        C[g_row * N + g_col] = __float2bfloat16_rn(acc);
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    uint64_t M = A.size(0);
    uint64_t N = B.size(0);
    uint64_t K = A.size(1);
    
    dim3 block(TX, TY);  // 32 x 32 = 1024 threads
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    
    size_t sm_bytes = (BM * BK + BK * BN) * sizeof(__nv_bfloat16);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    gemm_blocked_kernel<<<grid, block, sm_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(A.data_ptr()),
        static_cast<const __nv_bfloat16*>(B.data_ptr()),
        static_cast<__nv_bfloat16*>(C.data_ptr()),
        M, N, K);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_sm100::run);

}  // namespace gemm_sm100