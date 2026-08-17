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
static constexpr int BK = 16;
static constexpr int TX = 8;   // BN/8 = 8 threads along N
static constexpr int TY = BM;  // 64 threads along M => 512 threads/block

__global__ void gemm_blocked_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    uint64_t M, uint64_t N, uint64_t K)
{
    extern __shared__ __nv_bfloat16 smem[];
    
    // As[BM][BK]: index = r * BK + c, where r=0..BM-1, c=0..BK-1
    __nv_bfloat16* __restrict__ As = smem;
    // Bs[BK][BN]: index = r * BN + c, where r=0..BK-1, c=0..BN-1
    __nv_bfloat16* __restrict__ Bs = smem + BM * BK;
    
    int tx = threadIdx.x;  // 0..7
    int ty = threadIdx.y;  // 0..63
    
    uint64_t bm = blockIdx.y;
    uint64_t bn = blockIdx.x;
    
    uint64_t g_row = bm * BM + ty;  // global row for this thread
    bool row_valid = (g_row < M);
    
    // Thread owns 8 consecutive output columns within this block
    float acc[8] = {0.f};
    uint64_t col_start = bn * BN + 8 * tx;
    
    uint64_t num_k_tiles = (K + BK - 1) / BK;
    
    for (uint64_t t = 0; t < num_k_tiles; t++) {
        uint64_t k_start = t * BK;
        
        // ========== Load A tile ==========
        // As[r][c] where r = ty (this thread's row), c = thread shares K-range
        // 512 threads load BM*BK = 64*16 = 1024 elements => 2 per thread
        #pragma unroll
        for (int v = 0; v < 2; v++) {
            int c = 2 * tx + v;  // K-column within tile
            if (row_valid && (k_start + c) < K) {
                As[ty * BK + c] = A[g_row * K + k_start + c];
            } else {
                As[ty * BK + c] = __float2bfloat16_rn(0.f);
            }
        }
        
        // ========== Load B tile ==========
        // Bs[kr][col_in_block] = B[bn*BN + col_in_block][k_start + kr]
        // Bs has BK*BN = 16*64 = 1024 elements, 512 threads => 2 per thread
        // Thread (tx, ty): ty provides the K-row index for B (kr range)
        // Each thread loads 2 K-rows: kr = 2*tx and kr = 2*tx+1 won't work since we need 
        // all BN columns for each K row
        
        // Different approach: each thread loads one B value
        // Map thread to (kr, ln) in Bs space
        int flat_tid = ty * TX + tx;  // 0..511
        int kr = flat_tid / BN;       // 0..7 (need 0..15!)
        int ln = flat_tid % BN;       // 0..63
        
        // Need 2 iterations per thread to cover all 16 K-rows
        for (int iter = 0; iter < 2; iter++) {
            int kr_actual = kr + iter * (TX);  // 0,8,... or 1,9,...
            if (kr_actual < BK) {
                uint64_t global_col = bn * BN + ln;
                uint64_t k_off = k_start + kr_actual;
                if (global_col < N && k_off < K) {
                    Bs[kr_actual * BN + ln] = B[global_col * K + k_off];
                } else {
                    Bs[kr_actual * BN + ln] = __float2bfloat16_rn(0.f);
                }
            }
        }
        
        __syncthreads();
        
        // ========== Compute ==========
        #pragma unroll
        for (int kr = 0; kr < BK; kr++) {
            float a_val = __bfloat162float(As[ty * BK + kr]);
            #pragma unroll
            for (int lc = 0; lc < 8; lc++) {
                uint64_t global_col = col_start + lc;
                if (global_col < N) {
                    float b_val = __bfloat162float(Bs[kr * BN + (8 * tx + lc)]);
                    acc[lc] += a_val * b_val;
                }
            }
        }
        
        __syncthreads();
    }
    
    // ========== Store ==========
    #pragma unroll
    for (int lc = 0; lc < 8; lc++) {
        uint64_t global_col = col_start + lc;
        if (row_valid && global_col < N) {
            C[g_row * N + global_col] = __float2bfloat16_rn(acc[lc]);
        }
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    uint64_t M = A.size(0);
    uint64_t N = B.size(0);
    uint64_t K = A.size(1);
    
    dim3 block(TX, TY);  // 8 x 64 = 512 threads
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