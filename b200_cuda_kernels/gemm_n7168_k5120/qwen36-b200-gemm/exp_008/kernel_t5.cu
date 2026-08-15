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
static constexpr int BK = 16;  // Smaller K-tile for correctness first
static constexpr int TX = 16;  // threads along N: each thread handles 4 cols
static constexpr int TY = 64;  // threads along M: each thread handles 1 row

__global__ void gemm_blocked_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    uint64_t M, uint64_t N, uint64_t K)
{
    extern __shared__ __nv_bfloat16 smem[];
    
    // As[BM][BK] => As[row_in_block][k_in_tile]
    __nv_bfloat16* __restrict__ As = smem;
    // Bs[BK][BN] => Bs[k_in_tile][col_in_block]
    __nv_bfloat16* __restrict__ Bs = smem + BM * BK;
    
    int tx = threadIdx.x;  // 0..15
    int ty = threadIdx.y;  // 0..63
    
    uint64_t g_row = blockIdx.y * BM + ty;
    bool row_valid = (g_row < M);
    
    // This thread handles columns 4*tx .. 4*tx+3 within this block
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    
    uint64_t num_k_tiles = (K + BK - 1) / BK;
    
    for (uint64_t t = 0; t < num_k_tiles; t++) {
        uint64_t k_start = t * BK;
        
        // ----- Load A tile: As[ty][lk] -----
        // Each thread loads 4 consecutive K values
        #pragma unroll
        for (int v = 0; v < 4; v++) {
            int lk = 4 * tx + v;
            uint64_t k_off = k_start + lk;
            if (row_valid && k_off < K) {
                As[ty * BK + lk] = A[g_row * K + k_off];
            } else {
                As[ty * BK + lk] = __float2bfloat16_rn(0.f);
            }
        }
        
        // ----- Load B tile: Bs[lk][ln] -----
        // Each thread loads B values for its 4 columns and some K rows
        // Thread covers lk = 4*(t_BK) .. 4*(t_BK)+3 for each 4-column group
        // Since BK=16 and we have 16 threads, each thread handles exactly 1 K row
        #pragma unroll
        for (int v = 0; v < 4; v++) {
            int ln = 4 * tx + v;  // column within block (0..63)
            int lk = ty / 4;      // K index within tile? No, wrong mapping
            
            // Simpler: each thread (tx,ty) loads Bs[lk][4*tx+v] 
            // where lk ranges over [2*ty, 2*ty+1] stride 2... still complex.
            
            // Simplest correct approach: all threads cooperatively fill Bs
            int lk_val = 2 * (threadIdx.y) + 0;
            int ln_val = 4 * tx + v;
            if (lk_val < BK && ln_val < BN) {
                uint64_t blk_col = blockIdx.x * BN + ln_val;
                uint64_t k_off = k_start + lk_val;
                if (blk_col < N && k_off < K) {
                    Bs[lk_val * BN + ln_val] = B[blk_col * K + k_off];
                } else {
                    Bs[lk_val * BN + ln_val] = __float2bfloat16_rn(0.f);
                }
            }
        }
        
        __syncthreads();
        
        // ----- Compute -----
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float a_val = __bfloat162float(As[ty * BK + k]);
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                int ln = 4 * tx + c;
                if (blockIdx.x * BN + ln < N) {
                    float b_val = __bfloat162float(Bs[k * BN + ln]);
                    acc[c] += a_val * b_val;
                }
            }
        }
        
        __syncthreads();
    }
    
    // ----- Store -----
    #pragma unroll
    for (int c = 0; c < 4; c++) {
        int ln = 4 * tx + c;
        uint64_t blk_col = blockIdx.x * BN + ln;
        if (row_valid && blk_col < N) {
            C[g_row * N + blk_col] = __float2bfloat16_rn(acc[c]);
        }
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    uint64_t M = A.size(0);
    uint64_t N = B.size(0);
    uint64_t K = A.size(1);
    
    dim3 block(TX, TY);  // 16 x 64 = 1024 threads
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