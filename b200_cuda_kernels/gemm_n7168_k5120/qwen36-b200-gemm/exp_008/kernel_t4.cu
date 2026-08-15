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

static constexpr int BM = 64;  // M dimension per block
static constexpr int BN = 64;  // N dimension per block
static constexpr int BK = 32;  // K tile size

// Block dims: 16 threads in N, 64 threads in M => 1024 threads
static constexpr int TX = BN / 4;  // 16
static constexpr int TY = BM;      // 64

__global__ void gemm_blocked_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    uint64_t M, uint64_t N, uint64_t K)
{
    extern __shared__ __nv_bfloat16 smem[];
    
    // As[BM][BK] layout: As[ty][kl] = As[ty * BK + kl]
    __nv_bfloat16* __restrict__ As = smem;
    // Bs[BK][BN] layout: Bs[kl][ln] = Bs[kl * BN + ln]
    __nv_bfloat16* __restrict__ Bs = smem + BM * BK;
    
    int tx = threadIdx.x;  // 0..15
    int ty = threadIdx.y;  // 0..63
    
    uint64_t bm = blockIdx.y;
    uint64_t bn = blockIdx.x;
    
    // This thread owns C[ty][4*tx .. 4*tx+3] within its block
    uint64_t g_row = bm * BM + ty;
    bool row_in_bounds = (g_row < M);
    
    uint64_t acc4[4] = {0, 0, 0, 0};  // accumulate in u32 to hold fp32 bit patterns
    float acc_f[4] = {0.f, 0.f, 0.f, 0.f};
    
    uint64_t base_col0 = bn * BN + 4 * tx;
    bool col_valid[4];
    #pragma unroll
    for (int c = 0; c < 4; c++) {
        col_valid[c] = (base_col0 + c < N);
    }
    
    uint64_t num_k_tiles = (K + BK - 1) / BK;
    
    for (uint64_t tile_idx = 0; tile_idx < num_k_tiles; ++tile_idx) {
        uint64_t k_tile_start = tile_idx * BK;
        
        // ===================== Load A tile into As[BM][BK] =====================
        // Each thread (ty, tx) loads 4 consecutive K values for its row
        // Loads: As[ty][4*tx], As[ty][4*tx+1], As[ty][4*tx+2], As[ty][4*tx+3]
        #pragma unroll
        for (int v = 0; v < 4; v++) {
            int kl = 4 * tx + v;
            uint64_t k_off = k_tile_start + kl;
            if (row_in_bounds && k_off < K) {
                As[ty * BK + kl] = A[g_row * K + k_off];
            } else {
                As[ty * BK + kl] = __float2bfloat16_rn(0.f);
            }
        }
        
        // ===================== Load B tile into Bs[BK][BN] =====================
        // Bs[kl][ln] = Bs[kl * BN + ln]
        // For each of the 4 columns this thread owns, load 2 K values each iteration
        // Each thread loads: Bs[2*i][4*tx+c] and Bs[2*i+1][4*tx+c] for i=0..7, c=0..3
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            uint64_t g_col = base_col0 + c;
            if (!col_valid[c]) continue;
            
            uint64_t b_row_off = g_col * K;
            for (int i = 0; i < BK / 2; i += 2) {
                uint64_t k_off = k_tile_start + 2 * i;
                if (k_off + 1 < K) {
                    Bs[(2*i)   * BN + (4*tx + c)] = B[b_row_off + k_off     ];
                    Bs[(2*i+1) * BN + (4*tx + c)] = B[b_row_off + k_off + 1 ];
                    Bs[(2*i+2) * BN + (4*tx + c)] = B[b_row_off + k_off + 2 ];
                    Bs[(2*i+3) * BN + (4*tx + c)] = B[b_row_off + k_off + 3 ];
                } else {
                    Bs[(2*i)   * BN + (4*tx + c)] = __float2bfloat16_rn(0.f);
                    Bs[(2*i+1) * BN + (4*tx + c)] = __float2bfloat16_rn(0.f);
                    Bs[(2*i+2) * BN + (4*tx + c)] = __float2bfloat16_rn(0.f);
                    Bs[(2*i+3) * BN + (4*tx + c)] = __float2bfloat16_rn(0.f);
                }
            }
        }
        
        __syncthreads();
        
        // ===================== Compute =====================
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float a_val = __bfloat162float(As[ty * BK + k]);
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                if (col_valid[c]) {
                    float b_val = __bfloat162float(Bs[k * BN + (4 * tx + c)]);
                    acc_f[c] += a_val * b_val;
                }
            }
        }
        
        __syncthreads();
    }
    
    // ===================== Store =====================
    #pragma unroll
    for (int c = 0; c < 4; c++) {
        uint64_t g_col = base_col0 + c;
        if (row_in_bounds && col_valid[c]) {
            C[g_row * N + g_col] = __float2bfloat16_rn(acc_f[c]);
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
    
    // Shared memory: As[BM][BK] + Bs[BK][BN]
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