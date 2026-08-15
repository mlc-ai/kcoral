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

static constexpr int TM = 32;
static constexpr int TN = 32;
static constexpr int TK = 8;
static constexpr int BX = 32;
static constexpr int BY = 8;
static constexpr int TOTAL_THREADS = BX * BY; // 256

__device__ __forceinline__ float bf16_to_f32(const __nv_bfloat16& val) {
    return __bfloat162float(val);
}

__device__ __forceinline__ __nv_bfloat16 f32_to_bf16(float val) {
    return __float2bfloat16_rn(val);
}

__global__ void gemm_blocked_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    uint64_t M, uint64_t N, uint64_t K)
{
    // Shared memory: As[TM][TK] + Bs[TK][TN]
    extern __shared__ __nv_bfloat16 smem[];
    
    __nv_bfloat16* __restrict__ As = smem;
    __nv_bfloat16* __restrict__ Bs = smem + TM * TK;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    uint64_t bm = blockIdx.y;
    uint64_t bn = blockIdx.x;
    
    // Global indices for this thread's output element
    uint64_t g_row = bm * TM + ty;
    uint64_t g_col = bn * TN + tx;
    
    bool row_in_bounds = (g_row < M);
    bool col_in_bounds = (g_col < N);
    
    float acc = 0.f;
    
    uint64_t num_k_tiles = (K + TK - 1) / TK;
    
    for (uint64_t tile_idx = 0; tile_idx < num_k_tiles; ++tile_idx) {
        uint64_t k_tile_start = tile_idx * TK;
        bool k_valid = (k_tile_start < K);
        
        // Load A tile into shared memory
        // As[ty][tx] corresponds to A[g_row][k_tile_start + tx]
        if (row_in_bounds && k_valid && (k_tile_start + tx) < K) {
            As[ty * TK + tx] = A[g_row * K + k_tile_start + tx];
        } else {
            As[ty * TK + tx] = f32_to_bf16(0.f);
        }
        
        // Load B tile into shared memory (transposed view)
        // Bs[tx][ty] corresponds to B[g_col][k_tile_start + ty]
        if (col_in_bounds && k_valid && (k_tile_start + ty) < K) {
            Bs[tx * TN + ty] = B[g_col * K + k_tile_start + ty];
        } else {
            Bs[tx * TN + ty] = f32_to_bf16(0.f);
        }
        
        __syncthreads();
        
        // Compute partial dot product for this K-tile
        #pragma unroll
        for (int k = 0; k < TK; ++k) {
            acc += bf16_to_f32(As[ty * TK + k]) * bf16_to_f32(Bs[k * TN + tx]);
        }
        
        __syncthreads();
    }
    
    // Store result
    if (row_in_bounds && col_in_bounds) {
        C[g_row * N + g_col] = f32_to_bf16(acc);
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    uint64_t M = A.size(0);
    uint64_t N = B.size(0);
    uint64_t K = A.size(1);
    
    dim3 block(BX, BY);  // 32 x 8 = 256 threads
    dim3 grid((N + TN - 1) / TN, (M + TM - 1) / TM);
    
    // Shared memory: As[TM][TK] + Bs[TK][TN]
    size_t sm_bytes = (TM * TK + TK * TN) * sizeof(__nv_bfloat16);
    
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