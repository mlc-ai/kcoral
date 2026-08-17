#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace gemm_n7168_k5120 {

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 64;

__global__ __launch_bounds__(256, 2)
void gemm_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K) {

    __shared__ __nv_bfloat16 smemA[BM][BK];
    __shared__ __nv_bfloat16 smemB[BN][BK];

    const int tid = threadIdx.x;
    const int tx = tid & 15;   // 0..15
    const int ty = tid >> 4;   // 0..15

    const int m_block = blockIdx.x * BM;
    const int n_block = blockIdx.y * BN;

    float acc[8][8];
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            acc[i][j] = 0.0f;

    const int k_tiles = K / BK;
    #pragma unroll 1
    for (int kt = 0; kt < k_tiles; ++kt) {
        const int k_base = kt * BK;

        // Load A tile [BM, BK] using vectorized 16-byte stores
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int lin = tid * 4 + i;
            const int row = lin >> 3;          // lin / 8
            const int col = (lin & 7) << 3;    // (lin % 8) * 8
            const int gm = m_block + row;
            if (gm < M) {
                uint4 val = *reinterpret_cast<const uint4*>(&A[gm * K + k_base + col]);
                *reinterpret_cast<uint4*>(&smemA[row][col]) = val;
            } else {
                *reinterpret_cast<uint4*>(&smemA[row][col]) = make_uint4(0, 0, 0, 0);
            }
        }

        // Load B tile [BN, BK] using vectorized 16-byte stores
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int lin = tid * 4 + i;
            const int row = lin >> 3;
            const int col = (lin & 7) << 3;
            const int gn = n_block + row;
            uint4 val = *reinterpret_cast<const uint4*>(&B[gn * K + k_base + col]);
            *reinterpret_cast<uint4*>(&smemB[row][col]) = val;
        }
        __syncthreads();

        // Compute: process 2 K-elements per iteration
        #pragma unroll
        for (int k = 0; k < BK; k += 2) {
            __nv_bfloat162 a2[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                a2[i] = *reinterpret_cast<const __nv_bfloat162*>(&smemA[ty * 8 + i][k]);
            }
            __nv_bfloat162 b2[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                b2[j] = *reinterpret_cast<const __nv_bfloat162*>(&smemB[tx * 8 + j][k]);
            }
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                float2 af = __bfloat1622float2(a2[i]);
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    float2 bf = __bfloat1622float2(b2[j]);
                    acc[i][j] += af.x * bf.x;
                    acc[i][j] += af.y * bf.y;
                }
            }
        }
        __syncthreads();
    }

    // Store results
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int gm = m_block + ty * 8 + i;
        if (gm < M) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int gn = n_block + tx * 8 + j;
                if (gn < N) {
                    C[gm * N + gn] = __float2bfloat16(acc[i][j]);
                }
            }
        }
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    const int64_t N = B.size(0);  // 7168

    const __nv_bfloat16* A_ptr = static_cast<const __nv_bfloat16*>(A.data_ptr());
    const __nv_bfloat16* B_ptr = static_cast<const __nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());

    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN, 1);
    dim3 block(256, 1, 1);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    gemm_kernel<<<grid, block, 0, stream>>>(A_ptr, B_ptr, C_ptr, (int)M, (int)N, (int)K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_n7168_k5120::run);

}  // namespace gemm_n7168_k5120