#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;
using bf16 = __nv_bfloat16;

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace gemm_cuda {

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 16;
constexpr int NTHREADS = 256;

constexpr int A_TILE_BYTES = BM * BK * sizeof(bf16);
constexpr int B_TILE_BYTES = BN * BK * sizeof(bf16);
constexpr int SMEM_BYTES = 2 * (A_TILE_BYTES + B_TILE_BYTES);

__global__ __launch_bounds__(NTHREADS)
void gemm_kernel(const bf16* __restrict__ A, const bf16* __restrict__ B,
                 bf16* __restrict__ C, int M, int N, int K) {
    extern __shared__ char smem_raw[];
    bf16* smemA = reinterpret_cast<bf16*>(smem_raw);
    bf16* smemB = smemA + 2 * BM * BK;
    
    // Statically allocate float buffer for epilogue
    __shared__ float smemC[BM * BN];

    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int warp_m  = warp_id / 4;
    const int warp_n  = warp_id % 4;

    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;
    const int K_iters = K / BK;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][2];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);

    const int4 zero4 = make_int4(0, 0, 0, 0);

    // Load first tile
    {
        int row = tid / 2;
        int col = (tid % 2) * 8;
        int m = bm + row;
        int n = bn + row;
        *(int4*)&smemA[row * BK + col] = (m < M) ? *(const int4*)(A + m * K + col) : zero4;
        *(int4*)&smemB[row * BK + col] = (n < N) ? *(const int4*)(B + n * K + col) : zero4;
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major>  a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major>  b_frag[2];

    for (int ki = 0; ki < K_iters; ki++) {
        const int buf      = ki % 2;
        const int next_buf = 1 - buf;

        if (ki + 1 < K_iters) {
            int row = tid / 2;
            int col = (tid % 2) * 8;
            int m = bm + row;
            int n = bn + row;
            int k_off = (ki + 1) * BK;
            *(int4*)&smemA[next_buf * BM * BK + row * BK + col] =
                (m < M) ? *(const int4*)(A + m * K + k_off + col) : zero4;
            *(int4*)&smemB[next_buf * BN * BK + row * BK + col] =
                (n < N) ? *(const int4*)(B + n * K + k_off + col) : zero4;
        }

        #pragma unroll
        for (int ni = 0; ni < 2; ni++) {
            wmma::load_matrix_sync(b_frag[ni],
                &smemB[buf * BN * BK + (warp_n * 32 + ni * 16) * BK], BK);
        }

        #pragma unroll
        for (int mi = 0; mi < 4; mi++) {
            wmma::load_matrix_sync(a_frag,
                &smemA[buf * BM * BK + (warp_m * 64 + mi * 16) * BK], BK);
            #pragma unroll
            for (int ni = 0; ni < 2; ni++) {
                wmma::mma_sync(acc[mi][ni], a_frag, b_frag[ni], acc[mi][ni]);
            }
        }

        __syncthreads();
    }

    // Epilogue: store accumulators to float smem, convert to bf16, write to global
    #pragma unroll
    for (int mi = 0; mi < 4; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 2; ni++) {
            int row = warp_m * 64 + mi * 16;
            int col = warp_n * 32 + ni * 16;
            wmma::store_matrix_sync(&smemC[row * BN + col], acc[mi][ni], BN, wmma::mem_row_major);
        }
    }
    __syncthreads();

    // Convert float to bf16 and write to global with vectorized 8-byte stores
    for (int i = tid; i < BM * BN / 4; i += NTHREADS) {
        int idx = i * 4;
        int r = idx / BN;
        int c = idx % BN;
        int m = bm + r;
        int n = bn + c;
        if (m < M) {
            bf16 b0 = __float2bfloat16(smemC[idx]);
            bf16 b1 = __float2bfloat16(smemC[idx + 1]);
            bf16 b2 = __float2bfloat16(smemC[idx + 2]);
            bf16 b3 = __float2bfloat16(smemC[idx + 3]);
            uint32_t p0 = (uint32_t(*(uint16_t*)&b1) << 16) | uint32_t(*(uint16_t*)&b0);
            uint32_t p1 = (uint32_t(*(uint16_t*)&b3) << 16) | uint32_t(*(uint16_t*)&b2);
            *(uint2*)(C + (uint64_t)m * N + n) = make_uint2(p0, p1);
        }
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);

    const bf16* A_ptr = static_cast<const bf16*>(A.data_ptr());
    const bf16* B_ptr = static_cast<const bf16*>(B.data_ptr());
    bf16*       C_ptr = static_cast<bf16*>(C.data_ptr());

    dim3 block(NTHREADS);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    cudaFuncSetAttribute(gemm_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);

    gemm_kernel<<<grid, block, SMEM_BYTES, stream>>>(
        A_ptr, B_ptr, C_ptr, (int)M, (int)N, (int)K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_cuda::run);

}  // namespace gemm_cuda