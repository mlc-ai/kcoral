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

namespace gemm_bf16 {

constexpr int BM = 128;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int THREADS = 256;

__global__ __launch_bounds__(THREADS, 2)
void gemm_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K) {

    __shared__ __nv_bfloat16 smemA[2][BM * BK];
    __shared__ __nv_bfloat16 smemB[2][BN * BK];

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    const int m_block = blockIdx.x * BM;
    const int n_block = blockIdx.y * BN;

    // 8 warps, each handles 16 rows x 64 cols = 1 m-tile x 8 n-tiles
    // 8 mma per K step, 32 FP32 acc per thread
    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            acc[i][j] = 0.0f;

    const int k_tiles = K / BK;

    // Load first tile
    {
        // A: 128*16 = 2048 elements, 256 threads, 8 each (uint4)
        int a_row = threadIdx.x / 2;
        int a_col = (threadIdx.x % 2) * 8;
        int gm = m_block + a_row;
        uint4 val;
        if (gm < M) {
            val = *reinterpret_cast<const uint4*>(&A[gm * K + a_col]);
        } else {
            val = make_uint4(0, 0, 0, 0);
        }
        *reinterpret_cast<uint4*>(&smemA[0][a_row * BK + a_col]) = val;

        // B: 64*16 = 1024 elements, first 128 threads, 8 each (uint4)
        if (threadIdx.x < 128) {
            int b_row = threadIdx.x / 2;
            int b_col = (threadIdx.x % 2) * 8;
            int gn = n_block + b_row;
            val = *reinterpret_cast<const uint4*>(&B[gn * K + b_col]);
            *reinterpret_cast<uint4*>(&smemB[0][b_row * BK + b_col]) = val;
        }
    }
    __syncthreads();

    #pragma unroll 1
    for (int kt = 0; kt < k_tiles; kt++) {
        int buf = kt % 2;

        // Issue next load (double buffering)
        if (kt + 1 < k_tiles) {
            int next_buf = 1 - buf;
            int a_row = threadIdx.x / 2;
            int a_col = (threadIdx.x % 2) * 8;
            int gm = m_block + a_row;
            uint4 val;
            if (gm < M) {
                val = *reinterpret_cast<const uint4*>(&A[gm * K + (kt + 1) * BK + a_col]);
            } else {
                val = make_uint4(0, 0, 0, 0);
            }
            *reinterpret_cast<uint4*>(&smemA[next_buf][a_row * BK + a_col]) = val;

            if (threadIdx.x < 128) {
                int b_row = threadIdx.x / 2;
                int b_col = (threadIdx.x % 2) * 8;
                int gn = n_block + b_row;
                val = *reinterpret_cast<const uint4*>(&B[gn * K + (kt + 1) * BK + b_col]);
                *reinterpret_cast<uint4*>(&smemB[next_buf][b_row * BK + b_col]) = val;
            }
        }

        // Load A fragment (reused across all 8 n-tiles)
        int m_off = warp_id * 16;
        uint32_t a[4];
        a[0] = *reinterpret_cast<uint32_t*>(&smemA[buf][(m_off + lane_id/4) * BK + 2*(lane_id%4)]);
        a[1] = *reinterpret_cast<uint32_t*>(&smemA[buf][(m_off + lane_id/4 + 8) * BK + 2*(lane_id%4)]);
        a[2] = *reinterpret_cast<uint32_t*>(&smemA[buf][(m_off + lane_id/4) * BK + 2*(lane_id%4) + 8]);
        a[3] = *reinterpret_cast<uint32_t*>(&smemA[buf][(m_off + lane_id/4 + 8) * BK + 2*(lane_id%4) + 8]);

        // Compute 8 mma operations
        #pragma unroll
        for (int ni = 0; ni < 8; ni++) {
            int n_off = ni * 8;
            uint32_t b[2];
            b[0] = *reinterpret_cast<uint32_t*>(&smemB[buf][(n_off + lane_id/4) * BK + 2*(lane_id%4)]);
            b[1] = *reinterpret_cast<uint32_t*>(&smemB[buf][(n_off + lane_id/4) * BK + 2*(lane_id%4) + 8]);

            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(acc[ni][0]), "+f"(acc[ni][1]),
                  "+f"(acc[ni][2]), "+f"(acc[ni][3])
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                  "r"(b[0]), "r"(b[1])
            );
        }
        __syncthreads();
    }

    // Store results
    int m_off = warp_id * 16;
    #pragma unroll
    for (int ni = 0; ni < 8; ni++) {
        int n_off = ni * 8;
        int row0 = m_block + m_off + lane_id / 4;
        int row1 = m_block + m_off + lane_id / 4 + 8;
        int col0 = n_block + n_off + lane_id % 4;
        int col1 = n_block + n_off + lane_id % 4 + 4;

        if (row0 < M) {
            if (col0 < N) C[row0 * N + col0] = __float2bfloat16(acc[ni][0]);
            if (col1 < N) C[row0 * N + col1] = __float2bfloat16(acc[ni][2]);
        }
        if (row1 < M) {
            if (col0 < N) C[row1 * N + col0] = __float2bfloat16(acc[ni][1]);
            if (col1 < N) C[row1 * N + col1] = __float2bfloat16(acc[ni][3]);
        }
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(0);

    const __nv_bfloat16* A_ptr = static_cast<const __nv_bfloat16*>(A.data_ptr());
    const __nv_bfloat16* B_ptr = static_cast<const __nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_ptr = static_cast<__nv_bfloat16*>(C.data_ptr());

    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN, 1);
    dim3 block(THREADS, 1, 1);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    gemm_kernel<<<grid, block, 0, stream>>>(A_ptr, B_ptr, C_ptr, (int)M, (int)N, (int)K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_bf16::run);

}  // namespace gemm_bf16