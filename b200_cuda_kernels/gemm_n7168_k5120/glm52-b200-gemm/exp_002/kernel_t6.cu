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
constexpr int BN = 128;
constexpr int BK = 16;
constexpr int THREADS = 256;
constexpr int NUM_N_TILES = BN / 8;

__device__ __forceinline__ void cp_async_16(uint32_t smem_addr, const void* gmem_ptr) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__global__ __launch_bounds__(THREADS, 1)
void gemm_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K) {

    __shared__ __align__(16) __nv_bfloat16 smemA[2][BM * BK];
    __shared__ __align__(16) __nv_bfloat16 smemB[2][BN * BK];

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int m_block = blockIdx.x * BM;
    const int n_block = blockIdx.y * BN;

    float acc[NUM_N_TILES][4];
    #pragma unroll
    for (int i = 0; i < NUM_N_TILES; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            acc[i][j] = 0.0f;

    const int k_tiles = K / BK;

    // Prologue: issue first load via cp.async
    {
        int row = threadIdx.x / 2;
        int col = (threadIdx.x % 2) * 8;
        int gm = m_block + row;
        uint32_t a_smem = __cvta_generic_to_shared(&smemA[0][row * BK + col]);
        if (gm < M) {
            cp_async_16(a_smem, &A[gm * K + col]);
        } else {
            *reinterpret_cast<uint4*>(&smemA[0][row * BK + col]) = make_uint4(0, 0, 0, 0);
        }

        int gn = n_block + row;
        uint32_t b_smem = __cvta_generic_to_shared(&smemB[0][row * BK + col]);
        cp_async_16(b_smem, &B[gn * K + col]);
    }
    cp_async_commit();

    #pragma unroll 1
    for (int kt = 0; kt < k_tiles; kt++) {
        int buf = kt % 2;

        // Issue next load
        if (kt + 1 < k_tiles) {
            int next_buf = (kt + 1) % 2;
            int row = threadIdx.x / 2;
            int col = (threadIdx.x % 2) * 8;
            int gm = m_block + row;
            uint32_t a_smem = __cvta_generic_to_shared(&smemA[next_buf][row * BK + col]);
            if (gm < M) {
                cp_async_16(a_smem, &A[gm * K + (kt + 1) * BK + col]);
            } else {
                *reinterpret_cast<uint4*>(&smemA[next_buf][row * BK + col]) = make_uint4(0, 0, 0, 0);
            }

            int gn = n_block + row;
            uint32_t b_smem = __cvta_generic_to_shared(&smemB[next_buf][row * BK + col]);
            cp_async_16(b_smem, &B[gn * K + (kt + 1) * BK + col]);
            cp_async_commit();
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }
        __syncthreads();

        // Compute: load A fragment (reused across all n-tiles)
        int m_off = warp_id * 16;

        uint32_t a[4];
        a[0] = *reinterpret_cast<uint32_t*>(&smemA[buf][(m_off + lane_id/4) * BK + 2*(lane_id%4)]);
        a[1] = *reinterpret_cast<uint32_t*>(&smemA[buf][(m_off + lane_id/4 + 8) * BK + 2*(lane_id%4)]);
        a[2] = *reinterpret_cast<uint32_t*>(&smemA[buf][(m_off + lane_id/4) * BK + 2*(lane_id%4) + 8]);
        a[3] = *reinterpret_cast<uint32_t*>(&smemA[buf][(m_off + lane_id/4 + 8) * BK + 2*(lane_id%4) + 8]);

        #pragma unroll
        for (int ni = 0; ni < NUM_N_TILES; ni++) {
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

    // Epilogue: store results
    int m_off = warp_id * 16;
    #pragma unroll
    for (int ni = 0; ni < NUM_N_TILES; ni++) {
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