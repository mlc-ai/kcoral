#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace gemm_impl {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 32;
constexpr int BK_PAD = BK + 1;  // Pad B rows to avoid shared memory bank conflicts
constexpr int THREADS = 128;

__global__ __launch_bounds__(THREADS)
void gemm_kernel(const __nv_bfloat16* __restrict__ A,
                 const __nv_bfloat16* __restrict__ B,
                 __nv_bfloat16* __restrict__ C,
                 int M, int N, int K) {
    int bm = blockIdx.y;
    int bn = blockIdx.x;

    __shared__ __nv_bfloat16 A_smem[2][BM * BK];
    __shared__ __nv_bfloat16 B_smem[2][BN * BK_PAD];

    int tx = threadIdx.x;
    int thread_row = tx / 16;
    int thread_col = tx % 16;
    int row_base = thread_row * 8;
    int col_base = thread_col * 4;

    float acc[8][4];
    #pragma unroll
    for (int r = 0; r < 8; r++)
        #pragma unroll
        for (int c = 0; c < 4; c++)
            acc[r][c] = 0.0f;

    int num_bk = K / BK;

    auto load_tile = [&](int buf, int bk_idx) {
        {
            int row = tx / 2;
            int col_start = (tx % 2) * 16;
            int gm = bm * BM + row;
            int gk = bk_idx * BK + col_start;
            __nv_bfloat16* sptr = &A_smem[buf][row * BK + col_start];
            if (gm < M) {
                const __nv_bfloat16* gptr = &A[(size_t)gm * K + gk];
                *reinterpret_cast<uint4*>(sptr)     = *reinterpret_cast<const uint4*>(gptr);
                *reinterpret_cast<uint4*>(sptr + 8) = *reinterpret_cast<const uint4*>(gptr + 8);
            } else {
                *reinterpret_cast<uint4*>(sptr)     = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(sptr + 8) = make_uint4(0, 0, 0, 0);
            }
        }

        {
            int row = tx / 2;
            int col_start = (tx % 2) * 16;
            int gn = bn * BN + row;
            int gk = bk_idx * BK + col_start;
            __nv_bfloat16* sptr = &B_smem[buf][row * BK_PAD + col_start];
            if (gn < N) {
                const __nv_bfloat16* gptr = &B[(size_t)gn * K + gk];
                uint4 v0 = *reinterpret_cast<const uint4*>(gptr);
                uint4 v1 = *reinterpret_cast<const uint4*>(gptr + 8);
                __nv_bfloat16* vp0 = reinterpret_cast<__nv_bfloat16*>(&v0);
                __nv_bfloat16* vp1 = reinterpret_cast<__nv_bfloat16*>(&v1);
                #pragma unroll
                for (int i = 0; i < 8; i++) sptr[i] = vp0[i];
                #pragma unroll
                for (int i = 0; i < 8; i++) sptr[8 + i] = vp1[i];
            } else {
                #pragma unroll
                for (int i = 0; i < 16; i++) sptr[i] = __float2bfloat16(0.0f);
            }
        }
    };

    load_tile(0, 0);
    __syncthreads();

    for (int bk = 0; bk < num_bk; bk++) {
        int buf = bk % 2;

        if (bk + 1 < num_bk) {
            load_tile(1 - buf, bk + 1);
        }

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float a_vals[8];
            float b_vals[4];
            #pragma unroll
            for (int r = 0; r < 8; r++)
                a_vals[r] = __bfloat162float(A_smem[buf][(row_base + r) * BK + k]);
            #pragma unroll
            for (int c = 0; c < 4; c++)
                b_vals[c] = __bfloat162float(B_smem[buf][(col_base + c) * BK_PAD + k]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                #pragma unroll
                for (int c = 0; c < 4; c++)
                    acc[r][c] += a_vals[r] * b_vals[c];
        }

        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < 8; r++) {
        int gm = bm * BM + row_base + r;
        if (gm >= M) continue;
        int gn = bn * BN + col_base;
        __nv_bfloat16 vals[4];
        #pragma unroll
        for (int c = 0; c < 4; c++)
            vals[c] = __float2bfloat16(acc[r][c]);
        *reinterpret_cast<uint2*>(&C[(size_t)gm * N + gn]) =
            *reinterpret_cast<uint2*>(vals);
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    int64_t M = A.size(0);
    int64_t N = B.size(0);
    int64_t K = A.size(1);

    const __nv_bfloat16* A_data = static_cast<const __nv_bfloat16*>(A.data_ptr());
    const __nv_bfloat16* B_data = static_cast<const __nv_bfloat16*>(B.data_ptr());
    __nv_bfloat16* C_data = static_cast<__nv_bfloat16*>(C.data_ptr());

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(THREADS);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    gemm_kernel<<<grid, block, 0, stream>>>(A_data, B_data, C_data,
                                             (int)M, (int)N, (int)K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_impl::run);

}  // namespace gemm_impl