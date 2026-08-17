#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                         \
    cudaError_t err = (call);                                         \
    if (err != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                 \
                __FILE__, __LINE__, cudaGetErrorString(err));         \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)

/* ================================================================
 * Tunable constants
 * ================================================================
 * BM / BN – output tile size per block
 * BK      – K-tile width  (must divide actual K evenly for clean
 *           looping, otherwise padding logic applies)
 * THR     – threads per block  (= BM * BN, so each thread owns one
 *           output element in the tile)                          */
static constexpr int BM  = 16;
static constexpr int BN  = 16;
static constexpr int BK  = 16;
static constexpr int THR = BM * BN;          // 256

/* ================================================================
 * Kernel
 * ================================================================ */
__global__ void gemm_bw_kernel(
    const __nv_bfloat16 *__restrict__ A,
    const __nv_bfloat16 *__restrict__ B,
    __nv_bfloat16       *__restrict__ C,
    uint32_t M, uint32_t N, uint32_t K)
{
    /* ---------- shared memory ---------- */
    __shared__ __nv_bfloat16 As[BM][BK];
    __shared__ __nv_bfloat16 Bs[BN][BK];

    uint32_t tx    = threadIdx.x;
    uint32_t my_r  = tx % BM;               // row index within block tile
    uint32_t my_c  = tx / BM;               // col index within block tile

    uint32_t rs    = blockIdx.y * BM;       // global row start
    uint32_t ns    = blockIdx.x * BN;       // global col  start

    uint32_t nkt   = (K + BK - 1) / BK;     // number of K-tiles

    /* ---- accumulate one output element per thread ---- */
    float acc = 0.0f;

    for (uint32_t kt = 0; kt < nkt; ++kt) {
        uint32_t gk = kt * BK;             // global K offset

        /* --- cooperative load of A tile ---
         * A is [M, K] row-major.
         * Thread (my_r, my_c) loads row my_r, K-columns [gk .. gk+BK) */
        for (int k = 0; k < BK; ++k) {
            uint32_t gm = rs + my_r;
            if (gm < M && gk + k < K)
                As[my_r][k] = A[gm * K + gk + k];
            else
                As[my_r][k] = __float_as_bfloat16(0.0f);
        }

        /* --- cooperative load of B tile ---
         * B is [N, K] row-major.
         * Thread (my_r, my_c) loads row my_c, K-columns [gk .. gk+BK) */
        for (int k = 0; k < BK; ++k) {
            uint32_t gn = ns + my_c;
            if (gn < N && gk + k < K)
                Bs[my_c][k] = B[gn * K + gk + k];
            else
                Bs[my_c][k] = __float_as_bfloat16(0.0f);
        }

        __syncthreads();

        /* --- partial dot product over this K-slice --- */
        for (int k = 0; k < BK; ++k)
            acc += __bfloat162float(As[my_r][k]) *
                   __bfloat162float(Bs[my_c][k]);

        __syncthreads();
    }

    /* ---- store ---- */
    uint32_t out_r = rs + my_r;
    uint32_t out_c = ns + my_c;
    if (out_r < M && out_c < N)
        C[out_r * N + out_c] = __float2bfloat16(acc);
}

/* ================================================================
 * Host runner
 * ================================================================ */
namespace tvm_ffi_example_cuda {

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));

    uint32_t M = static_cast<uint32_t>(A.size(0));
    uint32_t N = static_cast<uint32_t>(B.size(0));
    uint32_t K = static_cast<uint32_t>(A.size(1));

    const __nv_bfloat16 *Ap = static_cast<const __nv_bfloat16 *>(A.data_ptr());
    const __nv_bfloat16 *Bp = static_cast<const __nv_bfloat16 *>(B.data_ptr());
    __nv_bfloat16       *Cp = static_cast<__nv_bfloat16 *>(C.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));

    if (M == 0 || N == 0 || K == 0) return;

    /* 2-D grid – each block computes one BM×BN output tile. */
    uint32_t gx = (N + BN - 1) / BN;
    uint32_t gy = (M + BM - 1) / BM;

    dim3 blk(THR);
    dim3 grd(gx, gy);

    gemm_bw_kernel<<<grd, blk, 0, stream>>>(Ap, Bp, Cp, M, N, K);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace tvm_ffi_example_cuda

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);